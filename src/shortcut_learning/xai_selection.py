"""Deterministic selection of test cases for explainability analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_method_predictions(
    root: str | Path,
    method: str,
    seeds: list[int],
) -> pd.DataFrame:
    """Load and concatenate per-seed prediction tables for one method."""
    root = Path(root)
    frames = []

    for seed in seeds:
        path = (
            root
            / "results"
            / "predictions"
            / method
            / f"seed_{seed}"
            / "test_predictions.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")

        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_consensus_table(
    erm: pd.DataFrame,
    balanced: pd.DataFrame,
    seeds: list[int],
) -> pd.DataFrame:
    """Build one row per image with correctness votes across seeds."""
    metadata_columns = [
        "img_id",
        "img_filename",
        "label",
        "background",
        "group",
    ]

    erm_metadata = (
        erm[metadata_columns]
        .drop_duplicates()
        .sort_values("img_id")
        .reset_index(drop=True)
    )
    balanced_metadata = (
        balanced[metadata_columns]
        .drop_duplicates()
        .sort_values("img_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        erm_metadata,
        balanced_metadata,
        check_dtype=False,
    )

    rows = []

    for row in erm_metadata.itertuples(index=False):
        img_id = int(row.img_id)

        erm_image = erm[erm["img_id"] == img_id]
        balanced_image = balanced[balanced["img_id"] == img_id]

        if len(erm_image) != len(seeds) or len(balanced_image) != len(seeds):
            raise RuntimeError(
                f"Image {img_id} does not have exactly one prediction per seed."
            )

        erm_votes = int(erm_image["correct"].sum())
        balanced_votes = int(balanced_image["correct"].sum())

        rows.append(
            {
                "img_id": img_id,
                "img_filename": row.img_filename,
                "label": int(row.label),
                "background": int(row.background),
                "group": int(row.group),
                "erm_correct_votes": erm_votes,
                "balanced_correct_votes": balanced_votes,
                "delta_correct_votes": balanced_votes - erm_votes,
                "erm_mean_confidence": float(erm_image["confidence"].mean()),
                "balanced_mean_confidence": float(balanced_image["confidence"].mean()),
            }
        )

    return pd.DataFrame(rows)


def add_reference_seed_status(
    consensus: pd.DataFrame,
    erm: pd.DataFrame,
    balanced: pd.DataFrame,
    reference_seed: int,
) -> pd.DataFrame:
    """Attach predictions from one fixed seed used for the displayed CAMs."""
    output = consensus.copy()

    erm_ref = erm[erm["seed"] == reference_seed][
        ["img_id", "prediction", "confidence", "correct"]
    ].rename(
        columns={
            "prediction": "erm_prediction",
            "confidence": "erm_confidence",
            "correct": "erm_correct",
        }
    )

    balanced_ref = balanced[balanced["seed"] == reference_seed][
        ["img_id", "prediction", "confidence", "correct"]
    ].rename(
        columns={
            "prediction": "balanced_prediction",
            "confidence": "balanced_confidence",
            "correct": "balanced_correct",
        }
    )

    output = output.merge(erm_ref, on="img_id", validate="one_to_one")
    output = output.merge(balanced_ref, on="img_id", validate="one_to_one")

    return output


def select_xai_cases(
    table: pd.DataFrame,
    per_category: int = 2,
) -> pd.DataFrame:
    """Select fixed XAI cases using prediction outcomes, never CAM appearance."""
    selected_parts = []
    used_ids: set[int] = set()

    corrected = table[
        table["group"].isin([1, 2])
        & (~table["erm_correct"].astype(bool))
        & (table["balanced_correct"].astype(bool))
    ].copy()

    corrected = corrected.sort_values(
        [
            "delta_correct_votes",
            "balanced_correct_votes",
            "erm_correct_votes",
            "img_id",
        ],
        ascending=[False, False, True, True],
    )
    corrected = corrected.head(per_category).copy()
    corrected["xai_category"] = "corrected_conflicting"
    selected_parts.append(corrected)
    used_ids.update(corrected["img_id"].astype(int).tolist())

    persistent = table[
        table["group"].isin([1, 2])
        & (~table["erm_correct"].astype(bool))
        & (~table["balanced_correct"].astype(bool))
        & (~table["img_id"].isin(used_ids))
    ].copy()

    persistent["total_correct_votes"] = (
        persistent["erm_correct_votes"] + persistent["balanced_correct_votes"]
    )

    persistent = persistent.sort_values(
        [
            "total_correct_votes",
            "erm_mean_confidence",
            "balanced_mean_confidence",
            "img_id",
        ],
        ascending=[True, False, False, True],
    )
    persistent = persistent.head(per_category).copy()
    persistent["xai_category"] = "persistent_conflicting"
    selected_parts.append(persistent)
    used_ids.update(persistent["img_id"].astype(int).tolist())

    stable = table[
        table["group"].isin([0, 3])
        & (table["erm_correct"].astype(bool))
        & (table["balanced_correct"].astype(bool))
        & (~table["img_id"].isin(used_ids))
    ].copy()

    stable["total_correct_votes"] = (
        stable["erm_correct_votes"] + stable["balanced_correct_votes"]
    )

    stable = stable.sort_values(
        [
            "total_correct_votes",
            "erm_mean_confidence",
            "balanced_mean_confidence",
            "img_id",
        ],
        ascending=[False, False, False, True],
    )
    stable = stable.head(per_category).copy()
    stable["xai_category"] = "stable_aligned"
    selected_parts.append(stable)

    selected = pd.concat(selected_parts, ignore_index=True)

    category_order = {
        "corrected_conflicting": 0,
        "persistent_conflicting": 1,
        "stable_aligned": 2,
    }
    selected["_category_order"] = selected["xai_category"].map(category_order)

    return (
        selected.sort_values(["_category_order", "img_id"])
        .drop(columns=["_category_order"])
        .reset_index(drop=True)
    )
