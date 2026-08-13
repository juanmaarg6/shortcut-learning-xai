"""Select deterministic Waterbirds examples for Grad-CAM analysis."""

from __future__ import annotations

from pathlib import Path

from shortcut_learning.xai_selection import (
    add_reference_seed_status,
    build_consensus_table,
    load_method_predictions,
    select_xai_cases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS = [42, 123, 456]
REFERENCE_SEED = 123
PER_CATEGORY = 2

OUTPUT_DIR = PROJECT_ROOT / "results" / "xai"
OUTPUT_PATH = OUTPUT_DIR / "selected_cases.csv"


def main() -> None:
    """Select fixed cases using predictions only, before generating any CAM."""
    erm = load_method_predictions(
        root=PROJECT_ROOT,
        method="erm",
        seeds=SEEDS,
    )
    balanced = load_method_predictions(
        root=PROJECT_ROOT,
        method="group_balanced",
        seeds=SEEDS,
    )

    consensus = build_consensus_table(
        erm=erm,
        balanced=balanced,
        seeds=SEEDS,
    )

    table = add_reference_seed_status(
        consensus=consensus,
        erm=erm,
        balanced=balanced,
        reference_seed=REFERENCE_SEED,
    )

    selected = select_xai_cases(
        table=table,
        per_category=PER_CATEGORY,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_PATH, index=False)

    print("XAI case selection")
    print("=" * 100)
    print(f"Seeds used for consensus: {SEEDS}")
    print(f"Reference seed for displayed CAMs: {REFERENCE_SEED}")
    print(
        "Selection policy: prediction outcomes and metadata only; "
        "CAM appearance is not used."
    )
    print()

    columns = [
        "xai_category",
        "img_id",
        "group",
        "label",
        "erm_correct_votes",
        "balanced_correct_votes",
        "erm_prediction",
        "balanced_prediction",
        "erm_confidence",
        "balanced_confidence",
        "img_filename",
    ]

    print(selected[columns].to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
