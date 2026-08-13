"""Audit Waterbirds metadata before any model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shortcut_learning.groups import GROUP_NAMES, group_id, is_aligned

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

SPLIT_NAMES = {
    0: "train",
    1: "val",
    2: "test",
}

LABEL_NAMES = {
    0: "landbird",
    1: "waterbird",
}

BACKGROUND_NAMES = {
    0: "land",
    1: "water",
}


def find_dataset_dir() -> Path:
    """Locate the extracted Waterbirds directory."""
    preferred = RAW_DIR / "waterbird_complete95_forest2water2"
    if (preferred / "metadata.csv").exists():
        return preferred

    matches = list(RAW_DIR.rglob("metadata.csv"))
    if len(matches) == 1:
        return matches[0].parent

    if not matches:
        raise FileNotFoundError(
            "metadata.csv was not found. Run scripts/00_download_waterbirds.py first."
        )

    raise RuntimeError(
        "Multiple metadata.csv files found under data/raw. "
        "Keep only one Waterbirds extraction."
    )


def load_metadata(dataset_dir: Path) -> pd.DataFrame:
    """Load metadata and add human-readable audit columns."""
    metadata_path = dataset_dir / "metadata.csv"
    df = pd.read_csv(metadata_path)

    required = {"img_filename", "y", "place", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    df = df.copy()
    df["split_name"] = df["split"].map(SPLIT_NAMES)
    df["label_name"] = df["y"].map(LABEL_NAMES)
    df["background_name"] = df["place"].map(BACKGROUND_NAMES)
    df["group_id"] = [
        group_id(int(y), int(place))
        for y, place in zip(df["y"], df["place"], strict=True)
    ]
    df["group_name"] = df["group_id"].map(GROUP_NAMES)
    df["aligned"] = [
        is_aligned(int(y), int(place))
        for y, place in zip(df["y"], df["place"], strict=True)
    ]

    return df


def validate_metadata(df: pd.DataFrame, dataset_dir: Path) -> None:
    """Run structural checks that should hold before training."""
    assert set(df["y"].unique()).issubset({0, 1})
    assert set(df["place"].unique()).issubset({0, 1})
    assert set(df["split"].unique()).issubset({0, 1, 2})
    assert not df["img_filename"].duplicated().any()

    missing_images = [
        path
        for path in df["img_filename"].map(dataset_dir.__truediv__)
        if not path.is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image files referenced by metadata are missing. "
            f"First missing path: {missing_images[0]}"
        )


def print_table(title: str, table: pd.DataFrame | pd.Series) -> None:
    """Print an audit table with a readable heading."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(table.to_string())


def main() -> None:
    """Run the complete first-pass Waterbirds audit."""
    dataset_dir = find_dataset_dir()
    df = load_metadata(dataset_dir)
    validate_metadata(df, dataset_dir)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Total examples: {len(df):,}")
    print(f"Metadata columns: {list(df.columns)}")

    split_counts = (
        df.groupby("split_name", sort=False)
        .size()
        .rename("n")
        .reindex(["train", "val", "test"])
    )
    print_table("Split sizes", split_counts)

    class_counts = pd.crosstab(
        df["split_name"],
        df["label_name"],
    ).reindex(["train", "val", "test"])
    print_table("Class counts by split", class_counts)

    group_counts = pd.crosstab(
        df["split_name"],
        df["group_name"],
    ).reindex(
        index=["train", "val", "test"],
        columns=[GROUP_NAMES[i] for i in range(4)],
        fill_value=0,
    )
    print_table("Four-group counts by split", group_counts)

    aligned_counts = pd.crosstab(
        df["split_name"],
        df["aligned"].map({True: "aligned", False: "conflicting"}),
    ).reindex(["train", "val", "test"])
    print_table("Aligned vs conflicting counts", aligned_counts)

    train = df[df["split_name"] == "train"]
    train_alignment_rate = train["aligned"].mean()
    print(f"\nTraining aligned fraction: {train_alignment_rate:.4f}")
    print(f"Training conflicting fraction: {1.0 - train_alignment_rate:.4f}")

    print("\nAudit passed: metadata structure and image paths are consistent.")


if __name__ == "__main__":
    main()
