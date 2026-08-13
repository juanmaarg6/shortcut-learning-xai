"""Create the first visual audit figures for the Waterbirds dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"

METADATA_PATH = DATASET_DIR / "metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures" / "dataset"

GROUP_NAMES = {
    0: "Landbird on land",
    1: "Landbird on water",
    2: "Waterbird on land",
    3: "Waterbird on water",
}

SPLIT_NAMES = {
    0: "Train",
    1: "Validation",
    2: "Test",
}


def load_metadata() -> pd.DataFrame:
    """Load Waterbirds metadata and add group information."""
    df = pd.read_csv(METADATA_PATH)

    df["group_id"] = 2 * df["y"] + df["place"]
    df["group_name"] = df["group_id"].map(GROUP_NAMES)
    df["split_name"] = df["split"].map(SPLIT_NAMES)

    return df


def plot_training_group_distribution(df: pd.DataFrame) -> None:
    """Plot the number of training samples in each Waterbirds group."""
    train_df = df[df["split"] == 0]

    counts = train_df["group_id"].value_counts().reindex(range(4), fill_value=0)

    labels = [GROUP_NAMES[group_id] for group_id in counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, counts.values)

    ax.set_title("Waterbirds Training Group Distribution")
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Group")

    ax.tick_params(axis="x", rotation=20)

    for bar, count in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "training_group_distribution.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_group_examples(
    df: pd.DataFrame,
    samples_per_group: int = 4,
    random_seed: int = 42,
) -> None:
    """Plot deterministic training examples from each Waterbirds group."""
    train_df = df[df["split"] == 0]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=samples_per_group,
        figsize=(3 * samples_per_group, 10),
    )

    for group_id in range(4):
        group_df = train_df[train_df["group_id"] == group_id]

        samples = group_df.sample(
            n=samples_per_group,
            random_state=random_seed,
        )

        for column, (_, row) in enumerate(samples.iterrows()):
            image_path = DATASET_DIR / row["img_filename"]

            with Image.open(image_path) as image:
                image = image.convert("RGB")

                ax = axes[group_id, column]
                ax.imshow(image)
                ax.axis("off")

                if column == 0:
                    ax.text(
                        -0.08,
                        0.5,
                        GROUP_NAMES[group_id],
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=11,
                        fontweight="bold",
                    )

    fig.suptitle(
        "Waterbirds Training Examples by Bird–Background Group",
        fontsize=14,
    )

    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.96])

    output_path = OUTPUT_DIR / "group_examples.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    """Generate the dataset visual audit."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()

    plot_training_group_distribution(df)
    plot_group_examples(df)

    print("\nDataset visualization completed.")


if __name__ == "__main__":
    main()
