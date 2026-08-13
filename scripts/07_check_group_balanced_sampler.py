"""Inspect one epoch of the Waterbirds group-balanced sampler."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from shortcut_learning.data import build_waterbirds_datasets
from shortcut_learning.sampling import build_group_balanced_sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"

SEED = 42


def main() -> None:
    """Draw one sampler epoch and report its realized group composition."""
    train_dataset = build_waterbirds_datasets(
        dataset_dir=DATASET_DIR,
        image_size=224,
    )["train"]

    groups = train_dataset.metadata["group_id"].astype(int).tolist()

    raw_counts = Counter(groups)

    sampler = build_group_balanced_sampler(
        groups=groups,
        seed=SEED,
    )

    sampled_indices = list(iter(sampler))
    sampled_groups = [groups[index] for index in sampled_indices]
    sampled_counts = Counter(sampled_groups)

    print("Group-balanced sampler audit")
    print("=" * 80)
    print(f"Seed: {SEED}")
    print(f"Epoch draws: {len(sampled_indices)}")
    print()

    print("Raw training counts")
    for group in range(4):
        print(f"G{group}: {raw_counts[group]:4d}")

    print("\nOne realized balanced epoch")
    for group in range(4):
        fraction = sampled_counts[group] / len(sampled_indices)
        print(f"G{group}: {sampled_counts[group]:4d} ({100.0 * fraction:5.2f}%)")

    print(
        "\nExpected probability per group: 25.00% "
        "(realized counts vary stochastically)."
    )


if __name__ == "__main__":
    main()
