"""Smoke-test the Waterbirds datasets and DataLoaders."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from shortcut_learning.data import (
    build_waterbirds_datasets,
    build_waterbirds_loaders,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"

BATCH_SIZE = 32
NUM_WORKERS = 4
IMAGE_SIZE = 224
SEED = 42


def print_dataset_summary() -> None:
    """Print split sizes and group counts directly from dataset metadata."""
    datasets = build_waterbirds_datasets(
        dataset_dir=DATASET_DIR,
        image_size=IMAGE_SIZE,
    )

    print("Dataset summary")
    print("=" * 80)

    for split_name, dataset in datasets.items():
        group_counts = Counter(dataset.metadata["group_id"].tolist())

        print(
            f"{split_name:>5}: "
            f"{len(dataset):>5} samples | "
            f"group counts = {dict(sorted(group_counts.items()))}"
        )


def check_batch(split_name: str, batch: dict[str, torch.Tensor]) -> None:
    """Validate the structure and tensor shapes of one collated batch."""
    images = batch["image"]
    labels = batch["label"]
    backgrounds = batch["background"]
    groups = batch["group"]

    print(f"\n{split_name.upper()} first batch")
    print("-" * 80)
    print(f"images:      shape={tuple(images.shape)}, dtype={images.dtype}")
    print(f"labels:      shape={tuple(labels.shape)}, dtype={labels.dtype}")
    print(f"backgrounds: shape={tuple(backgrounds.shape)}, dtype={backgrounds.dtype}")
    print(f"groups:      shape={tuple(groups.shape)}, dtype={groups.dtype}")
    print(f"group ids in batch: {sorted(groups.unique().tolist())}")

    expected_image_shape = (images.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)

    assert tuple(images.shape) == expected_image_shape
    assert labels.shape == (images.shape[0],)
    assert backgrounds.shape == (images.shape[0],)
    assert groups.shape == (images.shape[0],)
    assert labels.dtype == torch.int64
    assert backgrounds.dtype == torch.int64
    assert groups.dtype == torch.int64


def main() -> None:
    """Run a complete DataLoader smoke test."""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print_dataset_summary()

    loaders = build_waterbirds_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        check_batch(split_name, batch)

    print("\nDataLoader smoke test passed.")


if __name__ == "__main__":
    main()
