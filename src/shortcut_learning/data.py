"""Dataset and DataLoader utilities for Waterbirds."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from shortcut_learning.groups import group_id
from shortcut_learning.transforms import build_eval_transform, build_train_transform

SPLIT_TO_ID = {
    "train": 0,
    "val": 1,
    "test": 2,
}

ID_TO_SPLIT = {value: key for key, value in SPLIT_TO_ID.items()}


class WaterbirdsDataset(Dataset):
    """PyTorch dataset backed by the official Waterbirds metadata file.

    Each item is returned as a dictionary containing the transformed image,
    the bird label, the background attribute, the four-group identifier, and
    the original metadata/image path. Keeping the group information attached
    to every sample will make group-aware evaluation straightforward later.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        transform: Any | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split.lower()

        if self.split not in SPLIT_TO_ID:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {sorted(SPLIT_TO_ID)}."
            )

        metadata_path = self.dataset_dir / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Waterbirds metadata was not found: {metadata_path}"
            )

        metadata = pd.read_csv(metadata_path)

        required_columns = {"img_id", "img_filename", "y", "place", "split"}
        missing_columns = required_columns.difference(metadata.columns)
        if missing_columns:
            raise ValueError(
                "Waterbirds metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        split_id = SPLIT_TO_ID[self.split]
        metadata = metadata.loc[metadata["split"] == split_id].copy()
        metadata.reset_index(drop=True, inplace=True)

        if metadata.empty:
            raise ValueError(f"No samples were found for split '{self.split}'.")

        metadata["group_id"] = [
            group_id(int(label), int(background))
            for label, background in zip(metadata["y"], metadata["place"], strict=True)
        ]

        self.metadata = metadata
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return one Waterbirds example."""
        row = self.metadata.iloc[index]
        image_path = self.dataset_dir / str(row["img_filename"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return {
            "image": image,
            "label": int(row["y"]),
            "background": int(row["place"]),
            "group": int(row["group_id"]),
            "img_id": int(row["img_id"]),
            "path": str(image_path),
        }


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a DataLoader worker.

    PyTorch assigns each worker its own initial seed. We derive the other
    libraries' seeds from that value so random augmentations and sampling
    remain reproducible across runs.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_waterbirds_datasets(
    dataset_dir: str | Path,
    image_size: int = 224,
) -> dict[str, WaterbirdsDataset]:
    """Create train, validation, and test datasets with the correct transforms."""
    train_transform = build_train_transform(image_size=image_size)
    eval_transform = build_eval_transform(image_size=image_size)

    return {
        "train": WaterbirdsDataset(
            dataset_dir=dataset_dir,
            split="train",
            transform=train_transform,
        ),
        "val": WaterbirdsDataset(
            dataset_dir=dataset_dir,
            split="val",
            transform=eval_transform,
        ),
        "test": WaterbirdsDataset(
            dataset_dir=dataset_dir,
            split="test",
            transform=eval_transform,
        ),
    }


def build_waterbirds_loaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> dict[str, DataLoader]:
    """Create reproducible train, validation, and test DataLoaders."""
    datasets = build_waterbirds_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    return {
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            generator=generator,
            **common_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            shuffle=False,
            **common_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            shuffle=False,
            **common_kwargs,
        ),
    }
