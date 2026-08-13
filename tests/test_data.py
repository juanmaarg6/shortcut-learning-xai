"""Tests for Waterbirds dataset utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from shortcut_learning.data import WaterbirdsDataset
from shortcut_learning.transforms import build_eval_transform


@pytest.fixture
def tiny_waterbirds(tmp_path: Path) -> Path:
    """Create a minimal Waterbirds-like dataset for unit tests."""
    rows = []

    samples = [
        (0, "a.jpg", 0, 0, 0),
        (1, "b.jpg", 0, 1, 0),
        (2, "c.jpg", 1, 0, 1),
        (3, "d.jpg", 1, 1, 2),
    ]

    for img_id, filename, label, background, split in samples:
        image = Image.new(
            "RGB",
            size=(320, 240),
            color=(50 + img_id * 20, 100, 150),
        )
        image.save(tmp_path / filename)

        rows.append(
            {
                "img_id": img_id,
                "img_filename": filename,
                "y": label,
                "place": background,
                "split": split,
            }
        )

    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)

    return tmp_path


def test_dataset_filters_requested_split(tiny_waterbirds: Path) -> None:
    dataset = WaterbirdsDataset(
        dataset_dir=tiny_waterbirds,
        split="train",
    )

    assert len(dataset) == 2


def test_dataset_returns_expected_group(tiny_waterbirds: Path) -> None:
    dataset = WaterbirdsDataset(
        dataset_dir=tiny_waterbirds,
        split="train",
    )

    first = dataset[0]
    second = dataset[1]

    assert first["group"] == 0
    assert second["group"] == 1


def test_eval_transform_produces_expected_tensor_shape(
    tiny_waterbirds: Path,
) -> None:
    dataset = WaterbirdsDataset(
        dataset_dir=tiny_waterbirds,
        split="val",
        transform=build_eval_transform(),
    )

    sample = dataset[0]

    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)


def test_dataset_rejects_unknown_split(tiny_waterbirds: Path) -> None:
    with pytest.raises(ValueError):
        WaterbirdsDataset(
            dataset_dir=tiny_waterbirds,
            split="validation",
        )
