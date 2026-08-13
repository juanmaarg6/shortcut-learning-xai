"""Tests for final classifier inference utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from shortcut_learning.inference import predict_classifier


class TinyInferenceDataset(Dataset):
    """Small deterministic dataset with Waterbirds-like metadata."""

    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ]
        )
        self.labels = torch.tensor([0, 1, 0, 1])
        self.backgrounds = torch.tensor([0, 1, 1, 0])
        self.groups = torch.tensor([0, 3, 1, 2])

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "background": self.backgrounds[index],
            "group": self.groups[index],
            "img_id": index,
            "path": f"image_{index}.jpg",
        }


class IdentityClassifier(nn.Module):
    """Treat the two input features as class logits."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_predict_classifier_returns_all_rows() -> None:
    loader = DataLoader(
        TinyInferenceDataset(),
        batch_size=2,
        shuffle=False,
    )

    metrics, predictions = predict_classifier(
        model=IdentityClassifier(),
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert len(predictions) == 4
    assert metrics["n_examples"] == 4.0
    assert metrics["overall_accuracy"] == pytest.approx(1.0)
    assert metrics["worst_group_accuracy"] == pytest.approx(1.0)


def test_prediction_probabilities_sum_to_one() -> None:
    loader = DataLoader(
        TinyInferenceDataset(),
        batch_size=4,
        shuffle=False,
    )

    _, predictions = predict_classifier(
        model=IdentityClassifier(),
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        use_amp=False,
    )

    probability_sum = predictions["prob_landbird"] + predictions["prob_waterbird"]

    assert probability_sum.to_numpy() == pytest.approx([1.0, 1.0, 1.0, 1.0])
