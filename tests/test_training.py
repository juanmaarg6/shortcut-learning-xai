"""Tests for generic training utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from shortcut_learning.training import evaluate_model


class TinyDataset(Dataset):
    """Minimal deterministic binary classification dataset."""

    def __init__(self) -> None:
        self.images = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        self.labels = torch.tensor([0, 1, 0, 1])
        self.groups = torch.tensor([0, 1, 2, 3])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "group": self.groups[index],
        }


class IdentityClassifier(nn.Module):
    """Classifier whose two inputs are already class logits."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_evaluate_model_computes_perfect_accuracy() -> None:
    loader = DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
    )

    metrics = evaluate_model(
        model=IdentityClassifier(),
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert metrics["overall_accuracy"] == 1.0
    assert metrics["worst_group_accuracy"] == 1.0
    assert metrics["loss"] > 0.0
    assert metrics["n_examples"] == 4.0
