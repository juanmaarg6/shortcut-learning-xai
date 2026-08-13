"""Tests for Waterbirds model builders."""

from __future__ import annotations

import torch.nn as nn

from shortcut_learning.models import (
    build_resnet50_classifier,
    count_parameters,
)


def test_resnet50_has_two_output_classes() -> None:
    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=False,
    )

    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 2


def test_all_resnet50_parameters_are_trainable() -> None:
    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=False,
    )

    total, trainable = count_parameters(model)

    assert total > 0
    assert trainable == total
