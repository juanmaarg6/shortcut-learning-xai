"""Model builders for Waterbirds experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50_classifier(
    num_classes: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Build a ResNet-50 classifier for Waterbirds.

    When ``pretrained`` is True, ImageNet-1K V2 weights are loaded.
    The original 1000-class fully connected layer is replaced by a newly
    initialized linear layer for the requested number of output classes.
    All parameters remain trainable for full fine-tuning.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
