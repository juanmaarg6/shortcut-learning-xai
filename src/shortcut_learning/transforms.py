"""Image transformations used by the Waterbirds experiments."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build the stochastic augmentation pipeline used for training.

    The augmentation is intentionally modest. The project studies shortcut
    learning caused by background correlations, so we avoid aggressive
    transformations that could introduce an additional experimental factor.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    resize_size: int = 232,
) -> transforms.Compose:
    """Build the deterministic preprocessing pipeline for validation and test."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
