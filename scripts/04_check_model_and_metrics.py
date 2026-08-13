"""Smoke-test ResNet-50 and the group-aware metric implementation."""

from __future__ import annotations

from pathlib import Path

import torch

from shortcut_learning.data import build_waterbirds_loaders
from shortcut_learning.evaluation import compute_group_metrics
from shortcut_learning.models import (
    build_resnet50_classifier,
    count_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"

BATCH_SIZE = 8
NUM_WORKERS = 4
IMAGE_SIZE = 224
SEED = 42


def check_metric_sanity() -> None:
    """Run an easy-to-read synthetic example through the metric code."""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 0, 1, 0, 0]
    groups = [0, 0, 1, 1, 2, 2, 3, 3]

    metrics = compute_group_metrics(
        labels=labels,
        predictions=predictions,
        groups=groups,
    )

    print("Metric sanity check")
    print("=" * 80)

    for name, value in metrics.items():
        print(f"{name:>24}: {value:.4f}")


def check_model_forward() -> None:
    """Load ImageNet weights and run one real Waterbirds batch on the GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nModel smoke test")
    print("=" * 80)
    print(f"Device: {device}")

    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=True,
    )
    model.to(device)
    model.eval()

    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    loaders = build_waterbirds_loaders(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    batch = next(iter(loaders["val"]))
    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(images)
        else:
            logits = model(images)

    probabilities = torch.softmax(
        logits.float(),
        dim=1,
    )
    predictions = probabilities.argmax(dim=1)

    print(f"Input shape:  {tuple(images.shape)}")
    print(f"Logit shape:  {tuple(logits.shape)}")
    print(f"Predictions:  {predictions.cpu().tolist()}")
    print(f"Probability sums: {probabilities.sum(dim=1).cpu().tolist()}")

    assert logits.shape == (BATCH_SIZE, 2)
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(BATCH_SIZE, device=device),
        atol=1e-5,
    )


def main() -> None:
    """Run metric and model smoke tests."""
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()

    check_metric_sanity()
    check_model_forward()

    print("\nModel and metric smoke tests passed.")


if __name__ == "__main__":
    main()
