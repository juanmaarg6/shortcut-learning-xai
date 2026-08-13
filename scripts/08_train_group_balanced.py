"""Train the group-balanced ERM ResNet-50 mitigation on Waterbirds."""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shortcut_learning.data import (
    build_waterbirds_datasets,
    seed_worker,
)
from shortcut_learning.models import (
    build_resnet50_classifier,
    count_parameters,
)
from shortcut_learning.sampling import build_group_balanced_sampler
from shortcut_learning.training import evaluate_model, train_one_epoch
from shortcut_learning.utils import set_global_seed, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train group-balanced ERM on Waterbirds."
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit train batches for smoke tests only.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Limit validation batches for smoke tests only.",
    )

    return parser.parse_args()


def format_metric(value: float) -> str:
    """Format a probability metric as a percentage."""
    return f"{100.0 * value:6.2f}%"


def build_loaders(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    seed: int,
) -> tuple[dict[str, DataLoader], dict[int, int]]:
    """Build balanced train and ordinary validation/test DataLoaders."""
    datasets = build_waterbirds_datasets(
        dataset_dir=dataset_dir,
        image_size=image_size,
    )

    train_groups = datasets["train"].metadata["group_id"].astype(int).tolist()

    raw_counts = dict(sorted(Counter(train_groups).items()))

    train_sampler = build_group_balanced_sampler(
        groups=train_groups,
        seed=seed,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 10_000)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            sampler=train_sampler,
            shuffle=False,
            generator=loader_generator,
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

    return loaders, raw_counts


def save_checkpoint(
    *,
    model: nn.Module,
    epoch: int,
    seed: int,
    validation_metrics: dict[str, float],
    hyperparameters: dict[str, object],
    path: Path,
) -> None:
    """Save the best balanced-training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "validation_metrics": validation_metrics,
            "hyperparameters": hyperparameters,
        },
        path,
    )


def main() -> None:
    """Run one group-balanced training experiment."""
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    run_name = f"seed_{args.seed}"
    checkpoint_dir = (
        PROJECT_ROOT / "artifacts" / "checkpoints" / "group_balanced" / run_name
    )
    metrics_dir = PROJECT_ROOT / "results" / "metrics" / "group_balanced" / run_name

    checkpoint_path = checkpoint_dir / "best.pt"
    history_path = metrics_dir / "history.csv"
    summary_path = metrics_dir / "summary.json"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Waterbirds group-balanced ERM training")
    print("=" * 80)
    print(f"Device:          {device}")
    if device.type == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Seed:            {args.seed}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.learning_rate}")
    print(f"Weight decay:    {args.weight_decay}")
    print(f"AMP:             {use_amp}")
    print("Train sampler:   inverse group frequency, replacement=True")
    print("Expected groups: 25% / 25% / 25% / 25%")
    print("Epoch length:    original training-set length")
    print("Checkpoint rule: minimum validation cross-entropy loss")

    if args.max_train_batches is not None or args.max_val_batches is not None:
        print(
            "WARNING: batch limits are active. "
            "This is a smoke-test run, not a final experiment."
        )

    loaders, raw_group_counts = build_loaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )

    print(f"Raw train groups: {raw_group_counts}")

    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=True,
    ).to(device)

    total_parameters, trainable_parameters = count_parameters(model)

    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    hyperparameters = {
        "method": "GroupBalancedERM",
        "model": "resnet50",
        "pretrained_weights": "IMAGENET1K_V2",
        "full_fine_tuning": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "min_learning_rate": args.min_learning_rate,
        "amp": use_amp,
        "train_sampling": "inverse_group_frequency_with_replacement",
        "expected_group_probability": 0.25,
        "epoch_num_samples": len(loaders["train"].dataset),
        "raw_training_group_counts": raw_group_counts,
        "checkpoint_selection": "minimum_validation_cross_entropy_loss",
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_metrics: dict[str, float] | None = None
    history: list[dict[str, float | int]] = []

    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=args.max_train_batches,
        )

        val_metrics = evaluate_model(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            max_batches=args.max_val_batches,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }

        for name, value in train_metrics.items():
            row[f"train_{name}"] = value

        for name, value in val_metrics.items():
            row[f"val_{name}"] = value

        history.append(row)
        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)

            save_checkpoint(
                model=model,
                epoch=epoch,
                seed=args.seed,
                validation_metrics=val_metrics,
                hyperparameters=hyperparameters,
                path=checkpoint_path,
            )

        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"lr={current_lr:.2e} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"train acc={format_metric(train_metrics['overall_accuracy'])} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val acc={format_metric(val_metrics['overall_accuracy'])} | "
            f"val WGA={format_metric(val_metrics['worst_group_accuracy'])} | "
            f"gap={100.0 * val_metrics['shortcut_gap']:+6.2f} pp | "
            f"{epoch_seconds:.1f}s"
        )

    total_seconds = time.perf_counter() - training_start

    if best_val_metrics is None:
        raise RuntimeError("No validation checkpoint was produced.")

    summary = {
        "method": "GroupBalancedERM",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_metrics": best_val_metrics,
        "total_training_seconds": total_seconds,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "hyperparameters": hyperparameters,
    }

    write_json(summary, summary_path)

    print("\nBest validation checkpoint")
    print("=" * 80)
    print(f"Epoch:                 {best_epoch}")
    print(f"Validation loss:       {best_val_loss:.4f}")
    print(
        f"Overall accuracy:      {format_metric(best_val_metrics['overall_accuracy'])}"
    )
    print(
        "Worst-group accuracy:  "
        f"{format_metric(best_val_metrics['worst_group_accuracy'])}"
    )
    print(
        f"Aligned accuracy:      {format_metric(best_val_metrics['aligned_accuracy'])}"
    )
    print(
        "Conflicting accuracy:  "
        f"{format_metric(best_val_metrics['conflicting_accuracy'])}"
    )
    print(f"Shortcut gap:          {100.0 * best_val_metrics['shortcut_gap']:+.2f} pp")
    print(f"Checkpoint:             {checkpoint_path}")
    print(f"History:                {history_path}")
    print(f"Summary:                {summary_path}")
    print(f"Total time:             {total_seconds:.1f}s")

    if args.max_train_batches is not None or args.max_val_batches is not None:
        print(
            "\nSmoke-test run completed. "
            "Do not use these metrics as experimental results."
        )


if __name__ == "__main__":
    main()
