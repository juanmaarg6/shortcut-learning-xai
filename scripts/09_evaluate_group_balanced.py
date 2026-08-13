"""Evaluate frozen group-balanced checkpoints on the held-out Waterbirds test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shortcut_learning.data import WaterbirdsDataset, seed_worker
from shortcut_learning.inference import predict_classifier
from shortcut_learning.models import build_resnet50_classifier
from shortcut_learning.transforms import build_eval_transform
from shortcut_learning.utils import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "raw" / "waterbird_complete95_forest2water2"
)

DEFAULT_SEEDS = (42, 123, 456)

GROUP_DISPLAY_NAMES = {
    0: "Landbird on land",
    1: "Landbird on water",
    2: "Waterbird on land",
    3: "Waterbird on water",
}

AGGREGATE_METRICS = [
    "overall_accuracy",
    "worst_group_accuracy",
    "group_0_accuracy",
    "group_1_accuracy",
    "group_2_accuracy",
    "group_3_accuracy",
    "aligned_accuracy",
    "conflicting_accuracy",
    "shortcut_gap",
    "loss",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen group-balanced checkpoints on the held-out Waterbirds test set."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    return parser.parse_args()


def build_test_loader(
    dataset_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> DataLoader:
    """Build the deterministic held-out test DataLoader."""
    dataset = WaterbirdsDataset(
        dataset_dir=dataset_dir,
        split="test",
        transform=build_eval_transform(
            image_size=image_size,
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def load_frozen_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load one already-selected group-balanced checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = build_resnet50_classifier(
        num_classes=2,
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def make_relative_paths(
    predictions: pd.DataFrame,
    dataset_dir: Path,
) -> pd.DataFrame:
    """Replace machine-specific absolute paths with dataset-relative filenames."""
    predictions = predictions.copy()

    predictions["img_filename"] = predictions["path"].map(
        lambda value: Path(value).relative_to(dataset_dir).as_posix()
    )

    predictions.drop(
        columns=["path"],
        inplace=True,
    )

    return predictions


def print_seed_metrics(
    seed: int,
    checkpoint: dict,
    metrics: dict[str, float],
) -> None:
    """Print final test metrics for one balanced checkpoint."""
    print(f"\nSeed {seed}")
    print("-" * 80)
    print(f"Selected epoch:       {checkpoint['epoch']}")
    print(f"Test loss:            {metrics['loss']:.4f}")
    print(f"Overall accuracy:     {100.0 * metrics['overall_accuracy']:.2f}%")
    print(f"Worst-group accuracy: {100.0 * metrics['worst_group_accuracy']:.2f}%")

    for group_id in range(4):
        print(
            f"G{group_id} "
            f"({GROUP_DISPLAY_NAMES[group_id]}): "
            f"{100.0 * metrics[f'group_{group_id}_accuracy']:.2f}%"
        )

    print(f"Aligned accuracy:     {100.0 * metrics['aligned_accuracy']:.2f}%")
    print(f"Conflicting accuracy: {100.0 * metrics['conflicting_accuracy']:.2f}%")
    print(f"Shortcut gap:         {100.0 * metrics['shortcut_gap']:+.2f} pp")


def build_aggregate_table(
    seed_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean and sample standard deviation across seeds."""
    rows = []

    for metric in AGGREGATE_METRICS:
        values = seed_summary[metric]

        rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "n_seeds": int(values.notna().sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Evaluate all frozen balanced seeds and save result artifacts."""
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Final group-balanced test evaluation")
    print("=" * 80)
    print(f"Device:      {device}")
    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Seeds:       {args.seeds}")
    print("Test policy: frozen checkpoints; no model selection on test")

    test_loader = build_test_loader(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    if len(test_loader.dataset) != 5794:
        raise RuntimeError(
            "Unexpected Waterbirds test size: "
            f"{len(test_loader.dataset)} (expected 5794)."
        )

    criterion = nn.CrossEntropyLoss()
    seed_rows: list[dict[str, float | int]] = []

    for seed in args.seeds:
        checkpoint_path = (
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "group_balanced"
            / f"seed_{seed}"
            / "best.pt"
        )

        model, checkpoint = load_frozen_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        metrics, predictions = predict_classifier(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        if int(metrics["n_examples"]) != 5794:
            raise RuntimeError(
                f"Seed {seed} evaluated "
                f"{int(metrics['n_examples'])} examples instead of 5794."
            )

        predictions = make_relative_paths(
            predictions=predictions,
            dataset_dir=args.dataset_dir,
        )
        predictions.insert(
            0,
            "seed",
            seed,
        )

        prediction_dir = (
            PROJECT_ROOT / "results" / "predictions" / "group_balanced" / f"seed_{seed}"
        )
        metric_dir = (
            PROJECT_ROOT / "results" / "metrics" / "group_balanced" / f"seed_{seed}"
        )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        metric_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_path = prediction_dir / "test_predictions.csv"
        metric_path = metric_dir / "test_metrics.json"

        predictions.to_csv(
            prediction_path,
            index=False,
        )

        test_record = {
            "method": "GroupBalancedERM",
            "seed": seed,
            "selected_epoch": int(checkpoint["epoch"]),
            "checkpoint_path": str(checkpoint_path),
            "test_metrics": metrics,
            "prediction_path": str(prediction_path),
        }
        write_json(
            test_record,
            metric_path,
        )

        seed_row: dict[str, float | int] = {
            "seed": seed,
            "selected_epoch": int(checkpoint["epoch"]),
        }
        seed_row.update(metrics)
        seed_rows.append(seed_row)

        print_seed_metrics(
            seed=seed,
            checkpoint=checkpoint,
            metrics=metrics,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    seed_summary = pd.DataFrame(seed_rows)

    aggregate = build_aggregate_table(
        seed_summary=seed_summary,
    )

    aggregate_dir = PROJECT_ROOT / "results" / "metrics" / "group_balanced"
    aggregate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_summary_path = aggregate_dir / "test_seed_summary.csv"
    aggregate_path = aggregate_dir / "test_aggregate.csv"

    seed_summary.to_csv(
        seed_summary_path,
        index=False,
    )
    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    print("\nAggregate group-balanced test results")
    print("=" * 80)

    for row in aggregate.itertuples(index=False):
        if row.metric == "loss":
            print(f"{row.metric:>24}: {row.mean:.4f} ± {row.std:.4f}")
        else:
            print(
                f"{row.metric:>24}: "
                f"{100.0 * row.mean:6.2f}% "
                f"± {100.0 * row.std:5.2f} pp"
            )

    print("\nSaved")
    print("=" * 80)
    print(f"Per-seed summary: {seed_summary_path}")
    print(f"Aggregate table:  {aggregate_path}")
    print(
        "Per-example predictions: "
        "results/predictions/group_balanced/seed_*/test_predictions.csv"
    )
    print(
        "Per-seed test metrics:   "
        "results/metrics/group_balanced/seed_*/test_metrics.json"
    )


if __name__ == "__main__":
    main()
