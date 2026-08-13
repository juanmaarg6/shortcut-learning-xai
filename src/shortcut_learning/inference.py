"""Inference utilities for final Waterbirds evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shortcut_learning.evaluation import compute_group_metrics


def _autocast_context(device: torch.device, enabled: bool):
    """Return the appropriate autocast context for inference."""
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


@torch.inference_mode()
def predict_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a classifier and return metrics plus per-example predictions."""
    model.eval()

    running_loss = 0.0
    total_examples = 0
    rows: list[dict[str, Any]] = []

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_groups: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )
        labels = batch["label"].to(
            device,
            non_blocking=True,
        )
        groups = batch["group"].to(
            device,
            non_blocking=True,
        )

        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values

        batch_size = labels.shape[0]
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()
        groups_cpu = groups.cpu()
        probabilities_cpu = probabilities.cpu()
        confidence_cpu = confidence.cpu()

        backgrounds = batch["background"]
        img_ids = batch["img_id"]
        paths = batch["path"]

        for index in range(batch_size):
            rows.append(
                {
                    "img_id": int(img_ids[index]),
                    "path": str(paths[index]),
                    "label": int(labels_cpu[index]),
                    "background": int(backgrounds[index]),
                    "group": int(groups_cpu[index]),
                    "prediction": int(predictions_cpu[index]),
                    "prob_landbird": float(probabilities_cpu[index, 0]),
                    "prob_waterbird": float(probabilities_cpu[index, 1]),
                    "confidence": float(confidence_cpu[index]),
                    "correct": bool(predictions_cpu[index] == labels_cpu[index]),
                }
            )

        all_labels.append(labels_cpu)
        all_predictions.append(predictions_cpu)
        all_groups.append(groups_cpu)

    if total_examples == 0:
        raise RuntimeError("Inference loader produced no examples.")

    labels_tensor = torch.cat(all_labels)
    predictions_tensor = torch.cat(all_predictions)
    groups_tensor = torch.cat(all_groups)

    metrics = compute_group_metrics(
        labels=labels_tensor,
        predictions=predictions_tensor,
        groups=groups_tensor,
    )
    metrics["loss"] = running_loss / total_examples
    metrics["n_examples"] = float(total_examples)

    predictions_df = pd.DataFrame(rows)

    if len(predictions_df) != total_examples:
        raise RuntimeError(
            "Prediction table size does not match the number of evaluated examples."
        )

    return metrics, predictions_df
