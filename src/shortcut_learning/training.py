"""Training and validation utilities for Waterbirds classifiers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shortcut_learning.evaluation import compute_group_metrics


def _autocast_context(device: torch.device, enabled: bool):
    """Return the appropriate autocast context for the selected device."""
    if device.type == "cuda" and enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the tensors required for training/evaluation to the device."""
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    groups = batch["group"].to(device, non_blocking=True)

    return images, labels, groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train a classifier for one epoch and return aggregate metrics."""
    model.train()

    running_loss = 0.0
    total_examples = 0

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_groups: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels, groups = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        running_loss += loss.detach().item() * batch_size
        total_examples += batch_size

        predictions = logits.detach().argmax(dim=1)

        all_labels.append(labels.detach().cpu())
        all_predictions.append(predictions.cpu())
        all_groups.append(groups.detach().cpu())

    if total_examples == 0:
        raise RuntimeError("Training loader produced no examples.")

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

    return metrics


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a classifier without updating its parameters."""
    model.eval()

    running_loss = 0.0
    total_examples = 0

    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_groups: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images, labels, groups = _move_batch_to_device(batch, device)

        with _autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        running_loss += loss.item() * batch_size
        total_examples += batch_size

        predictions = logits.argmax(dim=1)

        all_labels.append(labels.cpu())
        all_predictions.append(predictions.cpu())
        all_groups.append(groups.cpu())

    if total_examples == 0:
        raise RuntimeError("Evaluation loader produced no examples.")

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

    return metrics
