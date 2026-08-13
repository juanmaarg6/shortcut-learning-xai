"""Group-aware evaluation metrics for Waterbirds."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

GROUP_IDS = (0, 1, 2, 3)
ALIGNED_GROUPS = (0, 3)
CONFLICTING_GROUPS = (1, 2)


def _to_numpy(values: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert labels, predictions, or groups to a one-dimensional NumPy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.asarray(values)

    values = np.asarray(values).reshape(-1)
    return values


def _accuracy_for_mask(
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return accuracy for a boolean mask, or NaN when the mask is empty."""
    if not np.any(mask):
        return float("nan")

    return float(np.mean(predictions[mask] == labels[mask]))


def compute_group_metrics(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    predictions: Sequence[int] | np.ndarray | torch.Tensor,
    groups: Sequence[int] | np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Compute overall and group-aware Waterbirds classification metrics.

    The shortcut gap is defined as:

        mean accuracy on aligned groups
        -
        mean accuracy on conflicting groups

    where aligned groups are G0/G3 and conflicting groups are G1/G2.
    Group means are macro averages, so each group contributes equally
    regardless of its number of examples.
    """
    labels_np = _to_numpy(labels)
    predictions_np = _to_numpy(predictions)
    groups_np = _to_numpy(groups)

    if not (len(labels_np) == len(predictions_np) == len(groups_np)):
        raise ValueError("labels, predictions, and groups must have the same length.")

    if len(labels_np) == 0:
        raise ValueError("Cannot compute metrics on an empty input.")

    unknown_groups = set(np.unique(groups_np)).difference(GROUP_IDS)
    if unknown_groups:
        raise ValueError(f"Unknown Waterbirds group ids: {sorted(unknown_groups)}")

    metrics: dict[str, float] = {
        "overall_accuracy": float(np.mean(predictions_np == labels_np)),
    }

    group_accuracies: dict[int, float] = {}

    for group in GROUP_IDS:
        mask = groups_np == group
        accuracy = _accuracy_for_mask(labels_np, predictions_np, mask)
        group_accuracies[group] = accuracy
        metrics[f"group_{group}_accuracy"] = accuracy

    present_group_accuracies = [
        accuracy for accuracy in group_accuracies.values() if not np.isnan(accuracy)
    ]

    metrics["worst_group_accuracy"] = float(min(present_group_accuracies))

    aligned_values = [
        group_accuracies[group]
        for group in ALIGNED_GROUPS
        if not np.isnan(group_accuracies[group])
    ]
    conflicting_values = [
        group_accuracies[group]
        for group in CONFLICTING_GROUPS
        if not np.isnan(group_accuracies[group])
    ]

    metrics["aligned_accuracy"] = float(np.mean(aligned_values))
    metrics["conflicting_accuracy"] = float(np.mean(conflicting_values))
    metrics["shortcut_gap"] = (
        metrics["aligned_accuracy"] - metrics["conflicting_accuracy"]
    )

    return metrics
