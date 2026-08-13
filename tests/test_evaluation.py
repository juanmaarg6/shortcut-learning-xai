"""Tests for Waterbirds group-aware evaluation metrics."""

from __future__ import annotations

import math

import pytest

from shortcut_learning.evaluation import compute_group_metrics


def test_group_metrics_are_computed_correctly() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    predictions = [0, 0, 0, 1, 0, 1, 0, 0]
    groups = [0, 0, 1, 1, 2, 2, 3, 3]

    metrics = compute_group_metrics(labels, predictions, groups)

    assert metrics["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["group_0_accuracy"] == pytest.approx(1.0)
    assert metrics["group_1_accuracy"] == pytest.approx(0.5)
    assert metrics["group_2_accuracy"] == pytest.approx(0.5)
    assert metrics["group_3_accuracy"] == pytest.approx(0.0)
    assert metrics["worst_group_accuracy"] == pytest.approx(0.0)

    assert metrics["aligned_accuracy"] == pytest.approx(0.5)
    assert metrics["conflicting_accuracy"] == pytest.approx(0.5)
    assert metrics["shortcut_gap"] == pytest.approx(0.0)


def test_shortcut_gap_uses_macro_group_averages() -> None:
    labels = [0, 0, 0, 0, 0, 0, 1, 1]
    predictions = [0, 0, 0, 0, 0, 0, 1, 0]
    groups = [0, 0, 0, 0, 0, 1, 2, 3]

    metrics = compute_group_metrics(labels, predictions, groups)

    # G0=1.0, G1=1.0, G2=1.0, G3=0.0
    # aligned=(G0+G3)/2=0.5
    # conflicting=(G1+G2)/2=1.0
    assert metrics["aligned_accuracy"] == pytest.approx(0.5)
    assert metrics["conflicting_accuracy"] == pytest.approx(1.0)
    assert metrics["shortcut_gap"] == pytest.approx(-0.5)


def test_missing_group_returns_nan_for_that_group() -> None:
    labels = [0, 0, 1]
    predictions = [0, 0, 1]
    groups = [0, 1, 3]

    metrics = compute_group_metrics(labels, predictions, groups)

    assert math.isnan(metrics["group_2_accuracy"])
    assert metrics["worst_group_accuracy"] == pytest.approx(1.0)


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        compute_group_metrics(
            labels=[0, 1],
            predictions=[0],
            groups=[0, 3],
        )


def test_metrics_reject_unknown_groups() -> None:
    with pytest.raises(ValueError):
        compute_group_metrics(
            labels=[0],
            predictions=[0],
            groups=[4],
        )
