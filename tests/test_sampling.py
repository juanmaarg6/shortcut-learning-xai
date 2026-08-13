"""Tests for group-balanced Waterbirds sampling."""

from __future__ import annotations

import pytest

from shortcut_learning.sampling import (
    build_group_balanced_sampler,
    compute_group_balanced_weights,
)


def test_each_group_has_equal_total_weight_mass() -> None:
    groups = [0] * 8 + [1] * 4 + [2] * 2 + [3] * 6

    weights = compute_group_balanced_weights(groups)

    masses = {}

    for group in range(4):
        masses[group] = float(
            weights[
                [index for index, value in enumerate(groups) if value == group]
            ].sum()
        )

    assert masses[0] == pytest.approx(1.0)
    assert masses[1] == pytest.approx(1.0)
    assert masses[2] == pytest.approx(1.0)
    assert masses[3] == pytest.approx(1.0)


def test_sampler_draws_original_dataset_length() -> None:
    groups = [0] * 8 + [1] * 4 + [2] * 2 + [3] * 6

    sampler = build_group_balanced_sampler(
        groups=groups,
        seed=42,
    )

    assert len(list(iter(sampler))) == len(groups)


def test_sampler_is_reproducible_for_same_seed() -> None:
    groups = [0] * 8 + [1] * 4 + [2] * 2 + [3] * 6

    first = list(
        iter(
            build_group_balanced_sampler(
                groups=groups,
                seed=123,
            )
        )
    )
    second = list(
        iter(
            build_group_balanced_sampler(
                groups=groups,
                seed=123,
            )
        )
    )

    assert first == second


def test_missing_group_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_group_balanced_weights([0, 0, 1, 2, 2])
