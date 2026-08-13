"""Group-balanced sampling utilities for Waterbirds."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler

EXPECTED_GROUPS = (0, 1, 2, 3)


def compute_group_balanced_weights(
    groups: Sequence[int],
) -> torch.Tensor:
    """Return inverse-frequency sample weights for the four Waterbirds groups.

    Each example receives weight ``1 / n_g``, where ``n_g`` is the number of
    training examples in its group. Therefore, every group contributes the
    same total probability mass to the sampler, regardless of its raw size.
    """
    group_list = [int(group) for group in groups]
    counts = Counter(group_list)

    missing = set(EXPECTED_GROUPS).difference(counts)
    if missing:
        raise ValueError(
            "Group-balanced sampling requires all four Waterbirds groups. "
            f"Missing groups: {sorted(missing)}"
        )

    unknown = set(counts).difference(EXPECTED_GROUPS)
    if unknown:
        raise ValueError(f"Unknown Waterbirds group ids: {sorted(unknown)}")

    return torch.tensor(
        [1.0 / counts[group] for group in group_list],
        dtype=torch.double,
    )


def build_group_balanced_sampler(
    groups: Sequence[int],
    seed: int,
) -> WeightedRandomSampler:
    """Build a reproducible sampler with equal expected mass per group.

    Sampling is performed with replacement and uses the original dataset
    length as the number of draws per epoch. This changes the group
    composition seen during training without changing the epoch size.
    """
    weights = compute_group_balanced_weights(groups)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
