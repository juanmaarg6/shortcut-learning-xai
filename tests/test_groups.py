"""Tests for Waterbirds group definitions."""

import pytest

from shortcut_learning.groups import group_id, is_aligned


@pytest.mark.parametrize(
    ("label", "background", "expected"),
    [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 2),
        (1, 1, 3),
    ],
)
def test_group_id(label: int, background: int, expected: int) -> None:
    assert group_id(label, background) == expected


@pytest.mark.parametrize(
    ("label", "background", "expected"),
    [
        (0, 0, True),
        (0, 1, False),
        (1, 0, False),
        (1, 1, True),
    ],
)
def test_is_aligned(label: int, background: int, expected: bool) -> None:
    assert is_aligned(label, background) is expected
