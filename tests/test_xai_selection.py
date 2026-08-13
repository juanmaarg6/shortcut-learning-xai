"""Tests for deterministic XAI case selection."""

from __future__ import annotations

import pandas as pd

from shortcut_learning.xai_selection import select_xai_cases


def test_xai_selection_uses_prediction_categories() -> None:
    table = pd.DataFrame(
        [
            {
                "img_id": 1,
                "group": 1,
                "erm_correct": False,
                "balanced_correct": True,
                "erm_correct_votes": 0,
                "balanced_correct_votes": 3,
                "delta_correct_votes": 3,
                "erm_mean_confidence": 0.9,
                "balanced_mean_confidence": 0.9,
            },
            {
                "img_id": 2,
                "group": 2,
                "erm_correct": False,
                "balanced_correct": False,
                "erm_correct_votes": 0,
                "balanced_correct_votes": 0,
                "delta_correct_votes": 0,
                "erm_mean_confidence": 0.9,
                "balanced_mean_confidence": 0.9,
            },
            {
                "img_id": 3,
                "group": 0,
                "erm_correct": True,
                "balanced_correct": True,
                "erm_correct_votes": 3,
                "balanced_correct_votes": 3,
                "delta_correct_votes": 0,
                "erm_mean_confidence": 0.95,
                "balanced_mean_confidence": 0.95,
            },
        ]
    )

    selected = select_xai_cases(table, per_category=1)

    assert selected["xai_category"].tolist() == [
        "corrected_conflicting",
        "persistent_conflicting",
        "stable_aligned",
    ]
