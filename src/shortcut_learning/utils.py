"""General utilities for reproducible Waterbirds experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Some CUDA operations may not have a deterministic implementation.
    # warn_only keeps the experiment running while surfacing such cases.
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write a dictionary as human-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
