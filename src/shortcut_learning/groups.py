"""Group definitions for the Waterbirds dataset."""

from __future__ import annotations

GROUP_NAMES = {
    0: "landbird_on_land",
    1: "landbird_on_water",
    2: "waterbird_on_land",
    3: "waterbird_on_water",
}


def group_id(label: int, background: int) -> int:
    """Return the canonical group identifier for (bird label, background).

    Waterbirds metadata uses:
    - y = 0: landbird
    - y = 1: waterbird
    - place = 0: land background
    - place = 1: water background

    The canonical identifier is 2 * y + place.
    """
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label}")
    if background not in (0, 1):
        raise ValueError(f"background must be 0 or 1, got {background}")
    return 2 * label + background


def is_aligned(label: int, background: int) -> bool:
    """Return True when bird type and background follow the spurious correlation."""
    return label == background
