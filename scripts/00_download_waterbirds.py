"""Download and safely extract the official Waterbirds tarball."""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://downloads.cs.stanford.edu/nlp/data/dro/"
    "waterbird_complete95_forest2water2.tar.gz"
)
ARCHIVE_NAME = "waterbird_complete95_forest2water2.tar.gz"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE_PATH = RAW_DIR / ARCHIVE_NAME
EXPECTED_DIR = RAW_DIR / "waterbird_complete95_forest2water2"


def _print_progress(block_count: int, block_size: int, total_size: int) -> None:
    """Print a compact download progress indicator."""
    if total_size <= 0:
        return

    downloaded = min(block_count * block_size, total_size)
    pct = 100.0 * downloaded / total_size
    print(
        f"\rDownloading: {pct:6.2f}% "
        f"({downloaded / 1024**2:.1f}/{total_size / 1024**2:.1f} MiB)",
        end="",
        flush=True,
    )


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract an archive while preventing path traversal."""
    destination = destination.resolve()

    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"Unsafe archive member: {member.name}")

    tar.extractall(destination)


def main() -> None:
    """Download Waterbirds if needed and extract it into data/raw."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXPECTED_DIR.exists() and (EXPECTED_DIR / "metadata.csv").exists():
        print(f"Dataset already exists: {EXPECTED_DIR}")
        return

    if not ARCHIVE_PATH.exists():
        print(f"Downloading official Waterbirds archive to:\n  {ARCHIVE_PATH}")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH, _print_progress)
        print()
    else:
        print(f"Using existing archive: {ARCHIVE_PATH}")

    print("Extracting dataset...")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        _safe_extract(tar, RAW_DIR)

    metadata_matches = list(RAW_DIR.rglob("metadata.csv"))
    if not metadata_matches:
        raise FileNotFoundError(
            "Extraction completed, but metadata.csv was not found under data/raw."
        )

    print("Done.")
    print(f"Metadata found at: {metadata_matches[0]}")


if __name__ == "__main__":
    main()
