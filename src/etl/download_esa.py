"""Download ESA-AD dataset from Zenodo.

Supports a configurable target directory (e.g. an external drive) and selecting
a subset of missions, so the dataset can be processed mission-by-mission.

Examples:
    # Download only Mission1 to an external drive
    python src/etl/download_esa.py --data-dir "/Volumes/MYDRIVE/esa-ad" --missions 1

    # Download all missions (default) to the repo's data/raw/esa-ad
    python src/etl/download_esa.py

The target directory can also be set via the ESA_DATA_DIR environment variable.
"""

import argparse
import os
import re
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "12528696"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DEFAULT_DATA_DIR = Path(os.environ.get("ESA_DATA_DIR", "data/raw/esa-ad"))


def get_file_list() -> list[dict]:
    """Fetch file metadata from Zenodo API."""
    response = requests.get(ZENODO_URL)
    response.raise_for_status()
    return response.json()["files"]


def mission_number(filename: str) -> int | None:
    """Extract the mission number from a Zenodo filename, if present."""
    match = re.search(r"Mission(\d+)", filename)
    return int(match.group(1)) if match else None


def download_file(url: str, dest: Path, size: int) -> None:
    """Download file with progress bar, resuming/skipping when already complete."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"Skipping {dest.name} (already downloaded)")
        return

    if dest.exists():
        print(
            f"Re-downloading {dest.name} (size mismatch: "
            f"{dest.stat().st_size} != {size}, likely a partial/corrupt download)"
        )

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with (
        open(dest, "wb") as f,
        tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Target directory for downloads (default: $ESA_DATA_DIR or data/raw/esa-ad)",
    )
    parser.add_argument(
        "--missions",
        type=str,
        default="all",
        help="Comma-separated mission numbers to download (e.g. '1' or '1,2,3'), or 'all'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.missions.strip().lower() == "all":
        wanted: set[int] | None = None
    else:
        wanted = {int(x) for x in args.missions.split(",") if x.strip()}

    print("Fetching ESA-AD file list from Zenodo...")
    files = get_file_list()

    selected = []
    for f in files:
        num = mission_number(f["key"])
        if wanted is None or (num is not None and num in wanted):
            selected.append(f)

    total_size = sum(f["size"] for f in selected)
    print(f"Target: {data_dir}")
    print(f"Selected {len(selected)} of {len(files)} files, total size: {total_size / 1e9:.2f} GB")

    for file_info in selected:
        url = file_info["links"]["self"]
        name = file_info["key"]
        size = file_info["size"]
        dest = data_dir / name
        download_file(url, dest, size)

    print("Download complete!")


if __name__ == "__main__":
    main()
