"""Download ESA-AD dataset from Zenodo."""

from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "12528696"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DATA_DIR = Path("data/raw/esa-ad")


def get_file_list() -> list[dict]:
    """Fetch file metadata from Zenodo API."""
    response = requests.get(ZENODO_URL)
    response.raise_for_status()
    return response.json()["files"]


def download_file(url: str, dest: Path, size: int) -> None:
    """Download file with progress bar."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"Skipping {dest.name} (already downloaded)")
        return

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
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching ESA-AD file list from Zenodo...")
    files = get_file_list()

    total_size = sum(f["size"] for f in files)
    print(f"Found {len(files)} files, total size: {total_size / 1e9:.2f} GB")

    for file_info in files:
        url = file_info["links"]["self"]
        name = file_info["key"]
        size = file_info["size"]
        dest = DATA_DIR / name
        download_file(url, dest, size)

    print("Download complete!")


if __name__ == "__main__":
    main()
