"""Download the ESA Anomaly Dataset from the Kaggle mirror, mission by mission.

Why Kaggle instead of Zenodo (deviation D3): the canonical Zenodo record
(10.5281/zenodo.12528696) throttles single-connection downloads to ~0.3-0.4 MB/s
(~3 h per mission) and does not support HTTP range requests, so it cannot be
parallelised. The Kaggle mirror `sammahoney/esa-anomaly-dataset` is byte-for-byte
the same size as the official manifest (11,664,533,376 B / 11.66 GB) and serves
over a fast CDN (minutes per mission).

Layout produced (flattening Kaggle's doubled ESA-MissionN/ESA-MissionN/ prefix):

    <data-dir>/ESA-Mission1/
        channels.csv  labels.csv  anomaly_types.csv  telecommands.csv
        channels/channel_1/channel_1      # pickled pandas DataFrame (datetime index)
        channels/channel_2/channel_2
        ...

Telecommands (per-file event data, ~800 small files) are skipped by default;
the anomaly-detection ETL does not use them. Pass --include-telecommands to keep.

Each per-channel file is <200 MB, so the output is FAT32-safe (no >4 GB files).

Examples:
    python src/etl/download_kaggle.py --data-dir "/Volumes/DUAL DRIVE/esa-ad" --mission 1
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

DATASET = "sammahoney/esa-anomaly-dataset"
DEFAULT_DATA_DIR = Path(os.environ.get("ESA_DATA_DIR", "data/raw/esa-ad"))

METADATA_FILES = ["channels.csv", "labels.csv", "anomaly_types.csv", "telecommands.csv"]


def kaggle_pull(remote_rel: str, dest_dir: Path, required: bool = True) -> Path | None:
    """Download one dataset file via the kaggle CLI and unzip its wrapper.

    Kaggle wraps single-file downloads in `<basename>.zip`; we extract the inner
    file (named like the original, no extension) into ``dest_dir`` and remove the
    wrapper. Returns the path to the extracted file, or None if the file does not
    exist on the dataset and ``required`` is False. Skips work if already present.
    """
    remote = f"ESA-Mission{MISSION}/ESA-Mission{MISSION}/{remote_rel}"
    basename = remote_rel.rsplit("/", 1)[-1]
    final = dest_dir / basename
    if final.exists() and final.stat().st_size > 0:
        return final

    dest_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            KAGGLE,
            "datasets",
            "download",
            DATASET,
            "-f",
            remote,
            "-p",
            str(dest_dir),
            "--force",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        if required:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        print(f"  (skipping {basename} — not present in this mission)")
        return None

    wrapper = dest_dir / f"{basename}.zip"
    if zipfile.is_zipfile(wrapper):
        with zipfile.ZipFile(wrapper) as z:
            z.extractall(dest_dir)
        wrapper.unlink()
    elif wrapper.exists():
        # Some files arrive unwrapped; just rename into place.
        wrapper.rename(final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--mission", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--include-telecommands", action="store_true")
    parser.add_argument(
        "--kaggle-bin",
        default=str(Path(sys.executable).with_name("kaggle")),
        help="Path to the kaggle CLI (default: alongside the current interpreter)",
    )
    args = parser.parse_args()

    global MISSION, KAGGLE
    MISSION = args.mission
    KAGGLE = args.kaggle_bin

    mission_dir = args.data_dir / f"ESA-Mission{MISSION}"
    channels_dir = mission_dir / "channels"

    print(f"Mission {MISSION} -> {mission_dir}")
    print("Downloading metadata CSVs...")
    required_csvs = {"channels.csv", "labels.csv", "anomaly_types.csv"}
    for csv in METADATA_FILES:
        kaggle_pull(csv, mission_dir, required=csv in required_csvs)

    channels = pd.read_csv(mission_dir / "channels.csv")["Channel"].tolist()
    print(f"{len(channels)} channels to fetch")
    for i, ch in enumerate(channels, 1):
        out = channels_dir / ch / ch
        if out.exists() and out.stat().st_size > 0:
            print(f"  [{i}/{len(channels)}] {ch} (cached)")
            continue
        print(f"  [{i}/{len(channels)}] {ch}", flush=True)
        kaggle_pull(f"channels/{ch}/{ch}", channels_dir / ch)

    if args.include_telecommands:
        tcs = pd.read_csv(mission_dir / "telecommands.csv").iloc[:, 0].tolist()
        print(f"{len(tcs)} telecommands to fetch")
        for i, tc in enumerate(tcs, 1):
            print(f"  [tc {i}/{len(tcs)}] {tc}", flush=True)
            kaggle_pull(f"telecommands/{tc}/{tc}", mission_dir / "telecommands" / tc)

    print(f"Mission {MISSION} download complete: {mission_dir}")


if __name__ == "__main__":
    main()
