"""Generate PNG plots of telemetry windows for AnomSeer-style visual detection.

Reads the full normalized window values that patch_telemetry.py stores in each
record's metadata ("values") and renders one clean PNG per window. (The earlier
version parsed the truncated instruction text and FABRICATED synthetic sine waves
when it could not recover enough values -- that produced fake telemetry and has
been removed.)

A per-split cap (--max-per-split) keeps the output bounded; the balanced JSONL can
contain tens of thousands of windows and we do not need a PNG for every one to seed
the vision model.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

PLOTS_DIR = Path("data/processed/plots")
SPLITS_DIR = Path("data/splits")


def plot_telemetry_window(
    data: list[float],
    output_path: Path,
    figsize: tuple = (8, 4),
    dpi: int = 100,
) -> None:
    """Generate a clean, title-free telemetry plot (model infers from the curve)."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(data))
    ax.plot(x, data, linewidth=1.5, color="#1f77b4")
    ax.fill_between(x, data, alpha=0.3, color="#1f77b4")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized Value")
    ax.set_xlim(0, len(data) - 1)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-split", type=int, default=2000)
    args = parser.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (PLOTS_DIR / split).mkdir(exist_ok=True)
        split_file = SPLITS_DIR / f"{split}.jsonl"
        if not split_file.exists():
            print(f"Skipping {split} - file not found")
            continue

        with open(split_file) as f:
            records = [json.loads(line) for line in f]
        if len(records) > args.max_per_split:
            records = records[: args.max_per_split]

        print(f"Generating {len(records)} plots for {split}...")
        meta_records = []
        for i, record in enumerate(tqdm(records, desc=split)):
            metadata = record["metadata"]
            values = metadata.get("values")
            if not values or len(values) < 3:
                # No fabrication: skip windows that lack stored values.
                continue

            output_path = PLOTS_DIR / split / f"{i:06d}.png"
            plot_telemetry_window(values, output_path)

            meta_records.append(
                {
                    "index": i,
                    "image_path": str(output_path),
                    "is_anomaly": metadata["is_anomaly"],
                    "mission": metadata["mission"],
                    "channel": metadata["channel"],
                }
            )

        meta_file = PLOTS_DIR / f"{split}_metadata.jsonl"
        with open(meta_file, "w") as f:
            for record in meta_records:
                f.write(json.dumps(record) + "\n")
        print(f"Saved {split}: {len(meta_records)} plots + metadata")


if __name__ == "__main__":
    main()
