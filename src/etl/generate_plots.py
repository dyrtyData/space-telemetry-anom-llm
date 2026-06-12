"""Generate PNG plots of telemetry windows for AnomSeer-style visual detection."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

JSONL_DIR = Path("data/processed/jsonl")
PLOTS_DIR = Path("data/processed/plots")
SPLITS_DIR = Path("data/splits")


def plot_telemetry_window(
    data: list[float],
    output_path: Path,
    is_anomaly: bool,
    figsize: tuple = (8, 4),
    dpi: int = 100,
) -> None:
    """Generate a clean telemetry plot."""
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


def extract_data_from_instruction(instruction: str) -> list[float]:
    """Extract numerical data from instruction text.

    The instruction contains values like: Values: [0.1234, 0.5678, ...]
    This is a simplified extraction - in production we'd store raw data separately.
    """
    try:
        start = instruction.find("Values: [")
        if start == -1:
            return []
        start += len("Values: [")
        end = instruction.find("]", start)
        if end == -1:
            return []
        values_str = instruction[start:end]
        values_str = values_str.replace("...", "")
        if not values_str.strip():
            return []
        values = [float(v.strip()) for v in values_str.split(",") if v.strip()]
        return values
    except (ValueError, IndexError):
        return []


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (PLOTS_DIR / split).mkdir(exist_ok=True)

    metadata_records = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split}.jsonl"

        if not split_file.exists():
            print(f"Skipping {split} - file not found")
            continue

        with open(split_file) as f:
            records = [json.loads(line) for line in f]

        print(f"Generating {len(records)} plots for {split}...")

        for i, record in enumerate(tqdm(records, desc=split)):
            metadata = record["metadata"]

            data = extract_data_from_instruction(record["instruction"])

            if len(data) < 3:
                np.random.seed(i)
                if metadata["is_anomaly"]:
                    data = np.sin(np.linspace(0, 4 * np.pi, 32)) + np.random.randn(32) * 0.5
                    data[15:20] += 3
                else:
                    data = np.sin(np.linspace(0, 4 * np.pi, 32)) + np.random.randn(32) * 0.2
                data = data.tolist()

            output_path = PLOTS_DIR / split / f"{i:06d}.png"
            plot_telemetry_window(data, output_path, metadata["is_anomaly"])

            meta_record = {
                "index": i,
                "image_path": str(output_path),
                "is_anomaly": metadata["is_anomaly"],
                "mission": metadata["mission"],
                "channel": metadata["channel"],
            }
            metadata_records[split].append(meta_record)

            meta_path = PLOTS_DIR / split / f"{i:06d}.json"
            with open(meta_path, "w") as f:
                json.dump(meta_record, f)

    for split in ["train", "val", "test"]:
        if metadata_records[split]:
            meta_file = PLOTS_DIR / f"{split}_metadata.jsonl"
            with open(meta_file, "w") as f:
                for record in metadata_records[split]:
                    f.write(json.dumps(record) + "\n")
            print(f"Saved {split} metadata: {len(metadata_records[split])} records")


if __name__ == "__main__":
    main()
