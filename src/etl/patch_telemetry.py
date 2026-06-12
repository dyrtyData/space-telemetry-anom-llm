"""Transform raw ESA-AD telemetry into training patches."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw/esa-ad")
PROCESSED_DIR = Path("data/processed")
JSONL_DIR = PROCESSED_DIR / "jsonl"
SPLITS_DIR = Path("data/splits")

WINDOW_SIZE = 32
STRIDE = 16


class RevINNormalizer:
    """Reversible Instance Normalization for time series."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize per-channel."""
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + self.eps
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Denormalize."""
        return x * self.std + self.mean


def load_mission_data(mission_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load telemetry and labels for a mission."""
    telemetry_file = mission_path / "telemetry.pkl"
    labels_file = mission_path / "labels.pkl"

    with open(telemetry_file, "rb") as f:
        telemetry = pickle.load(f)

    with open(labels_file, "rb") as f:
        labels = pickle.load(f)

    return telemetry, labels


def create_windows(
    telemetry: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[dict]:
    """Create rolling windows with labels."""
    windows = []
    n_samples = len(telemetry)

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window_data = telemetry[start:end]
        window_labels = labels[start:end]

        is_anomaly = window_labels.any()

        windows.append(
            {
                "start_idx": start,
                "end_idx": end,
                "data": window_data.tolist(),
                "is_anomaly": bool(is_anomaly),
                "anomaly_ratio": float(window_labels.mean()),
            }
        )

    return windows


def format_as_jsonl(windows: list[dict], channel_name: str, mission: str) -> list[dict]:
    """Format windows as instruction-response JSONL for LLM training."""
    records = []

    for w in windows:
        values_str = ", ".join([f"{v:.4f}" for v in np.array(w["data"]).flatten()[:10]])
        values_str += "..." if len(w["data"]) > 10 else ""

        instruction = (
            f"Analyze the following telemetry sequence from {mission} satellite, "
            f"channel {channel_name}. The sequence contains {len(w['data'])} timesteps. "
            f"Values: [{values_str}]\n\n"
            "Determine if this sequence shows anomalous behavior and explain your reasoning."
        )

        if w["is_anomaly"]:
            response = (
                "ANOMALY DETECTED. This sequence shows abnormal patterns that deviate "
                "from expected operational behavior."
            )
        else:
            response = (
                "NOMINAL. This sequence shows normal operational behavior within "
                "expected parameters."
            )

        records.append(
            {
                "instruction": instruction,
                "response": response,
                "metadata": {
                    "mission": mission,
                    "channel": channel_name,
                    "start_idx": w["start_idx"],
                    "end_idx": w["end_idx"],
                    "is_anomaly": w["is_anomaly"],
                    "anomaly_ratio": w["anomaly_ratio"],
                },
            }
        )

    return records


def main():
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    normalizer = RevINNormalizer()

    for mission_path in sorted(RAW_DIR.iterdir()):
        if not mission_path.is_dir():
            continue

        mission_name = mission_path.name
        print(f"Processing mission: {mission_name}")

        try:
            telemetry, labels = load_mission_data(mission_path)
        except FileNotFoundError:
            print(f"  Skipping {mission_name} - missing files")
            continue

        for channel in tqdm(telemetry.columns, desc=f"  Channels"):
            channel_data = telemetry[channel].values.reshape(-1, 1)
            channel_labels = (
                labels[channel].values
                if channel in labels.columns
                else np.zeros(len(channel_data))
            )

            normalized = normalizer.fit_transform(channel_data)

            windows = create_windows(normalized.flatten(), channel_labels)

            records = format_as_jsonl(windows, channel, mission_name)
            all_records.extend(records)

    output_file = JSONL_DIR / "all_patches.jsonl"
    with open(output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"Created {len(all_records)} patches")
    print(f"Anomalies: {sum(1 for r in all_records if r['metadata']['is_anomaly'])}")

    np.random.seed(42)
    indices = np.random.permutation(len(all_records))

    n_train = int(0.7 * len(indices))
    n_val = int(0.15 * len(indices))

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_file = SPLITS_DIR / f"{split_name}.jsonl"
        with open(split_file, "w") as f:
            for i in split_idx:
                f.write(json.dumps(all_records[i]) + "\n")
        print(f"{split_name}: {len(split_idx)} samples")


if __name__ == "__main__":
    main()
