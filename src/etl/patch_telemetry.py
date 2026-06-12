"""Transform raw ESA-AD telemetry into training patches.

Rewritten for the real ESA Anomaly Dataset structure (deviation D2). The dataset
does NOT ship `telemetry.pkl` / `labels.pkl` per mission as the original plan
assumed. Each mission directory looks like:

    ESA-Mission1/
        channels.csv      # Channel, Subsystem, Physical Unit, Group, Target(YES/NO)
        labels.csv        # ID, Channel, StartTime, EndTime  (ISO-8601 UTC intervals)
        anomaly_types.csv # ID, Class, Subclass, Category, ...
        channels/channel_N/channel_N   # pickled pandas DataFrame:
                                        #   DatetimeIndex 'datetime' + 1 float32 column

Scale note (deviation D4): a single channel holds millions of samples at ~90 s
cadence over many years. Naive windowing of all channels yields tens of millions
of windows. To produce a tractable, balanced LLM-training set we:
  1. resample each channel to a uniform cadence (--resample, default 1h),
  2. keep ALL anomalous windows, and
  3. subsample NOMINAL windows to --normal-ratio x (#anomalous), capped at
     --max-windows total.
Anomaly labels are mapped from time intervals in labels.csv onto the resampled
grid (a window is anomalous if any of its timesteps fall in a labelled interval).

Output schema is unchanged from the plan (instruction/response/metadata) so the
downstream Phase 3-5 formatting stays compatible; metadata now also carries the
normalized window `values` (for plot generation) and the window's time span.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_RAW_DIR = Path(os.environ.get("ESA_DATA_DIR", "data/raw/esa-ad"))
PROCESSED_DIR = Path("data/processed")
JSONL_DIR = PROCESSED_DIR / "jsonl"
SPLITS_DIR = Path("data/splits")

WINDOW_SIZE = 32
STRIDE = 16


class RevINNormalizer:
    """Reversible Instance Normalization for time series (per-channel)."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + self.eps
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def load_channel_series(channel_file: Path) -> pd.Series:
    """Load a per-channel pickled DataFrame and return a 1-column float Series."""
    with open(channel_file, "rb") as f:
        df = pickle.load(f)
    if isinstance(df, pd.DataFrame):
        series = df.iloc[:, 0]
    else:
        series = df  # already a Series
    series.index = pd.to_datetime(series.index)
    return series.astype("float32")


def load_labels(mission_dir: Path) -> pd.DataFrame:
    """Load labels.csv with parsed, tz-naive UTC interval bounds."""
    labels = pd.read_csv(mission_dir / "labels.csv")
    for col in ("StartTime", "EndTime"):
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    return labels


def anomaly_mask_for_channel(
    index: pd.DatetimeIndex, channel: str, labels: pd.DataFrame
) -> np.ndarray:
    """Boolean mask over `index`: True where the timestamp lies in a labelled
    anomaly interval for `channel`."""
    mask = np.zeros(len(index), dtype=bool)
    ch_labels = labels[labels["Channel"] == channel]
    idx_values = index.values
    for _, row in ch_labels.iterrows():
        start = np.datetime64(row["StartTime"])
        end = np.datetime64(row["EndTime"])
        mask |= (idx_values >= start) & (idx_values <= end)
    return mask


def create_windows(
    values: np.ndarray,
    mask: np.ndarray,
    timestamps: pd.DatetimeIndex,
    window_size: int,
    stride: int,
) -> list[dict]:
    """Create rolling windows tagged with anomaly status and time span."""
    windows = []
    n = len(values)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        wlabels = mask[start:end]
        windows.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "start_time": str(timestamps[start]),
                "end_time": str(timestamps[end - 1]),
                "data": values[start:end].astype(float).tolist(),
                "is_anomaly": bool(wlabels.any()),
                "anomaly_ratio": float(wlabels.mean()),
            }
        )
    return windows


def format_record(w: dict, channel_name: str, mission: str) -> dict:
    """Format one window as an instruction/response record for LLM training."""
    flat = np.asarray(w["data"]).flatten()
    values_str = ", ".join(f"{v:.4f}" for v in flat[:10])
    values_str += "..." if len(flat) > 10 else ""

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
            "NOMINAL. This sequence shows normal operational behavior within expected parameters."
        )

    return {
        "instruction": instruction,
        "response": response,
        "metadata": {
            "mission": mission,
            "channel": channel_name,
            "start_idx": w["start_idx"],
            "end_idx": w["end_idx"],
            "start_time": w["start_time"],
            "end_time": w["end_time"],
            "is_anomaly": w["is_anomaly"],
            "anomaly_ratio": w["anomaly_ratio"],
            "values": [round(float(v), 6) for v in w["data"]],
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--missions", default="all", help="e.g. '1' or '1,2,3' or 'all'")
    p.add_argument("--resample", default="1h", help="pandas offset alias, or 'none'")
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--target-only", action="store_true", default=True)
    p.add_argument("--all-channels", dest="target_only", action="store_false")
    p.add_argument(
        "--normal-ratio",
        type=float,
        default=3.0,
        help="nominal:anomalous window ratio after subsampling",
    )
    p.add_argument("--max-windows", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def discover_missions(data_dir: Path, missions: str) -> list[Path]:
    dirs = sorted(d for d in data_dir.glob("ESA-Mission*") if d.is_dir())
    if missions.strip().lower() != "all":
        wanted = {int(x) for x in missions.split(",") if x.strip()}
        dirs = [d for d in dirs if int(d.name.replace("ESA-Mission", "")) in wanted]
    return dirs


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    anomalous_records: list[dict] = []
    nominal_records: list[dict] = []

    for mission_dir in discover_missions(args.data_dir, args.missions):
        mission = mission_dir.name
        print(f"Processing {mission}")
        channels_meta = pd.read_csv(mission_dir / "channels.csv")
        if args.target_only:
            channels_meta = channels_meta[channels_meta["Target"] == "YES"]
        labels = load_labels(mission_dir)

        for channel in tqdm(channels_meta["Channel"].tolist(), desc=f"  {mission} channels"):
            channel_file = mission_dir / "channels" / channel / channel
            if not channel_file.exists():
                continue
            series = load_channel_series(channel_file)
            if args.resample.lower() != "none":
                series = series.resample(args.resample).mean().interpolate(limit_direction="both")
            series = series.dropna()
            if len(series) < args.window:
                continue

            values = RevINNormalizer().fit_transform(series.values.reshape(-1, 1)).flatten()
            mask = anomaly_mask_for_channel(series.index, channel, labels)
            windows = create_windows(values, mask, series.index, args.window, args.stride)

            for w in windows:
                rec = format_record(w, channel, mission)
                (anomalous_records if w["is_anomaly"] else nominal_records).append(rec)

    # Balance: keep all anomalous, subsample nominal to ratio, cap total.
    n_anom = len(anomalous_records)
    n_nominal_keep = min(len(nominal_records), int(args.normal_ratio * n_anom))
    if n_nominal_keep < len(nominal_records):
        keep_idx = rng.choice(len(nominal_records), size=n_nominal_keep, replace=False)
        nominal_records = [nominal_records[i] for i in keep_idx]

    all_records = anomalous_records + nominal_records
    if len(all_records) > args.max_windows:
        keep_idx = rng.choice(len(all_records), size=args.max_windows, replace=False)
        all_records = [all_records[i] for i in keep_idx]
    rng.shuffle(all_records)

    output_file = JSONL_DIR / "all_patches.jsonl"
    with open(output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    n_anom_final = sum(1 for r in all_records if r["metadata"]["is_anomaly"])
    print(
        f"Created {len(all_records)} patches ({n_anom_final} anomalous, "
        f"{len(all_records) - n_anom_final} nominal)"
    )

    # Train/val/test split (70/15/15).
    indices = rng.permutation(len(all_records))
    n_train = int(0.7 * len(indices))
    n_val = int(0.15 * len(indices))
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }
    for split_name, split_idx in splits.items():
        split_file = SPLITS_DIR / f"{split_name}.jsonl"
        with open(split_file, "w") as f:
            for i in split_idx:
                f.write(json.dumps(all_records[i]) + "\n")
        print(f"{split_name}: {len(split_idx)} samples")


if __name__ == "__main__":
    main()
