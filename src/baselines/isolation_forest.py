"""Quick Isolation Forest baseline for comparison with the LSTM and LLM detectors.

A cheap, training-free-ish per-channel baseline: window each channel (same 1h grid /
WINDOW_SIZE as the LSTM), fit an IsolationForest on the windowed feature vectors, and
score precision/recall/F1 against the labelled anomaly intervals.

Uses the shared ESA-AD loader (`src/etl/io.py`) -- never the superseded telemetry.pkl
path. No large artifacts are produced, so everything stays in the repo (results JSON).

Run:  ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad" make baseline-if
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.etl.io import (
    DEFAULT_RAW_DIR,
    anomaly_mask_for_channel,
    discover_missions,
    iter_channels,
    load_channel_series,
    load_labels,
    resample_series,
)

RESULTS_DIR = Path("results/isolation_forest")
WINDOW_SIZE = 32


def score_channel(
    series_values: np.ndarray,
    point_mask: np.ndarray,
    channel: str,
    mission: str,
    args: argparse.Namespace,
) -> dict:
    """Window a channel, fit IsolationForest, return precision/recall/F1."""
    n = len(series_values)
    if n < args.window + 1:
        return {"channel": channel, "mission": mission, "error": "too short"}

    windows, window_labels = [], []
    for i in range(0, n - args.window + 1, args.seq_stride):
        windows.append(series_values[i : i + args.window])
        window_labels.append(point_mask[i : i + args.window].max())

    X = np.asarray(windows, dtype="float32")
    y = np.asarray(window_labels) > 0
    X_scaled = StandardScaler().fit_transform(X)

    clf = IsolationForest(contamination=args.contamination, random_state=args.seed, n_jobs=-1)
    predicted = clf.fit_predict(X_scaled) == -1  # IF returns -1 for anomalies

    tp = int(np.sum(predicted & y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "channel": channel,
        "mission": mission,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_windows": int(len(X)),
        "n_anomaly_windows": int(y.sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--missions", default="all", help="e.g. '1' or '1,2,3' or 'all'")
    p.add_argument("--resample", default="1h", help="pandas offset alias, or 'none'")
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument(
        "--seq-stride", type=int, default=16, help="stride between windows (matches ETL)"
    )
    p.add_argument("--max-channels", type=int, default=5, help="channels per mission (bounds time)")
    p.add_argument("--contamination", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if "error" not in r and r["n_anomaly_windows"] > 0]
    if not scored:
        return {"n_channels_scored": 0}
    return {
        "n_channels_scored": len(scored),
        "avg_precision": float(np.mean([r["precision"] for r in scored])),
        "avg_recall": float(np.mean([r["recall"] for r in scored])),
        "avg_f1": float(np.mean([r["f1"] for r in scored])),
    }


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Isolation Forest baseline  |  raw data: {args.data_dir}")

    all_results: list[dict] = []
    for mission_dir in discover_missions(args.data_dir, args.missions):
        mission = mission_dir.name
        labels = load_labels(mission_dir)
        channels = list(iter_channels(mission_dir, target_only=True))[: args.max_channels]
        print(f"\n{mission}: scoring {len(channels)} channel(s)")

        for channel, channel_file in tqdm(channels, desc=f"  {mission}"):
            series = resample_series(load_channel_series(channel_file), args.resample)
            point_mask = anomaly_mask_for_channel(series.index, channel, labels)
            all_results.append(score_channel(series.values, point_mask, channel, mission, args))

    summary = summarize(all_results)
    output = {
        "summary": summary,
        "config": vars(args) | {"data_dir": str(args.data_dir)},
        "channels": all_results,
    }
    results_file = RESULTS_DIR / "if_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nIsolation Forest summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {results_file}")


if __name__ == "__main__":
    main()
