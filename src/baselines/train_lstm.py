"""Train an LSTM baseline for anomaly detection (Telemanom-style, per-channel).

Phase 2 baseline. This is intentionally a *classic* reconstruction-based detector so it
can be compared apples-to-apples against the LLM detectors in Phase 5:

  * Per channel, load the raw ESA-AD series via the shared loader (`src/etl/io.py`),
    resample to the SAME 1h grid the ETL/LLM use, and window it (WINDOW_SIZE=32).
  * Train an LSTM autoencoder on NORMAL windows only.
  * Flag windows whose reconstruction error exceeds a dynamic threshold
    (mean + z*std of the normal-window errors -- Telemanom-style).
  * Score precision / recall / F1 per channel against the labelled anomaly intervals.

Key Phase-2 decisions baked in (see plan §2 decision table):
  - keras 3 runs on the **torch** backend (no tensorflow installed): we set
    KERAS_BACKEND=torch BEFORE importing keras.
  - Loading goes through `src/etl/io.py` -- never the superseded telemetry.pkl path.
  - Raw data root comes from ESA_DATA_DIR (DEFAULT_RAW_DIR); trained models are LARGE and
    go under STAR_OUTPUT_DIR (defaults to the external DUAL DRIVE), never the near-full
    internal disk. Only the small metrics JSON is written into the repo (results/lstm/).

Runtime is bounded with --max-channels / --max-train-seq so `make baseline` finishes in
minutes; pass large values for a full per-channel sweep.

Run:  ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad" make baseline
"""

import os

# keras 3 needs a backend selected before import; tensorflow is NOT installed here.
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.etl.io import (  # noqa: E402
    DEFAULT_RAW_DIR,
    anomaly_mask_for_channel,
    discover_missions,
    iter_channels,
    load_channel_series,
    load_labels,
    resample_series,
)

#: Large artifacts (trained models) go under STAR_OUTPUT_DIR -- defaults to the external
#: drive so they never touch the near-full internal disk. Small metrics stay in the repo.
OUTPUT_ROOT = Path(os.environ.get("STAR_OUTPUT_DIR", "/Volumes/DUAL DRIVE/star-pipeline"))
MODELS_DIR = OUTPUT_ROOT / "models" / "lstm"
RESULTS_DIR = Path("results/lstm")

WINDOW_SIZE = 32
LSTM_UNITS = 64
EPOCHS = 20
BATCH_SIZE = 64
Z_SCORE = 3.0


def build_lstm_model(input_shape: tuple, lstm_units: int = LSTM_UNITS) -> keras.Model:
    """LSTM autoencoder: encode a window, then reconstruct it."""
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.LSTM(lstm_units // 2, return_sequences=False),
            layers.RepeatVector(input_shape[0]),
            layers.LSTM(lstm_units // 2, return_sequences=True),
            layers.LSTM(lstm_units, return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_shape[1])),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def create_sequences(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Stack rolling windows into a (n_windows, window_size, n_features) array."""
    seqs = [data[i : i + window_size] for i in range(0, len(data) - window_size + 1, stride)]
    return np.asarray(seqs, dtype="float32")


def dynamic_threshold(errors: np.ndarray, z_score: float = Z_SCORE) -> float:
    """Telemanom-style threshold: mean + z*std of the (normal) error distribution."""
    return float(np.mean(errors) + z_score * np.std(errors))


def train_channel_model(
    series_values: np.ndarray,
    point_mask: np.ndarray,
    channel_name: str,
    mission: str,
    args: argparse.Namespace,
) -> dict:
    """Train an LSTM autoencoder on one channel and score it."""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(series_values.reshape(-1, 1))

    sequences = create_sequences(normalized, args.window, stride=args.seq_stride)
    label_seqs = create_sequences(point_mask.reshape(-1, 1), args.window, stride=args.seq_stride)
    if len(sequences) == 0:
        return {"channel": channel_name, "mission": mission, "error": "too short to window"}

    # A window is anomalous if any timestep in it is labelled anomalous.
    window_is_anomaly = label_seqs.max(axis=(1, 2)) > 0
    normal_mask = ~window_is_anomaly

    train_sequences = sequences[normal_mask]
    if len(train_sequences) < 100:
        return {"channel": channel_name, "mission": mission, "error": "insufficient normal data"}

    # Cap training set so a single channel can't dominate wall-clock.
    if len(train_sequences) > args.max_train_seq:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(train_sequences), size=args.max_train_seq, replace=False)
        train_sequences = train_sequences[sel]

    model = build_lstm_model(input_shape=(args.window, 1))
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss", patience=3, restore_best_weights=True
    )
    history = model.fit(
        train_sequences,
        train_sequences,
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    losses = history.history["loss"]

    predictions = model.predict(sequences, verbose=0)
    errors = np.mean((sequences - predictions) ** 2, axis=(1, 2))
    threshold = dynamic_threshold(errors[normal_mask])

    predicted = errors > threshold
    actual = window_is_anomaly

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if args.save_models:
        model_path = MODELS_DIR / mission / f"{channel_name}.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    return {
        "channel": channel_name,
        "mission": mission,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "n_sequences": int(len(sequences)),
        "n_anomaly_windows": int(actual.sum()),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "epochs_ran": len(losses),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--missions", default="all", help="e.g. '1' or '1,2,3' or 'all'")
    p.add_argument("--resample", default="1h", help="pandas offset alias, or 'none'")
    p.add_argument("--window", type=int, default=WINDOW_SIZE)
    p.add_argument(
        "--seq-stride",
        type=int,
        default=16,
        help="stride between windows (default 16 matches ETL; stride=1 stalls on long channels)",
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument(
        "--max-channels",
        type=int,
        default=5,
        help="channels per mission to train (bounds runtime; use a big number for a full sweep)",
    )
    p.add_argument(
        "--max-train-seq", type=int, default=20000, help="cap normal windows per channel"
    )
    p.add_argument("--save-models", action="store_true", default=True)
    p.add_argument("--no-save-models", dest="save_models", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def summarize(results: list[dict]) -> dict:
    """Average precision/recall/F1 over channels that actually contain anomalies."""
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
    if args.save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"keras backend: {keras.backend.backend()}  |  raw data: {args.data_dir}")
    print(f"models -> {MODELS_DIR if args.save_models else '(not saved)'}")

    all_results: list[dict] = []
    for mission_dir in discover_missions(args.data_dir, args.missions):
        mission = mission_dir.name
        labels = load_labels(mission_dir)
        channels = list(iter_channels(mission_dir, target_only=True))[: args.max_channels]
        print(f"\n{mission}: training {len(channels)} channel(s)")

        for channel, channel_file in tqdm(channels, desc=f"  {mission}"):
            series = resample_series(load_channel_series(channel_file), args.resample)
            if len(series) < args.window:
                all_results.append({"channel": channel, "mission": mission, "error": "too short"})
                continue
            point_mask = anomaly_mask_for_channel(series.index, channel, labels)
            all_results.append(
                train_channel_model(series.values, point_mask, channel, mission, args)
            )

    summary = summarize(all_results)
    output = {
        "summary": summary,
        "config": vars(args) | {"data_dir": str(args.data_dir)},
        "channels": all_results,
    }
    results_file = RESULTS_DIR / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nLSTM baseline summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {results_file}")


if __name__ == "__main__":
    main()
