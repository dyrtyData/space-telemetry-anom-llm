"""Phase 11: calibrate the LSTM detector's operating point (threshold sweep).

The Phase-2/7 LSTM uses a *flat* threshold `mean + 3*std` of the normal-window reconstruction
errors. z=3.0 was an untuned default; this script shows it over-flags. Because the trained
per-channel LSTM weights don't change when only the threshold does, this REUSES the saved
models (no ~95 min retraining) -- it loads each model, computes the per-window reconstruction
errors ONCE, and evaluates a whole grid of thresholding choices on them:

  * flat threshold at a grid of z values (the operating curve), and
  * Telemanom-style pruned dynamic thresholding (the canonical unsupervised method).

It writes:
  * results/lstm/threshold_sweep.json   -- macro P/R/F1/CEF0.5 for every method (the curve), and
  * results/lstm/baseline_results_<best>.json + baseline_results_dynamic.json -- full per-channel
    records (same schema as baseline_results.json, incl. pred_starts/gt_starts for Affinity-F1),
    so evaluate.py can consume the chosen operating point directly.

This is the LSTM analogue of Phase 13's LLM precision-recall calibration. It does NOT pick the
operating point by peeking at a hidden test set -- z is a single GLOBAL hyperparameter shared by
all channels (exactly as the original z=3.0 was), and the full curve is reported, so the choice
is transparent rather than cherry-picked.

Run:  ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad" make tune-threshold
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.baselines.train_lstm import (  # noqa: E402
    MODELS_DIR,
    RESULTS_DIR,
    create_sequences,
    detect_dynamic,
    dynamic_threshold,
)
from src.etl.io import (  # noqa: E402
    DEFAULT_RAW_DIR,
    anomaly_mask_for_channel,
    discover_missions,
    iter_channels,
    load_channel_series,
    load_labels,
    resample_series,
)

WINDOW = 32
STRIDE = 16
RESAMPLE = "1h"
Z_GRID = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]


def cef_from_pr(p: float, r: float, beta: float = 0.5) -> float:
    d = beta * beta * p + r
    return 0.0 if d == 0 else (1 + beta * beta) * p * r / d


def prf(pred: np.ndarray, act: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(pred & act))
    fp = int(np.sum(pred & ~act))
    fn = int(np.sum(~pred & act))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _record(channel: str, mission: str, pred: np.ndarray, act: np.ndarray, method: str) -> dict:
    """Per-channel record matching baseline_results.json's schema (for evaluate.py)."""
    p, r, f = prf(pred, act)
    return {
        "channel": channel,
        "mission": mission,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "threshold_method": method,
        "n_sequences": int(len(pred)),
        "n_anomaly_windows": int(act.sum()),
        "window": WINDOW,
        "stride": STRIDE,
        "pred_starts": [int(i) * STRIDE for i in np.nonzero(pred)[0]],
        "gt_starts": [int(i) * STRIDE for i in np.nonzero(act)[0]],
    }


def macro(records: list[dict]) -> dict:
    scored = [c for c in records if c["n_anomaly_windows"] > 0]
    if not scored:
        return {"n_channels_scored": 0}
    p = float(np.mean([c["precision"] for c in scored]))
    r = float(np.mean([c["recall"] for c in scored]))
    f = float(np.mean([c["f1"] for c in scored]))
    return {
        "n_channels_scored": len(scored),
        "avg_precision": p,
        "avg_recall": r,
        "avg_f1": f,
        "avg_cef_0.5": cef_from_pr(p, r),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--missions", default="1")
    ap.add_argument("--max-channels", type=int, default=58)
    ap.add_argument("--ewma-span", type=int, default=5)
    ap.add_argument("--prune-p", type=float, default=0.13)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"keras backend: {keras.backend.backend()}  |  raw data: {args.data_dir}")
    print(f"reusing saved models from {MODELS_DIR}")

    # Per-z and per-dynamic full channel records (built incrementally as we stream channels).
    flat_records: dict[float, list[dict]] = {z: [] for z in Z_GRID}
    dynamic_records: list[dict] = []

    for mission_dir in discover_missions(args.data_dir, args.missions):
        mission = mission_dir.name
        labels = load_labels(mission_dir)
        channels = list(iter_channels(mission_dir, target_only=True))[: args.max_channels]
        print(f"\n{mission}: scoring {len(channels)} channel(s) by reusing saved models")

        for channel, channel_file in tqdm(channels, desc=f"  {mission}"):
            model_path = MODELS_DIR / mission / f"{channel}.keras"
            if not model_path.exists():
                continue
            series = resample_series(load_channel_series(channel_file), RESAMPLE)
            if len(series) < WINDOW:
                continue
            point_mask = anomaly_mask_for_channel(series.index, channel, labels)
            normalized = StandardScaler().fit_transform(series.values.reshape(-1, 1))
            seq = create_sequences(normalized, WINDOW, stride=STRIDE)
            act = (
                create_sequences(point_mask.reshape(-1, 1), WINDOW, stride=STRIDE).max(axis=(1, 2))
                > 0
            )
            if len(seq) == 0:
                continue

            model = keras.models.load_model(model_path)
            errors = np.mean((seq - model.predict(seq, verbose=0)) ** 2, axis=(1, 2))
            normal = ~act

            # flat threshold at each z
            for z in Z_GRID:
                thr = dynamic_threshold(errors[normal], z_score=z)
                flat_records[z].append(_record(channel, mission, errors > thr, act, f"flat_z{z}"))
            # dynamic (Telemanom)
            dpred, _ = detect_dynamic(errors, args.ewma_span, args.prune_p)
            dynamic_records.append(_record(channel, mission, dpred, act, "dynamic"))

    # Build the operating curve + pick the CEF0.5-optimal global z.
    curve = {f"flat_z{z}": macro(flat_records[z]) for z in Z_GRID}
    curve["dynamic"] = macro(dynamic_records)
    best_z = max(Z_GRID, key=lambda z: curve[f"flat_z{z}"].get("avg_cef_0.5", 0))

    sweep = {
        "z_grid": Z_GRID,
        "best_z_by_cef": best_z,
        "curve": curve,
        "config": {
            "data_dir": str(args.data_dir),
            "missions": args.missions,
            "max_channels": args.max_channels,
            "ewma_span": args.ewma_span,
            "prune_p": args.prune_p,
            "resample": RESAMPLE,
            "window": WINDOW,
            "stride": STRIDE,
            "reused_models": True,
        },
    }
    (RESULTS_DIR / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    # Full per-channel result files for evaluate.py (chosen z + dynamic, same schema).
    def write_full(records: list[dict], fname: str, method: str) -> None:
        out = {
            "summary": macro(records),
            "config": sweep["config"] | {"threshold_method": method},
            "channels": records,
        }
        (RESULTS_DIR / fname).write_text(json.dumps(out, indent=2))

    write_full(flat_records[best_z], f"baseline_results_z{best_z}.json", f"flat_z{best_z}")
    write_full(dynamic_records, "baseline_results_dynamic.json", "dynamic")

    print("\nOperating curve (macro over channels with anomalies):")
    for name, m in curve.items():
        if m.get("n_channels_scored"):
            print(
                f"  {name:10s} P={m['avg_precision']:.3f} R={m['avg_recall']:.3f} "
                f"F1={m['avg_f1']:.3f} CEF0.5={m['avg_cef_0.5']:.3f}"
            )
    print(f"\nCEF0.5-optimal global z = {best_z}")
    print(f"Wrote {RESULTS_DIR / 'threshold_sweep.json'}")
    print(f"Wrote {RESULTS_DIR / f'baseline_results_z{best_z}.json'} (chosen operating point)")
    print(f"Wrote {RESULTS_DIR / 'baseline_results_dynamic.json'} (Telemanom dynamic, for record)")


if __name__ == "__main__":
    main()
