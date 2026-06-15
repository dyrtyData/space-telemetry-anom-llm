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
    """Flat Telemanom threshold: mean + z*std of the (normal) error distribution.

    This is the Phase-2/7 baseline ("flat" method). It is *semi-supervised* -- it sets the
    cutoff from the true-normal window errors only -- and is preserved as the reproducible
    reference. Phase 11 adds the unsupervised pruned-dynamic method below.
    """
    return float(np.mean(errors) + z_score * np.std(errors))


# --------------------------------------------------------------------------- #
# Phase 11: Telemanom pruned dynamic error thresholding (Hundman et al., 2018)
# --------------------------------------------------------------------------- #
# The canonical Telemanom scheme, applied to our per-window reconstruction-error stream
# (windows are at a fixed stride on a CONTIGUOUS per-channel timeline, so the error series
# is an ordered signal -- smoothing + sequence grouping are meaningful, unlike the LLM's
# shuffled split). It is fully *unsupervised* (uses no labels to set the cutoff): a flat
# mean+z*std under-thresholds noisy channels and over-thresholds clean ones; this searches
# z per channel and then PRUNES marginal candidates whose error is not sufficiently above
# the noise floor, which is the main false-positive killer.


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Exponentially-weighted moving average (smooth the error stream before thresholding)."""
    if span <= 1 or len(x) == 0:
        return x.astype("float64")
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(x), dtype="float64")
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _contiguous_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_inclusive), ...] for each contiguous True run in a bool array."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def find_epsilon(
    e_s: np.ndarray, z_min: float = 2.5, z_max: float = 12.0, z_step: float = 0.5
) -> float:
    """Telemanom find_epsilon: pick the threshold that maximally cleans the error stream.

    For each candidate z, eps = mean + z*std; removing the errors above eps should sharply
    reduce both the mean and std of what remains, *without* flagging too many points or too
    many separate sequences. Maximizes (Δmean% + Δstd%) / (n_sequences^2 + n_anomalies).
    """
    mean_e = float(np.mean(e_s))
    std_e = float(np.std(e_s))
    fallback = mean_e + Z_SCORE * std_e
    if std_e == 0:
        return fallback

    best_score = -np.inf
    best_eps = fallback
    z = z_min
    while z <= z_max:
        eps = mean_e + z * std_e
        above = e_s >= eps
        n_anom = int(above.sum())
        if 0 < n_anom < 0.5 * len(e_s):
            kept = e_s[~above]
            if len(kept) > 0:
                d_mean = (mean_e - float(np.mean(kept))) / mean_e if mean_e else 0.0
                d_std = (std_e - float(np.std(kept))) / std_e if std_e else 0.0
                n_seqs = len(_contiguous_runs(above))
                score = (d_mean + d_std) / (n_seqs**2 + n_anom)
                if score > best_score:
                    best_score = score
                    best_eps = eps
        z += z_step
    return best_eps


def prune_sequences(
    e_s: np.ndarray, seqs: list[tuple[int, int]], p: float = 0.13
) -> list[tuple[int, int]]:
    """Telemanom pruning: drop anomaly sequences too close to the noise floor.

    Sort the sequences' peak errors (plus the max non-anomalous error) descending; walk the
    list and tentatively mark a sequence for removal when the % drop to the next peak is below
    `p`, but RESET that removal list whenever a large drop appears (everything above the last
    big drop is kept). Only the low-error tail of barely-separated candidates is pruned.
    """
    if not seqs:
        return []
    seq_max = [float(e_s[s : e + 1].max()) for s, e in seqs]
    flags = np.zeros(len(e_s), dtype=bool)
    for s, e in seqs:
        flags[s : e + 1] = True
    non_anom_max = float(e_s[~flags].max()) if (~flags).any() else 0.0

    order = list(np.argsort(seq_max)[::-1])  # sequence indices, highest peak first
    sorted_max = [seq_max[i] for i in order] + [non_anom_max]

    remove_positions: list[int] = []
    for i in range(len(sorted_max) - 1):
        top = sorted_max[i]
        drop = (top - sorted_max[i + 1]) / top if top > 0 else 0.0
        if drop < p:
            remove_positions.append(i)
        else:
            remove_positions = []  # a real separation -> keep everything up to here
    remove = {order[i] for i in remove_positions}
    return [seqs[i] for i in range(len(seqs)) if i not in remove]


def detect_dynamic(errors: np.ndarray, ewma_span: int, prune_p: float) -> tuple[np.ndarray, float]:
    """Run the full pruned-dynamic pipeline on a per-window error stream.

    Returns (predicted_bool_per_window, epsilon-in-error-space). Pipeline:
    log-transform -> EWMA smooth -> find_epsilon -> threshold -> group sequences -> prune.

    The log transform is essential: reconstruction MSE is positive and heavy-tailed, so a
    raw mean + z*std is dominated by a few catastrophic windows (epsilon explodes, recall
    collapses). log makes the error distribution roughly Gaussian, so mean + z*std -- and the
    Telemanom find_epsilon search over it -- behave as intended. The threshold is reported
    back in linear error space (exp).
    """
    linear = np.asarray(errors, dtype="float64")
    log_e = np.log(linear + 1e-12)
    e_s = _ewma(log_e, ewma_span)
    eps = find_epsilon(e_s)
    above = e_s >= eps
    # Prune on the *linear* error peaks (positive magnitudes, so the %-drop test is meaningful;
    # log peaks are negative for sub-1 errors and break the ratio).
    kept = prune_sequences(linear, _contiguous_runs(above), p=prune_p)
    predicted = np.zeros(len(errors), dtype=bool)
    for s, e in kept:
        predicted[s : e + 1] = True
    return predicted, float(np.exp(eps))


def train_channel_model(
    series_values: np.ndarray,
    point_mask: np.ndarray,
    channel_name: str,
    mission: str,
    args: argparse.Namespace,
    prior: dict | None = None,
) -> dict:
    """Train (or reuse) an LSTM autoencoder on one channel and score it.

    `prior` is the same channel's record from a previous run; when --reuse-models loads a
    saved model (skipping training), the loss history isn't regenerated, so initial/final
    loss are carried over from `prior` (same weights => same losses).
    """
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

    model_path = MODELS_DIR / mission / f"{channel_name}.keras"
    if args.reuse_models and model_path.exists():
        # Re-thresholding only: the LSTM weights don't change, so reuse the saved model and
        # carry the loss history from the prior run (avoids ~95 min of redundant retraining).
        model = keras.models.load_model(model_path)
        losses = [
            float((prior or {}).get("initial_loss", float("nan"))),
            float((prior or {}).get("final_loss", float("nan"))),
        ]
        epochs_ran = int((prior or {}).get("epochs_ran", 0))
    else:
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
        epochs_ran = len(losses)

    predictions = model.predict(sequences, verbose=0)
    errors = np.mean((sequences - predictions) ** 2, axis=(1, 2))

    if args.threshold == "dynamic":
        # Phase 11: unsupervised pruned dynamic thresholding over the contiguous error stream.
        predicted, threshold = detect_dynamic(errors, args.ewma_span, args.prune_p)
    else:
        # Flat (Phase 2/7): semi-supervised mean + z*std of the true-normal-window errors.
        # z is tunable (Phase 11): the untuned default 3.0 over-flags; ~4.0 lifts both F1 and
        # CEF0.5 (fewer false positives) -- see tune_threshold.py for the operating curve.
        threshold = dynamic_threshold(errors[normal_mask], z_score=args.z_score)
        predicted = errors > threshold
    actual = window_is_anomaly

    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if args.save_models and not (args.reuse_models and model_path.exists()):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

    # Persist a SPARSE per-window prediction record so evaluate.py can compute an
    # interval-aware Affinity-F1 (Phase 7). Windows are at stride `seq_stride` on the
    # resampled grid, so window i covers grid units [i*stride, i*stride + window).
    # We store only the *start indices* of anomalous windows (predicted + ground-truth),
    # not the full ~16k-window arrays -- keeps the JSON small while remaining exact.
    stride = int(args.seq_stride)
    pred_starts = [int(i) * stride for i in np.nonzero(predicted)[0]]
    gt_starts = [int(i) * stride for i in np.nonzero(actual)[0]]

    record = {
        "channel": channel_name,
        "mission": mission,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": threshold,
        "threshold_method": args.threshold,
        "n_sequences": int(len(sequences)),
        "n_anomaly_windows": int(actual.sum()),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "epochs_ran": epochs_ran,
        "window": int(args.window),
        "stride": stride,
        "pred_starts": pred_starts,
        "gt_starts": gt_starts,
    }
    # Phase 14: optionally attach the DENSE continuous per-window reconstruction error
    # (errors[i] is the window starting at i*stride). This is the continuous LSTM score the
    # ensemble fuses; it is far larger than the sparse start lists, so it is written to a
    # SEPARATE dump file (not baseline_results.json) and stripped before the metrics write.
    if args.dump_window_scores:
        record["window_errors"] = [round(float(e), 8) for e in errors]
    return record


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
        "--threshold",
        choices=["flat", "dynamic"],
        default="flat",
        help="flat = Phase-2/7 mean+3sigma (reproducible reference); "
        "dynamic = Phase-11 Telemanom pruned dynamic thresholding",
    )
    p.add_argument(
        "--z-score",
        type=float,
        default=Z_SCORE,
        help="z for flat threshold mean+z*std (Phase-2/7 default 3.0; ~4.0 CEF0.5/F1-optimal)",
    )
    p.add_argument(
        "--ewma-span",
        type=int,
        default=5,
        help="EWMA smoothing span for the error stream (dynamic threshold only; <=1 disables). "
        "Kept light: ESA anomalies are short (~3 windows), so heavy smoothing hurts recall.",
    )
    p.add_argument(
        "--prune-p",
        type=float,
        default=0.13,
        help="min %% drop between consecutive sequence peaks to keep a candidate (dynamic only)",
    )
    p.add_argument(
        "--results-file",
        default="baseline_results.json",
        help="output filename within results/lstm/ (use a distinct name per threshold method "
        "to keep both reproducible, e.g. baseline_results_dynamic.json)",
    )
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
    p.add_argument(
        "--reuse-models",
        action="store_true",
        help="load the saved per-channel model instead of retraining (re-thresholding only); "
        "loss history is carried over from --loss-source",
    )
    p.add_argument(
        "--loss-source",
        default="baseline_results.json",
        help="prior results file (in results/lstm/) to copy loss history from when --reuse-models",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip channels already present in results/lstm/baseline_results.json",
    )
    p.add_argument(
        "--dump-window-scores",
        default=None,
        help="Phase 14: also write the DENSE continuous per-window reconstruction error to this "
        "file (within results/lstm/) for score-level fusion. Keep --results-file distinct from "
        "the canonical baseline_results.json so the calibrated metrics file is not overwritten. "
        "Typical: --reuse-models --threshold flat --z-score 4.0 --dump-window-scores "
        "window_scores.json --results-file baseline_results_scoredump.json",
    )
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


def _atomic_write_json(obj, path: Path) -> None:
    """Write JSON via temp + os.replace so an interrupted write never corrupts the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def write_output(all_results: list[dict], args: argparse.Namespace, results_file: Path) -> dict:
    """Atomically (temp + replace) write the current results to disk and return the summary.

    Called after every channel so a death costs at most one channel, not the whole run
    (train_lstm.py is otherwise a single long pass with no checkpointing).

    When --dump-window-scores is set, the dense per-window error arrays are split into a
    SEPARATE dump file (keyed by mission/channel) and stripped from the metrics file, so
    baseline_results.json stays slim and the dump carries everything the ensemble needs.
    """
    if args.dump_window_scores:
        dump_path = RESULTS_DIR / args.dump_window_scores
        dump_channels = [
            {
                "mission": r["mission"],
                "channel": r["channel"],
                "window": r.get("window"),
                "stride": r.get("stride"),
                "threshold": r.get("threshold"),
                # errors[i] is the reconstruction error for the window starting at i*stride.
                "window_errors": r["window_errors"],
            }
            for r in all_results
            if "window_errors" in r
        ]
        _atomic_write_json(
            {
                "config": vars(args) | {"data_dir": str(args.data_dir)},
                "note": "window_errors[i] = LSTM reconstruction error for window start i*stride; "
                "map a test window via i = start_idx // stride.",
                "channels": dump_channels,
            },
            dump_path,
        )
        # Slim copy for the metrics file: drop the dense arrays.
        metrics_results = [
            {k: v for k, v in r.items() if k != "window_errors"} for r in all_results
        ]
    else:
        metrics_results = all_results

    summary = summarize(metrics_results)
    output = {
        "summary": summary,
        "config": vars(args) | {"data_dir": str(args.data_dir)},
        "channels": metrics_results,
    }
    _atomic_write_json(output, results_file)
    return summary


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"keras backend: {keras.backend.backend()}  |  raw data: {args.data_dir}")
    print(f"models -> {MODELS_DIR if args.save_models else '(not saved)'}")
    print(f"threshold method: {args.threshold}  |  output: results/lstm/{args.results_file}")

    results_file = RESULTS_DIR / args.results_file

    # --reuse-models: carry loss history from a prior run keyed by (mission, channel), since
    # loading a saved model doesn't regenerate it.
    loss_lookup: dict[tuple[str, str], dict] = {}
    if args.reuse_models:
        loss_file = RESULTS_DIR / args.loss_source
        if loss_file.exists():
            prior_run = json.loads(loss_file.read_text())
            loss_lookup = {
                (c.get("mission"), c.get("channel")): c for c in prior_run.get("channels", [])
            }
            print(f"reuse-models: loaded loss history for {len(loss_lookup)} channel(s)")
        else:
            print(f"reuse-models: no loss source at {loss_file} (loss fields will be NaN)")

    # --resume: keep channels already scored in a prior run; skip re-training them.
    all_results: list[dict] = []
    done: set[tuple[str, str]] = set()
    if args.resume and results_file.exists():
        prior_results = json.loads(results_file.read_text())
        all_results = prior_results.get("channels", [])
        done = {(c.get("mission"), c.get("channel")) for c in all_results}
        print(f"resume: {len(done)} channel(s) already scored, skipping those")

    for mission_dir in discover_missions(args.data_dir, args.missions):
        mission = mission_dir.name
        labels = load_labels(mission_dir)
        channels = list(iter_channels(mission_dir, target_only=True))[: args.max_channels]
        print(f"\n{mission}: training {len(channels)} channel(s)")

        for channel, channel_file in tqdm(channels, desc=f"  {mission}"):
            if (mission, channel) in done:
                continue
            series = resample_series(load_channel_series(channel_file), args.resample)
            if len(series) < args.window:
                all_results.append({"channel": channel, "mission": mission, "error": "too short"})
            else:
                point_mask = anomaly_mask_for_channel(series.index, channel, labels)
                all_results.append(
                    train_channel_model(
                        series.values,
                        point_mask,
                        channel,
                        mission,
                        args,
                        prior=loss_lookup.get((mission, channel)),
                    )
                )
            # Flush after every channel so an interruption is a cheap re-run (with --resume).
            write_output(all_results, args, results_file)

    summary = write_output(all_results, args, results_file)

    print("\nLSTM baseline summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {results_file}")


if __name__ == "__main__":
    main()
