"""Phase 14: score-level fusion of the detectors (text LLM + vision LLM + LSTM).

The three detectors are mirror images on the ESA-AD test split:

  * text LLM    — recall-oriented (default P 0.360 / R 0.609; calibratable, §6.4 / Phase 13)
  * vision LLM  — precision-oriented (P 0.769 / R 0.325)
  * LSTM        — best single detector (P 0.837 / R 0.432, z=4.0, Phase 11)

Because they make *independent* errors (numeric text vs. rendered image vs. reconstruction),
fusing their CONTINUOUS scores can trace the whole precision/recall frontier and, ideally,
Pareto-dominate any single model on F1 and CEF0.5. "Both must fire" (AND) and "either fires"
(OR) are only the two endpoints; a learned stacker finds the best point between/beyond them.

Inputs (per-window continuous scores in [0,1] unless noted):
  - text   : results/inference_test_scored.json     (Phase 13, --score)   — 4,500 windows
  - vision : results/inference_vision_scored.json    (Phase 14, --score)   — 2,000 PNGs
  - lstm   : results/lstm/window_scores.json          (train_lstm --dump-window-scores)
             dense reconstruction error per window; mapped to a test window by
             (mission, channel, start_idx) via i = start_idx // stride. Mission1 only.
  - the test split data/splits/test_with_advice.jsonl supplies each window's metadata
    (mission, channel, start_idx) for the LSTM map.

THE SHARED WINDOW SET is the 2,000 windows that have PNGs (the binding constraint). They are
exactly test-split lines 0..1999, and text/vision scored rows carry the same `index`, so the
three signals align by index (verified: vision idx i == text idx i == split line i).

Two fused variants are reported, each compared against its single models ON THE SAME windows:
  1. text+vision over ALL 2,000 shared windows (every window has both signals).
  2. text+vision+LSTM over the Mission1 subset (~1,378) where all three exist — the fairest
     head-to-head test of "does fusion beat the best single detector (LSTM)?".

Leakage control (deviation from the plan's "fit LR on the val split"): text & vision were
scored on TEST only, so to avoid train-on-test leakage WITHOUT a second (cloud) val-scoring
run, the learned stacker uses K-FOLD CROSS-VALIDATED STACKING — fit logistic regression on
k-1 folds, predict the held-out fold, concatenate the out-of-fold (OOF) predictions. No window
is scored by a model trained on it. The threshold is then swept over the OOF fused scores.
Weighted-sum, OR/AND, and 2-of-3 vote need no fitting and are reported as sanity baselines.

Usage:
    python src/inference/ensemble.py                 # default paths
    python src/inference/ensemble.py --no-plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

TEXT_SCORED = Path("results/inference_test_scored.json")
VISION_SCORED = Path("results/inference_vision_scored.json")
LSTM_DUMP = Path("results/lstm/window_scores.json")
TEST_SPLIT = Path("data/splits/test_with_advice.jsonl")

OUT_JSON = Path("results/ensemble_pr_curve.json")
OUT_PNG = Path("results/ensemble_pr_curve.png")
METRICS_ROWS = Path("results/ensemble_metrics.json")  # rows for evaluate.py

N_FOLDS = 5
SEED = 42


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cef_from_pr(precision: float, recall: float, beta: float = 0.5) -> float:
    """CEF0.5 / F-beta (beta=0.5 weights precision over recall)."""
    b2 = beta**2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> dict:
    """Confusion-matrix metrics for predicting ANOMALY when score >= thr."""
    pred = y_score >= thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": round(float(thr), 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cef_0.5": round(cef_from_pr(precision, recall), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def sweep(y_true: np.ndarray, y_score: np.ndarray, n_grid: int = 200) -> list[dict]:
    """Metrics over a uniform threshold grid spanning the score range (+ the 0.5 boundary)."""
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if hi <= lo:
        hi = lo + 1e-9
    grid = sorted(set(np.linspace(lo, hi, n_grid + 1).tolist() + [0.5]))
    return [metrics_at(y_true, y_score, t) for t in grid]


def best_by(curve: list[dict], key: str) -> dict:
    return max(curve, key=lambda p: p[key])


# --------------------------------------------------------------------------- #
# Load + align
# --------------------------------------------------------------------------- #
def _load_scored(path: Path) -> dict[int, dict]:
    """index -> scored row, from a {summary, results} scored file."""
    data = json.loads(path.read_text())
    return {r["index"]: r for r in data["results"]}


def _load_lstm_errors(path: Path) -> dict[tuple[str, str], dict]:
    """(mission, channel) -> {stride, window_errors[]} from the dense dump."""
    data = json.loads(path.read_text())
    out = {}
    for c in data["channels"]:
        out[(c["mission"], c["channel"])] = {
            "stride": int(c["stride"]),
            "errors": c["window_errors"],
        }
    return out


def lstm_score_for(meta: dict, lstm: dict[tuple[str, str], dict]) -> float | None:
    """LSTM reconstruction error for a test window, or None if no model / out of range.

    The LSTM windows are at stride `stride` on the contiguous per-channel grid, so the window
    whose start is `start_idx` is errors[start_idx // stride] (verified: every Mission1
    test-anomaly start_idx lands in the LSTM grid). Mission2/3 have no LSTM model -> None.
    """
    rec = lstm.get((meta["mission"], meta["channel"]))
    if rec is None:
        return None
    i = int(meta["start_idx"]) // rec["stride"]
    if 0 <= i < len(rec["errors"]):
        return float(rec["errors"][i])
    return None


def build_matrix() -> dict:
    """Assemble the aligned per-window signal matrix over the shared (PNG) window set."""
    text = _load_scored(TEXT_SCORED)
    vision = _load_scored(VISION_SCORED)
    lstm = _load_lstm_errors(LSTM_DUMP) if LSTM_DUMP.exists() else {}
    split = [json.loads(line) for line in TEST_SPLIT.open()]

    rows = []
    for idx in sorted(vision):  # the 2,000 PNG indices are the shared set
        v = vision[idx]
        t = text.get(idx)
        if t is None:
            continue
        meta = split[idx]["metadata"]
        # Sanity: the three sources must describe the same window.
        assert meta["mission"] == v["mission"] and meta["channel"] == v["channel"], (
            f"alignment mismatch at index {idx}"
        )
        rows.append(
            {
                "index": idx,
                "mission": meta["mission"],
                "channel": meta["channel"],
                "is_anomaly": 1 if meta["is_anomaly"] else 0,
                "text": float(t["score"]),
                "vision": float(v["score"]),
                "lstm_err": lstm_score_for(meta, lstm),
            }
        )
    return {"rows": rows, "has_lstm": bool(lstm)}


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(x)), float(np.std(x))
    return (x - mu) / sd if sd > 0 else x - mu


def oof_stack(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    """Out-of-fold logistic-regression stacking -> a leakage-free fused score per row.

    Each row's fused score is predicted by an LR fit on the OTHER folds, so no window is
    scored by a model that saw it. Features are standardized within each training fold.
    """
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for train_idx, test_idx in skf.split(X, y):
        mu = X[train_idx].mean(axis=0)
        sd = X[train_idx].std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit((X[train_idx] - mu) / sd, y[train_idx])
        oof[test_idx] = clf.predict_proba((X[test_idx] - mu) / sd)[:, 1]
    return oof


def full_stack_weights(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit one LR on all rows just to REPORT the learned per-modality weights (not for scoring)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit((X - mu) / sd, y)
    return {
        "coef": [round(float(c), 4) for c in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 4),
    }


def fuse_variant(rows: list[dict], feature_names: list[str], label: str) -> dict:
    """Build all fused scores + single-model baselines for one set of features, on `rows`."""
    y = np.array([r["is_anomaly"] for r in rows], dtype=int)
    n_pos = int(y.sum())
    base_rate = n_pos / len(y) if len(y) else 0.0

    # Per-modality score columns, normalized to a common scale. text/vision are already
    # probabilities in [0,1]; the LSTM error is unbounded positive, so z-normalize all for the
    # weighted-sum/vote (the stacker standardizes internally).
    raw = {f: np.array([r[f] for r in rows], dtype=float) for f in feature_names}
    norm = {f: _zscore(raw[f]) for f in feature_names}

    # --- single-model baselines on THIS window subset (the Pareto comparison targets) ---
    singles = {}
    for f in feature_names:
        s = raw[f]
        curve_f = sweep(y, s)
        # argmax point: 0.5 for the probability scores; for lstm_err use its own median split.
        thr = 0.5 if f in ("text", "vision") else float(np.median(s))
        singles[f] = {
            "argmax_point": metrics_at(y, s, thr),
            "auc_pr": round(float(average_precision_score(y, s)), 4),
            "best_cef_0.5": best_by(curve_f, "cef_0.5"),
            "best_f1": best_by(curve_f, "f1"),
        }

    # --- fused scores ---
    X = np.column_stack([raw[f] for f in feature_names])
    stack_oof = oof_stack(X, y)

    weighted = np.mean(np.column_stack([norm[f] for f in feature_names]), axis=1)

    # Vote: count modalities firing at their own argmax threshold; needs >= ceil(n/2) hits.
    fires = []
    for f in feature_names:
        thr = 0.5 if f in ("text", "vision") else float(np.median(raw[f]))
        fires.append((raw[f] >= thr).astype(int))
    vote_count = np.sum(np.column_stack(fires), axis=1)
    n_models = len(feature_names)
    need = (n_models // 2) + 1  # 2-of-3, or 2-of-2 (=AND) for two models
    vote_pred = (vote_count >= need).astype(int)

    fused = {
        "stacker_oof": stack_oof,
        "weighted_sum": weighted,
    }
    fused_summ = {}
    for name, score in fused.items():
        curve = sweep(y, score)
        fused_summ[name] = {
            "auc_pr": round(float(average_precision_score(y, score)), 4),
            "cef_0.5_optimal": best_by(curve, "cef_0.5"),
            "f1_optimal": best_by(curve, "f1"),
            "curve": curve,
        }

    # Vote + endpoints are hard predictions (no threshold sweep) -> single confusion matrix.
    def hard(pred: np.ndarray) -> dict:
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        tn = int(np.sum((pred == 0) & (y == 0)))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "cef_0.5": round(cef_from_pr(p, r), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    or_pred = (vote_count >= 1).astype(int)
    and_pred = (vote_count >= n_models).astype(int)
    review_bucket = int(np.sum((vote_count > 0) & (vote_count < n_models)))

    return {
        "label": label,
        "features": feature_names,
        "n_windows": len(rows),
        "n_anomalies": n_pos,
        "base_rate": round(base_rate, 4),
        "single_models": singles,
        "fused": {
            k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in fused_summ.items()
        },
        "fused_curves": {k: v["curve"] for k, v in fused_summ.items()},
        "vote": {"rule": f"{need}-of-{n_models}", **hard(vote_pred)},
        "endpoints": {"OR_any": hard(or_pred), "AND_all": hard(and_pred)},
        "disagreement_review_bucket": review_bucket,
        "stacker_weights_full_fit": full_stack_weights(X, y),
    }


# --------------------------------------------------------------------------- #
# Report rows for evaluate.py
# --------------------------------------------------------------------------- #
def _row(approach: str, pt: dict, n: int) -> dict:
    """A metric dict in evaluate.py's unified schema (validate-eval requires finite P/R/F1/CEF)."""
    return {
        "approach": approach,
        "precision": round(pt["precision"], 4),
        "recall": round(pt["recall"], 4),
        "f1": round(pt["f1"], 4),
        "cef_0.5": round(pt["cef_0.5"], 4),
        "affinity_f1": None,
        "format_compliance": 1.0,
        "advice_structured_frac": 0.0,
        "n_units": n,
        "unit": "shared windows",
        "averaging": "micro (pooled over shared windows)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not VISION_SCORED.exists():
        raise FileNotFoundError(
            f"Vision scores not found: {VISION_SCORED}\n"
            f"Run: python src/inference/eval_vision.py --score   (needs a GPU; see Phase 14)."
        )

    data = build_matrix()
    rows = data["rows"]
    print(
        f"Shared window set: {len(rows)} windows ({sum(r['is_anomaly'] for r in rows)} anomalies)"
    )

    variants = {}
    # Variant 1: text + vision over all shared windows.
    variants["text_vision"] = fuse_variant(
        rows, ["text", "vision"], "Ensemble (text+vision, stacked)"
    )

    # Variant 2: text + vision + LSTM over the Mission1 subset where all three exist.
    if data["has_lstm"]:
        m1 = [r for r in rows if r["lstm_err"] is not None]
        n_anom_m1 = sum(r["is_anomaly"] for r in m1)
        print(f"Mission1 (3-model) subset: {len(m1)} windows ({n_anom_m1} anomalies)")
        if len(m1) > 50:
            variants["text_vision_lstm"] = fuse_variant(
                m1, ["text", "vision", "lstm_err"], "Ensemble (text+vision+LSTM, stacked)"
            )

    # --- report rows (the CEF0.5-optimal stacker point per variant) ---
    report_rows = []
    for v in variants.values():
        pt = v["fused"]["stacker_oof"]["cef_0.5_optimal"]
        report_rows.append(_row(v["label"], pt, v["n_windows"]))

    out = {
        "phase": 14,
        "method": "score-level fusion; OOF k-fold logistic stacker (leakage-free), "
        "weighted-sum / 2-of-N vote / OR / AND baselines.",
        "shared_set": "2,000 windows with PNGs == test lines 0..1999 (text idx == vision idx).",
        "variants": variants,
        "report_rows": report_rows,
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    METRICS_ROWS.write_text(json.dumps(report_rows, indent=2))

    # --- console summary ---
    for key, v in variants.items():
        print(f"\n=== {v['label']} ({v['n_windows']} windows) ===")
        for f, s in v["single_models"].items():
            b = s["best_cef_0.5"]
            print(
                f"  single {f:>9}: AUC-PR={s['auc_pr']:.3f}  best-CEF0.5 "
                f"P={b['precision']:.3f} R={b['recall']:.3f} CEF={b['cef_0.5']:.3f}"
            )
        for name, fs in v["fused"].items():
            c = fs["cef_0.5_optimal"]
            print(
                f"  fused {name:>12}: AUC-PR={fs['auc_pr']:.3f}  CEF-opt "
                f"P={c['precision']:.3f} R={c['recall']:.3f} CEF={c['cef_0.5']:.3f}"
            )
        vt = v["vote"]
        print(
            f"  vote {vt['rule']}: P={vt['precision']:.3f} R={vt['recall']:.3f} "
            f"F1={vt['f1']:.3f}  | review={v['disagreement_review_bucket']}"
        )
        print(f"  stacker weights (full-fit): {v['stacker_weights_full_fit']}")

    print(f"\nWrote {OUT_JSON} and {METRICS_ROWS}")

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(
                1, len(variants), figsize=(6 * len(variants), 5), squeeze=False
            )
            for ax, (key, v) in zip(axes[0], variants.items()):
                for name, curve in v["fused_curves"].items():
                    rec = [p["recall"] for p in curve]
                    prec = [p["precision"] for p in curve]
                    ax.plot(rec, prec, lw=2, label=f"{name}")
                for f, s in v["single_models"].items():
                    p = s["argmax_point"]
                    ax.scatter([p["recall"]], [p["precision"]], zorder=5, label=f"{f} (argmax)")
                ax.axhline(v["base_rate"], ls="--", color="grey", lw=1)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(v["label"])
                ax.legend(loc="lower left", fontsize=7)
                ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_PNG, dpi=120)
            print(f"Plot written to {OUT_PNG}")
        except Exception as e:  # noqa: BLE001 - plotting is optional
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
