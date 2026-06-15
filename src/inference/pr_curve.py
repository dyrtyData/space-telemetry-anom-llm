"""Phase 13: precision-recall calibration for the text LLM detector.

The text LLM is reported at a single hard operating point (P 0.360 / R 0.609 — it
over-flags). This script turns the per-window continuous verdict scores produced by
``test_local_gguf.py --score`` (``results/inference_test_scored.json``) into a full
precision-recall curve: it sweeps a decision threshold over the score, computes
P / R / F1 / CEF0.5 at each, reports the AUC-PR (average precision), and surfaces a
higher-precision operating point that beats the default 0.360 precision at acceptable
recall.

CEF0.5 is the precision-weighted F-beta used throughout the project (beta=0.5), so the
curve is directly comparable to the detection bake-off in the analysis doc §6.1.

Usage:
    python src/inference/pr_curve.py                 # default in/out paths
    python src/inference/pr_curve.py --no-plot       # skip the PNG
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

SCORED_FILE = Path("results/inference_test_scored.json")
HARD_FILE = Path("results/inference_test.json")  # original generation run (for the default point)
OUT_JSON = Path("results/llm_pr_curve.json")
OUT_PNG = Path("results/llm_pr_curve.png")


def cef_from_pr(precision: float, recall: float, beta: float = 0.5) -> float:
    """CEF0.5 / F-beta (beta=0.5 weights precision over recall)."""
    beta_sq = beta**2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default=str(SCORED_FILE))
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--no-plot", action="store_true", help="Skip writing the PNG.")
    ap.add_argument(
        "--n-grid", type=int, default=200, help="Threshold grid resolution in [0,1]. Default 200."
    )
    args = ap.parse_args()

    scored_path = Path(args.scored)
    if not scored_path.exists():
        raise FileNotFoundError(
            f"Scored results not found: {scored_path}\n"
            f"Run: python src/inference/test_local_gguf.py --score --limit 0 --resume"
        )

    data = json.loads(scored_path.read_text())
    rows = data["results"]
    y_true = np.array([1 if r["is_anomaly"] else 0 for r in rows], dtype=int)
    y_score = np.array([r["score"] for r in rows], dtype=float)
    n_pos = int(y_true.sum())
    base_rate = n_pos / len(y_true)

    # AUC-PR (average precision) — the threshold-free summary of the curve.
    auc_pr = float(average_precision_score(y_true, y_score))
    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(y_true, y_score)

    # Threshold sweep: a uniform grid plus the natural 0.5 argmax boundary.
    grid = sorted(set(np.linspace(0.0, 1.0, args.n_grid + 1).tolist() + [0.5]))
    curve = [metrics_at(y_true, y_score, t) for t in grid]

    # Operating points of interest.
    argmax_pt = metrics_at(y_true, y_score, 0.5)
    cef_opt = max(curve, key=lambda p: p["cef_0.5"])
    f1_opt = max(curve, key=lambda p: p["f1"])
    # Highest-recall point whose precision clears 0.60 (a deployable false-alarm budget,
    # ~1 wrong alarm per 1.5 true ones vs. the default's ~2:1).
    hi_prec = [p for p in curve if p["precision"] >= 0.60 and p["recall"] > 0.0]
    hi_prec_pt = max(hi_prec, key=lambda p: p["recall"]) if hi_prec else None

    # The default reported operating point comes from the original generation run
    # (hard ANOMALY/NOMINAL verdicts), preserved for an apples-to-apples comparison.
    default_pt = None
    if HARD_FILE.exists():
        s = json.loads(HARD_FILE.read_text())["summary"]
        default_pt = {
            "source": "generation hard-verdict (results/inference_test.json)",
            "precision": s["precision"],
            "recall": s["recall"],
            "f1": s["f1"],
            "cef_0.5": round(cef_from_pr(s["precision"], s["recall"]), 4),
        }

    out = {
        "approach": "LLM detection (text, Qwen3-8B) — PR calibration",
        "n_samples": len(y_true),
        "n_anomalies": n_pos,
        "base_rate": round(base_rate, 4),
        "auc_pr": round(auc_pr, 4),
        "auc_pr_baseline_random": round(base_rate, 4),  # AP of a random classifier = prevalence
        "scoring": "verdict-token logprob: P(ANOMALY) = softmax(logit_ANOMALY, logit_NOMINAL)",
        "operating_points": {
            "default_hard_verdict": default_pt,
            "argmax_score_0.5": argmax_pt,
            "cef_0.5_optimal": cef_opt,
            "f1_optimal": f1_opt,
            "high_precision_p>=0.60": hi_prec_pt,
        },
        "curve": curve,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # --- console summary ---
    print(f"=== Text-LLM PR calibration ({len(y_true)} windows, {n_pos} anomalies) ===")
    print(f"  AUC-PR (avg precision): {auc_pr:.4f}   (random floor = base rate {base_rate:.3f})")
    if default_pt:
        print(
            f"  default (hard verdict): P={default_pt['precision']:.3f} "
            f"R={default_pt['recall']:.3f} F1={default_pt['f1']:.3f} "
            f"CEF0.5={default_pt['cef_0.5']:.3f}"
        )
    for name, p in [
        ("argmax @0.5", argmax_pt),
        ("CEF0.5-optimal", cef_opt),
        ("F1-optimal", f1_opt),
        ("high-prec (P>=.60)", hi_prec_pt),
    ]:
        if p:
            print(
                f"  {name:>20}: thr={p['threshold']:.3f}  P={p['precision']:.3f} "
                f"R={p['recall']:.3f}  F1={p['f1']:.3f}  CEF0.5={p['cef_0.5']:.3f}"
            )
    print(f"\nCurve written to {out_path}")

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec_arr, prec_arr, color="#1f77b4", lw=2, label=f"PR curve (AP={auc_pr:.3f})")
            ax.axhline(base_rate, ls="--", color="grey", lw=1, label=f"random ({base_rate:.2f})")
            for name, p, c in [
                ("default", default_pt, "#d62728"),
                ("CEF0.5-opt", cef_opt, "#2ca02c"),
                ("P≥.60", hi_prec_pt, "#9467bd"),
            ]:
                if p:
                    ax.scatter([p["recall"]], [p["precision"]], color=c, zorder=5, label=name)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Text LLM (Qwen3-8B) — precision-recall calibration")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_PNG, dpi=120)
            print(f"Plot written to {OUT_PNG}")
        except Exception as e:  # noqa: BLE001 - plotting is optional
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
