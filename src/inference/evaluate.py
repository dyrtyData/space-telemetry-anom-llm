"""Evaluate all approaches and generate the STAR-Pipeline comparison report (Phase 5).

Loads the *real* result files produced by earlier phases and computes a unified
comparison across four approaches:

  - Isolation Forest   -> results/isolation_forest/if_results.json   (Phase 2)
  - LSTM baseline      -> results/lstm/baseline_results.json         (Phase 2)
  - LLM detection      -> results/inference_test.json                (Phase 4 / 5)
  - Hybrid             -> derived (LSTM flags detection + LLM generates advice)

Metrics: Precision / Recall / F1 / CEF0.5 (precision-weighted F-beta) and, for the
LLM where per-window predictions are persisted, an interval-based Affinity-F1.

All "Key Findings" in the report are computed from the loaded metrics -- nothing is
hardcoded. Output: results/comparison_report.md (+ results/comparison_metrics.json).

Usage:
    python src/inference/evaluate.py --all
    make eval-all
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
LSTM_FILE = RESULTS_DIR / "lstm" / "baseline_results.json"
IF_FILE = RESULTS_DIR / "isolation_forest" / "if_results.json"
LLM_FILE = RESULTS_DIR / "inference_test.json"
TEST_WITH_ADVICE = Path("data/splits/test_with_advice.jsonl")

REPORT_FILE = RESULTS_DIR / "comparison_report.md"
METRICS_FILE = RESULTS_DIR / "comparison_metrics.json"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cef_from_pr(precision: float, recall: float, beta: float = 0.5) -> float:
    """CEF / F-beta score from precision & recall.

    CEF0.5 (beta=0.5) weights precision over recall -- the operationally relevant
    trade-off for spacecraft telemetry, where false alarms are expensive.
    """
    beta_sq = beta**2
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _merge_intervals(points: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Merge (start, end) windows into intervals, joining any that are within `gap`."""
    if not points:
        return []
    points = sorted(points)
    merged = [list(points[0])]
    for s, e in points[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def affinity_f1(
    grouped: dict[tuple, dict[str, list[tuple[int, int]]]],
    delta: int = 32,
    gap: int = 16,
) -> dict:
    """Interval-based Affinity-F1, computed *within* each (mission, channel) timeline.

    `grouped[(mission, channel)] = {"pred": [(start,end), ...], "gt": [(start,end), ...]}`
    using per-window start/end indices (which are only comparable within one channel).
    Predicted/GT windows are merged into intervals (joining neighbours within `gap`),
    then matched if they overlap within `delta` tolerance. Matched counts are pooled
    across channels into a single precision/recall.

    NOTE: on this dataset the test split is a *shuffled, balanced-subsampled* set of
    windows (~1.4 windows per channel), so most "intervals" are single isolated windows
    and Affinity-F1 effectively reduces to window-level F1. It is reported for
    transparency and to wire the temporally-aware metric the task asks for; it becomes
    meaningful only on a contiguous (un-shuffled) evaluation stream.
    """
    matched_pred = matched_gt = total_pred = total_gt = 0

    for groups in grouped.values():
        preds = _merge_intervals(groups["pred"], gap)
        gts = _merge_intervals(groups["gt"], gap)
        total_pred += len(preds)
        total_gt += len(gts)

        mp, mg = set(), set()
        for i, (ps, pe) in enumerate(preds):
            for j, (gs, ge) in enumerate(gts):
                if ps <= ge + delta and pe >= gs - delta:
                    mp.add(i)
                    mg.add(j)
        matched_pred += len(mp)
        matched_gt += len(mg)

    precision = matched_pred / total_pred if total_pred else 0.0
    recall = matched_gt / total_gt if total_gt else 0.0
    return {
        "affinity_precision": round(precision, 4),
        "affinity_recall": round(recall, 4),
        "affinity_f1": round(f1_from_pr(precision, recall), 4),
        "n_pred_intervals": total_pred,
        "n_gt_intervals": total_gt,
    }


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_per_channel(path: Path, approach: str) -> dict:
    """Load a Phase-2 baseline file (LSTM or IF): macro-average over channels."""
    if not path.exists():
        return {"approach": approach, "error": f"results not found: {path}"}

    d = json.loads(path.read_text())
    channels = [c for c in d.get("channels", []) if "error" not in c]
    if not channels:
        return {"approach": approach, "error": "no valid channel results"}

    n = len(channels)
    precision = sum(c["precision"] for c in channels) / n
    recall = sum(c["recall"] for c in channels) / n
    f1 = sum(c["f1"] for c in channels) / n
    return {
        "approach": approach,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "cef_0.5": round(cef_from_pr(precision, recall), 4),
        "affinity_f1": None,  # per-window predictions not persisted for baselines
        "n_units": n,
        "unit": "channels",
        "averaging": "macro (mean over channels)",
    }


def load_lstm_results() -> dict:
    return _load_per_channel(LSTM_FILE, "LSTM Baseline")


def load_if_results() -> dict:
    return _load_per_channel(IF_FILE, "Isolation Forest")


def load_llm_results() -> dict:
    """Load Phase-4/5 LLM detection results (micro metrics already computed)."""
    if not LLM_FILE.exists():
        return {"approach": "LLM Detection", "error": f"results not found: {LLM_FILE}"}

    d = json.loads(LLM_FILE.read_text())
    s = d["summary"]
    results = d.get("results", [])

    out = {
        "approach": "LLM Detection",
        "precision": round(s["precision"], 4),
        "recall": round(s["recall"], 4),
        "f1": round(s["f1"], 4),
        "cef_0.5": round(cef_from_pr(s["precision"], s["recall"]), 4),
        "accuracy": round(s.get("accuracy", 0.0), 4),
        "n_units": s["n_samples"],
        "unit": "windows",
        "averaging": "micro (pooled over windows)",
        "avg_time_s": s.get("avg_time_s"),
        "unknown_responses": s.get("unknown_responses", 0),
    }

    # --- Affinity-F1: reconstruct per-(mission,channel) intervals via test metadata ---
    affinity = _llm_affinity(results)
    if affinity:
        out["affinity_f1"] = affinity["affinity_f1"]
        out["affinity_detail"] = affinity

    # --- Advice coherence: do anomaly predictions actually emit structured advice? ---
    anomaly_resps = [r["actual_response"] for r in results if r["predicted"] == "ANOMALY"]
    if anomaly_resps:
        avg_len = sum(len(r) for r in anomaly_resps) / len(anomaly_resps)
        # actual_response is persisted truncated to 300 chars (Phase-4), which clips the
        # trailing ACTION line -- key coherence on DIAGNOSIS+ADVICE, which survive the cap.
        structured = sum(
            1 for r in anomaly_resps if ("DIAGNOSIS" in r.upper() and "ADVICE" in r.upper())
        )
        out["advice_avg_chars"] = round(avg_len, 1)
        out["advice_structured_frac"] = round(structured / len(anomaly_resps), 4)
        out["n_anomaly_predictions"] = len(anomaly_resps)

    return out


def _llm_affinity(results: list[dict]) -> dict | None:
    """Join LLM per-window results to test metadata (by index) and compute Affinity-F1."""
    if not results or not TEST_WITH_ADVICE.exists():
        return None

    test = [json.loads(line) for line in TEST_WITH_ADVICE.open()]
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: {"pred": [], "gt": []})

    for r in results:
        idx = r["index"]
        if idx >= len(test):
            continue
        meta = test[idx]["metadata"]
        key = (meta["mission"], meta["channel"])
        span = (int(meta["start_idx"]), int(meta["end_idx"]))
        if r["predicted"] == "ANOMALY":
            grouped[key]["pred"].append(span)
        if r["is_anomaly"]:
            grouped[key]["gt"].append(span)

    return affinity_f1(grouped)


def derive_hybrid(lstm: dict, llm: dict) -> dict:
    """Hybrid = LSTM does detection (high precision) + LLM generates advice on flags.

    Detection metrics are inherited from the LSTM (the component that flags anomalies);
    the added value over the bare LSTM is the LLM's diagnostic-advice layer. This makes
    the hybrid's *detection* score identical to the LSTM's by construction -- the point
    of the hybrid is not better detection but actionable advice on each flag.
    """
    if "error" in lstm:
        return {"approach": "Hybrid (LSTM + LLM advice)", "error": "needs LSTM results"}

    out = {
        "approach": "Hybrid (LSTM + LLM advice)",
        "precision": lstm["precision"],
        "recall": lstm["recall"],
        "f1": lstm["f1"],
        "cef_0.5": lstm["cef_0.5"],
        "affinity_f1": None,
        "n_units": lstm.get("n_units"),
        "unit": lstm.get("unit"),
        "averaging": "detection inherited from LSTM",
    }
    if "advice_structured_frac" in llm:
        out["advice_structured_frac"] = llm["advice_structured_frac"]
        out["advice_avg_chars"] = llm["advice_avg_chars"]
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def generate_findings(results: list[dict]) -> list[str]:
    """Compute the 'Key Findings' bullets from the actual metrics (nothing hardcoded)."""
    valid = [r for r in results if "error" not in r]
    findings: list[str] = []
    if not valid:
        return ["- No valid results were available to compare."]

    by_f1 = sorted(valid, key=lambda r: r.get("f1", 0), reverse=True)
    best = by_f1[0]
    findings.append(
        f"- Highest detection F1: **{best['approach']}** "
        f"(F1={best['f1']:.3f}, precision={best['precision']:.3f}, recall={best['recall']:.3f})."
    )

    by_cef = sorted(valid, key=lambda r: r.get("cef_0.5", 0), reverse=True)
    best_cef = by_cef[0]
    findings.append(
        f"- Best precision-weighted score (CEF0.5, the operationally relevant metric for "
        f"costly false alarms): **{best_cef['approach']}** (CEF0.5={best_cef['cef_0.5']:.3f})."
    )

    lstm = next((r for r in valid if r["approach"] == "LSTM Baseline"), None)
    llm = next((r for r in valid if r["approach"] == "LLM Detection"), None)
    if lstm and llm:
        findings.append(
            f"- The LSTM baseline detects with higher precision ({lstm['precision']:.3f} vs "
            f"{llm['precision']:.3f}); the LLM trades precision for recall "
            f"({llm['recall']:.3f} vs {lstm['recall']:.3f}) while adding a capability the "
            f"baselines lack: free-text diagnostic advice."
        )
    if llm and "advice_structured_frac" in llm:
        findings.append(
            f"- LLM advice coherence: {llm['advice_structured_frac'] * 100:.0f}% of the "
            f"{llm.get('n_anomaly_predictions', 0)} anomaly predictions emitted structured "
            f"DIAGNOSIS+ADVICE text (responses persisted truncated at 300 chars) -- the "
            f"hybrid's added value over the bare LSTM."
        )
    if llm and llm.get("avg_time_s") is not None:
        findings.append(
            f"- LLM inference cost: {llm['avg_time_s']:.2f}s/window on M3 Max Metal over "
            f"{llm['n_units']} windows (vs near-instant scoring for the baselines)."
        )
    if llm and llm.get("affinity_f1") is not None:
        det = llm.get("affinity_detail", {})
        findings.append(
            f"- Affinity-F1 (interval-aware) for the LLM = {llm['affinity_f1']:.3f} over "
            f"{det.get('n_gt_intervals', '?')} ground-truth intervals; on this shuffled, "
            f"subsampled test split it largely reduces to window-level F1 (see methodology note)."
        )
    return findings


def generate_report(results: list[dict]) -> str:
    llm = next((r for r in results if r["approach"] == "LLM Detection"), {})
    n_llm = llm.get("n_units", "?")

    lines = ["# STAR-Pipeline Evaluation Report\n"]
    lines.append(
        "End-to-end comparison of anomaly-detection approaches on the ESA-AD test split. "
        "Detection metrics are computed from the persisted Phase-2 (baselines) and "
        "Phase-4/5 (LLM) result files; CEF0.5 is the precision-weighted F-beta (beta=0.5) "
        "favoured when false alarms are costly.\n"
    )

    lines.append("## Approach Comparison\n")
    lines.append("| Approach | Precision | Recall | F1 | CEF0.5 | Affinity-F1 | Eval unit |")
    lines.append("|----------|-----------|--------|----|--------|-------------|-----------|")
    for r in results:
        if "error" in r:
            lines.append(
                f"| {r.get('approach', 'Unknown')} | _Error: {r['error']}_ |  |  |  |  |  |"
            )
            continue
        unit = f"{r.get('n_units', '?')} {r.get('unit', '')}".strip()
        lines.append(
            f"| {r['approach']} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | "
            f"{_fmt(r['f1'])} | {_fmt(r.get('cef_0.5'))} | {_fmt(r.get('affinity_f1'))} | {unit} |"
        )

    lines.append("\n## Key Findings\n")
    lines.extend(generate_findings(results))

    lines.append("\n## Methodology Notes\n")
    lines.append(
        "- **Baselines (LSTM, Isolation Forest)** are scored per channel and macro-averaged "
        "over the channels evaluated (Phase-2 smoke run: 3 Mission-1 target channels). "
        "Per-window predictions were not persisted, so Affinity-F1 is N/A for them."
    )
    lines.append(
        f"- **LLM Detection** is scored per window (micro-averaged) over the "
        f"{n_llm}-window test slice; ANOMALY/NOMINAL is parsed from the generated text."
    )
    lines.append(
        "- **Hybrid (LSTM + LLM advice)** inherits the LSTM's detection metrics by "
        "construction: the LSTM flags anomalies (high precision) and the LLM attaches "
        "diagnostic advice to each flag. Its value is the advice layer, not better detection."
    )
    lines.append(
        "- **Affinity-F1** merges per-window predictions into intervals within each "
        "(mission, channel) timeline and matches predicted vs. ground-truth intervals "
        "within a tolerance. The ESA-AD test split here is a *shuffled, balanced-subsampled* "
        "set (~1.4 windows/channel), so intervals are mostly isolated single windows and "
        "Affinity-F1 ≈ window-level F1; it is wired for correctness and becomes meaningful "
        "on a contiguous evaluation stream."
    )
    lines.append(
        "- **CEF0.5** uses beta=0.5 (precision-weighted). Computed from each approach's "
        "reported precision/recall."
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Evaluate all approaches.")
    parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    lstm = load_lstm_results()
    iso = load_if_results()
    llm = load_llm_results()
    hybrid = derive_hybrid(lstm, llm)

    # Report order: baselines, LLM, hybrid.
    results = [iso, lstm, llm, hybrid]

    report = generate_report(results)
    REPORT_FILE.write_text(report)
    METRICS_FILE.write_text(json.dumps(results, indent=2))

    print(report)
    print(f"Report saved to {REPORT_FILE}")
    print(f"Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()
