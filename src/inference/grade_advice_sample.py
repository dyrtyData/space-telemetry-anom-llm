"""Grade the fine-tuned text model's ADVICE quality semantically (Phase 9).

Phase 5/6 measured *detection* (F1/CEF) and the *structural* compliance of the advice
("99.6% of flags emit DIAGNOSIS+ADVICE text"). This phase answers the harder question:
of the advice the model emits, how much is actually **correct, actionable, and grounded**?
The judge is the in-session Claude (the session model) -- no API, no fine-tuning -- mirroring
the Phase-6 frontier pattern so the sample is frozen, reproducible, and the judging resumable.

Inputs (already on disk):
  - results/inference_test.json     fine-tuned text model, n=4500 (Phase 4/5). The advice IS
                                    each record's `actual_response` (truncated to 300 chars,
                                    which clips the trailing ACTION line -> grade DIAGNOSIS+ADVICE).
  - data/splits/test_with_advice.jsonl   per-window context (instruction, normalized values,
                                    metadata incl. start/end time and true anomaly_ratio).
  - data/labels/anomaly_advice.json gold advice (7,457 records) -- attached as an OPTIONAL
                                    per-(mission,channel) reference, not a strict join key.

Two modes (mirror src/inference/select_frontier_sample.py):

  --select  [--n 120]
      Freeze a seed-42 sample of the model's ANOMALY predictions, preserving the population's
      true-positive / false-positive ratio so the judge sees both correctly-flagged anomalies
      and false alarms. Writes a leak-complete judging file (the judge SHOULD see ground truth
      to grade correctness) to data/advice_grading/advice_sample.jsonl.

  --assemble PATH
      Read the frozen sample + a judgments JSON (list of
      {index, correctness, actionability, grounding, note}; each score 0-2) and write
      results/advice_grading_sample.json with overall + TP/FP-split summary statistics.

Usage:
    python src/inference/grade_advice_sample.py --select --n 120
    # ... in-session Claude scores each record -> results/advice_judgments.json ...
    python src/inference/grade_advice_sample.py --assemble results/advice_judgments.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

INFER_FILE = Path("results/inference_test.json")
TEST_FILE = Path("data/splits/test_with_advice.jsonl")
GOLD_FILE = Path("data/labels/anomaly_advice.json")
SAMPLE_FILE = Path("data/advice_grading/advice_sample.jsonl")
RESULTS_FILE = Path("results/advice_grading_sample.json")

SEED = 42


# --------------------------------------------------------------------------- #
# Select
# --------------------------------------------------------------------------- #
def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _gold_by_channel() -> dict[tuple, list[dict]]:
    """Index gold advice records by (mission, channel) for time-overlap lookup."""
    if not GOLD_FILE.exists():
        return {}
    gold = json.loads(GOLD_FILE.read_text())
    out: dict[tuple, list[dict]] = defaultdict(list)
    for g in gold:
        out[(g["mission"], g["channel"])].append(g)
    return out


def _gold_reference(meta: dict, gold_idx: dict[tuple, list[dict]]) -> dict | None:
    """Find the gold-advice record whose interval overlaps this window, if any.

    The gold advice is per anomaly *event*; a window is one slice of that event, so its
    start_time rarely equals the event start. We match by time-interval overlap within the
    same (mission, channel). Returns the overlapping (or nearest-start) gold record, trimmed
    to the fields a judge needs -- or None when the channel has no gold advice (e.g. the
    window is genuinely nominal, a false alarm).
    """
    candidates = gold_idx.get((meta["mission"], meta["channel"]))
    if not candidates:
        return None
    ws, we = _parse_time(meta.get("start_time")), _parse_time(meta.get("end_time"))
    best, best_overlap = None, None
    for g in candidates:
        gs, ge = _parse_time(g.get("start_time")), _parse_time(g.get("end_time"))
        if ws and we and gs and ge:
            overlap = (min(we, ge) - max(ws, gs)).total_seconds()
            if best_overlap is None or overlap > best_overlap:
                best, best_overlap = g, overlap
    chosen = best if (best is not None and best_overlap is not None and best_overlap > 0) else None
    if chosen is None:
        return None
    return {
        "pattern": chosen.get("pattern"),
        "severity": chosen.get("severity"),
        "subsystem": chosen.get("subsystem"),
        "physical_unit": chosen.get("physical_unit"),
        "advice": chosen.get("advice"),
        "recommended_action": chosen.get("recommended_action"),
    }


def select(n: int) -> list[dict]:
    """Freeze a seed-42 sample of ANOMALY predictions, preserving the TP/FP ratio."""
    if not INFER_FILE.exists():
        raise FileNotFoundError(
            f"{INFER_FILE} missing -- regenerate with `make eval-llm LIMIT=0` (Phase 9 readiness)."
        )
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_FILE}")

    infer = json.loads(INFER_FILE.read_text())["results"]
    test = [json.loads(line) for line in TEST_FILE.open()]
    gold_idx = _gold_by_channel()

    flagged = [r for r in infer if r.get("predicted") == "ANOMALY"]
    tp = [r for r in flagged if r["is_anomaly"]]  # correctly flagged true anomalies
    fp = [r for r in flagged if not r["is_anomaly"]]  # false alarms (window truly nominal)

    # Preserve the population's TP fraction in the sample (representative, honest).
    frac_tp = len(tp) / len(flagged) if flagged else 0
    n_tp = round(n * frac_tp)
    n_fp = n - n_tp

    import random

    rng = random.Random(SEED)
    rng.shuffle(tp)
    rng.shuffle(fp)
    picked = sorted(tp[:n_tp] + fp[:n_fp], key=lambda r: r["index"])

    rows = []
    for r in picked:
        idx = r["index"]
        meta = test[idx]["metadata"] if idx < len(test) else {}
        rows.append(
            {
                "index": idx,
                "mission": r["mission"],
                "channel": r["channel"],
                "is_anomaly": r["is_anomaly"],  # ground truth (judge sees this)
                "true_anomaly_ratio": meta.get("anomaly_ratio"),
                "window_start": meta.get("start_time"),
                "window_end": meta.get("end_time"),
                "instruction": test[idx]["instruction"] if idx < len(test) else "",
                "model_response": r.get("actual_response", ""),  # DIAGNOSIS/ADVICE (<=300 chars)
                "gold_reference": _gold_reference(meta, gold_idx) if meta else None,
            }
        )
    return rows


def do_select(n: int) -> None:
    rows = select(n)
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_tp = sum(1 for r in rows if r["is_anomaly"])
    print(
        f"Wrote {len(rows)} flagged windows to {SAMPLE_FILE} "
        f"({n_tp} true anomalies / {len(rows) - n_tp} false alarms)."
    )
    print(f"Frozen indices (seed={SEED}): {[r['index'] for r in rows]}")


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
RUBRIC = ("correctness", "actionability", "grounding")


def do_assemble(judgments_path: Path) -> None:
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Run --select first: {SAMPLE_FILE} missing")
    sample = {json.loads(line)["index"]: json.loads(line) for line in SAMPLE_FILE.open()}
    judged = {j["index"]: j for j in json.loads(Path(judgments_path).read_text())}

    missing = sorted(set(sample) - set(judged))
    if missing:
        raise ValueError(
            f"{len(missing)} sampled windows have no judgment (indices {missing[:10]}...). "
            "Every window in the frozen sample must be scored."
        )

    records = []
    for idx in sorted(sample):
        s, j = sample[idx], judged[idx]
        scores = {k: int(j[k]) for k in RUBRIC}
        for k, v in scores.items():
            if not 0 <= v <= 2:
                raise ValueError(f"index {idx}: {k}={v} out of range 0-2")
        records.append(
            {
                "index": idx,
                "mission": s["mission"],
                "channel": s["channel"],
                "is_anomaly": s["is_anomaly"],
                **scores,
                "total": sum(scores.values()),  # 0-6
                "note": j.get("note", ""),
            }
        )

    def stats(rows: list[dict]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        out = {"n": n}
        for k in RUBRIC:
            out[f"mean_{k}"] = round(sum(r[k] for r in rows) / n, 3)
        out["mean_total"] = round(sum(r["total"] for r in rows) / n, 3)
        # "Good advice" = scores >=1 on every rubric axis AND total >=4/6.
        out["pct_correct"] = round(sum(1 for r in rows if r["correctness"] >= 1) / n, 4)
        out["pct_actionable"] = round(sum(1 for r in rows if r["actionability"] >= 1) / n, 4)
        out["pct_grounded"] = round(sum(1 for r in rows if r["grounding"] >= 1) / n, 4)
        out["pct_high_quality"] = round(
            sum(1 for r in rows if r["total"] >= 4 and min(r[k] for k in RUBRIC) >= 1) / n, 4
        )
        return out

    tp_rows = [r for r in records if r["is_anomaly"]]
    fp_rows = [r for r in records if not r["is_anomaly"]]

    summary = {
        "approach": "Advice quality (semantic, in-session judge)",
        "judge": "Claude (session model) — no API, no fine-tuning",
        "n_samples": len(records),
        "seed": SEED,
        "rubric": "correctness / actionability / grounding, each 0-2 (total 0-6)",
        "overall": stats(records),
        "true_positives": stats(tp_rows),  # advice on correctly-flagged real anomalies
        "false_positives": stats(fp_rows),  # advice on false alarms (truly nominal windows)
        "note": (
            "Sampled (seed-42, TP/FP ratio preserved from the 1,898 anomaly predictions). "
            "The judge is the Claude session model; gold advice labels are synthetic "
            "(statistic-derived, Phase 1.5) and used only as an optional reference. Advice text "
            "is the model's response truncated to 300 chars (DIAGNOSIS+ADVICE; ACTION clipped). "
            "Correctness on a false alarm is necessarily low — the window is truly nominal — so "
            "the TP subset is the fairer measure of advisory quality when the model is right."
        ),
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "results": records}, indent=2))
    o = summary["overall"]
    hq = o["pct_high_quality"] * 100
    print(f"Wrote {RESULTS_FILE}")
    print(
        f"  overall: mean_total={o['mean_total']}/6  pct_high_quality={hq:.0f}%  "
        f"(TP n={summary['true_positives']['n']}, FP n={summary['false_positives']['n']})"
    )
    if tp_rows:
        t = summary["true_positives"]
        act, gnd = t["pct_actionable"] * 100, t["pct_grounded"] * 100
        print(
            f"  on true anomalies: mean_total={t['mean_total']}/6  "
            f"pct_actionable={act:.0f}%  pct_grounded={gnd:.0f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="Freeze the judging sample.")
    parser.add_argument("--n", type=int, default=120, help="Sample size (default 120).")
    parser.add_argument(
        "--assemble", default=None, help="Path to judgments JSON; write the final results file."
    )
    args = parser.parse_args()

    if args.select:
        do_select(args.n)
    if args.assemble:
        do_assemble(Path(args.assemble))
    if not args.select and not args.assemble:
        parser.error("pass --select and/or --assemble PATH")


if __name__ == "__main__":
    main()
