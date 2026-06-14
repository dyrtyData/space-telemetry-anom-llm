"""Select a frozen, reproducible stratified sample for the frontier zero-shot eval (Phase 6).

The frontier comparison has the *session model itself* (Claude) act as a zero-shot anomaly
detector on a small, fixed sample of the test split -- no API, no fine-tuning. To keep the
result reproducible we freeze the exact sample here:

  - seed-42 stratified shuffle of the 4,500-window test split,
  - keep the test split's ~25% anomalous balance,
  - take the first N (default 150).

Two modes:

  --select
      Write the frozen sample to data/frontier/frontier_sample.jsonl. Each line:
      {index, is_anomaly, mission, channel, severity, pattern, instruction, values}
      (`index` is the original line number in test_with_advice.jsonl).

  --assemble PATH
      Read the frozen sample + a classifications JSON (list of
      {index, predicted ("ANOMALY"|"NOMINAL"), response}) produced by the frontier model,
      join to ground truth, and write results/inference_frontier_sample.json in the SAME
      per-record schema as results/inference_test.json so evaluate.py can load it uniformly.

Usage:
    python src/inference/select_frontier_sample.py --select --n 150
    python src/inference/select_frontier_sample.py --assemble results/frontier_classifications.json
"""

import argparse
import json
import random
from pathlib import Path

TEST_FILE = Path("data/splits/test_with_advice.jsonl")
SAMPLE_FILE = Path("data/frontier/frontier_sample.jsonl")
RESULTS_FILE = Path("results/inference_frontier_sample.json")

SEED = 42
APPROACH = "Frontier zero-shot (Claude, n=%d sample)"


def _load_test() -> list[dict]:
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    return [json.loads(line) for line in TEST_FILE.open()]


def select(n: int) -> list[dict]:
    """Return a frozen stratified sample of size n (seed-42, ~25% anomalous)."""
    test = _load_test()
    anom = [i for i, r in enumerate(test) if r["metadata"]["is_anomaly"]]
    nom = [i for i, r in enumerate(test) if not r["metadata"]["is_anomaly"]]

    # Preserve the test split's anomalous fraction in the sample.
    frac = len(anom) / len(test)
    n_anom = round(n * frac)
    n_nom = n - n_anom

    rng = random.Random(SEED)
    rng.shuffle(anom)
    rng.shuffle(nom)
    picked = sorted(anom[:n_anom] + nom[:n_nom])

    rows = []
    for idx in picked:
        r = test[idx]
        m = r["metadata"]
        rows.append(
            {
                "index": idx,
                "is_anomaly": m["is_anomaly"],
                "mission": m["mission"],
                "channel": m["channel"],
                "severity": m.get("severity"),
                "pattern": m.get("pattern"),
                "instruction": r["instruction"],
                "values": m["values"],
            }
        )
    return rows


def do_select(n: int) -> None:
    rows = select(n)
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FILE, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_anom = sum(1 for r in rows if r["is_anomaly"])
    print(
        f"Wrote {len(rows)} windows to {SAMPLE_FILE} "
        f"({n_anom} anomalous, {len(rows) - n_anom} nominal)"
    )
    print(f"Frozen indices (seed={SEED}): {[r['index'] for r in rows]}")


def do_assemble(classifications_path: Path) -> None:
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Run --select first: {SAMPLE_FILE} missing")
    sample = {json.loads(line)["index"]: json.loads(line) for line in SAMPLE_FILE.open()}
    preds = {c["index"]: c for c in json.loads(Path(classifications_path).read_text())}

    missing = sorted(set(sample) - set(preds))
    if missing:
        raise ValueError(
            f"{len(missing)} sampled windows have no classification (indices {missing[:10]}...). "
            "The frontier model must classify every window in the frozen sample."
        )

    results = []
    for idx in sorted(sample):
        s = sample[idx]
        c = preds[idx]
        predicted = c["predicted"].upper()
        if predicted not in ("ANOMALY", "NOMINAL"):
            predicted = "UNKNOWN"
        expected = "ANOMALY" if s["is_anomaly"] else "NOMINAL"
        results.append(
            {
                "index": idx,
                "mission": s["mission"],
                "channel": s["channel"],
                "is_anomaly": s["is_anomaly"],
                "predicted": predicted,
                "correct": predicted == expected,
                "actual_response": c.get("response", "")[:300],
                "elapsed_s": 0.0,  # in-session zero-shot; no wall-clock measured
            }
        )

    n = len(results)
    tp = sum(1 for r in results if r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fp = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fn = sum(1 for r in results if r["is_anomaly"] and r["predicted"] != "ANOMALY")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "NOMINAL")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    summary = {
        "approach": APPROACH % n,
        "model": "Claude (session model, zero-shot)",
        "n_samples": n,
        "accuracy": round((tp + tn) / n, 4) if n else 0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown_responses": sum(1 for r in results if r["predicted"] == "UNKNOWN"),
        "seed": SEED,
        "partial": False,
        "note": (
            "Stratified sample only (frozen indices, seed 42). The frontier detector is the "
            "Claude session model reasoning zero-shot over the same normalized telemetry values "
            "and analyst system prompt; no fine-tuning, no API."
        ),
    }
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"Wrote {RESULTS_FILE}")
    print(f"  P={precision:.3f} R={recall:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select", action="store_true", help="Write the frozen sample.")
    parser.add_argument("--n", type=int, default=150, help="Sample size (default 150).")
    parser.add_argument(
        "--assemble",
        default=None,
        help="Path to classifications JSON; assemble the final results file.",
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
