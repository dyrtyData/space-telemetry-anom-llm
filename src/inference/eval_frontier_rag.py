"""Evaluate the frontier model (Claude) with RAG context on the frozen sample.

This script runs the frontier model with RAG-retrieved context — giving it the same
channel-specific history that the fine-tune implicitly learned. Comparison targets:
- Phase 6 frontier zero-shot (F1 ~0.254) - no context
- Phase 15 frontier + RAG - with k retrieved examples
- Fine-tune (F1 ~0.453) - context burned into weights

Usage:
    # Build RAG index first (one-time)
    python src/inference/build_rag_index.py --train data/splits/train.jsonl --out data/rag/

    # Run RAG-augmented frontier eval (in-session, no API cost on Max plan)
    python src/inference/eval_frontier_rag.py --sample data/frontier/frontier_sample.jsonl \
        --k 5 --out results/frontier_rag.json

    # OR: Generate prompts only for manual classification
    python src/inference/eval_frontier_rag.py --sample data/frontier/frontier_sample.jsonl \
        --k 5 --prompts-only --out data/frontier/frontier_rag_prompts.jsonl
"""

import argparse
import json
from pathlib import Path

from rag_retrieve import RAGRetriever, format_rag_context

SAMPLE_FILE = Path("data/frontier/frontier_sample.jsonl")
RESULTS_FILE = Path("results/inference_frontier_rag.json")
PROMPTS_FILE = Path("data/frontier/frontier_rag_prompts.jsonl")
RAG_DIR = Path("data/rag/")

SEED = 42
APPROACH = "Frontier (Claude) + RAG (k=%d)"


def build_rag_prompt(
    instruction: str,
    mission: str,
    channel: str,
    values: list[float],
    retriever: RAGRetriever,
    k: int = 5,
    include_labels: bool = True,
) -> str:
    """Build a RAG-augmented prompt for the frontier model."""
    neighbors = retriever.retrieve(mission, channel, values, k=k, include_labels=include_labels)
    context = format_rag_context(neighbors, mission, channel, include_labels=include_labels)

    prompt = f"""You are a spacecraft telemetry analyst. Below are {k} historical windows \
from channel {channel} that are similar to the current window, along with their \
ground-truth labels:

{context}

Now classify this new window:
{instruction}

Answer with exactly one word: ANOMALY or NOMINAL."""
    return prompt


def load_sample(sample_path: Path) -> list[dict]:
    """Load the frozen frontier sample."""
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {sample_path}. Run select_frontier_sample.py --select first."
        )
    return [json.loads(line) for line in sample_path.open()]


def generate_prompts(
    sample: list[dict],
    retriever: RAGRetriever,
    k: int = 5,
    include_labels: bool = True,
) -> list[dict]:
    """Generate RAG-augmented prompts for each sample window."""
    prompts = []
    for row in sample:
        prompt = build_rag_prompt(
            instruction=row["instruction"],
            mission=row["mission"],
            channel=row["channel"],
            values=row.get("values", []),
            retriever=retriever,
            k=k,
            include_labels=include_labels,
        )
        prompts.append(
            {
                "index": row["index"],
                "mission": row["mission"],
                "channel": row["channel"],
                "is_anomaly": row["is_anomaly"],
                "prompt": prompt,
            }
        )
    return prompts


def do_prompts_only(
    sample_path: Path,
    output_path: Path,
    k: int = 5,
    include_labels: bool = True,
) -> None:
    """Generate prompts for manual classification."""
    retriever = RAGRetriever(RAG_DIR)
    sample = load_sample(sample_path)
    prompts = generate_prompts(sample, retriever, k=k, include_labels=include_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(prompts)} RAG-augmented prompts to {output_path}")
    print(f"  k={k}, include_labels={include_labels}")
    print("\nTo classify manually, have the model classify each prompt and record the")
    print("predictions to a JSON file with format: [{index, predicted, response}, ...]")


def assemble_results(
    sample_path: Path,
    classifications_path: Path,
    output_path: Path,
    k: int,
) -> None:
    """Assemble results from manual classifications."""
    sample = {json.loads(line)["index"]: json.loads(line) for line in sample_path.open()}
    preds = {c["index"]: c for c in json.loads(Path(classifications_path).read_text())}

    missing = sorted(set(sample) - set(preds))
    if missing:
        raise ValueError(
            f"{len(missing)} sampled windows have no classification. "
            f"Missing indices: {missing[:10]}..."
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
                "elapsed_s": 0.0,
            }
        )

    n = len(results)
    tp = sum(1 for r in results if r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fp = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fn = sum(1 for r in results if r["is_anomaly"] and r["predicted"] != "ANOMALY")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "NOMINAL")
    unknown = sum(1 for r in results if r["predicted"] == "UNKNOWN")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    summary = {
        "approach": APPROACH % k,
        "model": f"Claude (session model) + RAG (k={k})",
        "n_samples": n,
        "accuracy": round((tp + tn) / n, 4) if n else 0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown_responses": unknown,
        "seed": SEED,
        "k": k,
        "partial": False,
        "note": (
            f"RAG-augmented frontier: same {n}-window sample as Phase-6 zero-shot, "
            f"but with k={k} retrieved training neighbors per window as context. "
            "Tests whether channel-specific history closes the gap vs the fine-tune."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"Wrote {output_path}")
    print(f"  P={precision:.3f} R={recall:.3f} F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=SAMPLE_FILE,
        help="Path to the frozen frontier sample",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors to retrieve (default 5)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Exclude labels from retrieved context (harder test)",
    )
    parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Generate prompts for manual classification",
    )
    parser.add_argument(
        "--assemble",
        type=Path,
        default=None,
        help="Path to classifications JSON; assemble final results",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: results/inference_frontier_rag.json)",
    )
    args = parser.parse_args()

    include_labels = not args.no_labels

    if args.prompts_only:
        out = args.out or PROMPTS_FILE
        do_prompts_only(args.sample, out, k=args.k, include_labels=include_labels)
    elif args.assemble:
        out = args.out or RESULTS_FILE
        assemble_results(args.sample, args.assemble, out, k=args.k)
    else:
        parser.error(
            "Use --prompts-only to generate prompts, or --assemble PATH to assemble results"
        )


if __name__ == "__main__":
    main()
