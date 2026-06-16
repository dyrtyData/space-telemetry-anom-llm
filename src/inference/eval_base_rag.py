"""Evaluate the un-fine-tuned base Qwen3-8B with RAG context on the full test split.

Apples-to-apples comparison with the fine-tuned text LLM:
- Fine-tune (F1 0.453) - 21k training windows burned into weights
- Base + RAG - retrieves k=5 examples per window from the same 21k training windows
- Base few-shot (F1 0.420) - only 2 examples, no retrieval
- Always-anomaly (F1 0.399) - trivial baseline

Usage:
    # Build RAG index first (one-time)
    python src/inference/build_rag_index.py --train data/splits/train.jsonl --out data/rag/

    # Download base GGUF (one-time)
    huggingface-cli download unsloth/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir models/base/

    # Run base + RAG eval (4,500 windows, ~4-6 hours)
    python src/inference/eval_base_rag.py --limit 0 --resume --checkpoint-every 250

Time estimate: ~4-6 hours for 4,500 windows at ~3-5 s/window. The context is longer
(k=5 retrieved examples) but output is short (just ANOMALY/NOMINAL verdict).
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from rag_retrieve import RAGRetriever, format_rag_context

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

TEST_FILE = Path("data/splits/test_with_advice.jsonl")
RAG_DIR = Path("data/rag/")
RESULTS_FILE = Path("results/inference_base_rag.json")

STAR_MODEL_DIR = Path(os.environ.get("STAR_MODEL_DIR", "models"))
BASE_GGUF = STAR_MODEL_DIR / "base" / "Qwen3-8B-Q4_K_M.gguf"

APPROACH = "Base Qwen3-8B + RAG (k=%d)"


def build_rag_prompt(
    instruction: str,
    mission: str,
    channel: str,
    values: list[float],
    retriever: RAGRetriever,
    k: int = 5,
) -> str:
    """Build a RAG-augmented prompt for the base model."""
    neighbors = retriever.retrieve(mission, channel, values, k=k, include_labels=True)
    context = format_rag_context(neighbors, mission, channel, include_labels=True)

    n_neighbors = len(neighbors)
    prompt = f"""Below are {n_neighbors} historical windows from channel {channel} that are \
similar to the current window, along with their ground-truth labels:

{context}

Now classify this new window:
{instruction}

Answer with exactly one word: ANOMALY or NOMINAL"""
    return prompt


def format_chatml_prompt(user_message: str, no_think: bool = True) -> str:
    """Format as ChatML for the model."""
    system_prompt = (
        "You are a spacecraft telemetry analyst. Analyze telemetry sequences "
        "and identify anomalies. Answer concisely."
    )
    if no_think:
        system_prompt += " /no_think"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_test_data(limit: int = 0) -> list[dict]:
    """Load test data with metadata."""
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    records = []
    with open(TEST_FILE) as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            record = json.loads(line)
            meta = record["metadata"]
            records.append(
                {
                    "index": i,
                    "instruction": record["instruction"],
                    "mission": meta["mission"],
                    "channel": meta["channel"],
                    "is_anomaly": meta["is_anomaly"],
                    "values": meta.get("values", []),
                    "expected_response": record.get("response", ""),
                }
            )
    return records


def parse_verdict(response: str) -> str:
    """Parse model response for ANOMALY/NOMINAL verdict."""
    text = response.strip().upper()

    if "<THINK>" in text:
        end = text.rfind("</THINK>")
        if end != -1:
            text = text[end + 8 :].strip()
        else:
            text = text[text.rfind("<THINK>") + 7 :].strip()

    if "ANOMALY" in text:
        return "ANOMALY"
    elif "NOMINAL" in text:
        return "NOMINAL"
    else:
        return "UNKNOWN"


def compute_summary(results: list[dict], k: int, partial: bool = True) -> dict:
    """Compute summary metrics from results."""
    n = len(results)
    if n == 0:
        return {"n_samples": 0, "partial": partial}

    tp = sum(1 for r in results if r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fp = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fn = sum(1 for r in results if r["is_anomaly"] and r["predicted"] != "ANOMALY")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "NOMINAL")
    unknown = sum(1 for r in results if r["predicted"] == "UNKNOWN")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / n if n else 0

    avg_time = sum(r.get("elapsed_s", 0) for r in results) / n if n else 0

    return {
        "approach": APPROACH % k,
        "model": f"Qwen3-8B base (Q4_K_M) + RAG (k={k})",
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "unknown_responses": unknown,
        "avg_time_s": round(avg_time, 3),
        "k": k,
        "partial": partial,
        "note": (
            f"Base Qwen3-8B (un-fine-tuned) with RAG: k={k} retrieved training "
            "neighbors per window. Tests whether RAG substitutes for fine-tuning."
        ),
    }


def write_results(results: list[dict], k: int, partial: bool, output_path: Path):
    """Write results atomically with a temp file."""
    summary = compute_summary(results, k, partial)
    data = {"summary": summary, "results": results}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    os.replace(temp_path, output_path)


def run_eval(
    model_path: Path,
    k: int = 5,
    limit: int = 0,
    resume: bool = False,
    checkpoint_every: int = 250,
    output_path: Path = RESULTS_FILE,
) -> None:
    """Run the base + RAG evaluation."""
    if Llama is None:
        raise ImportError("llama-cpp-python required. pip install llama-cpp-python")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Base GGUF not found: {model_path}\n"
            "Download with: huggingface-cli download unsloth/Qwen3-8B-GGUF "
            "Qwen3-8B-Q4_K_M.gguf --local-dir models/base/"
        )

    print(f"Loading base model: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False,
    )

    print(f"Loading RAG index: {RAG_DIR}")
    retriever = RAGRetriever(RAG_DIR)

    test_data = load_test_data(limit)
    print(f"Loaded {len(test_data)} test windows")

    results = []
    start_idx = 0

    if resume and output_path.exists():
        existing = json.loads(output_path.read_text())
        results = existing.get("results", [])
        start_idx = len(results)
        print(f"Resuming from window {start_idx}")

    for i, record in enumerate(test_data):
        if i < start_idx:
            continue

        rag_prompt = build_rag_prompt(
            instruction=record["instruction"],
            mission=record["mission"],
            channel=record["channel"],
            values=record["values"],
            retriever=retriever,
            k=k,
        )
        chatml = format_chatml_prompt(rag_prompt)

        start_time = time.time()
        output = llm(
            chatml,
            max_tokens=200,
            temperature=0.0,
            stop=["<|im_end|>"],
        )
        elapsed = time.time() - start_time

        response = output["choices"][0]["text"]
        predicted = parse_verdict(response)
        expected = "ANOMALY" if record["is_anomaly"] else "NOMINAL"

        results.append(
            {
                "index": record["index"],
                "mission": record["mission"],
                "channel": record["channel"],
                "is_anomaly": record["is_anomaly"],
                "predicted": predicted,
                "correct": predicted == expected,
                "actual_response": response[:300],
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            summary = compute_summary(results, k, partial=True)
            print(
                f"[{i + 1}/{len(test_data)}] "
                f"P={summary['precision']:.3f} R={summary['recall']:.3f} "
                f"F1={summary['f1']:.3f} ({elapsed:.2f}s)",
                flush=True,
            )

        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            write_results(results, k, partial=True, output_path=output_path)
            print(f"  Checkpointed at {i + 1} windows", flush=True)

    write_results(results, k, partial=False, output_path=output_path)
    summary = compute_summary(results, k, partial=False)
    print(f"\nFinal results written to {output_path}")
    print(f"  P={summary['precision']:.3f} R={summary['recall']:.3f} F1={summary['f1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=BASE_GGUF,
        help="Path to base GGUF model",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of neighbors to retrieve (default 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of windows to evaluate (0=all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Checkpoint interval (0 to disable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_FILE,
        help="Output file path",
    )
    args = parser.parse_args()

    run_eval(
        model_path=args.model,
        k=args.k,
        limit=args.limit,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
