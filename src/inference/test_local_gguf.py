"""Test GGUF model inference on M3 Max via Metal-accelerated llama-cpp-python.

Phase 4 (§4.3). Runs LOCALLY after the GGUF is downloaded from the cloud instance.

MUST-READ corrections vs. the original plan §4.3:
  - GGUF loaded from STAR_MODEL_DIR (default /Volumes/DUAL DRIVE/star-pipeline/models),
    never from local disk (which is nearly full).
  - Actual GGUF path: {STAR_MODEL_DIR}/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf
    (Unsloth names the file after the base model, not the project).
  - Test data: test_with_advice.jsonl (response includes DIAGNOSIS/ADVICE/ACTION so we can
    verify the model actually generates structured advice, not just ANOMALY/NOMINAL).
  - llama-cpp-python installed with Metal:
      CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
  - Default --limit 100 for Phase 4 smoke test; Phase 5 will run the full 4,500-sample split.

Usage:
    python src/inference/test_local_gguf.py
    python src/inference/test_local_gguf.py --limit 20
    python src/inference/test_local_gguf.py --limit 0   # run all 4,500 (Phase 5)
    STAR_MODEL_DIR=/custom/path python src/inference/test_local_gguf.py
"""

import argparse
import json
import os
import time
from pathlib import Path

STAR_MODEL_DIR = Path(os.environ.get("STAR_MODEL_DIR", "/Volumes/DUAL DRIVE/star-pipeline/models"))
GGUF_PATH = STAR_MODEL_DIR / "gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf"

TEST_FILE = Path("data/splits/test_with_advice.jsonl")
RESULTS_FILE = Path("results/inference_test.json")

SYSTEM_PROMPT = (
    "You are a spacecraft telemetry analyst. Analyze telemetry sequences and identify "
    "anomalies. When an anomaly is detected, provide a diagnosis, diagnostic advice, and "
    "a recommended action to help engineers resolve the issue."
)


def format_prompt(instruction: str) -> str:
    """Format as ChatML prompt (matches training format from format_for_unsloth.py)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_model(gguf_path: Path, verbose: bool = False):
    """Load GGUF model with Metal GPU acceleration."""
    from llama_cpp import Llama

    if not gguf_path.exists():
        raise FileNotFoundError(
            f"GGUF not found: {gguf_path}\n"
            f"Set STAR_MODEL_DIR env var or download the model from the cloud instance."
        )

    print(f"Loading GGUF: {gguf_path}")
    model = Llama(
        model_path=str(gguf_path),
        n_ctx=2048,
        n_gpu_layers=-1,  # offload all layers to Metal GPU
        verbose=verbose,
    )
    return model


def classify_response(response: str) -> str:
    """Extract ANOMALY/NOMINAL classification from model output."""
    upper = response.upper()
    if "ANOMALY DETECTED" in upper or upper.startswith("ANOMALY"):
        return "ANOMALY"
    if "NOMINAL" in upper:
        return "NOMINAL"
    return "UNKNOWN"


def compute_summary(results: list[dict], gguf_path: Path, partial: bool = False) -> dict:
    """Compute the metrics summary from a (possibly partial) results list.

    Uses each record's persisted ``elapsed_s`` so the summary is reconstructable on
    resume (no dependence on an in-memory timing list).
    """
    n = len(results)
    tp = sum(1 for r in results if r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fp = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "ANOMALY")
    fn = sum(1 for r in results if r["is_anomaly"] and r["predicted"] != "ANOMALY")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["predicted"] == "NOMINAL")
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_time = sum(r["elapsed_s"] for r in results) / n if n else 0
    unknown = sum(1 for r in results if r["predicted"] == "UNKNOWN")

    return {
        "approach": "LLM Detection (Qwen3-8B advice SFT Q4_K_M)",
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
        "gguf_path": str(gguf_path),
        "partial": partial,  # True while checkpointing mid-run; False on the final write
    }


def write_results(results: list[dict], gguf_path: Path, partial: bool) -> None:
    """Atomically write {summary, results} to RESULTS_FILE (temp + rename)."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    summary = compute_summary(results, gguf_path, partial=partial)
    tmp = RESULTS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    tmp.replace(RESULTS_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of test samples to run (0 = all 4,500 for Phase 5). Default: 100.",
    )
    parser.add_argument(
        "--gguf",
        default=None,
        help="Override GGUF path (default: derived from STAR_MODEL_DIR).",
    )
    parser.add_argument("--verbose", action="store_true", help="Show llama.cpp debug output.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Write a partial results checkpoint every N samples (durability). Default: 250.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial RESULTS_FILE (skip already-scored samples).",
    )
    args = parser.parse_args()

    gguf_path = Path(args.gguf) if args.gguf else GGUF_PATH

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    with open(TEST_FILE) as f:
        samples = [json.loads(line) for line in f]

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    # Resume: load prior results; samples are processed in deterministic order, so the
    # next index to run is len(prior results).
    results: list[dict] = []
    start = 0
    if args.resume and RESULTS_FILE.exists():
        prior = json.loads(RESULTS_FILE.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} samples in {RESULTS_FILE}. Nothing to do.")
            write_results(results, gguf_path, partial=False)
            return
        print(f"Resuming from sample {start}/{len(samples)} ({RESULTS_FILE}).", flush=True)

    model = load_model(gguf_path, verbose=args.verbose)
    print(f"Model loaded. n_gpu_layers={model.model_params.n_gpu_layers}", flush=True)
    print(f"Running inference on samples {start}..{len(samples)}...", flush=True)

    recent: list[float] = []
    for i in range(start, len(samples)):
        sample = samples[i]
        prompt = format_prompt(sample["instruction"])
        t0 = time.perf_counter()

        output = model(
            prompt,
            max_tokens=300,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
        )
        elapsed = time.perf_counter() - t0
        recent.append(elapsed)

        response = output["choices"][0]["text"].strip()
        predicted = classify_response(response)
        actual_anomaly = sample["metadata"]["is_anomaly"]
        expected_class = "ANOMALY" if actual_anomaly else "NOMINAL"

        results.append(
            {
                "index": i,
                "mission": sample["metadata"]["mission"],
                "channel": sample["metadata"]["channel"],
                "is_anomaly": actual_anomaly,
                "predicted": predicted,
                "correct": predicted == expected_class,
                "expected_response": sample["response"][:200],
                "actual_response": response[:300],
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 10 == 0:
            avg10 = sum(recent[-10:]) / 10
            print(f"  [{i + 1}/{len(samples)}] avg {avg10:.2f}s/sample", flush=True)
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            write_results(results, gguf_path, partial=True)
            print(f"  checkpoint: {len(results)} samples saved to {RESULTS_FILE}", flush=True)

    write_results(results, gguf_path, partial=False)
    s = compute_summary(results, gguf_path, partial=False)
    print(f"\n=== Inference Results ({s['n_samples']} samples) ===")
    print(f"  Accuracy:  {s['accuracy']:.3f}")
    print(f"  Precision: {s['precision']:.3f}  Recall: {s['recall']:.3f}  F1: {s['f1']:.3f}")
    print(
        f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}  "
        f"Unknown={s['unknown_responses']}"
    )
    print(f"  Avg time:  {s['avg_time_s']:.2f}s/sample")
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
