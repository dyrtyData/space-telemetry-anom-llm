"""Evaluate the fine-tuned Qwen3-VL vision detector on PNG telemetry plots (Phase 8).

Runs ON the Vast.ai GPU instance (Unsloth + transformers + the trained adapter), NOT
locally — multimodal GGUF on Metal is patchy, so the eval is done where the model trained
(plan §8.5 fallback). It writes results/inference_vision.json in the SAME per-record schema
as test_local_gguf.py so evaluate.py can load it with the shared detection loader.

Inputs : data/processed/plots/test_metadata.jsonl
         (each line: {"index", "image_path", "is_anomaly", "mission", "channel"})
         + the referenced PNGs under data/processed/plots/test/.
         + the trained LoRA dir (default models/lora/qwen3-vl-detection).
Outputs: results/inference_vision.json  ({summary, results}, schema below).

Usage on the instance (from the repo root so image_path resolves):
    python src/inference/eval_vision.py                 # all test windows
    python src/inference/eval_vision.py --limit 50      # smoke test
    python src/inference/eval_vision.py --resume        # resume after interruption

Durability: --checkpoint-every N (atomic temp+rename) + --resume, mirroring
test_local_gguf.py. Samples are processed in deterministic file order.
"""

import argparse
import json
import time
from pathlib import Path

PLOTS_DIR = Path("data/processed/plots")
DEFAULT_ADAPTER = "models/lora/qwen3-vl-detection"
RESULTS_FILE = Path("results/inference_vision.json")
MAX_SEQ_LENGTH = 2048

# Same prompt the model was fine-tuned with (train_detection.py::USER_PROMPT).
USER_PROMPT = (
    "This is a plot of a spacecraft telemetry sequence (normalized value vs. timestep). "
    "Does it show anomalous behaviour? Answer with ANOMALY DETECTED or NOMINAL."
)
APPROACH_LABEL = "LLM Detection (vision, Qwen3-VL)"


def classify_response(response: str) -> str:
    """Extract ANOMALY/NOMINAL from model output (matches test_local_gguf.py)."""
    upper = response.upper()
    if "ANOMALY DETECTED" in upper or upper.startswith("ANOMALY"):
        return "ANOMALY"
    if "NOMINAL" in upper:
        return "NOMINAL"
    return "UNKNOWN"


def compute_summary(results: list[dict], adapter: str, partial: bool = False) -> dict:
    """Compute the metrics summary from a (possibly partial) results list.

    Mirrors test_local_gguf.compute_summary so the JSON schema is identical and
    evaluate.py's shared loader works unchanged.
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
        "approach": APPROACH_LABEL,
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
        "adapter": adapter,
        "partial": partial,
    }


def write_results(results: list[dict], adapter: str, partial: bool, results_file: Path) -> None:
    """Atomically write {summary, results} (temp + rename)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(results, adapter, partial=partial)
    tmp = results_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    tmp.replace(results_file)


def load_test_samples(split: str) -> list[dict]:
    meta_file = PLOTS_DIR / f"{split}_metadata.jsonl"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing {meta_file} — upload data/processed/plots/ first.")
    with open(meta_file) as f:
        return [json.loads(line) for line in f]


def load_model(adapter: str):
    """Load the fine-tuned Qwen3-VL adapter (+ base) for inference via Unsloth."""
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=adapter,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, processor


def classify_image(model, processor, image) -> str:
    """Run one image through the VL model and return the raw generated text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    # Strip the prompt tokens; decode only the newly generated continuation.
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Trained VL LoRA dir.")
    parser.add_argument("--split", default="test", help="Metadata split to evaluate.")
    parser.add_argument("--limit", type=int, default=0, help="Cap samples (0 = all).")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results-file", default=str(RESULTS_FILE))
    args = parser.parse_args()

    results_file = Path(args.results_file)
    samples = load_test_samples(args.split)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    results: list[dict] = []
    start = 0
    if args.resume and results_file.exists():
        prior = json.loads(results_file.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} samples in {results_file}.", flush=True)
            write_results(results, args.adapter, partial=False, results_file=results_file)
            return
        print(f"Resuming from sample {start}/{len(samples)}.", flush=True)

    from PIL import Image

    model, processor = load_model(args.adapter)
    print(f"Model loaded from {args.adapter}. Scoring {start}..{len(samples)}.", flush=True)

    recent: list[float] = []
    for i in range(start, len(samples)):
        rec = samples[i]
        img_path = Path(rec["image_path"])
        t0 = time.perf_counter()
        try:
            image = Image.open(img_path).convert("RGB")
            response = classify_image(model, processor, image)
        except Exception as exc:  # noqa: BLE001 — record failures, don't abort the run
            response = f"ERROR: {exc}"
        elapsed = time.perf_counter() - t0
        recent.append(elapsed)

        predicted = classify_response(response)
        actual_anomaly = bool(rec["is_anomaly"])
        expected_class = "ANOMALY" if actual_anomaly else "NOMINAL"

        results.append(
            {
                "index": rec.get("index", i),
                "mission": rec["mission"],
                "channel": rec["channel"],
                "is_anomaly": actual_anomaly,
                "predicted": predicted,
                "correct": predicted == expected_class,
                "expected_response": expected_class,
                "actual_response": response[:300],
                "image_path": str(img_path),
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 10 == 0:
            avg10 = sum(recent[-10:]) / min(len(recent), 10)
            print(f"  [{i + 1}/{len(samples)}] avg {avg10:.2f}s/sample", flush=True)
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            write_results(results, args.adapter, partial=True, results_file=results_file)
            print(f"  checkpoint: {len(results)} saved to {results_file}", flush=True)

    write_results(results, args.adapter, partial=False, results_file=results_file)
    s = compute_summary(results, args.adapter, partial=False)
    print(f"\n=== Vision Eval ({s['n_samples']} samples) ===")
    print(f"  Accuracy:  {s['accuracy']:.3f}")
    print(f"  Precision: {s['precision']:.3f}  Recall: {s['recall']:.3f}  F1: {s['f1']:.3f}")
    print(f"  TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']} Unknown={s['unknown_responses']}")
    print(f"  Avg time:  {s['avg_time_s']:.2f}s/sample")
    print(f"\nResults saved to {results_file}", flush=True)


if __name__ == "__main__":
    main()
