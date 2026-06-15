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
    python src/inference/eval_vision.py                 # all test windows (fine-tuned adapter)
    python src/inference/eval_vision.py --limit 50      # smoke test
    python src/inference/eval_vision.py --resume        # resume after interruption
    python src/inference/eval_vision.py --base          # Phase 12: un-fine-tuned base zero-shot
                                                        #   -> results/inference_vision_base.json
    python src/inference/eval_vision.py --score         # Phase 14: continuous verdict score
                                                        #   -> results/inference_vision_scored.json

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
# Phase 14: continuous ANOMALY-vs-NOMINAL verdict score per window (mirror of
# test_local_gguf.py --score for the text LLM). Feeds src/inference/pr_curve.py (a vision
# PR curve) and src/inference/ensemble.py (the fusion input).
SCORED_RESULTS_FILE = Path("results/inference_vision_scored.json")
# Phase 12: un-fine-tuned base control (no adapter) — mirrors the Phase-6 text base/frontier
# controls for the vision modality. Same base weights the Phase-8 fine-tune was trained on
# (train_detection.py::DEFAULT_MODEL); the fine-tune merely added the LoRA adapter on top.
BASE_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
BASE_RESULTS_FILE = Path("results/inference_vision_base.json")
MAX_SEQ_LENGTH = 2048

# Same prompt the model was fine-tuned with (train_detection.py::USER_PROMPT).
USER_PROMPT = (
    "This is a plot of a spacecraft telemetry sequence (normalized value vs. timestep). "
    "Does it show anomalous behaviour? Answer with ANOMALY DETECTED or NOMINAL."
)
APPROACH_LABEL = "LLM Detection (vision, Qwen3-VL)"
BASE_APPROACH_LABEL = "LLM detection (vision, base zero-shot)"


def classify_response(response: str) -> str:
    """Extract ANOMALY/NOMINAL from model output (matches test_local_gguf.py)."""
    upper = response.upper()
    if "ANOMALY DETECTED" in upper or upper.startswith("ANOMALY"):
        return "ANOMALY"
    if "NOMINAL" in upper:
        return "NOMINAL"
    return "UNKNOWN"


def compute_summary(
    results: list[dict], adapter: str, partial: bool = False, approach: str = APPROACH_LABEL
) -> dict:
    """Compute the metrics summary from a (possibly partial) results list.

    Mirrors test_local_gguf.compute_summary so the JSON schema is identical and
    evaluate.py's shared loader works unchanged. `approach` distinguishes the
    fine-tuned detector (default) from the Phase-12 base zero-shot control.
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
        "approach": approach,
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


def write_results(
    results: list[dict],
    adapter: str,
    partial: bool,
    results_file: Path,
    approach: str = APPROACH_LABEL,
) -> None:
    """Atomically write {summary, results} (temp + rename)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(results, adapter, partial=partial, approach=approach)
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


def load_model(model_ref: str):
    """Load a Qwen3-VL model for inference via Unsloth.

    `model_ref` is either the fine-tuned LoRA adapter dir (loads base+adapter, the
    Phase-8 detector) or a bare base-model repo id (loads base weights only, the
    Phase-12 zero-shot control). FastVisionModel.from_pretrained resolves both.
    """
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_ref,
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


# --------------------------------------------------------------------------- #
# Phase 14: continuous verdict score (vision PR calibration + fusion input)
# --------------------------------------------------------------------------- #
# Exact mirror of test_local_gguf.py::score_prompt for the vision modality. The model
# emits its verdict as the FIRST generated token ("ANOMALY DETECTED..." / "NOMINAL...").
# Instead of decoding the full continuation and parsing a hard verdict, we read the
# first-step logits (generate(max_new_tokens=1, output_scores=True)) and take the relative
# softmax of the ANOMALY-vs-NOMINAL verdict tokens. That ratio is a deterministic, continuous
# per-window anomaly score in (0, 1) we can sweep a threshold over (pr_curve.py / ensemble.py).
# eval_vision already decodes greedily (do_sample=False), so this adds NO sampling bias — it
# only exposes the underlying confidence the hard verdict was already taking the argmax of.


def verdict_token_ids(processor) -> tuple[int, int]:
    """First-token ids for the two verdict words, as emitted at an assistant turn.

    Mirror of test_local_gguf.verdict_token_ids: the assistant content begins right after
    the generation prompt, so the words segment standalone. We take the FIRST sub-token of
    each verdict word from the processor's tokenizer (no special tokens, no leading space).
    """
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    anom = tok.encode("ANOMALY", add_special_tokens=False)
    nom = tok.encode("NOMINAL", add_special_tokens=False)
    return anom[0], nom[0]


def score_image(model, processor, image, anom_id: int, nom_id: int) -> dict:
    """Prefill one image+prompt and return the ANOMALY-vs-NOMINAL verdict score.

    score = softmax over the two verdict-token logits at the first generated position =
    exp(l_anom) / (exp(l_anom) + exp(l_nom)) = P(ANOMALY | {A, N}). Returns the score plus
    the raw logits for transparency, matching the text scored-row schema.
    """
    import torch

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
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # out.scores[0]: (batch=1, vocab) logits for the first generated token.
    logits = out.scores[0][0].to(torch.float32)
    l_anom = float(logits[anom_id].item())
    l_nom = float(logits[nom_id].item())
    m = max(l_anom, l_nom)
    import math

    ea, en = math.exp(l_anom - m), math.exp(l_nom - m)
    score = ea / (ea + en)
    return {
        "score": score,
        "logit_anomaly": round(l_anom, 4),
        "logit_nominal": round(l_nom, 4),
        "argmax": "ANOMALY" if l_anom >= l_nom else "NOMINAL",
    }


def compute_score_summary(
    results: list[dict],
    adapter: str,
    partial: bool = False,
    approach: str = "LLM Detection (vision, Qwen3-VL) — verdict score",
) -> dict:
    """Summary for a scored vision run: argmax-based confusion matrix + score stats."""
    n = len(results)
    tp = sum(1 for r in results if r["is_anomaly"] and r["argmax"] == "ANOMALY")
    fp = sum(1 for r in results if not r["is_anomaly"] and r["argmax"] == "ANOMALY")
    fn = sum(1 for r in results if r["is_anomaly"] and r["argmax"] != "ANOMALY")
    tn = sum(1 for r in results if not r["is_anomaly"] and r["argmax"] == "NOMINAL")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    avg_time = sum(r["elapsed_s"] for r in results) / n if n else 0
    return {
        "approach": approach,
        "scoring": "verdict-token logprob (1-step generate, argmax @ score=0.5)",
        "n_samples": n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "avg_time_s": round(avg_time, 3),
        "adapter": adapter,
        "partial": partial,
    }


def write_scored(results: list[dict], adapter: str, partial: bool, results_file: Path) -> None:
    """Atomically write {summary, results} for the scored vision run (temp + rename)."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_score_summary(results, adapter, partial=partial)
    tmp = results_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    tmp.replace(results_file)


def run_scoring(args, model_ref: str, samples: list[dict], results_file: Path) -> None:
    """Phase 14: capture a continuous verdict score per PNG (1-step prefill)."""
    from PIL import Image

    results: list[dict] = []
    start = 0
    if args.resume and results_file.exists():
        prior = json.loads(results_file.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} scored in {results_file}.", flush=True)
            write_scored(results, model_ref, partial=False, results_file=results_file)
            return
        print(f"Resuming scoring from sample {start}/{len(samples)}.", flush=True)

    model, processor = load_model(model_ref)
    anom_id, nom_id = verdict_token_ids(processor)
    print(f"Model loaded from {model_ref}. Verdict tokens: ANOMALY={anom_id} NOMINAL={nom_id}")
    print(f"Scoring samples {start}..{len(samples)} (1-step prefill)...", flush=True)

    recent: list[float] = []
    for i in range(start, len(samples)):
        rec = samples[i]
        img_path = Path(rec["image_path"])
        t0 = time.perf_counter()
        try:
            image = Image.open(img_path).convert("RGB")
            sc = score_image(model, processor, image, anom_id, nom_id)
        except Exception as exc:  # noqa: BLE001 — record failures, don't abort the run
            sc = {
                "score": 0.0,
                "logit_anomaly": 0.0,
                "logit_nominal": 0.0,
                "argmax": f"ERROR: {exc}",
            }
        elapsed = time.perf_counter() - t0
        recent.append(elapsed)

        results.append(
            {
                "index": rec.get("index", i),
                "mission": rec["mission"],
                "channel": rec["channel"],
                "is_anomaly": bool(rec["is_anomaly"]),
                "score": round(sc["score"], 6),
                "logit_anomaly": sc["logit_anomaly"],
                "logit_nominal": sc["logit_nominal"],
                "argmax": sc["argmax"],
                "image_path": str(img_path),
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 50 == 0:
            avg = sum(recent[-50:]) / len(recent[-50:])
            print(f"  [{i + 1}/{len(samples)}] avg {avg:.3f}s/sample", flush=True)
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            write_scored(results, model_ref, partial=True, results_file=results_file)
            print(f"  checkpoint: {len(results)} scored -> {results_file}", flush=True)

    write_scored(results, model_ref, partial=False, results_file=results_file)
    s = compute_score_summary(results, model_ref, partial=False)
    print(f"\n=== Vision verdict-score run ({s['n_samples']} samples) ===")
    print(f"  argmax P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")
    print(f"  TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}")
    print(f"  Avg time: {s['avg_time_s']:.3f}s/sample")
    print(f"\nScored results saved to {results_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Trained VL LoRA dir.")
    parser.add_argument(
        "--base",
        action="store_true",
        help="Phase 12: load the un-fine-tuned base VL model (no adapter) for a zero-shot "
        "control. Defaults the output to results/inference_vision_base.json and the approach "
        "label to the base row; identical prompt/decoding/parser otherwise.",
    )
    parser.add_argument("--split", default="test", help="Metadata split to evaluate.")
    parser.add_argument("--limit", type=int, default=0, help="Cap samples (0 = all).")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Phase 14: capture a continuous ANOMALY-vs-NOMINAL verdict score per PNG "
        "(1-step prefill logits) instead of decoding the verdict text. Writes to "
        "--results-file (default results/inference_vision_scored.json in score mode); "
        "feeds pr_curve.py + ensemble.py. Decoding is already greedy, so no sampling bias.",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Output JSON (default: inference_vision.json; --base: inference_vision_base.json; "
        "--score: inference_vision_scored.json).",
    )
    args = parser.parse_args()

    # --base loads the base weights only and writes the base-control file/label by default;
    # everything downstream (prompt, decoding, parser, schema) is identical to the fine-tune.
    model_ref = BASE_MODEL if args.base else args.adapter
    approach = BASE_APPROACH_LABEL if args.base else APPROACH_LABEL
    # In score mode the default output is a separate file so it never clobbers the hard-verdict
    # run (results/inference_vision.json).
    if args.score:
        default_file = SCORED_RESULTS_FILE
    else:
        default_file = BASE_RESULTS_FILE if args.base else RESULTS_FILE
    results_file = Path(args.results_file) if args.results_file else default_file

    samples = load_test_samples(args.split)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if args.score:
        run_scoring(args, model_ref, samples, results_file)
        return

    results: list[dict] = []
    start = 0
    if args.resume and results_file.exists():
        prior = json.loads(results_file.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} samples in {results_file}.", flush=True)
            write_results(
                results, model_ref, partial=False, results_file=results_file, approach=approach
            )
            return
        print(f"Resuming from sample {start}/{len(samples)}.", flush=True)

    from PIL import Image

    model, processor = load_model(model_ref)
    print(f"Model loaded from {model_ref}. Scoring {start}..{len(samples)}.", flush=True)

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
            write_results(
                results, model_ref, partial=True, results_file=results_file, approach=approach
            )
            print(f"  checkpoint: {len(results)} saved to {results_file}", flush=True)

    write_results(results, model_ref, partial=False, results_file=results_file, approach=approach)
    s = compute_summary(results, model_ref, partial=False, approach=approach)
    print(f"\n=== Vision Eval ({s['n_samples']} samples) ===")
    print(f"  Accuracy:  {s['accuracy']:.3f}")
    print(f"  Precision: {s['precision']:.3f}  Recall: {s['recall']:.3f}  F1: {s['f1']:.3f}")
    print(f"  TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']} Unknown={s['unknown_responses']}")
    print(f"  Avg time:  {s['avg_time_s']:.2f}s/sample")
    print(f"\nResults saved to {results_file}", flush=True)


if __name__ == "__main__":
    main()
