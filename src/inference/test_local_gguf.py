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
TRAIN_FILE = Path("data/splits/train_with_advice.jsonl")  # few-shot examples (no test leakage)
RESULTS_FILE = Path("results/inference_test.json")
SCORED_FILE = Path("results/inference_test_scored.json")  # Phase 13: --score output

SYSTEM_PROMPT = (
    "You are a spacecraft telemetry analyst. Analyze telemetry sequences and identify "
    "anomalies. When an anomaly is detected, provide a diagnosis, diagnostic advice, and "
    "a recommended action to help engineers resolve the issue."
)


def build_fewshot_prefix(n_each: int = 1) -> str:
    """Build ChatML few-shot example turns from the TRAIN split (never the test set).

    Used by the Phase-6 'prompting instead of fine-tuning' base baseline: showing the base
    model a worked NOMINAL and a worked ANOMALY example teaches it the exact terse output
    contract (verdict + DIAGNOSIS/ADVICE/ACTION) so it emits parseable verdicts instead of
    rambling. Examples are drawn from TRAIN to avoid any test-set leakage.
    """
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Few-shot needs the train split: {TRAIN_FILE}")
    train = [json.loads(line) for line in TRAIN_FILE.open()]
    nominal = [r for r in train if not r["metadata"]["is_anomaly"]][:n_each]
    anomaly = [r for r in train if r["metadata"]["is_anomaly"]][:n_each]
    turns = []
    for r in nominal + anomaly:  # show a nominal then an anomaly
        turns.append(
            f"<|im_start|>user\n{r['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{r['response']}<|im_end|>\n"
        )
    return "".join(turns)


def format_prompt(instruction: str, fewshot_prefix: str = "", no_think: bool = False) -> str:
    """Format as ChatML prompt (matches training format from format_for_unsloth.py).

    `fewshot_prefix` (optional) injects worked example turns after the system message;
    `no_think` appends Qwen3's ``/no_think`` switch so the base model answers directly
    instead of spending its token budget on a <think> block. Both default off, so the
    fine-tuned eval is byte-identical to Phase 4/5.
    """
    system = SYSTEM_PROMPT + (" /no_think" if no_think else "")
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"{fewshot_prefix}"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_model(gguf_path: Path, verbose: bool = False, logits_all: bool = False):
    """Load GGUF model with Metal GPU acceleration.

    ``logits_all=True`` is required for the Phase-13 verdict-score path: llama-cpp
    only populates its ``scores`` buffer for every position when this is set (with the
    default False, ``scores[n_tokens-1]`` reads back as zeros).
    """
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
        logits_all=logits_all,
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


# --------------------------------------------------------------------------- #
# Phase 13: continuous verdict score (precision-recall calibration)
# --------------------------------------------------------------------------- #
# The fine-tuned model emits the verdict as the FIRST assistant token
# ("ANOMALY DETECTED..." or "NOMINAL...", confirmed on 98.8% of the test split;
# the remaining ~1.2% open a <think> block first). Instead of sampling 300 tokens
# of advice and parsing a hard verdict, we read the model's logits at the first
# assistant position and take the relative log-probability of the ANOMALY vs the
# NOMINAL verdict token. That ratio is a deterministic, continuous per-window
# anomaly score in (0, 1) that we can sweep a threshold over (see pr_curve.py).
# Prefill-only (no generation) also makes the full 4,500-window run ~10x faster.


def verdict_token_ids(model) -> tuple[int, int]:
    """First-token ids for the two verdict words, as emitted at an assistant turn.

    The assistant content starts right after ``...assistant\\n`` (a newline boundary,
    so no leading space), which is exactly how ``tokenize(b"ANOMALY")`` /
    ``tokenize(b"NOMINAL")`` segment the words standalone.
    """
    anom = model.tokenize(b"ANOMALY", add_bos=False, special=False)
    nom = model.tokenize(b"NOMINAL", add_bos=False, special=False)
    return anom[0], nom[0]


def score_prompt(model, prompt: str, anom_id: int, nom_id: int) -> dict:
    """Prefill the prompt and return the ANOMALY-vs-NOMINAL verdict score.

    score = softmax over the two verdict-token logits at the first assistant
    position = exp(l_anom) / (exp(l_anom) + exp(l_nom)) = P(ANOMALY | {A, N}).
    Returns the score plus the raw logits for transparency/debugging.
    """
    import numpy as np

    tokens = model.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
    model.reset()
    model.eval(tokens)
    # llama-cpp-python writes the last evaluated token's logits to scores[n_tokens-1].
    logits = np.asarray(model.scores[model.n_tokens - 1], dtype=np.float64)
    l_anom = float(logits[anom_id])
    l_nom = float(logits[nom_id])
    # Numerically stable two-way softmax.
    m = max(l_anom, l_nom)
    ea, en = np.exp(l_anom - m), np.exp(l_nom - m)
    score = float(ea / (ea + en))
    return {
        "score": score,
        "logit_anomaly": round(l_anom, 4),
        "logit_nominal": round(l_nom, 4),
        "argmax": "ANOMALY" if l_anom >= l_nom else "NOMINAL",
    }


def compute_score_summary(
    results: list[dict],
    gguf_path: Path,
    partial: bool = False,
    approach: str = "LLM Detection (Qwen3-8B advice SFT) — verdict score",
) -> dict:
    """Summary for a scored run: argmax-based confusion matrix + score stats."""
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
        "scoring": "verdict-token logprob (prefill, argmax @ score=0.5)",
        "n_samples": n,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "avg_time_s": round(avg_time, 3),
        "gguf_path": str(gguf_path),
        "partial": partial,
    }


def write_scored(results: list[dict], gguf_path: Path, partial: bool, results_file: Path) -> None:
    """Atomically write {summary, results} for the scored run."""
    results_file.parent.mkdir(exist_ok=True)
    summary = compute_score_summary(results, gguf_path, partial=partial)
    tmp = results_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    tmp.replace(results_file)


def run_scoring(args, gguf_path: Path, samples: list[dict], results_file: Path) -> None:
    """Phase 13: capture a continuous verdict score per window (prefill-only)."""
    results: list[dict] = []
    start = 0
    if args.resume and results_file.exists():
        prior = json.loads(results_file.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} scored in {results_file}.")
            write_scored(results, gguf_path, partial=False, results_file=results_file)
            return
        print(f"Resuming scoring from sample {start}/{len(samples)}.", flush=True)

    model = load_model(gguf_path, verbose=args.verbose, logits_all=True)
    anom_id, nom_id = verdict_token_ids(model)
    print(
        f"Verdict tokens: ANOMALY={anom_id} ({model.detokenize([anom_id])!r}), "
        f"NOMINAL={nom_id} ({model.detokenize([nom_id])!r})",
        flush=True,
    )
    print(f"Scoring samples {start}..{len(samples)} (prefill-only)...", flush=True)

    recent: list[float] = []
    for i in range(start, len(samples)):
        sample = samples[i]
        prompt = format_prompt(sample["instruction"])
        t0 = time.perf_counter()
        sc = score_prompt(model, prompt, anom_id, nom_id)
        elapsed = time.perf_counter() - t0
        recent.append(elapsed)

        results.append(
            {
                "index": i,
                "mission": sample["metadata"]["mission"],
                "channel": sample["metadata"]["channel"],
                "is_anomaly": sample["metadata"]["is_anomaly"],
                "score": round(sc["score"], 6),
                "logit_anomaly": sc["logit_anomaly"],
                "logit_nominal": sc["logit_nominal"],
                "argmax": sc["argmax"],
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 50 == 0:
            avg = sum(recent[-50:]) / len(recent[-50:])
            print(f"  [{i + 1}/{len(samples)}] avg {avg:.3f}s/sample", flush=True)
        if args.checkpoint_every and (i + 1) % args.checkpoint_every == 0:
            write_scored(results, gguf_path, partial=True, results_file=results_file)
            print(f"  checkpoint: {len(results)} scored -> {results_file}", flush=True)

    write_scored(results, gguf_path, partial=False, results_file=results_file)
    s = compute_score_summary(results, gguf_path, partial=False)
    print(f"\n=== Verdict-score run ({s['n_samples']} samples) ===")
    print(f"  argmax P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")
    print(f"  TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}")
    print(f"  Avg time: {s['avg_time_s']:.3f}s/sample")
    print(f"\nScored results saved to {results_file}", flush=True)


def compute_summary(
    results: list[dict],
    gguf_path: Path,
    partial: bool = False,
    approach: str = "LLM Detection (Qwen3-8B advice SFT Q4_K_M)",
) -> dict:
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
        "gguf_path": str(gguf_path),
        "partial": partial,  # True while checkpointing mid-run; False on the final write
    }


def write_results(
    results: list[dict],
    gguf_path: Path,
    partial: bool,
    results_file: Path = RESULTS_FILE,
    approach: str = "LLM Detection (Qwen3-8B advice SFT Q4_K_M)",
) -> None:
    """Atomically write {summary, results} to ``results_file`` (temp + rename)."""
    results_file.parent.mkdir(exist_ok=True)
    summary = compute_summary(results, gguf_path, partial=partial, approach=approach)
    tmp = results_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    tmp.replace(results_file)


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
        help="Resume from an existing partial results file (skip already-scored samples).",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help=(
            "Override output path (default: results/inference_test.json). Use "
            "results/inference_base.json for the un-fine-tuned base-model control (Phase 6)."
        ),
    )
    parser.add_argument(
        "--approach-label",
        default="LLM Detection (Qwen3-8B advice SFT Q4_K_M)",
        help="Label stored in the summary's 'approach' field (cosmetic; evaluate.py relabels).",
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help=(
            "N few-shot examples PER CLASS from the TRAIN split (e.g. 1 -> 1 nominal + 1 "
            "anomaly). Phase-6 'prompting instead of fine-tuning' baseline. Default 0 (off)."
        ),
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Append Qwen3 '/no_think' so the base answers directly (skips <think> block).",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help=(
            "Phase 13: capture a continuous verdict score per window (prefill-only "
            "ANOMALY-vs-NOMINAL logprob) instead of generating advice. Writes to "
            "--results-file (default results/inference_test_scored.json in score mode). "
            "~10x faster than full generation; feeds src/inference/pr_curve.py."
        ),
    )
    args = parser.parse_args()

    gguf_path = Path(args.gguf) if args.gguf else GGUF_PATH
    # In score mode the default output is a separate file so it never clobbers the
    # generation results (results/inference_test.json).
    default_results = SCORED_FILE if args.score else RESULTS_FILE
    results_file = Path(args.results_file) if args.results_file else default_results
    approach_label = args.approach_label
    fewshot_prefix = build_fewshot_prefix(args.few_shot) if args.few_shot > 0 else ""
    if fewshot_prefix:
        print(f"Few-shot: {args.few_shot} example(s)/class from {TRAIN_FILE}", flush=True)
    if args.no_think:
        print("Decoding with Qwen3 /no_think (direct answer).", flush=True)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")

    with open(TEST_FILE) as f:
        samples = [json.loads(line) for line in f]

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if args.score:
        run_scoring(args, gguf_path, samples, results_file)
        return

    # Resume: load prior results; samples are processed in deterministic order, so the
    # next index to run is len(prior results).
    results: list[dict] = []
    start = 0
    if args.resume and results_file.exists():
        prior = json.loads(results_file.read_text())
        results = prior.get("results", [])
        start = len(results)
        if start >= len(samples):
            print(f"Already complete: {start} samples in {results_file}. Nothing to do.")
            write_results(
                results,
                gguf_path,
                partial=False,
                results_file=results_file,
                approach=approach_label,
            )
            return
        print(f"Resuming from sample {start}/{len(samples)} ({results_file}).", flush=True)

    model = load_model(gguf_path, verbose=args.verbose)
    print(f"Model loaded. n_gpu_layers={model.model_params.n_gpu_layers}", flush=True)
    print(f"Running inference on samples {start}..{len(samples)}...", flush=True)

    recent: list[float] = []
    for i in range(start, len(samples)):
        sample = samples[i]
        prompt = format_prompt(sample["instruction"], fewshot_prefix, args.no_think)
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
            write_results(
                results, gguf_path, partial=True, results_file=results_file, approach=approach_label
            )
            print(f"  checkpoint: {len(results)} samples saved to {results_file}", flush=True)

    write_results(
        results, gguf_path, partial=False, results_file=results_file, approach=approach_label
    )
    s = compute_summary(results, gguf_path, partial=False, approach=approach_label)
    print(f"\n=== Inference Results ({s['n_samples']} samples) ===")
    print(f"  Accuracy:  {s['accuracy']:.3f}")
    print(f"  Precision: {s['precision']:.3f}  Recall: {s['recall']:.3f}  F1: {s['f1']:.3f}")
    print(
        f"  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}  TN={s['tn']}  "
        f"Unknown={s['unknown_responses']}"
    )
    print(f"  Avg time:  {s['avg_time_s']:.2f}s/sample")
    print(f"\nResults saved to {results_file}", flush=True)


if __name__ == "__main__":
    main()
