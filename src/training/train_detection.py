"""Fine-tune Qwen3-VL-8B for visual anomaly detection (AnomSeer-style).

Phase 3 (§3.7). Runs ON the Vast.ai instance after the PNG plots are uploaded.

Inputs : data/processed/plots/{train,val}_metadata.jsonl
         (each line: {"index", "image_path", "is_anomaly", "mission", "channel"})
         + the referenced PNGs under data/processed/plots/{train,val}/.
Outputs: models/lora/qwen3-vl-detection/  (vision LoRA adapters + processor).

Usage on the instance (from the repo root so image_path is resolvable):
    python src/training/train_detection.py

Corrections vs. the original plan §3.7:
  - model id -> `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (verified on HF; the
    plan's `unsloth/Qwen3-VL-8B` does NOT exist).
  - Uses the real Unsloth vision SFT pattern: a list of {"messages": [...]} samples with
    a user image+text turn and an assistant label turn, plus UnslothVisionDataCollator.
    The plan's `dataset_text_field="text"` / `.map(format_example)` is NOT how vision SFT
    works in Unsloth and would fail.
  - Reads the metadata schema actually emitted by generate_plots.py (image_path/is_anomaly).
"""

import argparse
import json
from pathlib import Path

PLOTS_DIR = Path("data/processed/plots")
DEFAULT_MODEL = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
OUTPUT_DIR = "models/lora/qwen3-vl-detection"
MAX_SEQ_LENGTH = 2048

USER_PROMPT = (
    "This is a plot of a spacecraft telemetry sequence (normalized value vs. timestep). "
    "Does it show anomalous behaviour? Answer with ANOMALY DETECTED or NOMINAL."
)


def build_conversations(split: str, limit: int | None) -> list[dict]:
    """Read a *_metadata.jsonl split and build Unsloth vision conversation samples."""
    from PIL import Image

    meta_file = PLOTS_DIR / f"{split}_metadata.jsonl"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing {meta_file} — run generate_plots.py first.")

    samples: list[dict] = []
    with open(meta_file) as f:
        for line in f:
            rec = json.loads(line)
            img_path = Path(rec["image_path"])
            if not img_path.exists():
                continue
            label = "ANOMALY DETECTED" if rec["is_anomaly"] else "NOMINAL"
            samples.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_PROMPT},
                                {"type": "image", "image": Image.open(img_path).convert("RGB")},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": label}],
                        },
                    ]
                }
            )
            if limit and len(samples) >= limit:
                break
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per split (debug).")
    args = parser.parse_args()

    # Heavy imports inside main() so --help / lint work without a GPU stack.
    # IMPORTANT: import unsloth BEFORE trl/transformers/peft. Unsloth monkey-patches TRL's
    # SFTTrainer/SFTConfig (incl. correct VLM eos_token handling) at import time; if `trl`
    # is imported first, SFTTrainer binds to the UNPATCHED class and the Qwen3VL processor
    # trips a "<EOS_TOKEN>" not-in-vocab ValueError. Order matters here.
    from unsloth import FastVisionModel  # noqa: I001  (unsloth MUST precede trl; do not sort)
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_samples = build_conversations("train", args.limit)
    val_samples = build_conversations("val", args.limit)
    print(f"Vision SFT: {len(train_samples)} train / {len(val_samples)} val samples")

    FastVisionModel.for_training(model)

    # TRL 0.24 + Qwen3VLProcessor: SFTTrainer validates SFTConfig.eos_token against the
    # processor vocab. Unsloth's for_training() swaps tokenizer.eos_token for a literal
    # "<EOS_TOKEN>" placeholder that is NOT in the vocab, so reading it dynamically (or
    # leaving eos_token=None) makes SFTTrainer raise. Pin the canonical Qwen3 chat EOS,
    # which IS in the vocab (id 151645).
    eos_token = "<|im_end|>"

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,  # TRL 0.24: `tokenizer=` kwarg removed
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=train_samples,
        eval_dataset=val_samples,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_steps=50,
            eval_steps=50,
            eval_strategy="steps",
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            seed=42,
            # Vision SFT requirements (Unsloth): keep raw columns, skip text-only prep.
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=1,
            max_length=MAX_SEQ_LENGTH,  # TRL 0.24: `max_seq_length` renamed to `max_length` (D9)
            eos_token=eos_token,  # avoid TRL's "<EOS_TOKEN>" placeholder (not in VL vocab)
        ),
    )

    print(f"Starting vision training: {args.model} -> {args.output_dir}")
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Vision model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
