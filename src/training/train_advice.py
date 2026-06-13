"""Fine-tune Qwen3-8B for telemetry anomaly diagnosis + advice (Unsloth SFT).

Phase 3 (§3.6). Runs ON the Vast.ai RTX 4090 instance (NOT locally — Unsloth needs CUDA).

Inputs : data/formatted/{train,val}_chatml.jsonl  (single `text` field, ChatML;
         produced by src/training/format_for_unsloth.py)
Outputs: models/lora/qwen3-8b-advice/  (LoRA adapters + tokenizer) on the instance,
         then exported to GGUF by export_gguf.py before teardown.

Usage on the instance:
    python src/training/train_advice.py                       # uses defaults below
    python src/training/train_advice.py --config config/unsloth-train.yaml

Corrections vs. the original plan §3.6:
  - model id -> `unsloth/Qwen3-8B-unsloth-bnb-4bit` (verified on HF; the plan's
    `unsloth/Qwen3-8B-bnb-4bit` also exists but the Dynamic 4-bit variant is preferred).
  - `evaluation_strategy` -> `eval_strategy` (renamed in transformers >= 4.46).
  - dataset points at data/formatted/*_chatml.jsonl, not the raw splits.
"""

import argparse

DEFAULTS = {
    "model_name": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "max_seq_length": 2048,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "train_file": "data/formatted/train_chatml.jsonl",
    "eval_file": "data/formatted/val_chatml.jsonl",
    "text_field": "text",
    "output_dir": "models/lora/qwen3-8b-advice",
    "batch_size": 2,
    "grad_accum": 8,
    "epochs": 3,
    "lr": 2e-4,
    "warmup_ratio": 0.05,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
}


def load_config(path: str | None) -> dict:
    """Merge YAML config (if given) over DEFAULTS, flattening the nested structure."""
    cfg = dict(DEFAULTS)
    if not path:
        return cfg
    import yaml  # local import so the module imports without PyYAML when no config used

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    m = raw.get("model", {})
    lora = raw.get("lora", {})
    ds = raw.get("dataset", {})
    tr = raw.get("training", {})
    lg = raw.get("logging", {})
    out = raw.get("output", {})
    cfg.update(
        model_name=m.get("name", cfg["model_name"]),
        max_seq_length=m.get("max_seq_length", cfg["max_seq_length"]),
        lora_rank=lora.get("rank", cfg["lora_rank"]),
        lora_alpha=lora.get("alpha", cfg["lora_alpha"]),
        lora_dropout=lora.get("dropout", cfg["lora_dropout"]),
        target_modules=lora.get("target_modules", cfg["target_modules"]),
        train_file=ds.get("train_file", cfg["train_file"]),
        eval_file=ds.get("eval_file", cfg["eval_file"]),
        text_field=ds.get("text_field", cfg["text_field"]),
        output_dir=out.get("lora_dir", cfg["output_dir"]),
        batch_size=tr.get("batch_size", cfg["batch_size"]),
        grad_accum=tr.get("gradient_accumulation_steps", cfg["grad_accum"]),
        epochs=tr.get("num_epochs", cfg["epochs"]),
        lr=tr.get("learning_rate", cfg["lr"]),
        warmup_ratio=tr.get("warmup_ratio", cfg["warmup_ratio"]),
        optim=tr.get("optim", cfg["optim"]),
        lr_scheduler_type=tr.get("lr_scheduler_type", cfg["lr_scheduler_type"]),
        weight_decay=tr.get("weight_decay", cfg["weight_decay"]),
        max_grad_norm=tr.get("max_grad_norm", cfg["max_grad_norm"]),
        logging_steps=lg.get("logging_steps", cfg["logging_steps"]),
        save_steps=lg.get("save_steps", cfg["save_steps"]),
        eval_steps=lg.get("eval_steps", cfg["eval_steps"]),
    )
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to unsloth-train.yaml (optional).")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Heavy imports kept inside main() so `--help` and linting work without a GPU stack.
    # Import unsloth FIRST so its optimizations patch trl/transformers (per Unsloth warning).
    from unsloth import FastLanguageModel  # noqa: I001
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["eval_file"]},
    )

    # TRL 0.24 API: training args, dataset_text_field and max_length all live on SFTConfig;
    # the tokenizer is passed as processing_class (the old SFTTrainer kwargs were removed).
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        warmup_ratio=cfg["warmup_ratio"],
        optim=cfg["optim"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_steps=cfg["eval_steps"],
        eval_strategy="steps",  # NOT evaluation_strategy (removed in transformers>=4.46)
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
        dataset_text_field=cfg["text_field"],
        max_length=cfg["max_seq_length"],
        dataset_num_proc=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print(f"Starting training: {cfg['model_name']} -> {cfg['output_dir']}")
    trainer.train()

    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"Training complete! LoRA saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
