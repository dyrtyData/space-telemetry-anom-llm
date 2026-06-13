"""Export the fine-tuned advice LoRA to GGUF (run ON the Vast.ai instance).

Phase 3->4 bridge. Per the Phase 3 note, export GGUF before terminating the instance,
then pull the file down with scripts/cloud/download_models.sh.

Usage on the instance:
    python src/training/export_gguf.py
    python src/training/export_gguf.py --lora-dir models/lora/qwen3-8b-advice --quant q4_k_m

`save_pretrained_gguf` merges the LoRA into the base model and writes a single GGUF.
Q4_K_M is the safe, widely-supported default for local M3 (llama.cpp) inference.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-dir", default="models/lora/qwen3-8b-advice")
    parser.add_argument("--gguf-dir", default="models/gguf")
    parser.add_argument("--name", default="star-pipeline-advice")
    parser.add_argument("--quant", default="q4_k_m", help="llama.cpp quant type (e.g. q4_k_m).")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    gguf_dir = Path(args.gguf_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.lora_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    out = str(gguf_dir / args.name)
    print(f"Exporting {args.lora_dir} -> {out}.gguf (quant={args.quant}) ...")
    model.save_pretrained_gguf(out, tokenizer, quantization_method=args.quant)

    print(f"GGUF exported under {gguf_dir}")
    print("Download with: ./scripts/cloud/download_models.sh <instance_id>")


if __name__ == "__main__":
    main()
