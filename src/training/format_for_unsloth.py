"""Format JSONL splits into ChatML text for Unsloth SFT.

Phase 3 (§3.5). Reads the **advice-enriched** splits produced by Phase 1.5
(`data/splits/{train,val,test}_with_advice.jsonl`), where the diagnostic advice is
already merged into the ``response`` field (DIAGNOSIS / ADVICE / ACTION lines). We do
NOT re-key advice by a synthetic ``mission_channel_start-end`` id (the plan's original
§3.5 lookup always missed — the real ``anomaly_id`` is ``mission__channel__start_time``).

Output: ``data/formatted/{split}_chatml.jsonl`` with a single ``text`` field, which is
what ``config/unsloth-train.yaml`` / ``train_advice.py`` consume.

Run: ``python src/training/format_for_unsloth.py`` (or ``make format-train``).
"""

import argparse
import json
from pathlib import Path

SPLITS_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/formatted")

SYSTEM_PROMPT = (
    "You are a spacecraft telemetry analyst. Analyze telemetry sequences and identify "
    "anomalies. When an anomaly is detected, provide a diagnosis, diagnostic advice, and "
    "a recommended action to help engineers resolve the issue."
)


def format_as_chatml(instruction: str, response: str) -> str:
    """Wrap an instruction/response pair as a single ChatML training string."""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def pick_input(split: str) -> Path | None:
    """Prefer the advice-enriched split; fall back to the plain split."""
    enriched = SPLITS_DIR / f"{split}_with_advice.jsonl"
    plain = SPLITS_DIR / f"{split}.jsonl"
    if enriched.exists():
        return enriched
    if plain.exists():
        return plain
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to format (default: train,val,test).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        input_file = pick_input(split)
        if input_file is None:
            print(f"Skipping {split}: no input file found in {SPLITS_DIR}")
            continue

        output_file = OUTPUT_DIR / f"{split}_chatml.jsonl"
        n = n_anom = 0
        with open(input_file) as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                record = json.loads(line)
                text = format_as_chatml(record["instruction"], record["response"])
                f_out.write(json.dumps({"text": text}) + "\n")
                n += 1
                n_anom += bool(record["metadata"].get("is_anomaly"))

        print(
            f"Formatted {split}: {n} records ({n_anom} anomalous) "
            f"from {input_file.name} -> {output_file}"
        )


if __name__ == "__main__":
    main()
