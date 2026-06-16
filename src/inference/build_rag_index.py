"""Build per-channel FAISS index for RAG retrieval from training windows.

Creates embeddings of training windows using sentence-transformers and builds
a per-channel FAISS index for fast nearest-neighbor lookup. Each index contains
windows from a single (mission, channel) pair so retrieval returns channel-specific
context — matching what the fine-tune learned implicitly.

Usage:
    python src/inference/build_rag_index.py --train data/splits/train.jsonl --out data/rag/

Output:
    data/rag/{mission}__{channel}.faiss - per-channel FAISS indices
    data/rag/manifest.json - index metadata (n_windows, embedding_dim, model)
    data/rag/windows_by_channel.json - persisted window data for retrieval
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def format_window_text(mission: str, channel: str, values: list[float]) -> str:
    """Format window as text for embedding (matches the LLM input format)."""
    values_str = ", ".join([f"{v:.4f}" for v in values[:10]])
    if len(values) > 10:
        values_str += "..."
    return f"mission={mission} channel={channel} values=[{values_str}]"


def load_training_windows(train_path: Path) -> dict[str, list[dict]]:
    """Load training windows grouped by (mission, channel) key."""
    windows_by_channel: dict[str, list[dict]] = defaultdict(list)

    with open(train_path) as f:
        for line in tqdm(f, desc="Loading training windows"):
            record = json.loads(line)
            meta = record["metadata"]
            key = f"{meta['mission']}__{meta['channel']}"

            window_data = record.get("data", [])
            if not window_data:
                instr = record.get("instruction", "")
                start = instr.find("Values: [")
                if start != -1:
                    end = instr.find("]", start)
                    if end != -1:
                        try:
                            val_str = instr[start + 9 : end]
                            window_data = [
                                float(x.strip())
                                for x in val_str.replace("...", "").split(",")
                                if x.strip()
                            ]
                        except (ValueError, IndexError):
                            window_data = []

            windows_by_channel[key].append(
                {
                    "mission": meta["mission"],
                    "channel": meta["channel"],
                    "values": window_data,
                    "is_anomaly": meta["is_anomaly"],
                    "start_idx": meta.get("start_idx"),
                    "response": record.get("response", ""),
                }
            )

    return dict(windows_by_channel)


def build_index(
    windows_by_channel: dict[str, list[dict]],
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Build per-channel FAISS indices."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    manifest = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "channels": {},
    }

    for channel_key, windows in tqdm(windows_by_channel.items(), desc="Building indices"):
        if not windows:
            continue

        texts = [format_window_text(w["mission"], w["channel"], w["values"]) for w in windows]

        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        index_path = output_dir / f"{channel_key}.faiss"
        faiss.write_index(index, str(index_path))

        manifest["channels"][channel_key] = {
            "n_windows": len(windows),
            "n_anomalous": sum(1 for w in windows if w["is_anomaly"]),
            "index_file": f"{channel_key}.faiss",
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    windows_path = output_dir / "windows_by_channel.json"
    with open(windows_path, "w") as f:
        json.dump(windows_by_channel, f)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from training windows")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/splits/train.jsonl"),
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/rag"),
        help="Output directory for indices",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    args = parser.parse_args()

    print(f"Building RAG index from {args.train}")

    windows_by_channel = load_training_windows(args.train)
    total_windows = sum(len(w) for w in windows_by_channel.values())
    print(f"Loaded {total_windows} windows across {len(windows_by_channel)} channels")

    manifest = build_index(windows_by_channel, args.out, args.model)

    print("\nIndex built successfully:")
    print(f"  Output directory: {args.out}")
    print(f"  Channels indexed: {len(manifest['channels'])}")
    print(f"  Embedding dim: {manifest['embedding_dim']}")


if __name__ == "__main__":
    main()
