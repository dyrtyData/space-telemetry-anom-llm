"""RAG retrieval harness for fetching similar training windows.

Provides fast nearest-neighbor lookup of training windows by (mission, channel),
using the per-channel FAISS indices built by build_rag_index.py.

Usage:
    from rag_retrieve import RAGRetriever
    retriever = RAGRetriever("data/rag/")
    neighbors = retriever.retrieve("Mission1", "channel_41", [0.1, 0.2, ...], k=5)
"""

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def format_window_text(mission: str, channel: str, values: list[float]) -> str:
    """Format window as text for embedding (matches the LLM input format)."""
    values_str = ", ".join([f"{v:.4f}" for v in values[:10]])
    if len(values) > 10:
        values_str += "..."
    return f"mission={mission} channel={channel} values=[{values_str}]"


class RAGRetriever:
    """Retrieves similar training windows for a given query window."""

    def __init__(self, rag_dir: Path | str = "data/rag/"):
        self.rag_dir = Path(rag_dir)

        manifest_path = self.rag_dir / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        windows_path = self.rag_dir / "windows_by_channel.json"
        with open(windows_path) as f:
            self.windows_by_channel = json.load(f)

        print(f"Loading embedding model: {self.manifest['model']}")
        self.model = SentenceTransformer(self.manifest["model"])

        self.loaded_indices: dict[str, faiss.Index] = {}

    def _get_index(self, channel_key: str) -> faiss.Index | None:
        """Load and cache a per-channel FAISS index."""
        if channel_key in self.loaded_indices:
            return self.loaded_indices[channel_key]

        if channel_key not in self.manifest["channels"]:
            return None

        index_file = self.manifest["channels"][channel_key]["index_file"]
        index_path = self.rag_dir / index_file
        index = faiss.read_index(str(index_path))
        self.loaded_indices[channel_key] = index
        return index

    def retrieve(
        self,
        mission: str,
        channel: str,
        values: list[float],
        k: int = 5,
        include_labels: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the k most similar training windows for this (mission, channel).

        Args:
            mission: Mission name (e.g., "Mission1")
            channel: Channel name (e.g., "channel_41")
            values: Normalized telemetry values for the query window
            k: Number of neighbors to retrieve
            include_labels: Whether to include ground-truth labels in results

        Returns:
            List of dicts with {values, label, distance, response} for each neighbor.
            Returns empty list if no index exists for this channel.
        """
        channel_key = f"{mission}__{channel}"
        index = self._get_index(channel_key)
        if index is None:
            return []

        windows = self.windows_by_channel.get(channel_key, [])
        if not windows:
            return []

        query_text = format_window_text(mission, channel, values)
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding.astype(np.float32)

        k = min(k, len(windows))
        distances, indices = index.search(query_embedding, k)

        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            window = windows[idx]
            result = {
                "values": window["values"],
                "distance": float(dist),
                "rank": i,
            }
            if include_labels:
                result["label"] = "ANOMALY" if window["is_anomaly"] else "NOMINAL"
                result["response"] = window.get("response", "")
            results.append(result)

        return results

    def get_channel_stats(self, mission: str, channel: str) -> dict | None:
        """Get statistics for a channel's training data."""
        channel_key = f"{mission}__{channel}"
        return self.manifest["channels"].get(channel_key)


def format_rag_context(
    neighbors: list[dict],
    mission: str,
    channel: str,
    include_labels: bool = True,
) -> str:
    """Format retrieved neighbors as context for the LLM prompt."""
    if not neighbors:
        return f"No historical data available for channel {channel}."

    lines = [
        f"Historical windows from {mission}/{channel} similar to this one:",
        "",
    ]

    for i, n in enumerate(neighbors, 1):
        values_str = ", ".join([f"{v:.4f}" for v in n["values"][:10]])
        if len(n["values"]) > 10:
            values_str += "..."

        if include_labels:
            lines.append(f"  Example {i}: values=[{values_str}] -> {n['label']}")
        else:
            lines.append(f"  Example {i}: values=[{values_str}]")

    return "\n".join(lines)
