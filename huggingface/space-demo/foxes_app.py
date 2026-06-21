"""mini-FOXES XAI overlay -- Hugging Face Space (Gradio).

The repo's existing Space (`app.py`) is two text boxes; this companion view adds the **image
rendering the repo otherwise lacks**: it displays the per-patch flux-attribution overlay for a
selected held-out SDO/AIA EUV sample, the primary Explainable-AI artifact of the mini-FOXES
reproduction (a from-scratch ViT regressing GOES soft-X-ray flux; see
`huggingface/foxes-repro_MODELCARD.md` and `results/foxes_repro/report.md`).

Design notes
------------
* `predict()` is a plain function returning an **image path** + a markdown caption, so it can be
  called directly in a headless smoke test (no UI launch) -- that is exactly what
  `validate-foxes` / `make foxes-gradio-smoke` does.
* It prefers re-rendering live from the trained checkpoint via `src.foxes.visualize.render`
  (so a freshly selected sample produces a fresh overlay), but **falls back to the pre-rendered
  `results/foxes_repro/attribution_overlay.png`** when the checkpoint / HF pull / heavy deps are
  unavailable (e.g. a free CPU Space, or an offline CI smoke). Either way it returns a valid PNG.
* No model weights are bundled in the Space; on a real deployment, mount the trained checkpoint
  (or pull it from the model repo) and the live-render path activates automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Repo-root-relative artifact locations (this file lives in huggingface/space-demo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "foxes_repro"
ATTRIBUTION_PNG = OUT_DIR / "attribution_overlay.png"
ATTENTION_PNG = OUT_DIR / "attention_sanity.png"
VIZ_META_FILE = OUT_DIR / "viz_meta.json"

# Sample choices the UI offers. "pre-rendered" never touches the network (safe default for a
# free CPU Space / offline smoke); the others trigger a live render of a held-out FOXES sample.
SAMPLE_CHOICES = {
    "Pre-rendered held-out sample (no download)": {"mode": "prerendered"},
    "Held-out FOXES sample #0 (live render)": {"mode": "foxes", "sample_index": 0},
    "Held-out FOXES sample #1 (live render)": {"mode": "foxes", "sample_index": 1},
    "Synthetic random tensor (offline render)": {"mode": "synthetic"},
}
DEFAULT_CHOICE = next(iter(SAMPLE_CHOICES))


def _caption_from_meta() -> str:
    """Build a markdown caption from the viz_meta.json sidecar, if present."""
    if not VIZ_META_FILE.exists():
        return "Per-patch flux-attribution overlay (mini-FOXES)."
    meta = json.load(open(VIZ_META_FILE)).get("viz_meta", {})
    nan = float("nan")
    pred = meta.get("global_pred", nan)
    ferr = meta.get("faithfulness_abs_err", nan)
    gini = meta.get("attribution_gini", nan)
    brightness = meta.get("brightness_pearson_r_mean", nan)
    return (
        f"**Sample:** `{meta.get('sample_label', 'n/a')}`  \n"
        f"**Predicted SXR (norm. log-space):** {pred:.4f}  \n"
        f"**Faithfulness** (|sum per-patch - global|): {ferr:.2e}  \n"
        f"**Attribution Gini:** {gini:.3f} (0 = uniform; concentrated = structured)  \n"
        f"**Attribution vs EUV-brightness r (mean):** {brightness:.3f}"
    )


def _live_render(spec: dict) -> bool:
    """Try to (re)render the overlay live from the trained checkpoint. Return True on success.

    Best-effort: any failure (missing checkpoint, no HF access, heavy deps absent) returns False
    so the caller falls back to the pre-rendered PNG -- the Space stays functional regardless.
    """
    try:
        from src.foxes.visualize import render
    except Exception:
        return False
    args = argparse.Namespace(
        data="synthetic" if spec["mode"] == "synthetic" else "foxes",
        subsample_n=16,
        val_frac=0.25,
        sample_index=int(spec.get("sample_index", 0)),
        channel=0,
        block=-1,
        head=0,
        seed=42,
    )
    try:
        render(args)
        return ATTRIBUTION_PNG.exists()
    except Exception:
        return False


def predict(sample_choice: str = DEFAULT_CHOICE) -> tuple[str, str]:
    """Return (attribution_overlay_png_path, markdown_caption) for the chosen sample.

    The headless smoke (validate-foxes / make foxes-gradio-smoke) calls THIS directly and asserts
    the returned path exists and is a non-empty PNG -- no UI launch required.
    """
    spec = SAMPLE_CHOICES.get(sample_choice, {"mode": "prerendered"})
    if spec["mode"] != "prerendered":
        _live_render(spec)  # best-effort; falls through to the on-disk PNG either way

    if not ATTRIBUTION_PNG.exists():
        raise FileNotFoundError(
            f"{ATTRIBUTION_PNG} not found -- run `make foxes-figures` to render it first."
        )
    return str(ATTRIBUTION_PNG), _caption_from_meta()


def build_demo():
    """Construct the Gradio Blocks UI (imported lazily so the smoke path needs no gradio)."""
    import gradio as gr

    with gr.Blocks(title="mini-FOXES — EUV→SXR flux attribution (XAI)") as demo:
        gr.Markdown(
            "# mini-FOXES — per-patch flux-attribution overlay\n"
            "A from-scratch Vision Transformer regressing GOES soft-X-ray flux from a 7-channel "
            "SDO/AIA EUV stack. The overlay shows the model's **intrinsic spatial attribution** — "
            "the per-patch contributions that provably **sum to the global flux prediction** "
            "(the primary Explainable-AI artifact). Proof-of-mechanism miniature; see the model "
            "card + report for honest limitations."
        )
        choice = gr.Dropdown(choices=list(SAMPLE_CHOICES), value=DEFAULT_CHOICE, label="Sample")
        btn = gr.Button("Show attribution overlay")
        img = gr.Image(label="Per-patch flux-attribution overlay", type="filepath")
        cap = gr.Markdown()
        btn.click(fn=predict, inputs=choice, outputs=[img, cap])
        demo.load(fn=predict, inputs=choice, outputs=[img, cap])
    return demo


if __name__ == "__main__":
    build_demo().launch()
