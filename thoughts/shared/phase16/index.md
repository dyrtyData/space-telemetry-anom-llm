# Phase 16 — Mini-FOXES: ViT + Explainable-AI showcase (planning artifacts)

This folder holds the RPI planning artifacts for **Phase 16**, a standalone from-scratch
**Vision-Transformer + spatial Explainable-AI** showcase that reproduces a faithful *miniature* of
**FOXES** (Goodwin et al. 2026, ApJ, DOI [10.3847/1538-4357/ae5e4a](https://doi.org/10.3847/1538-4357/ae5e4a))
— multi-channel SDO/AIA EUV imagery → GOES soft-X-ray flux **regression**.

It is a **different task** from the repo's anomaly-detection bake-off (error in *dex*, not anomaly
precision) and is deliberately **kept off the comparison scoreboard**. It connects to the rest of the
repo by **shared engineering discipline** (same provenance, `validate-*` gates, matplotlib vocabulary,
Vast.ai training, report + model-card packaging) and adds the repo's first from-scratch `nn.Module`
and first image/heatmap rendering. See the README "Bonus — Mini-FOXES" section.

## Planning artifacts (RPI flow, in order)

1. [`01-research-questions-vit-xai-phase.md`](01-research-questions-vit-xai-phase.md) — the questions that scoped the phase
2. [`02-research-vit-xai-phase.md`](02-research-vit-xai-phase.md) — codebase + external (FOXES/timm/datasets) research
3. [`03-design-discussion-vit-xai-phase.md`](03-design-discussion-vit-xai-phase.md) — options weighed; decision to build a real miniature on a separate branch
4. [`04-structure-outline-mini-foxes-vit-xai.md`](04-structure-outline-mini-foxes-vit-xai.md) — the phased implementation plan (Phases 0–5) with per-phase validation and the Phase-3 cloud-run notes

## Deliverables (live elsewhere in the repo)

- **Code:** [`src/foxes/`](../../../src/foxes/) — `model.py` (`ViTLocal` + `InvertedAttentionBlock`), `attention.py`, `data.py`, `run.py`, `visualize.py`, `validate.py`
- **Report:** [`results/foxes_repro/report.md`](../../../results/foxes_repro/report.md)
- **Metrics:** [`results/foxes_repro/metrics.json`](../../../results/foxes_repro/metrics.json) — MAE 0.368 dex, Pearson r 0.943
- **XAI figures:** [`attribution_overlay.png`](../../../results/foxes_repro/attribution_overlay.png), [`attention_sanity.png`](../../../results/foxes_repro/attention_sanity.png)
- **HF model card:** [`huggingface/foxes-repro_MODELCARD.md`](../../../huggingface/foxes-repro_MODELCARD.md)
- **Gate / target:** `make validate-foxes` (13 checks, incl. faithfulness Σ per-patch ≈ global)

> Note: the original task brief is intentionally not included here (it contained personal
> job-application context not suitable for a public repo). The technical motivation — reproducing the
> published FOXES model to demonstrate ViT + XAI — is captured in the documents above.
