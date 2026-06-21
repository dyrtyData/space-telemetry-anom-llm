---
type: research-questions
---

# Research Questions

Context: the STAR-Pipeline repo (Phases 1–14 largely complete) is being extended with a
Vision-Transformer / Explainable-AI showcase motivated by the **FOXES** problem domain (Goodwin et
al. 2026) — Vision Transformers, Explainable AI (attention / spatial attribution), rare-event
regression, Surya-derived active-region masks, and operationalizing a published TRL-5 ViT toward
TRL-7/8. These questions
document how the project works today and gather the external context needed to scope a new
phase. They are descriptive only — research what exists, not what should be built.

## Current codebase — vision & model internals

1. In `src/training/train_detection.py` and `src/inference/eval_vision.py`, how does the
   current "vision" path actually work end to end — what model is loaded (Qwen3-VL via Unsloth
   `FastVisionModel`), what is its input (the matplotlib PNG line-plots from
   `src/etl/generate_plots.py`), and is any model architecture defined in-repo (custom
   `nn.Module`, `nn.MultiheadAttention`, ViT/patch-embedding, or `timm`/`torchvision` layers)
   versus loaded entirely from pretrained weights?

2. What does the image data layer look like today: how does `src/etl/generate_plots.py` render
   windows to PNGs, what are the image dimensions/channels (single-panel RGB line-plot vs.
   multi-channel scientific tensor), and what does the `*_metadata.jsonl` sidecar schema
   (`image_path`, `is_anomaly`, `mission`, `channel`) contain? How many plots exist per split?

## Current codebase — evaluation, MLOps & "showcase" surface

3. How is a new modeling approach plugged into the comparison harness today? Trace how
   `src/inference/evaluate.py` discovers and loads each approach's result file (the per-channel
   vs. per-window loader families), the `results/*.json` schemas it expects, the metrics it
   computes (`cef_from_pr`, `affinity_f1`), and the `Makefile` targets / `validate-*` gates that
   produce and check those files.

4. What operational / packaging surface already exists that signals "TRL-7/8" maturity — e.g.
   the Gradio demo (`huggingface/space-demo/app.py`), HuggingFace model cards
   (`huggingface/*_MODELCARD.md`), the Kaggle notebook (`contributions/kaggle/...ipynb`),
   provenance patterns (run `config`/`summary` snapshots, atomic checkpoint writes, `--resume`),
   and any API/MCP-style access patterns? How does the project currently present competencies?

5. How are results currently visualized (matplotlib conventions in `src/inference/pr_curve.py`
   and `generate_plots.py`, the PR-curve/comparison-report outputs) and how does the Gradio
   space present model output to a viewer? Document existing figure/plot and demo presentation
   patterns relevant to rendering attention/saliency maps.

## External — the FOXES baseline & required techniques

6. What does the FOXES paper (Goodwin et al. 2026, ApJ, DOI 10.3847/1538-4357/ae5e4a) actually
   specify: the ViT architecture (7-channel SDO/AIA stack → GOES 1–8 Å SXR flux, patch size,
   embedding dim, encoder/attention config, patch-level regression head, log-space target,
   global-context masking), the training/validation/test split procedure, the evaluation
   metrics, and how patch outputs + attention provide spatial attribution? Compare these specifics
   to the architecture summary given in the ticket.

7. What are the external building blocks the role names, and what are their concrete capabilities
   and interfaces: (a) the Surya solar foundation model and Surya-derived active-region
   segmentation masks — availability, data format, how masks are produced/consumed; (b) extracting
   raw attention matrices from PyTorch transformers via forward hooks on attention modules, plus
   attention-rollout / attention-map normalization back to image patches; (c) `timm` (and/or raw
   PyTorch) Vision Transformers configured for multi-channel (>3) scientific image tensors —
   patch-embed conv `in_chans`, pretrained-weight adaptation, and attention access.

8. What SDO/AIA EUV + GOES SXR data sources and tooling are commonly used in heliophysics ML
   (e.g. SunPy, JSOC/`drms`, AIA channel set, GOES/XRS flux retrieval, the SDOML / FDL datasets),
   including dataset sizes and access patterns — to understand what a reproduction or a
   small representative demo would require versus the ESA-AD satellite-telemetry data the repo
   uses today?
