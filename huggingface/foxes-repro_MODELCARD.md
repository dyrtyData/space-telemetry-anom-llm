---
license: mit
language:
- en
library_name: pytorch
pipeline_tag: image-to-image
tags:
- vision-transformer
- vit
- explainable-ai
- xai
- attention
- heliophysics
- solar
- sdo-aia
- goes
- euv
- soft-x-ray-flux
- regression
- foxes
datasets:
- griffingoodwin04/FOXES-Data
metrics:
- mae
- rmse
- pearsonr
model-index:
- name: mini-foxes-vit-euv-sxr-regression
  results:
  - task:
      type: image-regression
      name: EUV -> GOES soft-X-ray flux regression (FOXES reproduction, miniature)
    dataset:
      name: FOXES-Data (held-out subsample, 375 samples, 128x128)
      type: griffingoodwin04/FOXES-Data
      split: validation
    metrics:
    - type: mae
      value: 0.368
      name: MAE (dex)
    - type: rmse
      value: 0.411
      name: RMSE (dex)
    - type: pearsonr
      value: 0.943
      name: Pearson r (dex)
    - type: mae
      value: 0.689
      name: Constant-mean baseline MAE (dex)
---

# mini-FOXES -- EUV -> Soft-X-ray Flux ViT (FOXES reproduction, miniature)

A **from-scratch Vision Transformer** that regresses GOES soft-X-ray (SXR) flux from a 7-channel
SDO/AIA EUV image stack -- a small but **real, trained, mechanism-faithful miniature of FOXES**
(Goodwin et al. 2026). It reproduces the distinctive FOXES architecture (8x8 multi-channel patch
embed, 9x9 *inverted* non-local masked attention, no CLS token, a per-patch linear regression
head whose 256 scalars **sum to the global flux prediction**) and exposes the model's intrinsic
**spatial attribution map** as its primary Explainable-AI artifact. It is **honestly labeled as a
proof-of-mechanism miniature, not a full FOXES reproduction.**

## Results (FOXES-Data held-out subset, 375 samples, 128x128, in dex)

dex = log10 W/m^2 -- the unit the FOXES paper reports error in.

| Metric | Value | Note |
|---|---|---|
| **MAE (dex)** | **0.368** | beats the constant-mean baseline (0.689) by ~47% |
| RMSE (dex) | 0.411 | |
| **Pearson r** | **0.943** | predicted flux tracks true flux tightly in shape |
| Mean bias (dex) | 0.312 | residual systematic high-bias (see limitations) |
| Constant-mean baseline MAE (dex) | 0.689 | trivial predictor on the SAME held-out set |
| FOXES paper (full model) MAE (dex) | 0.051 | for reference -- NOT what this miniature achieves |
| FOXES paper naive baseline (dex) | 0.307 | our 0.368 is slightly above this -- it is a miniature |

**Why it's interesting:** it demonstrates raw ViT construction + spatial XAI on multi-channel
scientific tensors -- the per-patch head decomposes the prediction into a faithful spatial flux
attribution (the patch scalars provably sum to the global flux), and eager attention makes the
`(B, H, N, N)` weights extractable for a raw-attention sanity check (no rollout, faithful to the
paper). Trained for ~17 minutes for ~$0.15.

## Intended use & limitations

- **Direct use:** an EUV->SXR flux regressor + an intrinsic spatial-attribution XAI demo; a
  reference miniature for the FOXES architecture and its non-local masked attention.
- **Out of scope:** operational flux forecasting; full-resolution / full-cadence inference; any
  use needing the paper's 0.051 dex accuracy.
- **Limitations:**
  - **Miniature, not a full reproduction:** 128^2 (vs 512^2), 1,500 samples (vs ~1.4 TB), 40
    epochs / ~17 min (vs 2xA100 / 100 epochs). MAE 0.368 dex is slightly above the paper's 0.307
    naive baseline.
  - **Residual ~0.31 dex systematic bias** (de-biased residual ~0.27 dex) -- the model tracks
    shape (Pearson 0.94) far better than absolute level; a calibration limitation.
  - **Random (shuffled) split, not temporal** -- a temporal holdout is harder due to
    non-stationary solar activity (documented in the report).
  - **Surya AR masks stubbed** -- the `aux_mask` hook is wired + documented but fed a placeholder.

## Architecture

| | |
|---|---|
| Type | from-scratch `ViTLocal` (PyTorch `nn.Module`), adapting + citing FOXES `vit_patch_model_local.py` |
| Input | 7-channel SDO/AIA EUV stack (94/131/171/193/211/304/335 A), 128x128 |
| Patch embed | `Conv2d(7, 256, kernel=8, stride=8)` -> N = 256 patches |
| Attention | 9x9 *inverted* (non-local) masked attention, eager (materializable), all 8 blocks |
| Tokens | no CLS / class token (token count exactly N) |
| Head | per-patch `Linear(256, 1)`; the 256 scalars SUM to the global SXR prediction |
| Params | ~few M (embed 256, depth 8, heads 8, mlp 1024) |
| Output | global SXR flux (scalar) + per-patch attribution map (256-d, sums to global) |

## Training

| | |
|---|---|
| Data | `griffingoodwin04/FOXES-Data` subsample, 1,500 records (1,125 train / 375 held-out), streamed |
| Resolution | 512^2 -> 128^2 via one bilinear `F.interpolate` |
| Target | GOES XRSB flux -> normalized log-space |
| Loss / optimizer | Huber (delta=0.3), AdamW (lr 1e-4 cosine -> 1e-5, **weight_decay=0.0**) |
| Epochs | 40 |
| Compute | 1x NVIDIA RTX 4090 (Vast.ai), ~17 min, ~$0.15 |
| Provenance | config snapshot in every JSON, atomic writes, `--resume` / `--checkpoint-every` |

Two calibration fixes were needed during the run (both documented in the report): a **seeded
shuffled split** (a temporal tail-slice holdout injected a distribution-shift bias on
non-stationary solar activity) and **weight_decay=0.0** (AdamW's default decayed the per-patch
head bias -- the only constant-offset term in a sum-of-256-scalars prediction -- biasing flux
high).

## Usage (PyTorch)

```python
import torch
from src.foxes.model import ViTLocal

ckpt = torch.load("results/foxes_repro/checkpoint.pt", map_location="cpu", weights_only=False)
cfg = ckpt["config"]
model = ViTLocal(in_chans=cfg["in_chans"], img=cfg["img"], patch=cfg["patch"], embed=cfg["embed"],
                 depth=cfg["depth"], heads=cfg["heads"], mlp=cfg["mlp"],
                 local_window=cfg["local_window"], aux_mask_chans=cfg["aux_mask_chans"])
model.load_state_dict(ckpt["model"]); model.eval()

x = torch.randn(1, 7, 128, 128)                       # a 7-channel EUV stack (normalized [-1,1])
global_pred, per_patch = model(x)                     # scalar SXR flux + (1, 256) attribution map
assert torch.allclose(per_patch.sum(1), global_pred)  # faithfulness: patches SUM to the prediction
```

Extract raw attention for the XAI sanity figure:

```python
from src.foxes.attention import extract_attention
attn = extract_attention(model, x)                    # {block_idx: (B, H, N, N)} -- eager, materializable
```

## Citation & data attribution

Reproduction of **FOXES** -- Goodwin et al. 2026, ApJ, DOI
[10.3847/1538-4357/ae5e4a](https://doi.org/10.3847/1538-4357/ae5e4a). Trained on the authors'
published **FOXES-Data** (`griffingoodwin04/FOXES-Data`). Surya / SuryaBench (the documented
AR-mask drop-in) -- arXiv [2508.14112](https://arxiv.org/abs/2508.14112), masks
`nasa-ibm-ai4science/surya-bench-ar-segmentation`. This is an independent, honestly-labeled
miniature adapting + citing the public FOXES `vit_patch_model_local.py`; released under MIT.
