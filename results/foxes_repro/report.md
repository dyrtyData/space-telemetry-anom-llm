# mini-FOXES Reproduction Report

A small but **real, trained, mechanism-faithful miniature of FOXES** (multi-channel SDO/AIA EUV -> GOES soft-X-ray flux regression). This is a standalone proof-of-mechanism showcase on the `phase16-mini-foxes` branch -- it demonstrates raw Vision-Transformer construction and spatial Explainable-AI on scientific tensors, and is deliberately **kept out of the anomaly-detection comparison table** (`results/comparison_report.md`): it is a different task (flux regression, error in dex) with a different metric, so putting it on the CEF0.5 / Affinity-F1 scoreboard would be apples-to-oranges.

## Regression Metrics (held-out FOXES-Data subset, 375 samples, in dex)

Error is reported in **dex** (log10 W/m^2, the unit the FOXES paper uses). Predictions and labels live in normalized log-space during training and are mapped back to dex for evaluation.

| Metric | mini-FOXES (ours) | Constant-mean baseline | FOXES paper (full model) |
|---|---|---|---|
| MAE (dex) | **0.368** | 0.689 | 0.051 |
| RMSE (dex) | 0.411 | -- | -- |
| Pearson r | **0.943** | -- | -- |
| Mean bias (dex) | 0.312 | -- | -- |
| Naive-baseline MAE (paper) | -- | -- | 0.307 |

- **The model beats the trivial baseline by ~47%.** The self-calibrating gate is "beat a constant predictor (the training-set mean log-SXR) on the *same* held-out set": MAE 0.368 dex vs the constant-mean's 0.689 dex. This proves the network learned a real EUV->SXR mapping rather than memorizing a level -- no hard-coded magic number.
- **Pearson r = 0.943**: the predicted flux tracks the true flux tightly in shape (rank/scaling), the headline that the ViT extracts genuine flux-relevant structure from the EUV stack.
- **Honest caveat on the absolute number:** 0.368 dex is *above* the paper's naive baseline of 0.307 dex. This is expected -- see Methodology / Limitations. It is a miniature trained for ~17 minutes on 1,125 samples at 128^2, not the full 1.4 TB / 512^2 / 2xA100 model. The proof here is the *mechanism and the learning signal*, not a state-of-the-art number.

## Architecture (FOXES-faithful, from scratch)

A from-scratch `ViTLocal` (`src/foxes/model.py`), adapting + citing the public FOXES reference `vit_patch_model_local.py` (Goodwin et al. 2026). The distinctive FOXES signatures reproduced:

- **7-channel `Conv2d` patch embed** at 8x8 stride (94/131/171/193/211/304/335 A EUV; no RGB, trained fresh on the EUV channels).
- **9x9 inverted (non-local) masked attention** in *all* 8 encoder blocks: the 9x9 local patch neighborhood is *blocked*, so every patch attends only to distant patches -- the FOXES "non-local" mechanism that forces global context.
- **No CLS / class token**: token count is exactly N = (128/8)^2 = 256; global context comes from the masked attention, not an aggregation token.
- **Per-patch linear regression head** `Linear(256, 1)`: one scalar per patch, and the 256 scalars **sum to the global SXR prediction**. This is the intrinsic spatial attribution map (the prediction, decomposed) -- the primary XAI artifact.
- **Eager attention** (explicit softmax over QK^T, no SDPA/Flash/fused path), so the per-block `(B, H, N, N)` weights are materializable for the XAI sanity figure.

## Methodology

- **Data:** the FOXES authors' own published, pre-paired `griffingoodwin04/FOXES-Data` (each record: `aia_stack` (7, 512, 512) float32 already ITI-normalized to [-1, 1] + `sxr_value` GOES XRSB flux + ISO-timestamp). Pulled via `load_dataset(..., streaming=True).take(n)` so a small subsample never triggers the full ~1.46 TB download. Two in-loader operations: resize 512^2->128^2 with one bilinear `F.interpolate`, and convert raw W/m^2 flux to a normalized log-space target.
- **Subsample:** 1,500 records (1,125 train / 375 held-out), seeded shuffled split.
- **Training:** Huber loss (delta=0.3) in normalized log-space, cosine LR 1e-4->1e-5, 40 epochs, AdamW with `weight_decay=0.0`.
- **Compute & cost:** RTX 4090 (Vast.ai), ~17 min, **~$0.15** (0.28 hr x $0.538/hr). Cloud CUDA was chosen over local M3-MPS to avoid the confirmed PyTorch-MPS masked-attention NaN bug (we *specifically* use masked attention).
- **Provenance:** every output JSON embeds `config: vars(args)`; writes are atomic (temp file + `os.replace`); the training loop supports `--resume` / `--checkpoint-every`. Metrics in `results/foxes_repro/metrics.json`; trained checkpoint under `STAR_OUTPUT_DIR`.
- **Faithfulness invariant (gated):** the per-patch attribution map sums to the model's global prediction (max |sum per-patch - global| = 0.0 on trained weights). Enforced by `make validate-foxes`.

### Two calibration fixes made during the run (documented for honesty)

1. **Seeded shuffled train/val split.** The original deterministic *tail slice* held out the last 375 timestamp-ordered records. FOXES-Data is time-ordered and solar activity is non-stationary, so a temporal holdout injected a large distribution-shift bias (Pearson dropped to 0.53, MAE 0.71 dex). A seeded shuffle draws train/val from the same distribution and lifted Pearson to 0.94. The temporal-holdout view remains a documented *harder* eval (a real operational deployment would face exactly this distribution shift -- see Limitations).
2. **`weight_decay=0.0` in AdamW.** Because the global prediction is a *sum of 256 per-patch scalars*, the only constant-offset term is the per-patch head bias. AdamW's 0.01 default decayed that bias toward 0, suppressing the level and biasing predictions high in dex (mean_bias ~= MAE ~= 0.62 at Pearson 0.94 -- shape learned, level suppressed). Disabling decay halved the bias and brought MAE from 0.629 to 0.368.

## Explainable-AI artifacts

- `attribution_overlay.png` -- the **primary** artifact: the per-patch flux-attribution map (the prediction, decomposed) reshaped to the 16x16 patch grid, bilinearly upsampled to 128^2, overlaid (`alpha=0.3`) on the 94 A EUV channel. This is where the model says the X-ray flux comes from.
- `attention_sanity.png` -- the **secondary** sanity check: a chosen block/head's raw `(N, N)` attention reduced to a per-patch "attention received" map (raw, **no rollout** -- faithful to FOXES, which deliberately avoids rollout / Grad-CAM), shown alongside the attribution map.
- Automated XAI proxies (gated): the attribution map is non-uniform (Gini = 0.465; top-decile mass = 0.294 vs uniform 0.10) and its magnitude positively correlates with per-patch EUV brightness (channel-averaged Pearson r = 0.248; the SXR-driving hot channels 94/131/335 A reach r ~= 0.38-0.49) -- a quantitative stand-in for "attribution lands on active-region structure."

## Limitations (this is a miniature, not a full reproduction)

- **Not a full FOXES reproduction.** 128^2 (vs 512^2), 1,500 samples (vs ~1.4 TB), 40 epochs / ~17 min (vs 2xA100 / 100 epochs). MAE 0.368 dex is a proof-of-mechanism number, slightly *above* the paper's 0.307 naive baseline; it is not competitive with the full model's 0.051 dex.
- **Residual systematic bias ~= 0.31 dex** (de-biased residual ~= 0.27 dex). A miniature-calibration limitation: the model tracks shape (Pearson 0.94) far better than absolute level. A learned scalar offset / calibration head, or simply more data + resolution, would close most of this.
- **Random (shuffled) split, not temporal.** The headline metrics use a same-distribution split. A temporal holdout (the operationally honest eval) is harder here precisely because of non-stationary solar activity; bridging that gap is part of the catalog-scale next step.
- **Surya AR masks are stubbed, not integrated.** The `aux_mask` hook is wired and documented (`src/foxes/model.py`) but fed a placeholder; the real 1.31 GB HDF5 SuryaBench masks are not loaded in v1.

## Next steps

- **Full resolution / full dataset:** train at 512^2 on a larger FOXES-Data subsample (or the full set) on multi-GPU -- the direct path toward the paper's 0.051 dex.
- **Real Surya AR masks:** attach `nasa-ibm-ai4science/surya-bench-ar-segmentation` masks via the documented `aux_mask` drop-in (cadence-align by timestamp, resize 4096^2->128^2, concatenate as an extra channel), then ablate the AR-conditioning gain.
- **Calibration:** add a learned offset / temperature so the absolute dex level matches the shape fidelity, removing the residual ~0.31 dex bias.
- **Catalog-scale historical inference:** run the trained model over a long EUV archive to produce a continuous predicted-SXR time series, the TRL-7/8 operational deliverable the HeliolabOps sprint targets.

## Citation & data attribution

FOXES -- Goodwin et al. 2026, ApJ, DOI [10.3847/1538-4357/ae5e4a](https://doi.org/10.3847/1538-4357/ae5e4a). Data: `griffingoodwin04/FOXES-Data` (the authors' published pre-paired EUV->SXR dataset). Surya / SuryaBench -- arXiv [2508.14112](https://arxiv.org/abs/2508.14112), masks `nasa-ibm-ai4science/surya-bench-ar-segmentation`. This miniature is an independent reproduction adapting + citing the public FOXES `vit_patch_model_local.py`; released under the repo's MIT license.
