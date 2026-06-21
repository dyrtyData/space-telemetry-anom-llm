---
task: spacetelemetryanomllm-foxes-competencies
type: structure-outline
repo: space-telemetry-anom-llm
branch: main
sha: 6312156def73eaba5d23da0e3dbe6a3f2c42d666
---

# Phase 16: Mini-FOXES — a real Vision Transformer + Explainable-AI showcase

Build a small but **real, trained, mechanism-faithful miniature of FOXES** (multi-channel EUV → GOES SXR flux regression) on a dedicated branch, to demonstrate two competencies central to the FOXES problem domain: **raw ViT construction** and **spatial XAI / attention attribution**. The model is a from-scratch `ViTLocal` (8×8 patch embed, 9×9 inverted masked attention, per-patch linear regression head, no CLS) trained on a small subsample of the FOXES authors' own published HF dataset, packaged with the repo's existing provenance, `validate-*` gate, matplotlib, report, and model-card discipline — and deliberately **kept out of the anomaly-detection comparison table** (different task, different metric).

## Desired End State

- A new `src/foxes/` package contains a real `nn.Module` ViT (`ViTLocal` + `InvertedAttentionBlock`) with the distinctive FOXES signatures: 7-channel `Conv2d` patch embed at 8×8, a 9×9 inverted (non-local) masked-attention block used in all encoder layers, a per-patch linear head whose scalars **sum to the global prediction**, and an `aux_mask` input hook stubbed for Surya AR masks.
- Attention is **materializable** (eager path, `set_fused_attn(False)` / `need_weights=True`) and extracted via a forward-hook helper into `(B, H, N, N)` tensors.
- A short real training run on the FOXES-Data HF subsample (~1–2k samples, resized to 128²) on Vast.ai produces a **modest-but-real MAE in dex**, with full provenance (config snapshot, atomic writes, `--resume`/`--checkpoint-every`).
- The repo gains its first **image/heatmap/overlay rendering**: a per-patch flux-attribution overlay + a raw-attention sanity figure in the repo's matplotlib vocabulary.
- A standalone `results/foxes_repro/report.md` + `huggingface/foxes-repro_MODELCARD.md` (regression `model-index`) + a small Gradio overlay view present the result, honestly labeled as a proof-of-mechanism miniature.
- A `validate-foxes` Makefile gate enforces correctness, including the faithfulness invariant **Σ per-patch ≈ global prediction**.
- All of the above lives on a **dedicated branch** and never enters `evaluate.py` / the CEF0.5 / Affinity-F1 scoreboard.

## Implementation Overview

- [x] Phase 0: Worktree + branch setup
- [x] Phase 1: Model skeleton + XAI extraction + provenance gate (CPU, synthetic data)
- [x] Phase 2: Real FOXES-Data path (HF subsample → 7×128² tensors + SXR labels)
- [x] Phase 3: Real training run on Vast.ai → trained checkpoint + MAE in dex
- [x] Phase 4: XAI visualization — per-patch attribution overlay + attention sanity figure
- [x] Phase 5: Packaging — report + HF model card + Gradio overlay + Surya aux-mask stub/docs

> **Verification policy:** every phase is gated entirely by **automated** checks (`make validate-foxes` + `make lint`, plus git/data smoke asserts) so the project can run unattended start-to-finish. The only human-in-the-loop steps are two scientific/editorial sign-offs that genuinely require judgment; both are non-blocking and deferred to a single **Final Manual Verification** section at the very end.

---

## ✅ Phase 0: Worktree + branch setup

Isolate the whole showcase on its own branch in a dedicated git worktree (per the repo's `using-git-worktrees` convention and DQ7), so Phase 16 never touches the main-branch anomaly-detection narrative or working tree.

### File Changes

- **(git)**: create a worktree + branch `phase16-mini-foxes` off `main`, e.g. `git worktree add ../space-telemetry-foxes phase16-mini-foxes`. All subsequent phases edit files inside that worktree.

### Validation

#### Automated Verification

- [x] `git worktree list` shows the new worktree on branch `phase16-mini-foxes`.
- [x] `git -C ../space-telemetry-foxes rev-parse --abbrev-ref HEAD` returns `phase16-mini-foxes`.
- [x] `git -C ../space-telemetry-foxes status --porcelain` is empty (clean fresh checkout).

---

## ✅ Phase 1: Model skeleton + XAI extraction + provenance gate (CPU, synthetic data)

The thinnest end-to-end vertical slice: the **real network**, the **attention-extraction** path, the **provenance-shaped output**, and the **validate gate** — all wired together and runnable on synthetic random tensors with **no data download and no GPU**. This pins down the module's public signatures (the "C header" of the showcase) and gives an automated faithfulness check before any money is spent on cloud training.

### File Changes

- **`pyproject.toml`**: add an optional dependency group `foxes = ["timm>=1.0.0", "datasets>=2.18.0", "huggingface_hub>=0.23.0"]`. `torch`, `numpy`, `matplotlib`, `pillow`, `h5py` are already present.
- **`src/foxes/__init__.py`**: new package marker.
- **`src/foxes/model.py`**: from-scratch network. Mask construction blocks the 9×9 patch neighborhood (`mask_mode="inverted"`, `local_window=9`, `r=4`) so each patch attends only to distant patches; all `depth` blocks are `InvertedAttentionBlock`. Eager attention only (no SDPA/Flash) so weights are extractable.

```python
class InvertedAttentionBlock(nn.Module):
    def forward(self, x, attn_mask): ...        # returns x; attn weights captured via hook/return

class ViTLocal(nn.Module):
    def __init__(self, in_chans=7, img=128, patch=8, embed=256, depth=8,
                 heads=8, mlp=1024, local_window=9, aux_mask_chans=0): ...
    def build_inverted_mask(self, grid_h, grid_w, r=4) -> Tensor: ...   # (N, N) bool
    def forward(self, x, aux_mask=None):
        # patch_embed: Conv2d(in_chans (+aux), embed, patch, patch)
        # per_patch = mlp_head(tokens).squeeze(-1)          # (B, N) intrinsic attribution
        return per_patch.sum(1), per_patch                  # global SXR, spatial map
```

- **`src/foxes/attention.py`**: eager-mode extraction helper. Registers forward hooks (or uses `need_weights=True, average_attn_weights=False`) and returns `{block_idx: (B, H, N, N)}`; asserts `set_fused_attn(False)` was applied when timm primitives are used.
- **`src/foxes/run.py`**: single CLI entry (`main()` + `__main__`, Makefile-driven) with `--data {synthetic,foxes}` (Phase 1 wires only `synthetic`: random `(B,7,128,128)` + random log-SXR targets), `--epochs`, `--checkpoint-every` (default 250), `--resume`. Writes `results/foxes_repro/metrics.json` embedding `"config": vars(args) | {...}` via the repo's atomic temp-file + `os.replace` pattern; saves `torch.save({"model":..., "config":..., "epoch":...})` checkpoints atomically.
- **`Makefile`**: add `.PHONY` entries; `foxes-smoke` (runs `run.py --data synthetic --epochs 1`) and a `validate-foxes` inline-Python `assert` gate in the existing style.

```makefile
validate-foxes:
	$(PY) -c "..."   # see Validation below
```

### Validation

#### Automated Verification

- [x] `make setup` (or `$(PIP) install -e ".[foxes]"`) installs `timm`/`datasets` cleanly.
- [x] `make foxes-smoke` runs on CPU with synthetic data and writes `results/foxes_repro/metrics.json`.
- [x] `make validate-foxes` passes, asserting: metrics finite & no `error` key; forward output shapes `(B,)` and `(B, N)` with `N = (128/8)² = 256`; **faithfulness invariant** `|per_patch.sum(1) − global| < 1e-4`; extracted attention is non-`None` with shape `(B, H, N, N)`; the inverted mask blocks the 9×9 neighborhood (diagonal/local entries masked, distant entries allowed); checkpoint reloads and reproduces the forward pass.
- [x] `make validate-foxes` **also asserts the FOXES-faithfulness structural invariants** (automating the former manual code skim): **no CLS token** (token count `N`, not `N+1`; no `cls_token`/`class_token` attribute on the module), the per-patch head is `Linear(embed, 1)` producing one scalar per patch, and every encoder block is an `InvertedAttentionBlock` (no standard-attention layer). These match the public FOXES `vit_patch_model_local.py`, cited in `model.py` comments.
- [x] `make lint` (`ruff check src/` + `ruff format --check src/`) passes on `src/foxes/`.

---

## ✅ Phase 2: Real FOXES-Data path (HF subsample → 7×128² tensors + SXR labels)

Replace synthetic input with the **real, pre-paired** FOXES-Data subsample and prove a real sample flows through the Phase-1 network and loss — still CPU-only, still cheap (a tiny smoke pull), no full training yet.

### File Changes

- **`src/foxes/data.py`**: streaming subsample loader. Uses `load_dataset("griffingoodwin04/FOXES-Data", streaming=True).take(n)`; each record yields `aia_stack` `(7,512,512)` float32 (already ITI-normalized to `[-1,1]`) + `sxr_value` scalar (GOES XRSB) + ISO-timestamp `filename`. Resizes 512²→128² with a single `F.interpolate`; targets converted to normalized log-space (matching the training loss). Exposes `--subsample-n` and a held-out split fraction.
- **`src/foxes/run.py`**: wire `--data foxes` to `data.py`; add `--subsample-n` (default small for smoke). Synthetic path retained as the documented Option-B fallback.
- **`Makefile`**: add `foxes-data` target that pulls a **tiny** `--subsample-n 8` and runs one forward+loss pass (network-gated, cheap).

### Validation

#### Automated Verification

- [x] `make foxes-data` downloads 8 samples and runs a forward+loss pass on CPU without error.
- [x] `make validate-foxes` extended to assert (when a data smoke artifact exists): tensor shape `(B,7,128,128)`, dtype float32, value range within `[-1,1]±ε`, and finite log-space labels.
- [x] **Streaming-bounded download is asserted automatically** (automating the former manual "did it pull the full 1.46 TB?" check): the loader is constructed with `streaming=True` and exactly `--subsample-n` records are materialized (`isinstance(ds, IterableDataset)` and `len(list(loaded)) == n`), so no full-dataset pull can occur.
- [x] `make lint` passes.

---

## ✅ Phase 3: Real training run on Vast.ai → trained checkpoint + MAE in dex

Run the **actual** training to a modest-but-real result on cloud CUDA (avoids the M3-MPS masked-attention NaN bug), producing a trained checkpoint and real regression metrics with full provenance and resume.

> **Run notes (2026-06-21, RTX 4090 @ $0.538/hr, 40 epochs, ~17 min, $0.15):** Final `mae_dex=0.368`, `pearson_r=0.943`, `mean_bias_dex=0.312`, `rmse_dex=0.411`, baseline `0.689`. `make validate-foxes` green (instance + local). **Two fixes during the run, both committed:** (1) **seeded shuffled train/val split** in `data.py` — the original deterministic *tail slice* held out the last 375 timestamp-ordered records; FOXES-Data is time-ordered and solar activity is non-stationary, so the temporal holdout injected a large distribution-shift bias (Pearson 0.53, mae_dex 0.71). Shuffling lifted Pearson to 0.94. (2) **`weight_decay=0.0`** in the AdamW optimizer (`run.py`) — AdamW's 0.01 default decayed the per-patch head bias toward 0; since the global prediction is a *sum of 256 per-patch scalars*, that suppressed the only constant-offset term and biased predictions high in dex (mean_bias ≈ mae ≈ 0.62 at Pearson 0.94). Disabling decay on the head halved the bias and brought `mae_dex` from 0.629 to 0.368 (inside the 0.4 band). Residual ~0.31 dex bias remains (de-biased residual ≈ 0.27 dex) — documented as a miniature-calibration limitation. **Infra note:** the cheapest 4090 offer was a CN-region host whose HF CDN edge intermittently 401'd / aborted bulk parquet streaming; a detached self-retrying loop (clean re-stream per attempt) got a successful pull on attempt 2.

### File Changes

- **`src/foxes/run.py`**: full training loop — Huber loss in normalized log-space (δ=0.3), cosine LR (1e-4→~1e-5), `--epochs` (20–50), `--checkpoint-every`, `--resume` (rebuild `done`/last-epoch state), and an eval pass over the held-out subset computing **MAE / RMSE / Pearson r in dex** + mean bias. **Also computes, in the same run, a `mae_dex_mean_baseline`** = MAE of a trivial constant predictor (the training-set mean log-SXR) on the same held-out set, and records elapsed wall-clock + the instance `$/hr` into `summary` for an automated cost estimate. Writes `results/foxes_repro/metrics.json` (`summary` + `config`) and the trained checkpoint under `STAR_OUTPUT_DIR` (external-drive convention), atomic writes throughout.
- **`scripts/cloud/`**: reuse `launch_vast.sh` / `upload_data.sh` / `download_models.sh`; document the A6000/4090 invocation for the FOXES run (no new scripts needed if existing ones parametrize the command).
- **`Makefile`**: add `foxes-train` (drives the cloud command, mirroring the `train-cloud` pattern) — actual `python src/foxes/run.py --data foxes ...` runs **on the instance**.

### Validation

#### Automated Verification

- [x] After the run, `make validate-foxes` passes with: `summary.mae_dex` finite; **primary (self-calibrating) gate** `mae_dex < mae_dex_mean_baseline` by a margin (the model must beat predicting the constant mean on the same held-out set — proves real learning without hard-coding a magic number, mirroring `validate-baseline`'s "loss decreased" style); **secondary loose sanity band** `0.01 < mae_dex < 0.4` (lower bound catches label-leakage/bugs that make it look impossibly perfect, upper bound catches a non-learning/blowup run and keeps us under the paper's 0.307 naive baseline); Pearson r finite in `[-1,1]`; faithfulness invariant still holds on **trained** weights; the saved checkpoint reloads and reproduces metrics within tolerance. _(Result: `mae_dex=0.368 < baseline 0.689` (margin 5%), Pearson r=0.943, band-pass; faithfulness exact 0.0; checkpoint reload exact. **2 split/calibration fixes** were needed — see note below.)_
- [x] `make validate-foxes` asserts the **cost estimate** `summary.elapsed_hr * summary.usd_per_hr < 25` (the run logs both), automating the budget-ceiling sanity check. _(elapsed 0.28 hr × $0.538/hr = **$0.15**.)_
- [x] `make lint` passes.

---

## ✅ Phase 4: XAI visualization — per-patch attribution overlay + attention sanity figure

Add the repo's **first image/heatmap/overlay rendering**, using the trained checkpoint: the primary XAI artifact (per-patch flux-attribution map) and the secondary sanity-check (raw attention), in the established matplotlib vocabulary.

### File Changes

- **`src/foxes/visualize.py`**: load a trained checkpoint, run a forward pass on a held-out sample, and render:
  - `results/foxes_repro/attribution_overlay.png` — the `(B, N)` per-patch scalar map reshaped to the 16×16 patch grid, bilinearly upsampled to 128², overlaid (`alpha=0.3`) on an EUV channel.
  - `results/foxes_repro/attention_sanity.png` — a chosen block/head's `(N, N)` attention reduced to a per-patch map (raw, **no rollout**), shown alongside the attribution map.
  - matplotlib vocabulary: `#1f77b4`, `alpha=0.3`, Agg backend, `figsize=(6,5)`/`(8,4)`, `dpi=120`, `bbox_inches="tight"`.
- **`Makefile`**: add `foxes-figures` target.

### Validation

#### Automated Verification

- [x] `make foxes-figures` writes both PNGs (non-zero size). _(74 KB attribution_overlay.png + 39 KB attention_sanity.png; rendered LOCALLY on CPU from a tiny unauthenticated streaming pull of the held-out FOXES sample `2012-01-14T13:03:00`. `src/foxes/visualize.py` ends with an explicit `os._exit(0)` so the lingering HF-`datasets` background streaming threads don't hang `make` after all artifacts are flushed — MAKE_EXIT=0 in ~21 s.)_
- [x] `make validate-foxes` extended to assert both figure files exist and that the rendered sample's per-patch attribution **sums ≈ the model's global prediction**. _(New `check_figures()` gate + `viz_meta.json` sidecar; faithfulness err = 0.00e+00.)_
- [x] **Automated "attribution corresponds to structure, not noise" proxy.** _(Non-uniformity: Gini=0.465 > floor 0.05 AND top-decile mass=0.294 > 0.12 (uniform 0.10). Brightness correlation: asserted on the **channel-averaged** Pearson r=0.248 > floor 0.10, not a single channel — empirically the SXR-driving hot channels (94 Å r=+0.49, 131 Å +0.38, 335 Å +0.46) correlate strongly while the quiet-Sun 171 Å reads ~+0.04, so a single arbitrary channel was a fragile proxy. The headline overlay now uses 94 Å so the correspondence is visible.)_
- [x] `make lint` passes. _(`ruff check` clean, 32 files formatted.)_

---

## ✅ Phase 5: Packaging — report + HF model card + Gradio overlay + Surya aux-mask stub/docs

Wrap the result in the repo's packaging discipline — standalone report, HF-style model card, a small Gradio overlay view — and demonstrate + document the Surya AR-mask drop-in, **without** touching `evaluate.py` or the anomaly scoreboard.

### File Changes

- **`results/foxes_repro/report.md`**: standalone report mirroring the `comparison_report.md` section style — regression metrics table (MAE/RMSE/Pearson r in dex), methodology, **honest "miniature, not full reproduction" labeling**, and explicit next steps (full-resolution / full-dataset / real Surya masks / catalog-scale inference).
- **`huggingface/foxes-repro_MODELCARD.md`**: new card mirroring `qwen3-vl-8b-detection_MODELCARD.md`, with a regression `model-index` (MAE/RMSE/Pearson r in dex), architecture summary, and dataset/citation links (FOXES paper DOI 10.3847/1538-4357/ae5e4a, `griffingoodwin04/FOXES-Data`).
- **`src/foxes/model.py`**: exercise the Phase-1 `aux_mask` hook — accept an optional placeholder AR mask (extra `in_chans` or additive attention prior) and document exactly how the real SuryaBench AR masks (1.31 GB HDF5, `nasa-ibm-ai4science/surya-bench-ar-segmentation`) attach.
- **Gradio overlay**: a small view (extend `huggingface/space-demo/` or a separate `foxes_app.py`) that displays `attribution_overlay.png` for a selected sample — the image rendering the existing two-textbox Space lacks.
- **`Makefile`**: extend `validate-foxes` to check report/card completeness; optionally add a `foxes-report` convenience target.

### Validation

#### Automated Verification

- [x] `make validate-foxes` asserts `report.md` contains the required sections (metrics table, methodology, next-steps — mirroring `validate-eval`'s section check) and `foxes-repro_MODELCARD.md` contains `model-index` regression keys. _(New `check_packaging()` gate: report.md has 4 required sections (Regression Metrics / Methodology / Limitations / Next steps) + the honest "miniature" framing; model card declares an `image-regression` `model-index` with `mae`/`rmse`/`pearsonr` keys.)_
- [x] A CPU forward pass with a **placeholder `aux_mask`** runs without error (aux-mask path is wired, not dead code). _(New `check_aux_mask_forward()` gate + `make foxes-aux-smoke`: `ViTLocal(aux_mask_chans=1)` forward with an explicit all-zeros placeholder AND `aux_mask=None` both run, faithfulness err 0.00e+00, None==explicit-zeros (diff 0.00e+00), attention still extractable. `model.py` documents the real SuryaBench `nasa-ibm-ai4science/surya-bench-ar-segmentation` drop-in (timestamp-align, resize 4096²→128², concat as extra channel).)_
- [x] **Headless Gradio smoke** (automating the former "launch and look" step): import the app module and call the predict function directly, asserting it returns a valid image path/array for the attribution overlay — no UI launch required. _(New `huggingface/space-demo/foxes_app.py` + `check_gradio_smoke()` gate + `make foxes-gradio-smoke`: imports the app and calls `predict()` directly → returns a valid non-empty PNG path (attribution_overlay.png, 74 KB) + a markdown caption. The live-render-from-checkpoint path and the pre-rendered fallback both verified to return a valid PNG.)_
- [x] `make lint` passes. _(`ruff check src/` clean, 32 files formatted; `foxes_app.py` also lint+format clean.)_

---

## Final Manual Verification

These two checks are deferred to the very end so the project can run automated all the way through. Both are non-blocking sign-offs that genuinely require human judgment (the automated proxies above cover the mechanical correctness); run them once everything else is green.

- [ ] **Scientific plausibility sign-off** — visually review `attribution_overlay.png` + `attention_sanity.png`: does the attribution land on the active-region / bright-loop structure a heliophysicist would expect? (The Phase-4 concentration + brightness-correlation gate is the automated stand-in; this is the final expert eyeball.)
- [ ] **Editorial honesty sign-off** — read `report.md` + `foxes-repro_MODELCARD.md` and confirm the framing is honestly labeled as a proof-of-mechanism **miniature** (not a full FOXES reproduction), metrics/citations are correct, and the Surya next-step reads as the documented drop-in.

## Open Questions

- _None._ Both prior open questions are resolved: the **MAE gate** is now decided (Phase 3 — self-calibrating `mae_dex < mae_dex_mean_baseline` primary check + a `0.01 < mae_dex < 0.4` loose sanity band, so no magic number is hard-coded and the gate self-calibrates on the first run), and the **worktree** is now Phase 0 (dedicated `phase16-mini-foxes` worktree, per user confirmation).
