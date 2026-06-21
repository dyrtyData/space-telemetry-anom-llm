# Cloud training (Vast.ai)

The three scripts here drive the repo's cloud GPU workflow. They are **parametrized via env
vars**, so the same scripts serve both the main-narrative Unsloth fine-tunes and the Phase-16
mini-FOXES ViT run — no FOXES-specific script is needed.

| Script | Role |
|---|---|
| `launch_vast.sh` | search the cheapest matching offer and (with `--create`) launch an instance |
| `upload_data.sh` | rsync `src/` + config to the running instance |
| `download_models.sh` | pull the trained checkpoint back to `STAR_OUTPUT_DIR` (the external drive) |

---

## Phase 16 — mini-FOXES ViT (EUV → GOES SXR regression)

The mini-FOXES run does **not** need the Unsloth/bitsandbytes stack (it is a from-scratch
`nn.Module`, not a VLM LoRA). It needs only `datasets` + `timm` on top of the base PyTorch
image, and it pulls its data directly from HuggingFace **on the instance** (`griffingoodwin04/FOXES-Data`,
streaming-bounded to `--subsample-n` — never the full ~1.46 TB), so `upload_data.sh` only needs
to ship the code.

GPU: RTX **A6000** (48 GB, ~$0.39–0.45/hr) or RTX **4090** (24 GB, sufficient, ~$0.37/hr). The
run is ~10–12 min of pure train on ~1–2k samples × 20–50 epochs at 128²; total ~$1–3 including
spin-up + data pull (ingress free). Well under the $25 ceiling.

> Why cloud CUDA and not the local M3 Max: confirmed PyTorch-MPS bug where masked
> `nn.MultiheadAttention` yields NaNs — and this model is built around masked attention.

### 1. Launch an A6000 (or 4090) with the FOXES dependency stack

```bash
# A6000 (48 GB). For a 4090 use GPU_NAME=RTX_4090 MIN_GPU_RAM=23 (the script default).
GPU_NAME=RTX_A6000 MIN_GPU_RAM=47 \
ONSTART='pip install -q --no-cache-dir "datasets>=2.18.0" "timm>=1.0.0" "huggingface_hub>=0.23.0" pillow && echo FOXES_READY' \
./scripts/cloud/launch_vast.sh            # dry run: prints the cheapest offer

# add --create to actually launch (incurs charges):
GPU_NAME=RTX_A6000 MIN_GPU_RAM=47 \
ONSTART='pip install -q --no-cache-dir "datasets>=2.18.0" "timm>=1.0.0" "huggingface_hub>=0.23.0" pillow && echo FOXES_READY' \
./scripts/cloud/launch_vast.sh --create
```

### 2. Upload the code

`upload_data.sh` already rsyncs `src/` + config; the FOXES data is pulled on the instance from
HF, so no data upload is needed.

```bash
./scripts/cloud/upload_data.sh <instance_id>
```

### 3. Run the training on the instance

SSH in (`vastai ssh-url <instance_id>`), then from `/workspace/star-pipeline`:

```bash
# install the repo so `python -m src.foxes.run` resolves (or just run the module directly)
pip install -e ".[foxes]"

# the real run: ~1.5k samples, 40 epochs, batch 12, cosine LR 1e-4 -> 1e-5
STAR_OUTPUT_DIR=/workspace/star-pipeline/out \
python -m src.foxes.run --data foxes \
    --subsample-n 1500 --epochs 40 --batch 12 \
    --device cuda --usd-per-hr 0.40 \
    --checkpoint-every 5 --resume
```

`make foxes-train` wraps exactly this command (knobs: `FOXES_TRAIN_N`, `FOXES_EPOCHS`,
`FOXES_BATCH`, `FOXES_DEVICE`, `FOXES_USD_PER_HR`). `--resume` rebuilds the epoch/optimizer/
scheduler state from the last checkpoint, so a spot-instance interruption costs at most
`--checkpoint-every` epochs.

The run writes:
- `results/foxes_repro/metrics.json` — `summary` (MAE/RMSE/Pearson r in dex, mean bias,
  `mae_dex_mean_baseline`, `elapsed_hr`, `usd_per_hr`, `est_cost_usd`) + `config`;
- the trained checkpoint under `$STAR_OUTPUT_DIR/foxes_repro/checkpoint.pt`.

### 4. Pull the artifacts back

```bash
# the small metrics.json (scp it back to results/foxes_repro/ on the laptop)
scp -P <port> root@<host>:/workspace/star-pipeline/results/foxes_repro/metrics.json results/foxes_repro/

# the checkpoint -> STAR_OUTPUT_DIR (download_models.sh handles the LoRA/GGUF case; for the
# single FOXES checkpoint a direct scp is simplest):
scp -P <port> root@<host>:/workspace/star-pipeline/out/foxes_repro/checkpoint.pt \
    "${STAR_OUTPUT_DIR:-/Volumes/DUAL DRIVE/star-pipeline}/foxes_repro/checkpoint.pt"
```

### 5. Validate locally and DESTROY the instance

```bash
make validate-foxes      # activates the Phase-3 trained-result gates (mae_dex < baseline, etc.)
vastai destroy instance <instance_id>   # stop the meter
```
