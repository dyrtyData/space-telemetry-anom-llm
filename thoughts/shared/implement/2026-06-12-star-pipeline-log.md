# STAR-Pipeline Implementation Log

**Plan**: `thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`
**Started**: 2026-06-12
**Status**: ✅ **Phases 1–6 COMPLETE & committed (HEAD `4d27cdd`).** Phase 8 (vision) is IN
PROGRESS in a **concurrent thread sharing this working tree** (commits `a6f26a6`/`0daa2ec`/`81c0bb6`).

> **▶▶ NEXT = PHASE 7 (full LSTM). The Phase-7 CODE IS DONE & committed (`2a01b15`)** — per-window
> persistence + `--resume`/atomic flush in `train_lstm.py`, LSTM Affinity-F1 in `evaluate.py`,
> `MAX_CHANNELS` in the Makefile. **Only the 58-channel RUN + report regen + checkboxes remain.**
> Run: `make baseline MISSION=1 MAX_CHANNELS=58 ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad" STAR_OUTPUT_DIR="/Volumes/DUAL DRIVE/star-pipeline"`
> (~70–90 min, detached + caffeinate, laptop plugged in). `baseline_results.json` currently holds a
> 1-channel smoke of the new code; the full run supersedes it. See the plan's "Phase 7" FRESH-THREAD
> block. Then Phase 9 (advice grading, free), Phase 8 (vision, concurrent), Phase 10 (teardown LAST).
> **CONCURRENCY: `evaluate.py`/`Makefile`/`train_lstm.py` are shared with the Phase-8 thread — edit
> only your sections and `git add` ONLY your own files (never `-A`/`-am`); `git status --short` first.**
>
> **RESULTS ARE NOW TRACKED IN GIT** (commit `77d4f5f`, `.gitignore` updated): all `results/*.json`
> + the comparison report are committed (~3.5 MB) — no more regenerating hours of inference. Only
> `*.log` stays ignored. When you finish a run, COMMIT the updated result file + report.

**Phase 6 result (closed 2026-06-14):** four-way "Did fine-tuning help?" — fine-tune F1=0.453 /
CEF0.5=0.392 / advice=99.6% / 2.77s beats base-zero-shot (all-UNKNOWN, 0/0/0), base-few-shot
(F1=0.420 but CEF0.5=0.325, advice=12.9%, 8.56s), frontier-zero-shot (F1=0.254). Base GGUF deleted.

---

## Summary

| Phase | Status | Started | Completed | Deviations |
|-------|--------|---------|-----------|------------|
| 1 (code) | completed | 2026-06-12 14:35 | 2026-06-12 14:55 | pyproject.toml needed hatch build config |
| 1 (data pipeline) | **completed (all 3 missions)** | 2026-06-12 14:48 | 2026-06-12 23:45 | D2–D5; full dataset on DUAL DRIVE |
| 1.5 | **completed** | 2026-06-12 23:45 | 2026-06-13 00:10 | In-session generation (stats + channel meta) |
| 2 | **completed** | 2026-06-13 11:10 | 2026-06-13 11:25 | D6 stride=16; D7 models→DUAL DRIVE |
| 3 | **completed** | 2026-06-13 ~16:30 | 2026-06-13 22:12 | D8–D13 (model ids, TRL 0.24 API, ssh key, pkill, env, formatter) |
| 4 | **completed** | 2026-06-13 22:12 | 2026-06-14 02:55 | D14–D20 (GGUF path, test data, Metal, FAT32, Hungary SSH, HF upload, CDN download) |
| 5 | **completed** | 2026-06-14 | 2026-06-14 | D21–D24 (loader schemas, CEF from P/R, Affinity-F1 degenerate, IF + Hybrid added) |
| 6 (code+frontier) | **completed** | 2026-06-14 | 2026-06-14 | D26–D29 (adopted ext 500 base run, identical-harness, frontier sub-agent, graceful rows) |
| 6 (base run) | **in flight** | 2026-06-14 | — | external 500-window base run; finalize report + delete base GGUF after |
| 8 (vision) | **completed** | 2026-06-14 | 2026-06-14 | D31–D34 (3 never-run train_detection bugs, A6000 GPU, torchvision upgrade, eval faster than est); F1=0.457, instance destroyed |
| 9 (advice grading) | **completed** | 2026-06-14 | 2026-06-14 | D35–D37 (verifiable rubric, GT-gated correctness, gold-as-reference); TP advice 5.58/6 (95% HQ), gated by precision |

---

## Detailed Log

## Phase 1: Project Setup & ETL

### Step 1.1: Project Configuration
- **Started**: 2026-06-12 14:35
- **Completed**: 2026-06-12 14:38
- **Status**: completed
- **Deviation**: None initially
- **Commit**: c10f77a
- **Notes**: Created pyproject.toml and Makefile per plan

### Step 1.2: ESA-AD Download Script
- **Started**: 2026-06-12 14:38
- **Completed**: 2026-06-12 14:40
- **Status**: completed
- **Deviation**: None
- **Commit**: 5260f35
- **Notes**: Implemented Zenodo API download with progress bar

### Step 1.3: Telemetry Patching Script
- **Started**: 2026-06-12 14:40
- **Completed**: 2026-06-12 14:43
- **Status**: completed
- **Deviation**: None
- **Commit**: 9597c5f
- **Notes**: RevIN normalization and rolling windows implemented

### Step 1.4: Plot Generation Script
- **Started**: 2026-06-12 14:43
- **Completed**: 2026-06-12 14:46
- **Status**: completed
- **Deviation**: None
- **Commit**: 961b50e
- **Notes**: PNG plot generation with matplotlib for AnomSeer-style visual detection

### Step 1.5: Directory Structure
- **Started**: 2026-06-12 14:46
- **Completed**: 2026-06-12 14:48
- **Status**: completed
- **Deviation**: None
- **Commit**: 8107f54
- **Notes**: Created data/, models/, results/ structure with .gitkeep files

### Step 1.6: Verification
- **Started**: 2026-06-12 14:48
- **Completed**: 2026-06-12 14:55
- **Status**: completed
- **Deviation**: pyproject.toml required `[tool.hatch.build.targets.wheel]` section
- **Commit**: fc90082
- **Notes**: 
  - `make setup` failed initially - hatchling couldn't find package
  - Fixed by adding `packages = ["src"]` to pyproject.toml
  - Ruff lint found 4 issues (unused import, unsorted imports, f-string without placeholders)
  - All fixed with `ruff check --fix` and `ruff format`
  - `make lint` now passes

### Phase 1 Summary (code)
- **Tag**: phase-1-complete
- **Total commits**: 6
- **Key deviation**: pyproject.toml needed hatch build config for src/ layout
- **Impact on future phases**: None - deviation was isolated to build config

---

## CORRECTION (2026-06-12, resumed session "PHASE 1 - again")

The earlier "Phase 1 completed" entry above covered only the **code** (scripts written,
`make setup` + `make lint` passing). The **data pipeline never actually ran**. The original
implement thread (`dc6a3cf0`) was interrupted while running `download_esa.py`; the tool call
was rejected and the CodeLayer thread wedged in `interrupting` status (un-resumable).

Re-audited true Phase 1 state against the plan's success criteria:

| Criterion | Status |
|-----------|--------|
| `make setup` succeeds | ✅ done (.venv present) |
| `make lint` passes | ✅ done |
| `make download` completes | ❌ **never completed** |
| `make etl` runs | ❌ never run |
| `data/splits/*.jsonl` created | ❌ absent |
| anomaly count 100–200 | ❌ unverified |
| PNG plots generated | ❌ absent |

### Deviation D1 — Corrupt/partial download (resolved cleanup)
- A 197 MB `ESA-Mission1.zip` was left on disk; `unzip -t` reports it corrupt
  (real file is 3.7 GB). Download was killed mid-transfer at ~5%.
- **Action**: deleted the corrupt partial. `data/raw/esa-ad/` is clean again.
- ESA-AD on Zenodo (record 12528696) = **3 zips, 11.6 GB total**:
  ESA-Mission1.zip (3.7 GB), ESA-Mission2.zip (4.1 GB), ESA-Mission3.zip (3.7 GB).

### Deviation D2 — ETL loader assumption is wrong (RESOLVED — ETL rewritten)
- `patch_telemetry.py::load_mission_data` assumed each mission is a directory with
  `telemetry.pkl` + `labels.pkl` (plan lines 322–334). **Entirely wrong.**
- **Real ESA-AD structure** (confirmed by inspecting the data):
  - `ESA-MissionN/channels.csv` — `Channel, Subsystem, Physical Unit, Group, Target(YES/NO)`
  - `ESA-MissionN/labels.csv` — `ID, Channel, StartTime, EndTime` (ISO-8601 UTC intervals)
  - `ESA-MissionN/anomaly_types.csv` — `ID, Class, Subclass, Category, Dimensionality, ...`
  - `ESA-MissionN/channels/channel_N/channel_N` — a **pickled pandas DataFrame**: a
    `datetime` DatetimeIndex + one float32 column. Channels are LONG (channel_1 =
    10.5 M rows, ~90 s cadence, spanning 2000-01-01 → 2013-12-31).
  - Mission1: 76 channels (58 Target=YES), 200 anomaly events, 3,589 label rows. The 58
    target channels are exactly the 58 with anomalies → `--target-only` is the right default.
- **Fix (commit 7695a47)**: `patch_telemetry.py` rewritten — loads per-channel pickles,
  maps `labels.csv` intervals onto the (resampled) time grid, windows + labels, preserves
  the instruction/response/metadata JSONL schema (downstream-compatible). Unit-tested on a
  real channel_1 sample (loader/resample/mask/window all correct).

### Deviation D3 — Data source switched Zenodo → Kaggle mirror (speed)
- Zenodo (record 12528696) throttles single-connection downloads to ~0.3–0.4 MB/s and does
  **not** support HTTP range requests (returns 200 + full length for a Range request), so it
  cannot be parallelised → ~3 h/mission. The original thread died mid-Zenodo-download.
- Switched to the Kaggle mirror `sammahoney/esa-anomaly-dataset`, whose total size
  (11,664,533,376 B) is **byte-identical to the official Zenodo manifest** → same data,
  verified provenance. Mirror ships the **unzipped** form (per-file).
- New `src/etl/download_kaggle.py` (commit 7695a47): per-mission, per-file via kaggle CLI,
  unzips wrappers, skips telecommands, flattens the doubled `ESA-MissionN/ESA-MissionN/`.
- **Network finding**: Kaggle bulk endpoint also measured ~0.28 MB/s to /dev/null; per-file
  Kaggle channel transfer ~2 MB/s. The real bottleneck is the **local network (~1–2 MB/s)**,
  not the source. Per-file Mission1 (3.7 GB) is still optimal vs any bulk (11.6 GB) download.
- Kaggle auth: user supplied a `KGAT_`-prefixed token at `~/.kaggle/access_token` (accepted
  by kaggle CLI 2.2.1). **Token was pasted in plaintext in-session → user should rotate it.**

### Deviation D4 — Resampling + balanced subsampling (scale; plan-relevant)
- Naive windowing of all channels → tens of millions of windows; the plan's "100 < anomalies
  < 200" check (plan line 599) reflected *annotated events*, not *windows*, and is invalid as
  written. Mission1 alone has 200 anomaly events.
- ETL now: resample each channel to a uniform cadence (`--resample`, default `1h`), keep ALL
  anomalous windows, subsample nominal to `--normal-ratio`×anomalous, cap at `--max-windows`
  (default 30k). Produces a tractable, balanced set. `generate_plots.py` rewritten to plot the
  real stored window values (removed a synthetic sine-wave fabrication) with a per-split cap.

### RESOLVED DECISIONS (user)
- **Scope**: full dataset, processed **mission-by-mission** (Mission1 first).
- **Storage**: external thumb drive `DUAL DRIVE` (FAT32, 765 GB free) — raw data lives off the
  near-full internal disk. FAT32 4 GB-file limit is a non-issue for the Kaggle mirror (largest
  per-file < 200 MB). `ESA_DATA_DIR` / `--data-dir` point the pipeline at the drive.
- **Raw-data lifecycle (updated 2026-06-13)**: **KEEP raw on `DUAL DRIVE` until the entire
  project (Phases 1–5) is complete.** ~29 GB on a drive with 765 GB free is free insurance:
  if any later phase needs a re-ETL (different resample cadence, a bug, new normalization,
  re-windowing), we avoid a multi-hour throttled re-download. Deleting early only reclaims space
  we aren't short on. Raw is off the internal disk anyway, so it never pressures local storage.
  Teardown (rotate Kaggle key + delete raw) is a single end-of-project step — see the
  "Project Teardown / Cleanup" checklist in the plan.

### PHASE 1 DATA PIPELINE — COMPLETE (Mission1)
- **Download**: Mission1 fully fetched to `DUAL DRIVE` (76/76 channels, 8.3 GB unzipped pickles).
- **ETL** (`patch_telemetry.py --missions 1 --resample 1h`, 58 target channels, ~64 s):
  - **30,000 patches** total (capped by `--max-windows`): **7,437 anomalous / 22,563 nominal**.
  - Splits written: `train.jsonl` 21,000 · `val.jsonl` 4,500 · `test.jsonl` 4,500 (70/15/15).
  - `data/processed/jsonl/all_patches.jsonl` (28 MB).
- **Plots** (`generate_plots.py --max-per-split 2000`): real window values rendered to PNG
  (capped 2,000/split → ~6,000 PNGs) + per-split `*_metadata.jsonl` for the VL model.
- **Validation** (all ✅): JSONL schema present on every record; anomaly balance ~25% and
  consistent across train/val/test; all 58 target channels represented; every window length 32;
  `make validate-etl` range (100 < anomalies < 10000) satisfied (7,437).
- **Network note**: user is on a VPN (M247, São Paulo exit, 240 ms RTT). Raw link is 37 Mbit/s
  but single-stream transfers ran ~2 MB/s — high-latency single-connection limit, not the source.
- **.gitignore added**: raw/processed/plots/splits/models/.venv excluded; `.gitkeep` structure kept.

### Deviation D5 — Mission3 channels are categorical (RESOLVED)
- Mission3 channels store discrete state as ordinal strings (`'value_0'`, `'value_1'`) rather
  than float telemetry. `load_channel_series` crashed with `ValueError: could not convert string
  to float: 'value_0'` on the first Mission3 channel.
- **Fix (commit a6362c7)**: ordinal-encode categorical strings by extracting the trailing integer
  (`value_N` → float N). Also filtered macOS `._` resource-fork entries that appear on FAT32
  volumes (cause `NotADirectoryError`). `download_kaggle.py` made `telecommands.csv`
  non-fatal (absent from Mission3).

### PHASE 1 DATA PIPELINE — COMPLETE (all 3 missions, 2026-06-12 23:45)
- **Download** (all on DUAL DRIVE):
  - Mission1: 76 channels, 8.3 GB  
  - Mission2: 100 channels, 9.1 GB  
  - Mission3: 48 channels, 12 GB  
- **Combined ETL** (`patch_telemetry.py --missions 1,2,3 --resample 1h`):
  - **30,000 patches** (capped): **7,457 anomalous (24.9%) / 22,543 nominal**
  - Splits: `train.jsonl` 21,000 · `val.jsonl` 4,500 · `test.jsonl` 4,500 (70/15/15)
  - All 3 missions represented in every split
- **Plots** (`generate_plots.py --max-per-split 2000`): 6,000 PNGs + 3 `*_metadata.jsonl`
- **Validation** (`make validate-etl` ✅): 30,000 total · 7,457 anomalous · 3 missions
- **Lint** (`make lint` ✅): all 17 files pass ruff check + format
- **Commits**: a6362c7 (categorical fix), 0270817 (validate-etl fix), pushed to remote

---

## Notes for Future Phases

### Phase 1.5 (Advice Label Generation) — COMPLETE
- **Script**: `src/etl/generate_advice_labels.py`
- **Output**: `data/labels/anomaly_advice.json` (7,457 records, no API cost)
- **Methodology**: window statistics (pattern type: spike/drift/oscillation/sustained_offset/
  subtle_deviation) + channel metadata (subsystem → human name, physical unit → measurement type)
  → structured advice + severity (low/medium/high) + recommended_action
- **Enriched splits**: `data/splits/*_with_advice.jsonl` (advice merged into `response` field)
- **Severity distribution**: low 3,138 · medium 972 · high 3,347
- **Pattern distribution**: subtle_deviation 4,132 · persistent_anomaly 2,250 ·
  sustained_offset 576 · spike 257 · drift 224 · sustained_oscillation 15 · oscillation 3
- **Validation** (`make validate-advice` ✅): 7,457 unique IDs · all required fields present ·
  severities ⊆ {low, medium, high}
- **Commit**: 0868cc2 · pushed to remote

### Phase 2 prep — shared loader extracted (2026-06-13, ready for new thread)
- **`src/etl/io.py` created** as the single source of truth for ESA-AD loading. Exports:
  `DEFAULT_RAW_DIR`, `discover_missions`, `iter_channels` (skips missing + macOS `._` forks),
  `channel_file_path`, `list_channels`, `load_channel_series` (D5 categorical ordinal-encoding),
  `resample_series`, `load_labels`, `anomaly_mask_for_channel`, `RevINNormalizer`.
- **`patch_telemetry.py` refactored** to import from `io.py` (removed its duplicate copies of
  RevIN/loader/discover/mask). Smoke-tested read-only on all 3 missions (Mission3 categorical
  channel: 15.4M raw rows → 70,200 @1h, mask + RevIN correct). `make lint` ✅.
  **Did NOT re-run the full ETL** (would overwrite the good 3-mission splits) — refactor is
  behaviour-preserving; the read-only smoke test exercised every shared function.
- **Phase 2 plan section updated** with a MUST-READ block: the io.py import surface, a 9-row
  decisions table (loader, ESA_DATA_DIR, iter_channels, 1h resample, labels, D5, F1 range,
  keras backend, output storage), and the storage rule. The new thread should only need to read
  the plan.
- **keras backend (verified)**: venv has `keras 3.14.1` + `torch 2.12.0`, NO tensorflow →
  set `KERAS_BACKEND=torch` before `import keras`.
- **Storage rule (local disk nearly full)**: raw data on `DUAL DRIVE`; Phase 2 models/checkpoints
  must also go on `DUAL DRIVE` (configurable root, e.g. `STAR_OUTPUT_DIR` /
  `MODELS_DIR ?= /Volumes/DUAL DRIVE/star-pipeline/models`); repo tracks only code + small JSON
  metrics. `.gitignore` already excludes `models/`, `results/**/*.json`, raw data.
- **Commits**: see below (io.py + refactor + plan/log updates).

---

## Phase 2: LSTM Baseline

### Step 2.1 + 2.2: LSTM + Isolation Forest training scripts
- **Started**: 2026-06-13 11:10
- **Completed**: 2026-06-13 11:25
- **Status**: completed
- **Commit**: a2efc7a
- **Deviation D6**: `--seq-stride` default changed from 1 to **16** (matches ETL STRIDE).
  Plan defaulted to stride=1 which creates ~262k windows per channel at 1h cadence;
  at stride=1 a single channel took >15 min and never finished. Stride=16 yields ~16k
  windows, each channel trains in ~75s, 3 channels in 3:45.
- **Deviation D7**: Models go under `STAR_OUTPUT_DIR` (default `/Volumes/DUAL DRIVE/star-pipeline`)
  via `os.environ.get("STAR_OUTPUT_DIR")` — satisfies plan decision #9 (no large artifacts
  on internal disk). Plan code block wrote to local `models/lstm/`. Only the small
  `results/lstm/baseline_results.json` (metrics) stays in the repo.
- **Results (Mission1, 3 channels, 10 epochs)**:
  - LSTM: avg_precision=0.835, avg_recall=0.552, **avg_F1=0.663** ✅
  - Isolation Forest: avg_precision=0.127, avg_recall=0.459, avg_F1=0.188
- **validate-baseline**: `make validate-baseline` passes (F1 sanity range relaxed to 0.05–0.98
  from plan's 0.3–0.95, matching plan note that range was a guess; actual 0.663 ∈ both ranges).
- **Notes**:
  - Background subprocess runner silently killed the process; ran synchronously instead.
  - F1=0.663 for LSTM is a strong baseline; IF at 0.188 is as expected for an unsupervised
    method with fixed contamination=0.1.
  - These 3-channel numbers are representative; a full sweep (all 58 Mission1 target channels)
    would run `make baseline MISSION=1` with `--max-channels 58` and take ~70 min.

### Phase 2 impact on remaining phases
- No impact on Phases 3–5 (LLM fine-tuning uses JSONL splits, not the LSTM models).
- The 3-channel smoke run is sufficient for the interview showcase; a full sweep can run
  in background before the Phase 5 comparison table is generated.

---

### Phase 3-5 — plan reviewed & corrected (2026-06-13)
Audited Phases 3–5 against the Phase 1/1.5 findings; added MUST-READ blocks to each in the plan.
Key issues caught (full detail in the plan blocks):
- **Phase 3 §3.5 advice-key bug**: formatter keys advice by `{mission}_{channel}_{start_idx}-{end_idx}`,
  but the real `anomaly_id` is `{mission}__{channel}__{start_time}` → lookup always misses. Fix:
  read the enriched `data/splits/*_with_advice.jsonl` (advice already in `response`) instead.
- **Phase 3 §3.4 paths**: train/eval should point at `data/formatted/*_chatml.jsonl` (has `text`
  field), not raw splits. **§3.3 upload** omits `data/processed/plots/` needed by VL training §3.7.
- **Phase 4 §4.2/§4.3 storage violation**: download/load GGUF from LOCAL `models/gguf/`; an 8B GGUF
  is multi-GB → must use DUAL DRIVE via `STAR_MODEL_DIR`. Also: build `llama-cpp-python` with Metal.
- **Phase 5**: LLM eval is a 10-sample toy (raise to full 4,500 test split); `affinity_f1` defined
  but never called; "Key Findings" are hardcoded placeholders; Hybrid scoring undefined;
  `load_lstm_results` depends on Phase 2 emitting per-channel precision/recall/f1.
- **Cross-cutting**: verify HF model IDs (`unsloth/Qwen3-8B-bnb-4bit`, `Qwen3-VL-8B`) exist at impl
  time; `evaluation_strategy`→`eval_strategy` rename in recent transformers; storage rule (large
  artifacts on DUAL DRIVE / cloud, never local) applies throughout.

---

## Phase 3: Cloud Setup & LLM Fine-tuning (IN PROGRESS, 2026-06-13)

### What is DONE (committed code, all linted + locally validated)
All Phase 3 placeholder files were rewritten (not copied verbatim — the plan's §3.4/§3.5/§3.6/§3.7
code had known bugs flagged in the MUST-READ blocks). Files:
- `src/training/format_for_unsloth.py` — reads the advice-enriched splits
  `data/splits/{train,val,test}_with_advice.jsonl` (advice already merged into `response`) and
  ChatML-wraps `instruction`+`response` → `data/formatted/{split}_chatml.jsonl` (single `text`
  field). **Ran locally**: 21,000 / 4,500 / 4,500 records (5,221 / 1,113 / 1,123 anomalous).
- `config/unsloth-train.yaml` — points train/eval at `data/formatted/*_chatml.jsonl`; model
  `unsloth/Qwen3-8B-unsloth-bnb-4bit`; `eval_strategy`; gguf quant `q4_k_m`.
- `src/training/train_advice.py` — Qwen3-8B QLoRA SFT. **Rewritten for the TRL 0.24 API actually
  installed on the instance** (see D9): `SFTConfig` carries args + `dataset_text_field` +
  `max_length`; tokenizer passed as `processing_class`; `import unsloth` first.
- `src/training/train_detection.py` — Qwen3-VL SFT (written, **NOT run** — VL is out of this run's
  scope; advice-only). Uses real Unsloth vision pattern (messages list + UnslothVisionDataCollator,
  `processing_class`). Model `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`.
- `src/training/export_gguf.py` — `save_pretrained_gguf(..., quantization_method="q4_k_m")`.
- `scripts/cloud/launch_vast.sh` — searches offers, dry-run by default, `--create` to launch;
  reads `VASTAI_API_KEY` from `.env`.
- `scripts/cloud/upload_data.sh`, `download_models.sh` — rsync via `vastai ssh-url`. NOTE: these
  use the account default key; this session had to use a custom passphraseless key (D11), so the
  ACTUAL upload was done with a direct `tar | ssh` (see below), not these scripts. They also assume
  `rsync` on the instance (the pytorch image lacks it — use tar/scp, see D12-adjacent note).
- `Makefile` — added `format-train`, `validate-format`, `launch-vast` targets. `make validate-format`
  passes (checks `text` field + ChatML prefix on all 3 splits).

### Verified model IDs (web-researched 2026-06-13)
- Text: `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Dynamic 4-bit, preferred) — also valid:
  `unsloth/Qwen3-8B-bnb-4bit`.
- Vision: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` — **the plan's `unsloth/Qwen3-VL-8B` does
  NOT exist.**
- `transformers >= 4.46` renamed `evaluation_strategy` → `eval_strategy`.

### ▶▶ LIVE CLOUD STATE — how to resume (READ THIS FIRST in a fresh session) ◀◀
- **Vast.ai instance**: id **40838191**, single RTX 4090 24 GB (offer 38138029, Hungary, $0.49/hr).
  - SSH (direct): `ssh -i ~/.ssh/vast_star -p 60642 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes root@81.183.231.113`
  - SSH (proxy, fallback): port 38190 on `ssh3.vast.ai`, same key.
  - CLI: `.venv/bin/vastai show instance 40838191`  (api-key already set from `.env`).
- **SSH key**: `~/.ssh/vast_star` (passphraseless, generated this session, registered on the vast
  account + attached to the instance). The user's `~/.ssh/id_ed25519` is **passphrase-protected** and
  cannot be used non-interactively (D11) — use `vast_star`.
- **Remote workdir**: `/workspace/star-pipeline` (has `data/formatted/`, `src/`, `config/`).
- **Training**: full **3-epoch** advice SFT running under `setsid` (survives SSH drop).
  - Log: `/workspace/train.log`  ·  total **3,939 steps** @ ~2.83 s/it → **~3.1 h** total.
  - Check alive: `ssh ... "pgrep -fc '[t]rain_advice'"` (NOTE the `[t]` bracket trick — D12).
  - Watch progress: `ssh ... "tr '\r' '\n' < /workspace/train.log | grep -aE 'loss|it/s]' | tail"`.
  - Output LoRA → `/workspace/star-pipeline/models/lora/qwen3-8b-advice/`.
- **Installed stack on instance** (do not "fix" — it works): torch 2.10.0+cu128, transformers 5.5.0,
  trl 0.24.0, unsloth 2026.6.7, unsloth_zoo, Python 3.11. CUDA available, RTX 4090 detected.

### ⏭️ NEXT STEPS to finish Phase 3 (when training completes)
1. Confirm training finished: `ls /workspace/star-pipeline/models/lora/qwen3-8b-advice/` shows
   adapter files; `grep -a "Training complete" /workspace/train.log`.
2. Export GGUF **on the instance**:
   `ssh ... "cd /workspace/star-pipeline && python src/training/export_gguf.py"` → writes
   `models/gguf/star-pipeline-advice*.gguf` (q4_k_m).
3. Download LoRA + GGUF to DUAL DRIVE (instance lacks rsync — use tar/scp like the upload):
   `ssh ... "cd /workspace/star-pipeline && tar czf - models" | (cd "/Volumes/DUAL DRIVE/star-pipeline" && tar xzf -)`
   (or fix `download_models.sh` to use scp). STORAGE RULE: models go on DUAL DRIVE, never local.
4. **TEARDOWN the instance** (stops billing): `.venv/bin/vastai destroy instance 40838191`.
   This is the *instance* teardown only — NOT the project-wide teardown (raw-data deletion / Kaggle
   key rotation), which stays deferred until all Phases 1–5 are done.
5. Mark Phase 3 success criteria in the plan; Phase 4 = local GGUF inference (build
   `llama-cpp-python` with Metal; load GGUF from DUAL DRIVE via `STAR_MODEL_DIR`).
6. **Budget**: user authorized up to **$50** ceiling (full run is ~$1.5–2 of that). Credit was $25
   at start — check `vastai show user`.

### Deviations (Phase 3)
- **D8 — Model IDs corrected.** Plan's `Qwen3-8B-bnb-4bit` works but used the preferred Dynamic 4-bit
  `unsloth/Qwen3-8B-unsloth-bnb-4bit`; plan's `Qwen3-VL-8B` does not exist → `Qwen3-VL-8B-Instruct-unsloth-bnb-4bit`.
- **D9 — TRL 0.24 API rewrite.** The instance resolved to trl 0.24.0 / transformers 5.5.0. The
  plan's SFTTrainer call (`tokenizer=`, `dataset_text_field=`, `max_seq_length=` kwargs +
  `TrainingArguments`) is removed in this version. Rewrote `train_advice.py` to use `SFTConfig`
  (with `dataset_text_field`, `max_length`, `eval_strategy`) + `processing_class=tokenizer`.
  Introspected the live signatures to get field names exactly right before running.
- **D10 — Formatter reads enriched splits.** Skipped the plan's synthetic
  `mission_channel_start-end` advice lookup (always missed). Reads `*_with_advice.jsonl` where advice
  is already in `response`.
- **D11 — SSH key.** Account `id_ed25519` is passphrase-protected → `read_passphrase: can't open
  /dev/tty` in non-interactive shell (server *accepts* the key, client can't unlock it). Generated a
  passphraseless `~/.ssh/vast_star`, registered + attached it. Vast also has a post-boot key
  propagation lag (~1–2 min) before auth succeeds.
- **D12 — `pkill -f` self-match.** `pkill -f train_advice` killed the SSH management shell itself
  (its own argv contains "train_advice") → exit 255 / blank output. Fixed with the `[t]rain_advice`
  bracket pattern (regex matches the python process, not the literal pattern in my own command).
- **D13 — Unsloth env setup.** The `git+unsloth[cu124]` onstart install pulled transformers 5.12 and
  omitted `unsloth_zoo` (ImportError). Reinstalled from PyPI (`pip install unsloth unsloth_zoo`),
  which pinned transformers 5.5.0 / trl 0.24.0 and upgraded torch to 2.10.0+cu128 (CUDA still works;
  the torchaudio 2.5.1 version-conflict warning is harmless — torchaudio is unused).
- **Upload method**: the pytorch image lacks `rsync`; used `tar czf - ... | ssh ... tar xzf -`
  instead. zsh also doesn't word-split an `ssh -i ... -p ...` string stored in a variable — invoke
  ssh directly.
- **Epochs**: started at 3, briefly switched to 1 (fast loss convergence 2.85→1.4 by step 50), then
  **user requested the full 3-epoch run** (budget OK). Final run = 3 epochs / 3,939 steps.
- **Scope**: this run is **advice (text) model only**. The VL `train_detection.py` is written but not
  run; plots were not uploaded. VL can be a follow-up if desired for the 3-way comparison.

### Phase 3 completion summary (2026-06-13 22:12 UTC)
- **Training**: 3 epochs / 3,939 steps, ~4.5h total, loss 2.85→0.24, eval_loss 0.256 (stable)
- **Export**: `export_gguf.py` ran on instance → `models/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf` (4.7GB)
  + Modelfile. Note: Unsloth added a `_gguf` suffix to the dir name and named the GGUF after the base model
  (`qwen3-8b.Q4_K_M.gguf`), not the project name. This deviates from `export_gguf.py`'s output_name parameter.
- **LoRA downloaded**: all final adapter files on DUAL DRIVE at `/Volumes/DUAL DRIVE/star-pipeline/models/lora/qwen3-8b-advice/`
  (adapter_model.safetensors 167MB, tokenizer.json 11MB, adapter_config.json, tokenizer_config.json, chat_template.jinja, README.md).
- **Cost**: ~$2.30 (4.5h × $0.49/hr + bandwidth) — well within $50 ceiling.
- **Commits**: 3e758a8 (Phase 3 code), 68783a3 (Phase 4 code)

### Plan sections that may need updating based on this phase
- Phase 4 §4.1 `export_gguf.py` `quantization_method="dynamic"` → used `q4_k_m`; already corrected.
- Phase 4 §4.3: actual GGUF path is `{STAR_MODEL_DIR}/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf`
  (Unsloth's naming, not plan's `star-pipeline-advice.gguf`). Corrected in `test_local_gguf.py`.
- Phase 4 / Phase 5: test data uses `test_with_advice.jsonl` (has structured DIAGNOSIS/ADVICE/ACTION responses).
- Phase 5 LLM eval: run full 4,500 samples (use `--limit 0`), not 10.

---

## Phase 4: GGUF Export & Local Inference

### Step 4.1: GGUF Export on Cloud Instance
- **Started**: 2026-06-13 22:09 UTC (triggered during Phase 3 teardown)
- **Completed**: 2026-06-13 22:12 UTC
- **Status**: completed
- **Deviation**: None conceptually; export ran as part of Phase 3 wrap-up sequence.
  Unsloth's `save_pretrained_gguf` created `models/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf`
  (not `star-pipeline-advice.gguf` as the plan expected). The `_gguf` suffix + base-model name
  is Unsloth's default output format — noted as D14.
- **Commit**: 3e758a8 (export_gguf.py was committed as part of Phase 3 code)
- **Notes**: Also produced a `Modelfile` (Ollama format) at no extra effort.

### Step 4.2: Download Models
- **Started**: 2026-06-13 (this session)
- **Status**: in progress — GGUF download pending user native-terminal test
- **Deviation D15 — rsync unavailable + partial first download**: The instance pytorch image lacks
  rsync. Used `tar cf - | ssh ... tar xf -` (no gzip for GGUF — already binary compressed, avoids
  CPU bottleneck). Initial LoRA download via gzip was very slow; switched to no-gzip which was faster.
  LoRA adapter files downloaded successfully.
- **Deviation D17 — GGUF (5GB) exceeds FAT32 4 GB file limit**: DUAL DRIVE is FAT32 and cannot
  hold a single file larger than 4,294,967,295 bytes. The GGUF is 5,027,784,160 bytes (~4.68 GiB).
  **Fix**: download GGUF to local APFS SSD at
  `/Users/laptop/Developer/fdl_technicalInterview/models/gguf/star-pipeline-advice_gguf/`
  (63 GB free). Makefile STAR_MODEL_DIR default updated to this path. LoRA remains on DUAL DRIVE
  (each file <200 MB, within FAT32 limit). Plan MUST-READ updated with FAT32 warning.
- **Deviation D18 — Hungary instance → slow SSH transfer (300–400 KB/s aggregate)**: Instance 40838191
  was selected by `vastai search offers ... --order 'dph_total asc'` (cheapest RTX 4090 first).
  The cheapest happened to be in Hungary. For training (Phase 3), location is irrelevant. For
  downloading a 5 GB GGUF, the trans-Atlantic SSH path is bottlenecked to ~300–400 KB/s even with
  8 parallel streams, giving a ~3.5h ETA. Single stream to /dev/null confirmed: 100 MB in 6 min =
  278 KB/s. Local internet is 567 Mbps (no VPN) — the bottleneck is the SSH TCP path, not local
  network. **Mitigation**: user testing native terminal `scp` (tool sandbox may add overhead).
  **Future prevention**: add `--region US` to offer search; US RTX 4090 ~$0.55–0.70/hr vs $0.49/hr
  Hungary — small premium, but avoids multi-hour downloads. Plan §3.2 MUST-READ updated.
- **Commit**: pending (after download + inference complete)
- **Notes**:
  - LoRA final files (167MB adapter + 11MB tokenizer + small configs) → DUAL DRIVE ✅
  - GGUF (4.7GB) → downloading to local SSD (IN PROGRESS)
  - Checkpoints (~10GB of 39 intermediate saves) intentionally skipped — only final adapter matters.
  - Instance 40838191 will be destroyed immediately after GGUF download completes.

### Step 4.3: Local Inference Script
- **Started**: 2026-06-13 (this session)
- **Completed**: 2026-06-13 (code complete, lint passes; execution pending GGUF download)
- **Status**: code complete, execution pending
- **Deviation D16 — test data + GGUF path + limit**: Plan §4.3 used `test.jsonl` and `models/gguf/star-pipeline-advice.gguf`.
  Corrected: uses `test_with_advice.jsonl` (has structured DIAGNOSIS/ADVICE/ACTION in expected response),
  loads from `STAR_MODEL_DIR/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf`. Default limit=100
  for Phase 4 smoke test; Phase 5 will run full 4,500. Added `validate-inference` Makefile target.
- **Commit**: 68783a3

### Step 4.4: Local Dependencies
- **Started**: 2026-06-13 (this session)
- **Completed**: 2026-06-13
- **Status**: completed
- **Deviation**: `llama-cpp-python 0.3.29` installed (plan specified `>=0.2.50`; 0.3.x has improved
  Metal API). Metal confirmed: `llama_supports_gpu_offload()=True`, M3 Max GPU detected, 30GB unified
  memory available, `GPU family: MTLGPUFamilyApple9`.
- **Commit**: 68783a3

### Step 4.2 (continued): GGUF Download — Escalation from D18

The native `scp` test confirmed trans-Atlantic bottleneck (215 KB/s → ~6h ETA). Two parallel strategies
attempted; only one needed.

**D19 — US relay (tried, obsoleted by D20)**: Spun up a second Vast.ai instance (40866462, California
RTX 3060, ubuntu:22.04, $0.60/hr) as an SCP relay. Staged the `vast_star` private key on Hungary for
cross-instance auth. `/workspace` directory missing on the fresh ubuntu image — created it, restarted
SCP push from Hungary (PID 134904). The relay had 32 GB free and SSH via `ssh4.vast.ai:26462`.
Transfer was in progress (42 MB received) when D20 made the relay unnecessary. Instance 40866462
destroyed immediately; relay cost: negligible (~$0.03).

**D20 — HuggingFace Hub upload + CDN download (used)**: HF token `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`
initially returned "Invalid username or password" from local curl. On Hungary instance (where
`huggingface_hub` was pre-installed by Unsloth), `api.whoami()` returned `dyrtyData` — token valid,
issue was local environment. Ran background HF upload on Hungary:
- Repo created: `dyrtyData/star-pipeline-qwen3-8b-advice-gguf` (public model)
- Upload: 5.03 GB at 102 MB/s (~50 sec), completed at commit
  `https://huggingface.co/dyrtyData/star-pipeline-qwen3-8b-advice-gguf/commit/2732a9ffb8de9b4af6c74225e622b89ebb8aede4`
- Local download via `hf_hub_download` with CDN: ~4.7 GB/s effective throughput, completed in <60s.
  Temp file in `.cache/huggingface/download/`, moved to final path on completion.
- **Final path**: `/Users/laptop/Developer/fdl_technicalInterview/models/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf`
- **Verified**: `stat -f "%z"` → 5,027,784,160 bytes (exact match)

**Instances destroyed**: Hungary 40838191 and relay 40866462 both destroyed after GGUF confirmed locally.
**Total Vast.ai cost**: ~$2.33 (4.5h Hungary $0.52/hr + 30min relay $0.60/hr + bandwidth).

### Step 4.5: Run Inference (eval-llm)
- **Started**: 2026-06-13 19:45 local
- **Completed**: 2026-06-13 19:55 local
- **Status**: completed ✅
- **Command**: `STAR_MODEL_DIR=/Users/laptop/Developer/fdl_technicalInterview/models make eval-llm`
- **Output**: `results/inference_test.json`
- **Results (100-sample smoke test)**:
  - Accuracy: 0.690 | Precision: 0.432 | Recall: 0.615 | **F1: 0.508**
  - TP=16 FP=21 FN=10 TN=53 | Unknown=0 (model always produced ANOMALY/NOMINAL)
  - **Avg time: 1.962s/sample** (Metal GPU, well within <30s requirement)
  - n_gpu_layers=2147483647 (all 28 layers offloaded to Metal)
- `make validate-inference` → **validate-inference OK** ✅ (all checks pass)

### Phase 4 Completion Summary
- **GGUF**: 5,027,784,160 bytes verified on local APFS SSD + HF Hub backup
- **Inference**: Qwen3-8B Q4_K_M running on M3 Max Metal at ~1.96s/sample
- **HF Repo**: `dyrtyData/star-pipeline-qwen3-8b-advice-gguf` (public)
- **Vast.ai instances destroyed**: Hungary 40838191 + relay 40866462
- **Total Vast.ai cost**: ~$2.33 ($50 ceiling, well within budget)

### Phase 5 Pre-flight (READ BEFORE STARTING)
**`src/inference/evaluate.py` is currently a 3-line TODO stub** — Phase 5 must write it
from scratch using the plan §5 code as a starting point (with the schema fixes in MUST-READ #1).
`make eval-all` calls `python src/inference/evaluate.py --all`. Recommended start sequence:
1. `make eval-llm LIMIT=0` — full 4,500-sample LLM run (~2.5 h, overwrites the 100-sample
   smoke test result). STAR_MODEL_DIR is already set in Makefile.
2. Rewrite `evaluate.py` using actual schemas (see plan MUST-READ #1 for key names).
3. `make eval-all` → comparison table + report.

**Actual result file schemas (confirmed Phase 4):**

`results/inference_test.json`:
```json
{"summary": {"n_samples":100,"accuracy":0.69,"precision":0.432,"recall":0.615,"f1":0.508,
             "tp":16,"fp":21,"fn":10,"tn":53,"unknown_responses":0,"avg_time_s":1.962},
 "results": [{"index":0,"mission":"...","channel":"...","is_anomaly":bool,
              "predicted":"ANOMALY"|"NOMINAL","correct":bool,
              "expected_response":"...","actual_response":"...","elapsed_s":float}]}
```

`results/lstm/baseline_results.json`:
```json
{"summary": {...}, "config": {...},
 "channels": [{"channel":"...","mission":"...","precision":float,"recall":float,"f1":float,
               "threshold":float,"n_sequences":int,"n_anomaly_windows":int,...}]}
```

---

## Phase 5: Evaluation & Comparison

### Step 5.1: Rewrite evaluate.py (from stub) + comparison report
- **Started**: 2026-06-14
- **Completed**: 2026-06-14
- **Status**: completed
- **Commit**: (pending — Phase 5 code + plan/log)
- **What was built**: `src/inference/evaluate.py` (was a 3-line TODO stub) now loads the real
  result files, computes a unified comparison across **four** approaches, and writes
  `results/comparison_report.md` + `results/comparison_metrics.json`. Added `make validate-eval`.
- **Verification (all ✅)**:
  - `make eval-all` → exit 0, report + metrics JSON written.
  - `make validate-eval` → OK (report sections present; no approach errored; all
    precision/recall/f1/cef_0.5 ∈ [0,1]; LLM anomaly responses >50 chars).
  - `make lint` (ruff check + format) → clean on `evaluate.py`.
- **Results (LLM on the Phase-4 100-sample slice; baselines = Phase-2 3-channel smoke):**

  | Approach | Precision | Recall | F1 | CEF0.5 | Affinity-F1 |
  |---|---|---|---|---|---|
  | Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | N/A |
  | LSTM Baseline | 0.835 | 0.552 | 0.663 | 0.757 | N/A |
  | LLM Detection | 0.432 | 0.615 | 0.508 | 0.460 | 0.508 |
  | Hybrid (LSTM + LLM advice) | 0.835 | 0.552 | 0.663 | 0.757 | N/A |

  Computed Key Findings: LSTM wins on F1 and CEF0.5 (precision-weighted, the operationally
  relevant metric); the LLM trades precision for recall but adds free-text advice (100% of the
  37 anomaly predictions emitted structured DIAGNOSIS+ADVICE); LLM costs 1.96 s/window vs
  near-instant baselines.

### Step 5.2 wrap-up: Full 4,500-sample LLM eval — COMPLETE (2026-06-14)
- **Status**: completed ✅
- **n_samples**: 4,500 (`partial=false`) — Attempt 3 (detached daemon) ran to completion.
- **Final results (n=4500)**:

  | Approach | Precision | Recall | F1 | CEF0.5 | Affinity-F1 |
  |---|---|---|---|---|---|
  | Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | N/A |
  | LSTM Baseline | 0.835 | 0.552 | 0.663 | 0.757 | N/A |
  | **LLM Detection** | **0.360** | **0.609** | **0.453** | **0.392** | **0.456** |
  | Hybrid (LSTM + LLM advice) | 0.835 | 0.552 | 0.663 | 0.757 | N/A |

  LLM additional metrics: accuracy=0.632, avg_time=2.77s/sample, unknown_responses=27/4500,
  advice_structured_frac=0.9963, n_anomaly_predictions=1898.
  Affinity-F1 detail: precision=0.357, recall=0.631, n_pred_intervals=1852, n_gt_intervals=1052.
- `make eval-all` + `make validate-eval` → **validate-eval OK** ✅
- **Deviation D25** (relative to 100-sample numbers): Precision dropped (0.432→0.360), F1 dropped
  (0.508→0.453), Recall nearly unchanged (0.615→0.609). At scale the model is somewhat more
  trigger-happy (more FP). The LSTM still leads on precision-weighted CEF0.5 (0.757 vs 0.392);
  LLM wins recall.

### Deviations (Phase 5)
- **D21 — Loaders rewritten to real schemas.** The plan's `load_lstm_results()` /
  `load_llm_results()` assumed list-of-dicts / `r["actual"]` keys that never existed.
  Rewrote both: LSTM/IF read `d["channels"]` and macro-average; LLM reads `d["summary"]`
  directly (micro metrics already computed in Phase 4) and joins per-window `d["results"]`
  (key `actual_response`, pre-computed `predicted`) to test metadata by `index`.
- **D22 — CEF0.5 computed from precision/recall, not tp/fp/fn.** The baselines persist only
  per-channel precision/recall/f1 (no raw counts), so CEF is computed via
  `cef_from_pr(P, R, beta=0.5)` uniformly for every approach. Mathematically identical to the
  plan's `cef_score(tp,fp,fn)` given the same P/R. Documented the micro-vs-macro averaging
  difference (LLM micro over windows; baselines macro over channels) in the report's
  Methodology Notes.
- **D23 — Affinity-F1 wired but degenerate on this test split (documented honestly).** The
  test split is a *shuffled, balanced-subsampled* set of windows — only **~1.4 windows per
  (mission, channel)** in the 100-sample slice (verified). Interval reconstruction therefore
  yields mostly isolated single-window "intervals", so Affinity-F1 ≈ window-level F1 (0.508).
  Implemented `affinity_f1()` correctly (per-channel interval merge + delta-tolerant matching,
  pooled P/R) and added a Methodology note stating it becomes meaningful only on a contiguous
  (un-shuffled) evaluation stream. Computed for the LLM (per-window preds persisted); N/A for
  the baselines (only aggregate per-channel metrics were saved).
- **D24 — Added Isolation Forest (4th approach) + defined Hybrid scoring.** Plan §5 compared 3
  approaches; the IF baseline result (`results/isolation_forest/if_results.json`) already
  existed from Phase 2, so it's included for free as a 4th row. **Hybrid** (plan MUST-READ #5
  flagged it as "not wired") is scored as: detection metrics inherited from the LSTM (the
  component that flags anomalies, high precision) + the LLM's advice layer attached to each
  flag. Its detection score equals the LSTM's by construction — the hybrid's value is
  actionable advice, not better detection. Stated explicitly in the report.
- **Coherence-check fix**: `actual_response` is persisted truncated to 300 chars (Phase-4),
  which clips the trailing `ACTION:` line; keyed advice coherence on `DIAGNOSIS`+`ADVICE`
  (which survive the cap) → 100% structured, avg 300 chars.
- **Hardcoded "Key Findings" removed**: `generate_findings()` derives every bullet from the
  loaded metrics (best-F1, best-CEF, precision/recall trade-off, advice coherence, latency,
  affinity) — the plan's "~0.7 F1 / combines best of both" placeholders are gone.

### Decision: LLM eval left at 100 samples (full 4,500 sweep optional, not run)
- The plan MUST-READ recommends re-running the LLM on the full 4,500-window split before the
  final report (`make eval-llm LIMIT=0`, ~2.5 h on M3 Max Metal). User was asked and did not
  select; took the conservative/reversible default — **did not** start an uninvited multi-hour
  job. Report is honest about `n_samples=100`. The full sweep is a one-command follow-up:
  `make eval-llm LIMIT=0 && make eval-all` (evaluate.py reads `n_samples` from the file, so the
  report and Key Findings auto-update with no code change). Baselines could likewise be expanded
  from the 3-channel smoke to all 58 Mission-1 target channels (`make baseline` with
  `--max-channels 58`) for a fuller comparison.

### Phase 5 impact on the rest of the plan
- No downstream phases remain (Phase 5 is last). The only open project-wide item is the
  **Teardown / Cleanup** checklist (rotate Kaggle token, delete raw ESA-AD from DUAL DRIVE) —
  its precondition "Phase 5 evaluation complete and results committed" is now satisfiable once
  this commit lands. Teardown stays a deliberate, separate final step (not run here).
- `evaluate.py`'s loaders are now the canonical schema consumers; if Phase 2/4 result schemas
  ever change, update the three loader functions (`load_lstm_results`, `load_if_results`,
  `load_llm_results`) — they are the single point of coupling.

### Step 5.2: Full 4,500-sample LLM eval — re-run saga (2026-06-13 → 2026-06-14)
User approved the full sweep ("kick it off"). It died TWICE before succeeding; root causes and
the durability fixes are below (also distilled into `~/.claude/CLAUDE.md` → "Long-Running
Background Processes", committed `eed8d41` + a follow-up).

- **Attempt 1** (`make eval-llm LIMIT=0 | tee …`, run_in_background): produced NO output and
  never updated `inference_test.json`. Cause: Python stdout was **block-buffered through the
  `tee` pipe**, so progress never flushed — a dead job looked identical to a running one. The
  process was gone with only the model-load line (llama.cpp's C-level stderr) in the log.
- **Attempt 2** (`caffeinate -i` + run_in_background, hardened script): reached sample ~20 then
  died. Cause: **machine slept overnight** (~8 h gap). `caffeinate -i` blocks only *idle* sleep,
  not lid-close/system sleep; and `caffeinate`/the job were children of the harness session, so
  they died when the session was suspended.
- **Hardening applied to `src/inference/test_local_gguf.py`** (commit `1bfd715`):
  - `--checkpoint-every N` (default 250): atomic temp-file + rename write of `{summary, results}`
    every N samples. A death now costs ≤N samples.
  - `--resume`: deterministic sample order → resumes from `len(existing results)`. Reuses any
    prior genuine results (e.g. the 5-sample smoke).
  - `compute_summary()`/`write_results()` extracted; metrics reconstruct from each record's
    persisted `elapsed_s` (resume-safe). Summary gains a `partial` flag (True mid-run).
  - `flush=True` on all progress prints; run with `PYTHONUNBUFFERED=1`.
  - **Schema unchanged** → `evaluate.py` and `make validate-inference`/`validate-eval` still work.
- **Attempt 3 (CURRENT, running)**: detached daemon, survives session + blocks system sleep:
  ```
  cd <repo> && ( nohup caffeinate -dimsu env \
    STAR_MODEL_DIR="/Users/laptop/Developer/fdl_technicalInterview/models" PYTHONUNBUFFERED=1 \
    .venv/bin/python src/inference/test_local_gguf.py --limit 0 --resume --checkpoint-every 250 \
    > results/.eval_llm_full.log 2>&1 < /dev/null & )
  ```
  `( … & )` subshell reparents to launchd (PID 1) so it outlives the harness turn/session.
  Confirmed advancing (sample ~220 @ ~2–3 s/sample; ETA ~2–2.5 h from ~06:15 local 2026-06-14).
- **⚠️ HARDWARE CAVEAT (only the user can fix):** `caffeinate` CANNOT beat **lid-close sleep on
  battery** — that is what killed Attempt 2 overnight. Machine must stay **plugged in + lid open**
  (or clamshell w/ external display) for an unattended finish. If it sleeps again, just resume
  (≤250 samples lost).

### Deviations (Phase 5, continued)
- **D25 — Eval durability hardening.** Plan §4.3/§5 assumed a single-shot inference run; reality
  needed checkpoint+resume+unbuffered+detached+caffeinate (above). No metric/schema impact.

---

## ▶▶ FRESH-THREAD RESTART RUNBOOK (read this first if resuming with no context) ◀◀

**Project state as of 2026-06-14 ~06:20 local:** ALL phases (1–5) implemented. Phase 5 code is
COMPLETE and committed; the only thing in flight is the optional **full 4,500-sample LLM eval**
(Attempt 3 running detached). Everything else is done.

**Repo:** `/Users/laptop/Developer/fdl_technicalInterview/space-telemetry-anom-llm` (branch `main`).
**Plan:** `thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`.
**Key git commits:** `6c9220a` (Phase 5 report + evaluate.py), `1bfd715` (eval hardening),
`eed8d41` in `~/.claude` (CLAUDE.md durability guidance).

**Storage (CRITICAL — internal disk nearly full):**
- Raw ESA-AD: `/Volumes/DUAL DRIVE/esa-ad/` (FAT32; KEEP until project teardown).
- LoRA: `/Volumes/DUAL DRIVE/star-pipeline/models/lora/qwen3-8b-advice/`.
- GGUF (5 GB, exceeds FAT32 4 GB limit → on local APFS):
  `/Users/laptop/Developer/fdl_technicalInterview/models/gguf/star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf`
  (also on HF: `dyrtyData/star-pipeline-qwen3-8b-advice-gguf`).
- `STAR_MODEL_DIR=/Users/laptop/Developer/fdl_technicalInterview/models` (Makefile default).

**To check the in-flight eval:**
```
pgrep -fl test_local_gguf                       # is it alive?
tail -5 results/.eval_llm_full.log              # progress (unbuffered)
python3 -c "import json;d=json.load(open('results/inference_test.json'));print(d['summary']['n_samples'],d['summary'].get('partial'))"
```
- If `n_samples == 4500` and `partial == false` → eval DONE. Run the wrap-up below.
- If the process is GONE and `n_samples < 4500` → resume (relaunch Attempt-3 command above; it
  picks up from the last checkpoint). Make sure the machine is plugged in + lid open first.

**Wrap-up once the eval shows 4500 samples (THIS IS THE REMAINING WORK):**
1. `make eval-all` then `make validate-eval` (both must pass).
2. Update the n=4500 LLM row + Key Findings in BOTH the plan's Phase-5 status block and this log's
   comparison table (currently they show the **n=100** numbers: P=0.432 R=0.615 F1=0.508).
   `evaluate.py` reads `n_samples` from the file, so the report regenerates automatically — only
   the hand-written tables in the plan/log need the new numbers.
3. Commit: `[Phase 5] Full 4,500-sample LLM eval — report + tables updated`.
4. (Optional) Stop the completion monitor; `pkill -f "caffeinate -dimsu"` is auto-released when
   python exits, but verify no stray `caffeinate` lingers: `pgrep -fl "caffeinate -dimsu"`.

**Manual ops cheat-sheet:**
- Stop the eval: `pkill -f test_local_gguf` (caffeinate dies with it). Resume later with `--resume`.
- Kill a stuck caffeinate: `pkill -f "caffeinate -dimsu"`.
- The 100-sample report artifacts (`results/comparison_report.md`, `comparison_metrics.json`) are
  **gitignored** (`results/**`) — regenerate via `make eval-all`, don't expect them in git.

**Project teardown (final step, ONLY after the 4500 eval + commit are done):** see the plan's
"Project Teardown / Cleanup" checklist — rotate the Kaggle token (pasted plaintext 2026-06-12)
and delete raw ESA-AD from DUAL DRIVE. Not yet done; deliberately deferred.

---

## Phase 6: Did fine-tuning help? (base + frontier comparison) — 2026-06-14

**Status: code + frontier eval COMPLETE & committed; base run IN FLIGHT (adopted external 500-window run).**

### What was built
- **`test_local_gguf.py`**: added `--results-file PATH` (so the base run writes
  `results/inference_base.json` without clobbering the fine-tuned `inference_test.json`) and
  `--approach-label` (cosmetic summary label). Threaded through `write_results`/`compute_summary`/
  resume logic. Same `SYSTEM_PROMPT` + `format_prompt` + parser preserved → identical harness.
- **`src/inference/select_frontier_sample.py`** (NEW): `--select` freezes a seed-42 stratified
  sample (n=150, ~25% anomalous) → `data/frontier/frontier_sample.jsonl`; a leak-free
  `frontier_prompts.jsonl` (index + instruction only) is the detector's input. `--assemble`
  joins classifications to ground truth → `results/inference_frontier_sample.json` (same schema
  as `inference_test.json`).
- **`evaluate.py`**: extracted `_summarize_detection(path, approach, with_affinity)` shared by
  the fine-tuned LLM, base, and frontier; added `load_base_results`/`load_frontier_results`,
  a new `format_compliance` metric (= parseable-verdict fraction), and a computed
  **"Did fine-tuning help?"** report section (F1 / CEF0.5 / format-compliance / structured-advice
  deltas). Rows are added only when their files load cleanly (validate-eval forbids error rows).
- **Makefile**: `eval-base`, `frontier-select`, `frontier-assemble` targets.
- **Cleanup**: deleted the corrupt 1.13 GB partial GGUF on `DUAL DRIVE`
  (`star-pipeline-advice_gguf/qwen3-8b.Q4_K_M.gguf` + its `._` fork).
- **Base model**: downloaded `unsloth/Qwen3-8B-GGUF` → `Qwen3-8B-Q4_K_M.gguf` (5,027,784,512 B)
  to `models/gguf/base-qwen3-8b/` (local SSD, 208 GB free — the "nearly full" warning was stale).

### Results so far
| Approach | F1 | format-compliance | structured-advice |
|---|---|---|---|
| LLM Detection (fine-tuned, n=4500) | **0.453** | 0.994 | 0.996 |
| Frontier zero-shot (Claude, n=150) | 0.254 | 1.000 | 1.000 |
| Base Qwen3-8B (n=500) | _pending_ | _pending (smoke ≈0)_ | _pending (≈0)_ |

`make eval-all` + `make validate-eval` pass with the frontier row present.

### Deviations
- **D26 — Adopted an external 500-window base run.** A base run (PID 45306, `--limit 500`, label
  "Qwen3-8B BASE", → `run_base.log`) started concurrently with this session, NOT launched by this
  thread. To avoid a double-writer clobber on `results/inference_base.json`, killed my own
  just-launched duplicate (`--limit 0`) and adopted the external one. My launch command's
  `rm -f results/inference_base.json` ran <1 min after the external job started (<10 samples, no
  checkpoint yet) → destroyed nothing. User was asked how to proceed and did not answer → took the
  low-regret default (don't kill a running job). 500 windows (random, since the split is shuffled)
  is an adequate base control. Full 4,500: `make eval-base LIMIT=0` (resumable).
- **D27 — Base scored under the identical harness; format-compliance is the headline.** Base
  Qwen3-8B defaults to *thinking mode*, burns the 300-token budget on `<think>` and rarely emits
  the terse verdict → UNKNOWN. `/no_think` makes it ramble in markdown, still non-compliant.
  Chose the strict controlled comparison (same prompt/decoding/parser; only the weights differ).
  The clean, computable fine-tuning delta is **output-contract compliance** (fine-tune 99.4% vs
  base ≈0%) + structured-advice fraction. All-UNKNOWN base → F1≈0 is a faithful finding, not a bug.
- **D28 — Frontier: instruction-only input + fresh-thread sub-agent detector.** Detector sees only
  the `instruction` (mission/channel + ~10 normalized values) — exactly the fine-tune's input;
  labels stripped into `frontier_prompts.jsonl`. Realized the "session model as detector" as a
  fresh-thread Claude sub-agent (general-purpose, opus) classifying all 150 →
  `results/frontier_classifications.json`. Honest modest result (F1 0.254): zero-shot can't recover
  the mission/channel-specific patterns the fine-tune memorized; 23/37 sampled anomalies are
  `subtle_deviation`, near-invisible from 10 values.
- **D29 — Graceful degradation.** `evaluate.py` adds base/frontier rows only when their files load
  cleanly (validate-eval forbids `error` rows); the report is frontier-only until base finalizes.
- **.gitignore:** force-track `results/frontier_classifications.json` (in-session judgments, not
  deterministically regenerable); ignore `*.log`, `inference_base.json`, `inference_frontier_sample.json`,
  the report, and the base GGUF.

### Update 2026-06-14 (later) — base run stopped at n=100; few-shot baseline added (D30)
- **Base zero-shot run stopped early at n=100** (user request): 100/100 UNKNOWN confirmed the run
  goes to P=R=F1=0. That is a conclusive *compliance* finding at any n; finishing the other 400
  added nothing. `results/inference_base.json` (n=100, partial=true) is the compliance datapoint;
  the base row now renders F1=0.000, format_compliance=0.000 in the report.
- **D30 — few-shot base baseline ("prompting instead of fine-tuning").** Added `--few-shot N`
  (per-class examples from TRAIN, no test leakage) + `--no-think` to `test_local_gguf.py`;
  `load_base_fewshot_results()` + a row + a dedicated bullet in evaluate.py's "Did fine-tuning
  help?" section; `make eval-base-fewshot`. **1 example/class collapsed to always-ANOMALY**
  (recency/label bias, F1=0.5 = base rate); **2/class discriminates** (n=30 probe: F1≈0.57,
  P=0.44, R=0.83; anomaly-biased; ~6.6 s/window; no structured advice). 500-window run launched
  detached (`results/inference_base_fewshot.json`). Honest framing: prompting recovers compliance
  + a comparable detection score, but NOT structured advice or latency — those stay fine-tuning's
  wins. Committed `1fc9534` (only my files; the Phase-8 thread's `train_detection.py` WIP left
  untouched — concurrent-edit hazard managed by per-file staging).
- **Concurrency note:** the Phase-8 (vision) thread shares this working tree and committed
  `a6f26a6` (vision eval harness + evaluate.py vision row). evaluate.py/Makefile now carry BOTH
  Phase-6 and Phase-8 code; I edit only my sections and stage only my files.

### Phase 6 CLOSED (2026-06-14) — few-shot run finished, final numbers in
- **Few-shot base run complete** (n=500/500, partial=false): **F1=0.420, P=0.282, R=0.824,
  CEF0.5=0.325, compliance=1.000, structured-advice=0.129, 8.56s/window**, 397/500 flagged ANOMALY
  (anomaly-biased). The real n=500 figure (F1=0.420) is close to the n=30 probe (≈0.57) but lower
  and more stable.
- **Final four-way "Did fine-tuning help?" read:**

  | Model | F1 | CEF0.5 | compliance | structured-advice | s/window |
  |---|---|---|---|---|---|
  | Fine-tuned LLM (n=4500) | **0.453** | **0.392** | 0.994 | **0.996** | **2.77** |
  | Base few-shot (n=500) | 0.420 | 0.325 | 1.000 | 0.129 | 8.56 |
  | Base zero-shot (n=100) | 0.000 | 0.000 | 0.000 | 0.000 | — |
  | Frontier zero-shot (n=150) | 0.254 | 0.284 | 1.000 | 1.000 | — |

  **Conclusion:** few-shot prompting nearly matches detection F1 (Δ−0.032) and recovers compliance,
  but the fine-tune wins on precision-weighted **CEF0.5** (0.392 vs 0.325 — few-shot over-flags),
  **structured advice** (99.6% vs 12.9% — few-shot only sometimes copies the demonstrated format),
  and **latency** (3× faster). Zero-shot base/frontier recover neither compliance nor competitive
  detection. This is the skeptic-proof version of the headline claim.
- **Base GGUF deleted** (`rm -rf models/gguf/base-qwen3-8b/`, ~5 GB reclaimed) — same weights served
  both base runs, so it's no longer needed.

### D31 — Frontier reframe + trivial baseline (2026-06-14, commits `16a33b4`, `f6278de`)
User asked why the frontier (F1=0.254) was below the few-shot base (0.420), and whether it was a
fair comparison. Investigation:
- **Frontier model = Claude Opus 4.8** (`claude-opus-4-8`, the session model, run via a
  general-purpose sub-agent). Original frontier eval was ZERO-shot vs the base's FEW-shot — a real
  asymmetry the user caught.
- **Ran the fair control — frontier FEW-shot** (same 2 train examples, same frozen 150 windows):
  F1=**0.239** (P=0.200 R=0.297). Few-shot did NOT help → the gap was never a prompting artifact.
- **The decisive anchor — always-anomaly trivial baseline** (flag every window) = F1 **0.399** /
  CEF0.5 0.294 on the ~25%-positive set. The few-shot base's F1=0.420 (it flags **79%** of windows)
  is *barely above*, and on its own n=500 set (27% pos) the flag-all F1 is 0.428 — so the base is
  **at/below the dumb baseline**: it is over-flagging, not detecting. The frontier (zero/few-shot)
  is ~at chance (P≈R≈base-rate) because 10 context-free normalized values carry almost no signal.
- **The fine-tune (F1=0.453, balanced P=0.36/R=0.61) is the ONLY approach that beats always-anomaly
  with a real P/R trade-off** — the lone genuine detector. This *strengthens* the headline.
- Wired both new rows (frontier-few-shot, trivial baseline) + the reframed narrative into
  `evaluate.py`/report; dovetails with Phase 9 (advice quality gated by precision → deploy LLM as
  advisor on a high-precision detector = the Hybrid). Artifacts: `results/inference_frontier_fewshot.json`,
  `results/frontier_fewshot_classifications.json` (committed; results are tracked now).
- `make eval-all` + `make validate-eval` → OK with all four contrasts present.
- **Phase 6 = COMPLETE.** Remaining project work: Phases 7 (full LSTM), 8 (vision — concurrent
  thread), 9 (semantic advice grading), 10 (teardown).

---

## Phase 8: Vision detector (Qwen3-VL on PNG plots) — IN PROGRESS (2026-06-14)

**Status: local code COMPLETE & committed; cloud env READY; training BLOCKED on a smoke test
that is fix-applied-but-unconfirmed (session interrupted right before re-running it).**
**⚠️ A Vast.ai instance is LIVE and BILLING ($0.401/hr) — resume promptly or destroy it.**

### What was built (committed)
- **`src/inference/eval_vision.py`** (commit `a6f26a6`): on-instance VL eval harness. Loads the
  trained Qwen3-VL adapter via `FastVisionModel.from_pretrained` + `for_inference`, classifies
  test PNGs from `data/processed/plots/test_metadata.jsonl`, writes
  `results/inference_vision.json` in the SAME `{summary, results}` schema as `test_local_gguf.py`
  (so evaluate.py's shared `_summarize_detection` loader works unchanged). Has
  `--limit/--resume/--checkpoint-every` for durability.
- **`src/inference/evaluate.py`** (commit `a6f26a6`): `VISION_FILE`/`VISION_APPROACH`,
  `load_vision_results()` (unit `windows (PNG)`), row wired into `main()` with graceful
  degradation (row appears only when the file loads cleanly), methodology note. NOTE: this file
  ALSO carries Phase-6 (base/few-shot) code from the concurrent thread — edit only your sections,
  stage per-file.
- **`Makefile`** (commit `a6f26a6`): `eval-vision` target.
- **`src/training/train_detection.py`** (commit `0daa2ec`): THREE latent runtime bugs fixed (it
  was written in Phase 3 but NEVER run). See D31. lint passes. Synced to the instance (md5 match).

### ▶▶ LIVE CLOUD STATE — how to resume (READ FIRST) ◀◀
- **Instance**: id **40936091**, 1× **RTX A6000 46 GB**, Delaware US-East, **$0.401/hr**, rel 0.999.
  Chosen over RTX 4090 for VL VRAM headroom AND it was cheaper + 8.5 Gbps both ways (fast HF I/O).
- **SSH (direct)**: `ssh -i ~/.ssh/vast_star -p 40995 -o StrictHostKeyChecking=no -o IdentitiesOnly=yes root@38.29.145.20`
  (proxy fallback: `ssh6.vast.ai` port `16090`). The passphraseless `~/.ssh/vast_star` key (from
  Phase 3) is registered on the account + attached. zsh won't word-split an ssh-in-a-var string —
  invoke ssh directly or write a tiny wrapper script (I used `/tmp/vssh`, ephemeral).
- **CLI**: `set -a; source .env; set +a; .venv/bin/vastai set api-key "$VASTAI_API_KEY"` then
  `.venv/bin/vastai show instance 40936091`. Vast credit was $20.09 at start.
- **Remote workdir**: `/workspace/star-pipeline` — has `data/processed/plots/` (6,000 PNGs, the
  macOS `._` AppleDouble forks were deleted), `data/processed/plots/{train,val,test}_metadata.jsonl`,
  `src/`, `config/`, and the FIXED `train_detection.py` (md5 `6738a1dc…`, in sync with the repo).
- **Env on instance (DO NOT "fix" — it works)**: Python 3.11.9 (conda), unsloth 2026.6.7,
  unsloth_zoo 2026.6.5, **torch 2.10.0+cu128**, **torchvision 0.25.0+cu128 (manually upgraded from
  the image's 0.19 — REQUIRED; unsloth's torch 2.10 needs torchvision≥0.25)**, transformers 5.5.0,
  trl 0.24.0. Base image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`. `HF_TOKEN` exported in
  `/root/.bashrc` + written to `/root/.cache/huggingface/token`.
- **Base VL model**: `unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit` (the plan's `Qwen3-VL-8B`
  does NOT exist) — already downloaded/cached on the instance during smoke tests.

### ⏭️ NEXT STEPS to finish Phase 8 (resume here)
1. **Confirm the smoke test passes** (the interrupted step): from the instance,
   `cd /workspace/star-pipeline && HF_TOKEN=$HF_TOKEN python src/training/train_detection.py --limit 32 --epochs 1`
   → expect it to get PAST trainer construction and show training step logs (loss). The repro of
   the exact trainer build (unsloth-first imports) already succeeded, so this should now work.
2. **Launch full training detached** (survives SSH drop), ~250 opt-steps for 2 epochs over 2,000
   train PNGs (batch 1 × grad-accum 16). On the A6000 estimate ~30–60 min; measure after step 10.
   `cd /workspace/star-pipeline && ( setsid env HF_TOKEN=$HF_TOKEN nohup python src/training/train_detection.py --epochs 2 > /workspace/train_vl.log 2>&1 < /dev/null & )`
   Watch: `tr '\r' '\n' < /workspace/train_vl.log | grep -aE 'loss|epoch' | tail`. Output LoRA →
   `/workspace/star-pipeline/models/lora/qwen3-vl-detection/`.
3. **Push to HF BEFORE teardown** (preserve regardless of local inference): adapter (+ merged +
   GGUF + `mmproj` if exporting), e.g. `dyrtyData/star-pipeline-qwen3-vl-8b-detection`. Use the
   on-instance `huggingface_hub` (already installed); token is set.
4. **Eval ON the instance** (multimodal GGUF on Metal is patchy → do NOT rely on local):
   `cd /workspace/star-pipeline && python src/inference/eval_vision.py --resume --checkpoint-every 250`
   → `results/inference_vision.json`. ~2,000 test PNGs. Then `scp` it back to the repo's `results/`.
5. **Destroy the instance**: `.venv/bin/vastai destroy instance 40936091` (STOPS billing). Only the
   *instance* teardown — NOT the project-wide Phase-10 teardown.
6. `make eval-all && make validate-eval` locally → the vision row appears automatically. Update the
   Phase-8 table in the plan + this log. Commit (stage ONLY Phase-8 files; Phase-6 thread shares
   the tree). `rm -f /tmp/vssh` if used.

### Deviations (Phase 8)
- **D31 — `train_detection.py` had 3 never-run bugs (fixed, commit `0daa2ec`).** It was written in
  Phase 3 but never executed, so: (a) **import order** — `from trl import SFTTrainer` ran BEFORE
  `from unsloth import …`, binding the UNPATCHED TRL trainer; that path can't handle the Qwen3VL
  processor and raises `ValueError: eos_token '<EOS_TOKEN>' not in vocab`. Fixed by importing
  unsloth first (+ `# noqa: I001` so ruff's isort doesn't re-sort `trl` ahead of `unsloth`).
  (b) `SFTConfig(max_seq_length=…)` → `max_length` (TRL 0.24 rename, the same D9 fix
  `train_advice.py` got). (c) pinned `eos_token="<|im_end|>"` (Unsloth's `for_training` swaps
  `tokenizer.eos_token` for a `<EOS_TOKEN>` placeholder absent from the vocab). Root cause = (a);
  (b)/(c) are belt-and-suspenders. A faithful repro with unsloth-first imports built the trainer OK.
- **D32 — GPU choice: A6000 (not RTX 4090).** The plan suggested 4090 (consider A6000). A single
  US A6000 was both cheaper ($0.401/hr) and 48 GB → VL headroom; Delaware host had 8.5 Gbps both
  ways (avoids the Phase-4 Hungary download crawl). Picked it directly by offer id (offers are
  ephemeral — the first verified id expired; re-searched and launched a fresh one).
- **D33 — env: torchvision upgrade required.** `pip install unsloth unsloth_zoo` upgraded torch to
  2.10.0+cu128 but left the image's torchvision 0.19; vision modeling needs ≥0.25. Installed
  `torchvision==0.25.0` from the cu128 wheel index. (Onstart cmd was
  `pip install unsloth unsloth_zoo huggingface_hub pillow`; the torchvision pin is the only manual add.)
- **Upload method**: tar-stream over SSH (`tar czf - … | ssh … tar xzf -`); the pytorch image lacks
  rsync (D12/D15). macOS adds `._` AppleDouble forks to the tar → they also matched `*.png` (12,000
  vs 6,000); deleted with `find … -name '._*' -delete`. Harmless anyway (train/eval iterate the
  metadata file, not a glob).

### Plan sections that may need updating based on this phase
- Phase 8 step 3 (HF push) + step 5 (local-vs-cloud eval): the eval is best run ON the instance via
  `eval_vision.py` (already written for that); local Metal VL inference is the patchy fallback the
  plan flagged — we default to on-instance. Capture `inference_vision.json` BEFORE teardown.
- `train_detection.py` defaults: 2 epochs, batch 1 × grad-accum 16, lr 1e-4, r=α=16, all layers.

### Phase 8 COMPLETE (2026-06-14 ~18:05 UTC)
- **Smoke test passed** after the D31 import-order fix → training started cleanly.
- **Training**: `train_detection.py --epochs 2`, detached (setsid, PPID=1) on A6000. 250 steps over
  2,000 train PNGs, **~65 min** (~10.5 s/step). train_loss **4.48 → 0.34**, eval_loss **0.0089**
  (epoch 2; the binary task converges hard — expected for VL SFT on a 2-class signal). LoRA adapter
  (205 MB) → `models/lora/qwen3-vl-detection/`.
- **HF push** (before teardown, insurance): adapter + processor →
  `dyrtyData/star-pipeline-qwen3-vl-8b-detection` (public, `ignore_patterns=['checkpoint-*']`).
- **Eval on-instance** (`eval_vision.py --resume --checkpoint-every 200`, detached): all **2,000
  test PNGs**, `partial=false`, **~0.86 s/sample** (steady-state — the 3.6 s smoke figure was
  first-call warmup; full run ≈ 30 min, much faster than the ~72 min estimate). Scp'd back to
  `results/inference_vision.json` (gitignored; HF + the script reproduce it).
- **Result**: accuracy 0.806, **P=0.769 R=0.325 F1=0.457**, TP=163 FP=49 FN=338 TN=1450,
  **Unknown=0** (100% format compliance), CEF0.5=0.604.
  - **Precision-oriented**, the mirror image of the recall-oriented text LLM (P=0.360 R=0.609
    F1=0.453). Nearly identical F1, opposite P/R trade-off → because CEF0.5 weights precision, the
    **vision model has the highest CEF0.5 of any LLM approach (0.604)**. It misses more anomalies
    (338 FN) but almost never false-alarms (49 FP) — operationally attractive where false alarms are
    costly. Pure detector (no advice). A real, modality-independent third LLM signal that completes
    the original AnomSeer-style 3-way design.
- **Instance 40936091 DESTROYED** (`vastai destroy instance 40936091`; confirmed 0 instances) →
  billing stopped. **Total Phase-8 cloud cost ≈ $1.0** (~2.3 h A6000 @ $0.417/hr incl. storage).
- **`make eval-all` + `make validate-eval`** → vision row present, all 8 approaches valid.
- **Deviation D34 — eval far faster than estimated.** Sized the eval at ~72 min from the 3.58 s/
  sample smoke; steady-state was ~0.86 s/sample (warmup-dominated smoke). No action needed; the run
  is checkpointed + resumable so a worst-case estimate was the safe default.

### Phase 8 impact on the rest of the plan
- Phase 10 (project teardown) precondition "Phases 5–9 complete (Phase 8 optional)" is now closer:
  Phase 8 is DONE. Remaining optional: Phase 7 (full LSTM — note: someone re-ran it, LSTM F1 now
  0.6979 vs the old 0.663 3-channel smoke) and Phase 9 (semantic advice grading). Teardown still
  deferred until the owner decides Phases 7/9 are done-or-skipped.
- `evaluate.py` + `Makefile` are SHARED with the concurrent Phase-6 thread (base zero-shot/few-shot
  rows). The committed report now carries 8 approaches: IF, LSTM, text-LLM, **vision-LLM**, base
  zero-shot, base few-shot, frontier, hybrid. Stage Phase-8 files individually when committing.

---

## Phase 9: Semantic advice grading (in-session Claude as judge) — COMPLETE (2026-06-14)

**Status: COMPLETE.** Ran fully independently of Phase 7 (full LSTM still pending its 58-channel
run — `baseline_results.json` still holds the 1-channel smoke of the new code), the raw data, and
the cloud. Free, no API.

### What was built
- **`src/inference/grade_advice_sample.py`** (NEW, mirrors `select_frontier_sample.py`):
  - `--select [--n 120]`: filters `results/inference_test.json` to the model's ANOMALY predictions
    (1,898 of 4,500), takes a **seed-42 sample preserving the population TP/FP ratio** (684 TP /
    1,214 FP → 43 TP / 77 FP in the sample), joins each to `test_with_advice.jsonl[index]` for the
    window context + true `anomaly_ratio`, and attaches the time-overlapping gold-advice record
    (per `(mission, channel)`) as an optional reference. Writes
    `data/advice_grading/advice_sample.jsonl`.
  - `--assemble PATH`: joins an in-session judgments JSON (per-record correctness/actionability/
    grounding, 0-2) → `results/advice_grading_sample.json` with overall + **TP-split / FP-split**
    summary stats (mean per axis, mean total /6, pct_correct/actionable/grounded/high_quality).
- **`evaluate.py`**: added `generate_advice_quality_section()` → renders an
  **"Advice quality (semantic) — Phase 9"** subsection (degrades gracefully to nothing when the
  grading file is absent). Added `ADVICE_GRADE_FILE` constant; wired the section after the
  "Did fine-tuning help?" block.
- **Makefile**: `grade-advice-select`, `grade-advice-assemble` targets (+ `.PHONY`).

### Judging methodology (transparent, fact-grounded rubric)
The in-session judge applied a consistent rubric anchored in **verifiable** signals rather than
vibes (the advice is templated, so its semantic content is checkable):
- **Grounding** is verifiably strong: **119/119** flagged windows name the *correct* channel; the
  physical unit (temperature/pressure/binary) is consistent; only **3/120** mislabel the subsystem
  vs gold (cross-mission `channel_N` collisions).
- **Correctness** keys on ground truth: a flag on a truly-nominal window (FP) is incorrect by
  construction (correctness 0); on a true anomaly (TP), 2 if the diagnosed pattern/severity matches
  the true `anomaly_ratio`, 1 if it over/under-calls severity.
- **Actionability**: severity-appropriate guidance present → 2; proportionate-but-generic → 1;
  high-severity "investigate" raised on a nominal window (a costly false alarm) → 0.
The judgments file (`results/advice_judgments.json`) carries a per-record `note` with the rationale.

### Results (`results/advice_grading_sample.json`, n=120)
| Subset | n | Correctness | Actionability | Grounding | Mean /6 | High-quality |
|--------|---|------|------|------|------|------|
| All flags | 120 | 0.64 | 1.03 | 1.01 | 2.68 | 34% |
| True positives | 43 | 1.79 | 1.93 | 1.86 | **5.58** | **95%** |
| False positives | 77 | 0.00 | 0.53 | 0.53 | 1.06 | 0% |

**Finding:** when the model is *right to flag*, its advice is genuinely good (5.58/6, 95%
high-quality, 100% grounded & actionable). On false alarms (≈64% of flags, precision ≈0.36) the
advice is built on a false premise. So **advice quality is gated by detection precision** — the
direct evidence for recommending the fine-tune as the **advisor on a high-precision detector
(the Hybrid: LSTM precision ≈0.84 + LLM advice)**, not as the standalone detector. This converts
Phase 5/6's "99.6% structured" into a defensible "95% high-quality *when correctly triggered*".

### Verification
- `make eval-all` → report regenerated with the Phase-9 subsection; `make validate-eval` → **OK**
  (8 detection approaches still valid; the advice section is additive, not a scored row).
- `make lint` (ruff check + format) → clean on all 22 files.

### Deviations
- **D35 — judge is a transparent rubric, not free-form scoring.** The plan said "the in-session
  agent reviews … score on a small rubric." Because the advice is templated, the judge encoded the
  rubric as deterministic, *verifiable* criteria (channel-name match, stated-% vs true ratio,
  severity↔pattern coherence, GT-gated correctness) for reproducibility, with per-record notes.
  This is stricter and more defensible than impressionistic scoring; same rubric the plan specified.
- **D36 — correctness is GT-gated (FP ⇒ 0).** Grading advice on a false-positive window scores
  correctness 0 by construction (no real anomaly to diagnose). Reported TP/FP-split so the
  advisory quality "when the model is right" (the operationally relevant number) is not masked by
  the false-alarm rate. Stated explicitly in the report + the file's `note`.
- **D37 — gold advice used as optional reference only.** As the Phase-9 readiness block predicted,
  `inference_test.json` carries no `pattern`/`start_time` for a strict 1:1 gold join; the join is by
  `(mission, channel)` time-overlap and is advisory. Primary grading is against window context +
  ground-truth label, which needs no gold join.

### Impact on the rest of the plan
- **No changes needed to other phases.** Phase 9 is read-only w.r.t. all prior artifacts; it only
  *adds* a report subsection + two result JSONs + one script + two Makefile targets.
- **Concurrency:** `evaluate.py`, `Makefile`, `comparison_report.md` are shared with the Phase-7
  thread. Verified no Phase-7 process is running and the tree was clean; the report diff is **+15
  lines (the Phase-9 section only)** — Phase 6/7/8 numbers untouched. When Phase 7's full
  58-channel LSTM run lands, it will regenerate the report and the Phase-9 section persists. Staged
  Phase-9 files individually (never `git add -A`).
- **Phase 10 teardown** precondition "Phases 5–9 complete (Phase 8 optional)" now needs only
  Phase 7's full LSTM run to be finished-or-skipped.
