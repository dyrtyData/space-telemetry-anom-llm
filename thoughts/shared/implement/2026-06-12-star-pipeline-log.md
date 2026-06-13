# STAR-Pipeline Implementation Log

**Plan**: `thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`
**Started**: 2026-06-12
**Status**: Phase 1 COMPLETE — Phase 1.5 next

---

## Summary

| Phase | Status | Started | Completed | Deviations |
|-------|--------|---------|-----------|------------|
| 1 (code) | completed | 2026-06-12 14:35 | 2026-06-12 14:55 | pyproject.toml needed hatch build config |
| 1 (data pipeline) | **completed (all 3 missions)** | 2026-06-12 14:48 | 2026-06-12 23:45 | D2–D5; full dataset on DUAL DRIVE |
| 1.5 | **completed** | 2026-06-12 23:45 | 2026-06-13 00:10 | In-session generation (stats + channel meta) |
| 2 | pending | - | - | - |
| 3 | pending | - | - | - |
| 4 | pending | - | - | - |
| 5 | pending | - | - | - |

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

