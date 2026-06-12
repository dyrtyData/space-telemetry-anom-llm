# STAR-Pipeline Implementation Log

**Plan**: `thoughts/shared/plans/2026-06-12-star-pipeline-implementation.md`
**Started**: 2026-06-12
**Status**: In Progress

---

## Summary

| Phase | Status | Started | Completed | Deviations |
|-------|--------|---------|-----------|------------|
| 1 (code) | completed | 2026-06-12 14:35 | 2026-06-12 14:55 | pyproject.toml needed hatch build config |
| 1 (data pipeline) | **in progress** | 2026-06-12 14:48 | - | D2 ETL rewrite + D3 Kaggle source + D4 resample/subsample (see below) |
| 1.5 | pending | - | - | - |
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
- **Raw-data lifecycle**: raw is needed through Phase 2 (LSTM reads it too), so it stays until
  Phase 2; only then is it deletable. For "just Phase 1" we keep Mission1 raw on the drive.

### CURRENT STATUS (in progress)
- Mission1 downloading to the drive via `download_kaggle.py` (background, ~45 min, network-bound).
- Pending on completion: run `patch_telemetry.py` (Mission1) → splits, `generate_plots.py` →
  PNGs, then validate (schema, anomaly balance, plot presence) and re-update log/plan.

---

## Notes for Future Phases

### Phase 1.5 (Advice Label Generation)
- Awaiting ETL completion with **real data** before generating advice labels
- Will generate advice in-session to avoid API costs

### Phase 2 (LSTM Baseline)
- `train_lstm.py` / `isolation_forest.py` make the **same `telemetry.pkl`/`labels.pkl`
  assumption** as the ETL (plan lines 795–798, 868–871). They will need the same loader fix
  as D2. Recommend extracting a shared `load_mission_data()` into a small `src/etl/io.py`
  once the real ESA-AD structure is confirmed, and importing it from all three scripts.

### Phase 3-5 (Cloud Training, Export, Evaluation)
- No changes needed yet; depend on Phase 1 output format, which may shift slightly once D2 is
  resolved (JSONL schema is expected to stay stable).

