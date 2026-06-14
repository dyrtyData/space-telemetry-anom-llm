# STAR-Pipeline — Space Telemetry Anomaly Detection & Resolution

> An end-to-end, open-source pipeline that **detects anomalies** in real satellite telemetry and
> **generates human-readable diagnostic advice** — no external API vendor required.

Built on the **ESA Anomaly Dataset (ESA-AD)**, this project benchmarks four approaches to the same
task and fine-tunes an open-source LLM (Qwen3-8B) with QLoRA to act as a localized anomaly
advisor. It deliberately compares the *research-frontier* "LLM does everything" approach against
*classical* baselines, reports the result honestly, and lands on the architecture the data
actually supports.

---

## TL;DR results

Final evaluation on the ESA-AD test split (4,500 windows, 25% anomalous):

| Approach | Precision | Recall | F1 | CEF0.5† | Detects? | Advises? | Cost |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | ✅ | ❌ | ~instant |
| **LSTM baseline** (Telemanom-style) | **0.835** | 0.552 | **0.663** | **0.757** | ✅ | ❌ | ~instant |
| LLM detection (Qwen3-8B, QLoRA→GGUF) | 0.360 | **0.609** | 0.453 | 0.392 | ✅ | ✅ | 2.77 s/window |
| **Hybrid** (LSTM detect → LLM advise) | **0.835** | 0.552 | **0.663** | **0.757** | ✅ | ✅ | LLM only on flags |

† CEF0.5 = precision-weighted F-beta (β=0.5), the operationally relevant metric when false alarms
are costly.

**The honest headline:** a tuned LSTM is the stronger *detector* (2.3× the precision of the direct
LLM). But the LLM delivers what no baseline can — reliable structured advice (**99.6%** of its
flags include a `DIAGNOSIS` + `ADVICE` + `ACTION`). The architecture the numbers recommend is the
**hybrid**: a cheap high-precision detector that triggers an LLM advisor only when something fires.

📄 Full write-up: [`thoughts/shared/research/2026-06-14-results-analysis.md`](thoughts/shared/research/2026-06-14-results-analysis.md)
📊 Generated report: [`results/comparison_report.md`](results/comparison_report.md)

> ⚠️ **Read the caveat:** the baselines were scored on a 3-channel smoke subset with per-channel
> tuned thresholds, while the LLM was scored on the full 4,500-window cross-mission split. The
> detection gap is a believable *upper bound* on the LSTM's advantage, not a settled like-for-like
> result. See the analysis doc — leveling this field is the top follow-up.

---

## Why this project

The brief: a product manager needs **anomaly detection + end-user advice on the effect of a
change** for a mission-critical system, and wants to **own the model** rather than depend on an
external API. The open question — *can a fine-tuned open-source LLM do this?* — is answered
empirically here rather than asserted.

This is a technical showcase for an AI-engineering role: it demonstrates production-style ETL, a
classical-ML baseline, **LLM fine-tuning via QLoRA/Unsloth**, GGUF export, local Metal inference,
and a rigorous, honestly-reported evaluation.

---

## Architecture

```
                         ESA-AD raw telemetry (3 missions, 224 channels, ~29 GB)
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              │  Phase 1 — ETL (local, M3 Max)                              │
              │  resample 1h · RevIN normalize · 32-step rolling windows    │
              │  balanced subsample · → 30,000 instruction/response JSONL   │
              └─────────────────────────────┬─────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
  Phase 2 — Baselines               Phase 3 — LLM fine-tune            Phase 1.5 — Advice labels
  (local)                           (cloud: Vast.ai RTX 4090)          (in-session generation)
  • LSTM (per-channel               • Qwen3-8B + QLoRA (Unsloth)       • 7,457 anomaly advice
    error thresholding)             • r=16, α=16, lr 2e-4, 3 ep          records w/ severity
  • Isolation Forest                • → GGUF Q4_K_M export                + pattern
        │                                   │
        │                                   ▼
        │                           Phase 4 — Local inference (M3 Max, Metal GPU)
        │                           • llama-cpp-python, all layers offloaded
        │                                   │
        └───────────────────┬───────────────┘
                            ▼
              Phase 5 — Unified evaluation (make eval-all)
              Precision / Recall / F1 / CEF0.5 / Affinity-F1  →  comparison_report.md

  Recommended production design (the Hybrid):
     window → LSTM detector (cheap, P=0.835) ──flag──► Qwen3-8B advisor → operator-ready alert
```

---

## Dataset

[**ESA Anomaly Dataset (ESA-AD)**](https://doi.org/10.5281/zenodo.12528696) — the first
large-scale, real-life satellite telemetry benchmark with curated anomaly annotations, covering
three ESA missions. Chosen over NASA MSL/SMAP, whose Telemanom baseline is purpose-built and whose
quality is widely criticized.

- **Splits** (balanced-subsampled, ~25% anomalous): train **21,000** · val **4,500** · test **4,500**
- **Windowing:** 32-step rolling windows, stride 16, resampled to 1h, RevIN per-channel
  reversible instance normalization
- Raw data is **not** committed (it's ~29 GB); `make download` fetches it from the Kaggle mirror.

---

## Tech stack

| Layer | Choice |
|---|---|
| Base model | `unsloth/Qwen3-8B-bnb-4bit` |
| Fine-tuning | Unsloth + QLoRA (r=16, α=16, target = all-linear, lr 2e-4, 3 epochs) |
| Training compute | Vast.ai single RTX 4090 (24 GB), official Unsloth image — **~$2.33 total** |
| Quantization | GGUF `Q4_K_M` (Unsloth dynamic) |
| Local inference | `llama-cpp-python` with Metal GPU offload (M3 Max, 36 GB unified) |
| Baselines | Keras 3 (torch backend) LSTM · scikit-learn Isolation Forest |
| ETL / data | NumPy, pandas, h5py |
| Tooling | `ruff`, `pytest`, `make` |

---

## Reproducing the results

> **Prerequisites:** Python 3.11+, ~30 GB free disk (or an external drive) for raw data, and for
> the LLM steps a downloaded GGUF. Training (Phase 3) runs on a rented NVIDIA GPU, not locally.

```bash
# 0. Environment
make setup                       # venv + editable install ([dev,lstm] extras)

# 1. ETL — download ESA-AD (Kaggle mirror) and build windows/JSONL
make download ESA_DATA_DIR="/path/to/esa-ad"
make etl      ESA_DATA_DIR="/path/to/esa-ad"
make validate-etl

# 1.5 Advice labels (already committed; regenerate if desired)
make advice && make validate-advice

# 2. Classical baselines
make baseline       ESA_DATA_DIR="/path/to/esa-ad"   # LSTM
make baseline-if    ESA_DATA_DIR="/path/to/esa-ad"   # Isolation Forest
make validate-baseline

# 3. LLM fine-tune (on a Vast.ai RTX 4090 — see scripts/cloud/)
make format-train && make validate-format            # → ChatML JSONL
./scripts/cloud/launch_vast.sh                       # dry-run; --create to launch
#   (train_advice.py / export_gguf.py run ON the instance)

# 4. Local GGUF inference (M3 Max, Metal)
make install-local                                   # llama-cpp-python w/ Metal
make eval-llm LIMIT=0 \                               # full 4,500-window run
  STAR_MODEL_DIR="/path/to/models"
make validate-inference

# 5. Unified comparison report
make eval-all && make validate-eval                  # → results/comparison_report.md
```

The fine-tuned GGUF is published for convenience:
**[`dyrtyData/star-pipeline-qwen3-8b-advice-gguf`](https://huggingface.co/dyrtyData/star-pipeline-qwen3-8b-advice-gguf)** (Hugging Face).

### Durable long-running runs

The full LLM eval is a multi-hour job. `src/inference/test_local_gguf.py` supports
`--checkpoint-every N` and `--resume` (atomic checkpoint writes), so an interrupted run costs at
most one checkpoint. For an unattended run, keep the machine plugged in with the lid open
(`caffeinate` cannot override lid-close sleep on battery).

---

## Repository layout

```
src/
  etl/         download (Kaggle/Zenodo), RevIN windowing, advice-label + plot generation
  baselines/   train_lstm.py · isolation_forest.py
  training/    format_for_unsloth.py · train_advice.py · export_gguf.py   (cloud)
  inference/   test_local_gguf.py (Metal inference) · evaluate.py (unified report)
scripts/cloud/ Vast.ai launch / data upload / model download helpers
results/       comparison_report.md · comparison_metrics.json · per-approach JSON
thoughts/      research, plan, and a full implementation log (see below)
Makefile       every phase + a validate-* target encoding its success criteria
```

---

## Project journey & honest engineering log

This repo keeps its full paper trail — including the dead ends — because how the result was
reached is part of the showcase:

- **Plan:** [`thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`](thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md)
- **Implementation log** (every deviation D1–D25, the eval re-run saga, a fresh-thread restart
  runbook): [`thoughts/shared/implement/2026-06-12-star-pipeline-log.md`](thoughts/shared/implement/2026-06-12-star-pipeline-log.md)
- **Results analysis:** [`thoughts/shared/research/2026-06-14-results-analysis.md`](thoughts/shared/research/2026-06-14-results-analysis.md)

Notable real-world hurdles solved along the way: FAT32's 4 GB file limit forcing the 5 GB GGUF to
local SSD; a buffered-stdout + machine-sleep double failure that killed an overnight eval twice
(fixed with checkpoint/resume + unbuffered output + proper daemonization); and rewriting the
evaluation loaders to the *actual* persisted result schemas rather than the planned ones.

---

## Limitations

This is a showcase, not a deployed system. Known gaps, stated up front:

1. **Baselines are a 3-channel smoke run** with tuned thresholds — the detection head-to-head is
   an upper bound on the LSTM's advantage, not a like-for-like result.
2. **Different eval units** (baselines: 3 channels macro-averaged; LLM: 4,500 windows micro).
3. **Affinity-F1 is degenerate** on the shuffled test split (≈ window-level F1) — reported with
   that disclaimer.
4. **Advice is verified structurally (99.6%), not semantically** — no factuality/expert grading
   yet; advice labels were synthetically generated.
5. **Single fine-tune, no hyperparameter sweep**, and no detection-tuned LLM variant or P-R curve.

Each has a concrete next step in the [analysis doc](thoughts/shared/research/2026-06-14-results-analysis.md#9-recommended-next-steps-in-priority-order).

---

## License & data

Code is provided for review. The ESA-AD dataset is governed by its own license/terms (see Zenodo);
raw telemetry is not redistributed in this repo.
