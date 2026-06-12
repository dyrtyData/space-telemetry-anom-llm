---
date: 2026-06-12T16:04:27Z
researcher: Claude Code
git_commit: 74c2d26a455bb7b56d1a7c303b7b6f57a330e020
branch: main
repository: space-telemetry-anom-llm
topic: "STAR-Pipeline Codebase Research - Current State Analysis"
tags: [research, codebase, space-telemetry, anomaly-detection, llm-finetuning, star-pipeline]
status: complete
last_updated: 2026-06-12
last_updated_by: Claude Code
---

# Research: STAR-Pipeline Codebase Research - Current State Analysis

**Date**: 2026-06-12T16:04:27Z
**Researcher**: Claude Code
**Git Commit**: 74c2d26a455bb7b56d1a7c303b7b6f57a330e020
**Branch**: main
**Repository**: space-telemetry-anom-llm

## Research Question

What is the current state of the STAR-Pipeline (Space Telemetry Anomaly Detection & Resolution Pipeline) codebase, and what files/structures are relevant to the issue?

## Summary

The project is a **greenfield scaffold** with no implementation code. It contains:
- Empty placeholder files for the planned four-phase architecture (ETL, ML Baseline, LLM Fine-tuning, Inference)
- Comprehensive issue documentation defining the project requirements
- A Gemini exploratory thread documenting background research on the approach
- Context for an FDL (Frontier Development Lab) AI Engineering interview

All `.py`, `.yaml`, `.md` (README), and `requirements.txt` files in `src/` are 0 bytes. The project exists only as a directory structure and documentation.

## Detailed Findings

### Project Structure (Current State)

```
space-telemetry-anom-llm/
├── README.md                          # 0 bytes - empty
├── requirements.txt                   # 0 bytes - empty
├── data/
│   ├── processed/                     # empty directory
│   └── raw/                           # empty directory
├── notebooks/                         # empty directory
├── src/
│   ├── etl/
│   │   └── patch_telemetry.py         # 0 bytes - empty
│   ├── inference/
│   │   └── test_local_gguf.py         # 0 bytes - empty
│   └── training/
│       ├── config.yaml                # 0 bytes - empty
│       └── train.py                   # 0 bytes - empty
└── thoughts/
    └── issues/
        ├── Nina_FDL.png               # 193KB - FDL interview context
        ├── space_telemetry_anom_llm_ISSUE.md  # 7KB - main issue doc
        └── space-anomaly-detection-ai-advice_exploratoryThread.md  # 17KB
```

### Issue Documentation Analysis

**Location**: `thoughts/issues/space_telemetry_anom_llm_ISSUE.md`

The issue defines the STAR-Pipeline with these key components:

#### Hardware Constraints (Section 2)
- **Local (M3 Max MacBook)**: 14-core CPU, 30-core GPU, 36GB unified memory, 1TB SSD, macOS Tahoe 26.2
  - Purpose: Exploratory data analysis, ETL, GGUF model inference/testing
- **Cloud GPU**: Vast.ai or RunPod with RTX 4090 (24GB VRAM) or A6000 (48GB VRAM)
  - Purpose: LoRA fine-tuning (not to be done locally)

#### Reference Architectures (Section 3)
- **AnomLLM**: `github.com/rose-stl-lab/anomllm` - LLM-based anomaly detection
- **Time-LLM**: Time-series patching and prompt prefixing concepts
- **Unsloth**: QLoRA optimization for fine-tuning (docs at unsloth.ai)

#### Datasets (Section 4)
- **ESA Anomaly Dataset (ESA-AD)**: ~12GB real satellite telemetry with anomaly annotations
  - Reference: `d-nb.info/1361113049/34`
  - Benchmark: `kplabs-pl/ESA-ADB` on GitHub
- **NASA MSL & SMAP Telemetry**: Curiosity Rover and SMAP satellite data
  - Source: Kaggle `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`

#### Planned Four-Phase Architecture (Section 5)

| Phase | Name | Location | Purpose |
|-------|------|----------|---------|
| 1 | ETL & Data Transformation | Local | Process HDF5/CSV, create rolling windows, format as JSONL |
| 2 | Traditional ML Baseline | Local | Train LSTM/Autoencoder/Isolation Forest for comparison |
| 3 | LLM Fine-Tuning | Cloud | Unsloth + QLoRA on 7B-8B model (Qwen2.5/Llama-3) |
| 4 | Inference & Integration | Local | Export to GGUF, run on M3 Max, evaluate vs baseline |

#### Definition of Done (Section 6)
- Modular, scalable data pipelines converting telemetry to LLM-digestible patches
- Traditional ML baseline trained and evaluated
- Open-source LLM fine-tuned via Unsloth for anomaly detection + diagnostic advice
- Evaluation metrics (False Positives, Precision, Recall) comparing LLM vs baseline

### Gemini Exploratory Thread Analysis

**Location**: `thoughts/issues/space-anomaly-detection-ai-advice_exploratoryThread.md`

Documents a research conversation covering:

1. **Public Datasets**: ESA-AD (~12GB), NASA MSL/SMAP
2. **Open Source Projects**: ESA-ADB benchmark, Telemanom (NASA JPL LSTM framework)
3. **Traditional Detector Comparison**:
   - LSTM: Best for temporal dependencies (chosen by Telemanom)
   - Autoencoder: Good for complex multi-sensor relationships but weaker on sequences
   - Isolation Forest: Fast but ignores temporal sequence (high false positives)
4. **LLM Approach**: AnomLLM and Time-LLM prove LLMs can directly detect anomalies via tokenized numerical sequences
5. **Data Transformation**: Patching + projection - chunk time-series into rolling windows, project to embedding space
6. **Unsloth Hyperparameters**:
   - `r` (LoRA Rank): 16
   - `lora_alpha`: 32 (2x rank)
   - `target_modules`: all-linear
   - `learning_rate`: 2e-4
   - `per_device_train_batch_size`: 2-4
   - `max_steps`: 100 (initial validation)

### FDL Context

**Location**: `thoughts/issues/Nina_FDL.png`

The image shows a conversation about:
- FDL (Frontier Development Lab) interview context
- Interest in "space-based observing and working with satellite/space telescope data"
- This provides the motivation for choosing space telemetry datasets

### Files Relevant to Implementation

| Planned File | Purpose | Current State |
|--------------|---------|---------------|
| `src/etl/patch_telemetry.py` | Transform raw telemetry to JSONL patches | Empty (0 bytes) |
| `src/training/config.yaml` | Training hyperparameters | Empty (0 bytes) |
| `src/training/train.py` | Unsloth fine-tuning script | Empty (0 bytes) |
| `src/inference/test_local_gguf.py` | Local GGUF model inference | Empty (0 bytes) |
| `data/raw/` | Raw ESA-AD or NASA telemetry | Empty directory |
| `data/processed/` | JSONL training data | Empty directory |
| `requirements.txt` | Python dependencies | Empty (0 bytes) |
| `notebooks/` | EDA notebooks | Empty directory |

### Git History

```
74c2d26 - thoughts/issues/space_telemetry_anom_llm_ISSUE.md - gemini exploratory thread
7938235 - Add space anomaly detection AI advice notes to thoughts/issues
7015d75 - Initial project scaffold
```

Three commits total - project initialized 2026-06-11.

## Code References

- `thoughts/issues/space_telemetry_anom_llm_ISSUE.md` - Main issue specification
- `thoughts/issues/space-anomaly-detection-ai-advice_exploratoryThread.md` - Research thread
- `thoughts/issues/Nina_FDL.png` - FDL interview context
- `src/etl/patch_telemetry.py` - Planned ETL (empty)
- `src/training/train.py` - Planned training script (empty)
- `src/training/config.yaml` - Planned config (empty)
- `src/inference/test_local_gguf.py` - Planned inference (empty)

## Architecture Documentation

### Planned Data Flow

```
Raw Telemetry (HDF5/CSV)
        │
        ▼
┌───────────────────┐
│ patch_telemetry.py│  Phase 1: ETL
│ - Rolling windows │
│ - JSONL format    │
└───────────────────┘
        │
        ▼
   JSONL Dataset
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────────┐
│ LSTM │  │ Unsloth  │  Phase 2 & 3: Training
│ etc. │  │ QLoRA    │
└──────┘  └──────────┘
   │         │
   ▼         ▼
Baseline   Fine-tuned
Model      LLM (GGUF)
   │         │
   └────┬────┘
        │
        ▼
┌───────────────────┐
│test_local_gguf.py │  Phase 4: Inference
│ - Compare metrics │
│ - Precision/Recall│
└───────────────────┘
```

### Expected JSONL Format (from exploratory thread)

```json
{
  "instruction": "Analyze the following telemetry patch for subsystem anomalies. If an anomaly is detected, provide the root cause and mitigation advice.",
  "input": "[tokenized/patched numerical telemetry window]",
  "output": "Nominal." 
}
```
or
```json
{
  "instruction": "Analyze the following telemetry patch...",
  "input": "[telemetry window]",
  "output": "Anomaly detected in thermal regulator. Effect: System auto-throttling. Action: Reroute redundant power."
}
```

## Related Research

- AnomLLM: `github.com/rose-stl-lab/anomllm`
- ESA-ADB Benchmark: `github.com/kplabs-pl/ESA-ADB`
- Telemanom: `github.com/khundman/telemanom`
- Unsloth Docs: `unsloth.ai/docs`
- NASA MSL/SMAP Dataset: `kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl`

## Open Questions

1. **Dataset Selection**: ESA-AD (~12GB) vs NASA MSL/SMAP - which better fits FDL's "space-based observing" focus?
2. **Model Selection**: Qwen2.5-7B vs Llama-3-8B - which has better time-series reasoning?
3. **Patching Strategy**: What window size and stride for telemetry patches?
4. **Advice Generation**: How to synthesize "diagnostic advice" labels for the training data?
5. **ML Baseline**: Which traditional model (LSTM, Autoencoder, Isolation Forest) to implement first?
