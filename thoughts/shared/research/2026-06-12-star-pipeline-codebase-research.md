---
date: 2026-06-12T16:04:27Z
researcher: Claude Code
git_commit: 50183c8
branch: main
repository: space-telemetry-anom-llm
topic: "STAR-Pipeline Research - Validated Architecture & Implementation Guide"
tags: [research, codebase, space-telemetry, anomaly-detection, llm-finetuning, star-pipeline, unsloth, qwen3]
status: complete
last_updated: 2026-06-12
last_updated_by: Claude Code
---

# STAR-Pipeline Research: Validated Architecture & Implementation Guide

**Purpose**: This document provides the validated research findings and recommended architecture for the Space Telemetry Anomaly Detection & Resolution Pipeline (STAR-Pipeline). It is ready for the `/create_plan` workflow.

**Target**: FDL (Frontier Development Lab) AI Engineering interview showcase demonstrating fine-tuning capability for mission-critical infrastructure.

---

## Executive Summary

This project demonstrates the ability to fine-tune open-source LLMs for a localized business use case: spacecraft telemetry anomaly detection with diagnostic advice generation. The architecture compares **three approaches**:

1. **LSTM Baseline** (Telemanom) - Proven traditional ML approach (F1~0.71)
2. **LLM Detection** (AnomSeer-style) - Fine-tuned multimodal LLM on visual telemetry plots
3. **Hybrid** - LSTM detection + LLM diagnostic advice generation

**Key Value Proposition**: No external API dependency, data never leaves infrastructure, customized to specific telemetry patterns, demonstrates fine-tuning skill.

---

## Final Architecture (Three-Way Comparison)

```
ESA-AD Raw Telemetry (11.6 GB, 224 channels)
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                    PHASE 1: ETL                     │
│  patch_telemetry.py                                 │
│  - Download ESA-AD from Zenodo                      │
│  - RevIN normalize per channel                      │
│  - Create 32-step rolling windows (stride 16)       │
│  - Output: JSONL (text) + PNG plots (visual)        │
└─────────────────────────────────────────────────────┘
                │
    ┌───────────┼───────────────────────┐
    │           │                       │
    ▼           ▼                       ▼
┌─────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ APPROACH│ │ APPROACH 2:     │ │ APPROACH 3:         │
│ 1: LSTM │ │ LLM Detection   │ │ Hybrid              │
│         │ │ (AnomSeer-style)│ │ (LSTM + LLM Advice) │
├─────────┤ ├─────────────────┤ ├─────────────────────┤
│Telemanom│ │ Qwen3-8B VL     │ │ Telemanom detection │
│LSTM-DT  │ │ fine-tuned on   │ │ + Qwen3-8B advice   │
│per-chan │ │ PNG telemetry   │ │ generation          │
│         │ │ plots           │ │                     │
└────┬────┘ └────────┬────────┘ └──────────┬──────────┘
     │               │                      │
     ▼               ▼                      ▼
┌─────────────────────────────────────────────────────┐
│                 PHASE 4: EVALUATION                 │
│  - CEF0.5 (ESA benchmark metric)                    │
│  - Affinity-F1 (temporally-aware)                   │
│  - Compare: detection accuracy + advice quality     │
│  - Local inference on M3 Max (GGUF)                 │
└─────────────────────────────────────────────────────┘
```

---

## Validated Decisions

### 1. Dataset: ESA-AD (ESA-ADB)

| Attribute | Value |
|-----------|-------|
| Size | 11.6 GB compressed (3 missions) |
| Channels | 224 telemetry + 821 control signals |
| Events | 844 annotated, 148 anomalies |
| Format | Pickle/CSV |
| Source | [Zenodo](https://zenodo.org/records/12528696) |
| Benchmark | [Kaggle ESA-ADB Challenge](https://www.kaggle.com/competitions/esa-adb-challenge) |

**Why**: Expert-annotated by ESA engineers, actively benchmarked, ESA partnership with FDL Europe. NASA MSL/SMAP has documented quality issues and should NOT be used.

### 2. Model: Qwen3-8B (Not Qwen2.5)

| Attribute | Value |
|-----------|-------|
| Model ID | `unsloth/Qwen3-8B-bnb-4bit` |
| MATH score | Superior to Qwen2.5 (which scores 75.5% vs Llama's 51.9%) |
| Instruct variant | Yes - avoids untrained chat token issues |
| Vision variant | `unsloth/Qwen3-VL-8B` for AnomSeer approach |

**Why**: Qwen3 outperforms Qwen2.5, fully supported by Unsloth, use Instruct variant to avoid token issues.

### 3. GGUF Export: Unsloth Dynamic 2.0 Q4

**Use**: `quantization_method="dynamic"` or Dynamic 2.0 Q4

**Why**: Same 4.1 GB file size as q4_k_m but better perplexity (closer to Q5). Unsloth-specific optimization.

### 4. Cloud GPU: Vast.ai with RTX 4090

| Platform | Recommendation |
|----------|----------------|
| Vast.ai | Preferred - official `vastai/unsloth-studio` image |
| RunPod | No official template, use `unsloth/unsloth` Docker image |
| GPU | RTX 4090 (24GB VRAM) - sufficient for 8B QLoRA |

### 5. Evaluation Metrics

| Metric | Use For |
|--------|---------|
| CEF0.5 | ESA benchmark alignment (precision-weighted) |
| Affinity-F1 | LLM comparison (temporally-aware) |
| VUS-PR | TSB-AD benchmark compatibility |

**Warning**: Do NOT use Point Adjustment (PA) - it inflates scores artificially.

---

## Recommended Project Structure

```
space-telemetry-anom-llm/
├── README.md
├── requirements.txt
├── pyproject.toml                    # NEW: Python project config
├── Makefile                          # NEW: Automation commands
│
├── config/
│   └── unsloth-train.yaml            # NEW: Unsloth YAML config
│
├── data/
│   ├── raw/                          # ESA-AD downloads
│   ├── processed/
│   │   ├── jsonl/                    # Text-based patches
│   │   └── plots/                    # PNG telemetry plots (for AnomSeer)
│   └── splits/                       # Train/val/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_baseline_lstm.ipynb        # Telemanom baseline
│   └── 03_llm_detection.ipynb        # AnomSeer-style training
│
├── src/
│   ├── etl/
│   │   ├── download_esa.py           # NEW: Download ESA-AD from Zenodo
│   │   ├── patch_telemetry.py        # Rolling window creation
│   │   └── generate_plots.py         # NEW: PNG plot generation for VL model
│   │
│   ├── baselines/
│   │   ├── telemanom/                # NEW: LSTM baseline (fork/adapt)
│   │   └── isolation_forest.py       # NEW: Simple IF baseline
│   │
│   ├── training/
│   │   ├── config.yaml               # Unsloth training config
│   │   ├── train_advice.py           # Fine-tune for advice generation
│   │   └── train_detection.py        # NEW: Fine-tune for detection (AnomSeer)
│   │
│   └── inference/
│       ├── test_local_gguf.py        # M3 Max inference
│       └── evaluate.py               # NEW: CEF0.5 / Affinity-F1 metrics
│
├── models/                           # NEW: Trained model outputs
│   ├── lora/                         # LoRA adapters
│   ├── merged/                       # Merged 16-bit
│   └── gguf/                         # GGUF exports
│
└── thoughts/
    ├── issues/                       # Issue documentation
    └── shared/
        ├── research/                 # This document
        └── plans/                    # Implementation plans
```

---

## Automation & CLI Resources

### Unsloth CLI Commands

```bash
# Install (on cloud GPU)
curl -fsSL https://unsloth.ai/install.sh | sh

# Train from YAML config
unsloth train --config config/unsloth-train.yaml

# Export to GGUF
unsloth export --model ./models/lora --output ./models/gguf --quantization dynamic

# Launch Studio UI (for manual experimentation)
unsloth studio -H 0.0.0.0 -p 8000
```

### Vast.ai CLI Automation

```bash
# Install
pip install vastai

# Set API key
vastai set api-key YOUR_API_KEY

# Search for RTX 4090 instances
vastai search offers 'gpu_name=RTX_4090 reliability>0.95' --order 'dph_total asc'

# Create instance with Unsloth Studio
vastai create instance OFFER_ID \
  --image vastai/unsloth-studio:2026.6.3-cuda-12.9-py312 \
  --disk 50 \
  --env '-p 8000:8000 -p 8888:8888' \
  --onstart-cmd "unsloth studio -H 0.0.0.0 -p 8000" \
  --jupyter-lab
```

### Unsloth YAML Config Template

```yaml
# config/unsloth-train.yaml
model:
  name: "unsloth/Qwen3-8B-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  rank: 16
  alpha: 16
  dropout: 0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

dataset:
  train_file: "./data/processed/jsonl/train.jsonl"
  eval_file: "./data/processed/jsonl/eval.jsonl"

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_epochs: 3
  warmup_ratio: 0.05
  optim: "adamw_8bit"

output:
  lora_dir: "./models/lora"
  gguf_dir: "./models/gguf"

export:
  gguf: true
  quantizations: ["dynamic"]
```

### Unsloth GitHub Resources

| Resource | URL | Use |
|----------|-----|-----|
| Main repo | https://github.com/unslothai/unsloth | CLI, core library |
| Notebooks | https://github.com/unslothai/notebooks | Training templates |
| Qwen3 notebook | `Qwen3_(14B)-Reasoning-Conversational.ipynb` | Best starting template |
| Docker image | `unsloth/unsloth` | Self-hosted training |
| Vast.ai image | `vastai/unsloth-studio:2026.6.3-cuda-12.9-py312` | Cloud training |

---

## Implementation Notes for Planning Phase

### Phase 1: ETL (Local - M3 Max)

1. Download ESA-AD from Zenodo (~12GB)
2. Parse Pickle/CSV files into unified format
3. Apply RevIN normalization per channel
4. Create rolling windows (32 steps, stride 16)
5. Generate two outputs:
   - JSONL for text-based approach
   - PNG plots for AnomSeer visual approach
6. Create train/val/test splits (70/15/15)

### Phase 2: Baselines (Local)

1. **Telemanom LSTM**:
   - Fork/adapt from https://github.com/khundman/telemanom
   - Train per-channel LSTMs
   - Dynamic thresholding for anomaly flagging
   - Expected F1 ~0.71

2. **Isolation Forest** (quick comparison):
   - Windowed features (flatten 32-step window)
   - scikit-learn implementation
   - 10 minutes to implement

### Phase 3: LLM Fine-tuning (Cloud - Vast.ai)

1. **Advice Generation** (Qwen3-8B text):
   - Input: telemetry patch + anomaly flag
   - Output: diagnostic advice text
   - Use Claude/GPT-4 to generate training labels
   
2. **Detection** (Qwen3-VL-8B vision - AnomSeer style):
   - Input: PNG plot of telemetry window
   - Output: anomaly classification + explanation
   - Fine-tune with TimerPO or standard SFT

### Phase 4: Evaluation & Integration (Local)

1. Export models to GGUF (Dynamic 2.0)
2. Pull to M3 Max
3. Evaluate all three approaches:
   - LSTM baseline
   - LLM detection
   - Hybrid (LSTM + LLM advice)
4. Report CEF0.5, Affinity-F1, qualitative advice quality

---

## Hardware Requirements

| Phase | Hardware | Duration |
|-------|----------|----------|
| ETL | M3 Max (local) | ~1 hour |
| LSTM Baseline | M3 Max (local) | ~30 min/channel |
| LLM Fine-tuning | Vast.ai RTX 4090 | ~2-4 hours |
| Inference | M3 Max (local) | Real-time |

**Estimated cloud cost**: $5-15 total (RTX 4090 @ $0.30-0.80/hr)

---

## Key Research Sources

### LLM Time-Series Anomaly Detection
- [AnomLLM (ICLR 2025)](https://arxiv.org/abs/2410.05440) - Benchmarking study
- [AnomSeer (ICLR 2026)](https://arxiv.org/abs/2602.08868) - Fine-tuned multimodal LLM (84.4% F1 on synthetic)
- [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728) - Patching/reprogramming approach

### Datasets & Benchmarks
- [ESA-ADB (2024)](https://arxiv.org/abs/2406.17826) - Primary dataset
- [TSB-AD (NeurIPS 2024)](https://thedatumorg.github.io/TSB-AD/) - Benchmark framework

### Traditional ML
- [Telemanom (KDD 2018)](https://ar5iv.labs.arxiv.org/html/1802.04431) - LSTM baseline
- [M2AD (2025)](https://arxiv.org/abs/2504.15225) - Root-cause attribution

### Unsloth
- [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Qwen3 Tutorial](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune)
- [GGUF Export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)

---

## Instructions for Automation

**Note for implementation**: If anything can be set up with CLI, API, or MCP so that Claude Code can automate the process, do so. For manual steps that require browser interaction (e.g., Vast.ai web console, Kaggle dataset download), provide detailed step-by-step instructions for chrome-devtools or Cursor Browser Use.

### Automatable via CLI:
- Vast.ai instance creation (`vastai` CLI)
- Unsloth training (`unsloth train`)
- GGUF export (`unsloth export`)
- Git operations
- Python script execution

### Requires Manual/Browser Steps:
1. **Kaggle dataset download** - requires Kaggle API key setup
2. **Vast.ai account setup** - initial registration
3. **HuggingFace model upload** - if publishing results

For any manual steps, provide chrome-devtools or step-by-step instructions explaining concepts and reasoning.

---

## Open Items Resolved

All open questions from the initial research have been resolved:

| Question | Resolution |
|----------|------------|
| Dataset selection | ESA-AD (not NASA MSL/SMAP) |
| Model selection | Qwen3-8B (not Qwen2.5) |
| Architecture | Three-way comparison (LSTM, LLM detection, Hybrid) |
| GGUF quantization | Dynamic 2.0 |
| Cloud platform | Vast.ai preferred |
| Evaluation metrics | CEF0.5 + Affinity-F1 |

**This research document is ready for `/create_plan`.**
