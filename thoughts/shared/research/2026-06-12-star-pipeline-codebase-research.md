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

## Validated Recommendations (Post-Research)

### 1. Dataset Selection: ESA-AD (ESA-ADB)

**Recommendation**: Use ESA Anomaly Dataset as primary dataset.

**Reasoning**:
- **Benchmark credibility**: Only actively competed-on public benchmark (Kaggle competition May–Aug 2025)
- **Label quality**: Expert-annotated by ESA spacecraft operations engineers over 18 months
- **Scale**: 11.6 GB compressed, 224 real telemetry channels, 844 annotated events across 17.5 years
- **FDL alignment**: ESA partnership with FDL Europe signals domain awareness
- **Honest difficulty**: Paper concludes "no existing algorithm meets operational requirements" — compelling problem statement

**Do NOT use NASA MSL/SMAP as primary**:
- Documented quality issues: mislabeled ground truth, trivial anomalies
- "Multivariate" claim misleading: all channels except first are binary one-hot flags
- OPS-SAT paper (Nature 2025): "should not be used for time series anomaly detection benchmarking"

**Caveat on FDL alignment**: FDL's actual focus is heliophysics/solar observation (SDO, Parker Solar Probe) and Earth observation — not spacecraft health telemetry. Consider augmenting with SDO data if the role is heliophysics-focused.

**Sources**: [ESA-ADB Zenodo](https://zenodo.org/records/12528696), [arXiv 2406.17826](https://arxiv.org/abs/2406.17826), [Kaggle ESA-ADB Challenge](https://www.kaggle.com/competitions/esa-adb-challenge)

---

### 2. Model Selection: Qwen2.5-7B-Instruct

**Recommendation**: Use `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

**Reasoning**:
- **Numerical reasoning**: MATH benchmark 75.5% vs Llama-3.1-8B's 51.9% — 23.6-point gap
- **Structured output**: Explicitly trained on JSON extraction, more reliable formatting
- **GSM8K**: 91.6% vs Llama's 84.5%
- **Fine-tuning**: Unsloth provides pre-patched 4-bit variants with correct chat templates

**Alternative**: Qwen3-8B (April 2025) outperforms Qwen2.5 if starting fresh.

**Avoid Mistral-7B**: Community consensus it "is not as good as Qwen2.5 7B" on benchmarks.

**Sources**: [Qwen2.5-LLM Blog](https://qwenlm.github.io/blog/qwen2.5-llm/), [rankllms.com comparison](https://rankllms.com/compare/llama-3-1-8b-vs-qwen-2-5-7b/)

---

### 3. LLM vs Traditional Approach: Hybrid Architecture

**Critical Finding**: LLMs perform poorly on multivariate telemetry anomaly detection.

**Research evidence**:
- DeepSeek-V3 on aerospace telemetry: F1=0.47 (10% below TFMAE baseline)
- On multivariate out-of-loop data: "almost indistinguishable from random guessing"
- AnomLLM benchmark: Only tested on synthetic data, "no evidence they can understand more subtle real-world anomalies"

**Recommended Architecture** (adjusted from issue proposal):
```
Phase 1: ETL → Telemanom LSTM detects anomalies (proven F1~0.71)
Phase 2: Fine-tuned LLM generates diagnostic advice from LSTM output
```

This hybrid approach:
- Uses traditional ML where it excels (detection)
- Uses LLM where it excels (natural language explanation)
- Matches what the issue's friend actually asked for: "anomaly detection AND end user advice"

**If you must show LLM detection**: Fine-tune multimodal LLM (AnomSeer-style) on visual plots, not raw numbers. AnomSeer-7B achieves 84.4% F1 on AnomLLM benchmark — but only on synthetic data.

**Sources**: [arXiv 2601.12448](https://arxiv.org/html/2601.12448), [AnomSeer arXiv 2602.08868](https://arxiv.org/abs/2602.08868)

---

### 4. ML Baseline: Telemanom (LSTM + Dynamic Thresholding)

**Recommendation**: Implement Telemanom first.

**Reasoning**:
- Domain-specific, battle-tested on NASA datasets
- Apache 2.0, ready-to-run with SMAP/MSL data included
- Interpretable per-channel predictions (operators know which sensor triggered)
- No GPU required for basic version
- ESA used "Telemanom-ESA" as their baseline in ESA-ADB

**Honest benchmark numbers** (post-PA-correction):
- Telemanom: F1=0.713 on SMAP
- OmniAnomaly: F1=0.576 on SMD (PA-inflated numbers ~0.87)
- Published 90%+ F1 scores use "Point Adjustment" which inflates scores artificially

**Secondary baseline**: Isolation Forest (windowed) — 10 minutes to implement, gives non-temporal comparison.

**For root cause**: M2AD (2025) provides 81% top-1 sensor attribution accuracy.

**Implementation**: [github.com/khundman/telemanom](https://github.com/khundman/telemanom)

**Sources**: [Telemanom KDD 2018](https://ar5iv.labs.arxiv.org/html/1802.04431), [M2AD arXiv 2504.15225](https://arxiv.org/abs/2504.15225)

---

### 5. Patching Strategy for Time-Series Tokenization

**Recommendation**: Patch size 16-64 timesteps with stride = patch_size/2

**Best practices from research**:
1. **Apply RevIN** (Reversible Instance Normalization) before tokenization
2. **Patch, don't serialize**: Direct float serialization fragments into multi-tokens
3. **For frozen LLM**: Use cross-attention over text prototypes (Time-LLM approach)
4. **For fine-tuning**: Quantize to 512-4096 bins (Chronos approach) or AXIS-style integer serialization

**AXIS workaround** (if serializing numbers):
- Scale values, round to integers
- Use delimiter separation: "123, 124, 127"
- Minimizes token count while preserving ordering

**Sources**: [Time-LLM arXiv 2310.01728](https://arxiv.org/abs/2310.01728), [AXIS arXiv 2509.24378](https://arxiv.org/html/2509.24378v1)

---

### 6. Diagnostic Advice Labels

**Challenge**: ESA-AD and NASA datasets have anomaly labels but NO diagnostic advice text.

**Recommended approach**:
1. Use Telemanom to detect anomalies and identify which sensor/channel triggered
2. Create synthetic advice labels by:
   - Mapping channel IDs to subsystem names (thermal, power, attitude, etc.)
   - Generating templated advice: "Anomaly detected in {subsystem}. Effect: {effect}. Action: {action}"
   - Using LLM (Claude/GPT-4) to expand templates into natural language

**Alternative**: Use LLMAD's AnoCoT (Anomaly Detection Chain-of-Thought) prompting to generate explanations from a frozen LLM, then use those as training labels for the fine-tuned model.

**Sources**: [LLMAD KDD 2025 arXiv 2405.15370](https://arxiv.org/abs/2405.15370)

---

### 7. Unsloth Hyperparameters (Validated)

**Recommended configuration**:
```python
# LoRA config
r = 16                    # Bump to 32 if loss plateaus
lora_alpha = 16           # Equal to rank (Unsloth default)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
lora_dropout = 0          # Optimized for Unsloth kernels

# Training
learning_rate = 2e-4
num_train_epochs = 3
per_device_train_batch_size = 2
gradient_accumulation_steps = 8  # Effective batch = 16
warmup_ratio = 0.05
optim = "adamw_8bit"
use_gradient_checkpointing = "unsloth"  # Saves 30% VRAM
```

**GGUF export for M3 Max**: Use `q4_k_m` (4.1 GB) or Unsloth Dynamic 2.0 Q4 for best quality-to-speed tradeoff.

**Cloud GPU**: RTX 4090 (24GB) is sufficient and cost-effective for 7B QLoRA.

**Critical gotcha**: Qwen2.5 base model's chat tokens are untrained — always use instruct variant.

**Sources**: [Unsloth LoRA Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide), [Unsloth GGUF Docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)

---

### 8. Evaluation Metrics

**Critical warning**: ESA-ADB uses F0.5, not F1.

**Reasoning**: In operations, false alarms are very expensive (operator fatigue). F0.5 weights precision twice as heavily as recall.

**Do NOT use Point Adjustment (PA)**: A 2022 AAAI paper proved even random scorers achieve "SOTA" under PA. Use:
- **Affinity-F1** (temporally-aware) for LLM comparison
- **CEF0.5** (corrected event-wise F0.5) for ESA benchmark alignment
- **VUS-PR** (volume under surface, precision-recall) from TSB-AD

**Sources**: [AAAI 2022 PA critique arXiv 2109.05257](https://arxiv.org/abs/2109.05257), [ESA-ADB arXiv 2406.17826](https://arxiv.org/abs/2406.17826)

---

## Adjusted Architecture (Post-Research)

```
Raw Telemetry (ESA-AD HDF5/CSV)
        │
        ▼
┌───────────────────────┐
│ patch_telemetry.py    │  Phase 1: ETL
│ - RevIN normalize     │
│ - 16-64 step patches  │
│ - JSONL format        │
└───────────────────────┘
        │
        ▼
   Processed Dataset
        │
   ┌────┴────────────────┐
   │                     │
   ▼                     ▼
┌──────────────┐   ┌──────────────────┐
│ Telemanom    │   │ Qwen2.5-7B       │  Phase 2 & 3
│ LSTM-DT      │   │ QLoRA Fine-tune  │
│ (detection)  │   │ (advice gen)     │
└──────────────┘   └──────────────────┘
   │                     │
   ▼                     ▼
Channel-level        Diagnostic
Anomaly Flags        Advice Model
   │                     │
   └────────┬────────────┘
            │
            ▼
┌───────────────────────────┐
│ Integrated Pipeline       │  Phase 4: Inference
│ - LSTM detects anomaly    │
│ - LLM explains + advises  │
│ - CEF0.5 / Affinity-F1    │
└───────────────────────────┘
```

---

## Key Research Sources

### LLM Time-Series Anomaly Detection
- [AnomLLM (ICLR 2025)](https://arxiv.org/abs/2410.05440) - Benchmarking study showing LLM limitations
- [AnomSeer (ICLR 2026)](https://arxiv.org/abs/2602.08868) - Fine-tuned multimodal LLM achieving 84.4% on synthetic
- [LLM-TSAD (NeurIPS 2025)](https://openreview.net/forum?id=6rpy7X1Of8) - 66.6% F1 improvement via prompting
- [Time-LLM (ICLR 2024)](https://arxiv.org/abs/2310.01728) - Patching/reprogramming approach (forecasting, not detection)

### Datasets
- [ESA-ADB (2024)](https://arxiv.org/abs/2406.17826) - Primary recommendation
- [OPS-SAT (Nature 2025)](https://www.nature.com/articles/s41597-025-05035-3) - Small alternative
- [TSB-AD (NeurIPS 2024)](https://thedatumorg.github.io/TSB-AD/) - 1070 datasets, 40 algorithms

### Traditional ML
- [Telemanom (KDD 2018)](https://ar5iv.labs.arxiv.org/html/1802.04431) - LSTM baseline
- [M2AD (2025)](https://arxiv.org/abs/2504.15225) - Best root-cause attribution
- [STGLR (Sensors 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11769452/) - SOTA F1 with graph learning

### Unsloth
- [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [GGUF Export](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [Dynamic 2.0 Quantization](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
