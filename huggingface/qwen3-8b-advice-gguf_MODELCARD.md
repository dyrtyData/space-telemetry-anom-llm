---
license: mit
language:
- en
base_model:
- Qwen/Qwen3-8B
pipeline_tag: text-generation
library_name: gguf
tags:
- time-series
- anomaly-detection
- spacecraft-telemetry
- satellite
- qwen3
- qlora
- unsloth
- gguf
- llama-cpp
datasets:
- dyrtyData/esa-ad-star-splits
metrics:
- f1
- precision
- recall
model-index:
- name: star-pipeline-qwen3-8b-advice-gguf
  results:
  - task:
      type: time-series-anomaly-detection
      name: Spacecraft Telemetry Anomaly Detection + Advice
    dataset:
      name: ESA Anomaly Dataset (ESA-AD), STAR-Pipeline test split
      type: dyrtyData/esa-ad-star-splits
      split: test
    metrics:
    - type: f1
      value: 0.453
      name: Window-level F1
    - type: precision
      value: 0.360
    - type: recall
      value: 0.609
    - type: cef0.5
      value: 0.392
      name: CEF0.5 (precision-weighted F-beta)
    - type: accuracy
      value: 0.632
---

# STAR-Pipeline — Qwen3-8B Telemetry Anomaly Advisor (GGUF)

A **QLoRA fine-tune of Qwen3-8B** that reads a window of spacecraft telemetry (as text) and emits
an `ANOMALY`/`NOMINAL` verdict **plus** structured diagnostic advice
(`DIAGNOSIS` / `ADVICE` / `ACTION`). Exported to **GGUF `Q4_K_M`** for local CPU/Metal inference —
no external API required.

> **TL;DR for deployment:** this model is strongest as the **advisor** in a two-stage *hybrid*
> (a high-precision detector flags an anomaly → this model explains it), **not** as a standalone
> detector. On windows correctly flagged as anomalous, its advice scores **5.58/6 (95%
> high-quality)**; as a standalone detector it over-flags (precision 0.36). See *Intended use*.

## Results (ESA-AD STAR-Pipeline test split, 4,500 windows, 25% anomalous)

| Metric | Value | Note |
|---|---|---|
| F1 (window-level) | **0.453** | balanced P/R |
| Precision | 0.360 | |
| Recall | 0.609 | |
| CEF0.5 | 0.392 | precision-weighted (ESA-aligned) |
| Accuracy | 0.632 | |
| Output-format compliance | 99.4% | emits parseable verdict |
| Structured advice on flags | 99.6% | DIAGNOSIS/ADVICE/ACTION present |
| Advice quality on true positives | **5.58 / 6** | 95% high-quality, 100% grounded |
| Latency | 2.77 s/window | M3 Max, Metal, all layers offloaded |

**How to read these:** the fine-tune is the **only** LLM-family approach (vs un-fine-tuned base
zero/few-shot and a frontier model) that beats a trivial *always-anomaly* baseline (F1 0.399) with a
*balanced* precision/recall — i.e. it learned mission/channel-specific priors that prompting cannot
supply. A classical LSTM still out-*detects* it as a single model (F1 0.553, P 0.837) — though
reading this model's confidence properly calibrates its precision from 0.360 to 0.838, and a
text+vision+LSTM ensemble beats every single detector (P 0.922). This model's unique value remains the
advice layer. Full bake-off: see the project repo.

## Intended use & limitations

- **Direct use:** generate human-readable diagnostic advice for a flagged telemetry window in a
  monitoring/FDIR assistant.
- **Recommended architecture (hybrid):** `window → cheap high-precision detector (e.g. LSTM) → if
  flagged → this model writes the advice → operator`. Advice quality is *gated by detection
  precision*; on false alarms the advice is built on a false premise (~1/6).
- **Out of scope:** standalone safety-critical anomaly triggering (precision 0.36 → alert fatigue);
  non-space telemetry without domain adaptation; real-time per-window screening at scale (2.77 s/window).
- **Limitations:** trained on 3 ESA missions (generalization to other spacecraft untested);
  advice labels were synthetically generated; single fine-tune (no hyperparameter sweep); stored
  advice was truncated at 300 chars during eval.

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-8B` (via `unsloth/Qwen3-8B-bnb-4bit`, 4-bit QLoRA) |
| Method | Unsloth + QLoRA — r=16, α=16, target = all-linear, dropout 0 |
| Optimizer / LR | adamw_8bit, lr 2e-4, cosine, warmup 5%, 3 epochs |
| Data | 21,000 ChatML instruction/response windows (ESA-AD derived) |
| Loss | 2.85 → 0.24 (eval ≈ 0.256, stable) |
| Compute | 1× NVIDIA RTX 4090 (Vast.ai), ~4.5 h, ~$2.30 |
| Export | GGUF `Q4_K_M` (Unsloth dynamic), ~4.7 GB |

## Usage (llama-cpp-python, Metal/CPU)

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="dyrtyData/star-pipeline-qwen3-8b-advice-gguf",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1,   # all layers to Metal GPU; use 0 for CPU-only
    n_ctx=2048,
)
SYSTEM = "You are a spacecraft telemetry analyst. Classify the window as ANOMALY or NOMINAL; if ANOMALY, give DIAGNOSIS/ADVICE/ACTION."
window = "mission=Mission1 channel=channel_41 values=[0.12, 0.14, ...]"  # normalized window
out = llm.create_chat_completion(messages=[
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": window},
], max_tokens=300)
print(out["choices"][0]["message"]["content"])
```

## Citation & data attribution

Trained on the **ESA Anomaly Dataset (ESA-AD)** — Kotowski et al., 2024
([arXiv:2406.17826](https://arxiv.org/abs/2406.17826), [Zenodo 10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696)),
licensed **CC BY 3.0 IGO** (© ESA / Airbus / KP Labs). This fine-tune and its outputs are released
under MIT. Project repository: STAR-Pipeline (see model card metadata).
