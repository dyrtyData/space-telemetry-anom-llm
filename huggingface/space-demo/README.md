---
title: STAR-Pipeline Telemetry Anomaly Advisor
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# STAR-Pipeline — Telemetry Anomaly Advisor (demo)

Interactive demo of a QLoRA-fine-tuned **Qwen3-8B** that reads a spacecraft-telemetry window and
returns an `ANOMALY`/`NOMINAL` verdict + structured `DIAGNOSIS`/`ADVICE`/`ACTION`.

**Hardware notes**
- **Free CPU basic** (default): works, but an 8B Q4_K_M model generates slowly (~10–40 s/analysis).
- **ZeroGPU** (needs HF PRO): set Space env var `N_GPU_LAYERS=-1` for interactive speed.
- To demo the **vision** model instead (most visually striking), rewrite `app.py` to load
  `dyrtyData/star-pipeline-qwen3-vl-8b-detection` via `transformers`+`peft` on ZeroGPU and accept a
  PNG telemetry-plot upload.

Model: [`dyrtyData/star-pipeline-qwen3-8b-advice-gguf`](https://huggingface.co/dyrtyData/star-pipeline-qwen3-8b-advice-gguf).
Project: STAR-Pipeline. Data: ESA-AD (CC BY 3.0 IGO).
