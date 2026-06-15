---
license: mit
language:
- en
base_model:
- Qwen/Qwen3-VL-8B-Instruct
pipeline_tag: image-text-to-text
library_name: peft
tags:
- time-series
- anomaly-detection
- spacecraft-telemetry
- satellite
- vision-language-model
- qwen3-vl
- qlora
- unsloth
- multimodal
datasets:
- dyrtyData/esa-ad-star-splits
metrics:
- f1
- precision
- recall
model-index:
- name: star-pipeline-qwen3-vl-8b-detection
  results:
  - task:
      type: time-series-anomaly-detection
      name: Spacecraft Telemetry Anomaly Detection (vision, AnomSeer-style)
    dataset:
      name: ESA Anomaly Dataset (ESA-AD), STAR-Pipeline rendered PNG test split
      type: dyrtyData/esa-ad-star-splits
      split: test
    metrics:
    - type: f1
      value: 0.457
      name: Window-level F1
    - type: precision
      value: 0.769
    - type: recall
      value: 0.325
    - type: cef0.5
      value: 0.604
      name: CEF0.5 (precision-weighted F-beta)
    - type: accuracy
      value: 0.806
---

# STAR-Pipeline — Qwen3-VL-8B Telemetry Anomaly Detector (vision, LoRA)

A **QLoRA fine-tune of Qwen3-VL-8B** that detects spacecraft-telemetry anomalies by **looking at a
rendered PNG plot** of a telemetry window (the *AnomSeer-style* multimodal approach) and emitting
`ANOMALY`/`NOMINAL`. This repo contains the **LoRA adapter + processor** (apply on top of the base
VL model). It is a **pure detector** — it does not generate diagnostic advice.

## Results (ESA-AD STAR-Pipeline test split, 2,000 rendered PNGs)

| Metric | Value | Note |
|---|---|---|
| F1 | 0.457 | |
| **Precision** | **0.769** | precision-oriented |
| Recall | 0.325 | conservative (misses more) |
| **CEF0.5** | **0.604** | highest of any LLM approach in the project |
| Accuracy | 0.806 | |
| Format compliance | 100% | 0 unparseable responses |
| Latency | 0.86 s/window | A6000 |

**Why it's interesting:** it is the **precision-oriented mirror image** of the text model (which is
recall-oriented: P 0.360 / R 0.609) at nearly identical F1. It almost never false-alarms (49 FP vs
1,450 TN), which makes it attractive as a low-false-alarm **cross-check** in an ensemble. The two
modalities fail differently, so requiring *both* to fire is a natural high-confidence gate.

## Intended use & limitations

- **Direct use:** a second, modality-independent detection signal on rendered telemetry plots; an
  ensemble cross-check on a recall-oriented detector.
- **Out of scope:** generating advice (no advice head); non-plot inputs; spacecraft/missions far
  from the ESA-AD distribution.
- **Limitations:** trained on 3 ESA missions; converged very fast (eval loss 0.0089 on a 2-class
  task) so generalization to unseen missions is untested; recall is low by design (precision-biased).

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct` (4-bit QLoRA via Unsloth) |
| Method | QLoRA — r=16, α=16, all-linear, dropout 0 |
| Optimizer / LR | lr 1e-4, warmup 10%, batch 1 × grad-accum 16, 2 epochs |
| Data | 2,000 rendered PNG telemetry-window plots (ESA-AD derived) |
| Loss | 4.48 → 0.34 (eval 0.0089) |
| Compute | 1× NVIDIA A6000 48 GB (Vast.ai), ~65 min, ~$1.00 |

## Usage (transformers + peft)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from PIL import Image

base = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "dyrtyData/star-pipeline-qwen3-vl-8b-detection")
processor = AutoProcessor.from_pretrained("dyrtyData/star-pipeline-qwen3-vl-8b-detection")

img = Image.open("telemetry_window.png")
messages = [{"role": "user", "content": [
    {"type": "image"}, {"type": "text", "text": "Analyze this telemetry plot. ANOMALY or NOMINAL?"}]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, return_tensors="pt", images=[img]).to(model.device)
print(processor.decode(model.generate(**inputs, max_new_tokens=16)[0], skip_special_tokens=True))
```

## Citation & data attribution

Trained on the **ESA Anomaly Dataset (ESA-AD)** — Kotowski et al., 2024
([arXiv:2406.17826](https://arxiv.org/abs/2406.17826), [Zenodo 10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696)),
**CC BY 3.0 IGO** (© ESA / Airbus / KP Labs). Approach follows the AnomSeer line of multimodal
LLM anomaly detection. Adapter released under MIT.
