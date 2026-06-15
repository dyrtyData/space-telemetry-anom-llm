# Draft GitHub issue for rose-stl-lab/anomllm

**Where to post:** https://github.com/rose-stl-lab/anomllm/issues/new
**Why an issue first:** the repo has no CONTRIBUTING guide; opening a scoped issue to gauge
maintainer interest before a PR is the right etiquette (and references their open Issue #4).

---

**Title:** Proposal: add ESA-AD as a real-world dataset + Qwen3 (text & vision) model entries

**Body:**

Thanks for AnomLLM — the Affinity-F1-based evaluation and the synthetic anomaly taxonomy are a great
foundation. The paper notes the benchmark currently can't assess LLMs on *subtle real-world*
anomalies (it's evaluated on synthetic series). I've been working on exactly that gap and would like
to contribute it back if you're open to it.

**What I'd add**

1. **A real-world dataset: ESA-AD** (ESA Anomaly Dataset — real satellite telemetry, 3 missions,
   expert-annotated anomalies, CC BY 3.0 IGO; arXiv:2406.17826). I have a windowing + loader adapter
   that fits the existing time-series-plus-label schema, including the *subtle deviation* anomaly
   class that the synthetic suite doesn't cover. Affinity-F1 (your primary metric) is already what I
   report, so the evaluation plugs in directly.
2. **Two new model entries:** a QLoRA-fine-tuned `Qwen3-8B` (numeric-text input) and a
   `Qwen3-VL-8B` (rendered-plot / vision input, AnomSeer-style). Both are public on HF. This extends
   your model table with a *fine-tuned* (not just prompted) open-8B data point, which I think is a
   useful contrast to the GPT-4o-mini / Gemini / Qwen-VL prompted results.

**Headline finding (relevant to your thesis):** on ESA-AD, a frontier model prompted zero- and
few-shot sits ~at chance on context-free 10-value windows, while a fine-tuned 8B clears a trivial
always-anomaly baseline with balanced precision/recall — but a classical LSTM still out-detects both.
It's a clean real-world corroboration of "LLMs aren't yet competitive detectors, but fine-tuning
recovers domain priors prompting can't."

**Proposed scope (happy to split):**
- (A) dataset loader + ESA-AD adapter under your dataset format
- (B) the two Qwen3 model configs + an eval run, with a results table addition
- (C) separately, I can help with #4 (publishing a benchmark dataset to the HF Hub) if useful

Would a PR along these lines be welcome? If so, any preferences on dataset directory layout / config
conventions so I match your structure? Repo + full write-up: <STAR-Pipeline repo URL>.
