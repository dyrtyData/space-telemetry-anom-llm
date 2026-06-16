# STAR-Pipeline — Space Telemetry Anomaly Detection & Resolution

> An end-to-end, **open-source** pipeline that **detects anomalies** in real satellite telemetry and
> **generates human-readable diagnostic advice** — no external API vendor required.

Built on the **ESA Anomaly Dataset (ESA-AD)**, this project runs a like-for-like **bake-off of 15
approaches** to the same task: fine-tuned LLMs (Qwen3-8B text, Qwen3-VL-8B vision), classical
baselines (LSTM, Isolation Forest), un-fine-tuned bases with and without RAG, a frontier model with
and without RAG, a trivial baseline, and a **learned ensemble** that fuses the detectors' confidence
scores. The headline finding: **retrieval beats training for detection**. Base+RAG (F1 0.531) beats
fine-tuning (0.453) while requiring no training — and it's fully sovereign (local GGUF + local FAISS).
Frontier+RAG (F1 0.825) sets the ceiling. Fine-tuning earns its place for the **advice layer**, which
RAG cannot produce.

📄 **Full analysis:** [`thoughts/shared/research/2026-06-14-results-analysis.md`](thoughts/shared/research/2026-06-14-results-analysis.md)
🎓 **Learn it from scratch (plain-language):** [`thoughts/shared/research/2026-06-14-plain-language-walkthrough.md`](thoughts/shared/research/2026-06-14-plain-language-walkthrough.md)
📊 **Generated metrics report:** [`results/comparison_report.md`](results/comparison_report.md)

---

## TL;DR results

Final evaluation on the ESA-AD test material (LLM: 4,500 windows, **25.0% anomalous**; LSTM: all 58
Mission-1 channels; vision: 2,000 rendered PNGs). **Read every row against the trivial baseline.**

| Approach | Precision | Recall | F1 | CEF0.5† | Detects? | Advises? | Cost |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | ✅ | ❌ | ~instant |
| **LSTM baseline** (Telemanom-style, 58 ch) | **0.837** | 0.432 | **0.553** | **0.705** | ✅ | ❌ | ~instant |
| LLM detection — **text** (Qwen3-8B QLoRA→GGUF) | 0.360 ‡ | **0.609** | 0.453 | 0.392 | ✅ | ✅ | 2.77 s/window |
| LLM detection — **vision** (Qwen3-VL-8B, PNG plots) | 0.769 | 0.325 | 0.457 | 0.604 | ✅ | ❌ | 0.86 s/window |
| Base Qwen3-8B — zero-shot (no fine-tune) | 0 | 0 | 0 | 0 | ❌ | ❌ | — |
| Base Qwen3-8B — few-shot (2-ex, no fine-tune) | 0.282 | 0.824 | 0.420 | 0.325 | ⚠️ over-flags | ⚠️ 13% | 8.56 s/window |
| **Base Qwen3-8B + RAG (k=5)** ⬥ | 0.447 | 0.654 | **0.531** | 0.478 | ✅ | ❌ | retrieval + LLM |
| Base Qwen3-VL — zero-shot (no fine-tune) | 0.310 | 0.403 | 0.350 | 0.325 | ⚠️ ~chance | ❌ | — |
| Frontier (Claude) — zero-shot | 0.308 | 0.216 | 0.254 | 0.284 | ⚠️ ~chance | ✅ | — |
| Frontier (Claude) — few-shot | 0.200 | 0.297 | 0.239 | 0.214 | ⚠️ ~chance | ✅ | — |
| **Frontier (Claude) + RAG (k=5)** ⬥ | **1.000** | 0.703 | **0.825** | **0.922** | ✅ | ✅ | retrieval + API |
| **Always-anomaly (trivial baseline)** | 0.250 | 1.000 | 0.399 | 0.294 | ❌ | ❌ | free |
| **Ensemble** (text+vision+LSTM, stacked) ✦ | **0.922** | 0.486 | **0.636** | **0.781** | ✅ | ❌ | runs all 3 |
| **Hybrid** (ensemble detect → LLM advise) | **0.922** | 0.486 | **0.636** | **0.781** | ✅ | ✅ | LLM on flags |

† CEF0.5 = precision-weighted F-beta (β=0.5), the operationally relevant metric when false alarms are
costly (the ESA-benchmark-aligned score).
‡ The text LLM's **as-deployed** point; reading its confidence properly (deterministic decode + a
tuned threshold) lifts precision to **0.838** — the over-flagging is a calibration artifact, not a
capacity limit.
✦ The ensemble is scored on the **shared subset** where all three detectors have a score (1,378
Mission-1 windows), **not** the 4,500-window / 58-channel basis above — so it is not a like-for-like
swap; the claim is that on identical windows the fusion beats every single model.
⬥ **RAG rows (Phase 15)** use the same frozen sample as the frontier control for apples-to-apples
comparison. RAG retrieves k=5 labeled training neighbors per window, providing the channel history the
context-free models lacked. Base+RAG beats fine-tuning; Frontier+RAG is the best detector overall —
but reintroduces API dependence.

**The honest headline (five sentences):**
1. **For detection, retrieval beats training.** Base+RAG (F1 0.531) beats fine-tuning (0.453) — and
   it's also sovereign (local GGUF + local FAISS, no API). This inverts the original thesis: for
   detection, fine-tuning is not the best approach.
2. **Frontier+RAG (F1 0.825, P 1.000) sets the ceiling** — the target sovereign approaches should aim
   for. The gap from 0.531 to 0.825 shows model capability still matters once context is supplied.
3. **The ensemble (P 0.922) remains the strongest overall detector**, but it requires fine-tuned
   models. For maximum precision when training cost is acceptable, this is still the design.
4. **Fine-tuning earns its place for the advice layer, not detection.** The structured
   `DIAGNOSIS / ADVICE / ACTION` requires the fine-tune; Base+RAG produces detection only. Advice is
   graded **95% high-quality when the flag is correct**.
5. **The revised architecture:** `Base+RAG or LSTM (detection, no training) → fine-tuned LLM (advice
   on flags)`. Use retrieval for detection (better, cheaper), fine-tuning for explanation (the one
   thing RAG can't do).

---

## Why this project

The brief: a product needs **anomaly detection + end-user advice on the effect of a change** for a
mission-critical system, and the team wants to **own the model** rather than depend on an external
API. The open question — *can a fine-tuned open-source LLM do this, and is fine-tuning even worth it
over prompting an API?* — is answered **empirically** here.

The target capability is explicitly *"the ability to move beyond external API wrappers and
successfully fine-tune open-source foundation models for highly specific, localized business use
cases."* So the project demonstrates production-style ETL, classical-ML baselines, **LLM fine-tuning
via QLoRA/Unsloth** (text *and* vision), GGUF export, local Metal inference, and a rigorous,
honestly-reported evaluation — including the controls most fine-tuning demos skip.

---

## Where this sits in the field

This is not a new algorithm; it's a rigorous **application study** that runs the field's open-source
gold-standard pieces as one honest bake-off on real ESA data:

| Reference | Role here |
|---|---|
| **ESA-ADB** (`kplabs-pl/ESA-ADB`) | Source of the **dataset** and the **CEF / Affinity-F1** evaluation metrics (and the point-adjustment metric we deliberately avoid). |
| **Telemanom** (NASA JPL, KDD 2018) | The LSTM-reconstruction-error **method**, re-applied to ESA-AD as our strong baseline. |
| **AnomLLM** (ICLR 2025) | The "LLM-as-direct-detector" idea (text) — reproduced and its precision/recall trade-off confirmed. |
| **AnomSeer** (ICLR 2026) | The **vision** approach (LLM reads rendered plots) — reproduced as the Qwen3-VL detector. |
| **Time-LLM** (ICLR 2024) | Informed the **windowing/patching** ETL. |

**Contribution:** a like-for-like comparison across *four model families* and *three input modalities*
(plus a learned fusion of them), a defensible answer to *"did the fine-tuning actually help?"* —
and an uncomfortable finding: **for detection, retrieval beats training**. The transferable findings:
(1) **Base+RAG > fine-tuning for detection** (0.531 vs 0.453), and it's also sovereign; (2) an LLM
detector's **over-flagging can be a calibration artifact**; (3) **fusing detectors Pareto-beats every
one of them**; (4) fine-tuning's value shifts to the **advice layer** (the one thing RAG can't do);
and (5) **Frontier+RAG (0.825) sets the ceiling** sovereign approaches should aim for.

---

## Architecture

```
                  ESA-AD raw telemetry (3 missions, 224 channels, ~29 GB)
                                       │
          ┌────────────────────────────┴────────────────────────────┐
          │  Phase 1 — ETL (local, M3 Max)                           │
          │  resample 1h · RevIN normalize · 32-step rolling windows │
          │  balanced subsample · → 30,000 instruction/response JSONL│
          └────────────────────────────┬────────────────────────────┘
                                       │
   ┌──────────────┬────────────────────┼─────────────────────┬───────────────┐
   ▼              ▼                    ▼                     ▼               ▼
 Phase 2        Phase 3            Phase 8              Phase 1.5         Phase 6/12
 Baselines      LLM fine-tune      Vision fine-tune     Advice labels     Controls
 (local)        (cloud RTX 4090)   (cloud A6000)        (in-session)      (base/frontier/
 • LSTM 58 ch   • Qwen3-8B QLoRA   • Qwen3-VL-8B QLoRA  • 7,457 records    trivial + VL base)
 • Iso-Forest   • →GGUF Q4_K_M     • on PNG plots
        │              │                  │
        ▼              ▼                  ▼
   Phase 11       Phase 4            Phase 8 eval
   LSTM calib.    Local Metal        vision scoring ─────────────────────┐
   z=4.0 →        inference          (cloud A6000)                       │
   P=0.837        (M3 Max)                                               │
        │              │                  │                              │
        │              ▼                  │                              │
        │         Phase 13                │                              │
        │         Text-LLM calib.         │                              │
        │         PR curve, AUC 0.678     │                              │
        │         P 0.360→0.838           │                              │
        │              │                  │                              │
        └──────────────┴──────────────────┴──────────────────────────────┤
                                       │                                 │
                                       ▼                                 │
                        Phase 5/7/9 — Unified single-model evaluation    │
                        P / R / F1 / CEF0.5 / Affinity-F1                │
                        + semantic advice grading (120-flag sample)      │
                                       │                                 │
                                       ▼                                 │
                        Phase 14 — Ensemble (score-level fusion) ◄───────┘
                        leakage-free OOF stacker over continuous scores
                        text + vision + LSTM → P 0.922, CEF0.5 0.781
                        (the strongest detector)
                                       │
                                       ▼
                              → results/comparison_report.md

  Recommended production design (the Hybrid):
     window → LSTM screen (cheap, P=0.837) ──flag──► fused ensemble confirm (P=0.922)
              ──confirm──► Qwen3-8B advisor → operator-ready alert
```

---

## The phases

| Phase | Output |
|---|---|
| **1 — ETL** | ESA-AD → 30,000 windows (24.8% anomalous), split 21k/4.5k/4.5k, RevIN-normalized, 1h resample |
| **1.5 — Advice labels** | 7,457 structured `DIAGNOSIS/ADVICE/ACTION` records (in-session, no API) |
| **2 — Baselines** | LSTM (Telemanom-style) + Isolation Forest |
| **3 — Fine-tune (cloud)** | Qwen3-8B QLoRA (r=16, α=16, lr 2e-4, 3 ep) on Vast.ai RTX 4090 → GGUF Q4_K_M, ~$2.30 |
| **4 — Local inference** | GGUF on M3 Max via Metal; llama-cpp-python, all layers offloaded |
| **5 — Evaluation** | Full 4,500-window LLM run (hardened/resumable); unified comparison report |
| **6 — Did fine-tuning help?** | Base zero/few-shot, frontier zero/few-shot, trivial baseline — all on the identical harness |
| **7 — Level the field** | LSTM expanded to all 58 channels on contiguous timelines; real Affinity-F1 |
| **8 — Vision detector** | Qwen3-VL-8B on PNG plots (AnomSeer-style), F1 0.457, P 0.769, ~$1 |
| **9 — Advice grading** | 120-flag sample graded; 5.58/6 on true positives (95% high-quality) |
| **11 — LSTM calibration** | Tuned the decision threshold → canonical LSTM P 0.837 / CEF0.5 0.705 (Telemanom dynamic thresholding tried, found unsuitable) |
| **12 — Vision base control** | Un-fine-tuned Qwen3-VL zero-shot: F1 0.350 (below trivial) — completes the "did fine-tuning help?" table for vision |
| **13 — Text-LLM calibration** | Continuous verdict score + PR curve (AUC-PR 0.678); the over-flagging is a calibration artifact (precision 0.360 → 0.838) |
| **14 — Ensemble** | Leakage-free stacked fusion of text+vision+LSTM scores: **P 0.922 / CEF0.5 0.781** |
| **15 — RAG comparison** | Per-channel FAISS indices from 21k training windows; k=5 retrieval. **Frontier+RAG F1 0.825** (vs 0.254), **Base+RAG F1 0.531** (beats fine-tune 0.453). Validates the context hypothesis. |

See the [analysis doc](thoughts/shared/research/2026-06-14-results-analysis.md) for how each result
was produced.

---

## Dataset

[**ESA Anomaly Dataset (ESA-AD)**](https://doi.org/10.5281/zenodo.12528696) — the first large-scale,
real-life satellite telemetry benchmark with curated anomaly annotations, covering three ESA
missions. Chosen over NASA MSL/SMAP, whose Telemanom baseline is purpose-built and whose label
quality is widely criticized.

- **Splits** (balanced-subsampled, ~25% anomalous): train **21,000** · val **4,500** · test **4,500**
- **Windowing:** 32-step rolling windows, stride 16, resampled to 1h, RevIN per-channel reversible
  normalization
- Raw data is **not** committed (~29 GB); `make download` fetches it from the Kaggle mirror.

---

## Tech stack

| Layer | Choice |
|---|---|
| Base models | `unsloth/Qwen3-8B` (text) · `unsloth/Qwen3-VL-8B` (vision) |
| Fine-tuning | Unsloth + QLoRA (r=16, α=16, all-linear; text lr 2e-4 / 3 ep, vision lr 1e-4 / 2 ep) |
| Training compute | Vast.ai — RTX 4090 (text) + A6000 (vision, base control, vision scoring) — **~$4.2 total** |
| Quantization / export | GGUF `Q4_K_M` (Unsloth dynamic) |
| Local inference | `llama-cpp-python` with Metal GPU offload (M3 Max, 36 GB unified) |
| Baselines | Keras 3 (torch backend) LSTM · scikit-learn Isolation Forest |
| ETL / data | NumPy, pandas, h5py, matplotlib (plot rendering) |
| Tooling | `ruff`, `pytest`, `make` |

Published models (Hugging Face): text GGUF
[`dyrtyData/star-pipeline-qwen3-8b-advice-gguf`](https://huggingface.co/dyrtyData/star-pipeline-qwen3-8b-advice-gguf)
· vision adapter `dyrtyData/star-pipeline-qwen3-vl-8b-detection`.

---

## Reproducing the results

> **Prerequisites:** Python 3.11+, ~30 GB free disk for raw data, and for the LLM steps a downloaded
> GGUF. Training (Phases 3 & 8) runs on a rented NVIDIA GPU, not locally.

```bash
# 0. Environment
make setup                       # venv + editable install ([dev,lstm] extras)

# 1. ETL — download ESA-AD (Kaggle mirror) and build windows/JSONL
make download ESA_DATA_DIR="/path/to/esa-ad"
make etl      ESA_DATA_DIR="/path/to/esa-ad"
make validate-etl

# 1.5 Advice labels (already committed; regenerate if desired)
make advice && make validate-advice

# 2 / 7. Classical baselines  (MAX_CHANNELS=58 for the full Phase-7 sweep)
make baseline    MISSION=1 MAX_CHANNELS=58 ESA_DATA_DIR="/path/to/esa-ad"
make baseline-if ESA_DATA_DIR="/path/to/esa-ad"
make validate-baseline

# 3 / 8. LLM fine-tunes (on rented GPUs — see scripts/cloud/)
make format-train && make validate-format            # → ChatML JSONL
./scripts/cloud/launch_vast.sh                       # dry-run; --create to launch

# 4. Local GGUF inference (M3 Max, Metal)
make install-local
make eval-llm LIMIT=0 STAR_MODEL_DIR="/path/to/models"   # full 4,500-window run
make validate-inference

# 5–9. Unified comparison + controls + advice grading
make eval-base eval-base-fewshot                     # Phase 6 text base controls
make frontier-select frontier-assemble               # Phase 6 frontier control
make eval-vision                                     # Phase 8 vision row
make grade-advice-select grade-advice-assemble       # Phase 9 advice grade
make eval-all && make validate-eval                  # → results/comparison_report.md

# 11–14. Detection hardening (LSTM calib, vision base, text PR, ensemble)
make tune-threshold ESA_DATA_DIR="/path/to/esa-ad"   # Phase 11: LSTM z-threshold sweep
# Phase 12 (vision base control) was run on cloud A6000 — see scripts/cloud/
make eval-llm --score ...                            # Phase 13: text-LLM scored output
python src/inference/pr_curve.py --scored results/inference_test_scored.json \
       --out results/llm_pr_curve.json               #   → PR curve + AUC-PR 0.678
make lstm-window-scores ESA_DATA_DIR="/path/to/esa-ad"  # Phase 14: LSTM continuous scores
make eval-vision-score                               # Phase 14: vision continuous scores
make ensemble                                        # Phase 14: fused stacker → CEF0.5 0.781

# 15. RAG comparison (Phase 15)
make build-rag-index                                 # Build per-channel FAISS indices
make frontier-rag-prompts                            # Generate RAG-augmented prompts
# ... classify prompts in-session ...
make frontier-rag-assemble CLASSIFICATIONS=results/frontier_rag_classifications.json
make eval-base-rag LIMIT=100                         # Base+RAG smoke test

make eval-all                                        # regenerate comparison_report.md
```

### Durable long-running runs

The full LLM eval is a multi-hour job. `src/inference/test_local_gguf.py` supports
`--checkpoint-every N` and `--resume` (atomic checkpoint writes), so an interrupted run costs at most
one checkpoint. For an unattended run, keep the machine plugged in with the lid open (`caffeinate`
cannot override lid-close sleep on battery). This habit paid off when both the 4,500-window eval and
the 58-channel LSTM sweep died mid-run and resumed for free.

---

## Repository layout

```
src/
  etl/         download (Kaggle/Zenodo), RevIN windowing, advice-label + PNG plot generation
  baselines/   train_lstm.py · isolation_forest.py
  training/    format_for_unsloth.py · train_advice.py · train_detection.py · export_gguf.py  (cloud)
  inference/   test_local_gguf.py (Metal) · eval_vision.py · evaluate.py (unified report)
               select_frontier_sample.py · grade_advice_sample.py
scripts/cloud/ Vast.ai launch / data upload / model download helpers
results/       comparison_report.md · comparison_metrics.json · per-approach JSON
thoughts/      research, plan, full implementation log, results analysis, plain-language walkthrough
Makefile       every phase + a validate-* target encoding its success criteria
```

---

## Project journey & honest engineering log

This repo keeps its full paper trail — including the dead ends — because *how* the result was reached
is part of the showcase. The work logged **52 numbered deviations** (a wrong data-loader assumption,
a FAT32 4 GB wall, a trans-Atlantic transfer bottleneck solved via the HF CDN, a TRL API rewrite,
three latent bugs in never-run vision code, a two-failure eval-durability saga, Telemanom's dynamic
thresholding turning out to be unsuitable here, and Qwen3's `/no_think` directive needed in the system
prompt for RAG inference).

- **Plan:** [`thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`](thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md)
- **Implementation log (D1–D49):** [`thoughts/shared/implement/2026-06-12-star-pipeline-log.md`](thoughts/shared/implement/2026-06-12-star-pipeline-log.md)
- **Results analysis:** [`thoughts/shared/research/2026-06-14-results-analysis.md`](thoughts/shared/research/2026-06-14-results-analysis.md)
- **Plain-language walkthrough:** [`thoughts/shared/research/2026-06-14-plain-language-walkthrough.md`](thoughts/shared/research/2026-06-14-plain-language-walkthrough.md)

---

## Limitations

This is a showcase, not a deployed system. Residual gaps, stated up front:

1. **The ensemble is on a different evaluation unit** than the single detectors — the shared windows
   where all three have a score (1,378 Mission-1 windows), not the 4,500-window / 58-channel basis. Its
   Pareto-win claim is *internal* (on identical windows it beats each component), not a like-for-like
   master-table swap.
2. **Mission scope.** The LSTM (and so the 3-model ensemble) covers Mission 1 only — the other
   missions' LSTMs were never trained. The text LLM was evaluated across all three missions, a residual
   coverage asymmetry; a fully like-for-like rematch would score everything on one contiguous stream.
3. **Frontier control is a sample (n=150)**, fed deliberately context-free input — but Phase 15 tested
   **Frontier+RAG** on the same sample, which reached F1 0.825 with perfect precision, validating that
   context, not capability, was the bottleneck.
4. **Advice labels are synthetic** (in-session generated), and the advice grade is a 120-flag
   LLM-judged sample, not a human-SME panel — the top lever toward mission-critical advice.
5. **The Hybrid's detection score is inherited** from its detector; only its advice layer is new.
6. **Single fine-tune, no hyperparameter sweep**; the vision model has no advice head and converged
   fast (generalization to a new mission untested); **1-hour resampling is lossy** for sub-hour
   transients.

Each has a concrete next step in the
[analysis doc](thoughts/shared/research/2026-06-14-results-analysis.md#10-open-next-steps).

---

## Presentation materials

Generated via [NotebookLM](https://notebooklm.google.com/notebook/8bdf78b7-edfd-45c7-b39c-7418957184b5)
from the project documentation. **Ask questions interactively** via the chat, or explore:

| Material | Link | Local copy |
|----------|------|------------|
| **Video walkthrough** | [NotebookLM](https://notebooklm.google.com/notebook/8bdf78b7-edfd-45c7-b39c-7418957184b5/artifact/339a7f38-74f5-44a8-a1ae-4aff9d32c876) | — (65 MB, not in repo) |
| **Slide deck** | [NotebookLM](https://notebooklm.google.com/notebook/8bdf78b7-edfd-45c7-b39c-7418957184b5/artifact/a4632c10-05a7-4cf7-9536-c4350e5beb45) | [`thoughts/shared/notebookLM/STAR-Pipeline_Hybrid_Anomaly_Detection.pdf`](thoughts/shared/notebookLM/STAR-Pipeline_Hybrid_Anomaly_Detection.pdf) |
| **Infographic** | [NotebookLM](https://notebooklm.google.com/notebook/8bdf78b7-edfd-45c7-b39c-7418957184b5/artifact/3dc5fd18-2e32-4cb2-8992-997801528bfe) | [`thoughts/shared/notebookLM/Satellite_Anomaly_Detection_Using_LLMs.png`](thoughts/shared/notebookLM/Satellite_Anomaly_Detection_Using_LLMs.png) |
| **Mind map** | [NotebookLM](https://notebooklm.google.com/notebook/8bdf78b7-edfd-45c7-b39c-7418957184b5/artifact/32077d52-0340-40fe-bee5-5282a8624da9) | [`thoughts/shared/notebookLM/Mind Map.png`](thoughts/shared/notebookLM/Mind%20Map.png) |

---

## License & data

Code in this repository is released under the **MIT License** (see [`LICENSE`](LICENSE)). The
**ESA Anomaly Dataset** is governed by its own license — **CC BY 3.0 IGO** (attribution required,
commercial use permitted, no share-alike) — per its [Zenodo record](https://doi.org/10.5281/zenodo.12528696);
raw telemetry is **not** redistributed here. Any derived artifacts published elsewhere (e.g. the
fine-tuned models on Hugging Face) attribute ESA / the ESA-AD authors accordingly.
