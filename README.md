# STAR-Pipeline — Space Telemetry Anomaly Detection & Resolution

> An end-to-end, **open-source** pipeline that **detects anomalies** in real satellite telemetry and
> **generates human-readable diagnostic advice** — no external API vendor required.

Built on the **ESA Anomaly Dataset (ESA-AD)**, this project runs a like-for-like **bake-off of ten
approaches** to the same task and fine-tunes two open-source LLMs (Qwen3-8B text, Qwen3-VL-8B
vision) with QLoRA. It deliberately pits the *research-frontier* "LLM does everything" idea against
*classical* baselines **and** against the un-fine-tuned base model, a frontier model, and a trivial
baseline — then reports the result honestly and lands on the architecture the data actually
supports.

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
| **LSTM baseline** (Telemanom-style, 58 ch) | **0.785** | 0.451 | **0.552** | **0.684** | ✅ | ❌ | ~instant |
| LLM detection — **text** (Qwen3-8B QLoRA→GGUF) | 0.360 | **0.609** | 0.453 | 0.392 | ✅ | ✅ | 2.77 s/window |
| LLM detection — **vision** (Qwen3-VL-8B, PNG plots) | 0.769 | 0.325 | 0.457 | 0.604 | ✅ | ❌ | 0.86 s/window |
| Base Qwen3-8B — zero-shot (no fine-tune) | 0 | 0 | 0 | 0 | ❌ | ❌ | — |
| Base Qwen3-8B — few-shot (2-ex, no fine-tune) | 0.282 | 0.824 | 0.420 | 0.325 | ⚠️ over-flags | ⚠️ 13% | 8.56 s/window |
| Frontier (Claude) — zero-shot | 0.308 | 0.216 | 0.254 | 0.284 | ⚠️ ~chance | ✅ | — |
| Frontier (Claude) — few-shot | 0.200 | 0.297 | 0.239 | 0.214 | ⚠️ ~chance | ✅ | — |
| **Always-anomaly (trivial baseline)** | 0.250 | 1.000 | 0.399 | 0.294 | ❌ | ❌ | free |
| **Hybrid** (LSTM detect → LLM advise) | **0.785** | 0.451 | **0.552** | **0.684** | ✅ | ✅ | LLM only on flags |

† CEF0.5 = precision-weighted F-beta (β=0.5), the operationally relevant metric when false alarms
are costly (the ESA-benchmark-aligned score).

**The honest headline (three sentences):**
1. As a pure **detector**, the tuned **LSTM wins** (F1 0.552, 2.2× the precision of the direct
   text-LLM, best CEF0.5, real Affinity-F1 0.649) — matching the published literature.
2. Yet **fine-tuning is justified and survives a skeptic**: read against the trivial *always-anomaly*
   baseline (free F1 0.399), the fine-tuned LLM is the **only** LLM-family approach that clears it
   with a *balanced* precision/recall — a few-shot base "matches" its F1 only by over-flagging ~80%
   of windows, and a frontier model prompted two ways sits **at chance** on this context-free input.
3. The LLM's unique value is **advice**: graded **95% high-quality when the flag is correct** — but
   gated by detection precision — so the design the data recommends is the **hybrid** (a cheap
   high-precision detector triggers an LLM advisor).

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

**Contribution:** a like-for-like comparison across *four model families* and *three input
modalities*, plus a defensible answer to *"did the fine-tuning actually help?"* framed against a
trivial baseline — with the transferable finding that **direct LLM detection is signal-limited on
context-free input**, while a fine-tuned 8B model beats both a few-shot base and a frontier model by
learning localized priors prompting can't supply.

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
 Phase 2        Phase 3            Phase 8              Phase 1.5         Phase 6
 Baselines      LLM fine-tune      Vision fine-tune     Advice labels     Controls
 (local)        (cloud RTX 4090)   (cloud A6000)        (in-session)      (base/frontier/
 • LSTM 58 ch   • Qwen3-8B QLoRA   • Qwen3-VL-8B QLoRA  • 7,457 records    trivial)
 • Iso-Forest   • →GGUF Q4_K_M     • on PNG plots
        │              │                  │
        │              ▼                  │
        │       Phase 4 — Local Metal inference (M3 Max, llama-cpp-python)
        │              │                  │
        └──────────────┴──────────────────┴─────────────► Phase 5/7/9 — Unified evaluation
                                                            P / R / F1 / CEF0.5 / Affinity-F1
                                                            + semantic advice grading
                                                            → results/comparison_report.md

  Recommended production design (the Hybrid):
     window → LSTM detector (cheap, P=0.785) ──flag──► Qwen3-8B advisor → operator-ready alert
              (optional: Qwen3-VL vision model as a low-false-alarm cross-check)
```

---

## The nine phases

| Phase | Output |
|---|---|
| **1 — ETL** | ESA-AD → 30,000 windows (24.8% anomalous), split 21k/4.5k/4.5k, RevIN-normalized, 1h resample |
| **1.5 — Advice labels** | 7,457 structured `DIAGNOSIS/ADVICE/ACTION` records (in-session, no API) |
| **2 — Baselines** | LSTM (Telemanom-style) + Isolation Forest (3-channel smoke) |
| **3 — Fine-tune (cloud)** | Qwen3-8B QLoRA (r=16, α=16, lr 2e-4, 3 ep) on Vast.ai RTX 4090 → GGUF Q4_K_M, ~$2.30 |
| **4 — Local inference** | GGUF on M3 Max via Metal; llama-cpp-python, all layers offloaded |
| **5 — Evaluation** | Full 4,500-window LLM run (hardened/resumable); unified comparison report |
| **6 — Did fine-tuning help?** | Base zero/few-shot, frontier zero/few-shot, trivial baseline — all on the identical harness |
| **7 — Level the field** | LSTM expanded to all 58 channels; honest F1 0.552; real Affinity-F1 0.649 |
| **8 — Vision detector** | Qwen3-VL-8B on PNG plots (AnomSeer-style), F1 0.457, P 0.769, ~$1 |
| **9 — Advice grading** | 120-flag sample graded; 5.58/6 on true positives (95% high-quality) |

> Phases 6–9 were added specifically to close the four methodological gaps the Phase-5 review
> flagged (no base-vs-fine-tuned comparison, 3-channel-only LSTM, vision detector unrun, advice
> graded only for shape). See the [analysis doc](thoughts/shared/research/2026-06-14-results-analysis.md).

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
| Training compute | Vast.ai — RTX 4090 (text) + A6000 (vision) — **~$3.33 total** |
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
make eval-base eval-base-fewshot                     # Phase 6 base controls
make frontier-select frontier-assemble               # Phase 6 frontier control
make eval-vision                                     # Phase 8 vision row
make grade-advice-select grade-advice-assemble       # Phase 9 advice grade
make eval-all && make validate-eval                  # → results/comparison_report.md
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
is part of the showcase. The work ran across ~87 sessions over three days and logged **38 numbered
deviations** (a wrong data-loader assumption, a FAT32 4 GB wall, a trans-Atlantic transfer
bottleneck solved via the HF CDN, a TRL API rewrite, three latent bugs in never-run vision code, and
a two-failure eval-durability saga).

- **Plan:** [`thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md`](thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md)
- **Implementation log (D1–D38):** [`thoughts/shared/implement/2026-06-12-star-pipeline-log.md`](thoughts/shared/implement/2026-06-12-star-pipeline-log.md)
- **Results analysis:** [`thoughts/shared/research/2026-06-14-results-analysis.md`](thoughts/shared/research/2026-06-14-results-analysis.md)
- **Plain-language walkthrough:** [`thoughts/shared/research/2026-06-14-plain-language-walkthrough.md`](thoughts/shared/research/2026-06-14-plain-language-walkthrough.md)

---

## Limitations

This is a showcase, not a deployed system. The four largest Phase-5 gaps are now **closed** (base-vs-
fine-tuned → Phase 6; full 58-channel LSTM → Phase 7; vision detector → Phase 8; semantic advice
grading → Phase 9). Residual gaps, stated up front:

1. **Eval-unit asymmetry reduced, not eliminated** — LSTM macro-averaged over 58 contiguous
   channels; LLM micro-averaged over 4,500 shuffled cross-mission windows. A fully like-for-like
   rematch would score both on one identical contiguous stream.
2. **Frontier control is a sample (n=150)**, fed deliberately context-free input — a hard sanity
   check, not a full frontier benchmark.
3. **Advice labels are synthetic** (in-session generated), and the advice grade is a 120-window
   LLM-judged sample, not a human-SME panel.
4. **The Hybrid's detection score is inherited** (= the LSTM's by construction); only its advice
   layer is new.
5. **Single fine-tune, no hyperparameter sweep**, no detection-tuned LLM variant or P-R curve, and
   the vision model has no advice head.
6. **1-hour resampling is lossy** for sub-hour transients; cross-mission generalization untested.

Each has a concrete next step in the
[analysis doc](thoughts/shared/research/2026-06-14-results-analysis.md#10-recommended-next-steps-priority-order).

---

## License & data

Code in this repository is released under the **MIT License** (see [`LICENSE`](LICENSE)). The
**ESA Anomaly Dataset** is governed by its own license — **CC BY 3.0 IGO** (attribution required,
commercial use permitted, no share-alike) — per its [Zenodo record](https://doi.org/10.5281/zenodo.12528696);
raw telemetry is **not** redistributed here. Any derived artifacts published elsewhere (e.g. the
fine-tuned models on Hugging Face) attribute ESA / the ESA-AD authors accordingly.
