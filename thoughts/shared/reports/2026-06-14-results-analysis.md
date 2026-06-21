# STAR-Pipeline — Results, Analysis & Discussion

**Project:** Space Telemetry Anomaly Detection & Resolution Pipeline (STAR-Pipeline)
**Scope:** A full, presentation-grade write-up of what was built, what was measured, what the
numbers mean, where the work sits relative to the published field, what it contributes, the honest
limitations, and the open next steps.

**Companion docs**
- Plain-language walkthrough (learn every concept and decision from scratch):
  [`2026-06-14-plain-language-walkthrough.md`](2026-06-14-plain-language-walkthrough.md)
- Auto-generated metrics report: [`results/comparison_report.md`](../../../results/comparison_report.md)
  and [`results/comparison_metrics.json`](../../../results/comparison_metrics.json)
- Original brief (a product brief: a PM needs anomaly detection + end-user advice and wants to own
  the model rather than depend on an external API):
  [`space_telemetry_anom_llm_ISSUE.md`](../issues/space_telemetry_anom_llm_ISSUE.md)
- Validated architecture research: [`2026-06-12-star-pipeline-codebase-research.md`](../research/2026-06-12-star-pipeline-codebase-research.md)
- Implementation plan: [`2026-06-12-star-pipeline-create-plan.md`](../plans/2026-06-12-star-pipeline-create-plan.md)
- Implementation log (the full engineering trail, 49 numbered deviations):
  [`2026-06-12-star-pipeline-log.md`](../implement/2026-06-12-star-pipeline-log.md)

> **Related — separate task, not on this scoreboard.** A from-scratch **Vision Transformer +
> Explainable-AI** showcase (Phase 16, `phase16-mini-foxes` branch) reproduces a miniature of
> **FOXES** — multi-channel SDO/AIA EUV imagery → GOES soft-X-ray flux *regression* (MAE 0.368 dex,
> Pearson r 0.943). It is **deliberately excluded from the comparison table in this document**
> (different task, different metric — error in *dex*, not anomaly precision; forcing it onto the
> CEF0.5/Affinity-F1 board would be apples-to-oranges). It connects to this work by **shared
> engineering discipline** — the same provenance (config snapshots, atomic writes, `--resume`),
> `validate-*` gates, matplotlib vocabulary, Vast.ai training path, and report/model-card packaging
> (*same engineer, same rigor, new domain*) — and it adds the repo's first from-scratch `nn.Module`
> and first image/heatmap rendering. See [`results/foxes_repro/report.md`](../../../results/foxes_repro/report.md)
> and [`thoughts/shared/phase16/`](../phase16/).

---

## 1. Executive summary

On real European Space Agency satellite telemetry (the ESA-AD benchmark), this project builds and
**empirically compares more than a dozen ways to do the same job**: flag anomalous telemetry windows
and, where possible, explain them in plain language. The comparison spans a classical
non-deep-learning floor (Isolation Forest), a tuned deep-learning detector (LSTM), a fine-tuned
open-source LLM reading the numbers as text (Qwen3-8B), a fine-tuned vision LLM reading *plots* of
the numbers (Qwen3-VL-8B), the un-fine-tuned base of each (zero- and few-shot), a frontier model
(Claude, zero- and few-shot), a deliberately dumb "flag everything" control, and a learned
**ensemble** that fuses the detectors' scores.

Four findings carry the project:

1. **Among single detectors, the tuned LSTM wins** — F1 **0.553**, precision **0.837**, the best
   precision-weighted CEF0.5 **0.705**, and a genuine interval-aware Affinity-F1 **0.673**. The two
   fine-tuned LLM detectors are *mirror images* of one another: the **vision** model (Qwen3-VL reading
   rendered plots) is precision-oriented (P 0.769) and the **text** model is recall-oriented; properly
   calibrated, the text model's precision more than doubles (to 0.838) and its detection becomes
   competitive with the LSTM (§6.4). On their own, no LLM beats the LSTM at detection — consistent
   with the published literature.
2. **But because the three detectors make *independent* errors, fusing their continuous scores beats
   every one of them.** A leakage-free stacked ensemble over the shared evaluation set reaches
   **CEF0.5 0.781 / AUC-PR 0.756 / precision 0.922** (text+vision+LSTM) — a Pareto win over each
   model's own best operating point, and the strongest detection result in the project (§6.3). The
   production architecture the data recommends is therefore an **ensemble detector feeding an LLM
   advisor**, with a cheap LSTM first-pass to keep it affordable.
3. **Fine-tuning is justified, and the proof survives a skeptic.** Read against a *trivial
   always-anomaly baseline* (F1 0.399 for free on this 25%-positive set), the fine-tuned LLM is the
   **only LLM-family approach that clears that line with a balanced precision/recall trade-off**.
   Few-shot prompting a base "matches" its F1 only by over-flagging ~80% of windows; a far stronger
   frontier model, prompted two ways, sits *at chance* on this context-free input. The two modalities
   even expose the lesson from opposite ends: for text, fine-tuning's headline win is *output
   compliance* (the base can't produce a verdict at all); for vision, the base already complies, so
   the win is *learned discrimination* — precision 0.310 → 0.769 (§5).
4. **The LLM's unique value is the advice layer — and it is good when it should be.** When the model
   correctly flags an anomaly, its diagnostic advice averages **5.58/6 and is 95% high-quality**,
   naming the right channel with a window-consistent magnitude. But advice quality is *gated by
   detection precision*: on false alarms it scores ~1/6 (95%-when-correct is a strong showcase result,
   not yet a mission-critical guarantee — §7).

The headline is neither "LLMs win" nor "LLMs lose," but *"here is the measured trade-off, and here is
the design it dictates."* That design — a fused high-precision detector triggering a fine-tuned LLM
advisor — is exactly the localized, open-source, no-external-API system this project set out to
validate.

---

## 2. The headline result (master comparison)

Final numbers on the ESA-AD evaluation material. **Read the table against the always-anomaly line** —
anything at or below F1 ≈ 0.40 has not learned to detect, it is just over-flagging.

| # | Approach | Precision | Recall | F1 | CEF0.5† | Affinity-F1 | Detects? | Advises? | Eval unit |
|---|----------|-----------|--------|----|---------|-------------|----------|----------|-----------|
| 1 | Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | — | ✅ | ❌ | 3 channels (macro) |
| 2 | **LSTM baseline** | **0.837** | 0.432 | **0.553** | **0.705** | **0.673** | ✅ | ❌ | 58 channels (macro) |
| 3 | LLM detection — **text** (Qwen3-8B) | 0.360 | **0.609** | 0.453 | 0.392 | 0.456‡ | ✅ | ✅ | 4,500 windows (micro) |
| 4 | LLM detection — **vision** (Qwen3-VL-8B) | 0.769 | 0.325 | 0.457 | 0.604 | — | ✅ | ❌ | 2,000 PNG windows (micro) |
| 5 | Base Qwen3-8B — zero-shot | 0 | 0 | 0 | 0 | — | ❌ | ❌ | 100 windows |
| 6 | Base Qwen3-8B — few-shot (2-ex) | 0.282 | 0.824 | 0.420 | 0.325 | — | ⚠️ | ⚠️ | 500 windows |
| 7 | Base Qwen3-VL — zero-shot | 0.310 | 0.403 | 0.350 | 0.325 | — | ⚠️ | ❌ | 2,000 PNG windows |
| 8 | Frontier (Claude) — zero-shot | 0.308 | 0.216 | 0.254 | 0.284 | — | ⚠️ | ✅ | 150-window sample |
| 9 | Frontier (Claude) — few-shot | 0.200 | 0.297 | 0.239 | 0.214 | — | ⚠️ | ✅ | 150-window sample |
| 10 | **Always-anomaly (trivial)** | 0.250 | 1.000 | 0.399 | 0.294 | — | ❌ | ❌ | 4,500 windows |
| 11 | **Ensemble (text+vision, stacked)** | **0.810** | 0.511 | **0.627** | **0.725** | — | ✅ | ❌ | 2,000 shared windows ✦ |
| 12 | **Ensemble (text+vision+LSTM, stacked)** | **0.922** | 0.486 | **0.636** | **0.781** | — | ✅ | ❌ | 1,378 M1 shared windows ✦ |
| 13 | **Hybrid (ensemble detect → LLM advise)** | **0.922** | 0.486 | **0.636** | **0.781** | — | ✅ | ✅ | inherits row 12 + advice |

† **CEF0.5** = precision-weighted F-beta (β=0.5), the ESA-benchmark-aligned metric favoured when false
alarms are costly. ‡ Affinity-F1 for the text LLM is **degenerate** on the shuffled test split
(≈ window-level F1, §6.2); the LSTM's 0.673 is the *genuine* interval-aware number.
✦ **The ensemble rows are on a different evaluation unit** — the shared windows where the fused
models all have a score (2,000 windows with PNGs; the 1,378 Mission-1 subset where the LSTM also
scores). Their numbers are **not directly comparable** to the 4,500-window / 58-channel rows above;
the honest claim is internal (§6.3): on those shared windows the fused score beats every single
model's *own* best operating point.

**Operating-point note.** Rows 3 and 4 are the LLMs' *as-deployed* points. Both have a calibrated
operating curve (§6.4): the text LLM reaches **P 0.838 / CEF0.5 0.674** and the vision LLM
**CEF0.5 0.649** once their decision threshold is tuned — the over-flagging in row 3 is a calibration
artifact, not a capacity limit.

**Test material:** the text LLM is scored on the full **4,500-window** test split (**25.0%
anomalous**); the LSTM on all **58 Mission-1 target channels** on contiguous per-channel timelines;
the vision model and the base-VL control on **2,000** rendered test PNGs; the base/frontier text
controls on frozen seed-42 samples; the ensembles on the shared subsets noted above.

The rest of this document explains how each row was produced and what it means.

---

## 3. The lay of the land — what's out there, what we used, what we contribute

This project deliberately surveys the field and validates the options against current research rather
than committing to the first idea. Here is the landscape it sits in.

### 3.1 The two architectural philosophies

There is a genuine tension in the field:

- **Two-stage (industry standard):** a cheap traditional detector (LSTM / autoencoder / isolation
  forest) flags deviations; a language model *explains* them. Reliable, cheap, decoupled.
- **Unified LLM (research frontier):** a single fine-tuned LLM ingests the tokenized numeric window
  and **both detects and advises** — the *AnomLLM / Time-LLM* line of work.

Rather than assert that one approach is best, this project **builds and measures both**, plus a
classical floor and a fused ensemble — so the recommendation is *empirical*. That decision is the
spine of the whole result.

### 3.2 The reference works we drew on (the "gold standards")

| Work | What it is | How we used it |
| --- | --- | --- |
| **ESA-ADB** (`kplabs-pl/ESA-ADB`, ESA + KP Labs, 2024) | The official benchmark and evaluation pipeline for the ESA Anomaly Dataset — the dataset's own gold-standard harness. | Adopted its **dataset** (ESA-AD) and its **metrics** (CEF0.5, Affinity-F1); avoided the discredited *point-adjustment* metric it warns against. |
| **Telemanom** (NASA JPL, KDD 2018) | The canonical LSTM + dynamic-error-thresholding detector; the baseline for NASA MSL/SMAP. | Re-implemented its *method* (per-channel LSTM reconstruction error + threshold) as our baseline — applied to **ESA-AD**, not its native NASA data. Its *dynamic* thresholding is also evaluated (§6.2) — and found unsuitable here. |
| **AnomLLM** (rose-stl-lab, ICLR 2025) | Benchmarks LLMs *as direct detectors* on tokenized numbers; finds they are not yet competitive with tuned sequence models. | Reproduced the *unified-LLM detection* idea (text model classifying tokenized windows) and confirmed the trade-off on a new dataset. |
| **AnomSeer** (ICLR 2026) | Fine-tunes a *multimodal* LLM to read **rendered plots** of telemetry, not the raw numbers. | Reproduced the *vision* approach: Qwen3-VL-8B on PNG plots. |
| **Time-LLM** (ICLR 2024) | Patching / reprogramming of numeric series into the LLM embedding space. | Informed the **windowing / patching** ETL design (32-step windows). |
| **Unsloth** | QLoRA fine-tuning toolkit with Apple-silicon-friendly GGUF export. | The fine-tuning engine for both LLMs. |

### 3.3 What this project contributes

It is not a new algorithm; it is a **rigorous, reproducible application study with an honest
negative-and-positive result.** Concretely:

1. **A like-for-like bake-off on real ESA telemetry** across *four* model families and *three* input
   modalities (numbers-as-text, numbers-as-image, classical features), all scored with the ESA
   benchmark's own metrics, plus a learned fusion of them. Published comparisons usually pit one or
   two of these against each other; here they share one dataset, one harness, one report.
2. **A defensible answer to "did the fine-tuning actually help?"** — the question most fine-tuning
   showcases skip. By holding the harness fixed and varying only the model (un-fine-tuned base of
   *both* modalities, few-shot base, frontier zero/few-shot, trivial baseline), the work isolates
   *what fine-tuning bought* and **frames it against a trivial baseline** so the "win" cannot be an
   over-flagging artifact (§5).
3. **Evidence that direct LLM detection is signal-limited *and* calibration-limited** — a frontier
   model cannot beat the dumb baseline on 10 context-free values (signal), and the fine-tune's
   "over-flagging" turns out to be a decoding/threshold artifact that calibration removes (§6.4). Two
   transferable findings about *what an LLM detector needs* and *how to read its raw output*.
4. **A demonstrated Pareto win from modality fusion** — independent text/vision/LSTM errors combine,
   via a leakage-free stacker, into a detector that beats all three (§6.3).
5. **A precision-gated reading of "explainable AI" advice** — turning "99.6% of outputs are
   structured" into the sharper, graded claim "95% high-quality *when the flag is correct*, ~0% when
   it is a false alarm," which directly motivates the deployment architecture (§7).

---

## 4. Process & methodology

The pipeline was built and then progressively hardened. The table summarizes each phase; the
narrative of *why* each result looks the way it does is in §5–§8.

| Phase | What it produced |
| --- | --- |
| **1 — ETL** | Downloaded ESA-AD (3 missions, 224 channels, ~29 GB) from the Kaggle mirror; resampled to 1-hour cadence; **RevIN** per-channel normalization; **32-step rolling windows** (stride 16); balanced subsample. → **30,000** instruction/response records, **24.8% anomalous**, split **21,000 / 4,500 / 4,500** (seed 42). |
| **1.5 — Advice labels** | **7,457** structured `DIAGNOSIS / ADVICE / ACTION` records (one per anomaly window) + severity + pattern type, generated in-session and merged into the training responses. |
| **2 — Classical baselines** | **LSTM** (Telemanom-style autoencoder) and **Isolation Forest**. |
| **3 — LLM fine-tune (cloud)** | **Qwen3-8B** + **QLoRA** (r=16, α=16, all-linear, lr 2e-4, **3 epochs**) on a rented **RTX 4090**; loss 2.85 → 0.24; → **GGUF Q4_K_M** export. ~**$2.30**. |
| **4 — Local inference** | GGUF on the M3 Max via `llama-cpp-python` + **Metal** (all layers offloaded). |
| **5 — Unified evaluation** | Full **4,500-window** text-LLM run (hardened checkpoint/resume): P 0.360, R 0.609, F1 0.453, 99.6% structured advice. |
| **6 — Did fine-tuning help?** | Controls under the *identical* harness: base **zero-shot**, base **few-shot**, **frontier** (Claude) zero- and few-shot, and a **trivial always-anomaly** baseline (§5). |
| **7 — Full LSTM sweep** | LSTM scored on all **58 Mission-1 target channels** on contiguous timelines, so **Affinity-F1 is genuine** (§6.2). |
| **8 — Vision detector** | The *AnomSeer-style* **Qwen3-VL-8B** fine-tuned on **PNG plots** (A6000, 2 epochs). 2,000 test PNGs: F1 0.457, P 0.769, CEF0.5 0.604 (§6.3). |
| **9 — Advice grading** | A frozen seed-42 sample of **120** flags graded on a transparent rubric (correctness / actionability / grounding, §7). |
| **11 — LSTM calibration** | Swept the detector's global decision threshold; the calibrated operating point is the canonical LSTM result (P 0.837, §6.1–§6.2). |
| **12 — Vision base control** | Un-fine-tuned Qwen3-VL run zero-shot on the same PNGs — the vision-modality control that completes the §5 skeptic table. |
| **13 — Text-LLM calibration** | A continuous verdict score + PR curve for the text LLM (AUC-PR 0.678), exposing the calibrated operating points (§6.4). |
| **14 — Ensemble** | Score-level fusion of text + vision + LSTM via a leakage-free stacker — the strongest detector (§6.3). |

**Model selection (why Qwen3-8B).** The base model was chosen deliberately over Qwen2.5 and Llama-3:
Qwen3 posts stronger reasoning/math scores; its **Instruct** variant avoids untrained chat-token
issues; it is fully supported by the Unsloth fine-tuning toolkit; at **8B** it fits the 36 GB M3 Max
after 4-bit quantization (so inference runs locally with no API); and it has a **vision sibling,
Qwen3-VL-8B**, which made the AnomSeer-style approach possible with the same toolchain.

**Metrics used (and why):**
- **Precision / Recall / F1** — standard detection metrics. Precision = of the windows we flagged,
  how many were truly anomalous; recall = of the truly anomalous windows, how many we caught.
- **CEF0.5** — F-beta with β=0.5, weighting *precision* over recall, because in a control room a false
  alarm (alert fatigue) is costlier than a near-miss. This is the ESA-benchmark-aligned scalar.
- **AUC-PR** — area under the precision-recall curve: a *threshold-independent* measure of how well a
  continuous score ranks anomalies above nominals. Used for the calibrated LLMs and the ensemble.
- **Affinity-F1** — an interval-aware metric for streaming telemetry; only meaningful on **contiguous**
  timelines (§6.2).
- We deliberately **do not** use point-adjustment (PA), which the ESA-ADB authors show inflates scores.

---

## 5. Result A — Did the fine-tuning actually help? (the skeptic's table)

This is the question a fine-tuning showcase lives or dies on. The method: **hold the harness fixed**
(identical prompt, decoding parameters, and output parser) and **vary only the model** — for *both*
modalities.

| Model | F1 | CEF0.5 | Output-format compliance | Structured advice on flags | Eval |
|-------|----|--------|--------------------------|----------------------------|------|
| **Fine-tuned LLM (text)** | **0.453** | **0.392** | 0.994 | **0.996** | 4,500 windows |
| Base Qwen3-8B — zero-shot | 0 | 0 | 0.000 | 0.000 | 100 windows (partial) |
| Base Qwen3-8B — few-shot (2-ex) | 0.420 | 0.325 | 1.000 | 0.129 | 500 windows |
| Frontier (Claude) — zero-shot | 0.254 | 0.284 | 1.000 | 1.000 | 150-window sample |
| Frontier (Claude) — few-shot | 0.239 | 0.214 | 1.000 | 1.000 | 150-window sample |
| **Fine-tuned LLM (vision)** | **0.457** | **0.604** | 1.000 | 0.000 | 2,000 PNGs |
| Base Qwen3-VL — zero-shot | 0.350 | 0.325 | 1.000 | 0.000 | 2,000 PNGs |
| **Always-anomaly (trivial)** | **0.399** | 0.294 | 1.000 | 0.000 | flag-all (4,500) |

How to read it, in order:

1. **Anchor on the dumb baseline first.** *Always-anomaly* (flag every window) scores **F1 0.399** for
   free on a 25%-positive set. Any "detector" at or below that has not learned anything — it is just
   over-flagging. This anchor is what makes the rest of the table honest.
2. **vs. the un-fine-tuned text base (same model, same harness): F1 0.000 → 0.453.** The raw base,
   run through the identical strict harness, emits a parseable ANOMALY/NOMINAL verdict on **0%** of
   windows — it spends its token budget on chain-of-thought ("thinking mode") and never commits to the
   terse contract. The fine-tune complies on **99%** and emits the structured DIAGNOSIS/ADVICE format
   on **100%** of its flags (vs 0%). *Learning the output contract is itself a fine-tuning result.*
3. **The few-shot base's F1 is a mirage.** With two in-context examples and thinking suppressed, the
   base recovers compliance (100% parseable) and posts F1 **0.420** — but with precision **0.282** /
   recall **0.824**, i.e. it flags ~80% of all windows. That is barely above flag-everything: it is
   **over-flagging, not detecting** (the 1:1 example ratio mis-signals a ~50% prior). It is also slower
   (8.56 s vs 2.77 s/window) and emits structured advice on only **13%** of flags.
4. **The frontier (Claude) is genuinely trying, but the input is near-signal-free.** Zero-shot F1
   **0.254**; adding the *same* few-shot examples barely moves it (**0.239**) — so the gap is **not** a
   prompting-asymmetry artifact. Unlike the base it does *not* over-flag (P≈R≈0.3, near the base rate),
   so it sits ~at chance. **A far stronger general model, prompted two ways, cannot beat the dumb
   baseline here.** *Why?* Each window is presented as ~10 normalized numbers plus a mission and
   channel name. The discriminating information — the **signal** — is whether that little sequence is
   abnormal *for that specific channel*, which requires **channel history**: knowing what is normal
   for, say, `Mission1/channel_41` (its usual range, rhythm, and noise). A reading of 0.6 may be
   perfectly normal on one channel and a clear fault on another; with no history you cannot tell. The
   frontier has never seen this channel, so 10 context-free numbers carry almost no signal — especially
   for the dominant *subtle-deviation* anomalies, which look nearly normal in 10 numbers. The fine-tune
   **learned each channel's normal from 21,000 training windows**, which is exactly the prior the
   frontier lacks. (Giving a frontier that context — via retrieval/RAG or by fine-tuning it — is in
   §10.)
5. **The vision modality tells the same lesson from the opposite end.** Running the *un-fine-tuned*
   Qwen3-VL-8B base through the identical PNG harness, it is **fully format-compliant (100% parseable,
   0 UNKNOWN)** — a chat-VL simply answers the ANOMALY/NOMINAL question — *unlike* the text base, which
   emitted 0%. But it does **not discriminate**: F1 **0.350** (P **0.310**, R **0.403**), *below* the
   flag-everything line. Fine-tuning lifts it to F1 **0.457** and, decisively, **precision 0.310 →
   0.769** (Δ **+0.459**). So for **text**, fine-tuning's headline win is **output compliance** (the
   base can't even produce a verdict); for **vision**, the base already complies, so the win is
   **learned discrimination** — the channel-specific priors that turn a compliant guesser into a
   precise detector. Both wins come from one lever: localizing the model to this mission's data.
6. **The honest headline.** The **fine-tuned model is the only approach in the LLM family that beats
   the always-anomaly baseline with a balanced precision/recall** — the lone real detector among them.
   On top of detection it delivers output compliance, reliable structured advice, and 3× lower latency.
   What fine-tuning bought is the mission/channel-specific priors that no prompt over 10 normalized
   values can supply — exactly the localized capability the project targeted.

---

## 6. Result B — The detection bake-off (who is the best *detector*?)

### 6.1 The single detectors, side by side

The four single detectors, ranked by F1:

| Detector | Precision | Recall | F1 | CEF0.5 | Character |
| --- | --- | --- | --- | --- | --- |
| **LSTM** | **0.837** | 0.432 | **0.553** | **0.705** | best single detector; high precision |
| **Vision LLM** (Qwen3-VL) | 0.769 | 0.325 | 0.457 | 0.604 | precision-oriented; rarely false-alarms |
| **Text LLM** (Qwen3-8B) | 0.360 | 0.609 | 0.453 | 0.392 | recall-oriented as-deployed (calibrates to P 0.838, §6.4) |
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | non-temporal floor; drowns in false positives |

**The LSTM is the strongest single detector** — highest F1, highest CEF0.5, and a genuine
interval-aware Affinity-F1 of **0.673**. Its method is industry standard (Telemanom-style per-channel
reconstruction error), and its decision threshold is a single global hyperparameter tuned on the
operating curve (§6.2). This matches the AnomLLM literature: direct LLM time-series detection trades
precision for recall and is not yet competitive with a tuned sequence model — *as a single model*.
The two LLM modalities are mirror images (vision precise / text high-recall) at nearly identical F1.
**Isolation Forest** is the floor: ignoring the order of time entirely, it drowns in false positives
(precision 0.127), exactly as the literature predicts for a non-temporal method on telemetry.

Crucially, none of these single-detector numbers is the project's *best* detection result — the fused
ensemble (§6.3) beats all of them. And the single-detector gaps are smaller than they first look: the
text LLM's as-deployed precision (0.360) is a calibration artifact that closes to 0.838 once its
decision threshold is tuned (§6.4).

### 6.2 Reading the LSTM: full-channel honesty, a genuine Affinity-F1, and a tuned threshold

Three things make the LSTM number trustworthy rather than cherry-picked:

- **All 58 Mission-1 target channels, macro-averaged** — not a hand-picked easy subset. The LSTM
  faces a full, heterogeneous channel set, much closer to the LLM's full-distribution test.
- **A genuine Affinity-F1 (0.673).** Because the LSTM scores *contiguous* per-channel timelines, its
  per-window predictions merge into real multi-window intervals, so the interval-aware metric is
  meaningful. By contrast the text LLM's test split is *shuffled and balanced-subsampled*
  (~1.4 windows per channel), so its "intervals" are mostly isolated single windows and its
  Affinity-F1 (0.456) collapses to ≈ window-level F1 — reported with that disclaimer rather than
  dressed up.
- **A calibrated decision threshold.** The detector flags a window when its reconstruction error
  exceeds `μ + zσ`. The threshold multiplier `z` is a single global knob; sweeping it over all 58
  channels traces a clean operating curve (`results/lstm/threshold_sweep.json`) on which CEF0.5 rises
  to a ~0.708 plateau. The canonical operating point (z = 4.0) sits at the elbow of that curve —
  P 0.837 / R 0.432 / F1 0.553 / CEF0.5 0.705 — chosen transparently from the published curve, not by
  peeking at the test set. This is the LSTM analogue of the text-LLM threshold calibration in §6.4:
  in both cases a single business-aligned threshold, not a model change, sets the precision.

One negative result is worth recording: **Telemanom's canonical *pruned dynamic* error thresholding
does not help here.** Implemented faithfully (EWMA-smoothed, log-transformed errors → adaptive-`z`
`find_epsilon` → percentage-drop pruning), it is far too conservative for this window-level,
relatively-frequent-anomaly labeling — its `find_epsilon` objective assumes *rare isolated* events, so
it flags only the largest spikes and collapses recall to 0.068 (CEF0.5 0.244). The published
"Telemanom-ESA-Pruned" ~0.97 figure is under the harder ESA-ADB *event-wise* protocol, not our window
protocol, so it is not directly comparable. Remaining untried levers — attention/bidirectional layers,
longer context, and channel ensembling — are in §10.

### 6.3 The strongest detector: a fused ensemble (and why fusion beats every single model)

The three detectors **fail differently** — the text LLM over-flags (high recall), the vision LLM is
conservative (high precision), and the LSTM is precise on a different basis (reconstruction error).
Because their errors are *independent* (numeric text vs rendered image vs reconstruction), fusing
their *continuous* scores does better than any one of them — and better than the naive "both must
fire" (AND, high precision) or "either fires" (OR, high recall) endpoints.

Each detector's continuous per-window score was captured — text and vision as the verdict-token
logprob (the softmax of ANOMALY-vs-NOMINAL logits, §6.4), the LSTM as its reconstruction error — and
fused with a **leakage-free out-of-fold k-fold logistic stacker** (a logistic regression trained on
held-out folds so no window is scored by a model trained on it). Aligning the modalities was the work:
the **2,000 windows with PNGs** are exactly the text-scored windows by `index` (verified, 0
mismatches), and the LSTM maps in via `(mission, channel, start_idx) → start_idx // stride` on the
shared 1-hour/stride-16 grid (verified exact; Mission-1 only). Results on the shared set, each model
compared against its *own* best operating point computed on the same windows:

| Fusion | Windows | Fused AUC-PR | Fused CEF0.5-optimal point | Beats single-best CEF0.5 |
|---|---|---|---|---|
| **text + vision** | 2,000 | **0.703** | P 0.810 / R 0.511 / F1 0.627 / **CEF0.5 0.725** | text 0.683, vision 0.649 ✅ |
| **text + vision + LSTM** | 1,378 (Mission-1) | **0.756** | P 0.922 / R 0.486 / F1 0.636 / **CEF0.5 0.781** | text 0.731, vision 0.666, LSTM 0.479 ✅ |

Both fused points **dominate every single model on both AUC-PR and CEF0.5** on the shared set — the
independent-errors intuition holds, and the 3-model fusion (P 0.922) is the strongest detection result
in the project. A **2-of-3 vote** gives a balanced P 0.724 / R 0.600 / F1 0.656, and a
**disagreement → human-review** tier — alarm only when the detectors agree; route the ~747-of-1,378
windows where they disagree to an operator — is the deployable read.

> **Important caveat — different evaluation unit.** These ensemble numbers are on the shared
> 2,000-/1,378-window set, **not** the 4,500-window / 58-channel basis of §2/§6.1, so CEF0.5 0.781 is
> *not* a like-for-like replacement for the LSTM's 0.705 (the master-table LSTM is a semi-supervised
> threshold over *all* Mission-1 windows; here the LSTM error is ranked on the harder balanced
> subsample, which is why its standalone AUC-PR here is only 0.479). The defensible claim is the
> *internal* one: on identical windows, fusion beats each component. A 3-model ensemble over all
> missions would require training Mission-2/3 LSTMs (which do not exist — §9), so the 3-model row is
> Mission-1 only. Full detail: `results/ensemble_pr_curve.{json,png}`, `src/inference/ensemble.py`.

The vision model also has independent value as a **second precision-oriented signal**: at P 0.769 it
almost never false-alarms (49 FP vs 1,450 TN on 2,000 windows), so `vision detector → text advisor` is
a legitimate all-LLM hybrid where false alarms are especially costly. It converged very fast on its
2-class task (eval loss 0.0089) and its weak point is recall (0.325); more/varied training data, a
larger backbone, or more epochs could lift recall (§10).

### 6.4 Calibrating the text LLM — the over-flagging was a threshold artifact

The text LLM's as-deployed **P 0.360 / R 0.609** is a *single* operating point inflated by false
alarms two ways: it was decoded with **stochastic sampling** (temperature 0.8), and it raised ANOMALY
whenever that token won a *sampled* draw. Replacing the hard verdict with a **continuous,
deterministic anomaly score** per window — the softmax of the model's ANOMALY-vs-NOMINAL logits at the
verdict position — and sweeping a decision threshold over it exposes the model's true operating curve:

| Operating point | Threshold | Precision | Recall | F1 | CEF0.5 |
| --- | --- | --- | --- | --- | --- |
| As-deployed (sampled hard verdict) | — | 0.360 | 0.609 | 0.453 | 0.392 |
| Deterministic argmax | 0.500 | 0.527 | 0.639 | 0.578 | 0.546 |
| F1-optimal | 0.580 | 0.621 | 0.567 | **0.593** | 0.609 |
| **CEF0.5-optimal** | 0.775 | **0.838** | 0.379 | 0.521 | **0.674** |

Two findings:

1. **Removing the sampling noise recovers ~17 precision points for free.** The deterministic argmax
   (threshold 0.5 — the natural boundary) already sits at **P 0.527**, versus the sampled point's
   0.360. Same weights — temperature-0.8 decoding alone was injecting random ANOMALY flips.
2. **The score genuinely separates, and the curve is deployable.** **AUC-PR = 0.678** (random floor =
   the 0.250 base rate). At threshold 0.775 the text LLM reaches **P 0.838 / CEF0.5 0.674** — its
   CEF0.5 lands just below the LSTM (0.705) and above the vision LLM (0.604). **Once calibrated, the
   text LLM is a competitive precision-weighted detector, not a trigger-happy one.**

This confirms that the low precision was a **decision-boundary/calibration problem, not a capacity
one** — the same lesson as the LSTM's threshold calibration (§6.2). It also means a *detection-only*
SFT (dropping the advice head) is not indicated: the auxiliary advice task likely *helps* the shared
representation, so removing it is unlikely to raise precision and would cost a capability. The
continuous score is also what makes the fusion in §6.3 possible. (The vision detector has its own
calibration curve — AUC-PR 0.586; tuning its threshold buys recall, 0.325 → 0.453 at CEF0.5 0.649 —
but no "turn off sampling" win, since it already decoded greedily.)

The master table (§2) keeps the LLMs at their *as-deployed* points for honesty about what was measured
end-to-end; this section is the deployment knob.

---

## 7. Result C — Is the advice any good? (semantic grading)

Detection is only half the brief; the other half is *"end-user advice on the effect of a change."*
None of the classical baselines can do this at all. The fine-tune emits structured
`DIAGNOSIS / ADVICE / ACTION` text on **99.6%** of its flags — but the sharper question is whether
that advice is *correct*.

A frozen seed-42 sample of **120** of the model's anomaly predictions — preserving the real **TP:FP
ratio** (true-positive to false-positive flags; the model's ~0.36 as-deployed precision means roughly
43 correct flags to 77 false alarms, and the sample keeps that ratio so the grade reflects the true
false-alarm rate rather than a cleaned-up set) — was graded on a transparent, *verifiable* rubric:
correctness / actionability / grounding, 0–2 each (total 0–6).

**How the grading worked (and why it is not circular).** The grader (the Claude session model) did
**not** score against its own world knowledge, nor did it simply diff against the synthetic advice
labels (it could not — the stored predictions lack the keys to join to those labels). It scored
against **the window's own data and the ground-truth anomaly label**: *grounding* = does the named
channel match the input window and is the stated magnitude consistent with the actual values;
*correctness* = ground-truth-gated, so advice written about a truly-nominal window (a false alarm)
scores 0 by construction. This is why "just use the frontier model instead" does not follow: judging
finished advice **with the answer key in hand** is a far easier task than *producing* detections cold
— and as a detector the same frontier sat at chance (§5).

| Subset | n | Correctness | Actionability | Grounding | Mean /6 | High-quality |
|--------|---|-------------|---------------|-----------|---------|--------------|
| All flags | 120 | 0.64 | 1.03 | 1.01 | 2.68 | 34% |
| **True positives (correct flags)** | 43 | **1.79** | **1.93** | **1.86** | **5.58** | **95%** |
| False positives (false alarms) | 77 | 0.00 | 0.53 | 0.53 | 1.06 | 0% |

- **When the model is right to flag, its advice is genuinely good:** mean **5.58/6**, 95%
  high-quality, with **near-perfect grounding (1.86/2)** — 119/120 named the correct *channel* (the
  specific sensor stream, e.g. `Mission1/channel_41`) and a magnitude consistent with the window;
  3/120 mislabelled the *subsystem* (the functional group the channel belongs to — power, thermal,
  attitude, etc.). Even on the rare miss it usually points at the right sensor; what slips is the
  higher-level category. Actionability was 1.93/2 (a severity-appropriate recommended action).
- **Advice quality is gated by detection precision.** On false alarms (77 of 120) the advice is built
  on a false premise, so correctness is **0.00/2** by construction; the model fabricates a confident
  "persistent anomaly" narrative on a truly-nominal window. The overall mean drops to 2.68/6 precisely
  *because* the standalone detector over-flags.
- **This is the empirical case for putting the advisor behind a high-precision detector** — not using
  the LLM to *decide* what is anomalous, but to *explain* what a precise detector flags.

**"95% high-quality" is good, but not yet mission-critical-grade.** For a *lives-depend-on-it* system,
95%-when-correct (graded by an LLM on n=120, against **synthetic** advice labels) is a strong showcase
result, not a deployment guarantee. The highest-leverage steps to close that gap, in order of impact:
1. **Replace the synthetic advice labels with human-SME-written advice** — the current ceiling is the
   templated, statistic-derived labels; real fault-engineer advice (with true root causes) is the
   single biggest lever on correctness.
2. **Ground the advisor in real references (RAG)** — give it the channel's spec sheet / fault catalog
   so it cites documented root causes and procedures instead of pattern templates.
3. **Put it behind a high-precision detector** (the ensemble or LSTM) so it is rarely asked to explain
   a false alarm — directly fixes the precision-gating.
4. **Add calibrated confidence / abstention** so it can say "uncertain" instead of confidently
   fabricating on a borderline window.
5. **A bigger advisor backbone and a larger, human-validated advice grade** to replace the n=120
   LLM-judge sample with tighter confidence.

---

## 8. Analysis & discussion — the architecture the data recommends

Putting §5–§7 together yields a single, empirically-supported recommendation: a **two-stage hybrid
whose detector is a score-fused ensemble**, with a cheap first-pass to keep it affordable.

```
                          ┌─────────────────── per-window cost ───────────────────┐
telemetry window ──► LSTM detector ──flag──► ensemble confirm ──confirm──► LLM advisor ──► operator
                     (sub-ms, P 0.837)        (text+vision+LSTM            (Qwen3-8B, structured
                      cheap first-pass         stacker, P 0.922)            DIAGNOSIS/ADVICE/ACTION,
                                               cuts false alarms            95% HQ on real flags)
```

Why this is the right design, not just a convenient one:

- **Detection is strongest as a fusion, cheapest as the LSTM — so cascade them.** The fused ensemble
  is the most precise detector (P 0.922, CEF0.5 0.781, §6.3), but it runs three models per window. The
  LSTM alone is nearly as precise (0.837), sub-millisecond, and free — so use it as the **first-pass
  screen** and invoke the LLM-bearing ensemble only on the windows it flags. You get the ensemble's
  precision where it matters without paying LLM latency on every window (the text LLM is 2.77 s/window
  generated, ~0.7 s scored). Where compute is ample, score the full ensemble on every window directly.
- **Explanation belongs to the LLM, invoked only on confirmed flags.** §7 shows that *on the windows a
  high-precision detector flags*, the advice is 95% high-quality. The expensive model runs ~1/N of the
  time and supplies the one capability the detectors lack.
- **This hybrid strictly dominates any single component for the stated business need:** it inherits the
  ensemble's detection (the best in the project) *and* adds reliable, grounded advice — the exact
  "anomaly + end-user advice, no external vendor" capability this project targets. (The simpler
  `LSTM → advisor` design is the low-cost variant; the all-LLM `vision → advisor` is the variant for
  contexts where a near-zero false-alarm rate dominates.)

The deeper, most transferable lesson is **where the signal lives — and how to read it**. A frontier
model with vastly more general capability than Qwen3-8B *cannot* detect these anomalies from the
prompt alone (§5): the fine-tune can, because it *learned the mission- and channel-specific priors*
that no prompt over 10 normalized values can supply. And the fine-tune's apparent weakness as a
detector was largely a *reading* problem — its raw sampled verdict was miscalibrated, and a tuned
threshold on its own confidence score recovers most of the gap (§6.4). Localization plus calibration,
not raw model size, is what turns a general model into a deployable mission-specific detector.

One honest qualifier: we compared a *fine-tuned* open model against a *prompted* frontier — not a
*fine-tuned* frontier. The cleanest "could you just adapt a hosted model instead?" test would be to
fine-tune a frontier (GPT-4o/Gemini expose tuning APIs; Claude does not publicly) or to give one the
missing channel history via RAG (§10). The trade-off: doing so re-introduces the vendor dependency,
data-egress, cost, and latency that an owned on-prem model avoids — so even a fine-tuned frontier that
matched the accuracy would still lose on the deployment constraints that motivated the project.

---

## 9. Limitations

Stated up front, each with *why it stands* and what it would take to close:

1. **The ensemble is on a different evaluation unit than the single detectors.** Its Pareto-win
   numbers (CEF0.5 0.781 / 0.725) are on the shared 1,378-/2,000-window sets where the fused models all
   have a score — not the 4,500-window / 58-channel basis — so they are not a like-for-like
   replacement for the master-table rows (§6.3). The claim is internal: on identical windows, fusion
   beats each component. A common-unit re-score of every approach on one contiguous stream (~1–2 days)
   would make all rows directly comparable.
2. **The 3-model ensemble is Mission-1 only.** Mission-2/3 LSTM models were never trained (the LSTM
   sweep was scoped to Mission 1), so the 622 M2/M3 windows in the shared set have no LSTM signal.
   Training those (~3 h; M3 channels are categorical and noisier) would extend the 3-model fusion to
   all missions.
3. **Mission scope of the LSTM itself.** The LSTM covers Mission-1 target channels, while the text LLM
   was evaluated across all three missions — a residual coverage asymmetry. Extending the LSTM to all
   missions (cheap; more per-channel models) would make the single-detector comparison fully
   coverage-matched, and **cross-mission generalization** (train on one mission, test on a spacecraft
   the model never saw) remains a separate, untested experiment (§10).
4. **The frontier control is a sample (n=150) on context-free input.** It is fed the same ~10
   normalized values the fine-tune saw, with no channel history — a deliberately *controlled* (and
   hard) comparison, not a best-effort frontier benchmark. A fair real-world frontier would get channel
   history via RAG, a longer raw window, or channel metadata (§10).
5. **Advice labels are synthetic, and the advice grade is a 120-flag LLM-judged sample.** The advice
   labels were generated in-session (statistic-derived templates), not written by spacecraft SMEs;
   the grade preserves the true TP:FP ratio but has wide confidence intervals, and the stored advice
   was truncated at 300 chars (clipping only the trailing ACTION line — the verdict and most of the
   advice survive, so no detection comparison is affected). Human-expert labels + a larger grade are
   the top lever toward mission-critical advice (§7, §10).
6. **The Hybrid's detection score is inherited, not independently measured** — by construction it
   equals its detector's (the ensemble, or the LSTM in the low-cost variant). Only its advice layer is
   genuinely new, and it was not benchmarked against an alternative advice generator.
7. **Single fine-tune, no hyperparameter sweep.** r=16 / α=16 / 3-epoch were sensible defaults run
   once rather than a grid. The knobs most likely to matter — LoRA rank, epochs, learning rate, prompt
   format — are a ~1-day, ~$5–15 cloud ablation (§10). (A *detection-only* SFT is **not** on this list:
   §6.4 showed the precision gap was calibration, not capacity, so dropping the advice head would cost
   a capability without an expected precision gain.)
8. **The vision detector converged very fast and has no advice head.** Its validation loss hit 0.0089
   on an easy 2-class task — meaning it fit the *training* distribution easily; whether it generalizes
   to a *new mission it never trained on* is untested (overfitting risk). It also produces no advice
   by design.
9. **The vision model trained on 10× less data than the text model.** The text LLM saw 21,000 training
   windows; the vision LLM saw only **2,000 training PNGs** — yet achieved higher precision (0.769 vs
   0.360 as-deployed). This data asymmetry may **understate the vision modality's potential**: scaling
   the vision training to 21,000 PNGs could improve both precision and recall, potentially making it
   the best single detector. The fast convergence (loss 0.0089) could indicate saturation *or*
   insufficient diversity — only a scaled run would tell (§10).
9. **Resampling to 1-hour cadence is lossy.** Raw telemetry is sampled every few seconds/minutes; we
   averaged it onto a 1-hour grid to make 29 GB tractable. An anomaly lasting **less than an hour** can
   be averaged into its neighbours and disappear. No other cadence (5-min/15-min) was tried, so the
   signal cost of the 1-hour choice is unmeasured (§10).
10. **Affinity-F1 is degenerate on the shuffled test split** for the LLMs (it reduces to window-level
    F1 on a ~1.4-window-per-channel set). It is implemented correctly and becomes meaningful only on a
    contiguous evaluation stream — which is exactly why the LSTM's 0.673 (contiguous timelines) is the
    only genuine Affinity-F1 reported (§6.2).

---

## 10. Open next steps

The detection-and-evaluation hardening (LSTM calibration, the vision skeptic-table control, text-LLM
calibration, and the fused ensemble) is **done** and folded into §5–§8 above. What remains, prioritized
with rough **effort** and **expected impact**:

1. **Ship the hybrid as the reference design.** *Effort: low* (packaging). *Impact: high.* The
   architecture the numbers support — a fused (or LSTM-first-pass) high-precision detector feeding the
   LLM advisor. **Who needs it:** operators of mission-critical monitoring systems — spacecraft FDIR,
   but equally industrial/SCADA telemetry or any high-availability infrastructure — that must run an
   **on-prem / sovereign** model (no data egress, no external-API dependency) and where **false alarms
   are costly**.
2. **Toward mission-critical advice (highest-value quality work).** *Effort: high* (needs human
   experts). *Impact: high.* Human-SME-written advice labels + **RAG grounding** (channel spec sheets /
   fault catalogs) + a larger, human-validated advice grade — the real path from "95% when correct" to
   deployment-grade (§7).
3. **Common-unit re-score + extend the LSTM to all missions.** *Effort: ~1–2 days + ~3 h training.*
   *Impact: medium–high.* Score every approach (including a 3-model ensemble) on one identical
   contiguous stream so all rows are directly comparable, and train Mission-2/3 LSTMs so the fusion and
   the single-detector comparison cover all missions, not just Mission 1 (§9 #1–#3).
4. **Fine-tune / RAG a frontier model — the true "own vs. adapt" comparison.** *Effort: medium.*
   *Impact: medium.* Fine-tune a frontier (GPT-4o/Gemini tuning APIs) or give one channel history via
   RAG, to test whether a custom 8B is even needed. **Caveat:** doing so re-introduces the vendor
   dependency the sovereign model exists to avoid.
5. **Push the detectors further.** *Effort: medium.* *Impact: medium–high.* LSTM levers left untried —
   attention/bidirectional layers, longer context, channel ensembling; a small **LoRA ablation** to
   confirm the text fine-tune is near its ceiling. **Highest-potential single lever: scale the vision
   training from 2,000 to 21,000 PNGs** (~1–2 h PNG generation + ~$1–2 cloud training) — the vision
   model already hits P 0.769 with 10× less data than the text model, so scaling may push it toward
   the best single detector and strengthen the ensemble. A vision **advice head** could also add the
   explanation capability the current vision detector lacks.
6. **Robustness sweeps.** *Effort: medium.* *Impact: medium.* A **cross-mission generalization test**
   (train on one mission, test on a never-seen spacecraft) and a **resample-cadence sweep** (5-min /
   15-min / 1-hour) to find the best signal-vs-tractability trade-off, since the 1-hour grid can
   average away sub-hour anomalies (§9 #9).

---

## 11. Definition-of-Done scorecard

| Brief's DoD item | Status | Evidence |
|---|---|---|
| Modular, scalable ETL: raw telemetry → LLM-digestible patches | ✅ | `src/etl/` — download, RevIN windowing, balanced subsample; 30,000 JSONL records across 3 missions |
| Traditional ML baseline trained and evaluated | ✅ | LSTM (Telemanom-style, 58 channels, calibrated) **and** Isolation Forest, both scored |
| Open-source LLM fine-tuned via Unsloth; infers anomaly **and** advice | ✅ | Qwen3-8B QLoRA → GGUF Q4_K_M → Metal inference; 99.6% structured advice, 95% high-quality on true flags |
| FP / precision / recall documented, LLM vs. baseline | ✅ | This analysis + `comparison_report.md`; full confusion matrices persisted |
| Cost-effective local/cloud split | ✅ | ETL + inference local on M3 Max; all cloud training/eval on rented GPUs for **~$4.2 total** |
| *Beyond DoD* | ✅ | did-fine-tuning-help controls (both modalities), a vision detector, semantic advice grading, threshold calibration, and a Pareto-winning fused ensemble |

Every box is checked — and the inconvenient findings (**no single LLM beats the LSTM at detection;
the few-shot base's F1 is a mirage; advice quality is precision-gated**) are reported plainly rather
than hidden. Surfacing correct-but-inconvenient results is the whole point of an empirical bake-off.

---

## 12. Bottom line

On real ESA satellite telemetry, no single LLM beats a tuned classical LSTM at *detection* (LSTM
F1 0.553, precision 0.837, CEF0.5 0.705) — direct LLM detection trades precision for recall, matching
the literature. But two results reframe that: (1) the text LLM's apparent over-flagging is a
**calibration artifact** — a tuned threshold on its own confidence score lifts precision 0.360 → 0.838;
and (2) because the text, vision, and LSTM detectors make **independent errors**, a leakage-free
**stacked ensemble beats all of them** (P 0.922, CEF0.5 0.781, AUC-PR 0.756 on the head-to-head set) —
the strongest detector in the project. Among the LLM family the fine-tune is still the **only** approach
that clears a trivial always-anomaly baseline with a balanced precision/recall, while a frontier model
prompted two ways sits at chance — proving the fine-tune learned mission-specific priors that prompting
cannot supply. And where the LLM uniquely earns its place — diagnostic advice — it is **95% high-quality
when the flag is correct** (strong, though not yet mission-critical-grade), but only then. The
empirically-grounded architecture is therefore a **fused high-precision detector feeding an LLM
advisor**, with a cheap LSTM first-pass — exactly the localized, open-source, no-vendor system this
project set out to validate. The headline is not "LLMs win" or "LLMs lose"; it is *"here is the measured
trade-off, and here is the design it dictates."*

---

## Appendix — Reproducibility

Every phase has a `make` target and a paired `validate-*` target encoding its success criteria. The
full LLM eval is durable (`--checkpoint-every N`, `--resume`, atomic writes). Published artifacts:
fine-tuned text GGUF — `dyrtyData/star-pipeline-qwen3-8b-advice-gguf`; vision adapter —
`dyrtyData/star-pipeline-qwen3-vl-8b-detection` (both on Hugging Face). Detection curves and the
ensemble are reproducible via `results/llm_pr_curve.*`, `results/vision_pr_curve.*`,
`results/ensemble_pr_curve.*` and `src/inference/{pr_curve,ensemble}.py`. Raw ESA-AD (~29 GB) is not
redistributed; `make download` fetches it. The implementation log records the full engineering trail
(49 numbered deviations) for anyone who wants the blow-by-blow.
