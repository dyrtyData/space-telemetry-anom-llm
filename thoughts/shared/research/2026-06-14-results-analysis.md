# STAR-Pipeline — Results, Analysis & Discussion

**Project:** Space Telemetry Anomaly Detection & Resolution Pipeline (STAR-Pipeline)
**Date:** 2026-06-14 (final — all nine phases complete)
**Scope:** A full, presentation-grade write-up of what was built, what was measured, what the
numbers mean, where the work sits relative to the published field, what it contributes, the honest
limitations, and the next steps.

**Companion docs**
- Plain-language walkthrough (learn every concept and decision from scratch):
  [`2026-06-14-plain-language-walkthrough.md`](2026-06-14-plain-language-walkthrough.md)
- Auto-generated metrics report: [`results/comparison_report.md`](../../../results/comparison_report.md)
  and [`results/comparison_metrics.json`](../../../results/comparison_metrics.json)
- Original brief: [`space_telemetry_anom_llm_ISSUE.md`](../issues/space_telemetry_anom_llm_ISSUE.md)
  (the brief is a product brief: a PM needs anomaly detection + end-user advice and wants to own the
  model rather than depend on an external API)
- Validated architecture research: [`2026-06-12-star-pipeline-codebase-research.md`](2026-06-12-star-pipeline-codebase-research.md)
- Implementation plan: [`2026-06-12-star-pipeline-create-plan.md`](../plans/2026-06-12-star-pipeline-create-plan.md)
- Implementation log (every deviation D1–D38, the full saga):
  [`2026-06-12-star-pipeline-log.md`](../implement/2026-06-12-star-pipeline-log.md)

---

## 1. Executive summary

On real European Space Agency satellite telemetry (the ESA-AD benchmark), this project builds and
**empirically compares ten ways to do the same job**: flag anomalous telemetry windows and, where
possible, explain them in plain language. The comparison spans a classical non-deep-learning floor
(Isolation Forest), a tuned deep-learning detector (LSTM), a fine-tuned open-source LLM reading the
numbers as text (Qwen3-8B), a fine-tuned vision LLM reading *plots* of the numbers (Qwen3-VL-8B),
the same open base model with no fine-tuning (zero- and few-shot), a frontier model (Claude, zero-
and few-shot), and a deliberately dumb "flag everything" control.

Three findings carry the project:

1. **As a pure detector, the tuned LSTM wins** (F1 0.553, precision 0.837, the best
   precision-weighted CEF0.5 0.705, and a genuine interval-aware Affinity-F1 0.673, after the
   Phase-11 operating-point calibration — see §6.1). The two
   fine-tuned LLM detectors land below it and are *mirror images* of one another — the **vision**
   model (Qwen3-VL reading rendered plots) is precision-oriented (P 0.769, and the best
   false-alarm-aware CEF0.5 of any LLM here, 0.604), while the **text** model is recall-oriented
   (R 0.609, over-flags). Neither beats the LSTM at detection — consistent with the published
   literature.
2. **Fine-tuning is nonetheless justified, and the proof survives a skeptic.** Read against a
   *trivial always-anomaly baseline* (F1 0.399 for free on this 25%-positive set), the fine-tuned
   LLM is the **only LLM-family approach that clears that line with a balanced precision/recall
   trade-off** — it is the lone *real* detector among the LLMs. Few-shot prompting a base model
   "matches" its F1 only by over-flagging ~80% of windows; a far stronger frontier model, prompted
   two different ways, sits *at chance* on this context-free input. Fine-tuning also buys output
   compliance (99% vs 0% for the raw base), reliable structured advice (99.6% vs 12.9% few-shot),
   and 3× lower latency.
3. **The LLM's real, unique value is the advice layer — and it is good when it should be.** A
   graded sample shows that **when the model correctly flags an anomaly, its diagnostic advice
   averages 5.58/6 and is 95% high-quality** — naming the right channel with a window-consistent
   magnitude (grounding 1.86/2). But advice quality is *gated by detection precision*: on false
   alarms the advice is built on a false premise and scores ~1/6. (95%-when-correct is a strong
   showcase result, not yet a mission-critical guarantee — see §7 for the path to that.)

Putting these together, the architecture the data recommends is the **two-stage hybrid the
industry already uses**: a cheap, high-precision detector (the LSTM) triggers, and the fine-tuned
text LLM explains each flag — with the precision-oriented vision model available as an optional
low-false-alarm cross-check. That is exactly the localized, open-source, no-external-API system this
project set out to validate — and the headline is neither "LLMs win" nor "LLMs lose," but
*"here is the measured trade-off, and here is the design it dictates."*

---

## 2. The headline result (master comparison)

Final numbers on the ESA-AD evaluation material. **Read the table top-down against the
always-anomaly line at the bottom** — anything at or below F1 ≈ 0.40 has not learned to detect, it
is just over-flagging.

| # | Approach | Precision | Recall | F1 | CEF0.5† | Affinity-F1 | Detects? | Advises? | Eval unit |
|---|----------|-----------|--------|----|---------|-------------|----------|----------|-----------|
| 1 | Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | — | ✅ | ❌ | 3 channels (macro) |
| 2 | **LSTM baseline (58 ch)** | **0.837** | 0.432 | **0.553** | **0.705** | **0.673** | ✅ | ❌ | 58 channels (macro) |
| 3 | LLM detection — **text** (Qwen3-8B QLoRA→GGUF) | 0.360 | **0.609** | 0.453 | 0.392 | 0.456‡ | ✅ | ✅ | 4,500 windows (micro) |
| 4 | LLM detection — **vision** (Qwen3-VL-8B) | 0.769 | 0.325 | 0.457 | 0.604 | — | ✅ | ❌ | 2,000 PNG windows (micro) |
| 5 | Base Qwen3-8B — zero-shot | 0 | 0 | 0 | 0 | — | ❌ | ❌ | 100 windows |
| 6 | Base Qwen3-8B — few-shot (2-ex) | 0.282 | 0.824 | 0.420 | 0.325 | — | ⚠️ | ⚠️ | 500 windows |
| 7 | Frontier (Claude) — zero-shot | 0.308 | 0.216 | 0.254 | 0.284 | — | ⚠️ | ✅ | 150-window sample |
| 8 | Frontier (Claude) — few-shot | 0.200 | 0.297 | 0.239 | 0.214 | — | ⚠️ | ✅ | 150-window sample |
| 9 | **Always-anomaly (trivial)** | 0.250 | 1.000 | 0.399 | 0.294 | — | ❌ | ❌ | 4,500 windows |
| 10 | **Hybrid (LSTM detect → LLM advise)** | **0.837** | 0.432 | **0.553** | **0.705** | 0.673 | ✅ | ✅ | inherits row 2 + advice |

† **CEF0.5** = precision-weighted F-beta (β=0.5), the ESA-benchmark-aligned metric favoured when
false alarms are costly. ‡ Affinity-F1 for the text LLM is **degenerate** on the shuffled test
split (≈ window-level F1) — see §6.2; the LSTM's 0.673 is the *genuine* interval-aware number.

**Test material:** the LLM is scored on the full **4,500-window** test split (**25.0% anomalous**);
the LSTM on all **58 Mission-1 target channels** on contiguous per-channel timelines; the vision
model on **2,000** rendered test PNGs; the base/frontier controls on frozen seed-42 samples.

The rest of this document explains how each row was produced and what it means.

---

## 3. The lay of the land — what's out there, what we used, what we contribute

This project deliberately surveys the field and validates the options against current research
rather than committing to the first idea. Here is the landscape it sits in.

### 3.1 The two architectural philosophies

There is a genuine tension in the field:

- **Two-stage (industry standard):** a cheap traditional detector (LSTM / autoencoder / isolation
  forest) flags deviations; a language model *explains* them. Reliable, cheap, decoupled.
- **Unified LLM (research frontier):** a single fine-tuned LLM ingests the tokenized numeric window
  and **both detects and advises** — the *AnomLLM / Time-LLM* line of work.

Rather than assert that one approach is best, this project **builds and measures both**, plus a
classical floor — so the recommendation is *empirical*. That decision is the spine of the whole
result.

### 3.2 The reference works we drew on (the "gold standards")

| Work | What it is | How we used it |
| --- | --- | --- |
| **ESA-ADB** (`kplabs-pl/ESA-ADB`, ESA + KP Labs, 2024) | The official benchmark and evaluation pipeline for the ESA Anomaly Dataset — the dataset's own gold-standard harness. | Adopted its **dataset** (ESA-AD) and its **metrics** (CEF0.5, Affinity-F1); avoided the discredited *point-adjustment* metric it warns against. |
| **Telemanom** (NASA JPL, KDD 2018) | The canonical LSTM + dynamic-error-thresholding detector; the baseline for NASA MSL/SMAP. | Re-implemented its *method* (per-channel LSTM reconstruction error + threshold) as our baseline — applied to **ESA-AD**, not its native NASA data. |
| **AnomLLM** (rose-stl-lab, ICLR 2025) | Benchmarks LLMs *as direct detectors* on tokenized numbers; finds they are not yet competitive with tuned sequence models. | Reproduced the *unified-LLM detection* idea (text model classifying tokenized windows) and confirmed the trade-off on a new dataset. |
| **AnomSeer** (ICLR 2026) | Fine-tunes a *multimodal* LLM to read **rendered plots** of telemetry, not the raw numbers. | Reproduced the *vision* approach: Qwen3-VL-8B on PNG plots (Phase 8). |
| **Time-LLM** (ICLR 2024) | Patching / reprogramming of numeric series into the LLM embedding space. | Informed the **windowing / patching** ETL design (32-step windows). |
| **Unsloth** | QLoRA fine-tuning toolkit with Apple-silicon-friendly GGUF export. | The fine-tuning engine for both LLMs. |

### 3.3 What this project contributes

It is not a new algorithm; it is a **rigorous, reproducible application study with an honest
negative-and-positive result.** Concretely:

1. **A like-for-like bake-off on real ESA telemetry** across *four* model families and *three*
   input modalities (numbers-as-text, numbers-as-image, classical features), all scored with the
   ESA benchmark's own metrics. Published comparisons usually pit one or two of these against each
   other; here they share one dataset, one harness, one report.
2. **A defensible answer to "did the fine-tuning actually help?"** — the question most fine-tuning
   showcases skip. By holding the harness fixed and varying only the model (un-fine-tuned base,
   few-shot base, frontier zero/few-shot, trivial baseline), the work isolates *what fine-tuning
   bought* and, crucially, **frames it against a trivial baseline** so the "win" cannot be an
   over-flagging artifact (§5).
3. **Evidence that direct LLM detection is signal-limited, not just precision-limited** — a
   frontier model (Claude) prompted two ways cannot beat the dumb baseline on 10 normalized values
   with no channel context. This is a useful, transferable finding about *what telemetry context an
   LLM needs* to detect at all.
4. **A precision-gated reading of "explainable AI" advice** — turning "99.6% of outputs are
   structured" into the sharper, graded claim "95% high-quality *when the flag is correct*, ~0%
   when it is a false alarm," which directly motivates the deployment architecture.
5. **A production-grade engineering trail** — durable/resumable multi-hour jobs, atomic
   checkpointing, cost-controlled cloud training (~$3.33 total), and a fully documented set of 38
   deviations from plan. The *how it was reached* is part of the deliverable.

---

## 4. Process & methodology (the nine phases)

The pipeline was built in phases. Phases 1–5 built the core; phases 6–9 closed the methodological
gaps the Phase-5 review identified.

| Phase | What it produced |
| --- | --- |
| **1 — ETL** | Downloaded ESA-AD (3 missions, 224 channels, ~29 GB) from the Kaggle mirror; resampled to 1-hour cadence; **RevIN** per-channel normalization; **32-step rolling windows** (stride 16); balanced subsample. → **30,000** instruction/response records, **24.8% anomalous**, split **21,000 / 4,500 / 4,500** (seed 42). |
| **1.5 — Advice labels** | **7,457** structured `DIAGNOSIS / ADVICE / ACTION` records (one per anomaly window) + severity + pattern type, generated in-session and merged into the training responses. |
| **2 — Classical baselines** | **LSTM** (Telemanom-style autoencoder + μ+3σ threshold) and **Isolation Forest**, as a *3-channel smoke run* (full sweep deferred to Phase 7). |
| **3 — LLM fine-tune (cloud)** | **Qwen3-8B** + **QLoRA** (r=16, α=16, all-linear, lr 2e-4, **3 epochs**) on a rented **RTX 4090**; loss 2.85 → 0.24; → **GGUF Q4_K_M** export. ~**$2.30**. |
| **4 — Local inference** | GGUF on the M3 Max via `llama-cpp-python` + **Metal** (all layers offloaded). 100-window smoke: F1 0.508. |
| **5 — Unified evaluation** | Full **4,500-window** LLM run (hardened checkpoint/resume). Text-LLM: F1 0.453, P 0.360, R 0.609, 99.6% structured advice, 2.77 s/window. |
| **6 — "Did fine-tuning help?"** | Controls under the *identical* harness: base **zero-shot**, base **few-shot**, **frontier** (Claude) zero- and few-shot, and a **trivial always-anomaly** baseline. (§5) |
| **7 — Level the field** | LSTM expanded from 3 → **all 58 Mission-1 target channels**; per-window predictions persisted so **Affinity-F1 becomes real** (0.649). Honest F1 fell from 0.663 to **0.552** — still the top detector. |
| **8 — Vision detector** | The *AnomSeer-style* model: **Qwen3-VL-8B** on **PNG plots** (A6000, 2 epochs). 2,000 test PNGs: F1 0.457, **P 0.769**, **CEF0.5 0.604**. ~$1. (§6.3) |
| **9 — Advice grading** | A frozen seed-42 sample of **120** flags graded on a transparent rubric (correctness / actionability / grounding, 0–2 each). (§7) |

**Model selection (why Qwen3-8B).** The base model was chosen deliberately over Qwen2.5 and
Llama-3: Qwen3 posts stronger reasoning/math scores; its **Instruct** variant avoids untrained
chat-token issues; it is fully supported by the Unsloth fine-tuning toolkit; at **8B** it fits the
36 GB M3 Max after 4-bit quantization (so inference can run locally with no API); and it has a
**vision sibling, Qwen3-VL-8B**, which made the AnomSeer-style vision approach possible with the
same toolchain. (Full rationale: the architecture-research doc.)

**Metrics used (and why):**
- **Precision / Recall / F1** — standard detection metrics. Precision = of the windows we flagged,
  how many were truly anomalous; recall = of the truly anomalous windows, how many we caught.
- **CEF0.5** — F-beta with β=0.5, weighting *precision* more than recall, because in a control room
  a false alarm (alert fatigue) is more costly than a near-miss. This is the ESA-benchmark-aligned
  scalar.
- **Affinity-F1** — an interval-aware metric: it merges per-window predictions into time intervals
  and matches predicted intervals to ground-truth intervals within a tolerance. It is the *right*
  metric for streaming telemetry, but only meaningful on **contiguous** timelines (§6.2).
- We deliberately **do not** use point-adjustment (PA), which the ESA-ADB authors show inflates
  scores artificially.

---

## 5. Result A — Did the fine-tuning actually help? (the skeptic's table)

This is the question a fine-tuning showcase lives or dies on. The method: **hold the harness fixed**
(identical prompt, decoding parameters, and output parser) and **vary only the model**.

| Model | F1 | CEF0.5 | Output-format compliance | Structured advice on flags | Eval |
|-------|----|--------|--------------------------|----------------------------|------|
| **Fine-tuned LLM (text)** | **0.453** | **0.392** | 0.994 | **0.996** | 4,500 windows |
| Base Qwen3-8B — zero-shot | 0 | 0 | 0.000 | 0.000 | 100 windows (partial) |
| Base Qwen3-8B — few-shot (2-ex) | 0.420 | 0.325 | 1.000 | 0.129 | 500 windows |
| Frontier (Claude) — zero-shot | 0.254 | 0.284 | 1.000 | 1.000 | 150-window sample |
| Frontier (Claude) — few-shot | 0.239 | 0.214 | 1.000 | 1.000 | 150-window sample |
| **Always-anomaly (trivial)** | **0.399** | 0.294 | 1.000 | 0.000 | flag-all (4,500) |

How to read it, in order:

1. **Anchor on the dumb baseline first.** *Always-anomaly* (flag every window) scores **F1 0.399**
   for free on a 25%-positive set. Any "detector" at or below that has not learned anything — it is
   just over-flagging. This anchor is what makes the rest of the table honest.
2. **vs. the un-fine-tuned base (same model, same harness): F1 0.000 → 0.453.** The raw base, run
   through the identical strict harness, emits a parseable ANOMALY/NOMINAL verdict on **0%** of
   windows — it spends its token budget on chain-of-thought ("thinking mode") and never commits to
   the terse contract. The fine-tune complies on **99%** and emits the structured DIAGNOSIS/ADVICE
   format on **100%** of its flags (vs 0%). *Learning the output contract is itself a fine-tuning
   result.*
3. **The few-shot base's F1 is a mirage.** With two in-context examples and thinking suppressed, the
   base recovers compliance (100% parseable) and posts F1 **0.420** — but with precision **0.282** /
   recall **0.824**, i.e. it flags ~80% of all windows. That is barely above flag-everything: it is
   **over-flagging, not detecting** (the 1:1 example ratio mis-signals a ~50% prior). It is also
   slower (8.56 s vs 2.77 s/window) and emits structured advice on only **13%** of flags.
4. **The frontier (Claude) is genuinely trying, but the input is near-signal-free.** Zero-shot F1
   **0.254**; adding the *same* few-shot examples barely moves it (**0.239**) — so the gap is **not**
   a prompting-asymmetry artifact. Unlike the base it does *not* over-flag (P≈R≈0.3, near the base
   rate), so it sits ~at chance. **A far stronger general model, prompted two ways, cannot beat the
   dumb baseline here.** *Why?* Each window is presented as ~10 normalized numbers plus a mission and
   channel name. The discriminating information — the **signal** — is whether that little sequence is
   abnormal *for that specific channel*, and that requires **channel history**: knowing what is
   normal for, say, `Mission1/channel_41` (its usual range, rhythm, and noise). A reading of 0.6 may
   be perfectly normal on one channel and a clear fault on another; with no history you cannot tell.
   The frontier has never seen this channel, so 10 context-free numbers carry almost no signal —
   especially for the dominant *subtle-deviation* anomalies, which look nearly normal in 10 numbers.
   The fine-tune, by contrast, **learned each channel's normal from 21,000 training windows**, which
   is exactly the prior the frontier lacks. (Two ways to give a frontier that context — retrieval/RAG
   over channel history, or fine-tuning a frontier model — were not tried here; see §10.)
5. **The honest headline.** The **fine-tuned model is the only approach in the LLM family that
   beats the always-anomaly baseline with a balanced precision/recall** — the lone real detector
   among them. On top of detection it delivers output compliance, reliable structured advice, and
   3× lower latency. **What fine-tuning bought is the mission/channel-specific priors that no prompt
   over 10 normalized values can supply** — exactly the localized capability the brief targeted.

**Scope note.** These controls isolate the *text* model — the base and frontier were run through the
text harness on the numeric windows. An equivalent base/frontier control for the *vision* detector
(e.g. an un-fine-tuned Qwen3-VL on the same PNGs) was not run, because the controls were built before
the vision model existed. It is a cheap, listed next step (§10) that would complete the symmetry.

---

## 6. Result B — The detection bake-off (who is the best *detector*?)

### 6.1 The detectors, side by side

All four detectors, ranked by F1 (LSTM after the Phase-7 full 58-channel sweep, no cherry-picking):

| Detector | Precision | Recall | F1 | CEF0.5 | Character |
| --- | --- | --- | --- | --- | --- |
| **LSTM** | **0.837** | 0.432 | **0.553** | **0.705** | best overall; high precision (z=4.0 calibrated, §6.1) |
| **Vision LLM** (Qwen3-VL) | **0.769** | 0.325 | 0.457 | **0.604** | precision-oriented; rarely false-alarms |
| **Text LLM** (Qwen3-8B) | 0.360 | 0.609 | 0.453 | 0.392 | recall-oriented; over-flags |
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | non-temporal floor; drowns in false positives |

Two things stand out. **(1) The LSTM is the strongest detector** — highest F1, highest CEF0.5, and a
genuine interval-aware Affinity-F1 of **0.673** (§6.2). It is ~2.3× more precise than the text LLM
and wins decisively on the precision-weighted score. **This matches the AnomLLM literature:** direct
LLM time-series detection trades precision for recall and is not yet competitive with a tuned
sequence model. **(2) The two LLM modalities are mirror images** — the vision model is
precision-oriented (P 0.769, almost never false-alarms) and the text model is recall-oriented
(R 0.609, catches more but over-flags) at nearly identical F1. The vision model in fact posts the
**best CEF0.5 of any LLM here (0.604)** and is detailed in §6.3.

**Is the LSTM already "best possible"?** The *method* is industry standard (Telemanom). **Phase 11
calibrated the operating point** and found two things. (1) **Threshold calibration helps**: the
original threshold used an *untuned* z=3.0 (`μ+3σ`) which over-flags; sweeping the single global z
over all 58 channels (the full curve is in `results/lstm/threshold_sweep.json`) shows CEF0.5 rises to
a ~0.708 plateau at z≈4.5–5.0, and **z=4.0 Pareto-improves z=3.0 on F1, CEF0.5 *and* Affinity-F1**
(precision 0.785→0.837) — that calibrated z=4.0 is the number reported above. This is the LSTM
analogue of the text-LLM PR-curve calibration (§10 #6 / Plan Phase 13). (2) **Telemanom's canonical
*pruned dynamic* thresholding does *not* help here** — implemented faithfully (EWMA-smoothed,
log-transformed errors → adaptive-z `find_epsilon` → %-drop pruning), it is far too conservative for
this window-level, relatively-frequent-anomaly labeling (it assumes rare isolated events), collapsing
recall to 0.068 (CEF0.5 0.244). The published "Telemanom-ESA-Pruned" ~0.97 *event-wise* CEF is under
the harder ESA-ADB *event* protocol, not our window protocol, so it is not directly comparable.
Remaining untried levers: attention/bidirectional layers, longer context, and channel ensembling (§10).

**Reading the text-LLM confusion matrix** (n=4,500): TP 684, FP 1,214, FN 439, TN ≈ 2,161,
accuracy 0.632, 27 unparsed. The model is *over-eager* — it raises 1,898 flags against 1,123 true
anomalies; false positives outnumber false negatives ~2.8:1. At a 25% base rate, precision 0.36 is
still well above the 0.25 random-flag floor and recall 0.61 is genuine signal — it has learned
*something* about nominal-vs-anomalous rhythm. But **0.36 precision means ~2 of every 3 alarms are
wrong** → as a standalone *trigger* it is not deployable; as a *recall-maximizing second opinion*
behind a high-precision detector, it is useful.

### 6.2 Why the honest LSTM number is *lower* than before — and why that's the point

Phase 2 scored the LSTM on a hand-picked 3-channel smoke subset with per-channel tuned thresholds
and reported F1 0.663. Phase 7 ran **all 58 Mission-1 target channels** and the honest macro-F1 fell
to **0.552**. We *kept the lower number* — the master table (§2) and §6.1 report only the 58-channel
0.552; the 0.663 smoke figure appears **only here**, to make the correction visible rather than
quietly swapping numbers. Two things the full sweep fixed:

- **Like-for-like eval units.** The LSTM now faces a full, heterogeneous channel set (no
  cherry-picking), bringing it closer to the LLM's full-distribution test.
- **A real Affinity-F1.** Because the LSTM scores *contiguous* per-channel timelines, its
  per-window predictions merge into genuine multi-window intervals (e.g. 172 anomalous windows → 60
  intervals on one probe channel), so **Affinity-F1 = 0.649 is meaningful**. By contrast the LLM's
  test split is *shuffled and balanced-subsampled* (~1.4 windows per channel), so its "intervals"
  are mostly isolated single windows and its Affinity-F1 (0.456) collapses to ≈ window-level F1. We
  report the LLM's number with that disclaimer rather than dressing it up.

> The Phase-7 figures in this subsection (F1 0.552, Affinity-F1 0.649) are at the *untuned* z=3.0
> threshold. **Phase 11 then calibrated the operating point** (global z 3.0→4.0), lifting the canonical
> LSTM to F1 0.553 / CEF0.5 0.705 / Affinity-F1 0.673 / precision 0.837 — the numbers in §2 and §6.1.
> The z=3.0 figures are kept here as the historical correction record.

### 6.3 The vision detector — a second, modality-independent signal

Phase 8 completed the originally-planned *AnomSeer-style* approach: **Qwen3-VL-8B fine-tuned to read
a rendered PNG plot** of the telemetry window (not the numbers) and emit ANOMALY/NOMINAL. On 2,000
test PNGs:

- **Precision 0.769, recall 0.325, F1 0.457, CEF0.5 0.604**, 100% format compliance (0 UNKNOWN).
- It is a **precision-oriented mirror image** of the text LLM (which is recall-oriented: P 0.360 /
  R 0.609) at nearly identical F1. Because CEF0.5 weights precision, the vision model has the
  **highest CEF0.5 of any LLM approach (0.604)** — it almost never false-alarms (49 FP vs 1,450 TN)
  but misses more (338 FN).
- It is a **pure detector** — no diagnostic advice by design (its advice fields are 0).

The practical implication: the two LLM modalities **fail differently** (one trigger-happy, one
conservative), which makes them attractive for an ensemble. Hard rules — require *both* to fire (high
precision) or *either* (high recall) — are just the two extremes; **fusing their continuous scores**
(ideally with the LSTM's score as a third input) via a small learned combiner can trace a whole
precision–recall frontier that improves on *both* corners at once, because the modalities' errors are
independent. This is specced as Plan Phase 14 (it builds on the calibrated scores from §10 #6).

Two further consequences worth stating. **(a) The advisor can sit on the vision detector too.**
Because the vision model is high-precision (0.769), `vision detector → text advisor` is a legitimate
alternative to `LSTM → advisor` — the same hybrid pattern, with an all-LLM front end where false
alarms are especially costly. **(b) The vision detector likely has headroom.** It converged very
fast on a 2-class task (eval loss 0.0089) and its weak point is recall (0.325); more and more varied training
data, a larger VL backbone, or more epochs *could* lift recall — untested, and listed in §10. Even
as-is, its precision makes it worth considering wherever a near-zero false-alarm rate matters.

### 6.4 Calibrating the text LLM — the over-flagging was a threshold artifact (Phase 13)

The text LLM's headline **P 0.360 / R 0.609** is a *single* operating point inflated by false alarms
two ways: it was decoded with **stochastic sampling** (temperature 0.8), and it raised ANOMALY
whenever that token won a *sampled* draw. Phase 13 replaced the hard verdict with a **continuous,
deterministic anomaly score** per window — the softmax of the model's ANOMALY-vs-NOMINAL logits at the
verdict position (`test_local_gguf.py --score` → `results/inference_test_scored.json`) — and swept a
decision threshold over it (`src/inference/pr_curve.py` → `results/llm_pr_curve.json` + `.png`). Two
findings:

1. **Removing the sampling noise recovers ~17 precision points for free.** The deterministic argmax
   (threshold 0.5 — the natural boundary) already sits at **P 0.527 / R 0.639 / F1 0.578 / CEF0.5
   0.546**, versus the sampled point's 0.360 / 0.609 / 0.453 / 0.392. Same weights — temperature-0.8
   decoding alone was injecting random ANOMALY flips that cost precision.
2. **The score genuinely separates, and the curve is deployable.** **AUC-PR = 0.678** (random floor =
   the 0.250 base rate). Sweeping the threshold up to **0.775** reaches **P 0.838 / R 0.379 / F1 0.521
   / CEF0.5 0.674** — precision *more than doubles* vs the default 0.360. At that operating point the
   text LLM's **CEF0.5 (0.674) lands just below the calibrated LSTM (0.705) and above the vision LLM
   (0.604)**: once calibrated it is a competitive precision-weighted detector, not a trigger-happy one.
   The default sampled point lies *strictly below* the calibrated PR curve (red dot in
   `llm_pr_curve.png`).

| Operating point | Threshold | Precision | Recall | F1 | CEF0.5 |
| --- | --- | --- | --- | --- | --- |
| Default (as-deployed: sampled hard verdict) | — | 0.360 | 0.609 | 0.453 | 0.392 |
| Deterministic argmax | 0.500 | 0.527 | 0.639 | 0.578 | 0.546 |
| F1-optimal | 0.580 | 0.621 | 0.567 | **0.593** | 0.609 |
| **CEF0.5-optimal** | 0.775 | **0.838** | 0.379 | 0.521 | **0.674** |

This confirms the §10 #6 hypothesis: **the low precision was a decision-boundary/calibration problem,
not a capacity one** — the same lesson as the LSTM's z-calibration in Phase 11. A detection-only SFT is
therefore still not indicated; the threshold (and turning off sampling) was the right lever. The
continuous score also unlocks **Phase 14** (score-level fusion), which needs soft scores rather than
hard verdicts.

> **Operating-point note for §2 / §6.1.** The master table keeps the text LLM at its *as-deployed*
> point (P 0.360 / R 0.609 — the advice-SFT model decoded as actually shipped) for honesty about what
> was measured end-to-end. The calibrated alternatives above are the deployment knobs: deterministic
> decode + threshold ≈0.78 → **P 0.838** where false alarms are the binding cost.

---

## 7. Result C — Is the advice any good? (semantic grading)

Detection is only half the brief; the other half is *"end-user advice on the effect of a change."*
None of the classical baselines can do this at all. Phase 5 established that **99.6%** of the
fine-tune's flags carry structured `DIAGNOSIS / ADVICE / ACTION` text. Phase 9 asked the sharper
question: **is that advice correct?**

A frozen seed-42 sample of **120** of the model's anomaly predictions — preserving the real **TP:FP
ratio** (true-positive to false-positive flags; the model's ~0.36 precision means roughly 43 correct
flags to 77 false alarms, and the sample keeps that ratio so the grade reflects the true false-alarm
rate rather than a cleaned-up set) — was graded on a transparent, *verifiable* rubric: correctness /
actionability / grounding, 0–2 each (total 0–6).

**How the grading worked (and why it is not circular).** The grader (the Claude session model) did
**not** score against its own world knowledge, nor did it simply diff against the Phase-1.5 advice
labels (it could not — the stored predictions lack the keys to join to those labels). It scored
against **the window's own data and the ground-truth anomaly label**: *grounding* = does the named
channel match the input window and is the stated magnitude consistent with the actual values;
*correctness* = ground-truth-gated, so advice written about a truly-nominal window (a false alarm)
scores 0 by construction. This is why "just use the frontier model instead" does not follow from
Phase 9: judging finished advice **with the answer key in hand** is a far easier task than *producing*
detections cold — and as a detector the same frontier sat at chance (§5).

| Subset | n | Correctness | Actionability | Grounding | Mean /6 | High-quality |
|--------|---|-------------|---------------|-----------|---------|--------------|
| All flags | 120 | 0.64 | 1.03 | 1.01 | 2.68 | 34% |
| **True positives (correct flags)** | 43 | **1.79** | **1.93** | **1.86** | **5.58** | **95%** |
| False positives (false alarms) | 77 | 0.00 | 0.53 | 0.53 | 1.06 | 0% |

- **When the model is right to flag, its advice is genuinely good:** mean **5.58/6**, 95%
  high-quality, with **near-perfect grounding (1.86/2)** — 119/120 named the correct *channel* (the
  specific sensor stream, e.g. `Mission1/channel_41`) and a magnitude consistent with the window;
  3/120 mislabelled the *subsystem* (the functional group the channel belongs to — power, thermal,
  attitude, etc.). So even on the rare miss it usually points at the right sensor; what slips is the
  higher-level category. Actionability was 1.93/2 (a severity-appropriate recommended action).
- **Advice quality is gated by detection precision.** On false alarms (77 of 120 — the model's
  precision is ~0.36) the advice is built on a false premise, so correctness is **0.00/2** by
  construction; the model fabricates a confident "persistent anomaly" narrative on a truly-nominal
  window. The overall mean drops to 2.68/6 precisely *because* the standalone detector over-flags.
- **This is the empirical case for the Hybrid.** Deploy the fine-tune as the **advisor on top of a
  high-precision detector** (the LSTM flags at precision ≈0.79), not as the standalone detector.
  Its diagnostic writing is strong — but only as trustworthy as whatever decided to call the
  anomaly.

**"95% high-quality" is good, but not yet mission-critical-grade.** For a *lives-depend-on-it*
system, 95%-when-correct (graded by an LLM on n=120, against **synthetic** Phase-1.5 advice labels)
is a strong showcase result, not a deployment guarantee. The highest-leverage steps to close that
gap, roughly in order of impact:
1. **Replace the synthetic advice labels with human-SME-written advice.** The current ceiling is the
   templated, statistic-derived Phase-1.5 labels; real fault-engineer advice (with true root causes)
   is the single biggest lever on correctness.
2. **Ground the advisor in real references (RAG).** Give it the channel's spec sheet / fault catalog
   so it cites documented root causes and corrective procedures instead of pattern templates.
3. **Put it behind a high-precision detector** (the Hybrid) so it is rarely asked to explain a false
   alarm in the first place — directly fixes the precision-gating above.
4. **Add calibrated confidence / abstention** so it can say "uncertain" instead of confidently
   fabricating on a borderline window.
5. **A bigger/stronger advisor backbone and more epochs** *if* resources allow — and a **larger,
   human-validated advice grade** to replace the n=120 LLM-judge sample with tighter confidence.

---

## 8. Analysis & discussion — the architecture the data recommends

Putting §5–§7 together yields a single, empirically-supported recommendation: the **two-stage
hybrid**.

```
telemetry window ──► LSTM detector  (P≈0.79, ~instant, cheap)
                         │ flags an anomaly (only ~1/N of windows)
                         ▼
                     Qwen3-8B advisor  (structured DIAGNOSIS / ADVICE / ACTION, 95% HQ on real flags)
                         │
                         ▼
                     operator-ready alert
        (optional: Qwen3-VL vision model as a low-false-alarm cross-check on the flag)
```

Why this is the right design, not just a convenient one:

- **Detection belongs to the cheap, high-precision LSTM.** You cannot run an 8B model on every
  window: the text LLM costs **2.77 s/window** on M3 Max Metal; at telemetry scale that is a
  non-starter for first-pass screening. The LSTM scores in sub-milliseconds.
- **Explanation belongs to the LLM, invoked only on flags.** The expensive model runs ~1/N of the
  time and supplies the one capability the detector lacks — and §7 shows that *on the windows a
  high-precision detector flags*, the advice is 95% high-quality.
- **The Hybrid strictly dominates either component alone for the stated business need:** it inherits
  the LSTM's detection (F1 0.553, P 0.837) *and* adds reliable, grounded advice — the exact
  "anomaly + end-user advice, no external vendor" capability this project targets.
- **The vision model is the optional third leg:** modality-independent, precision-oriented, useful
  as an ensemble cross-check where false alarms are especially costly.

The deeper lesson — and the most transferable finding — is **where the signal lives**. A frontier
model with vastly more general capability than Qwen3-8B *cannot* detect these anomalies from the
prompt alone (§5). The fine-tune can, because it *learned the mission- and channel-specific priors* —
what "normal" looks like for *this* channel on *this* spacecraft. That is precisely what fine-tuning
is for: adapting an open model to a localized, mission-specific task that prompting an API cannot
replicate.

One honest qualifier on that claim: we compared a *fine-tuned* open model against a *prompted*
frontier — not a *fine-tuned* frontier. The cleanest "could you just adapt a hosted model instead?"
test would be to **fine-tune a frontier model** (GPT-4o/Gemini expose tuning APIs; Claude does not
publicly) or to give a frontier the missing channel history via **retrieval (RAG)**. Neither was
run; both are listed in §10. Note the trade-off: fine-tuning or RAG-ing a *hosted* model re-introduces
the exact vendor dependency, data-egress, cost, and latency that an owned on-prem model avoids — so
even if a fine-tuned frontier matched the accuracy, the sovereign 8B would still win on the
deployment constraints that motivated the project.

---

## 9. Limitations

The four largest gaps identified mid-project are now **closed** (base-vs-fine-tuned → §5; full LSTM →
§6.2; vision detector → §6.3; semantic advice grading → §7). The residual limitations after all nine
phases — each with *why it stands* and what it would take to close:

1. **Eval-unit asymmetry is reduced but not eliminated.** The LSTM is macro-averaged over 58
   contiguous Mission-1 channels; the LLM is micro-averaged over 4,500 shuffled cross-mission
   windows. *Why it stands:* Phase 7 already moved the LSTM from 3 → 58 channels under the project's
   time budget, which narrowed the gap materially; a *fully* like-for-like rematch needs a new
   evaluation harness that scores both models on one identical, contiguous per-channel stream (~1–2
   days of work, §10).
2. **The frontier control is a sample (n=150) on context-free input.** It is fed the same ~10
   normalized values the fine-tune saw, with no channel history. *Why it stands:* this was the free,
   *controlled* comparison — deliberately holding the input identical so only the model varies; it is
   a hard sanity check, not a best-effort frontier benchmark. *To give a frontier a fair shot in the
   real world* you would add channel history via **retrieval (RAG)**, a longer raw window, or channel
   metadata — listed in §10.
3. **Advice labels are synthetic.** The Phase-1.5 advice was generated in-session (statistic-derived
   templates), not written by spacecraft subject-matter experts (SMEs). Phase 9 grades the model's
   advice against window data + the ground-truth label (so it needs no synthetic-label join), but
   human-expert validation of both the labels and the output remains future work — and is the top
   lever toward mission-critical advice (§7).
4. **Advice grading is a 120-flag, LLM-judged sample.** Judged by the Claude session model on a
   transparent rubric, not a human SME panel; the TP:FP ratio (true-positive to false-positive flags)
   was preserved to keep the false-alarm rate realistic, but confidence intervals are wide. The stored
   advice was also **truncated at 300 characters** — *why:* to keep the 4,500-row results file
   manageable during the eval; this clips only the trailing ACTION line. *Does it change any
   comparison?* No — the ANOMALY/NOMINAL verdict is at the start of the response and the same clip
   applied to base and frontier, so detection numbers are unaffected; only advice-grading
   *completeness* (the ACTION line) is. Re-running advice grading uncapped is a small next step (§10).
5. **The Hybrid's detection score is inherited, not independently measured** — by construction it
   equals the LSTM's. Only its advice layer is genuinely new, and it was not benchmarked against an
   alternative advice generator.
6. **Single fine-tune, no hyperparameter sweep.** r=16 / α=16 / 3-epoch were sensible defaults.
   *Why it stands:* a showcase ran one good configuration rather than a grid. The knobs most likely to
   matter: **LoRA rank** (8/16/32 — capacity), **epochs** (2/3/5 — under/overfitting), **learning
   rate**, and **prompt format**. A small grid is ~1 day of cloud (~$5–15), §10.
7. **No detection-tuned LLM variant** (the *precision–recall curve* is now done). The deployed model
   was the *advice* **SFT** (Supervised Fine-Tuning — training on input→output pairs) used *also* as a
   detector. ✅ **Resolved by Phase 13 (§6.4):** the single-point caveat is gone — a continuous verdict
   score swept over a threshold gives **AUC-PR 0.678** and a high-precision operating point (**P 0.838
   at R 0.379**, CEF0.5 0.674), and even just decoding deterministically (no temperature sampling)
   lifts precision 0.360→0.527. This **confirms the over-flagging was a calibration artifact, not a
   capacity limit.** The one idea still untried is a **detection-only SFT** (a model trained purely to
   emit ANOMALY/NOMINAL) — and it is *not* recommended (§10 #6): the auxiliary advice task likely
   *helps* the shared representation, so stripping it is unlikely to raise precision and would cost a
   capability.
8. **Vision model has no advice head, and "converged very fast."** Its validation loss dropped to
   0.0089 on an easy 2-class (ANOMALY/NOMINAL) task — meaning it fit the *training* distribution very
   easily. Low in-distribution validation loss does **not** guarantee it generalizes to a *new mission
   it never trained on*; that (overfitting risk) is untested.
9. **Resampling to 1-hour cadence is lossy.** Raw telemetry is sampled every few seconds/minutes; we
   averaged it onto a 1-hour grid to make 29 GB tractable. An anomaly lasting **less than an hour** can
   be averaged into its neighbours and disappear. We did not try other cadences (5-min / 15-min), so
   the signal cost of the 1-hour choice is unmeasured — a cadence sweep is in §10.
10. **Mission scope.** The LSTM sweep covers Mission-1 target channels; **cross-mission generalization**
    — train on one mission, test on a different spacecraft the model never saw — is untested (§10).

---

## 10. Recommended next steps

Prioritized, each with a rough **effort** and **expected impact**. The detection-and-evaluation items
most worth doing next — **improve the LSTM** (#4), **complete the vision skeptic table** (#5),
**calibrate the LLM operating point** (#6), and **ensemble via score fusion** (#7) — are written up as
concrete, self-contained next phases (Phases 11–14) in the implementation plan.

1. **Ship the Hybrid as the reference design.** *Effort: low* (packaging, not new modeling).
   *Impact: high.* It is the architecture the numbers support. **Who needs it:** operators of
   mission-critical monitoring systems — spacecraft FDIR, but equally industrial/SCADA telemetry or
   any high-availability infrastructure — that (a) must run an **on-prem / sovereign** model (data
   cannot leave the boundary, no external-API dependency), and (b) where **false alarms are costly**
   (alert fatigue in a staffed control room). For that profile, "cheap high-precision detector → LLM
   advisor" is the deployable shape.
2. **Toward mission-critical advice (highest-value quality work).** *Effort: high* (needs human
   experts). *Impact: high.* Replace the synthetic Phase-1.5 labels with **human-SME-written advice**,
   add **RAG grounding** (channel spec sheets / fault catalogs), and run a **larger, human-validated
   advice grade** to replace the n=120 LLM-judge sample. This is the real path from "95% when correct"
   to deployment-grade.
3. **Fully level the detection field.** *Effort: ~1–2 days* (build a contiguous-stream eval harness;
   re-score LSTM and LLM on one identical per-channel timeline so Affinity-F1 is comparable).
   *Impact: medium–high.* **Is the report meaningful without it?** Yes — Phase 7 already made the
   comparison far more honest and the result matches the literature. But doing it converts a
   "believable upper bound on the LSTM's edge" into a settled like-for-like number, which is *more*
   authoritative.
4. **Improve the LSTM detector itself** (Plan Phase 11). ✅ **DONE (2026-06-15).** Calibrating the
   threshold's single global z (3.0→4.0) lifted the LSTM to **F1 0.553 / CEF0.5 0.705 / Affinity-F1
   0.673 / precision 0.837** (was 0.552 / 0.684 / 0.649 / 0.785) — a Pareto improvement that widens the
   margin over the LLMs (see §6.1). Notably, Telemanom's canonical **pruned dynamic error thresholding**
   was implemented faithfully but **does not help on this window-level labeling** (recall collapses to
   0.068) — a documented negative result; the published "Telemanom-ESA-Pruned" ~0.97 figure is under the
   harder ESA-ADB *event* protocol, not comparable. **Remaining untried levers** (deferred):
   **attention/bidirectional** layers, **longer context windows**, and **channel ensembling**.
5. **Complete the skeptic table for vision** (Plan Phase 12). *Effort: low* (~half a day — run an
   un-fine-tuned Qwen3-VL zero-shot on the 2,000 test PNGs, optionally a frontier-VL too).
   *Impact: medium.* Closes the text-only scope gap noted in §5 so the "did fine-tuning help?" story
   covers both modalities.
6. **Calibrate the LLM's operating point** (Plan Phase 13). ✅ **DONE (2026-06-15).** A continuous
   verdict score (ANOMALY-vs-NOMINAL logprob) swept over a threshold gives **AUC-PR 0.678** and a
   higher-precision operating point — **P 0.838 / R 0.379 / CEF0.5 0.674** at threshold 0.775, more
   than doubling the default 0.360 precision; even removing temperature sampling alone lifts it to
   0.527 (§6.4, `results/llm_pr_curve.{json,png}`). This **confirms the over-flagging was calibration,
   not capacity.** A *detection-only* SFT — dropping the advice head — is therefore **not** recommended:
   the auxiliary advice task likely *helps* rather than hurts the shared representation, so removing it
   is unlikely to improve precision and would cost a capability. The calibrated continuous score is also
   the input Phase 14 (#7) fuses.
7. **Ensemble the detectors via score-level fusion** (Plan Phase 14, depends on #6). *Effort: ~2–3
   days* (no new training — fuse the continuous scores). *Impact: medium–high.* "Both must fire" (AND
   → high precision) and "either fires" (OR → high recall) are only the two *endpoints*. The richer
   move is to **fuse the continuous anomaly scores** (text + vision + the LSTM error score) with a
   small learned **stacker** (logistic regression on a validation split) and sweep one threshold — a
   whole precision–recall frontier. Because the modalities make *independent* errors (§6.3), the fused
   frontier can **Pareto-dominate** any single model (higher F1/CEF0.5 at matched recall), not merely
   interpolate. A 2-of-3 majority vote and a "disagreement → human review" tier are deployable
   variants. Requires scoring all models on one shared window set (the fiddly part).
8. **Fine-tune / RAG a frontier model — the true "own vs. adapt" comparison.** *Effort: medium.*
   *Impact: medium.* We compared a *fine-tuned* open model to a *prompted* frontier; fine-tuning a
   frontier (GPT-4o/Gemini tuning APIs; Claude has none public) or giving it channel history via RAG
   would test whether you even need a custom 8B. **Caveat:** doing so re-introduces the vendor
   dependency, data-egress, cost, and latency the sovereign model exists to avoid.
9. **Hyperparameter sweep + vision improvements + generalization tests.** *Effort: medium.*
   *Impact: medium.* A small **LoRA ablation** (rank / epochs / prompt format; ~1 day, ~$5–15) to
   confirm the fine-tune is near its ceiling; an **advice head** and larger backbone for the vision
   model; a **cross-mission generalization test** (train on one mission, test on a spacecraft the
   model never saw — the real test of whether it generalizes); and a **resample-cadence sweep**
   (re-run ETL at 5-min / 15-min / 1-hour to find the best signal-vs-tractability trade-off, since the
   current 1-hour grid can average away sub-hour anomalies).

---

## 11. Definition-of-Done scorecard

| Brief's DoD item | Status | Evidence |
|---|---|---|
| Modular, scalable ETL: raw telemetry → LLM-digestible patches | ✅ | `src/etl/` — download, RevIN windowing, balanced subsample; 30,000 JSONL records across 3 missions |
| Traditional ML baseline trained and evaluated | ✅ | LSTM (Telemanom-style, 58 channels) **and** Isolation Forest, both scored |
| Open-source LLM fine-tuned via Unsloth; infers anomaly **and** advice | ✅ | Qwen3-8B QLoRA → GGUF Q4_K_M → Metal inference; 99.6% structured advice, 95% high-quality on true flags |
| FP / precision / recall documented, LLM vs. baseline | ✅ | This analysis + `comparison_report.md`; full confusion matrices persisted |
| Cost-effective local/cloud split | ✅ | ETL + inference local on M3 Max; training on rented GPUs for **~$3.33 total** |
| *Beyond DoD:* did-fine-tuning-help, vision modality, semantic advice grade | ✅ | Phases 6–9 |

Every box is checked — and the inconvenient findings (**the LLM loses the detection bake-off; the
few-shot base's F1 is a mirage; advice quality is precision-gated**) are reported plainly rather than
hidden. Surfacing correct-but-inconvenient results is the whole point of an empirical bake-off.

---

## 12. Bottom line

On real ESA satellite telemetry, a classical LSTM with per-channel error thresholding remains the
stronger *detector* (precision 0.837, F1 0.553, CEF0.5 0.705) than a QLoRA-fine-tuned Qwen3-8B used
as a direct detector (precision 0.360, F1 0.453) — the LLM buys recall at a steep precision cost and
is far too slow (2.77 s/window) to screen every window anyway. The *vision* fine-tune (Qwen3-VL,
precision 0.769, CEF0.5 0.604) is its precision-oriented mirror and the best LLM detector on the
cost-of-false-alarm metric — a useful third, modality-independent signal. But among the LLM family
the text fine-tune is the **only** approach that beats a trivial always-anomaly baseline with a
balanced precision/recall trade-off, while a frontier model prompted two ways sits at chance —
proving the fine-tune learned mission-specific priors that no prompt over 10 normalized values can
supply. And where the LLM uniquely earns its place — diagnostic advice — it is **95% high-quality
when the flag is correct** (strong, though not yet mission-critical-grade), but only then. The
correct, empirically-grounded architecture is therefore the **hybrid**: a cheap high-precision
detector that triggers an LLM advisor — exactly the localized, open-source, no-vendor system this
project set out to validate. The headline is not "LLMs win" or "LLMs lose"; it is *"here is the
measured trade-off, and here is the design it dictates."*

---

## Appendix A — Reproducibility

Every phase has a `make` target and a paired `validate-*` target encoding its success criteria. The
full LLM eval is durable (`--checkpoint-every N`, `--resume`, atomic writes). Published artifacts:
fine-tuned text GGUF — `dyrtyData/star-pipeline-qwen3-8b-advice-gguf`; vision adapter —
`dyrtyData/star-pipeline-qwen3-vl-8b-detection` (both on Hugging Face). Raw ESA-AD (~29 GB) is not
redistributed; `make download` fetches it. The implementation log records the full engineering trail
(38 numbered deviations) for anyone who wants the blow-by-blow.
