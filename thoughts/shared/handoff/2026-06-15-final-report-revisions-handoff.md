# HANDOFF — Final-report revisions (fresh-thread continuation)

**Date:** 2026-06-15
**Trigger:** Pre-autocompact handoff. User supplied a revision list in
`thoughts/shared/issues/FinalReport_FollowUp.md`. This doc captures the full task + pre-computed
answers so a fresh thread can execute without re-deriving.

---

## 0. What this task is

Revise the project's TWO final, public-facing documents per the user's follow-up list:
- **A** = `thoughts/shared/reports/2026-06-14-results-analysis.md` (rigorous report)
- **B** = `thoughts/shared/reports/2026-06-14-plain-language-walkthrough.md` (learn-from-scratch companion)

The user's revision requests are in `thoughts/shared/issues/FinalReport_FollowUp.md` (items 1–14;
there is no item 12). This repo is going PUBLIC, so the framing must read as a standalone public
report, NOT as answers to an interviewer/prompt. **Track everything with git** (the user works in
small atomic commits with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; do NOT push
unless asked; on `main` is fine here — prior commits were straight to main).

Several requests also ask to ADD next-steps / possible new phases to the plan doc
`thoughts/shared/plans/2026-06-12-star-pipeline-create-plan.md` (see §4 below).

---

## 1. Project state (so you have the numbers without re-reading everything)

9-phase project: fine-tune open LLMs for ESA satellite-telemetry anomaly detection + advice.
**Final master comparison** (source of truth = `results/comparison_metrics.json`, test split = 4,500
windows, 25.0% anomalous):

| Approach | P | R | F1 | CEF0.5 | Affinity-F1 | unit |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | — | 3 ch |
| **LSTM (58 ch)** | **0.785** | 0.451 | **0.552** | **0.684** | **0.649** | 58 ch |
| LLM text (Qwen3-8B QLoRA) | 0.360 | 0.609 | 0.453 | 0.392 | 0.456‡ | 4500 win |
| LLM vision (Qwen3-VL-8B) | 0.769 | 0.325 | 0.457 | 0.604 | — | 2000 PNG |
| Base Qwen3-8B zero-shot | 0 | 0 | 0 | 0 | — | 100 win |
| Base Qwen3-8B few-shot (2-ex) | 0.282 | 0.824 | 0.420 | 0.325 | — | 500 win |
| Frontier (Claude) zero-shot | 0.308 | 0.216 | 0.254 | 0.284 | — | 150 win |
| Frontier (Claude) few-shot | 0.200 | 0.297 | 0.239 | 0.214 | — | 150 win |
| Always-anomaly (trivial) | 0.250 | 1.000 | 0.399 | 0.294 | — | 4500 win |
| Hybrid (LSTM+LLM advice) | 0.785 | 0.451 | 0.552 | 0.684 | 0.649 | inherits LSTM |

‡ degenerate on shuffled split. Text-LLM confusion (n=4500): TP 684, FP 1214, FN 439, TN ~2161, 27
unparsed, acc 0.632, 2.77 s/win, 99.6% structured advice. Vision (n=2000): TP 163, FP 49, FN 338,
TN 1450, 100% compliance, 0.86 s/win, ~$1 train. Advice grade (Phase 9, n=120 = 43 TP/77 FP):
All 2.68/6 (34% HQ); TP 5.58/6 (95% HQ, grounding 1.86, 119/120 right channel, 3/120 wrong
subsystem); FP 1.06/6 (0% HQ). Training: text r16/α16/all-linear/lr2e-4/3ep RTX4090 ~$2.30; vision
r16/α16/lr1e-4/2ep A6000 ~$1; total cloud ~$3.33. Dataset: 30,000 windows (24.8% anom),
21k/4.5k/4.5k, 32-step windows stride 16, 1h resample, RevIN.

**5 public-prep commits already done** (latest `8552ac2`): docs rewrite, hygiene/de-personalize, HF
cards (`huggingface/`), contribution playbook (`contributions/`), competitive assessment. Don't redo
those.

---

## 2. REVISION CHECKLIST — every item, with what to change + pre-computed answers

Apply to A (results-analysis) and B (walkthrough) as noted. Item numbers match the follow-up doc.

### Item 1 — Internal consistency
After all edits, pass through BOTH docs and check consistency within each and between them (numbers,
section names, claims). Notable: B currently calls the LSTM "best detector" and barely mentions
vision; A's §6.1 also omits vision — both need the vision model surfaced (see items 7/8).

### Item 2 (general consumption — both docs, mostly B)
- **B §10 "Interview Q&A (anticipate these)"** → rename to neutral **"FAQ"** (or "Common questions");
  drop "anticipate these." It's a public repo.
- **B "hardest engineering part" answer** → REPLACE the eval-durability answer (user says that was
  quick/easy). Choose something more impressive to a domain SME. **Best candidates:** (a) reverse-
  engineering the actual ESA-AD on-disk structure (per-channel pickles, categorical Mission-3
  channels) and building the RevIN windowing/balanced-subsample ETL that turns 29 GB of raw multi-
  mission telemetry into a clean 30k-window instruction set; or (b) designing the *controlled*
  "did-fine-tuning-help" evaluation (identical harness, trivial-baseline anchor, frontier control)
  that makes the fine-tuning claim survive a skeptic. Recommend (b) — it's the methodological
  highlight.
- **A §"Honest limitations (state these before an interviewer does)"** → rename to just
  **"Limitations"**.
- **A §"One-paragraph takeaway"** → rename to **"Executive summary"** or **"TL;DR"** (note: A already
  opens with an "Executive summary" §1 — use **"Bottom line"** or **"In one paragraph"** to avoid a
  duplicate heading, OR move/merge. Check for the clash during the consistency pass.)

### Item 2/FINAL REPORT (A — de-prompt the framing). Reword these so they read as a report, not as
answers to a user prompt:
- "The brief explicitly asked the agent to *survey the field…*" → e.g. "This project deliberately
  surveys the field and validates options rather than committing to the first idea. Here is the
  landscape it sits in."
- "A naïve project picks one and asserts it is best. This project builds and measures both…" → keep
  the substance, drop the "naïve project" framing: "Rather than assert one approach, this project
  builds and measures both… so the recommendation is empirical."
- "which is exactly the senior-engineering signal the brief asked for." → delete the
  brief/senior-signal clause; state the property directly ("…a rigorous, reproducible comparison").
- "That is the precise thing fine-tuning is for, and the precise thing the brief wanted
  demonstrated:…" → "That is precisely what fine-tuning is for: adapting an open model to a
  localized, mission-specific task that prompting an API cannot replicate."
- "**Ship the Hybrid as the reference design** — …the one the business needs." → name the business
  type: operators of **mission-critical monitoring systems** (spacecraft FDIR, industrial/SCADA
  telemetry, any high-availability system) **that need on-prem/sovereign models** (data can't leave
  the boundary, no external-API dependency) and where **false alarms are costly** (alert fatigue in a
  control room). Frame the hybrid as the design for that profile.

### Item 3 (A — remove meta/process-history & interviewer references; this is a final report)
DELETE these passages outright:
- The "**Note on this document's history**" callout block (the Phase-5-earlier-version note).
- "The earlier (Phase-5) version of this analysis listed 'no base-vs-fine-tuned comparison' as the
  *single biggest gap*. This table is that gap, closed — and reframed so it survives a skeptic."
  → keep only a clean lead-in to the table (e.g. "This isolates the fine-tuning effect…").
- "(This is a systems-level point an interviewer will probe.)" → delete the parenthetical.
- The DoD §11 "Every box is checked, and — more important for a senior signal — the inconvenient
  findings… The brief asked the agent to validate and explain…" → reword to state the findings are
  reported transparently, without the brief/senior-signal meta.
- **Appendix B (The deviation trail)** → user says remove it. DELETE Appendix B. (The deviations
  live in the implementation log already; the report doesn't need the 38-deviation paragraph.)
  Check that nothing else references "Appendix B."

### Item 4 (B — make the "what would you do next" FAQ answer actually human-understandable)
The terse list ("ship the hybrid; level the detection field; calibrate threshold; widen advice
grade; ensemble") must be FULLY fleshed out — each item explained in plain language WITH the *why*.
Use the plain-language explanations from §3 of this handoff (channel history, calibrated threshold,
ensemble, etc.).

### Item 5 (A + B — advice quality: reframe 95% as "good but not mission-critical-enough," explain
the judge, explain channel/subsystem). THIS IS IMPORTANT — user had a key realization:
- **95% is NOT enough for "lives depend on it" mission-critical.** Report 95% as *generally good*,
  then add concrete recommendations to push toward production-grade (see §3 "how to improve advice").
- **Clarify the LLM-as-judge ≠ source-of-truth confusion (the user's realization is correct):**
  In Phase 9 the Claude judge did **NOT** use its own knowledge as ground truth, and did **NOT**
  simply diff against the Phase-1.5 advice labels (it couldn't — `inference_test.json` lacks the
  `pattern`/`start_time` keys to join to the gold advice; gold advice was an *optional reference*).
  The judge scored on a **transparent, verifiable rubric keyed to the ground-truth `is_anomaly`
  label + the window's own statistics**: grounding = does the named channel match the input window's
  channel, is the stated magnitude consistent with the actual values; correctness = GT-gated (a flag
  on a truly-nominal window scores 0 by construction). So the judge is checking the model's advice
  against *the window + the known answer key*, not against the judge's world knowledge.
  → **Therefore "just use the frontier model instead" does NOT follow:** (a) as a *detector* the
  frontier sat at chance (F1 0.25, below the trivial baseline) — it cannot do the detection job at
  all on this input; (b) the judge role (grading finished advice with the answer key in hand) is a
  much easier task than the detector role and says nothing about the frontier's ability to *produce*
  detections/advice cold; (c) sovereignty/privacy, cost, latency, and API-dependency still favor the
  owned model. The real apples-to-apples "could you just adapt a frontier model?" test would be to
  **fine-tune a frontier model** (or give it channel context via RAG) — which we did NOT do (Claude
  has no public fine-tuning; some frontiers like GPT-4o/Gemini do) → add to next steps (item 7).
- **Explain "wrong channel" / "subsystem" with an example.** A *channel* is one sensor's telemetry
  stream (e.g. `Mission1/channel_41`, say a battery-bus voltage). A *subsystem* is the functional
  group it belongs to (power, thermal, attitude/AOCS, propulsion, comms). "Named the right channel"
  = the advice correctly says the anomaly is in channel_41 (not channel_12). "Mislabelled the
  subsystem" = it named the right channel but attributed it to, e.g., "thermal" when channel_41 is a
  "power" channel. **Resolve the user's confusion:** grounding was 1.86/2 (not perfect) — the
  "100% grounded / 100% severity-appropriate action" phrasing is an overstatement and INCONSISTENT
  with "3/120 mislabelled subsystem." FIX: state it as "119/120 correct channel, 117/120 correct
  subsystem, magnitude consistent — i.e. grounding 1.86/2, strong but not perfect." So: yes, on true
  positives it's doing a good job, but "100% grounded" should become "near-perfect grounding (1.86/2)."

### Item 6 (A + B — LSTM: how to improve / did we do industry-standard best?)
We used the *industry-standard Telemanom method* (per-channel LSTM autoencoder + dynamic μ+3σ
threshold) but a **basic** version. The official ESA-ADB "Telemanom-ESA-**Pruned**" reportedly scores
~0.97 event-wise CEF on Mission 1 → our LSTM has real headroom. How to improve: (a) Telemanom's
**pruned dynamic error thresholding** (not a flat μ+3σ); (b) per-channel hyperparameter/threshold
tuning; (c) bidirectional/attention or deeper layers; (d) longer context windows; (e) ensemble
across channels. State plainly: "method = industry standard; implementation = a solid baseline with
clear headroom toward the pruned/tuned variant."

### Item 7 (A + B — detector section + the frontier "no channel history/signal" explanation + add
fine-tune-frontier/RAG to next steps + surface the VISION model in the comparison)
- **Explain "channel history" and "signal" with project examples.** The model sees, per window:
  the mission + channel name + ~10 normalized values. "Signal" = the discriminative information that
  separates anomalous from nominal. "No channel history" = the frontier has never learned what is
  *normal for this specific channel/mission* (its baseline range, periodicity, typical noise), so 10
  context-free normalized numbers carry almost no signal — especially for the dominant
  `subtle_deviation` anomaly class, which looks nearly normal in 10 numbers. The **fine-tuned model
  learned those per-channel/per-mission priors** from 21,000 training windows, so it can tell "this
  shape is off *for channel_41*." Example to use: a value of 0.6 might be perfectly normal for one
  channel and a clear anomaly for another; without channel history you can't know which.
- **How to give the frontier that context (the user's question):** (a) **RAG** — retrieve recent/
  historical windows for the same channel and put them in the prompt so the frontier has a baseline;
  (b) **fine-tune a frontier model** (GPT-4o/Gemini support tuning APIs; Claude does not publicly) —
  this is the true "could we just adapt a frontier instead of owning an 8B?" comparison. Neither was
  done. → **Add both to next steps**, noting the sovereignty trade-off (fine-tuning a hosted frontier
  re-introduces the vendor dependency the project set out to avoid).
- **Surface the VISION fine-tune in the comparison (genuine gap the user caught):** the "did
  fine-tuning help?" skeptic table (A §5) and §6.1 and the one-paragraph takeaway are **text-only**.
  WHY: Phase 6 (the base/frontier controls) was built BEFORE Phase 8 (vision existed yet), and the
  controls were wired to the text harness. There is **no un-fine-tuned-base-VL or frontier-VL
  control**. FIX: (1) add the vision row into §6.1's "strongest detector" discussion side-by-side
  (it has the best CEF0.5 of any LLM, 0.604, and P 0.769); (2) mention vision in the one-paragraph
  takeaway; (3) add a **"vision base control" to next steps** explaining it wasn't run because Phase 6
  predates Phase 8 and a base Qwen3-VL would also need the format contract — but it *should* be run
  to complete the symmetry. Consider whether to add it as a real new phase vs. next-step (user asked).
  Recommend: next-step now + note in plan; running it is cheap (zero-shot base VL on the 2000 PNGs).

### Item 8 (A — vision model placement + "bigger model would help?" + stack advisor on vision)
- **§6.1 "The LSTM is the strongest detector" must list the vision model side-by-side** (it was run
  later but this is a final report — order by result, not by chronology). Vision P 0.769 / CEF0.5
  0.604 deserves equal billing; currently it's siloed in §6.3.
- **§6.2 mentions the 3-channel smoke (0.663) but those results appear nowhere else** — either (a)
  add a one-line note that 0.663 was the Phase-2 smoke number now superseded, or (b) keep it but make
  clear it's not in the master table on purpose. Make it self-consistent.
- **Would a bigger/better model or more training help the vision detector?** Yes plausibly: it
  converged very fast (see item 13.8) so more data/regularization could improve generalization; a
  larger VL backbone or more epochs *might* lift recall (its weak point, 0.325). Add as a measured
  "possible, untested" next-step, not a promise.
- **Stack the advisor on the vision detector?** YES — this is a valid architecture and worth stating:
  vision is high-precision (0.769), so `vision detector (high P) → text advisor` is a legitimate
  alternative/addition to `LSTM → advisor`, useful where false alarms are especially costly. Tie to
  the existing "vision as optional third leg / ensemble cross-check" line.

### Item 9 (B — remind what Phase 1.5 was; how generated; one model for both?)
**Phase 1.5 = the advice training labels.** For each of the 7,457 anomaly windows it generated a
structured `DIAGNOSIS / ADVICE / ACTION` record + severity + pattern type, **in-session and
templated from the window's own statistics** (NOT from an external API, NOT human-written): the
pattern type (subtle_deviation / persistent_anomaly / etc.) and severity were derived from the
anomaly_ratio / shape of the window, and turned into templated advice text. These were **merged into
the training responses** so the text model could learn to *write* advice. **One model does both:**
the Qwen3-8B text SFT was trained to output the ANOMALY/NOMINAL verdict AND the advice in one
response — so detection and advice come from the same model (that's why "LLM detection (text)" and
the advice layer are the same checkpoint). The **vision** model is detection-only (separate, no
advice). So "could we train one for both?" — we DID, for the text model. Make this clear in B.

### Item 10 (A — formatting: words running together)
Fix run-together words in the tables in **§3.2 "The reference works we drew on"** and **§4 "Process &
methodology (the nine phases)."** Likely markdown table cells where text wrapped/concatenated. Render
and inspect; add spaces/fix pipes.

### Item 11 (A — base-model decision logic missing)
The user is right that A doesn't explain WHY Qwen3-8B. It IS documented in
`thoughts/shared/research/2026-06-12-star-pipeline-codebase-research.md` §2: chosen over Qwen2.5 /
Llama-3 because — higher MATH/reasoning scores; Instruct variant avoids untrained-chat-token issues;
fully supported by Unsloth; 8B fits the 36 GB M3 Max after 4-bit quant; has a vision sibling
(Qwen3-VL-8B) enabling the AnomSeer approach. ADD a short "Model selection" note to A's methodology
(§4) and to B.

### Item 13 (A §9 Limitations — explain the *why-not-done* for each, define terms)
For EACH limitation, add a clause on *why it wasn't done* (time/scope of a showcase) and what it'd
take. Specifics:
- **Eval-unit asymmetry** (LSTM macro/58-contiguous vs LLM micro/4500-shuffled): not done because
  Phase 7 already moved 3→58 channels under time budget; a *fully* like-for-like rematch needs both
  scored on one identical contiguous per-channel stream (rebuild eval harness; ~1–2 days). Say so.
- **Frontier control sample (n=150)**: explain how you'd give it richer context in the real world
  (RAG over channel history; longer raw window; channel metadata) and why not done here (Phase 6 was
  the free/no-cloud control on the *same context-free input* the fine-tune saw — a deliberately
  controlled, hard comparison, not a best-effort frontier benchmark).
- **300-char advice clip**: explain WHY — during the 4,500-window eval the stored `actual_response`
  was truncated to 300 chars to keep `inference_test.json` manageable; this clipped the trailing
  ACTION line. **Would any comparison change without the clip?** No — the verdict (ANOMALY/NOMINAL)
  is at the start, and the SAME clip applied to base/frontier, so detection numbers are unaffected;
  only advice-grading completeness (the ACTION line) is affected. Re-running advice grading without
  the clip is a small next-step. **Also DEFINE "TP:FP ratio"**: ratio of true-positive to
  false-positive flags. The Phase-9 sample preserved the model's real 684:1214 (~36% precision) ratio
  as 43:77 so the grade reflects the true false-alarm rate, not a cleaned-up sample.
- **#6 Hyperparameter sweep**: not done — single sensible config under showcase time budget. Most
  useful knobs to sweep: LoRA rank (8/16/32), epochs (2/3/5), learning rate, prompt format, window
  size. Est. ~1 day of cloud per small grid (~$5–15). Good future plan.
- **#7 SFT / detection-only / calibrated threshold**: DEFINE **SFT** = Supervised Fine-Tuning
  (training on input→output pairs — what we did). "Detection-only SFT" = a model fine-tuned purely to
  emit ANOMALY/NOMINAL (no advice) which might calibrate precision/recall differently. **"Calibrated
  decision threshold"** = instead of the model's hard verdict, take a continuous anomaly score and
  choose a cutoff that trades recall for precision (move along a precision–recall curve). Not explored
  because the deployed model was the advice SFT used at one operating point. Whether detection-only
  "would do better" is an open hypothesis (don't assert it — say "could shift the operating point,
  untested"). Next-step.
- **#8 "converged very fast (eval loss 0.0089 on a 2-class task)"**: explain — the vision model's
  validation loss dropped extremely low on an easy 2-class (ANOMALY/NOMINAL) task, meaning it fit the
  training distribution very easily. Low in-distribution val loss does NOT guarantee it generalizes
  to a *new mission it never trained on* — that's untested (overfitting risk).
- **#9 "Resampling to 1-hour cadence is lossy"**: explain — raw telemetry is sampled every few
  seconds/minutes; we averaged it onto a 1-hour grid to make 29 GB tractable. An anomaly lasting <1
  hour can get averaged into its neighbors and disappear. We didn't try other cadences (5-min/15-min)
  so we don't know how much signal the 1-hour choice cost. Next-step = resample-cadence sweep.

### Item 14 (A §10 Next Steps — add effort estimates, explain items)
- **#2 "Fully level the detection field"**: add effort (~1–2 days to build a contiguous-stream eval
  harness + re-score). Answer the user's question explicitly: the report is *still meaningful* without
  it (Phase 7 already made it far more honest, and the literature corroborates the LSTM>LLM result),
  but doing it would convert "believable upper bound" into a settled like-for-like number — *more*
  authoritative, not required. Note whether to add to plan.
- **Ensemble idea** (user likes it): flesh out — combine recall-oriented text LLM + precision-
  oriented vision LLM (e.g. "both must fire" = high precision, "either fires" = high recall);
  est. ~2–3 days (no training, just score-combination logic over existing per-window outputs).
  **Add to the plan doc as a proposed next phase.**
- **#7 "Cross-mission generalization test & resample-cadence sweep"**: explain — cross-mission =
  train on Mission 1, test on Missions 2/3 (spacecraft the model never saw) to measure whether it
  generalizes beyond its training distribution; resample-cadence sweep = re-run ETL at 5-min/15-min/
  1-hour to find the best signal-vs-tractability trade-off (see item 13.9).

---

## 3. Plain-language explanations to reuse (for B and the de-jargoned A bits)

- **Channel / subsystem:** channel = one sensor stream (`Mission1/channel_41`); subsystem = the
  functional group (power, thermal, AOCS/attitude, propulsion, comms). Example: channel_41 might be a
  battery-bus voltage in the *power* subsystem.
- **Signal / channel history:** "signal" = the info that separates anomalous from normal; "channel
  history" = knowing what's normal *for that specific channel*. 0.6 may be normal for one channel,
  anomalous for another — without history you can't tell.
- **Calibrated decision threshold / P–R curve:** turn the model's score into a tunable cutoff to
  trade recall for precision; the P–R curve plots that trade-off across cutoffs (one model = many
  operating points, not one).
- **SFT:** Supervised Fine-Tuning — training on input→output examples.
- **TP:FP ratio:** true-positive flags : false-positive flags; preserving it in a sample keeps the
  real false-alarm rate.
- **Ensemble:** combine multiple detectors' outputs (e.g. require both text+vision to agree).
- **How to improve the *advice* toward mission-critical:** (1) replace synthetic Phase-1.5 labels
  with **human-SME-written advice** (the biggest lever — current ceiling is the templated labels);
  (2) **retrieval/grounding** — give the advisor the channel's spec sheet / fault catalog (RAG) so it
  cites real root causes, not patterns; (3) **bigger/stronger advisor backbone** or more epochs IF
  resources allow; (4) **put the advisor behind a high-precision detector** (LSTM/vision) so it's
  rarely built on a false premise; (5) **calibrated confidence/abstention** — let it say "uncertain"
  instead of fabricating; (6) **larger + human-validated advice grade** to replace the n=120 LLM-judge
  sample. State clearly: 95% high-quality-on-TP is good for a showcase but NOT sufficient for
  life-critical deployment without (1)+(2)+(6).

---

## 4. Plan-doc additions the user is interested in (`…/plans/2026-06-12-star-pipeline-create-plan.md`)
The user asked "should we add to the plan?" for several. Propose adding a **"Phase 11 — Post-report
hardening / next steps"** block (or extend the existing next-steps), covering:
1. Vision base/frontier control (complete the skeptic table symmetry) — cheap.
2. Fully-leveled contiguous-stream eval (LSTM + LLM, real Affinity-F1) — ~1–2 days.
3. Text+vision **ensemble** — ~2–3 days.
4. LLM P–R curve / calibrated threshold + detection-only SFT.
5. LoRA **hyperparameter sweep**.
6. **Fine-tune a frontier model** and/or **RAG-with-channel-history frontier** — the true "own-model
   vs adapt-frontier" comparison (note sovereignty trade-off).
7. **Cross-mission generalization** + **resample-cadence sweep**.
8. **Human-SME advice labels** + larger advice grade (path to mission-critical).
Confirm with user whether they want these written into the plan now or just referenced as next-steps
in the report.

---

## 5. Open decisions to confirm with the user
- License = MIT confirmed? (already committed MIT; flagged Apache-2.0 as alt.)
- HF repo slugs in `huggingface/` cards assume `dyrtyData/...` (text: `star-pipeline-qwen3-8b-advice-gguf`,
  vision: `star-pipeline-qwen3-vl-8b-detection`, dataset: `esa-ad-star-splits`) — confirm exact IDs.
- Whether to ADD the next-step phases to the plan doc now (item 4 above) vs. just list in the report.
- Whether to actually RUN the cheap vision base control before finalizing (user floated it).

## 6. Workflow notes
- Source of truth for numbers: `results/comparison_metrics.json` + `results/comparison_report.md`
  (auto-generated; do NOT hand-edit the report — regenerated by `make eval-all`).
- After editing A and B: do the item-1 consistency pass, then commit (atomic, Co-Authored-By trailer).
- `FinalReport_FollowUp.md` itself: it's the user's request doc; once revisions are done, ask whether
  to keep, archive, or delete it (it contains some first-person/learning phrasing).
- The deleted `space-anomaly-detection-ai-advice_exploratoryThread.md` (raw chat) and `Nina_FDL.png`
  were removed in commit `6e8d34a` — don't reference them.
