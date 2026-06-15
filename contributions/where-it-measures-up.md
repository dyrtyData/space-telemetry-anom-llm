# Where this project measures up — an honest competitive assessment

Researched 2026-06-14. The question: *are the results good enough to be worth submitting to arXiv,
TSB-AD, ESA-ADB, AnomLLM, etc.?* Short answer: **the numbers are not the contribution — the
rigor is.** Here's the evidence.

## The one caveat that governs everything: protocol mismatch

This project's metrics are computed on a **shuffled, balanced-subsampled** test set (~25% anomalous,
window-level micro-averaging). Every official benchmark uses a **different protocol**:

| | This project | ESA-ADB official / Kaggle | TSB-AD-M |
|---|---|---|---|
| Unit | window-level | **event-wise** (60 s tolerance) | range/point w/ buffer |
| Stream | shuffled windows | **full continuous** timeline | continuous |
| Class balance | balanced (~25% pos) | **real, heavily imbalanced** (anomalies rare) | real |
| Scalar | CEF0.5 (from P/R) | event-wise CEF0.5 | **VUS-PR** |

**Consequence:** the names match (CEF0.5, Affinity-F1) but the denominators don't. My CEF0.5 of
0.684 (LSTM) is **not comparable** to an official event-wise CEF0.5, and balanced subsampling removes
the class imbalance that makes the real benchmark hard — a model that looks good on a balanced subset
can collapse on the true rare-anomaly distribution. **To make any ranked claim, the method must be
re-run under that benchmark's exact harness.** This is the single biggest robustness gap.

## Where each result sits vs the published literature

### LLM detection — squarely "normal for the field," not exceptional
My text F1 **0.453** and vision F1 **0.457** land right inside the published LLM-TSAD range on real
data:
- Mistral zero-shot detector: F1 0.525; DeepSeek-V3 direct: 0.469; multimodal LLMs: ~0.45–0.53
  (range-/variate-wise); aerospace-synthetic best LLM ~0.47–0.61.
- Consensus finding across all of them: **8B-scale LLMs lag a tuned LSTM/CNN by ~10–30% F1**, worse
  on multivariate real data. My LSTM (0.552) > my LLM (0.453) **reproduces this exactly.**

So the LLM result is a *faithful corroboration* of the literature on a new, hard, real dataset — a
legitimate data point, but **not a number that "wins."**

### LSTM / Isolation Forest — credible classical tier, but unranked
TSB-AD-M leaderboard (VUS-PR): **xLSTMAD 0.37 (SOTA) · CNN/LSTMAD/PCA ~0.31 (prior-SOTA tier) ·
IForest ~0.20 (bottom)**. My IForest being weak (F1 0.188) is *directionally consistent* with their
IForest (0.20) — a good internal-consistency sign. A well-tuned LSTM submitted there would plausibly
land in the **~0.31 mid/upper-classical tier** — respectable, not SOTA. But I can't claim a number
until it's run on their 1,070 series.

### ESA-ADB official — the bar is very high
On the official event-wise protocol, **Telemanom-ESA-Pruned reportedly scores ~0.97 CEF0.5 on
Mission 1**, and an edge-deployment paper hit ~0.93. That's a different universe from window-level
0.68. Competing here means re-running on the real imbalanced event-wise stream **and** Docker-packaging
to their TimeEval harness — high effort, high bar.

## Per-venue verdict: is it worth it?

| Venue | Worth it? | Why |
|---|---|---|
| **Hugging Face** (cards, dataset, demo, collection) | **Yes — do now** | Zero gating, high signal, no SOTA needed. Turns weights into shipped models. |
| **Kaggle Code-tab notebook** | **Yes — do now** | Closed for scoring, but a clean reference notebook reaches the exact community. No bar. |
| **AnomLLM issue/PR** | **Yes — high value** | Their benchmark is *synthetic-only*; ESA-AD + a fine-tuned 8B fills a stated gap. Open the issue first. |
| **arXiv preprint** | **Yes** | No peer-review bar; an *application + honest-negative-result + reproducibility* paper is a respected type. Your literature-corroborating finding is publishable as-is. |
| **ML4ITS workshop** (next edition) | **Yes — best academic home** | Has a **presentation-only track with no novelty requirement** and a named "Time Series for Space Applications" session. 2025 deadline passed; watch for 2026. |
| **TSB-AD leaderboard** | **Yes, if you invest ~2–3 days** | Open, explicit submission process, real public leaderboard. Submit the **LSTM + IForest** (not the slow 8B LLM). Expect mid-tier (~0.31), not SOTA — but a *verifiable* "on a NeurIPS benchmark leaderboard" claim. |
| **ESA-ADB leaderboard/PR** | **Only as a deliberate project** | High effort (Docker + event-wise re-eval) and a ~0.97 bar. Not worth it just to "place"; worth it only if you want the ESA-ADB-protocol number and the packaging exercise. |
| **Kaggle competition (scored)** | **No** | Closed Aug 2025; no late scored entries. |

## What would make a *ranked/published* claim solid

In priority order (these are the project's own "next steps," and they're what an interviewer/reviewer
would ask for):
1. **Re-evaluate on one benchmark's exact protocol** — easiest is TSB-AD (wrap LSTM/IForest, get a
   real VUS-PR on their data). This is the cheapest path to a *verifiable external* number.
2. **One fully like-for-like contiguous-stream eval** (event-wise) so Affinity-F1/CEF are
   apples-to-apples and approximate the ESA-ADB protocol.
3. **A P–R curve for the LLM** (sweep the decision threshold) instead of a single operating point.
4. **A larger advice-grading sample** (+ ideally one human-SME spot-check).

## Bottom line

The project's worth is **not** a leaderboard-topping score — on every benchmark, a tuned classical
model still out-detects an 8B LLM, and that's exactly what the literature says. Its worth is a
**rigorous, honest, reproducible comparative study on real ESA telemetry**, including the controls and
negative results most fine-tuning showcases omit. That is **immediately** shareable (HF, Kaggle
notebook, AnomLLM, arXiv, ML4ITS presentation track) with no SOTA claim required. A *ranked* external
claim needs a protocol-matched re-evaluation — and the only one with a good effort:payoff ratio is
**TSB-AD**.
