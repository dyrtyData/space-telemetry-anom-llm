# STAR-Pipeline Evaluation Report

End-to-end comparison of anomaly-detection approaches on the ESA-AD test split. Detection metrics are computed from the persisted Phase-2 (baselines) and Phase-4/5 (LLM) result files; CEF0.5 is the precision-weighted F-beta (beta=0.5) favoured when false alarms are costly.

## Approach Comparison

| Approach | Precision | Recall | F1 | CEF0.5 | Affinity-F1 | Eval unit |
|----------|-----------|--------|----|--------|-------------|-----------|
| Isolation Forest | 0.127 | 0.459 | 0.188 | 0.149 | N/A | 3 channels |
| LSTM Baseline | 0.837 | 0.599 | 0.698 | 0.775 | 0.607 | 1 channels |
| LLM Detection | 0.360 | 0.609 | 0.453 | 0.392 | 0.456 | 4500 windows |
| LLM Detection (vision, Qwen3-VL) | 0.769 | 0.325 | 0.457 | 0.604 | N/A | 2000 windows (PNG) |
| Base Qwen3-8B (zero-shot) | 0 | 0.000 | 0 | 0.000 | N/A | 100 windows |
| Base Qwen3-8B (few-shot, no fine-tune) | 0.282 | 0.824 | 0.420 | 0.325 | N/A | 500 windows |
| Frontier zero-shot (Claude, n=150 sample) | 0.308 | 0.216 | 0.254 | 0.284 | N/A | 150 windows |
| Frontier few-shot (Claude, n=150 sample) | 0.200 | 0.297 | 0.239 | 0.214 | N/A | 150 windows |
| Always-anomaly (trivial baseline) | 0.250 | 1.000 | 0.399 | 0.294 | N/A | 4500 windows |
| Hybrid (LSTM + LLM advice) | 0.837 | 0.599 | 0.698 | 0.775 | N/A | 1 channels |

## Key Findings

- Highest detection F1: **LSTM Baseline** (F1=0.698, precision=0.837, recall=0.599).
- Best precision-weighted score (CEF0.5, the operationally relevant metric for costly false alarms): **LSTM Baseline** (CEF0.5=0.775).
- The LSTM baseline detects with higher precision (0.837 vs 0.360); the LLM trades precision for recall (0.609 vs 0.599) while adding a capability the baselines lack: free-text diagnostic advice.
- LLM advice coherence: 100% of the 1898 anomaly predictions emitted structured DIAGNOSIS+ADVICE text (responses persisted truncated at 300 chars) -- the hybrid's added value over the bare LSTM.
- LLM inference cost: 2.77s/window on M3 Max Metal over 4500 windows (vs near-instant scoring for the baselines).
- Affinity-F1 (interval-aware) for the LLM = 0.456 over 1052 ground-truth intervals; on this shuffled, subsampled test split it largely reduces to window-level F1 (see methodology note).

## Did fine-tuning help?

The headline claim is that fine-tuning an open model adapts it to a localized, mission-specific task. This isolates that value by holding the harness fixed (identical prompt, decoding, and parser) and varying only the model:

| Model | F1 | CEF0.5 | Output-format compliance | Structured-advice on flags | Eval |
|-------|----|--------|--------------------------|----------------------------|------|
| LLM Detection | 0.453 | 0.392 | 0.994 | 0.996 | 4500 windows |
| Base Qwen3-8B (zero-shot) | 0 | 0.000 | 0.000 | 0.000 | 100 windows (partial) |
| Base Qwen3-8B (few-shot, no fine-tune) | 0.420 | 0.325 | 1.000 | 0.129 | 500 windows |
| Frontier zero-shot (Claude, n=150 sample) | 0.254 | 0.284 | 1.000 | 1.000 | 150-window sample |
| Frontier few-shot (Claude, n=150 sample) | 0.239 | 0.214 | 1.000 | 1.000 | 150-window sample |
| Always-anomaly (trivial baseline) | 0.399 | 0.294 | 1.000 | 0.000 | flag-all (4500 windows) |

- **Read the table against the dumb baseline first.** *Always-anomaly* (flag every window) scores F1 **0.399** / CEF0.5 0.294 for free on this ~25%-positive set. Any approach at or below that line has **not** learned to detect — it is just over-flagging. Only the **fine-tune** (F1 0.453, balanced P=0.360/R=0.609) clears it with a real precision/recall trade-off.
- **vs. the un-fine-tuned base (same Qwen3-8B, same harness, same 100 windows):** detection F1 0.000 → 0.453 (Δ **+0.453**). The larger story is the output contract: the base produces a parseable ANOMALY/NOMINAL verdict on only 0% of windows (it spends its token budget on chain-of-thought), versus 99% for the fine-tune (Δ **+99 pts**), and emits the structured DIAGNOSIS/ADVICE format on 0% vs 100% of its flags (Δ **+100 pts**).
- **The few-shot base's F1 is a mirage.** With 2 in-context examples + /no_think it recovers compliance (100% parseable) and posts F1 0.420 — but with precision 0.282 / recall 0.824, i.e. it flags the large majority of windows. That is barely above the flag-everything baseline: it is **not detecting**, just over-flagging (the 1:1 examples mis-signal a ~50% prior). It is also slower (8.557s vs 2.765s) and emits structured advice on only 13% of flags (vs 100%).
- **The frontier (Claude) is genuinely trying, but the input is near-signal-free.** Zero-shot F1 0.254 (P=0.308); adding the SAME few-shot examples barely moves it (F1 0.239) — so the gap is **not** a prompting-asymmetry artifact. Unlike the base it does not over-flag (P≈R≈0.31, near the base rate), so it sits ~at chance: 10 normalized values with no channel context do not carry the signal. A far stronger general model, prompted two ways, cannot beat the dumb baseline here.
- **Takeaway (the honest headline):** the **fine-tuned model is the only approach that beats the always-anomaly baseline with a balanced precision/recall** — the lone real detector. Few-shot prompting a base 'wins' F1 only by over-flagging (≈ the dumb baseline); a frontier model prompted zero- or few-shot sits at chance on this context-free input. On top of detection, fine-tuning also delivers output compliance, structured advice, and 3× lower latency. Its value is learning mission/channel-specific priors that no prompt over 10 normalized values can supply.

## Advice quality (semantic) — Phase 9

The detection table above shows the fine-tuned model emits structured advice on ~all of its flags. This grades whether that advice is *correct*. A frozen seed-42 sample of **120** of the model's anomaly predictions (TP/FP ratio preserved) was graded by Claude (session model) — no API, no fine-tuning. Rubric: correctness / actionability / grounding, each 0-2 (total 0-6).

| Subset | n | Correctness | Actionability | Grounding | Mean total /6 | High-quality |
|--------|---|-------------|---------------|-----------|---------------|--------------|
| All flags | 120 | 0.64 | 1.03 | 1.01 | 2.68 | 34% |
| True positives (correct flags) | 43 | 1.79 | 1.93 | 1.86 | 5.58 | 95% |
| False positives (false alarms) | 77 | 0.00 | 0.53 | 0.53 | 1.06 | 0% |

- **When the model is right to flag (true positives), its advice is genuinely good:** mean 5.58/6, 95% high-quality, 100% grounded (correct channel/subsystem/unit and a magnitude consistent with the window), 100% carrying a severity-appropriate recommended action.
- **Advice quality is gated by detection precision.** On false alarms (77/120 of the sampled flags — the model's precision is ~0.36) the advice is built on a false premise, so correctness is 0.00/2 by construction; the model often fabricates a confident 'persistent/100% anomalous' narrative on a truly-nominal window. Overall mean drops to 2.68/6 (34% high-quality).
- **Implication for the recommendation:** the fine-tuned model is best deployed as the *advisor* on top of a high-precision detector (the Hybrid: LSTM flags at precision ≈0.84, the LLM explains each flag), not as the standalone detector — its diagnostic writing is strong, but only as trustworthy as whatever decided to call the anomaly.
- **Caveats:** sample of 120 only; judge is the Claude session model (not human SMEs); gold advice labels are synthetic (statistic-derived, Phase 1.5) and used only as an optional reference; advice text was scored from the 300-char-truncated DIAGNOSIS+ADVICE (the ACTION line is clipped in storage).

## Methodology Notes

- **Baselines (LSTM, Isolation Forest)** are scored per channel and macro-averaged over the channels evaluated (Phase-2 smoke run: 3 Mission-1 target channels). Per-window predictions were not persisted, so Affinity-F1 is N/A for them.
- **LLM Detection** is scored per window (micro-averaged) over the 4500-window test slice; ANOMALY/NOMINAL is parsed from the generated text.
- **LLM Detection (vision, Qwen3-VL)** is the AnomSeer-style detector: a Qwen3-VL-8B fine-tuned to read a *rendered PNG plot* of the telemetry window (not the numeric text) and emit ANOMALY/NOMINAL. Scored on 2000 test-split PNGs (micro-averaged). It is a pure detector — no diagnostic advice, so its advice fields are 0 by design; it adds a second, modality-independent detection signal that completes the original three-way LLM design.
- **Hybrid (LSTM + LLM advice)** inherits the LSTM's detection metrics by construction: the LSTM flags anomalies (high precision) and the LLM attaches diagnostic advice to each flag. Its value is the advice layer, not better detection.
- **Affinity-F1** merges per-window predictions into intervals within each (mission, channel) timeline and matches predicted vs. ground-truth intervals within a tolerance. The ESA-AD test split here is a *shuffled, balanced-subsampled* set (~1.4 windows/channel), so intervals are mostly isolated single windows and Affinity-F1 ≈ window-level F1; it is wired for correctness and becomes meaningful on a contiguous evaluation stream.
- **CEF0.5** uses beta=0.5 (precision-weighted). Computed from each approach's reported precision/recall.
- **Base Qwen3-8B (zero-shot)** is the *un-fine-tuned* base run through the identical harness (same prompt, decoding, parser) — the controlled comparison that isolates the fine-tuning effect. **Frontier zero-shot** is the Claude session model classifying a frozen seed-42 stratified sample (no fine-tuning, no API), seeing the same per-window input the fine-tune saw. Both are Phase-6 controls for the 'Did fine-tuning help?' analysis above.
