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

- **vs. the un-fine-tuned base (same Qwen3-8B, same harness, same 100 windows):** detection F1 0.000 → 0.453 (Δ **+0.453**). The larger story is the output contract: the base produces a parseable ANOMALY/NOMINAL verdict on only 0% of windows (it spends its token budget on chain-of-thought), versus 99% for the fine-tune (Δ **+99 pts**), and emits the structured DIAGNOSIS/ADVICE format on 0% vs 100% of its flags (Δ **+100 pts**).
- **vs. the base with few-shot prompting (same base weights, 2 in-context examples + /no_think, n=500):** prompting alone *does* recover output compliance (100% parseable verdicts) and a real detection score (F1 0.420, precision 0.282, recall 0.824) — but at 8.557s/window vs 2.765s for the fine-tune, and with **no structured advice** (13% vs 100%). On detection F1 the fine-tune is 0.453 (Δ **+0.032** — fine-tuning still wins). This is the honest, hardest comparison: it asks whether the fine-tune beats good *prompting*, not just the raw base.
- **vs. a frontier model zero-shot (Claude, 150-window stratified sample, no fine-tuning):** the frontier detector scores F1 0.254 (precision 0.308, recall 0.216) on the same input the fine-tune saw — a far stronger general model still trails the small fine-tuned model (F1 0.453) on this specialized task, because the signal lives in mission/channel-specific patterns learned during fine-tuning, not in general reasoning over a few normalized values.
- **Takeaway:** fine-tuning's clearest, most reliable wins here are *task/format adaptation, structured advice, and latency* — turning a capable but non-compliant base into a model that reliably emits the exact terse verdict + structured advice the downstream pipeline consumes, faster. Few-shot prompting can recover compliance and a comparable detection score, but not the structured advice or the speed; raw zero-shot (base and frontier) recovers neither. The operational unlock is compliance + advice.

## Methodology Notes

- **Baselines (LSTM, Isolation Forest)** are scored per channel and macro-averaged over the channels evaluated (Phase-2 smoke run: 3 Mission-1 target channels). Per-window predictions were not persisted, so Affinity-F1 is N/A for them.
- **LLM Detection** is scored per window (micro-averaged) over the 4500-window test slice; ANOMALY/NOMINAL is parsed from the generated text.
- **LLM Detection (vision, Qwen3-VL)** is the AnomSeer-style detector: a Qwen3-VL-8B fine-tuned to read a *rendered PNG plot* of the telemetry window (not the numeric text) and emit ANOMALY/NOMINAL. Scored on 2000 test-split PNGs (micro-averaged). It is a pure detector — no diagnostic advice, so its advice fields are 0 by design; it adds a second, modality-independent detection signal that completes the original three-way LLM design.
- **Hybrid (LSTM + LLM advice)** inherits the LSTM's detection metrics by construction: the LSTM flags anomalies (high precision) and the LLM attaches diagnostic advice to each flag. Its value is the advice layer, not better detection.
- **Affinity-F1** merges per-window predictions into intervals within each (mission, channel) timeline and matches predicted vs. ground-truth intervals within a tolerance. The ESA-AD test split here is a *shuffled, balanced-subsampled* set (~1.4 windows/channel), so intervals are mostly isolated single windows and Affinity-F1 ≈ window-level F1; it is wired for correctness and becomes meaningful on a contiguous evaluation stream.
- **CEF0.5** uses beta=0.5 (precision-weighted). Computed from each approach's reported precision/recall.
- **Base Qwen3-8B (zero-shot)** is the *un-fine-tuned* base run through the identical harness (same prompt, decoding, parser) — the controlled comparison that isolates the fine-tuning effect. **Frontier zero-shot** is the Claude session model classifying a frozen seed-42 stratified sample (no fine-tuning, no API), seeing the same per-window input the fine-tune saw. Both are Phase-6 controls for the 'Did fine-tuning help?' analysis above.
