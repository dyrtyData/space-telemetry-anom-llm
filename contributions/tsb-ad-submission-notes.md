# TSB-AD submission notes

**Repo:** https://github.com/TheDatumOrg/TSB-AD · **Leaderboard:** https://thedatumorg.github.io/TSB-AD/

**Process (from their README):** implement a `Run_<YourModel>_Detector.py` in the `benchmark_exp`
folder on the `TSB-AD-algo` branch and either open a PR or email it to the maintainers
(`liu.11085@osu.edu` / `paparrizos.1@osu.edu`). They run and score it against their 1,070 curated
series and add it to the leaderboard. Metric: **VUS-PR** (and friends). No dataset adapter needed —
they evaluate on *their* data, so this demonstrates **generalization**, not your ESA-AD numbers.

**What to submit (best fit):** the **classical detectors** — Isolation Forest and the LSTM
autoencoder — which map cleanly onto their detector interface (input array → anomaly score array).
TSB-AD already benchmarks IForest and LSTMAD, so frame your entries as tuned variants and compare.

**What *not* to submit:** the 8B LLM detector. It's slow (2.77 s/window), TSB-AD doesn't measure the
advice that is its actual value, and per-window LLM scoring across 1,070 series is impractical. Tell
the LLM story on Hugging Face + the analysis doc instead.

**Interface sketch:**
```python
# Run_STAR_LSTM_Detector.py  (mirror an existing Run_*.py in benchmark_exp/)
def run_STAR_LSTM(data, **hp):
    # data: (T, C) array; return per-timestep anomaly score (higher = more anomalous)
    # reuse src/baselines/train_lstm.py logic: per-channel reconstruction error
    ...
    return scores
```

**Effort:** ~2–3 days. **Acceptance likelihood:** high — the process exists specifically for this.

**Honest framing for an interview:** "I contributed my baselines to an actively-maintained NeurIPS
benchmark and they appear on its public leaderboard" is a concrete, verifiable contribution claim.
