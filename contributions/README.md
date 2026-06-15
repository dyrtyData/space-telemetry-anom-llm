# Contribution playbook

Where this project can be contributed back, with current status (researched 2026-06-14) and the
concrete path for each. Ordered by likelihood-of-acceptance × signal.

| Venue | Status | Path | In this folder |
|---|---|---|---|
| **TSB-AD leaderboard** | ✅ open, explicit process | Submit a `Run_<Name>_Detector.py` (LSTM / IForest fit cleanly) via PR or email to the maintainers; they run it and add it to the live leaderboard. VUS-PR metric; multivariate ceiling ~0.31. | `tsb-ad-submission-notes.md` |
| **Hugging Face** | ✅ no gating | Upload model cards + dataset card + demo Space + Collection. | see `../huggingface/` |
| **AnomLLM** (`rose-stl-lab/anomllm`) | 🟡 active, no CONTRIBUTING | Open the scoped issue (their synthetic-only gap fits ESA-AD perfectly), then PR. | `anomllm-issue-draft.md` |
| **Kaggle** `esa-adb-challenge` | ⛔ closed Aug 2025 (no scored entry) | Publish a notebook in the competition's **Code tab** — visible to the community, no gating. | `kaggle/star-pipeline-esa-ad.ipynb` |
| **ESA-ADB** (`kplabs-pl/ESA-ADB`) | 🟡 active, Docker-packaging required | Open an issue proposing your detectors; if welcomed, package each as a Docker container to their TimeEval harness + add results. Higher effort. | — |
| **Telemanom** (`khundman/telemanom`) | ⛔ effectively abandoned | Don't PR — **fork + modernize** (Py3.11, ESA-AD adapter), publish your own; helps the users stuck on its broken data links. | — |
| **AnomSeer** (ICLR 2026) | ⏳ no public code yet | Watch the OpenReview page; once code drops, run a head-to-head with the Qwen3-VL model. | — |

## Robustness work that unlocks the stronger venues

- **TSB-AD:** wrap LSTM + Isolation Forest in their detector interface (~2–3 days). The 8B LLM is an
  awkward fit (slow; no advice metric) — submit the classical detectors there, tell the LLM story on HF.
- **arXiv / ML4ITS-style workshop:** an *application + honest-negative-result + reproducibility*
  paper is a respected contribution type (no SOTA claim required). To make it solid, first close the
  residual eval gaps: one fully like-for-like contiguous-stream eval, a P–R curve for the LLM, and a
  larger advice-grading sample. See the analysis doc's "next steps."

## Data license reminder

ESA-AD is **CC BY 3.0 IGO** — attribution required, commercial OK, no share-alike. Publish *derived*
artifacts (windowed JSONL, plots, weights) with attribution; do **not** redistribute the raw dataset.
