---
license: cc-by-3.0
pretty_name: ESA-AD STAR-Pipeline Splits (windowed, instruction/response)
task_categories:
- time-series-forecasting
tags:
- time-series
- anomaly-detection
- spacecraft-telemetry
- satellite
- instruction-tuning
source_datasets:
- esa/anomaly-dataset
language:
- en
size_categories:
- 10K<n<100K
---

# ESA-AD STAR-Pipeline Splits (windowed, instruction/response)

A **derived, preprocessed** version of the **ESA Anomaly Dataset (ESA-AD)** prepared for the
STAR-Pipeline project: continuous multivariate satellite telemetry sliced into fixed windows and
formatted as instruction/response records for LLM fine-tuning and evaluation, plus rendered PNG
plots for the vision approach.

> **This is a derivative work, not the raw dataset.** The raw ESA-AD telemetry is **not**
> redistributed here — only windowed/normalized splits and rendered plots derived from it. To
> obtain the raw data, see the original [Zenodo record](https://doi.org/10.5281/zenodo.12528696).

## Contents

| Split | Records | Anomalous |
|---|---|---|
| train | 21,000 | ~24.8% |
| validation | 4,500 | ~25% |
| test | 4,500 | 25.0% |

- **Windowing:** 32-step rolling windows, stride 16, resampled to 1-hour cadence.
- **Normalization:** RevIN (per-channel reversible instance normalization).
- **Coverage:** 3 ESA missions, 224 telemetry channels.
- **Record schema (text):** `{mission, channel, is_anomaly, instruction, response}` where the
  response for anomalies carries `DIAGNOSIS / ADVICE / ACTION` lines (advice labels generated
  synthetically in-session — treat as a reference, not ground-truth SME advice).
- **Vision variant:** PNG plots per window with `{image_path, is_anomaly, mission, channel}` metadata.

## Intended use

Fine-tuning / evaluating anomaly detectors (text or vision LLMs, classical baselines) on real
spacecraft telemetry, with a held-out test split for reproducible comparison. Metrics used in the
project: window-level P/R/F1, **CEF0.5**, and **Affinity-F1**.

## Limitations & caveats

- The test split is **shuffled and balanced-subsampled** (~25% anomalous), which makes interval-aware
  metrics (Affinity-F1) degenerate; for true streaming evaluation use a contiguous per-channel timeline.
- 1-hour resampling is lossy for sub-hour transient anomalies.
- Advice labels are synthetic (statistic-derived), included only as an optional reference field.

## License & attribution (important)

Derived from the **ESA Anomaly Dataset (ESA-AD)**, licensed **CC BY 3.0 IGO**
(attribution required; commercial use permitted; no share-alike). You must attribute the original
creators when using this derivative:

> ESA Anomaly Dataset (ESA-AD) — © European Space Agency / Airbus Defence and Space / KP Labs.
> Kotowski et al. (2024), *European Space Agency Benchmark for Anomaly Detection in Satellite
> Telemetry*, [arXiv:2406.17826](https://arxiv.org/abs/2406.17826),
> [Zenodo 10.5281/zenodo.12528696](https://doi.org/10.5281/zenodo.12528696).

```bibtex
@article{kotowski2024esaadb,
  title  = {European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry},
  author = {Kotowski, Krzysztof and others},
  journal= {arXiv preprint arXiv:2406.17826},
  year   = {2024},
  doi    = {10.5281/zenodo.12528696}
}
```
