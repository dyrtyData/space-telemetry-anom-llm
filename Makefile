.PHONY: setup download download-zenodo etl baseline baseline-if validate-baseline format-train launch-vast train-cloud export eval-all eval-lstm eval-llm install-local validate-inference clean lint format validate-etl validate-format validate-advice advice

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# Where raw ESA-AD lives. Override to use an external drive, e.g.:
#   make download ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad"
ESA_DATA_DIR ?= data/raw/esa-ad
MISSION ?= 1

# Setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev,lstm]"

# ETL Pipeline -- primary download is the Kaggle mirror (fast CDN, byte-identical
# to the official Zenodo manifest). Zenodo kept as a fallback (download-zenodo).
download:
	$(PY) src/etl/download_kaggle.py --data-dir "$(ESA_DATA_DIR)" --mission $(MISSION)

download-zenodo:
	$(PY) src/etl/download_esa.py --data-dir "$(ESA_DATA_DIR)" --missions $(MISSION)

etl:
	ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/etl/patch_telemetry.py --missions $(MISSION)
	$(PY) src/etl/generate_plots.py

# Baselines -- keras 3 runs on the torch backend (no tensorflow installed); raw data
# comes from ESA_DATA_DIR, trained models go under STAR_OUTPUT_DIR (external drive).
baseline:
	KERAS_BACKEND=torch ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/train_lstm.py --missions $(MISSION)

baseline-if:
	ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/isolation_forest.py --missions $(MISSION)

validate-baseline:
	$(PY) -c "\
import json, math; \
d = json.load(open('results/lstm/baseline_results.json')); \
s = d['summary']; \
ch = [c for c in d['channels'] if 'error' not in c]; \
print('Summary:', s); \
assert s.get('n_channels_scored', 0) > 0, 'No channels scored'; \
vals = [c[k] for c in ch for k in ('precision','recall','f1','final_loss','initial_loss')]; \
assert all(math.isfinite(v) for v in vals), 'Non-finite metric found'; \
dec = [c for c in ch if 'final_loss' in c and c['final_loss'] < c['initial_loss']]; \
assert dec, 'No channel showed a loss decrease'; \
f1 = s['avg_f1']; \
print(f'avg_f1={f1:.3f} (sanity range 0.05 < f1 < 0.98)'); \
assert 0.05 < f1 < 0.98, f'avg_f1 {f1} outside sanity range (recalibrate if needed)'; \
print('validate-baseline OK') \
"

# Phase 3 -- LLM fine-tuning (cloud). format-train runs LOCALLY; the rest drive Vast.ai.
# train_advice/train_detection run ON the instance (Unsloth needs CUDA), not via make.
format-train:
	$(PY) src/training/format_for_unsloth.py

validate-format:
	$(PY) -c "\
import json; \
files = {s: f'data/formatted/{s}_chatml.jsonl' for s in ['train','val','test']}; \
[__import__('os').path.exists(p) or (_ for _ in ()).throw(AssertionError(f'missing {p}')) for p in files.values()]; \
recs = {s: [json.loads(l) for l in open(p)] for s, p in files.items()}; \
[ (_ for _ in ()).throw(AssertionError(f'{s}: bad keys')) for s, rs in recs.items() for r in rs if set(r) != {'text'}]; \
[ (_ for _ in ()).throw(AssertionError(f'{s}: not chatml')) for s, rs in recs.items() if not rs[0]['text'].startswith('<|im_start|>system')]; \
print({s: len(rs) for s, rs in recs.items()}); \
print('validate-format OK') \
"

# Dry run: search offers + print cheapest (no charges). Add --create to launch.
launch-vast:
	./scripts/cloud/launch_vast.sh

# Phase 4 -- Local GGUF inference on M3 Max.
# STAR_MODEL_DIR must point to where the GGUF was downloaded (default: DUAL DRIVE).
STAR_MODEL_DIR ?= /Volumes/DUAL DRIVE/star-pipeline/models

# Install llama-cpp-python with Metal GPU support (M3 Max). Run once after Phase 4 download.
install-local:
	CMAKE_ARGS="-DLLAMA_METAL=on" $(PIP) install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# Run inference smoke test (default 100 samples). Use LIMIT=0 for full 4,500 (Phase 5).
LIMIT ?= 100
eval-llm:
	STAR_MODEL_DIR="$(STAR_MODEL_DIR)" $(PY) src/inference/test_local_gguf.py --limit $(LIMIT)

validate-inference:
	$(PY) -c "\
import json, math, os; \
p = 'results/inference_test.json'; \
assert os.path.exists(p), f'Missing {p} -- run make eval-llm first'; \
d = json.load(open(p)); \
s = d['summary']; \
rs = d['results']; \
print('Summary:', s); \
assert s['n_samples'] > 0, 'No samples'; \
assert all(math.isfinite(s[k]) for k in ('precision','recall','f1','accuracy')), 'Non-finite metric'; \
assert 0 <= s['f1'] <= 1, 'F1 out of range'; \
assert s['avg_time_s'] < 30, f'Avg time {s[\"avg_time_s\"]:.1f}s too slow (expect <30s on M3 Max)'; \
assert s['unknown_responses'] < s['n_samples'] * 0.2, 'Too many unparseable responses'; \
long_enough = sum(1 for r in rs if len(r['actual_response']) > 10); \
assert long_enough == s['n_samples'], f'Short responses found ({s[\"n_samples\"] - long_enough})'; \
kw_ok = sum(1 for r in rs if 'ANOMALY' in r['actual_response'].upper() or 'NOMINAL' in r['actual_response'].upper()); \
print(f'Responses with ANOMALY/NOMINAL keyword: {kw_ok}/{s[\"n_samples\"]}'); \
assert kw_ok > s['n_samples'] * 0.8, 'Model not producing expected keywords'; \
print('validate-inference OK') \
"

# Evaluation
eval-all:
	$(PY) src/inference/evaluate.py --all

eval-lstm:
	$(PY) src/baselines/train_lstm.py

# Utilities
lint:
	$(VENV)/bin/ruff check src/
	$(VENV)/bin/ruff format --check src/

format:
	$(VENV)/bin/ruff format src/

advice:
	ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/etl/generate_advice_labels.py

validate-advice:
	$(PY) -c "\
import json; \
advice = json.load(open('data/labels/anomaly_advice.json')); \
ids = [a['anomaly_id'] for a in advice]; \
assert len(ids) == len(set(ids)), 'Duplicate anomaly_ids found'; \
required = {'advice','severity','recommended_action','pattern','mission','channel'}; \
missing = [a['anomaly_id'] for a in advice if not required.issubset(a)]; \
assert not missing, f'{len(missing)} records missing required fields'; \
sevs = {a['severity'] for a in advice}; \
assert sevs <= {'low','medium','high'}, f'Unexpected severity values: {sevs}'; \
print(f'advice OK: {len(advice)} records, severities={dict((s, sum(1 for a in advice if a[\"severity\"]==s)) for s in [\"low\",\"medium\",\"high\"])}') \
"

validate-etl:
	$(PY) -c "\
import json; \
files = ['data/splits/train.jsonl', 'data/splits/val.jsonl', 'data/splits/test.jsonl']; \
records = [json.loads(l) for f in files for l in open(f)]; \
total = len(records); \
anomalies = sum(r['metadata']['is_anomaly'] for r in records); \
missions = {r['metadata']['mission'] for r in records}; \
print(f'Total: {total}, Anomalies: {anomalies} ({100*anomalies/total:.1f}%), Missions: {sorted(missions)}'); \
assert total == 30000, f'Expected 30000, got {total}'; \
assert 100 < anomalies < 10000, f'Anomaly count {anomalies} out of range'; \
assert len(missions) == 3, f'Expected 3 missions, got {missions}' \
"

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
