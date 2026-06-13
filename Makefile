.PHONY: setup download download-zenodo etl baseline train-cloud export eval-all clean lint format validate-etl

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

# Baselines
baseline:
	$(PY) src/baselines/train_lstm.py

# Evaluation
eval-all:
	$(PY) src/inference/evaluate.py --all

eval-lstm:
	$(PY) src/baselines/train_lstm.py

eval-llm:
	$(PY) src/inference/test_local_gguf.py

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
