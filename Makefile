.PHONY: setup download etl baseline train-cloud export eval-all clean lint format validate-etl

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# Setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev,lstm]"

# ETL Pipeline
download:
	$(PY) src/etl/download_esa.py

etl: download
	$(PY) src/etl/patch_telemetry.py
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

validate-etl:
	$(PY) -c "import json; \
		files = ['data/splits/train.jsonl', 'data/splits/val.jsonl', 'data/splits/test.jsonl']; \
		total = 0; anomalies = 0; \
		[exec('total += 1; anomalies += int(json.loads(line)[\"metadata\"][\"is_anomaly\"])') for f in files for line in open(f)]; \
		print(f'Total: {total}, Anomalies: {anomalies}'); \
		assert 100 < anomalies < 10000, f'Anomaly count {anomalies} out of expected range'"

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
