.PHONY: setup download download-zenodo etl baseline baseline-if tune-threshold validate-baseline format-train launch-vast train-cloud export eval-all eval-lstm eval-llm validate-eval install-local validate-inference clean lint format validate-etl validate-format validate-advice advice eval-base eval-base-fewshot frontier-select frontier-assemble eval-vision eval-vision-score lstm-window-scores vision-pr-curve ensemble grade-advice-select grade-advice-assemble foxes-smoke foxes-data foxes-train foxes-figures foxes-report foxes-aux-smoke foxes-gradio-smoke validate-foxes

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# Where raw ESA-AD lives. Override to use an external drive, e.g.:
#   make download ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad"
ESA_DATA_DIR ?= data/raw/esa-ad
MISSION ?= 1
# Channels per mission for the LSTM baseline. Default 5 (quick smoke); set to 58 for the
# full Mission-1 target-channel sweep (Phase 7): make baseline MAX_CHANNELS=58
MAX_CHANNELS ?= 5
# LSTM thresholding method (Phase 11). flat = Phase-2/7 mean+3sigma (reproducible
# reference); dynamic = Telemanom pruned dynamic thresholding. Write each method to its own
# RESULTS_FILE so both stay reproducible, then make the winner canonical for evaluate.py:
#   make baseline MISSION=1 MAX_CHANNELS=58 THRESHOLD=dynamic RESULTS_FILE=baseline_results_dynamic.json
THRESHOLD ?= flat
RESULTS_FILE ?= baseline_results.json

# Setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -e ".[dev,lstm,foxes]"

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
	KERAS_BACKEND=torch ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/train_lstm.py --missions $(MISSION) --max-channels $(MAX_CHANNELS) --threshold $(THRESHOLD) --results-file $(RESULTS_FILE) --resume

baseline-if:
	ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/isolation_forest.py --missions $(MISSION)

# Phase 11: calibrate the LSTM operating point. REUSES the saved per-channel models (no
# retraining) to sweep the flat-threshold z grid + the Telemanom dynamic method in one pass,
# writing results/lstm/threshold_sweep.json + the CEF0.5-optimal per-channel result file.
tune-threshold:
	KERAS_BACKEND=torch ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/tune_threshold.py --missions $(MISSION) --max-channels $(MAX_CHANNELS)

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
# STAR_MODEL_DIR points to the directory containing gguf/ and lora/ subdirectories.
# Default is local SSD (D17: GGUF is 5GB, exceeds FAT32 4GB file limit on DUAL DRIVE).
# Override if you move the GGUF elsewhere: make eval-llm STAR_MODEL_DIR="/path/to/models"
STAR_MODEL_DIR ?= ./models

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

# Phase 6: un-fine-tuned base Qwen3-8B control, IDENTICAL harness (same prompt/decoding/
# parser as eval-llm) -- isolates the fine-tuning effect. Point --gguf at the base GGUF.
# Long run (~10s/sample, base spends its budget "thinking"); detach + resume for durability:
#   ( nohup caffeinate -dimsu env PYTHONUNBUFFERED=1 make eval-base > run_base.log 2>&1 < /dev/null & )
BASE_GGUF ?= models/gguf/base-qwen3-8b/Qwen3-8B-Q4_K_M.gguf
eval-base:
	$(PY) src/inference/test_local_gguf.py --gguf "$(BASE_GGUF)" \
		--results-file results/inference_base.json \
		--approach-label "Base Qwen3-8B (zero-shot)" \
		--limit $(LIMIT) --resume --checkpoint-every 250

# Phase 6: "prompting instead of fine-tuning" baseline — same base weights, but 2 in-context
# examples per class (from TRAIN, no test leakage) + Qwen3 /no_think so it emits parseable
# verdicts. The fair, hardest comparison: does the fine-tune beat good prompting on detection?
eval-base-fewshot:
	$(PY) src/inference/test_local_gguf.py --gguf "$(BASE_GGUF)" \
		--results-file results/inference_base_fewshot.json \
		--approach-label "Base Qwen3-8B (few-shot, no fine-tune)" \
		--few-shot 2 --no-think --limit $(LIMIT) --resume --checkpoint-every 100

# Phase 6: frozen stratified frontier-eval sample (seed 42). frontier-select writes the
# leak-free prompts; the frontier model (Claude session) classifies into
# results/frontier_classifications.json; frontier-assemble joins to ground truth.
frontier-select:
	$(PY) src/inference/select_frontier_sample.py --select --n 150

frontier-assemble:
	$(PY) src/inference/select_frontier_sample.py --assemble results/frontier_classifications.json

# Phase 9: semantic advice grading (free, in-session judge). grade-advice-select freezes a
# seed-42 sample of the fine-tuned model's anomaly predictions + window context + gold ref to
# data/advice_grading/advice_sample.jsonl; the Claude session model scores each on a 0-2 rubric
# (correctness/actionability/grounding) into results/advice_judgments.json; grade-advice-assemble
# joins them into results/advice_grading_sample.json (consumed by evaluate.py's report).
grade-advice-select:
	$(PY) src/inference/grade_advice_sample.py --select --n $(or $(N),120)

grade-advice-assemble:
	$(PY) src/inference/grade_advice_sample.py --assemble results/advice_judgments.json

# Phase 8: Qwen3-VL vision detector eval. Runs ON the Vast.ai instance (Unsloth + the
# trained adapter), NOT locally — multimodal GGUF on Metal is patchy, so we score where
# it trained and scp results/inference_vision.json back. From the repo root on the instance:
#   python src/inference/eval_vision.py --resume --checkpoint-every 250
eval-vision:
	$(PY) src/inference/eval_vision.py --limit $(LIMIT) --resume --checkpoint-every 250

# Phase 14: vision continuous verdict score (1-step prefill logits) over the 2,000 test PNGs.
# Runs ON the Vast.ai instance like eval-vision; scp results/inference_vision_scored.json back.
#   python src/inference/eval_vision.py --score --resume --checkpoint-every 250
eval-vision-score:
	$(PY) src/inference/eval_vision.py --score --resume --checkpoint-every 250

# Phase 14: derive the LSTM continuous per-window score (reuse the saved M1 models, ~6 min).
# Writes results/lstm/window_scores.json (dense errors) WITHOUT touching baseline_results.json.
lstm-window-scores:
	KERAS_BACKEND=torch ESA_DATA_DIR="$(ESA_DATA_DIR)" $(PY) src/baselines/train_lstm.py \
		--missions 1 --max-channels 58 --threshold flat --z-score 4.0 \
		--reuse-models --loss-source baseline_results.json \
		--results-file baseline_results_scoredump.json --dump-window-scores window_scores.json

# Phase 14: vision PR curve (mirror of the text Phase-13 curve) + score-level fusion ensemble.
vision-pr-curve:
	$(PY) src/inference/pr_curve.py --scored results/inference_vision_scored.json \
		--out results/vision_pr_curve.json --hard results/inference_vision.json \
		--label "Vision LLM (Qwen3-VL-8B)"

ensemble:
	$(PY) src/inference/ensemble.py

# Evaluation -- Phase 5: unified comparison report across all approaches.
# Phase 6 adds Base + Frontier rows automatically when their result files are present.
# Phase 8 adds the vision detector row when results/inference_vision.json is present.
eval-all:
	$(PY) src/inference/evaluate.py --all

eval-lstm:
	$(PY) src/baselines/train_lstm.py

# Phase 5 success criteria: report exists, has required sections, metrics in [0,1],
# no approach errored, and LLM anomaly responses are substantive (advice coherence).
validate-eval:
	$(PY) -c "\
import json, math, os; \
rep = 'results/comparison_report.md'; \
met = 'results/comparison_metrics.json'; \
assert os.path.exists(rep), f'Missing {rep} -- run make eval-all first'; \
assert os.path.exists(met), f'Missing {met} -- run make eval-all first'; \
text = open(rep).read(); \
[assert_sec for assert_sec in [None] if 'Approach Comparison' in text] or (_ for _ in ()).throw(AssertionError('missing Approach Comparison section')); \
('Key Findings' in text) or (_ for _ in ()).throw(AssertionError('missing Key Findings section')); \
results = json.load(open(met)); \
errs = [r['approach'] for r in results if 'error' in r]; \
assert not errs, f'Approaches with errors: {errs}'; \
vals = [r[k] for r in results for k in ('precision','recall','f1','cef_0.5')]; \
assert all(math.isfinite(v) and 0 <= v <= 1 for v in vals), 'Metric out of [0,1] or non-finite'; \
llm = next(r for r in results if r['approach'] == 'LLM Detection'); \
assert llm.get('advice_avg_chars', 0) > 50, f'LLM anomaly responses too short: {llm.get(\"advice_avg_chars\")}'; \
print('validate-eval OK:', {r['approach']: r['f1'] for r in results}) \
"

# Phase 16 (mini-FOXES ViT + XAI, dedicated branch) -- NOT part of the anomaly comparison.
# Phase 1: thinnest end-to-end slice on CPU + synthetic random tensors (no data, no GPU).
# foxes-smoke trains 1 epoch and writes results/foxes_repro/metrics.json + a checkpoint;
# validate-foxes is the inline correctness/faithfulness gate (factored into src/foxes/validate.py).
foxes-smoke:
	$(PY) -m src.foxes.run --data synthetic --epochs 1

# Phase 2: pull a TINY (8-sample) streaming subsample of the real FOXES-Data and run one
# forward+loss pass on CPU. streaming=True + .take(n) bounds the pull (never the full ~1.46 TB).
FOXES_SUBSAMPLE_N ?= 8
foxes-data:
	$(PY) -m src.foxes.run --data foxes --subsample-n $(FOXES_SUBSAMPLE_N) --epochs 1

# Phase 3: the REAL training run -- runs ON the Vast.ai instance (A6000/4090; CUDA avoids the
# M3-MPS masked-attention NaN bug), mirroring the train-cloud pattern (launch-vast ->
# upload_data.sh -> run -> download_models.sh). See scripts/cloud/README.md for the full
# A6000/4090 invocation. The trained checkpoint lands under STAR_OUTPUT_DIR; the small
# metrics.json comes back to results/foxes_repro/. Knobs (override on the CLI):
#   FOXES_TRAIN_N (subsample, 1-2k)  FOXES_EPOCHS (20-50)  FOXES_DEVICE  FOXES_USD_PER_HR
FOXES_TRAIN_N ?= 1500
FOXES_EPOCHS ?= 40
FOXES_BATCH ?= 12
FOXES_DEVICE ?= cuda
FOXES_USD_PER_HR ?= 0.40
foxes-train:
	$(PY) -m src.foxes.run --data foxes \
		--subsample-n $(FOXES_TRAIN_N) --epochs $(FOXES_EPOCHS) --batch $(FOXES_BATCH) \
		--device $(FOXES_DEVICE) --usd-per-hr $(FOXES_USD_PER_HR) \
		--checkpoint-every 5 --resume

# Phase 4: the repo's FIRST image / heatmap / overlay rendering. Loads the trained checkpoint,
# runs a forward pass on a held-out FOXES sample (tiny streaming pull -- never the full ~1.46 TB),
# and writes the per-patch flux-attribution overlay + the raw-attention sanity figure + a
# viz_meta.json sidecar the validate gate asserts (faithfulness + non-uniformity + brightness
# correlation). Runs LOCALLY on CPU -- no cloud, no GPU. FOXES_VIZ_N is the held-out pull size.
FOXES_VIZ_N ?= 16
foxes-figures:
	$(PY) -m src.foxes.visualize --data foxes --subsample-n $(FOXES_VIZ_N)

# Phase 5: packaging. foxes-report is a convenience target that (re)renders the figures and runs
# the validate gate -- the report.md + foxes-repro_MODELCARD.md + foxes_app.py are authored
# artifacts, so this just confirms the figures + completeness gates are green. foxes-aux-smoke
# exercises the Surya AR-mask drop-in on a placeholder mask (CPU, no data); foxes-gradio-smoke is
# the headless Gradio check (calls foxes_app.predict() directly, no UI launch). All three are also
# folded into validate-foxes, so `make validate-foxes` alone covers the Phase-5 automated checks.
foxes-report: foxes-figures validate-foxes
	@echo "foxes-report: report.md + model card + figures + Gradio overlay validated."

foxes-aux-smoke:
	$(PY) -c "from src.foxes.validate import check_aux_mask_forward; check_aux_mask_forward()"

foxes-gradio-smoke:
	$(PY) -c "from src.foxes.validate import check_gradio_smoke; check_gradio_smoke()"

validate-foxes:
	$(PY) -m src.foxes.validate

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
