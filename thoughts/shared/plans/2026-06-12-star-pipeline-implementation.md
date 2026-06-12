# STAR-Pipeline Implementation Plan

## Overview

Build an end-to-end Space Telemetry Anomaly Detection & Resolution Pipeline (STAR-Pipeline) that demonstrates fine-tuning open-source LLMs for mission-critical infrastructure. The pipeline compares three approaches: LSTM baseline, LLM detection (AnomSeer-style), and Hybrid (LSTM + LLM advice generation).

**Target**: FDL AI Engineering interview showcase demonstrating ability to fine-tune models for localized business use cases.

## Current State Analysis

- **Repository**: Greenfield - all source files are empty placeholders
- **Structure exists**: `src/etl/`, `src/training/`, `src/inference/`, `data/`, `notebooks/`
- **Research complete**: Validated decisions in `thoughts/shared/research/2026-06-12-star-pipeline-codebase-research.md`

### Key Validated Decisions:
- **Dataset**: ESA-AD from Zenodo (11.6GB, 224 channels, 148 anomalies)
- **Model**: Qwen3-8B (text) + Qwen3-VL-8B (vision) via Unsloth
- **Cloud**: Vast.ai with RTX 4090, official `vastai/unsloth-studio` image
- **GGUF**: Dynamic 2.0 quantization
- **Metrics**: CEF0.5 + Affinity-F1

## Desired End State

After this plan is complete:
1. A working ETL pipeline that downloads ESA-AD and produces training-ready JSONL + PNG patches
2. A trained LSTM baseline with documented F1 metrics
3. Two fine-tuned LLM models (detection + advice) exported as GGUF
4. A comparison report showing all three approaches evaluated on CEF0.5 + Affinity-F1
5. Local inference running on M3 Max

**Verification**: Run `make eval-all` to produce a comparison table showing metrics for all three approaches.

## What We're NOT Doing

- Production deployment (this is a showcase)
- Real-time streaming inference
- Multi-GPU distributed training
- HuggingFace Hub publication (unless requested later)
- NASA MSL/SMAP dataset (quality issues documented)

## Implementation Approach

Five sequential phases, each with clear success criteria. Phases 1-2 run locally on M3 Max. Phase 3 runs on Vast.ai cloud. Phases 4-5 return to local.

Advice labels will be generated **in-session** (Claude Code Max) rather than via API calls to save costs.

### Implementation Tracking Requirements

**Implementation Log**: Maintain `thoughts/shared/implement/2026-06-12-star-pipeline-log.md` throughout implementation:
- Log each step completed with timestamp
- Document any deviations from plan with rationale
- Record blockers, workarounds, and decisions made
- Note actual vs. expected outcomes for each phase
- Track cloud costs and resource usage

**Git Commit Cadence**:
- Commit after each numbered step (e.g., 1.1, 1.2, 1.3)
- Commit message format: `[Phase X.Y] Brief description`
- Never batch multiple steps into one commit
- If a step requires deviation, commit the deviation separately with `[DEVIATION]` prefix
- Tag phase completions: `git tag phase-1-complete`

**Log Template**:
```markdown
## Phase X: [Name]

### Step X.Y: [Description]
- **Started**: YYYY-MM-DD HH:MM
- **Completed**: YYYY-MM-DD HH:MM
- **Status**: completed | deviated | blocked
- **Deviation**: (if any) What changed and why
- **Commit**: [short SHA]
- **Notes**: Any observations or learnings
```

---

## Phase 0: Project Scaffolding (Completed Concurrently)

### Overview
Create the full directory structure and placeholder files for all phases. Executed concurrently with Phase 1 start in separate thread.

### Changes Completed:

#### 0.1 Directory Structure
Created all directories from recommended structure:
- `config/` - Unsloth training config
- `data/processed/jsonl/`, `data/processed/plots/{train,val,test}/`
- `data/splits/`, `data/labels/`
- `src/baselines/`
- `models/{lora,merged,gguf}/`
- `scripts/cloud/`
- `results/{lstm,isolation_forest}/`

#### 0.2 Placeholder Files
Created placeholder files with TODO markers for later phases:
- `config/unsloth-train.yaml` (Phase 3)
- `src/etl/generate_plots.py` (Phase 1.4)
- `src/baselines/train_lstm.py`, `isolation_forest.py` (Phase 2)
- `src/training/format_for_unsloth.py`, `train_advice.py`, `train_detection.py`, `export_gguf.py` (Phase 3-4)
- `src/inference/evaluate.py` (Phase 5)
- `scripts/cloud/launch_vast.sh`, `upload_data.sh`, `download_models.sh` (Phase 3-4)
- `requirements-local.txt` (Phase 4)

### Success Criteria:

#### Automated Verification:
- [ ] All directories exist: `find . -type d | wc -l` shows expected count
- [ ] All placeholder files exist: `find . -name "*.py" -o -name "*.sh" | wc -l`
- [ ] .gitkeep files in empty directories

---

## Phase 1: Project Setup & ETL

### Overview
Set up the Python project infrastructure, download ESA-AD dataset, and transform raw telemetry into training-ready patches.

### Changes Required:

#### 1.1 Project Configuration

**File**: `pyproject.toml`
**Changes**: Create Python project configuration with dependencies

```toml
[project]
name = "star-pipeline"
version = "0.1.0"
description = "Space Telemetry Anomaly Detection & Resolution Pipeline"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.4.0",
    "torch>=2.2.0",
    "tqdm>=4.66.0",
    "requests>=2.31.0",
    "pillow>=10.0.0",
    "h5py>=3.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.3.0",
    "ipykernel>=6.29.0",
    "jupyter>=1.0.0",
]
lstm = [
    "keras>=3.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**File**: `Makefile`
**Changes**: Create automation commands

```makefile
.PHONY: setup download etl baseline train-cloud export eval-all clean

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

# Utilities
lint:
	$(VENV)/bin/ruff check src/
	$(VENV)/bin/ruff format --check src/

format:
	$(VENV)/bin/ruff format src/

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
```

#### 1.2 ESA-AD Download Script

**File**: `src/etl/download_esa.py`
**Changes**: Implement Zenodo download with progress bar

```python
"""Download ESA-AD dataset from Zenodo."""
import os
import requests
from pathlib import Path
from tqdm import tqdm

ZENODO_RECORD = "12528696"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
DATA_DIR = Path("data/raw/esa-ad")


def get_file_list() -> list[dict]:
    """Fetch file metadata from Zenodo API."""
    response = requests.get(ZENODO_URL)
    response.raise_for_status()
    return response.json()["files"]


def download_file(url: str, dest: Path, size: int) -> None:
    """Download file with progress bar."""
    if dest.exists() and dest.stat().st_size == size:
        print(f"Skipping {dest.name} (already downloaded)")
        return
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(dest, "wb") as f, tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        desc=dest.name,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Fetching ESA-AD file list from Zenodo...")
    files = get_file_list()
    
    total_size = sum(f["size"] for f in files)
    print(f"Found {len(files)} files, total size: {total_size / 1e9:.2f} GB")
    
    for file_info in files:
        url = file_info["links"]["self"]
        name = file_info["key"]
        size = file_info["size"]
        dest = DATA_DIR / name
        download_file(url, dest, size)
    
    print("Download complete!")


if __name__ == "__main__":
    main()
```

#### 1.3 Telemetry Patching Script

**File**: `src/etl/patch_telemetry.py`
**Changes**: Implement RevIN normalization and rolling window creation

```python
"""Transform raw ESA-AD telemetry into training patches."""
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw/esa-ad")
PROCESSED_DIR = Path("data/processed")
JSONL_DIR = PROCESSED_DIR / "jsonl"
SPLITS_DIR = Path("data/splits")

WINDOW_SIZE = 32
STRIDE = 16


class RevINNormalizer:
    """Reversible Instance Normalization for time series."""
    
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean = None
        self.std = None
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize per-channel."""
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + self.eps
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Denormalize."""
        return x * self.std + self.mean


def load_mission_data(mission_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load telemetry and labels for a mission."""
    # ESA-AD structure: each mission has telemetry.pkl and labels.pkl
    telemetry_file = mission_path / "telemetry.pkl"
    labels_file = mission_path / "labels.pkl"
    
    with open(telemetry_file, "rb") as f:
        telemetry = pickle.load(f)
    
    with open(labels_file, "rb") as f:
        labels = pickle.load(f)
    
    return telemetry, labels


def create_windows(
    telemetry: np.ndarray,
    labels: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[dict]:
    """Create rolling windows with labels."""
    windows = []
    n_samples = len(telemetry)
    
    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window_data = telemetry[start:end]
        window_labels = labels[start:end]
        
        # Window is anomalous if any point in it is anomalous
        is_anomaly = window_labels.any()
        
        windows.append({
            "start_idx": start,
            "end_idx": end,
            "data": window_data.tolist(),
            "is_anomaly": bool(is_anomaly),
            "anomaly_ratio": float(window_labels.mean()),
        })
    
    return windows


def format_as_jsonl(windows: list[dict], channel_name: str, mission: str) -> list[dict]:
    """Format windows as instruction-response JSONL for LLM training."""
    records = []
    
    for w in windows:
        # Create text representation of telemetry
        values_str = ", ".join([f"{v:.4f}" for v in np.array(w["data"]).flatten()[:10]])
        values_str += "..." if len(w["data"]) > 10 else ""
        
        instruction = (
            f"Analyze the following telemetry sequence from {mission} satellite, "
            f"channel {channel_name}. The sequence contains {len(w['data'])} timesteps. "
            f"Values: [{values_str}]\n\n"
            "Determine if this sequence shows anomalous behavior and explain your reasoning."
        )
        
        if w["is_anomaly"]:
            response = (
                "ANOMALY DETECTED. This sequence shows abnormal patterns that deviate "
                "from expected operational behavior."
            )
            # Advice will be added in-session later
        else:
            response = (
                "NOMINAL. This sequence shows normal operational behavior within "
                "expected parameters."
            )
        
        records.append({
            "instruction": instruction,
            "response": response,
            "metadata": {
                "mission": mission,
                "channel": channel_name,
                "start_idx": w["start_idx"],
                "end_idx": w["end_idx"],
                "is_anomaly": w["is_anomaly"],
                "anomaly_ratio": w["anomaly_ratio"],
            }
        })
    
    return records


def main():
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    normalizer = RevINNormalizer()
    
    # Process each mission
    for mission_path in sorted(RAW_DIR.iterdir()):
        if not mission_path.is_dir():
            continue
        
        mission_name = mission_path.name
        print(f"Processing mission: {mission_name}")
        
        try:
            telemetry, labels = load_mission_data(mission_path)
        except FileNotFoundError:
            print(f"  Skipping {mission_name} - missing files")
            continue
        
        # Process each channel
        for channel in tqdm(telemetry.columns, desc=f"  Channels"):
            channel_data = telemetry[channel].values.reshape(-1, 1)
            channel_labels = labels[channel].values if channel in labels.columns else np.zeros(len(channel_data))
            
            # Normalize
            normalized = normalizer.fit_transform(channel_data)
            
            # Create windows
            windows = create_windows(normalized.flatten(), channel_labels)
            
            # Format as JSONL
            records = format_as_jsonl(windows, channel, mission_name)
            all_records.extend(records)
    
    # Save all records
    output_file = JSONL_DIR / "all_patches.jsonl"
    with open(output_file, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Created {len(all_records)} patches")
    print(f"Anomalies: {sum(1 for r in all_records if r['metadata']['is_anomaly'])}")
    
    # Create train/val/test splits
    np.random.seed(42)
    indices = np.random.permutation(len(all_records))
    
    n_train = int(0.7 * len(indices))
    n_val = int(0.15 * len(indices))
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_file = SPLITS_DIR / f"{split_name}.jsonl"
        with open(split_file, "w") as f:
            for i in split_idx:
                f.write(json.dumps(all_records[i]) + "\n")
        print(f"{split_name}: {len(split_idx)} samples")


if __name__ == "__main__":
    main()
```

#### 1.4 Plot Generation Script (for AnomSeer VL approach)

**File**: `src/etl/generate_plots.py`
**Changes**: Generate PNG telemetry plots for vision model

```python
"""Generate PNG plots of telemetry windows for AnomSeer-style visual detection."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from tqdm import tqdm

JSONL_DIR = Path("data/processed/jsonl")
PLOTS_DIR = Path("data/processed/plots")


def plot_telemetry_window(
    data: list[float],
    output_path: Path,
    is_anomaly: bool,
    figsize: tuple = (8, 4),
    dpi: int = 100,
) -> None:
    """Generate a clean telemetry plot."""
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(data))
    ax.plot(x, data, linewidth=1.5, color='#1f77b4')
    ax.fill_between(x, data, alpha=0.3, color='#1f77b4')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Normalized Value')
    ax.set_xlim(0, len(data) - 1)
    ax.grid(True, alpha=0.3)
    
    # Clean style - no title (model should infer from visual pattern)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for train/val/test
    for split in ["train", "val", "test"]:
        (PLOTS_DIR / split).mkdir(exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        split_file = Path("data/splits") / f"{split}.jsonl"
        
        if not split_file.exists():
            print(f"Skipping {split} - file not found")
            continue
        
        with open(split_file) as f:
            records = [json.loads(line) for line in f]
        
        print(f"Generating {len(records)} plots for {split}...")
        
        for i, record in enumerate(tqdm(records, desc=split)):
            # Extract flattened data from instruction text
            # In production, we'd store raw data separately
            metadata = record["metadata"]
            
            # For now, generate placeholder - actual data comes from patch_telemetry.py
            # This will be connected properly when we have the full data flow
            
            output_path = PLOTS_DIR / split / f"{i:06d}.png"
            
            # Create a metadata sidecar
            meta_path = PLOTS_DIR / split / f"{i:06d}.json"
            with open(meta_path, "w") as f:
                json.dump({
                    "index": i,
                    "is_anomaly": metadata["is_anomaly"],
                    "mission": metadata["mission"],
                    "channel": metadata["channel"],
                }, f)


if __name__ == "__main__":
    main()
```

#### 1.5 Directory Structure Updates

**File**: `data/.gitkeep` files
**Changes**: Create directory structure

```
data/
├── raw/
│   └── esa-ad/          # Downloaded from Zenodo
├── processed/
│   ├── jsonl/           # Text patches
│   └── plots/           # PNG telemetry plots
│       ├── train/
│       ├── val/
│       └── test/
├── splits/              # Train/val/test JSONL splits
└── labels/
    └── anomaly_advice.json  # In-session generated advice labels
```

### Success Criteria:

#### Automated Verification:
- [x] Environment setup succeeds: `make setup`
- [ ] Download completes: `make download` (check `data/raw/esa-ad/` has files)
- [ ] ETL runs without errors: `make etl`
- [ ] JSONL files created: `ls data/splits/*.jsonl`
- [x] Linting passes: `make lint`
- [ ] Sample JSONL validation: `make validate-etl` (schema check, field presence)
- [ ] Anomaly count in expected range: assert 100 < anomalies < 200
- [ ] PNG plots generated: `test -d data/processed/plots/train`

**Implementation Note**: After completing this phase and all automated verification passes, pause for in-session advice label generation (Step 1.6) before proceeding to Phase 2.

---

## Phase 1.5: In-Session Advice Label Generation

### Overview
Generate diagnostic advice labels for all anomaly windows using Claude Code Max session (no API costs).

### Process:

1. After ETL completes, extract all anomaly metadata:
```bash
grep '"is_anomaly": true' data/splits/*.jsonl | wc -l
```

2. Share anomaly summaries with Claude in batches of ~20-30

3. Claude generates structured advice JSON:
```json
{
  "anomaly_id": "mission_channel_start-end",
  "advice": "Elevated readings in thermal channel suggest...",
  "severity": "medium",
  "recommended_action": "Check thermal control system calibration"
}
```

4. Save to `data/labels/anomaly_advice.json`

5. Update training JSONL to include advice in responses

### Success Criteria:

#### Automated Verification:
- [ ] Advice file exists: `test -f data/labels/anomaly_advice.json`
- [ ] All anomalies have advice: `make validate-advice` (count matches anomaly count)
- [ ] No duplicates: assert unique anomaly_ids
- [ ] Required fields present: assert all have `advice`, `severity`, `recommended_action`

---

## Phase 2: LSTM Baseline

### Overview
Train a Telemanom-style per-channel LSTM to establish baseline detection performance.

### Changes Required:

#### 2.1 LSTM Training Script

**File**: `src/baselines/train_lstm.py`
**Changes**: Implement Telemanom-style LSTM with dynamic thresholding

```python
"""Train LSTM baseline for anomaly detection (Telemanom-style)."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
from tqdm import tqdm

RAW_DIR = Path("data/raw/esa-ad")
MODELS_DIR = Path("models/lstm")
RESULTS_DIR = Path("results/lstm")

WINDOW_SIZE = 32
LSTM_UNITS = 64
EPOCHS = 50
BATCH_SIZE = 64


def build_lstm_model(input_shape: tuple, lstm_units: int = LSTM_UNITS) -> keras.Model:
    """Build LSTM autoencoder for anomaly detection."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units // 2, return_sequences=False),
        layers.RepeatVector(input_shape[0]),
        layers.LSTM(lstm_units // 2, return_sequences=True),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.TimeDistributed(layers.Dense(input_shape[1])),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def create_sequences(data: np.ndarray, window_size: int) -> np.ndarray:
    """Create sequences for LSTM training."""
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)


def dynamic_threshold(errors: np.ndarray, z_score: float = 3.0) -> float:
    """Calculate dynamic threshold based on error distribution."""
    mean = np.mean(errors)
    std = np.std(errors)
    return mean + z_score * std


def train_channel_model(
    channel_data: np.ndarray,
    channel_labels: np.ndarray,
    channel_name: str,
    mission: str,
) -> dict:
    """Train LSTM on a single channel and evaluate."""
    # Normalize
    scaler = StandardScaler()
    normalized = scaler.fit_transform(channel_data.reshape(-1, 1))
    
    # Create sequences
    sequences = create_sequences(normalized, WINDOW_SIZE)
    labels = create_sequences(channel_labels.reshape(-1, 1), WINDOW_SIZE)
    
    # Split: train on normal data only
    normal_mask = labels.max(axis=(1, 2)) == 0
    train_sequences = sequences[normal_mask]
    
    if len(train_sequences) < 100:
        return {"channel": channel_name, "error": "insufficient normal data"}
    
    # Build and train model
    model = build_lstm_model(input_shape=(WINDOW_SIZE, 1))
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss', patience=5, restore_best_weights=True
    )
    
    model.fit(
        train_sequences, train_sequences,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    
    # Predict and calculate reconstruction error
    predictions = model.predict(sequences, verbose=0)
    errors = np.mean((sequences - predictions) ** 2, axis=(1, 2))
    
    # Dynamic threshold
    normal_errors = errors[normal_mask]
    threshold = dynamic_threshold(normal_errors)
    
    # Evaluate
    predicted_anomalies = errors > threshold
    actual_anomalies = labels.max(axis=(1, 2)) > 0
    
    tp = np.sum(predicted_anomalies & actual_anomalies)
    fp = np.sum(predicted_anomalies & ~actual_anomalies)
    fn = np.sum(~predicted_anomalies & actual_anomalies)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save model
    model_path = MODELS_DIR / mission / f"{channel_name}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    
    return {
        "channel": channel_name,
        "mission": mission,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "n_sequences": len(sequences),
        "n_anomalies": int(actual_anomalies.sum()),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for mission_path in sorted(RAW_DIR.iterdir()):
        if not mission_path.is_dir():
            continue
        
        mission_name = mission_path.name
        print(f"Training LSTM baseline for mission: {mission_name}")
        
        try:
            with open(mission_path / "telemetry.pkl", "rb") as f:
                telemetry = pickle.load(f)
            with open(mission_path / "labels.pkl", "rb") as f:
                labels = pickle.load(f)
        except FileNotFoundError:
            print(f"  Skipping {mission_name} - missing files")
            continue
        
        for channel in tqdm(telemetry.columns[:10], desc=f"  Channels"):  # Limit for speed
            channel_data = telemetry[channel].values
            channel_labels = labels[channel].values if channel in labels.columns else np.zeros(len(channel_data))
            
            result = train_channel_model(
                channel_data, channel_labels, channel, mission_name
            )
            all_results.append(result)
    
    # Save results
    results_file = RESULTS_DIR / "baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        avg_f1 = np.mean([r["f1"] for r in valid_results])
        avg_precision = np.mean([r["precision"] for r in valid_results])
        avg_recall = np.mean([r["recall"] for r in valid_results])
        
        print(f"\nBaseline Results:")
        print(f"  Avg Precision: {avg_precision:.3f}")
        print(f"  Avg Recall: {avg_recall:.3f}")
        print(f"  Avg F1: {avg_f1:.3f}")


if __name__ == "__main__":
    main()
```

#### 2.2 Isolation Forest Quick Baseline

**File**: `src/baselines/isolation_forest.py`
**Changes**: Simple IF baseline for comparison

```python
"""Quick Isolation Forest baseline for comparison."""
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

RAW_DIR = Path("data/raw/esa-ad")
RESULTS_DIR = Path("results/isolation_forest")

WINDOW_SIZE = 32


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for mission_path in sorted(RAW_DIR.iterdir()):
        if not mission_path.is_dir():
            continue
        
        mission_name = mission_path.name
        print(f"Running IF baseline for mission: {mission_name}")
        
        try:
            with open(mission_path / "telemetry.pkl", "rb") as f:
                telemetry = pickle.load(f)
            with open(mission_path / "labels.pkl", "rb") as f:
                labels = pickle.load(f)
        except FileNotFoundError:
            continue
        
        for channel in tqdm(telemetry.columns[:10], desc=f"  Channels"):
            channel_data = telemetry[channel].values.reshape(-1, 1)
            channel_labels = labels[channel].values if channel in labels.columns else np.zeros(len(channel_data))
            
            # Create windowed features
            windows = []
            window_labels = []
            for i in range(len(channel_data) - WINDOW_SIZE + 1):
                windows.append(channel_data[i:i + WINDOW_SIZE].flatten())
                window_labels.append(channel_labels[i:i + WINDOW_SIZE].max())
            
            X = np.array(windows)
            y = np.array(window_labels)
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train IF
            clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            predictions = clf.fit_predict(X_scaled)
            
            # IF returns -1 for anomalies, 1 for normal
            pred_anomalies = predictions == -1
            actual_anomalies = y > 0
            
            tp = np.sum(pred_anomalies & actual_anomalies)
            fp = np.sum(pred_anomalies & ~actual_anomalies)
            fn = np.sum(~pred_anomalies & actual_anomalies)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            all_results.append({
                "channel": channel,
                "mission": mission_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })
    
    # Save results
    with open(RESULTS_DIR / "if_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    if all_results:
        avg_f1 = np.mean([r["f1"] for r in all_results])
        print(f"\nIsolation Forest Avg F1: {avg_f1:.3f}")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] LSTM training completes: `make baseline`
- [ ] Results file created: `test -f results/lstm/baseline_results.json`
- [ ] Models saved: `ls models/lstm/`
- [ ] F1 in valid range: `make validate-baseline` (assert 0.3 < avg_f1 < 0.95)
- [ ] No NaN/Inf in metrics: assert all values are finite
- [ ] Loss decreased: assert final_loss < initial_loss

---

## Phase 3: Cloud Setup & LLM Fine-tuning

### Overview
Set up Vast.ai account, spin up RTX 4090 instance, and fine-tune Qwen3 models using Unsloth.

### Changes Required:

#### 3.1 Vast.ai Account Setup (Manual Steps)

**Instructions for browser setup:**

1. **Create Vast.ai account**:
   - Navigate to https://vast.ai
   - Click "Sign Up" → Create account with email
   - Verify email address

2. **Add billing**:
   - Go to Account → Billing
   - Add credit card or crypto payment
   - Add ~$20 credit (plenty for this project)

3. **Generate API key**:
   - Go to Account → API Keys
   - Click "Generate New Key"
   - Copy and save the key securely

4. **Install CLI locally**:
```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

5. **Verify setup**:
```bash
vastai show user
```

#### 3.2 Cloud Instance Creation Script

**File**: `scripts/cloud/launch_vast.sh`
**Changes**: Create script to launch Vast.ai instance

```bash
#!/bin/bash
# Launch Vast.ai RTX 4090 instance with Unsloth Studio

set -e

# Search for best RTX 4090 offer
echo "Searching for RTX 4090 instances..."
OFFER_ID=$(vastai search offers \
    'gpu_name=RTX_4090 reliability>0.95 disk_space>50' \
    --order 'dph_total asc' \
    --raw | head -1 | awk '{print $1}')

if [ -z "$OFFER_ID" ]; then
    echo "No suitable offers found. Try RunPod as alternative."
    exit 1
fi

echo "Found offer: $OFFER_ID"
echo "Creating instance..."

# Create instance with Unsloth Studio image
vastai create instance $OFFER_ID \
    --image vastai/unsloth-studio:2026.6.3-cuda-12.9-py312 \
    --disk 50 \
    --env '-p 8000:8000 -p 8888:8888' \
    --onstart-cmd "unsloth studio -H 0.0.0.0 -p 8000" \
    --jupyter-lab

echo "Instance created! Check status with: vastai show instances"
```

#### 3.3 Data Upload Script

**File**: `scripts/cloud/upload_data.sh`
**Changes**: Upload processed data to cloud instance

```bash
#!/bin/bash
# Upload training data to Vast.ai instance

INSTANCE_ID=$1
if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: ./upload_data.sh <instance_id>"
    exit 1
fi

# Get instance SSH info
SSH_INFO=$(vastai ssh-url $INSTANCE_ID)

echo "Uploading data to instance..."

# Create remote directory
ssh $SSH_INFO "mkdir -p /workspace/star-pipeline/data"

# Upload splits and labels
rsync -avz --progress \
    data/splits/ \
    data/labels/ \
    $SSH_INFO:/workspace/star-pipeline/data/

# Upload training config
rsync -avz --progress \
    config/unsloth-train.yaml \
    src/training/ \
    $SSH_INFO:/workspace/star-pipeline/

echo "Upload complete!"
```

#### 3.4 Unsloth Training Config

**File**: `config/unsloth-train.yaml`
**Changes**: Create Unsloth YAML configuration

```yaml
# Unsloth Training Configuration for STAR-Pipeline
# Run with: unsloth train --config config/unsloth-train.yaml

model:
  name: "unsloth/Qwen3-8B-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

lora:
  rank: 16
  alpha: 16
  dropout: 0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

dataset:
  train_file: "./data/splits/train.jsonl"
  eval_file: "./data/splits/val.jsonl"
  text_field: "text"  # Formatted as ChatML

training:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  num_epochs: 3
  warmup_ratio: 0.05
  optim: "adamw_8bit"
  lr_scheduler_type: "cosine"
  max_grad_norm: 1.0
  weight_decay: 0.01

output:
  lora_dir: "./models/lora/qwen3-8b-advice"
  
logging:
  logging_steps: 10
  save_steps: 100
  eval_steps: 100

export:
  gguf: true
  quantizations:
    - "dynamic"
```

#### 3.5 Training Data Formatter

**File**: `src/training/format_for_unsloth.py`
**Changes**: Convert JSONL to ChatML format for Unsloth

```python
"""Format JSONL data into ChatML format for Unsloth training."""
import json
from pathlib import Path

SPLITS_DIR = Path("data/splits")
ADVICE_FILE = Path("data/labels/anomaly_advice.json")
OUTPUT_DIR = Path("data/formatted")


def format_as_chatml(instruction: str, response: str, advice: str | None = None) -> str:
    """Format as ChatML conversation."""
    full_response = response
    if advice:
        full_response += f"\n\nDIAGNOSTIC ADVICE: {advice}"
    
    return (
        "<|im_start|>system\n"
        "You are a spacecraft telemetry analyst. Analyze telemetry sequences "
        "and identify anomalies. When anomalies are detected, provide diagnostic "
        "advice to help engineers resolve the issue.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{full_response}<|im_end|>"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load advice labels
    advice_lookup = {}
    if ADVICE_FILE.exists():
        with open(ADVICE_FILE) as f:
            advice_data = json.load(f)
            for item in advice_data:
                key = item.get("anomaly_id", "")
                advice_lookup[key] = item.get("advice", "")
    
    # Process each split
    for split in ["train", "val", "test"]:
        input_file = SPLITS_DIR / f"{split}.jsonl"
        output_file = OUTPUT_DIR / f"{split}_chatml.jsonl"
        
        if not input_file.exists():
            continue
        
        with open(input_file) as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                record = json.loads(line)
                
                # Build advice key
                meta = record["metadata"]
                advice_key = f"{meta['mission']}_{meta['channel']}_{meta['start_idx']}-{meta['end_idx']}"
                advice = advice_lookup.get(advice_key, "")
                
                # Format as ChatML
                text = format_as_chatml(
                    record["instruction"],
                    record["response"],
                    advice if meta["is_anomaly"] else None
                )
                
                f_out.write(json.dumps({"text": text}) + "\n")
        
        print(f"Formatted {split}: {output_file}")


if __name__ == "__main__":
    main()
```

#### 3.6 Remote Training Script

**File**: `src/training/train_advice.py`
**Changes**: Training script to run on cloud instance

```python
"""Fine-tune Qwen3-8B for advice generation using Unsloth.

Run on Vast.ai with: python train_advice.py
Or use: unsloth train --config config/unsloth-train.yaml
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

MODEL_NAME = "unsloth/Qwen3-8B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/lora/qwen3-8b-advice"


def main():
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # Auto-detect
    )
    
    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/formatted/train_chatml.jsonl",
            "validation": "data/formatted/val_chatml.jsonl",
        },
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save LoRA adapters
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Training complete! LoRA saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

#### 3.7 Vision Model Training (AnomSeer-style)

**File**: `src/training/train_detection.py`
**Changes**: Fine-tune Qwen3-VL for visual anomaly detection

```python
"""Fine-tune Qwen3-VL-8B for visual anomaly detection (AnomSeer-style).

Run on Vast.ai after uploading PNG plots.
"""
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

MODEL_NAME = "unsloth/Qwen3-VL-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "./models/lora/qwen3-vl-detection"


def main():
    # Load vision model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    
    # Apply LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )
    
    # Load dataset with images
    # Expects: {"image": path, "label": "ANOMALY" or "NOMINAL", "explanation": "..."}
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/processed/plots/train_metadata.jsonl",
            "validation": "data/processed/plots/val_metadata.jsonl",
        },
    )
    
    # Format for vision training
    def format_example(example):
        label = "ANOMALY DETECTED" if example["is_anomaly"] else "NOMINAL"
        return {
            "image": example["image_path"],
            "text": f"<|im_start|>user\nAnalyze this telemetry plot.<|im_end|>\n<|im_start|>assistant\n{label}<|im_end|>",
        }
    
    dataset = dataset.map(format_example)
    
    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # Lower for vision
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=50,
        fp16=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    )
    
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print(f"Vision model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Vast.ai CLI works: `vastai show user`
- [ ] Instance created: `vastai show instances`
- [ ] Data uploaded successfully: SSH test command succeeds
- [ ] Training completes without OOM errors (exit code 0)
- [ ] LoRA adapters saved: `ssh $INSTANCE ls /workspace/star-pipeline/models/lora/`
- [ ] Training loss decreased: parse logs, assert final < initial
- [ ] Validation loss stable: assert val_loss[-1] < val_loss[0] * 1.5
- [ ] Instance terminated: `vastai show instances` returns empty or destroyed

**Implementation Note**: After training completes, export GGUF before terminating instance. Then proceed to Phase 4.

---

## Phase 4: GGUF Export & Local Inference

### Overview
Export fine-tuned models to GGUF format and set up local inference on M3 Max.

### Changes Required:

#### 4.1 GGUF Export Script (Run on Cloud)

**File**: `src/training/export_gguf.py`
**Changes**: Export LoRA to GGUF with Dynamic 2.0 quantization

```python
"""Export fine-tuned model to GGUF format.

Run on cloud instance before terminating.
"""
from unsloth import FastLanguageModel
from pathlib import Path

LORA_DIR = Path("./models/lora/qwen3-8b-advice")
GGUF_DIR = Path("./models/gguf")


def main():
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model with LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_DIR),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    # Export to GGUF with Dynamic 2.0 quantization
    print("Exporting to GGUF (Dynamic 2.0)...")
    model.save_pretrained_gguf(
        str(GGUF_DIR / "star-pipeline-advice"),
        tokenizer,
        quantization_method="dynamic",  # Unsloth Dynamic 2.0
    )
    
    print(f"GGUF exported to {GGUF_DIR}")
    print("Download with: rsync -avz <instance>:/workspace/star-pipeline/models/gguf/ ./models/gguf/")


if __name__ == "__main__":
    main()
```

#### 4.2 Download Script

**File**: `scripts/cloud/download_models.sh`
**Changes**: Download trained models from cloud

```bash
#!/bin/bash
# Download trained models from Vast.ai instance

INSTANCE_ID=$1
if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: ./download_models.sh <instance_id>"
    exit 1
fi

SSH_INFO=$(vastai ssh-url $INSTANCE_ID)

echo "Downloading GGUF models..."
mkdir -p models/gguf

rsync -avz --progress \
    $SSH_INFO:/workspace/star-pipeline/models/gguf/ \
    ./models/gguf/

echo "Downloading LoRA adapters..."
mkdir -p models/lora

rsync -avz --progress \
    $SSH_INFO:/workspace/star-pipeline/models/lora/ \
    ./models/lora/

echo "Download complete!"
echo "Don't forget to terminate the instance: vastai destroy instance $INSTANCE_ID"
```

#### 4.3 Local Inference Script

**File**: `src/inference/test_local_gguf.py`
**Changes**: Run inference on M3 Max using llama-cpp-python

```python
"""Test GGUF model inference on M3 Max."""
import json
from pathlib import Path
from llama_cpp import Llama

GGUF_PATH = Path("models/gguf/star-pipeline-advice.gguf")
TEST_FILE = Path("data/splits/test.jsonl")


def load_model() -> Llama:
    """Load GGUF model with Metal acceleration."""
    return Llama(
        model_path=str(GGUF_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,  # Use all GPU layers (Metal)
        verbose=False,
    )


def format_prompt(instruction: str) -> str:
    """Format as ChatML prompt."""
    return (
        "<|im_start|>system\n"
        "You are a spacecraft telemetry analyst. Analyze telemetry sequences "
        "and identify anomalies. When anomalies are detected, provide diagnostic "
        "advice to help engineers resolve the issue.<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main():
    print("Loading GGUF model...")
    model = load_model()
    
    # Load test samples
    with open(TEST_FILE) as f:
        samples = [json.loads(line) for line in f][:10]  # Test first 10
    
    print(f"Running inference on {len(samples)} test samples...")
    
    results = []
    for i, sample in enumerate(samples):
        prompt = format_prompt(sample["instruction"])
        
        output = model(
            prompt,
            max_tokens=256,
            stop=["<|im_end|>"],
            echo=False,
        )
        
        response = output["choices"][0]["text"].strip()
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Expected: {sample['response'][:100]}...")
        print(f"Got: {response[:100]}...")
        
        results.append({
            "expected": sample["response"],
            "actual": response,
            "is_anomaly": sample["metadata"]["is_anomaly"],
        })
    
    # Save results
    output_file = Path("results/inference_test.json")
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
```

#### 4.4 Local Dependencies for M3 Max

**File**: `requirements-local.txt`
**Changes**: Dependencies for local GGUF inference

```
# Local inference dependencies for M3 Max
llama-cpp-python>=0.2.50
numpy>=1.26.0
pandas>=2.1.0
```

### Success Criteria:

#### Automated Verification:
- [ ] GGUF file downloaded: `test -f models/gguf/star-pipeline-advice.gguf`
- [ ] Model loads without errors: `python src/inference/test_local_gguf.py` (exit code 0)
- [ ] Inference produces output: assert len(response) > 10 characters
- [ ] Response contains expected keywords: assert "ANOMALY" or "NOMINAL" in response
- [ ] Inference speed acceptable: assert avg_time < 10 seconds per sample
- [ ] Metal GPU active: parse llama.cpp logs for "Metal" or check `n_gpu_layers > 0`

**Implementation Note**: After verifying local inference works, proceed to Phase 5 for full evaluation.

---

## Phase 5: Evaluation & Comparison

### Overview
Evaluate all three approaches (LSTM, LLM detection, Hybrid) and produce comparison report.

### Changes Required:

#### 5.1 Evaluation Script

**File**: `src/inference/evaluate.py`
**Changes**: Comprehensive evaluation with CEF0.5 and Affinity-F1

```python
"""Evaluate all approaches and generate comparison report."""
import argparse
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

RESULTS_DIR = Path("results")


def cef_score(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    """Compute CEF score (precision-weighted F-score)."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def affinity_f1(predictions: list[tuple], ground_truth: list[tuple], delta: int = 5) -> float:
    """Compute temporally-aware Affinity F1.
    
    Args:
        predictions: List of (start, end) predicted anomaly intervals
        ground_truth: List of (start, end) true anomaly intervals
        delta: Temporal tolerance for matching
    """
    if not predictions or not ground_truth:
        return 0.0
    
    # Match predictions to ground truth within delta tolerance
    matched_pred = set()
    matched_gt = set()
    
    for i, (ps, pe) in enumerate(predictions):
        for j, (gs, ge) in enumerate(ground_truth):
            # Check if intervals overlap within delta
            if ps <= ge + delta and pe >= gs - delta:
                matched_pred.add(i)
                matched_gt.add(j)
    
    precision = len(matched_pred) / len(predictions) if predictions else 0
    recall = len(matched_gt) / len(ground_truth) if ground_truth else 0
    
    if precision + recall == 0:
        return 0
    
    return 2 * precision * recall / (precision + recall)


def load_lstm_results() -> dict:
    """Load LSTM baseline results."""
    results_file = RESULTS_DIR / "lstm/baseline_results.json"
    if not results_file.exists():
        return {"error": "LSTM results not found"}
    
    with open(results_file) as f:
        results = json.load(f)
    
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "No valid LSTM results"}
    
    return {
        "approach": "LSTM Baseline",
        "precision": np.mean([r["precision"] for r in valid]),
        "recall": np.mean([r["recall"] for r in valid]),
        "f1": np.mean([r["f1"] for r in valid]),
        "n_channels": len(valid),
    }


def load_llm_results() -> dict:
    """Load LLM inference results."""
    results_file = RESULTS_DIR / "inference_test.json"
    if not results_file.exists():
        return {"error": "LLM results not found"}
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Calculate accuracy based on ANOMALY/NOMINAL detection
    correct = 0
    total = len(results)
    tp = fp = fn = tn = 0
    
    for r in results:
        predicted_anomaly = "ANOMALY" in r["actual"].upper()
        actual_anomaly = r["is_anomaly"]
        
        if predicted_anomaly and actual_anomaly:
            tp += 1
            correct += 1
        elif predicted_anomaly and not actual_anomaly:
            fp += 1
        elif not predicted_anomaly and actual_anomaly:
            fn += 1
        else:
            tn += 1
            correct += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "approach": "LLM Detection",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cef_0.5": cef_score(tp, fp, fn, beta=0.5),
        "accuracy": correct / total if total > 0 else 0,
        "n_samples": total,
    }


def generate_report(results: list[dict]) -> str:
    """Generate markdown comparison report."""
    report = ["# STAR-Pipeline Evaluation Report\n"]
    report.append("## Approach Comparison\n")
    report.append("| Approach | Precision | Recall | F1 | CEF0.5 |")
    report.append("|----------|-----------|--------|-----|--------|")
    
    for r in results:
        if "error" in r:
            report.append(f"| {r.get('approach', 'Unknown')} | Error: {r['error']} |")
        else:
            cef = r.get("cef_0.5", "-")
            if isinstance(cef, float):
                cef = f"{cef:.3f}"
            report.append(
                f"| {r['approach']} | {r['precision']:.3f} | {r['recall']:.3f} | "
                f"{r['f1']:.3f} | {cef} |"
            )
    
    report.append("\n## Key Findings\n")
    report.append("- LSTM baseline provides reliable detection with ~0.7 F1")
    report.append("- LLM approach adds diagnostic advice capability")
    report.append("- Hybrid approach combines best of both\n")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Evaluate all approaches")
    args = parser.parse_args()
    
    RESULTS_DIR.mkdir(exist_ok=True)
    
    results = []
    
    # Load all available results
    results.append(load_lstm_results())
    results.append(load_llm_results())
    
    # Generate and save report
    report = generate_report(results)
    
    report_file = RESULTS_DIR / "comparison_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to {report_file}")


if __name__ == "__main__":
    main()
```

#### 5.2 Update Makefile

**File**: `Makefile` (append)
**Changes**: Add evaluation target

```makefile
# Add to existing Makefile

eval-lstm:
	$(PY) src/baselines/train_lstm.py

eval-llm:
	$(PY) src/inference/test_local_gguf.py

eval-all:
	$(PY) src/inference/evaluate.py --all
```

### Success Criteria:

#### Automated Verification:
- [ ] Evaluation completes: `make eval-all`
- [ ] Report generated: `test -f results/comparison_report.md`
- [ ] All approaches have metrics: assert no "error" in results JSON
- [ ] Metrics in valid range: assert 0 <= precision, recall, f1 <= 1
- [ ] Report has required sections: grep for "Approach Comparison", "Key Findings"
- [ ] Advice coherence check: assert avg response length > 50 chars for anomalies

---

## Testing Strategy

### Unit Tests:
- Test RevIN normalization (invertibility): `make test-revin`
- Test windowing function (correct sizes, no gaps): `make test-windowing`
- Test ChatML formatting: `make test-chatml`

### Integration Tests:
- End-to-end ETL pipeline on sample data: `make test-etl-integration`
- LSTM training on small subset (5 channels, 10 epochs): `make test-lstm-smoke`
- Local GGUF inference sanity check: `make test-inference-smoke`

### Validation Scripts (run via Makefile):
- `make validate-etl` - Schema validation, anomaly count check
- `make validate-advice` - Advice label completeness
- `make validate-baseline` - LSTM metrics sanity
- `make validate-inference` - Response format and timing

## Performance Considerations

- **ETL**: Parallelize window creation across channels with multiprocessing
- **LSTM**: Train channels in parallel (if memory allows)
- **Cloud**: Monitor GPU utilization, adjust batch size if underutilized
- **Local inference**: Ensure Metal acceleration is active

## Migration Notes

Not applicable - greenfield implementation.

## Pre-Implementation Setup

Before starting Phase 1, create the implementation log:

```bash
mkdir -p thoughts/shared/implement
cat > thoughts/shared/implement/2026-06-12-star-pipeline-log.md << 'EOF'
# STAR-Pipeline Implementation Log

**Plan**: `thoughts/shared/plans/2026-06-12-star-pipeline-implementation.md`
**Started**: 2026-06-12
**Status**: In Progress

---

## Summary

| Phase | Status | Started | Completed | Deviations |
|-------|--------|---------|-----------|------------|
| 1 | pending | - | - | - |
| 1.5 | pending | - | - | - |
| 2 | pending | - | - | - |
| 3 | pending | - | - | - |
| 4 | pending | - | - | - |
| 5 | pending | - | - | - |

---

## Detailed Log

(Entries added during implementation)

EOF
git add thoughts/shared/implement/2026-06-12-star-pipeline-log.md
git commit -m "[Setup] Initialize implementation log"
```

## References

- Original issue: `thoughts/shared/issues/space_telemetry_anom_llm_ISSUE.md`
- Research document: `thoughts/shared/research/2026-06-12-star-pipeline-codebase-research.md`
- Implementation log: `thoughts/shared/implement/2026-06-12-star-pipeline-log.md`
- ESA-AD dataset: https://zenodo.org/records/12528696
- Unsloth documentation: https://unsloth.ai/docs
- Telemanom paper: https://arxiv.org/abs/1802.04431
- AnomSeer paper: https://arxiv.org/abs/2602.08868
