"""Shared I/O for the real ESA Anomaly Dataset (ESA-AD) structure.

Single source of truth for loading ESA-AD — imported by `patch_telemetry.py`
(Phase 1 ETL), `train_lstm.py`, and `isolation_forest.py` (Phase 2 baselines) so
the loader logic lives in exactly one place.

ESA-AD does NOT ship `telemetry.pkl` / `labels.pkl` per mission (the original plan's
assumption, deviation D2). Each mission directory looks like:

    ESA-Mission1/
        channels.csv      # Channel, Subsystem, Physical Unit, Group, Target(YES/NO)
        labels.csv        # ID, Channel, StartTime, EndTime  (ISO-8601 UTC intervals)
        anomaly_types.csv # ID, Class, Subclass, Category, ...
        channels/channel_N/channel_N   # pickled pandas DataFrame:
                                        #   DatetimeIndex + 1 column (float OR categorical)

Notes baked in from the Phase 1 deviations:
  - D4: channels are LONG (up to ~15M rows at ~90 s cadence). Resample before windowing.
  - D5: Mission3 channels are CATEGORICAL ('value_0', 'value_1', ...). `load_channel_series`
    ordinal-encodes them (extract trailing int). macOS writes '._<name>' resource-fork
    entries on FAT32 drives; `iter_channels` skips them.
  - Data lives on the external DUAL DRIVE; read `ESA_DATA_DIR` rather than hard-coding.
"""

import os
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

#: Default raw-data root. Override via ESA_DATA_DIR, e.g.
#: ESA_DATA_DIR="/Volumes/DUAL DRIVE/esa-ad"
DEFAULT_RAW_DIR = Path(os.environ.get("ESA_DATA_DIR", "data/raw/esa-ad"))


class RevINNormalizer:
    """Reversible Instance Normalization for time series (per-channel)."""

    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + self.eps
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def discover_missions(data_dir: Path, missions: str = "all") -> list[Path]:
    """Return the ESA-Mission* directories under `data_dir`, filtered by `missions`.

    `missions` is "all" or a comma-separated list of mission numbers, e.g. "1" or "1,2,3".
    """
    dirs = sorted(d for d in Path(data_dir).glob("ESA-Mission*") if d.is_dir())
    if missions.strip().lower() != "all":
        wanted = {int(x) for x in missions.split(",") if x.strip()}
        dirs = [d for d in dirs if int(d.name.replace("ESA-Mission", "")) in wanted]
    return dirs


def channel_file_path(mission_dir: Path, channel: str) -> Path:
    """Path to a channel's pickled DataFrame: <mission>/channels/<channel>/<channel>."""
    return mission_dir / "channels" / channel / channel


def list_channels(mission_dir: Path, target_only: bool = True) -> list[str]:
    """Return channel names from channels.csv (filtered to Target==YES by default)."""
    meta = pd.read_csv(mission_dir / "channels.csv")
    if target_only:
        meta = meta[meta["Target"] == "YES"]
    return meta["Channel"].tolist()


def iter_channels(mission_dir: Path, target_only: bool = True) -> Iterator[tuple[str, Path]]:
    """Yield (channel_name, channel_file) for each readable channel in a mission.

    Skips channels whose file is missing and macOS '._' resource-fork entries that
    appear on FAT32 volumes (they otherwise raise NotADirectoryError / load garbage).
    """
    for channel in list_channels(mission_dir, target_only=target_only):
        channel_file = channel_file_path(mission_dir, channel)
        if not channel_file.exists() or channel_file.name.startswith("."):
            continue
        yield channel, channel_file


def load_channel_series(channel_file: Path) -> pd.Series:
    """Load a per-channel pickled DataFrame and return a float32 Series w/ DatetimeIndex.

    Mission3 channels are categorical (strings like 'value_0', 'value_1') rather than
    continuous floats (deviation D5). We ordinal-encode them: extract the trailing
    integer so 'value_0'->0.0, 'value_1'->1.0, etc. This preserves the discrete state
    signal and lets downstream normalisation proceed normally.
    """
    with open(channel_file, "rb") as f:
        df = pickle.load(f)
    if isinstance(df, pd.DataFrame):
        series = df.iloc[:, 0]
    else:
        series = df  # already a Series
    series.index = pd.to_datetime(series.index)
    if series.dtype == object:
        # Ordinal-encode categorical strings of the form 'value_N'
        series = series.str.extract(r"(\d+)$", expand=False).astype("float32")
    return series.astype("float32")


def resample_series(series: pd.Series, rule: str = "1h") -> pd.Series:
    """Resample to a uniform cadence (mean) and interpolate gaps. `rule='none'` is a no-op.

    Use the SAME rule across the ETL and the baselines so the LSTM/LLM detectors operate
    on an identical time grid (apples-to-apples comparison) and training stays tractable.
    """
    if rule.lower() == "none":
        return series.dropna()
    return series.resample(rule).mean().interpolate(limit_direction="both").dropna()


def load_labels(mission_dir: Path) -> pd.DataFrame:
    """Load labels.csv with parsed, tz-naive UTC interval bounds."""
    labels = pd.read_csv(mission_dir / "labels.csv")
    for col in ("StartTime", "EndTime"):
        labels[col] = pd.to_datetime(labels[col], utc=True).dt.tz_localize(None)
    return labels


def anomaly_mask_for_channel(
    index: pd.DatetimeIndex, channel: str, labels: pd.DataFrame
) -> np.ndarray:
    """Boolean mask over `index`: True where the timestamp lies in a labelled anomaly
    interval for `channel`."""
    mask = np.zeros(len(index), dtype=bool)
    ch_labels = labels[labels["Channel"] == channel]
    idx_values = index.values
    for _, row in ch_labels.iterrows():
        start = np.datetime64(row["StartTime"])
        end = np.datetime64(row["EndTime"])
        mask |= (idx_values >= start) & (idx_values <= end)
    return mask
