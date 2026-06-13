"""Generate structured diagnostic advice labels for all anomalous telemetry windows.

Phase 1.5 of STAR-Pipeline. Advice is derived in-session from:
  - Window statistics (pattern type: spike, drift, oscillation, sustained offset)
  - Channel metadata (subsystem, physical unit) from channels.csv
  - Anomaly severity (anomaly_ratio)

Output: data/labels/anomaly_advice.json  (list of advice records)
        data/splits/*_with_advice.jsonl   (original splits + advice merged into response)
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path("data/splits")
LABELS_DIR = Path("data/labels")
DATA_DIR = Path(os.environ.get("ESA_DATA_DIR", "data/raw/esa-ad"))

# Plausible satellite subsystem names for anonymised ESA-AD labels
SUBSYSTEM_MAP = {
    "subsystem_1": "Power Subsystem",
    "subsystem_2": "Attitude & Orbit Control",
    "subsystem_3": "Thermal Control",
    "subsystem_4": "Payload / Instruments",
    "subsystem_5": "Telemetry & Data Handling",
    "subsystem_6": "Propulsion",
    "subsystem_7": "Command & Control",
    "subsystem_8": "Structure & Mechanisms",
    "subsystem_9": "Communications",
    "subsystem_10": "Electrical Power Distribution",
}

UNIT_MAP = {
    "physical_unit_1": "voltage (V)",
    "physical_unit_2": "current (A)",
    "physical_unit_3": "temperature (°C)",
    "physical_unit_4": "pressure (Pa)",
    "physical_unit_5": "angular rate (°/s)",
    "physical_unit_6": "quaternion component",
    "physical_unit_7": "power (W)",
    "physical_unit_8": "binary state",
    "physical_unit_9": "frequency (Hz)",
    "physical_unit_10": "flux",
    "physical_unit_11": "acceleration (m/s²)",
    "physical_unit_12": "magnetic field (T)",
}


def classify_pattern(values: list[float], anomaly_ratio: float) -> tuple[str, str]:
    """Return (pattern_name, statistical_description) for a window."""
    v = np.array(values, dtype=float)
    mean_v = float(v.mean())
    std_v = float(v.std())
    peak_to_peak = float(v.max() - v.min())
    # Linear trend via first/last quarter means
    first_q = v[: len(v) // 4].mean()
    last_q = v[-len(v) // 4 :].mean()
    trend_slope = last_q - first_q

    if anomaly_ratio >= 0.75:
        if abs(mean_v) > 2.0:
            pattern = "sustained_offset"
            desc = (
                f"persistent offset (mean={mean_v:.2f}σ) throughout "
                f"{int(anomaly_ratio * 100)}% of the window"
            )
        elif std_v > 2.5:
            pattern = "sustained_oscillation"
            desc = (
                f"high-amplitude oscillation (σ={std_v:.2f}) over "
                f"{int(anomaly_ratio * 100)}% of the window"
            )
        else:
            pattern = "persistent_anomaly"
            desc = f"anomalous behaviour across {int(anomaly_ratio * 100)}% of the window"
    elif std_v > 3.0 or peak_to_peak > 6.0:
        pattern = "spike"
        pct = int(anomaly_ratio * 100)
        desc = f"transient spike (peak-to-peak={peak_to_peak:.2f}σ, {pct}% of window flagged)"
    elif abs(trend_slope) > 1.5:
        direction = "upward" if trend_slope > 0 else "downward"
        pattern = "drift"
        desc = f"monotonic {direction} drift (Δ={trend_slope:.2f}σ across window)"
    elif std_v > 1.8:
        pattern = "oscillation"
        desc = (
            f"elevated variability (σ={std_v:.2f}) with "
            f"{int(anomaly_ratio * 100)}% anomalous timesteps"
        )
    else:
        pattern = "subtle_deviation"
        desc = (
            f"subtle deviation ({int(anomaly_ratio * 100)}% of timesteps outside nominal envelope)"
        )

    return pattern, desc


def severity(anomaly_ratio: float) -> str:
    if anomaly_ratio >= 0.5:
        return "high"
    if anomaly_ratio >= 0.15:
        return "medium"
    return "low"


PATTERN_ADVICE = {
    "sustained_offset": (
        "Channel has drifted to a persistent off-nominal level. Check sensor calibration "
        "or zero-point drift. Verify reference baseline against housekeeping archive."
    ),
    "sustained_oscillation": (
        "Channel shows high-amplitude periodic oscillation. Investigate control-loop "
        "instability or coupling from an adjacent subsystem."
    ),
    "persistent_anomaly": (
        "Channel exhibits prolonged off-nominal behaviour. Cross-check with adjacent "
        "channels in the same subsystem group for correlated faults."
    ),
    "spike": (
        "Transient spike detected. Likely causes include electrostatic discharge, "
        "single-event upset, or a brief external stimulus. Monitor for recurrence."
    ),
    "drift": (
        "Monotonic trend indicates gradual degradation or environmental shift. "
        "Evaluate long-term trends and compare against expected mission lifecycle profile."
    ),
    "oscillation": (
        "Elevated variability suggests intermittent noise source or unstable operating "
        "point. Check for loose connections, vibration coupling, or thermal cycling effects."
    ),
    "subtle_deviation": (
        "Low-amplitude deviation from nominal envelope. Could be sensor noise floor "
        "increase, minor calibration shift, or early-stage degradation. Log and monitor."
    ),
}

SEVERITY_ACTION = {
    "high": (
        "Escalate to mission operations. "
        "Consider safing the affected subsystem pending investigation."
    ),
    "medium": (
        "Schedule inspection at next ground contact. "
        "Cross-correlate with event log and FDIR records."
    ),
    "low": "Flag for trend monitoring. Review at next routine housekeeping downlink.",
}


def load_channel_meta(data_dir: Path) -> dict[tuple[str, str], dict]:
    """Return {(mission, channel): {subsystem, physical_unit, group}} for all missions."""
    meta = {}
    for mdir in sorted(data_dir.glob("ESA-Mission*")):
        csv_path = mdir / "channels.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            key = (mdir.name, row["Channel"])
            meta[key] = {
                "subsystem": row.get("Subsystem", "unknown"),
                "physical_unit": row.get("Physical Unit", "unknown"),
                "group": row.get("Group", "?"),
            }
    return meta


def make_advice_record(record: dict, ch_meta: dict) -> dict:
    meta = record["metadata"]
    values = meta["values"]
    ar = meta["anomaly_ratio"]
    pattern, stat_desc = classify_pattern(values, ar)
    sev = severity(ar)

    mission = meta["mission"]
    channel = meta["channel"]
    key = (mission, channel)
    ch = ch_meta.get(key, {})
    subsystem_raw = ch.get("subsystem", "unknown")
    unit_raw = ch.get("physical_unit", "unknown")
    group = ch.get("group", "?")

    subsystem_raw = str(subsystem_raw) if not isinstance(subsystem_raw, str) else subsystem_raw
    unit_raw = str(unit_raw) if not isinstance(unit_raw, str) else unit_raw
    subsystem_name = SUBSYSTEM_MAP.get(subsystem_raw, subsystem_raw.replace("_", " ").title())
    unit_name = UNIT_MAP.get(unit_raw, unit_raw.replace("_", " "))

    anomaly_id = f"{mission}__{channel}__{meta['start_time'].replace(' ', 'T')}"

    advice_text = (
        f"{subsystem_name} channel {channel} (group {group}, {unit_name}) shows {stat_desc}. "
        f"{PATTERN_ADVICE[pattern]}"
    )
    action = SEVERITY_ACTION[sev]

    return {
        "anomaly_id": anomaly_id,
        "mission": mission,
        "channel": channel,
        "subsystem": subsystem_name,
        "physical_unit": unit_name,
        "group": str(group),
        "start_time": meta["start_time"],
        "end_time": meta["end_time"],
        "anomaly_ratio": ar,
        "pattern": pattern,
        "severity": sev,
        "advice": advice_text,
        "recommended_action": action,
    }


def build_enriched_response(record: dict, advice: dict) -> str:
    base = record["response"]
    if not advice:
        return base
    return (
        f"{base}\n\n"
        f"DIAGNOSIS: {advice['pattern'].replace('_', ' ').upper()} "
        f"(severity: {advice['severity']})\n"
        f"ADVICE: {advice['advice']}\n"
        f"ACTION: {advice['recommended_action']}"
    )


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ch_meta = load_channel_meta(DATA_DIR)

    split_files = sorted(SPLITS_DIR.glob("*.jsonl"))
    if not split_files:
        raise FileNotFoundError(f"No split files found in {SPLITS_DIR}")

    all_advice: list[dict] = []
    advice_by_id: dict[str, dict] = {}

    print("Generating advice labels for anomalous windows...")
    for split_file in split_files:
        records = [json.loads(line) for line in open(split_file)]
        anomalous = [r for r in records if r["metadata"]["is_anomaly"]]
        print(f"  {split_file.name}: {len(anomalous)} anomalous records")
        for r in anomalous:
            adv = make_advice_record(r, ch_meta)
            if adv["anomaly_id"] not in advice_by_id:
                advice_by_id[adv["anomaly_id"]] = adv
                all_advice.append(adv)

    out_path = LABELS_DIR / "anomaly_advice.json"
    with open(out_path, "w") as f:
        json.dump(all_advice, f, indent=2)
    print(f"\nSaved {len(all_advice)} advice records -> {out_path}")

    severity_counts = {}
    pattern_counts = {}
    for a in all_advice:
        severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1
        pattern_counts[a["pattern"]] = pattern_counts.get(a["pattern"], 0) + 1
    print(f"Severity: {severity_counts}")
    print(f"Patterns: {dict(sorted(pattern_counts.items(), key=lambda x: -x[1]))}")

    # Write enriched splits with advice merged into the response field
    print("\nWriting enriched splits...")
    for split_file in split_files:
        records = [json.loads(line) for line in open(split_file)]
        out_file = SPLITS_DIR / split_file.name.replace(".jsonl", "_with_advice.jsonl")
        with open(out_file, "w") as f:
            for r in records:
                if r["metadata"]["is_anomaly"]:
                    meta = r["metadata"]
                    start = meta["start_time"].replace(" ", "T")
                    aid = f"{meta['mission']}__{meta['channel']}__{start}"
                    adv = advice_by_id.get(aid)
                    r = dict(r)
                    r["response"] = build_enriched_response(r, adv)
                    if adv:
                        r["metadata"]["advice_id"] = aid
                        r["metadata"]["severity"] = adv["severity"]
                        r["metadata"]["pattern"] = adv["pattern"]
                f.write(json.dumps(r) + "\n")
        print(f"  {out_file.name}: {len(records)} records")

    print("\nPhase 1.5 complete.")


if __name__ == "__main__":
    main()
