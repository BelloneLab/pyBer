"""Read behavior tables without changing their clocks or source measurements.

Inference is deliberately separate from GUI selection: explicit column overrides
remain available when a recording uses a naming convention we do not recognize.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIME_ALIASES = ("time", "time_s", "time_sec", "time_seconds", "trial_time",
                "trial_time_s", "recording_time", "recording_time_s",
                "timestamp_software", "timestamp_camera", "timestamp")


def _usable_time(values):
    """Require a measured, ordered clock with at least one positive time step."""
    finite = values[np.isfinite(values)]
    differences = np.diff(finite)
    return finite.size >= 2 and np.all(differences >= 0) and np.any(differences > 0)


def detect_time_column(df):
    """Find a known numeric clock without converting unrelated table columns."""
    by_key = {_key(name): name for name in df.columns}
    for key in TIME_ALIASES:
        if key in by_key:
            name = by_key[key]
            values = _numeric_values(df[name], clock=True)
            if _usable_time(values):
                return name
    return None


def _key(name):
    """Normalize labels for matching while retaining original export labels."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _numeric_values(series, *, clock=False):
    """Convert a column once; known clock labels also accept elapsed hh:mm:ss."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if clock and np.any(~np.isfinite(values)) and not pd.api.types.is_numeric_dtype(series):
        text = series.astype("string").str.strip()
        parts = text.str.extract(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:[.,]\d+)?)$")
        valid = parts[1].notna().to_numpy()
        if np.any(valid):
            values = values.copy()
            hours = pd.to_numeric(parts[0].fillna("0"), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            minutes = pd.to_numeric(parts[1], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            seconds = pd.to_numeric(parts[2].str.replace(",", ".", regex=False), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            values[valid] = (hours * 3600 + minutes * 60 + seconds)[valid]
    return values


def read_behavior_csv(path):
    """Sniff common delimiters once, then use pandas' fast C CSV reader."""
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        sample = stream.read(65536)
    lines = "\n".join(line for line in sample.splitlines() if not line.startswith("#"))
    try:
        delimiter = csv.Sniffer().sniff(lines, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    return pd.read_csv(path, sep=delimiter, comment="#", encoding="utf-8-sig", low_memory=False)


def infer_table(df, *, time_column=None, behavior_columns=None, x_column=None, y_column=None):
    """Infer clocks, binary states and coordinate pairs with optional overrides.

    Arrays retain source row order and timestamps. Missing detections become NaN
    in coordinate copies only. Automatic inference excludes acquisition metadata;
    explicit selections may use any numeric column. No synchronization or time
    rebasing is performed here.
    """
    names = list(df.columns)
    keys = {name: _key(name) for name in names}
    by_key = {key: name for name, key in keys.items()}
    numeric = {}
    for name in names:
        series = df[name]
        # Convert every column once, including text boolean states.
        if pd.api.types.is_bool_dtype(series):
            numeric[name] = series.to_numpy(dtype=float, na_value=np.nan)
        else:
            if not pd.api.types.is_numeric_dtype(series):
                series = series.replace({"True": 1, "False": 0, "true": 1, "false": 0,
                                         "TRUE": 1, "FALSE": 0, "yes": 1, "no": 0})
            numeric[name] = _numeric_values(series, clock=keys[name] in TIME_ALIASES or name == time_column)

    time_candidates = {by_key[key]: numeric[by_key[key]].copy() for key in TIME_ALIASES
                       if key in by_key and _usable_time(numeric[by_key[key]])}
    auto_time_column = next(iter(time_candidates), None)
    if time_column is not None:
        if time_column not in numeric or not _usable_time(numeric[time_column]):
            raise ValueError("Selected time column must contain increasing numeric timestamps.")
        time_candidates.setdefault(time_column, numeric[time_column].copy())
    else:
        time_column = auto_time_column

    def coordinate_identity(key):
        tokens = key.split("_")
        axes = [token for token in tokens if token in {"x", "y"}]
        if len(axes) != 1:
            return None
        axis = axes[0]
        return axis, "_".join(token for token in tokens if token != axis)

    axes = {}
    for name, key in keys.items():
        identity = coordinate_identity(key)
        if identity:
            axis, stem = identity
            axes.setdefault(stem, {})[axis] = name
    pairs = [(part["x"], part["y"]) for part in axes.values() if "x" in part and "y" in part]
    if (x_column is None) != (y_column is None):
        raise ValueError("Select both X and Y coordinate columns.")
    if x_column is not None:
        if x_column not in numeric or y_column not in numeric:
            raise ValueError("Selected coordinate column is missing.")
        pairs = [(x_column, y_column)] + [pair for pair in pairs if pair != (x_column, y_column)]

    trajectory = {}
    valid_pairs = []
    masked_counts = {}
    for x_name, y_name in pairs:
        x, y = numeric[x_name].copy(), numeric[y_name].copy()
        valid = np.isfinite(x) & np.isfinite(y)
        key = keys[x_name]
        animal = re.match(r"(mouse_\d+|animal)_", key)
        if animal:
            prefix = animal.group(1)
            # Pykaboo uses -1 for absent detections, including keypoints.
            if "timestamp_software" in by_key or "timestamp_camera" in by_key:
                valid &= ~((x == -1) | (y == -1))
            for suffix, minimum in [("detected", 0), ("confidence", 0), ("class_id", -1)]:
                column = by_key.get(prefix + "_" + suffix)
                if column is not None:
                    values = numeric[column]
                    valid &= np.isfinite(values) & (values > minimum)
            likelihood_key = re.sub(r"_x$", "_likelihood", key)
            likelihood = by_key.get(likelihood_key)
            if likelihood is not None:
                valid &= np.isfinite(numeric[likelihood]) & (numeric[likelihood] > 0)
        x[~valid], y[~valid] = np.nan, np.nan
        if np.any(valid):
            trajectory[x_name], trajectory[y_name] = x, y
            valid_pairs.append((x_name, y_name))
            masked_counts[x_name] = int(np.count_nonzero(~valid))

    coordinate_names = {name for pair in pairs for name in pair}
    pykaboo = "timestamp_software" in by_key and any(key.startswith("live_") or key.startswith("mouse_1_") for key in by_key)

    def metadata(key):
        if key in TIME_ALIASES:
            return True
        if key.startswith(("timestamp", "frame_", "camera_", "live_", "recording_", "hardware_")):
            return True
        if key.endswith(("_id", "_count", "_confidence", "_likelihood", "_prob", "_vector")):
            return True
        return key in {"time", "trial_time", "recording_time", "animal_id", "session", "trial", "experiment",
                       "condition", "arena", "date", "gain_db", "sensor_width", "sensor_height", "exposure_time_us",
                       "behavior_state", "behavior_active", "behavior_decision_frame", "behavior_backend",
                       "ttl_state", "animal_detected", "mouse_1_detected", "mouse_2_detected"}

    behaviors = {}
    explicit_behaviors = set(behavior_columns) if behavior_columns is not None else None
    if explicit_behaviors is not None and not explicit_behaviors.issubset(numeric):
        raise ValueError("Selected behavior column is missing.")
    for name, values in numeric.items():
        if pykaboo and keys[name].startswith("behavior_"):
            values = values.copy()
            values[values == -1] = np.nan
        finite = values[np.isfinite(values)]
        binary = finite.size > 0 and np.all((finite == 0) | (finite == 1))
        if explicit_behaviors is not None:
            if name in explicit_behaviors:
                if not binary:
                    raise ValueError(f"Behavior column '{name}' must contain binary 0/1 states.")
                behaviors[name] = values.copy()
            continue
        key = keys[name]
        if name == time_column or name in coordinate_names or metadata(key):
            continue
        if pykaboo and not (key.startswith("behavior_") or key in {"gate", "sync", "injection", "dev2_laser_ttl", "user_flag_event", "user_flag_ttl"}):
            continue
        if binary:
            behaviors[name] = values.copy()
    # Older Pykaboo releases serialize states instead of writing dedicated columns.
    # Parse each distinct state vector once, preserving missing values as NaN.
    vector_name = by_key.get("behavior_state_vector")
    if explicit_behaviors is None and vector_name is not None:
        vectors = df[vector_name].fillna("").astype(str)
        parsed = {}
        state_names = set()
        for vector in vectors.unique():
            states = {}
            for field in vector.split("|"):
                label, separator, raw = field.partition("=")
                if separator and raw.strip() in {"0", "1", "0.0", "1.0"}:
                    states[label.strip()] = float(raw)
            parsed[vector] = states
            state_names.update(states)
        for label in sorted(state_names):
            name = "behavior_" + label.removeprefix("behavior_")
            if name not in behaviors and name not in df.columns:
                mapping = {vector: states.get(label, np.nan) for vector, states in parsed.items()}
                behaviors[name] = vectors.map(mapping).to_numpy(dtype=float)
    # Preserve generic continuous channels for spatial/activity selectors, without
    # presenting Pykaboo timing, confidence, IDs or acquisition settings as tracks.
    for name, values in numeric.items():
        if name == time_column or name in coordinate_names or name in behaviors or metadata(keys[name]):
            continue
        if not pykaboo and np.count_nonzero(np.isfinite(values)) >= 2:
            trajectory[name] = values.copy()

    def pair_priority(pair):
        key = keys[pair[0]]
        priority = {"mouse_1_center_x": 0, "x": 1, "x_center": 2, "center_x": 2, "animal_center_x": 3}
        return priority.get(key, 4), names.index(pair[0])

    valid_pairs.sort(key=pair_priority)
    default = (x_column, y_column) if x_column is not None and (x_column, y_column) in valid_pairs else (valid_pairs[0] if valid_pairs else None)
    return {"time_column": time_column,
            "auto_time_column": auto_time_column, "time_candidates": time_candidates,
            "time": numeric[time_column].copy() if time_column is not None else np.array([], dtype=float),
            "behaviors": behaviors, "trajectory": trajectory, "coordinate_pairs": valid_pairs,
            "default_coordinate_pair": default,
            "report": {"format": "Pykaboo" if pykaboo else "Generic table", "rows": len(df),
                       "time_column": time_column, "time_candidates": list(time_candidates), "behavior_columns": list(behaviors),
                       "coordinate_pairs": valid_pairs, "masked_coordinate_samples": masked_counts,
                       "time_rebased": False}}
