"""Rescale a dF/F trace by the size of its own spontaneous events.

The "prominence normalized" output divides the fitted-reference dF/F trace by
one number: the mean size of the largest events found in a baseline region of
the same recording. Two event-size measures are available.

``prominence`` (default, the original MATLAB workflow)
    The trace is centred on the median of the baseline region, events are
    ranked by peak prominence (height above the higher of the two surrounding
    troughs) and ``out = (dF/F - median) / mean(top prominences)``.

``amplitude`` (Pascoli et al., Nat Neurosci 2026, Extended Data Fig. 1)
    "We selected the top 10% spontaneous events and used their mean amplitude
    to rescale the whole trace." Events are local maxima whose dF/F value is at
    least ``min_amplitude`` (0.2 % dF/F in the paper), ranked by that dF/F
    value, and ``out = dF/F / mean(top amplitudes)``. The paper does not state
    the minimum spacing or prominence of an event; both stay user settings.

The baseline region is a time window, the whole recording, or the recording
minus (or restricted to) windows around events. Events come from the A/D
channel or from an event file in any of the formats the postprocessing tab
reads: EthoVision exports, binary state tables with a time column, onset
tables (one column of onset times per behavior) and plain event-time lists.

Nothing in this module modifies its inputs or the files it reads.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from signal_events import detect_peaks, estimate_noise


# ---------------------------------------------------------------------------
# Option identifiers (stored in ProcessingParams / config files)
# ---------------------------------------------------------------------------

METHOD_PROMINENCE = "prominence"   # rank by prominence, centre on baseline median
METHOD_AMPLITUDE = "amplitude"     # rank by dF/F peak value, divide only

UNITS_ABSOLUTE = "absolute"        # thresholds in dF/F (0.002 = 0.2 %)
UNITS_NOISE = "noise"              # thresholds in multiples of baseline noise sigma

ROLE_EXCLUDE = "exclude"           # baseline = recording outside event windows
ROLE_INCLUDE = "include"           # baseline = inside event windows only

SOURCE_WINDOW = "window"
SOURCE_EVENTS = "events"           # A/D (DIO) channel
SOURCE_FILE = "file"
SOURCE_WHOLE = "whole"

ALL_EVENT_LABELS = ""              # use every label of the event file

# Values published in Pascoli et al. 2026 (Methods, "Fiber photometry").
# Spacing and prominence of an event are not specified there, so they are
# deliberately absent from this preset.
PAPER_PRESET: Dict[str, Any] = {
    "prominence_method": METHOD_AMPLITUDE,
    "prominence_percent_top": 0.10,
    "prominence_threshold_units": UNITS_ABSOLUTE,
    "prominence_min_amplitude": 0.002,
    "prominence_exclude_before_s": 5.0,
    "prominence_exclude_after_s": 30.0,
    "prominence_event_role": ROLE_EXCLUDE,
}

# Rolling-median window used to remove slow drift before measuring noise.
NOISE_BASELINE_WINDOW_S = 10.0


# ---------------------------------------------------------------------------
# Event files
# ---------------------------------------------------------------------------

@dataclass
class EventTable:
    """Events read from one file: label -> (onsets, offsets), seconds.

    Point events (onset lists) have ``offset == onset``. Bouts from state
    columns keep their observed offsets, including bouts that are still
    ongoing at the first or last sample, because those samples are also
    "during the event" and must not leak into a spontaneous baseline.
    """

    path: str
    format: str
    events: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    note: str = ""

    def labels(self) -> List[str]:
        return list(self.events.keys())

    def select(self, label: str = ALL_EVENT_LABELS) -> Tuple[np.ndarray, np.ndarray]:
        """Onsets/offsets of one label, or of every label merged when empty."""
        if label and label in self.events:
            on, off = self.events[label]
            return np.asarray(on, float).copy(), np.asarray(off, float).copy()
        if label and label not in self.events:
            return np.array([], float), np.array([], float)
        if not self.events:
            return np.array([], float), np.array([], float)
        on = np.concatenate([np.asarray(v[0], float) for v in self.events.values()])
        off = np.concatenate([np.asarray(v[1], float) for v in self.events.values()])
        order = np.argsort(on, kind="stable")
        return on[order], off[order]

    def count(self, label: str = ALL_EVENT_LABELS) -> int:
        return int(self.select(label)[0].size)


def bouts_from_states(time: np.ndarray, state: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Return (onsets, offsets) of every high run of a sampled state column.

    A run that is high on the first (last) sample starts (ends) there. Missing
    samples count as low. The offset is the first low sample after the run,
    or the last sample for a run that reaches the end.
    """
    t = np.asarray(time, float).reshape(-1)
    x = np.asarray(state, float).reshape(-1)
    n = min(t.size, x.size)
    t, x = t[:n], x[:n]
    if n == 0:
        return np.array([], float), np.array([], float)
    high = np.isfinite(x) & (x > float(threshold)) & np.isfinite(t)
    if not np.any(high):
        return np.array([], float), np.array([], float)
    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    ends = np.flatnonzero(high & ~np.r_[high[1:], False])  # last high sample of each run
    off_idx = np.minimum(ends + 1, n - 1)
    return t[starts].astype(float), t[off_idx].astype(float)


def _numeric_event_list(df) -> np.ndarray:
    """Event times from a plain list (first column named like a time, else first numeric)."""
    import pandas as pd

    hint = re.compile(r"(time|event|onset|start|sec)", re.IGNORECASE)
    columns = [c for c in df.columns if hint.search(str(c))] + list(df.columns)
    for col in columns:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size:
            return np.sort(np.unique(values))
    return np.array([], float)


def _read_table(path: str):
    """Read CSV/TSV/TXT or the first sheet of an XLSX; EthoVision sheets are cleaned."""
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        with pd.ExcelFile(path) as xls:  # close at once: open handles lock files on Windows
            sheet = xls.sheet_names[0] if xls.sheet_names else None
        if sheet is None:
            raise ValueError("The workbook has no sheet.")
        try:
            from pathlib import Path
            from ethovision_process_gui import clean_sheet, find_header_row

            find_header_row(Path(path), sheet)
        except (ValueError, ImportError):
            return pd.read_excel(path, sheet_name=sheet), "spreadsheet"
        return clean_sheet(Path(path), sheet, interpolate=True), "EthoVision"
    from behavior_import import read_behavior_csv

    return read_behavior_csv(path), "table"


def read_event_table(path: str) -> EventTable:
    """Read an event/behavior file with the same format rules as postprocessing.

    Order of interpretation:
      1. onset table (one column of onset times per behavior),
      2. binary state columns on a recognised time column (EthoVision,
         Pykaboo, generic 0/1 tables): every high run becomes a bout,
      3. a plain list of event times (legacy single-column CSV).

    Raises ``ValueError`` with a readable message when nothing usable is found.
    Results are cached on (path, modification time, size).
    """
    if not path or not os.path.isfile(path):
        raise ValueError("Event file not found.")
    stat = os.stat(path)
    return _read_event_table_cached(os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _read_event_table_cached(path: str, _mtime_ns: int, _size: int) -> EventTable:
    from behavior_import import detect_onset_table, infer_table, select_identity_rows

    df, source = _read_table(path)
    if df is None or len(df.index) == 0:
        raise ValueError("The event file is empty.")
    # Multi-animal exports must be split before they can be used; the message
    # explains how many streams the file contains.
    df, _identity = select_identity_rows(df, None)

    onsets = detect_onset_table(df)
    if onsets is not None:
        events = {
            str(label): (np.asarray(v, float), np.asarray(v, float))
            for label, v in onsets["onsets"].items()
        }
        return EventTable(path, "onset table", events)

    inferred = infer_table(df)
    time = np.asarray(inferred.get("time", np.array([], float)), float)
    behaviors = inferred.get("behaviors") or {}
    if inferred.get("time_column") and behaviors and time.size:
        events = {str(name): bouts_from_states(time, values) for name, values in behaviors.items()}
        fmt = "EthoVision states" if source == "EthoVision" else "binary states"
        return EventTable(path, fmt, events, note=f"time column '{inferred.get('time_column')}'")
    if behaviors and not inferred.get("time_column"):
        raise ValueError(
            "Binary state columns were found but no time column (time, Trial time, timestamp...). "
            "Add a time column in seconds so the states can be placed on the recording."
        )

    times = _numeric_event_list(df)
    if times.size == 0:
        raise ValueError("No event times found (expected onset columns, 0/1 states with a time column, or a time list).")
    return EventTable(path, "event list", {"events": (times, times.copy())})


# ---------------------------------------------------------------------------
# Baseline region
# ---------------------------------------------------------------------------

def event_window_mask(
    time: np.ndarray,
    onsets: np.ndarray,
    offsets: Optional[np.ndarray],
    before_s: float,
    after_s: float,
) -> np.ndarray:
    """True for samples within [onset - before, offset + after] of any event."""
    t = np.asarray(time, float)
    on = np.asarray(onsets, float).reshape(-1)
    off = on.copy() if offsets is None else np.asarray(offsets, float).reshape(-1)
    if off.size != on.size:
        off = on.copy()
    off = np.where(np.isfinite(off) & (off >= on), off, on)
    before, after = max(0.0, float(before_s)), max(0.0, float(after_s))
    inside = np.zeros(t.shape, dtype=bool)
    finite_t = np.isfinite(t)
    order = np.argsort(np.where(finite_t, t, np.inf), kind="stable")
    t_sorted = t[order]
    for a, b in zip(on, off):
        if not np.isfinite(a):
            continue
        lo = np.searchsorted(t_sorted, a - before, side="left")
        hi = np.searchsorted(t_sorted, b + after, side="right")
        inside[order[lo:hi]] = True
    return inside & finite_t


def _window_mask(t: np.ndarray, start_s: float, end_s: float) -> Tuple[np.ndarray, str]:
    """Explicit window relative to the first sample; end <= 0 means 'to the end'."""
    finite_t = np.isfinite(t)
    if not np.any(finite_t):
        return np.zeros(t.shape, bool), "no finite time"
    t_lo = float(np.nanmin(t[finite_t]))
    t_hi = float(np.nanmax(t[finite_t]))
    a = t_lo + max(0.0, float(start_s))
    b = t_lo + float(end_s) if float(end_s) > 0 else t_hi
    if b <= a:
        return finite_t.copy(), "empty window; whole recording used"
    return finite_t & (t >= a) & (t <= b), ""


def baseline_region(
    time: np.ndarray,
    params: Any,
    dio_onsets: Optional[np.ndarray] = None,
    dio_offsets: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Select the samples whose events define the scale.

    Returns ``mask``, ``source`` (what was actually used), ``n_events``,
    ``warnings`` and, for file sources, ``file_format`` / ``labels``.
    """
    t = np.asarray(time, float)
    source = str(getattr(params, "prominence_baseline_source", SOURCE_EVENTS) or SOURCE_EVENTS).lower()
    role = str(getattr(params, "prominence_event_role", ROLE_EXCLUDE) or ROLE_EXCLUDE).lower()
    before = float(getattr(params, "prominence_exclude_before_s", 0.0) or 0.0)
    after = float(getattr(params, "prominence_exclude_after_s", 0.0) or 0.0)
    info: Dict[str, Any] = {"source": source, "role": role, "n_events": 0, "warnings": []}

    if source == SOURCE_WINDOW:
        mask, note = _window_mask(
            t,
            float(getattr(params, "prominence_baseline_start_s", 0.0) or 0.0),
            float(getattr(params, "prominence_baseline_end_s", 0.0) or 0.0),
        )
        if note:
            info["warnings"].append(f"Baseline window: {note}.")
        info["mask"] = mask
        return info
    if source == SOURCE_WHOLE:
        info["mask"] = np.isfinite(t)
        return info

    on = np.asarray(dio_onsets if dio_onsets is not None else [], float)
    off = np.asarray(dio_offsets if dio_offsets is not None else on, float)
    if source == SOURCE_FILE:
        path = str(getattr(params, "prominence_event_file_path", "") or "")
        label = str(getattr(params, "prominence_event_label", ALL_EVENT_LABELS) or ALL_EVENT_LABELS)
        try:
            table = read_event_table(path)
            info["file_format"] = table.format
            info["labels"] = table.labels()
            on, off = table.select(label)
            info["event_label"] = label or "all labels"
            if label and label not in table.events:
                info["warnings"].append(f"Label '{label}' is not in the event file.")
        except Exception as exc:  # unreadable file: say so, use the A/D events instead
            info["warnings"].append(f"Event file unusable ({exc}); A/D events used instead.")
            info["source"] = SOURCE_EVENTS
    info["n_events"] = int(np.isfinite(on).sum())

    inside = event_window_mask(t, on, off, before, after)
    finite_t = np.isfinite(t)
    if role == ROLE_INCLUDE:
        mask = inside
        if info["n_events"] == 0:
            info["warnings"].append("No events: 'inside event windows' selects nothing.")
        elif not np.any(mask):
            info["warnings"].append("Event windows contain no samples (check before/after and the time axis).")
    else:
        mask = finite_t & ~inside
        if info["n_events"] == 0:
            info["warnings"].append("No events found: the whole recording is the baseline.")
    finite_on = on[np.isfinite(on)]
    if finite_on.size and np.any(finite_t):
        t_lo, t_hi = float(np.nanmin(t)), float(np.nanmax(t))
        outside = np.sum((finite_on < t_lo) | (finite_on > t_hi))
        if outside == finite_on.size:
            info["warnings"].append(
                f"All {finite_on.size} events lie outside the recording ({t_lo:.1f}-{t_hi:.1f} s): "
                "event times must use the recording clock."
            )
    info["mask"] = mask
    return info


def mask_intervals(time: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Contiguous True runs of ``mask`` as (start_s, end_s) pairs."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool)
    if t.size != m.size or t.size == 0:
        return []
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False])
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------

def _sampling_rate(t: np.ndarray, fs_used: float) -> float:
    fs = float(fs_used) if fs_used is not None else float("nan")
    if not math.isfinite(fs) or fs <= 0:
        dt = np.diff(t[np.isfinite(t)])
        dt = dt[dt > 0]
        fs = 1.0 / float(np.median(dt)) if dt.size else float("nan")
    return fs


def _detect(t, d, min_prom, min_amp, min_dist):
    """Peaks of ``d`` (segment-aware) with prominence/height/spacing floors."""
    found = detect_peaks(
        t, d,
        prominence=max(0.0, float(min_prom)),
        min_height=float(min_amp) if (math.isfinite(min_amp) and min_amp > 0) else 0.0,
        min_distance_sec=max(0.0, float(min_dist)),
        auc_half_window_sec=0.0,
    )
    return np.asarray(found["indices"], int), np.asarray(found["prominences"], float)


def normalize_by_event_scale(
    signal: np.ndarray,
    time: np.ndarray,
    params: Any,
    fs_used: float,
    dio_onsets: Optional[np.ndarray] = None,
    dio_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Divide ``signal`` by the mean size of its top baseline events.

    Returns the normalized trace and a statistics dictionary that holds every
    number needed to reproduce or report the scale, plus overlay arrays in the
    output's units (``candidate_*`` for all detected events, ``selected_*``
    for the top fraction used in the mean).
    """
    y = np.asarray(signal, float)
    t = np.asarray(time, float)
    method = str(getattr(params, "prominence_method", METHOD_PROMINENCE) or METHOD_PROMINENCE).lower()
    if method not in (METHOD_PROMINENCE, METHOD_AMPLITUDE):
        method = METHOD_PROMINENCE
    units = str(getattr(params, "prominence_threshold_units", UNITS_ABSOLUTE) or UNITS_ABSOLUTE).lower()
    top_fraction = float(getattr(params, "prominence_percent_top", 0.10) or 0.0)
    top_fraction = min(1.0, max(0.0, top_fraction))
    min_prom_u = max(0.0, float(getattr(params, "prominence_min_peak", 0.0) or 0.0))
    max_size_u = float(getattr(params, "prominence_max_peak", 0.0) or 0.0)
    min_amp_u = max(0.0, float(getattr(params, "prominence_min_amplitude", 0.0) or 0.0))
    min_dist = max(0.0, float(getattr(params, "prominence_min_distance_s", 0.0) or 0.0))

    stats: Dict[str, Any] = {
        "method": method, "units": units, "top_fraction": top_fraction,
        "show_overlay": bool(getattr(params, "prominence_show_peaks_overlay", True)),
        "warnings": [], "scale": float("nan"), "scale_sem": float("nan"),
        "n_candidates": 0, "n_selected": 0, "center": 0.0,
    }
    empty = np.array([], float)
    for key in ("candidate_times", "candidate_peak", "candidate_base",
                "selected_times", "selected_peak", "selected_base", "selected_size"):
        stats[key] = empty
    if y.size == 0 or t.size != y.size:
        stats.update(scale_source="no_data", baseline_intervals=[], baseline_duration_s=0.0)
        return np.full_like(y, np.nan), _with_legacy_keys(stats, y)

    region = baseline_region(t, params, dio_onsets, dio_offsets)
    keep = np.asarray(region.pop("mask"), bool) & np.isfinite(t)
    stats["warnings"].extend(region.pop("warnings", []))
    stats["baseline"] = region
    stats["baseline_source"] = region.get("source", "")
    stats["baseline_intervals"] = mask_intervals(t, keep)
    finite_keep = keep & np.isfinite(y)
    fs = _sampling_rate(t, fs_used)
    duration = float(np.sum(finite_keep) / fs) if math.isfinite(fs) and fs > 0 else float("nan")
    stats["baseline_duration_s"] = duration
    stats["fs"] = fs
    if not np.any(finite_keep):
        stats["warnings"].append("The baseline region holds no valid sample; the whole recording is used.")
        keep = np.isfinite(t)
        finite_keep = keep & np.isfinite(y)
        stats["baseline_intervals"] = mask_intervals(t, keep)
        stats["baseline_duration_s"] = float(np.sum(finite_keep) / fs) if math.isfinite(fs) and fs > 0 else float("nan")

    # Detection trace: centred for the prominence recipe, raw dF/F for amplitude.
    median = float(np.nanmedian(y[finite_keep])) if np.any(finite_keep) else float(np.nanmedian(y))
    center = median if method == METHOD_PROMINENCE else 0.0
    stats["baseline_median"] = median
    stats["center"] = center
    d = y - center

    noise = estimate_noise(t, d, baseline_window_sec=NOISE_BASELINE_WINDOW_S, baseline_mask=finite_keep)
    sigma = float(noise.get("noise_sigma", float("nan")))
    stats["noise_sigma"] = sigma
    unit = 1.0
    if units == UNITS_NOISE:
        if math.isfinite(sigma) and sigma > 0:
            unit = sigma
        else:
            stats["warnings"].append("Baseline noise could not be measured; thresholds read as dF/F.")
            units = UNITS_ABSOLUTE
            stats["units"] = units
    min_prom, min_amp = min_prom_u * unit, min_amp_u * unit
    max_size = max_size_u * unit if max_size_u > 0 else float("inf")
    stats.update(min_prominence=min_prom, min_amplitude=min_amp, max_size=max_size, min_distance_s=min_dist)

    idx, prom = _detect(t, d, min_prom, min_amp, min_dist)
    inside = keep[idx] if idx.size else np.array([], bool)
    idx, prom = idx[inside], prom[inside]
    amp = d[idx]
    size = prom if method == METHOD_PROMINENCE else amp
    ok = np.isfinite(size) & (size > 0) & (size <= max_size)
    idx, prom, amp, size = idx[ok], prom[ok], amp[ok], size[ok]
    scale_source = f"top_{method}"

    if idx.size == 0 and (min_prom > 0 or min_amp > 0):
        idx2, prom2 = _detect(t, d, 0.0, 0.0, min_dist)
        inside2 = keep[idx2] if idx2.size else np.array([], bool)
        idx2, prom2 = idx2[inside2], prom2[inside2]
        amp2 = d[idx2]
        size2 = prom2 if method == METHOD_PROMINENCE else amp2
        ok2 = np.isfinite(size2) & (size2 > 0) & (size2 <= max_size)
        if np.any(ok2):
            idx, prom, amp, size = idx2[ok2], prom2[ok2], amp2[ok2], size2[ok2]
            scale_source = f"top_{method}_floor_ignored"
            stats["warnings"].append(
                "No event reached the minimum prominence/amplitude; the floor was ignored. "
                "Lower the threshold or check its units."
            )

    n_cand = int(idx.size)
    n_sel = int(math.ceil(n_cand * top_fraction)) if n_cand else 0
    order = np.argsort(-size, kind="stable")
    sel = np.sort(order[:n_sel])
    scale = float(np.mean(size[sel])) if n_sel else float("nan")
    sem = float(np.std(size[sel], ddof=1) / math.sqrt(n_sel)) if n_sel > 1 else (0.0 if n_sel == 1 else float("nan"))

    if not (math.isfinite(scale) and scale > 1e-12):
        # No event at all: a robust noise-based scale keeps the trace visible.
        base_arr = d[finite_keep] if np.any(finite_keep) else d[np.isfinite(d)]
        scale, sem = float("nan"), float("nan")
        if base_arr.size >= 5 and math.isfinite(sigma) and sigma > 1e-12:
            tail = np.abs(base_arr - np.nanmedian(base_arr))
            tail = tail[tail >= 2.0 * sigma]
            scale = float(np.mean(tail)) if tail.size else sigma
            scale_source = "noise_fallback"
            stats["warnings"].append("No baseline event detected: scaled by baseline noise instead.")

    if math.isfinite(scale) and scale > 1e-12:
        out = d / scale
    else:
        out = d.copy()
        scale_source = "centered_only_no_scale"
        stats["warnings"].append("No usable scale: the trace is only centred.")
        scale = float("nan")

    stats.update(
        scale=scale, scale_sem=sem, scale_source=scale_source,
        n_candidates=n_cand, n_selected=n_sel,
        candidate_rate_hz=(n_cand / duration) if duration and math.isfinite(duration) and duration > 0 else float("nan"),
        selected_rate_hz=(n_sel / duration) if duration and math.isfinite(duration) and duration > 0 else float("nan"),
    )
    if math.isfinite(scale) and scale > 0 and n_cand:
        peak_out = amp / scale
        base_out = (amp - prom) / scale if method == METHOD_PROMINENCE else np.zeros_like(amp)
        stats.update(
            candidate_times=t[idx].astype(float), candidate_peak=peak_out, candidate_base=base_out,
            selected_times=t[idx[sel]].astype(float), selected_peak=peak_out[sel],
            selected_base=base_out[sel], selected_size=size[sel].astype(float),
        )
        stats["_selected_index"] = idx[sel].astype(int)
    return out, _with_legacy_keys(stats, y)


def _with_legacy_keys(stats: Dict[str, Any], y: np.ndarray) -> Dict[str, Any]:
    """Keys read by older callers (output context, ProcessedTrial fields)."""
    idx = np.asarray(stats.pop("_selected_index", np.array([], int)), int)
    stats["mean_amplitude"] = stats.get("scale", float("nan"))
    stats["sem_amplitude"] = stats.get("scale_sem", float("nan"))
    stats["n_peaks"] = float(stats.get("n_selected", 0))
    stats["duration_s"] = stats.get("baseline_duration_s", float("nan"))
    stats["peak_times"] = np.asarray(stats.get("selected_times", np.array([], float)), float)
    stats["peak_values"] = y[idx].astype(float) if idx.size else np.array([], float)
    stats["effective_min_peak"] = float(stats.get("min_prominence", float("nan")))
    stats["mad_noise_sigma"] = float(stats.get("noise_sigma", float("nan")))
    stats.setdefault("baseline_intervals", [])
    stats.setdefault("baseline_source", "")
    return stats


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100.0 * x:.3g}%" if math.isfinite(x) else "n/a"


def describe_scale(stats: Optional[Dict[str, Any]]) -> Tuple[str, bool]:
    """Plain-language summary for the settings panel; second item flags warnings."""
    if not stats:
        return "", False
    method = stats.get("method", METHOD_PROMINENCE)
    measure = "prominence" if method == METHOD_PROMINENCE else "amplitude"
    lines: List[str] = []
    dur = float(stats.get("baseline_duration_s", float("nan")))
    region = stats.get("baseline") or {}
    src = {
        SOURCE_WINDOW: "time window", SOURCE_WHOLE: "whole recording",
        SOURCE_EVENTS: "A/D events", SOURCE_FILE: "event file",
    }.get(str(stats.get("baseline_source", "")), str(stats.get("baseline_source", "")))
    if region.get("source") in (SOURCE_EVENTS, SOURCE_FILE):
        role = "inside" if region.get("role") == ROLE_INCLUDE else "outside"
        label = region.get("event_label")
        label_txt = f" '{label}'" if label and label != "all labels" else ""
        src = f"{role} {region.get('n_events', 0)}{label_txt} {src.replace('A/D events', 'A/D event')}s"
        src = src.replace("event files", "file events")
    lines.append(f"Baseline: {dur:.0f} s ({src})" if math.isfinite(dur) else f"Baseline: {src}")
    n_c, n_s = int(stats.get("n_candidates", 0)), int(stats.get("n_selected", 0))
    r_c, r_s = float(stats.get("candidate_rate_hz", float("nan"))), float(stats.get("selected_rate_hz", float("nan")))
    rate = lambda r: f"{r:.3g} Hz" if math.isfinite(r) else "n/a"  # noqa: E731
    top = float(stats.get("top_fraction", 0.0))
    lines.append(f"Events: {n_c} ({rate(r_c)}); top {100 * top:.3g}% used: {n_s} ({rate(r_s)})")
    scale = float(stats.get("scale", float("nan")))
    sem = float(stats.get("scale_sem", float("nan")))
    if math.isfinite(scale):
        lines.append(f"Scale: mean {measure} {_pct(scale)} dF/F (SEM {_pct(sem)}); output 1 = {_pct(scale)} dF/F")
    sigma = float(stats.get("noise_sigma", float("nan")))
    if math.isfinite(sigma):
        lines.append(f"Baseline noise sigma: {_pct(sigma)} dF/F")
    warnings = list(stats.get("warnings") or [])
    lines.extend(f"Warning: {w}" for w in warnings)
    return "\n".join(lines), bool(warnings)


def context_text(stats: Optional[Dict[str, Any]]) -> str:
    """One-line description for the output plot title and exports."""
    if not stats:
        return ""
    method = stats.get("method", METHOD_PROMINENCE)
    name = "prominence" if method == METHOD_PROMINENCE else "SE amplitude"
    parts = [
        f"Scale [{name}, {stats.get('scale_source', '')}]: {_pct(float(stats.get('scale', float('nan'))))} dF/F",
        f"top {100 * float(stats.get('top_fraction', 0.0)):.3g}% of {int(stats.get('n_candidates', 0))} events "
        f"(n={int(stats.get('n_selected', 0))})",
        f"baseline {float(stats.get('baseline_duration_s', float('nan'))):.3g}s",
    ]
    if stats.get("warnings"):
        parts.append("WARNING: " + "; ".join(stats["warnings"]))
    return ", ".join(parts)


def dio_bouts(time: np.ndarray, dio: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """High runs of the A/D channel as (onsets, offsets)."""
    if dio is None:
        return np.array([], float), np.array([], float)
    return bouts_from_states(time, np.asarray(dio, float), 0.5)


__all__: Sequence[str] = (
    "EventTable", "read_event_table", "bouts_from_states", "event_window_mask", "baseline_region",
    "normalize_by_event_scale", "describe_scale", "context_text", "dio_bouts", "PAPER_PRESET",
)
