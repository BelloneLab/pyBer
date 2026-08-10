# analysis_core.py
from __future__ import annotations

import os
import re
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

from scipy.signal import butter, sosfiltfilt, resample_poly, savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d, median_filter, maximum_filter1d
from PySide6 import QtCore  # for QRunnable signals

# Optional: Lasso (requires scikit-learn). If unavailable, we fall back to OLS.
try:
    from sklearn.linear_model import Lasso
except Exception:
    Lasso = None

# Baseline correction (pybaselines)
try:
    from pybaselines.api import Baseline
except Exception:
    from pybaselines import Baseline

from sensor_registry import SENSOR_UNKNOWN, assess_sensor_trace, get_sensor


# =============================================================================
# Data models and user-facing option lists
# =============================================================================

BASELINE_METHODS = ["asls", "arpls", "airpls"]

# Fitting methods for "fitted reference" motion correction.
# - OLS: ordinary least squares
# - Lasso: sparse regression (requires sklearn)
# - RLM (HuberT): robust regression via IRLS with Huber weighting (no extra deps)
REFERENCE_FIT_METHODS = [
    "OLS (recommended)",
    "Lasso",
    "RLM (HuberT)",
]

SMOOTHING_METHODS = [
    "Savitzky-Golay",
    "Moving average",
    "Moving median",
]

ARTIFACT_HANDLING_MODES = [
    "Interpolate",
    "Cut",
    "Strong local low-pass",
    "Do nothing",
]

SMART_ARTIFACT_MODE = "Smart multi-evidence"
ARTIFACT_DETECTION_MODES = [
    SMART_ARTIFACT_MODE,
    "Adaptive MAD (windowed)",
    "Global MAD (raw)",
]

BAND_LIMITED_INVERTED_ISO_MODE = "dFF (motion corrected with band-limited inverted isobestic)"

# Output modes
OUTPUT_MODES = [
    # 1) dFF (non motion corrected)
    "dFF (non motion corrected)",
    # 2) zscore of dFF (non motion corrected)
    "zscore (non motion corrected)",
    # 3) dFF motion corrected via subtraction (dFF_sig - dFF_ref)
    "dFF (motion corrected via subtraction)",
    # 4) zscore of dFF motion corrected via subtraction
    "zscore (motion corrected via subtraction)",
    # 5) zscore(subtractions) = z(dFF_sig) - z(dFF_ref)
    "zscore (subtractions)",
    # 6) dFF motion corrected with fitted ref = (sig_f - fitted_ref) / fitted_ref
    "dFF (motion corrected with fitted ref)",
    # 7) dFF motion corrected after inverting the isobestic before fitting
    "dFF (motion corrected with inverted isobestic fit)",
    # 8) dFF corrected by fitting only the short-timescale inverted isobestic component
    BAND_LIMITED_INVERTED_ISO_MODE,
    # 9) zscore of that fitted-ref dFF
    "zscore (motion corrected with fitted ref)",
    # 10) prominence-normalized fitted-ref dFF
    "prominence normalized (motion corrected with fitted ref)",
    # 11) raw signal (processed 465 trace after artifact handling/filtering/resampling)
    "Raw signal (465)",
]


@dataclass
class ProcessingParams:
    # -------------------------
    # Artifact detection
    # -------------------------
    artifact_detection_enabled: bool = True
    artifact_mode: str = SMART_ARTIFACT_MODE
    artifact_handling: str = "Interpolate"
    mad_k: float = 8.0
    adaptive_window_s: float = 5.0
    artifact_pad_s: float = 0.25

    # -------------------------
    # Filtering
    # -------------------------
    lowpass_hz: float = 12.0
    filter_order: int = 3
    smoothing_enabled: bool = False
    smoothing_method: str = "Savitzky-Golay"
    smoothing_window_s: float = 0.200
    smoothing_polyorder: int = 2

    # -------------------------
    # Decimation / resampling
    # -------------------------
    target_fs_hz: float = 100.0

    # -------------------------
    # Baseline via pybaselines
    # -------------------------
    baseline_method: str = "airpls"  # asls | arpls | airpls
    baseline_lambda: float = 1e9
    baseline_diff_order: int = 2
    baseline_max_iter: int = 50
    baseline_tol: float = 1e-3
    asls_p: float = 0.01

    # -------------------------
    # Output selection
    # -------------------------
    # Default chosen to be explicit and widely used in photometry workflows.
    output_mode: str = "dFF (motion corrected with fitted ref)"

    # -------------------------
    # Signal polarity
    # -------------------------
    invert_polarity: bool = False
    sensor_id: str = SENSOR_UNKNOWN

    # -------------------------
    # Reference fit options (used by "fitted ref" output modes)
    # -------------------------
    reference_fit: str = "OLS (recommended)"  # OLS | Lasso | RLM (HuberT)

    # Lasso hyperparameter (only used if reference_fit == "Lasso")
    lasso_alpha: float = 1e-3

    # Window used by the band-limited inverted isobestic correction. The trace
    # is high-passed by subtracting a rolling median over this window before the
    # reference coupling is estimated.
    band_limited_reference_window_s: float = 60.0

    # Robust regression (Huber) hyperparameters (only used if reference_fit == "RLM (HuberT)")
    rlm_huber_t: float = 1.345  # classic Huber threshold (in sigma units)
    rlm_max_iter: int = 50
    rlm_tol: float = 1e-6

    # Prominence normalization options. The "baseline" is the segment used
    # to estimate the peak-prominence scale. Multiple sources are supported:
    #   - "events": exclude windows around DIO trigger rising edges
    #   - "window": use an explicit [start, end] interval (e.g. 5 min before task)
    #   - "file":   exclude windows around event times loaded from a CSV/XLSX
    #   - "whole":  the entire trace
    prominence_baseline_source: str = "events"
    prominence_baseline_start_s: float = 0.0
    prominence_baseline_end_s: float = 0.0  # 0 means "extend to end of trace"
    prominence_event_file_path: str = ""
    prominence_percent_top: float = 0.10
    prominence_exclude_before_s: float = 0.0
    prominence_exclude_after_s: float = 0.0
    prominence_min_peak: float = 0.0
    prominence_max_peak: float = 1e6
    prominence_show_peaks_overlay: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProcessingParams":
        p = ProcessingParams()
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


@dataclass
class SectionAdvice:
    """Advice for one settings panel, split into what to do and why.

    ``headline`` is a single plain-language instruction, ``settings`` are the
    concrete values to dial in (label, value), and ``why`` holds the evidence
    behind them, one self-contained sentence per entry.
    """
    headline: str = ""
    settings: List[Tuple[str, str]] = field(default_factory=list)
    why: List[str] = field(default_factory=list)

    def as_text(self) -> str:
        """Flatten to the one-line form kept in ``PreprocessingRecommendation.sections``."""
        parts = [self.headline.strip()] + [w.strip() for w in self.why]
        return " ".join(p for p in parts if p)


@dataclass
class PreprocessingRecommendation:
    """Data-driven preprocessing recommendation for one raw recording."""
    params: ProcessingParams
    confidence: float
    summary: str
    sections: Dict[str, str] = field(default_factory=dict)
    advice: Dict[str, SectionAdvice] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ArtifactDetectionResult:
    """Structured result from the smart artifact detector."""
    mask: np.ndarray
    core_mask: np.ndarray
    signal_core_mask: np.ndarray
    reference_core_mask: np.ndarray
    score: np.ndarray
    signal_score: np.ndarray
    reference_score: np.ndarray
    regions: List[Tuple[float, float]] = field(default_factory=list)
    core_regions: List[Tuple[float, float]] = field(default_factory=list)
    region_sources: List[str] = field(default_factory=list)
    region_scores: List[float] = field(default_factory=list)
    summary: str = ""


@dataclass
class LoadedTrial:
    path: str
    channel_id: str
    time: np.ndarray
    signal_465: np.ndarray
    reference_405: np.ndarray
    sampling_rate: float
    trigger_time: Optional[np.ndarray] = None
    trigger: Optional[np.ndarray] = None
    trigger_name: str = ""
    # Support for multiple triggers (DIO/AOUT)
    triggers: Dict[str, np.ndarray] = field(default_factory=dict)
    trigger_times: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class LoadedDoricFile:
    path: str
    channels: List[str]
    time_by_channel: Dict[str, np.ndarray]
    signal_by_channel: Dict[str, np.ndarray]
    reference_by_channel: Dict[str, np.ndarray]
    digital_time: Optional[np.ndarray]
    digital_by_name: Dict[str, np.ndarray]
    trigger_time_by_name: Dict[str, np.ndarray]
    trigger_by_name: Dict[str, np.ndarray]

    def make_trial(self, channel: str, trigger_name: Optional[str] = None, trigger_names: Optional[List[str]] = None) -> LoadedTrial:
        t = self.time_by_channel[channel]
        sig = self.signal_by_channel[channel]
        ref = self.reference_by_channel.get(channel)
        if ref is None or np.asarray(ref).size == 0:
            ref = np.full_like(np.asarray(sig, float), np.nan, dtype=float)
        else:
            ref = np.asarray(ref, float)

        trig_t = None
        trig = None
        trig_name = trigger_name or ""
        all_triggers = {}
        all_trigger_times = {}

        if trigger_name:
            if trigger_name in self.trigger_by_name:
                trig = np.asarray(self.trigger_by_name[trigger_name], float)
                trig_t = self.trigger_time_by_name.get(trigger_name, None)
                if trig_t is not None:
                    trig_t = np.asarray(trig_t, float)
                elif self.digital_time is not None and trigger_name in self.digital_by_name:
                    trig_t = np.asarray(self.digital_time, float)
                elif trig.size == t.size:
                    trig_t = np.asarray(t, float)

                if trig_t is not None and trig_t.size and trig.size:
                    n = min(trig_t.size, trig.size)
                    trig_t = trig_t[:n]
                    trig = trig[:n]

                # Align analog signals to selected trigger time base for overlays/event alignment.
                if trig_t is not None and trig_t.size and t.size and trig_t.size != t.size:
                    sig = np.interp(trig_t, t, sig)
                    if np.isfinite(ref).any():
                        ref = np.interp(trig_t, t, ref)
                    else:
                        ref = np.full_like(sig, np.nan, dtype=float)
                    t = trig_t
            elif self.digital_time is not None and trigger_name in self.digital_by_name:
                # Backward compatibility for sessions loaded before trigger map support.
                trig_t = self.digital_time
                trig = self.digital_by_name[trigger_name]

        # Multiple triggers (for export)
        names_to_collect = list(trigger_names) if trigger_names else []
        if trigger_name and trigger_name not in names_to_collect:
            names_to_collect.append(trigger_name)

        for name in names_to_collect:
            if name in self.trigger_by_name:
                val = np.asarray(self.trigger_by_name[name], float)
                vt = self.trigger_time_by_name.get(name, None)
                if vt is not None:
                    vt = np.asarray(vt, float)
                elif self.digital_time is not None and name in self.digital_by_name:
                    vt = np.asarray(self.digital_time, float)
                elif val.size == t.size:
                    vt = np.asarray(t, float)
                
                if vt is not None:
                    all_triggers[name] = val
                    all_trigger_times[name] = vt

        fs = 1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan

        return LoadedTrial(
            path=self.path,
            channel_id=channel,
            time=np.asarray(t, float),
            signal_465=np.asarray(sig, float),
            reference_405=np.asarray(ref, float),
            sampling_rate=float(fs) if np.isfinite(fs) else np.nan,
            trigger_time=np.asarray(trig_t, float) if trig_t is not None else None,
            trigger=np.asarray(trig, float) if trig is not None else None,
            trigger_name=trig_name,
            triggers=all_triggers,
            trigger_times=all_trigger_times,
        )


@dataclass
class ProcessedTrial:
    path: str
    channel_id: str

    time: np.ndarray
    raw_signal: np.ndarray
    raw_reference: np.ndarray

    # Threshold envelope on 465 (for display only)
    raw_thr_hi: Optional[np.ndarray] = None
    raw_thr_lo: Optional[np.ndarray] = None

    # Exact pre-processing raw arrays used for the top trace display. These are
    # kept separate from raw_signal/raw_reference because the latter are aligned
    # to the processed/export timebase for backward compatibility.
    raw_display_time: Optional[np.ndarray] = None
    raw_display_signal: Optional[np.ndarray] = None
    raw_display_reference: Optional[np.ndarray] = None
    raw_display_thr_hi: Optional[np.ndarray] = None
    raw_display_thr_lo: Optional[np.ndarray] = None
    raw_display_ref_thr_hi: Optional[np.ndarray] = None
    raw_display_ref_thr_lo: Optional[np.ndarray] = None
    raw_display_dio_time: Optional[np.ndarray] = None
    raw_display_dio: Optional[np.ndarray] = None

    # Optional analog/digital trigger channel aligned to processed time
    dio: Optional[np.ndarray] = None
    dio_name: str = ""
    # Support for multiple triggers
    triggers: Dict[str, np.ndarray] = field(default_factory=dict)

    # Processing intermediates
    sig_f: Optional[np.ndarray] = None
    ref_f: Optional[np.ndarray] = None
    baseline_sig: Optional[np.ndarray] = None
    baseline_ref: Optional[np.ndarray] = None

    # Final selected output
    output: Optional[np.ndarray] = None
    output_label: str = ""
    output_context: str = ""
    outputs: Dict[str, np.ndarray] = field(default_factory=dict)
    sensor_label: str = ""
    sensor_check: Dict[str, Any] = field(default_factory=dict)

    artifact_regions_sec: Optional[List[Tuple[float, float]]] = None
    artifact_regions_auto_sec: Optional[List[Tuple[float, float]]] = None
    artifact_regions_auto_core_sec: Optional[List[Tuple[float, float]]] = None
    artifact_regions_auto_source: Optional[List[str]] = None

    # Overlay data for the prominence-normalized output mode (None when the
    # active output mode does not use prominence normalization).
    prominence_peak_times: Optional[np.ndarray] = None
    prominence_peak_values: Optional[np.ndarray] = None
    prominence_baseline_intervals: Optional[List[Tuple[float, float]]] = None
    prominence_threshold: float = float("nan")
    prominence_baseline_source: str = ""

    # Optional synchronized timebase. When present, postprocessing can use this
    # camera/behavior-aligned time column instead of the original photometry time.
    sync_aligned_time: Optional[np.ndarray] = None
    sync_report: Dict[str, Any] = field(default_factory=dict)

    fs_actual: float = np.nan
    fs_target: float = np.nan
    fs_used: float = np.nan


@dataclass
class ExportSelection:
    raw: bool = True
    isobestic: bool = True
    output: bool = True
    dio: bool = True
    baseline_sig: bool = True
    baseline_ref: bool = True
    csv_metadata: bool = True
    output_modes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": bool(self.raw),
            "isobestic": bool(self.isobestic),
            "output": bool(self.output),
            "dio": bool(self.dio),
            "baseline_sig": bool(self.baseline_sig),
            "baseline_ref": bool(self.baseline_ref),
            "csv_metadata": bool(self.csv_metadata),
            "output_modes": list(self.output_modes or []),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "ExportSelection":
        if not isinstance(data, dict):
            return cls()
        modes = data.get("output_modes", [])
        if not isinstance(modes, list):
            modes = []
        return cls(
            raw=bool(data.get("raw", True)),
            isobestic=bool(data.get("isobestic", True)),
            output=bool(data.get("output", True)),
            dio=bool(data.get("dio", True)),
            baseline_sig=bool(data.get("baseline_sig", True)),
            baseline_ref=bool(data.get("baseline_ref", True)),
            csv_metadata=bool(data.get("csv_metadata", True)),
            output_modes=[str(m).strip() for m in modes if str(m or "").strip()],
        )


# =============================================================================
# Export helpers
# =============================================================================

def safe_stem_from_metadata(path: str, channel: str, meta: Dict[str, str]) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    a = (meta or {}).get("animal_id", "").strip()
    s = (meta or {}).get("session", "").strip()
    t = (meta or {}).get("trial", "").strip()

    def clean(x: str) -> str:
        x = re.sub(r"\s+", "_", x)
        x = re.sub(r"[^A-Za-z0-9_\-\.]", "", x)
        return x

    if a:
        parts = [clean(a)]
        if s:
            parts.append(clean(s))
        if t:
            parts.append(clean(t))
        parts.append(clean(channel))
        return "_".join(parts)
    return f"{clean(base)}_{clean(channel)}"


# =============================================================================
# Processed-trace export format  ("pyBer processed trace" v1.0)
# =============================================================================
#
# One gold-standard layout shared by the preprocessing CSV/HDF5 export and the
# postprocessing sync-aligned re-export. Design principles:
#
#   * Simple, stable headers for downstream scripts and Prism. The processed
#     output is written under its FAMILY name ("dFF" / "z-score" / "prominence"
#     / "signal_465"), never a generic "output" column. Structural columns are
#     fixed ("time", "time_aligned", "raw_465", "raw_405", "baseline_465",
#     "baseline_405"); triggers keep their real names ("DIO01", ...).
#   * The exact nature of every output (which motion-correction variant, the
#     reference fit, the baseline, and all processing parameters) lives in a
#     sidecar JSON file "<stem>.pyber.json". HDF5 also embeds that same JSON so
#     the file stays self-contained. CSV carries NO comment/metadata lines.
#   * "What you select is what you get": the previewed/primary output is always
#     present and is flagged primary in the sidecar; there is no duplicated or
#     unrequested column.
#
PYBER_FORMAT_VERSION = "1.0"
PYBER_FORMAT_KIND = "processed_trace"
PYBER_SIDECAR_SUFFIX = ".pyber.json"

# Fixed structural column / dataset names (identical across CSV and HDF5).
COL_TIME = "time"
COL_TIME_ALIGNED = "time_aligned"
COL_RAW_465 = "raw_465"
COL_RAW_405 = "raw_405"
COL_BASELINE_465 = "baseline_465"
COL_BASELINE_405 = "baseline_405"

_RESERVED_COLUMN_NAMES = {
    COL_TIME, COL_TIME_ALIGNED, COL_RAW_465, COL_RAW_405,
    COL_BASELINE_465, COL_BASELINE_405,
}

# Map each OUTPUT_MODES label to (family, variant). "family" is the bare column
# name used when the output is primary (or the first written of its family);
# "variant" disambiguates additional same-family outputs as "<family>__<variant>".
_OUTPUT_MODE_INFO: Dict[str, Tuple[str, str]] = {
    "dFF (non motion corrected)": ("dFF", "nomc"),
    "zscore (non motion corrected)": ("z-score", "nomc"),
    "dFF (motion corrected via subtraction)": ("dFF", "sub"),
    "zscore (motion corrected via subtraction)": ("z-score", "sub"),
    "zscore (subtractions)": ("z-score", "zdiff"),
    "dFF (motion corrected with fitted ref)": ("dFF", "fitref"),
    "dFF (motion corrected with inverted isobestic fit)": ("dFF", "invfitref"),
    BAND_LIMITED_INVERTED_ISO_MODE: ("dFF", "bandinvfitref"),
    "zscore (motion corrected with fitted ref)": ("z-score", "fitref"),
    "prominence normalized (motion corrected with fitted ref)": ("prominence", "fitref"),
    "Raw signal (465)": ("signal_465", "raw"),
}

_VARIANT_MOTION_CORRECTION = {
    "nomc": "none",
    "sub": "subtraction",
    "zdiff": "zscore_subtraction",
    "fitref": "fitted_ref",
    "invfitref": "inverted_fitted_ref",
    "bandinvfitref": "band_limited_inverted_fitted_ref",
    "raw": "none",
}


def output_label_type(label: str) -> str:
    """Short output label type. Kept for backward compatibility with old readers;
    new code should prefer output_family()."""
    lab = (label or "").strip().lower()
    if "zscore" in lab or "z-score" in lab or "z score" in lab:
        return "z-score"
    if "dff" in lab:
        return "dFF"
    if "raw signal" in lab or lab.startswith("raw"):
        return "raw_signal"
    if "prominence" in lab:
        return "prominence"
    return "output"


def output_family(label: str) -> str:
    """Family-level column name for an output mode (dFF / z-score / prominence / signal_465)."""
    info = _OUTPUT_MODE_INFO.get(str(label or "").strip())
    if info:
        return info[0]
    lab = (label or "").strip().lower()
    if "zscore" in lab or "z-score" in lab or "z score" in lab:
        return "z-score"
    if "prominence" in lab:
        return "prominence"
    if "dff" in lab:
        return "dFF"
    if "raw signal" in lab or "signal" in lab or lab.startswith("raw"):
        return "signal_465"
    return "output"


def output_variant_key(label: str) -> str:
    """Short variant tag distinguishing same-family output modes (fitref, sub, nomc, ...)."""
    info = _OUTPUT_MODE_INFO.get(str(label or "").strip())
    if info:
        return info[1]
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(label or "").strip()).strip("_").lower()
    return key or "output"


def output_units(label: str) -> str:
    """Interpretive units string for an output family."""
    return {
        "dFF": "dF/F",
        "z-score": "z-score (median/MAD)",
        "prominence": "prominence-normalized",
        "signal_465": "a.u. (processed 465)",
    }.get(output_family(label), "a.u.")


def _unique_export_name(name: str, used: set) -> str:
    base = str(name or "output").strip() or "output"
    out = base
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def assign_output_column_names(labels: List[str]) -> List[Tuple[str, str]]:
    """Assign a stable, unique column/dataset name to each output label (primary first).

    The first output of a family gets the bare family name (e.g. "dFF"); any
    further same-family outputs get "<family>__<variant>". Names never collide
    with each other or with reserved structural columns.
    """
    used: set = set(_RESERVED_COLUMN_NAMES)
    family_taken: set = set()
    assigned: List[Tuple[str, str]] = []
    for label in labels:
        fam = output_family(label)
        candidate = f"{fam}__{output_variant_key(label)}" if fam in family_taken else fam
        name = _unique_export_name(candidate, used)
        family_taken.add(fam)
        assigned.append((str(label), name))
    return assigned


def _output_items_for_export(
    processed: ProcessedTrial,
    selection: ExportSelection,
) -> List[Tuple[str, np.ndarray]]:
    """Return output traces in requested export order, falling back to the selected output."""
    source = getattr(processed, "outputs", None) or {}
    items: List[Tuple[str, np.ndarray]] = []

    if source:
        requested = [str(m).strip() for m in (selection.output_modes or []) if str(m or "").strip()]
        seen = set()
        for label in requested:
            if label in source and label not in seen:
                items.append((label, np.asarray(source[label], float)))
                seen.add(label)
        for label, values in source.items():
            if label not in seen:
                items.append((str(label), np.asarray(values, float)))
                seen.add(label)

    if not items and processed.output is not None:
        label = str(processed.output_label or "output")
        items.append((label, np.asarray(processed.output, float)))

    return items


def _pyber_now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def _json_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def plan_processed_columns(
    processed: ProcessedTrial,
    selection: ExportSelection,
) -> Tuple[List[Dict[str, Any]], str]:
    """Build the ordered column plan shared by the CSV and HDF5 writers.

    Returns (columns, primary_output_name) where each column is a dict with at
    least {name, role, values}. Roles: time, time_aligned, raw_signal,
    isosbestic, output, baseline_signal, baseline_reference, trigger.
    """
    t = np.asarray(processed.time, float)
    n = t.size

    def _fit(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        a = np.asarray(arr, float)
        return a if a.size == n else None

    columns: List[Dict[str, Any]] = []
    used: set = set()

    def _add(name: str, role: str, values: np.ndarray, **extra: Any) -> None:
        unique = _unique_export_name(name, used)
        entry = {"name": unique, "role": role, "values": np.asarray(values, float)}
        entry.update(extra)
        columns.append(entry)

    _add(COL_TIME, "time", t, units="s")
    for nm in (COL_TIME, COL_TIME_ALIGNED, COL_RAW_465, COL_RAW_405, COL_BASELINE_465, COL_BASELINE_405):
        used.add(nm)  # reserve structural names up front

    aligned = _fit(getattr(processed, "sync_aligned_time", None))
    if aligned is not None:
        columns.append({"name": COL_TIME_ALIGNED, "role": "time_aligned",
                        "values": aligned, "units": "s"})

    if selection.raw:
        raw = _fit(processed.raw_signal)
        columns.append({"name": COL_RAW_465, "role": "raw_signal",
                        "values": raw if raw is not None else np.full(n, np.nan)})
    if selection.isobestic:
        iso = _fit(processed.raw_reference)
        columns.append({"name": COL_RAW_405, "role": "isosbestic",
                        "values": iso if iso is not None else np.full(n, np.nan)})

    primary_output_name = ""
    if selection.output:
        output_items = _output_items_for_export(processed, selection)
        assigned = assign_output_column_names([lab for lab, _ in output_items])
        for (label, values), (_, colname) in zip(output_items, assigned):
            used.add(colname)
            vals = _fit(values)
            is_primary = not primary_output_name
            if is_primary:
                primary_output_name = colname
            columns.append({
                "name": colname, "role": "output",
                "values": vals if vals is not None else np.full(n, np.nan),
                "label": str(label), "family": output_family(label),
                "variant": output_variant_key(label), "units": output_units(label),
                "primary": is_primary,
            })

    if getattr(selection, "baseline_sig", False):
        b = _fit(processed.baseline_sig)
        if b is not None:
            columns.append({"name": COL_BASELINE_465, "role": "baseline_signal", "values": b})
    if getattr(selection, "baseline_ref", False):
        b = _fit(processed.baseline_ref)
        if b is not None:
            columns.append({"name": COL_BASELINE_405, "role": "baseline_reference", "values": b})

    if selection.dio:
        primary_dio = str(processed.dio_name or "").strip()
        dio = _fit(processed.dio)
        if dio is not None:
            _add(primary_dio or "dio", "trigger", dio, dio_name=(primary_dio or "dio"))
        for name, val in (getattr(processed, "triggers", None) or {}).items():
            if str(name) == primary_dio:
                continue
            v = _fit(val)
            if v is not None:
                _add(str(name), "trigger", v, dio_name=str(name))

    return columns, primary_output_name


def _split_subject(metadata: Optional[Dict[str, str]]) -> Dict[str, Any]:
    md = dict(metadata or {})
    subject: Dict[str, Any] = {}
    for key in ("animal_id", "session", "trial", "treatment"):
        if key in md:
            subject[key] = md.get(key, "")
    custom: Dict[str, str] = {}
    for k, v in md.items():
        if k in ("animal_id", "session", "trial", "treatment"):
            continue
        name = str(k)[len("custom:"):] if str(k).startswith("custom:") else str(k)
        custom[name] = v
    if custom:
        subject["custom"] = custom
    return subject


def build_processed_metadata(
    processed: ProcessedTrial,
    columns: List[Dict[str, Any]],
    primary_output_name: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
    params: Any = None,
    source_path: Optional[str] = None,
    channel: Optional[str] = None,
    created_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the sidecar / embedded metadata document for a processed export."""
    output_cols = [c for c in columns if c.get("role") == "output"]
    outputs: Dict[str, Any] = {}
    for c in output_cols:
        variant = str(c.get("variant", ""))
        entry = {
            "column": c["name"],
            "label": str(c.get("label", "")),
            "family": str(c.get("family", "")),
            "variant": variant,
            "units": str(c.get("units", "")),
            "primary": bool(c.get("primary")),
            "motion_correction": _VARIANT_MOTION_CORRECTION.get(variant, ""),
        }
        if variant in ("fitref", "invfitref", "bandinvfitref") and params is not None:
            entry["reference_fit"] = str(getattr(params, "reference_fit", "") or "")
            if variant == "bandinvfitref":
                entry["band_limited_reference_window_s"] = _json_float(
                    getattr(params, "band_limited_reference_window_s", np.nan)
                )
        outputs[c["name"]] = entry

    col_list: List[Dict[str, Any]] = []
    for c in columns:
        entry: Dict[str, Any] = {"name": c["name"], "role": c.get("role", "")}
        if c.get("units"):
            entry["units"] = c["units"]
        if c.get("dio_name"):
            entry["dio_name"] = c["dio_name"]
        if c.get("role") == "output":
            entry["primary"] = bool(c.get("primary"))
        col_list.append(entry)

    triggers = [str(c.get("dio_name") or c["name"]) for c in columns if c.get("role") == "trigger"]
    has_aligned = any(c["name"] == COL_TIME_ALIGNED for c in columns)

    if hasattr(params, "to_dict"):
        processing = params.to_dict()
    elif isinstance(params, dict):
        processing = dict(params)
    else:
        processing = {}

    sensor = get_sensor(str(processing.get("sensor_id", SENSOR_UNKNOWN)))
    sensor_check = getattr(processed, "sensor_check", {}) or {}
    sensor_meta = {
        "id": sensor.sensor_id,
        "name": sensor.name,
        "family": sensor.family,
        "target": sensor.target,
        "color": sensor.color,
        "direction": sensor.direction,
        "excitation_nm": sensor.excitation_nm,
        "emission_nm": sensor.emission_nm,
        "isobestic_nm": sensor.isobestic_nm,
        "rise": sensor.rise,
        "decay": sensor.decay,
        "affinity": sensor.affinity,
        "dynamic_range": sensor.dynamic_range,
        "recommended_fs_hz": _json_float(sensor.recommended_fs_hz),
        "recommended_lowpass_hz": _json_float(sensor.recommended_lowpass_hz),
        "notes": sensor.notes,
        "paper_url": sensor.paper_url,
        "source": sensor.source,
        "trace_check": sensor_check if isinstance(sensor_check, dict) else {},
    }

    return {
        "pyber_format": PYBER_FORMAT_KIND,
        "pyber_format_version": PYBER_FORMAT_VERSION,
        "created_utc": created_utc or _pyber_now_iso(),
        "source": {
            "file": os.path.basename(str(source_path or processed.path or "")),
            "path": str(source_path or processed.path or ""),
            "channel": str(channel or processed.channel_id or ""),
        },
        "subject": _split_subject(metadata),
        "sampling_rate_hz": {
            "actual": _json_float(processed.fs_actual),
            "target": _json_float(processed.fs_target),
            "used": _json_float(processed.fs_used),
        },
        "time": {
            "primary": COL_TIME,
            "aligned": COL_TIME_ALIGNED if has_aligned else None,
            "units": "s",
        },
        "primary_output": str(primary_output_name),
        "outputs": outputs,
        "columns": col_list,
        "triggers": triggers,
        "output_context": str(getattr(processed, "output_context", "") or ""),
        "processing": processing,
        "sensor": sensor_meta,
        "sync": (getattr(processed, "sync_report", {}) or {}),
    }


def sidecar_path_for(data_path: str) -> str:
    """Return the sidecar metadata path for a given CSV/HDF5 data path."""
    return os.path.splitext(str(data_path))[0] + PYBER_SIDECAR_SUFFIX


def write_processed_sidecar(data_path: str, meta: Dict[str, Any]) -> str:
    """Write the metadata document beside a data file as <stem>.pyber.json."""
    out = sidecar_path_for(data_path)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return out


def read_processed_sidecar(data_path: str) -> Optional[Dict[str, Any]]:
    """Read the sidecar metadata for a data file, if present."""
    side = sidecar_path_for(data_path)
    if not os.path.isfile(side):
        return None
    try:
        with open(side, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# =============================================================================
# Processed-trace import (single reader shared by both GUI tabs)
# =============================================================================

def _parse_processed_number(text: Any) -> float:
    """Parse a numeric cell, tolerating European decimals and HH:MM:SS times."""
    s = str(text or "").strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        pass
    try:
        return float(s.replace(" ", "").replace(",", "."))
    except Exception:
        pass
    return coerce_time_value(s)


def _looks_like_trigger_col(name: str) -> bool:
    key = str(name or "").strip().lower().replace(" ", "").replace("_", "")
    return (
        key == "dio" or key.startswith("dio") or key.startswith("ttl")
        or key.startswith("trigger") or "sync" in key or "barcode" in key
    )


def load_processed_csv(path: str) -> Optional[ProcessedTrial]:
    """Load a processed-trace CSV (pyBer v1.0 sidecar-aware, with legacy fallback)."""
    import csv
    try:
        with open(path, "r", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return None
    if not rows:
        return None

    try:
        sidecar = read_processed_sidecar(path)
    except Exception:
        sidecar = None
    if not isinstance(sidecar, dict):
        sidecar = None

    output_context = ""
    output_label = ""
    sensor_label = ""
    sensor_check: Dict[str, Any] = {}
    if sidecar is not None:
        output_context = str(sidecar.get("output_context", "") or "")
        sensor_meta = sidecar.get("sensor", {})
        if isinstance(sensor_meta, dict):
            sensor_label = str(sensor_meta.get("name", "") or "")
            check = sensor_meta.get("trace_check", {})
            sensor_check = check if isinstance(check, dict) else {}
    if not output_context:
        for r in rows:
            if r and str(r[0]).strip().lower().startswith("# output_context:"):
                output_context = str(r[0]).split(":", 1)[1].strip()
                break

    rows = [r for r in rows if r and any(str(cell).strip() for cell in r)]
    data_rows = [r for r in rows if not str(r[0]).lstrip().startswith("#")]
    if not data_rows:
        return None

    raw_header = [str(h).strip() for h in data_rows[0]]
    header = [h.lower() for h in raw_header]

    def _find_col(names: List[str]) -> Optional[int]:
        for name in names:
            key = str(name or "").strip().lower()
            if key and key in header:
                return header.index(key)
        return None

    sidecar_role_idx: Dict[int, str] = {}
    sidecar_trigger_names: Dict[int, str] = {}
    primary_output_name = ""
    time_name = ""
    aligned_name = ""
    if sidecar is not None:
        primary_output_name = str(sidecar.get("primary_output", "") or "")
        time_meta = sidecar.get("time") if isinstance(sidecar.get("time"), dict) else {}
        time_name = str((time_meta or {}).get("primary", "") or "")
        aligned_name = str((time_meta or {}).get("aligned", "") or "")
        for col in (sidecar.get("columns") or []):
            if not isinstance(col, dict):
                continue
            idx = _find_col([col.get("name", "")])
            if idx is None:
                continue
            role = str(col.get("role", ""))
            sidecar_role_idx[idx] = role
            if role == "trigger":
                sidecar_trigger_names[idx] = str(col.get("dio_name") or raw_header[idx])
        outs = sidecar.get("outputs") if isinstance(sidecar.get("outputs"), dict) else {}
        prim = (outs or {}).get(primary_output_name) if isinstance(outs, dict) else None
        if isinstance(prim, dict):
            output_label = str(prim.get("label", "") or "")

    def _role_index(role: str) -> Optional[int]:
        for idx, r in sidecar_role_idx.items():
            if r == role:
                return idx
        return None

    time_idx = _find_col([time_name]) if time_name else None
    if time_idx is None:
        time_idx = _role_index("time")
    if time_idx is None:
        time_idx = header.index("time") if "time" in header else None

    output_idx = _find_col([primary_output_name]) if primary_output_name else None
    if output_idx is None:
        output_idx = _role_index("output")
    if output_idx is None:
        output_idx = _find_col([
            "dff", "z-score", "zscore", "z score", "prominence", "signal_465",
            "output", "raw_signal",
            "raw_465", "raw", "isobestic", "raw_405",
            "reference", "reference_405", "ref", "dio",
            "baseline_465", "baseline_405",
        ])
    has_header = time_idx is not None and output_idx is not None

    raw_idx = _role_index("raw_signal") if sidecar is not None else None
    if raw_idx is None and has_header:
        raw_idx = _find_col(["raw_465", "raw", "signal", "signal_465"])
    iso_idx = _role_index("isosbestic") if sidecar is not None else None
    if iso_idx is None and has_header:
        iso_idx = _find_col(["raw_405", "isobestic", "isosbestic", "reference", "reference_405", "ref"])

    aligned_idx = _find_col([aligned_name]) if aligned_name else None
    if aligned_idx is None and has_header:
        aligned_idx = _find_col(["time_aligned", "aligned_time", "sync_aligned_time"])

    dio_idx = None
    trigger_cols: List[Tuple[int, str]] = []
    if has_header:
        if sidecar_trigger_names:
            for idx, name in sidecar_trigger_names.items():
                trigger_cols.append((idx, str(name or raw_header[idx])))
        else:
            seen: set = set()
            for idx, name in enumerate(raw_header):
                if idx == time_idx:
                    continue
                label = str(name or header[idx] or f"column_{idx + 1}").strip()
                if not _looks_like_trigger_col(label):
                    continue
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                trigger_cols.append((idx, label))
        dio_idx = _find_col(["dio"])
        if dio_idx is None and len(trigger_cols) == 1:
            dio_idx = trigger_cols[0][0]

    data_rows = data_rows[1:] if has_header else data_rows

    time: List[float] = []
    output: List[float] = []
    raw_vals: List[float] = []
    iso_vals: List[float] = []
    dio_vals: List[float] = []
    aligned_vals: List[float] = []
    trigger_vals: Dict[str, List[float]] = {name: [] for _, name in trigger_cols}

    for r in data_rows:
        if time_idx is None or output_idx is None:
            continue
        if len(r) <= max(time_idx, output_idx):
            continue
        try:
            tval = _parse_processed_number(r[time_idx])
            oval = float(r[output_idx])
        except Exception:
            continue
        if not np.isfinite(tval):
            continue
        time.append(tval)
        output.append(oval)
        if raw_idx is not None:
            try:
                raw_vals.append(float(r[raw_idx]) if len(r) > raw_idx else np.nan)
            except Exception:
                raw_vals.append(np.nan)
        if iso_idx is not None:
            try:
                iso_vals.append(float(r[iso_idx]) if len(r) > iso_idx else np.nan)
            except Exception:
                iso_vals.append(np.nan)
        if dio_idx is not None:
            try:
                dio_vals.append(float(r[dio_idx]) if len(r) > dio_idx else np.nan)
            except Exception:
                dio_vals.append(np.nan)
        if aligned_idx is not None:
            try:
                aligned_vals.append(float(r[aligned_idx]) if len(r) > aligned_idx else np.nan)
            except Exception:
                aligned_vals.append(np.nan)
        for trig_idx, trig_name in trigger_cols:
            try:
                trigger_vals[trig_name].append(float(r[trig_idx]) if len(r) > trig_idx else np.nan)
            except Exception:
                trigger_vals[trig_name].append(np.nan)

    if not time:
        return None

    t = np.asarray(time, float)
    out = np.asarray(output, float)
    raw = np.asarray(raw_vals, float) if raw_idx is not None and len(raw_vals) == len(time) else np.full_like(t, np.nan)
    iso = np.asarray(iso_vals, float) if iso_idx is not None and len(iso_vals) == len(time) else np.full_like(t, np.nan)
    dio_arr = np.asarray(dio_vals, float) if dio_idx is not None and len(dio_vals) == len(time) else None
    dio_name = ""
    if dio_arr is not None and dio_idx is not None and 0 <= dio_idx < len(raw_header):
        dio_name = str(raw_header[dio_idx] or "DIO").strip() or "DIO"
    triggers: Dict[str, np.ndarray] = {}
    for trig_name, vals in trigger_vals.items():
        arr = np.asarray(vals, float)
        if arr.size == t.size and int(np.sum(np.isfinite(arr))) >= 2:
            triggers[str(trig_name)] = arr
    if dio_arr is not None and int(np.sum(np.isfinite(dio_arr))) >= 2:
        triggers.setdefault(dio_name or "DIO", dio_arr)
    sync_aligned = (
        np.asarray(aligned_vals, float)
        if aligned_idx is not None and len(aligned_vals) == len(time)
        else None
    )

    if not output_label:
        output_label = "Imported CSV"
        if has_header and output_idx is not None:
            col = raw_header[output_idx] if output_idx < len(raw_header) else header[output_idx]
            col_l = str(col).strip().lower()
            if col and col_l != "output":
                pretty = {
                    "zscore": "z-score", "dff": "dFF",
                    "raw_signal": "Raw signal (465)", "signal_465": "Raw signal (465)",
                    "prominence": "Prominence normalized",
                }.get(col_l, str(col))
                output_label = f"Imported CSV ({pretty})"

    return ProcessedTrial(
        path=path,
        channel_id="import",
        time=t,
        raw_signal=raw,
        raw_reference=iso,
        dio=dio_arr,
        dio_name=dio_name,
        triggers=triggers,
        sig_f=None,
        ref_f=None,
        baseline_sig=None,
        baseline_ref=None,
        output=out,
        output_label=output_label,
        output_context=output_context,
        sensor_label=sensor_label,
        sensor_check=sensor_check,
        sync_aligned_time=sync_aligned,
        sync_report={"status": "imported", "method": "imported time_aligned"} if sync_aligned is not None else {},
        artifact_regions_sec=None,
        fs_actual=np.nan,
        fs_target=np.nan,
        fs_used=np.nan,
    )


def load_processed_h5(path: str) -> Optional[ProcessedTrial]:
    """Load a processed-trace HDF5 (pyBer v1.0 embedded/sidecar-aware, legacy fallback)."""
    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return None
            g = f["data"]
            if "time" not in g:
                return None
            gattrs = g.attrs
            t = np.asarray(g["time"][()], float)

            def _ds(name: Optional[str]) -> Optional[np.ndarray]:
                return np.asarray(g[name][()], float) if name and name in g else None

            meta: Optional[Dict[str, Any]] = None
            raw_meta = f.attrs.get("pyber_meta_json", "") if hasattr(f, "attrs") else ""
            if raw_meta:
                try:
                    parsed = json.loads(str(raw_meta))
                    meta = parsed if isinstance(parsed, dict) else None
                except Exception:
                    meta = None
            if meta is None:
                try:
                    side = read_processed_sidecar(path)
                    meta = side if isinstance(side, dict) else None
                except Exception:
                    meta = None

            roles: Dict[str, str] = {}
            trigger_meta: List[Tuple[str, str]] = []
            primary_output_name = ""
            if meta is not None:
                primary_output_name = str(meta.get("primary_output", "") or "")
                for col in (meta.get("columns") or []):
                    if isinstance(col, dict) and col.get("name"):
                        nm = str(col["name"])
                        role = str(col.get("role", ""))
                        roles[nm] = role
                        if role == "trigger":
                            trigger_meta.append((nm, str(col.get("dio_name") or nm)))

            out = None
            out_name = primary_output_name or str(gattrs.get("primary_output", "") or "")
            if out_name and out_name in g:
                out = _ds(out_name)
            if out is None:
                for nm, role in roles.items():
                    if role == "output" and nm in g:
                        out, out_name = _ds(nm), nm
                        break
            if out is None:
                for nm in ("output", "dFF", "z-score", "zscore", "prominence", "signal_465",
                           "raw_465", "raw", "raw_405", "isobestic", "dio",
                           "baseline_465", "baseline_405"):
                    if nm in g:
                        out, out_name = _ds(nm), nm
                        break
            if out is None:
                return None

            raw_sig = _ds("raw_465")
            if raw_sig is None:
                raw_sig = _ds("raw")
            if raw_sig is None:
                raw_sig = np.full_like(t, np.nan)
            raw_ref = _ds("raw_405")
            if raw_ref is None:
                raw_ref = _ds("isobestic")
            if raw_ref is None:
                raw_ref = np.full_like(t, np.nan)

            triggers: Dict[str, np.ndarray] = {}
            if trigger_meta:
                for ds_name, logical in trigger_meta:
                    arr = _ds(ds_name)
                    if arr is not None and arr.size == t.size:
                        triggers[str(logical)] = arr
            else:
                reserved = {"time", "time_aligned", "sync_aligned_time", "raw_465", "raw_405",
                            "raw", "isobestic", "baseline_465", "baseline_405", "output", out_name}
                for nm in g.keys():
                    if nm in reserved or not _looks_like_trigger_col(nm):
                        continue
                    arr = _ds(nm)
                    if arr is not None and arr.size == t.size:
                        triggers[str(nm)] = arr

            dio_name = str(gattrs.get("dio_name", "") or "")
            dio = _ds("dio")
            if dio is None and dio_name and dio_name in g:
                dio = _ds(dio_name)
            if dio is None and len(triggers) == 1:
                only_name, only_arr = next(iter(triggers.items()))
                dio = only_arr
                if not dio_name:
                    dio_name = only_name
            if dio is not None and dio.size == t.size:
                triggers.setdefault(dio_name or "DIO", dio)

            output_label = str(gattrs.get("output_label", "") or "")
            if not output_label and meta is not None:
                outs = meta.get("outputs") if isinstance(meta.get("outputs"), dict) else {}
                prim = (outs or {}).get(primary_output_name)
                if isinstance(prim, dict):
                    output_label = str(prim.get("label", "") or "")
            if not output_label:
                output_label = "Imported H5"
            output_context = str(gattrs.get("output_context", "") or "")
            sensor_label = ""
            sensor_check: Dict[str, Any] = {}
            if isinstance(meta, dict):
                sensor_meta = meta.get("sensor", {})
                if isinstance(sensor_meta, dict):
                    sensor_label = str(sensor_meta.get("name", "") or "")
                    check = sensor_meta.get("trace_check", {})
                    sensor_check = check if isinstance(check, dict) else {}
            fs_actual = float(gattrs.get("fs_actual", np.nan))
            fs_target = float(gattrs.get("fs_target", np.nan))
            fs_used = float(gattrs.get("fs_used", np.nan))

            sync_aligned = _ds("time_aligned")
            if sync_aligned is None:
                sync_aligned = _ds("sync_aligned_time")

            sync_report: Dict[str, Any] = {}
            raw_report = gattrs.get("sync_report_json", "")
            if raw_report:
                try:
                    parsed = json.loads(str(raw_report))
                    sync_report = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    sync_report = {}
    except Exception:
        return None

    return ProcessedTrial(
        path=path,
        channel_id="import",
        time=t,
        raw_signal=raw_sig,
        raw_reference=raw_ref,
        dio=dio,
        dio_name=dio_name,
        triggers=triggers,
        sig_f=None,
        ref_f=None,
        baseline_sig=None,
        baseline_ref=None,
        output=out,
        output_label=output_label,
        output_context=output_context,
        sensor_label=sensor_label,
        sensor_check=sensor_check,
        sync_aligned_time=sync_aligned,
        sync_report=sync_report,
        artifact_regions_sec=None,
        fs_actual=fs_actual,
        fs_target=fs_target,
        fs_used=fs_used,
    )


def export_processed_csv(
    path: str,
    processed: ProcessedTrial,
    metadata: Optional[Dict[str, str]] = None,
    selection: Optional[ExportSelection] = None,
    params: Any = None,
    created_utc: Optional[str] = None,
    write_sidecar: bool = True,
) -> None:
    import csv

    selection = selection if isinstance(selection, ExportSelection) else ExportSelection()
    columns, primary_output_name = plan_processed_columns(processed, selection)
    n = int(np.asarray(processed.time, float).size)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c["name"] for c in columns])
        for i in range(n):
            row = []
            for c in columns:
                v = float(c["values"][i])
                row.append(v if np.isfinite(v) else np.nan)
            w.writerow(row)

    if write_sidecar:
        meta = build_processed_metadata(
            processed, columns, primary_output_name,
            metadata=metadata, params=params, created_utc=created_utc,
        )
        write_processed_sidecar(path, meta)


def export_processed_h5(
    path: str,
    processed: ProcessedTrial,
    metadata: Optional[Dict[str, str]] = None,
    selection: Optional[ExportSelection] = None,
    params: Any = None,
    created_utc: Optional[str] = None,
    write_sidecar: bool = True,
) -> None:
    selection = selection if isinstance(selection, ExportSelection) else ExportSelection()
    columns, primary_output_name = plan_processed_columns(processed, selection)
    meta = build_processed_metadata(
        processed, columns, primary_output_name,
        metadata=metadata, params=params, created_utc=created_utc,
    )

    with h5py.File(path, "w") as f:
        f.attrs["pyber_format"] = PYBER_FORMAT_KIND
        f.attrs["pyber_format_version"] = PYBER_FORMAT_VERSION
        # Self-contained: embed the full sidecar document.
        try:
            f.attrs["pyber_meta_json"] = json.dumps(meta, default=str)
        except Exception:
            pass

        g = f.create_group("data")
        for c in columns:
            g.create_dataset(c["name"], data=np.asarray(c["values"], float), compression="gzip")
            if c.get("role") == "output":
                ds = g[c["name"]]
                ds.attrs["label"] = str(c.get("label", ""))
                ds.attrs["family"] = str(c.get("family", ""))
                ds.attrs["variant"] = str(c.get("variant", ""))
                ds.attrs["units"] = str(c.get("units", ""))
                ds.attrs["primary"] = bool(c.get("primary"))

        output_names = [c["name"] for c in columns if c.get("role") == "output"]
        trigger_names = [str(c.get("dio_name") or c["name"]) for c in columns if c.get("role") == "trigger"]
        g.attrs["primary_output"] = str(primary_output_name)
        g.attrs["output_columns"] = json.dumps(output_names)
        g.attrs["trigger_columns"] = json.dumps(trigger_names)
        prim = next((c for c in columns if c.get("role") == "output" and c.get("primary")), None)
        g.attrs["output_label"] = str(prim.get("label", "")) if prim else str(processed.output_label or "")
        g.attrs["output_context"] = str(getattr(processed, "output_context", "") or "")
        g.attrs["dio_name"] = str(processed.dio_name or "")
        g.attrs["fs_actual"] = float(processed.fs_actual)
        g.attrs["fs_used"] = float(processed.fs_used)
        g.attrs["fs_target"] = float(processed.fs_target)
        g.attrs["export_selection"] = json.dumps(selection.to_dict())
        if any(c["name"] == COL_TIME_ALIGNED for c in columns):
            try:
                g.attrs["sync_report_json"] = json.dumps(getattr(processed, "sync_report", {}) or {})
            except Exception:
                pass

        # Structured subject metadata group (flat key/value, for quick browsing).
        if metadata:
            mg = f.create_group("metadata")
            for k, v in metadata.items():
                mg.attrs[str(k)] = str(v)

    if write_sidecar:
        write_processed_sidecar(path, meta)


# =============================================================================
# Math helpers
# =============================================================================

def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation scaled to be comparable to std for normal data."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs (used after masking artifacts)."""
    y = np.asarray(y, float).copy()
    bad = ~np.isfinite(y)
    if not np.any(bad):
        return y
    good = np.where(~bad)[0]
    if good.size < 2:
        return y
    y[bad] = np.interp(np.where(bad)[0], good, y[good])
    return y


def regions_from_mask(time: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Convert a boolean mask into contiguous time regions (start, end) for display/export."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool)
    idx = np.where(m)[0]
    if idx.size == 0:
        return []
    regions: List[Tuple[float, float]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            regions.append((float(t[start]), float(t[prev])))
            start = i
            prev = i
    regions.append((float(t[start]), float(t[prev])))
    return regions


def _pad_mask_by_seconds(time: np.ndarray, mask: np.ndarray, pad_s: float) -> np.ndarray:
    """Pad a boolean time mask by pad_s seconds on both sides."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool).copy()
    if m.size == 0 or t.size < 2 or float(pad_s) <= 0:
        return m
    n = min(t.size, m.size)
    t = t[:n]
    m = m[:n]
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return np.asarray(mask, bool)
    pad_n = int(max(0, round(float(pad_s) / max(dt, 1e-12))))
    if pad_n <= 0:
        return np.asarray(mask, bool)
    idx = np.where(m)[0]
    padded = m.copy()
    for i in idx:
        a = max(0, i - pad_n)
        b = min(padded.size, i + pad_n + 1)
        padded[a:b] = True
    if padded.size != np.asarray(mask).size:
        out = np.zeros_like(np.asarray(mask, bool), dtype=bool)
        out[: min(out.size, padded.size)] = padded[: min(out.size, padded.size)]
        return out
    return padded


def apply_manual_regions(time: np.ndarray, mask: np.ndarray, regions: List[Tuple[float, float]]) -> np.ndarray:
    """OR a user-provided list of regions (sec) into the artifact mask."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool).copy()
    for (a, b) in (regions or []):
        t0, t1 = (min(a, b), max(a, b))
        m |= (t >= t0) & (t <= t1)
    return m


def remove_manual_regions(time: np.ndarray, mask: np.ndarray, regions: List[Tuple[float, float]]) -> np.ndarray:
    """Remove (unmask) user-provided regions (sec) from the artifact mask."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool).copy()
    for (a, b) in (regions or []):
        t0, t1 = (min(a, b), max(a, b))
        m &= ~((t >= t0) & (t <= t1))
    return m


def _lowpass_sos(x: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    """Zero-phase low-pass filtering with a Butterworth SOS filter."""
    if not np.isfinite(fs) or fs <= 0 or cutoff <= 0:
        return np.asarray(x, float)

    y = np.asarray(x, float)
    if np.any(~np.isfinite(y)):
        y = interpolate_nans(y)

    nyq = 0.5 * fs
    wn = min(0.999, max(1e-6, cutoff / nyq))
    sos = butter(order, wn, btype="low", output="sos")
    return np.asarray(sosfiltfilt(sos, y), float)


def _normalize_artifact_handling(value: object) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("cut"):
        return "Cut"
    if "low" in text and "pass" in text:
        return "Strong local low-pass"
    if text.startswith("do") or text in {"none", "nothing", "off", "ignore"}:
        return "Do nothing"
    return "Interpolate"


def _normalize_artifact_mode(value: object) -> str:
    text = str(value or "").strip()
    low = text.lower()
    if "smart" in low or "multi" in low:
        return SMART_ARTIFACT_MODE
    if low in {"adaptive_mad", "adaptive mad", "adaptive mad (windowed dx)"} or low.startswith("adaptive"):
        return "Adaptive MAD (windowed)"
    if low in {"global_mad", "global mad", "global mad (dx)"} or low.startswith("global"):
        return "Global MAD (raw)"
    return SMART_ARTIFACT_MODE


def _strong_local_lowpass_artifacts(
    x: np.ndarray,
    mask: np.ndarray,
    fs: float,
    base_cutoff_hz: float,
    filter_order: int,
) -> np.ndarray:
    y = np.asarray(x, float).copy()
    m = np.asarray(mask, bool)
    if y.size == 0 or m.size != y.size or not np.any(m):
        return y
    bridged = y.copy()
    bridged[m] = np.nan
    bridged = interpolate_nans(bridged)
    if np.any(~np.isfinite(bridged)):
        return y

    # Use a clearly stronger local cutoff than the main anti-aliasing filter.
    try:
        base_cutoff = float(base_cutoff_hz)
    except Exception:
        base_cutoff = 12.0
    if not np.isfinite(base_cutoff) or base_cutoff <= 0:
        base_cutoff = 12.0
    cutoff = min(2.0, max(0.05, 0.25 * base_cutoff))
    order = int(max(3, min(6, int(filter_order) + 1)))
    try:
        replacement = _lowpass_sos(bridged, fs, cutoff, order)
    except Exception:
        replacement = bridged
    y[m] = replacement[m]
    return y


def _apply_artifact_handling(
    sig: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray,
    fs: float,
    params: ProcessingParams,
) -> Tuple[np.ndarray, np.ndarray, str]:
    handling = _normalize_artifact_handling(getattr(params, "artifact_handling", "Interpolate"))
    sig_corr = np.asarray(sig, float).copy()
    ref_corr = np.asarray(ref, float).copy()
    m = np.asarray(mask, bool)
    if sig_corr.size == 0 or ref_corr.size == 0 or m.size != sig_corr.size or not np.any(m):
        return sig_corr, ref_corr, handling

    if handling == "Do nothing":
        return sig_corr, ref_corr, handling

    if handling == "Cut":
        # IMPORTANT: don't return the raw signal here. The downstream cut
        # step (after resampling) removes the artifact samples, but the
        # low-pass filter runs BEFORE that step - if the artifact is still
        # in the signal it gets smeared across neighbouring samples and
        # creates an even worse artefact at the cut boundary.
        # Fill the artifact region with interpolated values so the filter
        # sees a smooth trace; those samples are then dropped at the cut
        # step and never reach the output anyway.
        sig_corr[m] = np.nan
        ref_corr[m] = np.nan
        return interpolate_nans(sig_corr), interpolate_nans(ref_corr), handling

    if handling == "Strong local low-pass":
        cutoff = float(getattr(params, "lowpass_hz", 12.0))
        order = int(getattr(params, "filter_order", 3))
        return (
            _strong_local_lowpass_artifacts(sig_corr, m, fs, cutoff, order),
            _strong_local_lowpass_artifacts(ref_corr, m, fs, cutoff, order),
            handling,
        )

    sig_corr[m] = np.nan
    ref_corr[m] = np.nan
    return interpolate_nans(sig_corr), interpolate_nans(ref_corr), handling


def _window_samples_from_seconds(
    fs: float,
    window_s: float,
    *,
    minimum: int = 1,
    require_odd: bool = False,
) -> int:
    if not np.isfinite(fs) or fs <= 0:
        return int(max(1, minimum))
    n = int(round(float(window_s) * float(fs)))
    n = max(int(minimum), n)
    if require_odd and (n % 2 == 0):
        n += 1
    return int(n)


def _apply_optional_smoothing(x: np.ndarray, fs: float, params: ProcessingParams) -> np.ndarray:
    """
    Optional smoothing stage applied on the processed timebase.
    Supported methods:
    - Savitzky-Golay
    - Moving average
    - Moving median
    """
    y = np.asarray(x, float)
    if y.size < 3:
        return y
    if not bool(getattr(params, "smoothing_enabled", False)):
        return y

    method = str(getattr(params, "smoothing_method", "Savitzky-Golay") or "Savitzky-Golay").strip()
    window_s = float(getattr(params, "smoothing_window_s", 0.0))
    if not np.isfinite(window_s) or window_s <= 0:
        return y

    if np.any(~np.isfinite(y)):
        y = interpolate_nans(y)

    if method.startswith("Savitzky"):
        polyorder = int(max(1, getattr(params, "smoothing_polyorder", 2)))
        win = _window_samples_from_seconds(fs, window_s, minimum=polyorder + 2, require_odd=True)
        if win > y.size:
            win = y.size if (y.size % 2 == 1) else max(1, y.size - 1)
        if win <= polyorder:
            polyorder = max(1, min(polyorder, win - 1))
        if win < 3 or win <= polyorder:
            return y
        try:
            return np.asarray(savgol_filter(y, window_length=int(win), polyorder=int(polyorder), mode="interp"), float)
        except Exception:
            return y

    if method.startswith("Moving average"):
        win = _window_samples_from_seconds(fs, window_s, minimum=1, require_odd=False)
        if win <= 1:
            return y
        try:
            return np.asarray(uniform_filter1d(y, size=int(win), mode="nearest"), float)
        except Exception:
            return y

    if method.startswith("Moving median"):
        win = _window_samples_from_seconds(fs, window_s, minimum=3, require_odd=True)
        if win > y.size:
            win = y.size if (y.size % 2 == 1) else max(1, y.size - 1)
        if win <= 1:
            return y
        try:
            return np.asarray(median_filter(y, size=int(win), mode="nearest"), float)
        except Exception:
            return y

    return y


def _compute_resample_ratio(fs: float, target_fs: float) -> Tuple[int, int, float]:
    """Compute a rational resampling ratio (up/down) limited to manageable denominators."""
    from fractions import Fraction
    ratio = float(target_fs) / float(fs)
    frac = Fraction(ratio).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    fs_used = fs * up / down
    return int(up), int(down), float(fs_used)


def _resample_pair_to_target_fs(
    t: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    fs: float,
    target_fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Resample two signals together (preserving alignment) using polyphase filtering.
    We only resample if fs is substantially above target_fs (i.e., true decimation).
    """
    t = np.asarray(t, float)
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)

    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(target_fs) or target_fs <= 0:
        return t, x1, x2, fs

    # If already near/below target, do not resample (avoid unnecessary distortion).
    if fs <= target_fs * 1.05:
        return t, x1, x2, fs

    up, down, fs_used = _compute_resample_ratio(fs, target_fs)

    def _rp(x: np.ndarray) -> np.ndarray:
        # resample_poly signature differs across SciPy versions (padtype optional)
        try:
            return resample_poly(x, up, down, padtype="line")
        except TypeError:
            return resample_poly(x, up, down)

    y1 = _rp(x1)
    y2 = _rp(x2)

    n = min(y1.size, y2.size)
    y1 = y1[:n]
    y2 = y2[:n]

    dt_new = 1.0 / fs_used
    t_new = t[0] + np.arange(n, dtype=float) * dt_new

    return t_new, np.asarray(y1, float), np.asarray(y2, float), fs_used


def _compute_signal_envelope(
    t: np.ndarray,
    x: np.ndarray,
    k: float,
    mode: str,
    window_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a high/low "envelope" for display, based on median +/- k * MAD.
    In adaptive mode, median and MAD are computed in sliding windows.
    """
    y = np.asarray(x, float)
    if y.size == 0:
        return y.copy(), y.copy()
    finite_y = y[np.isfinite(y)]
    if finite_y.size == 0:
        empty = np.full_like(y, np.nan, dtype=float)
        return empty, empty

    if mode.startswith("Adaptive"):
        dt = float(np.nanmedian(np.diff(t))) if t.size > 2 else 1.0
        wN = int(max(25, round(window_s / max(dt, 1e-12))))
        stride = max(10, wN // 4)

        center = np.full_like(y, np.nan, dtype=float)
        spread = np.full_like(y, np.nan, dtype=float)
        for start in range(0, y.size, stride):
            end = min(y.size, start + wN)
            seg = y[start:end]
            seg = seg[np.isfinite(seg)]
            if seg.size < 5:
                continue
            center[start:end] = float(np.median(seg))
            spread[start:end] = float(k * _mad(seg))

        center = interpolate_nans(center)
        spread = interpolate_nans(spread)
        return center + spread, center - spread

    med = float(np.median(finite_y))
    sp = float(k * _mad(finite_y))
    hi = np.full_like(y, med + sp, dtype=float)
    lo = np.full_like(y, med - sp, dtype=float)
    return hi, lo


def _mask_outside_signal_envelope(t: np.ndarray, y: np.ndarray, hi: np.ndarray, lo: np.ndarray, pad_s: float) -> np.ndarray:
    """Return samples outside the displayed raw-signal MAD envelope, with optional time padding."""
    tt = np.asarray(t, float)
    yy = np.asarray(y, float)
    upper = np.asarray(hi, float)
    lower = np.asarray(lo, float)
    n = min(tt.size, yy.size, upper.size, lower.size)
    if n <= 0:
        return np.zeros_like(yy, dtype=bool)
    tt = tt[:n]
    yy = yy[:n]
    upper = upper[:n]
    lower = lower[:n]
    spread = 0.5 * np.abs(upper - lower)
    core = (
        np.isfinite(tt)
        & np.isfinite(yy)
        & np.isfinite(upper)
        & np.isfinite(lower)
        & (spread > 1e-12)
        & ((yy > upper) | (yy < lower))
    )
    mask = np.zeros_like(yy, dtype=bool)
    mask[:n] = core
    mask = _pad_mask_by_seconds(tt, mask, pad_s)
    if y.size != mask.size:
        out = np.zeros_like(y, dtype=bool)
        out[: min(out.size, mask.size)] = mask[: min(out.size, mask.size)]
        return out
    return mask


def detect_artifacts_global_dx(time: np.ndarray, x: np.ndarray, k: float, pad_s: float) -> np.ndarray:
    """
    Artifact detection via the same raw-signal MAD envelope shown in the GUI:
    - Compute median +/- k * MAD(signal)
    - Flag samples outside that visible envelope
    - Optionally pad around detections (pad_s)
    """
    t = np.asarray(time, float)
    y = np.asarray(x, float)
    hi, lo = _compute_signal_envelope(t, y, float(k), "Global MAD", 0.0)
    return _mask_outside_signal_envelope(t, y, hi, lo, pad_s)


def detect_artifacts_adaptive(time: np.ndarray, x: np.ndarray, k: float, window_s: float, pad_s: float) -> np.ndarray:
    """
    Artifact detection via the same adaptive raw-signal MAD envelope shown in the GUI:
    - For each window: median +/- k * MAD(signal_window)
    - Flag samples outside that visible envelope
    - Optionally pad around detections (pad_s)
    """
    t = np.asarray(time, float)
    y = np.asarray(x, float)
    if t.size < 3:
        return np.zeros_like(y, dtype=bool)
    hi, lo = _compute_signal_envelope(t, y, float(k), "Adaptive MAD", float(window_s))
    return _mask_outside_signal_envelope(t, y, hi, lo, pad_s)


def _mask_runs_at_least(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True runs with at least min_len samples."""
    m = np.asarray(mask, bool)
    out = np.zeros_like(m, dtype=bool)
    if m.size == 0:
        return out
    min_n = max(1, int(min_len))
    idx = np.where(m)[0]
    if idx.size == 0:
        return out
    start = int(idx[0])
    prev = int(idx[0])
    for raw_i in idx[1:]:
        i = int(raw_i)
        if i == prev + 1:
            prev = i
            continue
        if prev - start + 1 >= min_n:
            out[start:prev + 1] = True
        start = i
        prev = i
    if prev - start + 1 >= min_n:
        out[start:prev + 1] = True
    return out


def _expand_bool_mask_samples(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand True samples by a fixed sample radius."""
    m = np.asarray(mask, bool)
    if m.size == 0 or int(radius) <= 0:
        return m.copy()
    width = int(radius) * 2 + 1
    return uniform_filter1d(m.astype(float), size=width, mode="nearest") > 0.0


def _local_median_mad(
    y: np.ndarray,
    fs: float,
    window_s: float,
    *,
    min_window: int = 7,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rolling median and robust spread with floors to avoid brittle divisions."""
    arr = np.asarray(y, float)
    if arr.size == 0:
        return arr.copy(), arr.copy(), 0
    clean = interpolate_nans(arr)
    if np.any(~np.isfinite(clean)):
        fill = float(np.nanmedian(arr[np.isfinite(arr)])) if np.isfinite(arr).any() else 0.0
        clean = np.where(np.isfinite(clean), clean, fill)
    win = _window_samples_from_seconds(fs, window_s, minimum=min_window, require_odd=True)
    win = min(max(int(win), min_window), max(min_window, int(clean.size) // 2 * 2 + 1))
    if win >= clean.size:
        center = np.full_like(clean, float(np.nanmedian(clean)), dtype=float)
    else:
        center = np.asarray(median_filter(clean, size=win, mode="nearest"), float)
    dev = np.abs(clean - center)
    if win >= clean.size:
        local_mad = np.full_like(clean, _mad(dev), dtype=float)
    else:
        local_mad = np.asarray(median_filter(dev, size=win, mode="nearest"), float)
    spread = 1.4826 * local_mad
    finite_spread = spread[np.isfinite(spread) & (spread > 0)]
    global_floor = _mad(clean - center)
    if not np.isfinite(global_floor) or global_floor <= 0:
        global_floor = _mad(clean)
    if not np.isfinite(global_floor) or global_floor <= 0:
        global_floor = float(np.nanstd(clean))
    if not np.isfinite(global_floor) or global_floor <= 0:
        global_floor = 1.0
    local_floor = float(np.nanquantile(finite_spread, 0.20)) if finite_spread.size else global_floor
    floor = max(1e-12, 0.10 * float(global_floor), 0.25 * float(local_floor))
    spread = np.where(np.isfinite(spread) & (spread > floor), spread, floor)
    return center, spread, int(win)


def _smart_channel_artifact_features(
    time: np.ndarray,
    y: np.ndarray,
    fs: float,
    k: float,
    window_s: float,
    *,
    channel_label: str,
    protect_positive_transients: bool,
) -> Dict[str, np.ndarray]:
    """Compute robust channel-level artifact evidence masks and scores."""
    t = np.asarray(time, float)
    raw = np.asarray(y, float)
    n = min(t.size, raw.size)
    t = t[:n]
    raw = raw[:n]
    if n == 0:
        empty = np.array([], dtype=bool)
        return {
            "core": empty,
            "candidate": empty,
            "score": np.array([], dtype=float),
            "amp": empty,
            "amp_pos": empty,
            "amp_neg": empty,
            "slope": empty,
            "curve": empty,
            "level": empty,
            "plateau": empty,
            "dropout": empty,
            "nonfinite": empty,
            "label": np.array([], dtype=object),
        }

    finite = np.isfinite(raw)
    clean = interpolate_nans(raw)
    if np.any(~np.isfinite(clean)):
        fill = float(np.nanmedian(raw[finite])) if np.any(finite) else 0.0
        clean = np.where(np.isfinite(clean), clean, fill)

    win_s = float(window_s)
    if not np.isfinite(win_s) or win_s <= 0:
        win_s = 5.0
    win_s = float(_clip_float(win_s, max(0.25, 8.0 / max(float(fs), 1e-9)), 60.0))
    center, spread, _ = _local_median_mad(clean, fs, win_s, min_window=7)
    residual = clean - center
    amp_z = np.abs(residual) / spread

    step = np.empty_like(clean)
    step[0] = 0.0
    step[1:] = np.diff(clean)
    slope_window_s = float(_clip_float(min(win_s, 1.0), max(0.20, 5.0 / max(float(fs), 1e-9)), 5.0))
    slope_center, slope_spread, _ = _local_median_mad(step, fs, slope_window_s, min_window=5)
    slope_z = np.abs(step - slope_center) / slope_spread

    curve = np.empty_like(clean)
    curve[0] = 0.0
    curve[1:] = np.diff(step)
    curve_center, curve_spread, _ = _local_median_mad(curve, fs, slope_window_s, min_window=5)
    curve_z = np.abs(curve - curve_center) / curve_spread

    fast_center, _, _ = _local_median_mad(clean, fs, max(0.25, min(1.0, win_s * 0.25)), min_window=5)
    slow_center, slow_spread, _ = _local_median_mad(clean, fs, max(win_s, 2.0), min_window=9)
    level_z = np.abs(fast_center - slow_center) / slow_spread

    k_val = float(k)
    if not np.isfinite(k_val) or k_val <= 0:
        k_val = 8.0
    amp_thr = max(5.0, k_val)
    slope_thr = max(5.0, 0.75 * k_val)
    curve_thr = max(6.0, 0.85 * k_val)
    level_thr = max(4.0, 0.65 * k_val)

    edge_n = _window_samples_from_seconds(fs, 0.15, minimum=2)
    if edge_n > 0 and slope_z.size > 2 * edge_n:
        slope_z[:edge_n] = 0.0
        slope_z[-edge_n:] = 0.0
        curve_z[:edge_n] = 0.0
        curve_z[-edge_n:] = 0.0

    amp_pos = (residual > 0) & (amp_z >= amp_thr)
    amp_neg = (residual < 0) & (amp_z >= amp_thr)
    amp = amp_pos | amp_neg
    slope = slope_z >= slope_thr
    curve_mask = curve_z >= curve_thr
    if edge_n > 0 and slope.size > 2 * edge_n:
        slope[:edge_n] = False
        slope[-edge_n:] = False
        curve_mask[:edge_n] = False
        curve_mask[-edge_n:] = False
    level = _mask_runs_at_least(level_z >= level_thr, _window_samples_from_seconds(fs, 0.20, minimum=1))

    finite_clean = clean[np.isfinite(clean)]
    if finite_clean.size:
        q01 = float(np.nanquantile(finite_clean, 0.01))
        q99 = float(np.nanquantile(finite_clean, 0.99))
        dyn = max(float(q99 - q01), 1e-12)
    else:
        q01 = q99 = 0.0
        dyn = 1.0
    flat_step = np.abs(step) <= max(1e-12, 1e-5 * dyn)
    flat_run = _mask_runs_at_least(flat_step, _window_samples_from_seconds(fs, 0.25, minimum=3))
    plateau = flat_run & ((clean <= q01) | (clean >= q99) | (amp_z >= max(4.0, 0.70 * amp_thr)))
    dropout = finite & (clean <= q01) & (amp_z >= max(4.0, 0.70 * amp_thr))
    nonfinite = ~finite

    abrupt_with_amplitude = (slope | curve_mask) & (amp_z >= 0.60 * amp_thr)
    if protect_positive_transients:
        level_artifact = level & (amp_neg | abrupt_with_amplitude | dropout | plateau | nonfinite)
        channel_core = amp_neg | abrupt_with_amplitude | level_artifact | plateau | dropout | nonfinite
    else:
        level_artifact = level & ((amp_z >= 0.60 * amp_thr) | dropout | plateau | nonfinite)
        channel_core = amp | abrupt_with_amplitude | level_artifact | plateau | dropout | nonfinite

    candidate = (
        (amp_z >= 0.85 * amp_thr)
        | (slope_z >= 0.85 * slope_thr)
        | (curve_z >= 0.85 * curve_thr)
        | (level_z >= 0.90 * level_thr)
        | plateau
        | dropout
        | nonfinite
    )

    score = np.nanmax(
        np.vstack([
            amp_z / max(amp_thr, 1e-12),
            slope_z / max(slope_thr, 1e-12),
            curve_z / max(curve_thr, 1e-12),
            level_z / max(level_thr, 1e-12),
            plateau.astype(float),
            dropout.astype(float),
            nonfinite.astype(float),
        ]),
        axis=0,
    )
    score = np.where(np.isfinite(score), score, 0.0)
    labels = np.full(n, str(channel_label), dtype=object)
    return {
        "core": np.asarray(channel_core, bool),
        "candidate": np.asarray(candidate, bool),
        "score": np.asarray(score, float),
        "amp": np.asarray(amp, bool),
        "amp_pos": np.asarray(amp_pos, bool),
        "amp_neg": np.asarray(amp_neg, bool),
        "slope": np.asarray(slope, bool),
        "curve": np.asarray(curve_mask, bool),
        "level": np.asarray(level, bool),
        "plateau": np.asarray(plateau, bool),
        "dropout": np.asarray(dropout, bool),
        "nonfinite": np.asarray(nonfinite, bool),
        "label": labels,
    }


def _smart_region_sources(
    time: np.ndarray,
    regions: List[Tuple[float, float]],
    core_regions: List[Tuple[float, float]],
    sig_feat: Dict[str, np.ndarray],
    ref_feat: Dict[str, np.ndarray],
    shared_mask: np.ndarray,
    score: np.ndarray,
) -> Tuple[List[str], List[float]]:
    """Build compact human-readable evidence labels for artifact regions."""
    t = np.asarray(time, float)
    names = [
        ("amp", "amp"),
        ("slope", "slope"),
        ("curve", "curve"),
        ("level", "level"),
        ("plateau", "plateau"),
        ("dropout", "drop"),
        ("nonfinite", "nan"),
    ]
    sources: List[str] = []
    scores: List[float] = []
    for idx, (a, b) in enumerate(regions):
        if idx < len(core_regions):
            ca, cb = core_regions[idx]
        else:
            ca, cb = a, b
        in_core = (t >= float(ca)) & (t <= float(cb))
        if not np.any(in_core):
            in_core = (t >= float(a)) & (t <= float(b))
        parts: List[str] = []
        for prefix, feat in (("465", sig_feat), ("405", ref_feat)):
            tags = [tag for key, tag in names if np.any(np.asarray(feat.get(key, []), bool) & in_core)]
            if tags:
                parts.append(f"{prefix}:{'+'.join(tags[:3])}")
        if np.any(np.asarray(shared_mask, bool) & in_core):
            parts.append("shared")
        local_score = float(np.nanmax(score[in_core])) if np.any(in_core) else 0.0
        if np.isfinite(local_score):
            parts.append(f"Q={local_score:.2g}")
            scores.append(local_score)
        else:
            scores.append(0.0)
        sources.append("; ".join(parts) if parts else "smart")
    return sources, scores


def detect_artifacts_smart(
    time: np.ndarray,
    signal_465: np.ndarray,
    reference_405: Optional[np.ndarray] = None,
    *,
    k: float = 8.0,
    window_s: float = 5.0,
    pad_s: float = 0.25,
    fs: Optional[float] = None,
) -> ArtifactDetectionResult:
    """
    Detect fiber photometry artifacts using multi-evidence robust scoring.

    Evidence channels:
    - local Hampel-style amplitude residuals
    - derivative and curvature shocks
    - sustained level shifts
    - flat extreme plateaus and dropouts
    - temporally shared 405/465 evidence

    The 465 channel is protected against ordinary slow positive transients unless
    the evidence is abrupt, very large, or also appears in the reference channel.
    """
    t = np.asarray(time, float)
    sig = np.asarray(signal_465, float)
    n = min(t.size, sig.size)
    t = t[:n]
    sig = sig[:n]
    if reference_405 is None:
        ref = np.full(n, np.nan, dtype=float)
    else:
        ref_arr = np.asarray(reference_405, float)
        if ref_arr.size < n:
            ref = np.full(n, np.nan, dtype=float)
        else:
            ref = ref_arr[:n]
    if n == 0:
        empty_bool = np.array([], dtype=bool)
        empty_float = np.array([], dtype=float)
        return ArtifactDetectionResult(
            mask=empty_bool,
            core_mask=empty_bool,
            signal_core_mask=empty_bool,
            reference_core_mask=empty_bool,
            score=empty_float,
            signal_score=empty_float,
            reference_score=empty_float,
        )

    fs_val = float(fs) if fs is not None and np.isfinite(float(fs)) and float(fs) > 0 else np.nan
    if not np.isfinite(fs_val) or fs_val <= 0:
        fs_val = 1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else 10.0
    if not np.isfinite(fs_val) or fs_val <= 0:
        fs_val = 10.0

    finite_ref = ref[np.isfinite(ref)]
    has_reference = bool(finite_ref.size >= max(10, int(0.02 * n)) and np.nanstd(finite_ref) > 1e-12)
    sig_feat = _smart_channel_artifact_features(
        t,
        sig,
        fs_val,
        k,
        window_s,
        channel_label="465",
        protect_positive_transients=True,
    )
    if has_reference:
        ref_feat = _smart_channel_artifact_features(
            t,
            ref,
            fs_val,
            k,
            window_s,
            channel_label="405",
            protect_positive_transients=False,
        )
    else:
        ref_feat = {
            "core": np.zeros(n, dtype=bool),
            "candidate": np.zeros(n, dtype=bool),
            "score": np.zeros(n, dtype=float),
            "amp": np.zeros(n, dtype=bool),
            "amp_pos": np.zeros(n, dtype=bool),
            "amp_neg": np.zeros(n, dtype=bool),
            "slope": np.zeros(n, dtype=bool),
            "curve": np.zeros(n, dtype=bool),
            "level": np.zeros(n, dtype=bool),
            "plateau": np.zeros(n, dtype=bool),
            "dropout": np.zeros(n, dtype=bool),
            "nonfinite": np.zeros(n, dtype=bool),
        }

    near_radius = _window_samples_from_seconds(fs_val, 0.12, minimum=1)
    near_width = int(max(1, 2 * near_radius + 1))
    sig_score = np.asarray(sig_feat["score"], float)
    ref_score = np.asarray(ref_feat["score"], float)
    sig_near_ref = _expand_bool_mask_samples(np.asarray(sig_feat["candidate"], bool), near_radius)
    ref_near_sig = _expand_bool_mask_samples(np.asarray(ref_feat["candidate"], bool), near_radius)
    sig_score_near = maximum_filter1d(sig_score, size=near_width, mode="nearest") if sig_score.size else sig_score
    ref_score_near = maximum_filter1d(ref_score, size=near_width, mode="nearest") if ref_score.size else ref_score
    shared = (
        np.asarray(sig_feat["candidate"], bool)
        & ref_near_sig
        & (sig_score >= 0.90)
        & (ref_score_near >= 0.90)
        & ((sig_score >= 1.00) | (ref_score_near >= 1.00))
    ) | (
        np.asarray(ref_feat["candidate"], bool)
        & sig_near_ref
        & (ref_score >= 0.90)
        & (sig_score_near >= 0.90)
        & ((ref_score >= 1.00) | (sig_score_near >= 1.00))
    )
    sig_core = np.asarray(sig_feat["core"], bool) | shared
    ref_core = np.asarray(ref_feat["core"], bool) | shared
    core = sig_core | ref_core

    score = np.maximum(sig_score, ref_score)
    score = np.where(shared, np.maximum(score, 1.00), score)

    core_fraction = float(np.mean(core)) if core.size else 0.0
    if core_fraction > 0.20:
        stricter = core & (score >= 1.25)
        if np.any(stricter):
            core = stricter
            sig_core = sig_core & core
            ref_core = ref_core & core

    padded = _pad_mask_by_seconds(t, core, float(pad_s))
    regions = regions_from_mask(t, padded)
    core_regions: List[Tuple[float, float]] = []
    for a, b in regions:
        in_region = (t >= float(a)) & (t <= float(b)) & core
        if np.any(in_region):
            idx = np.where(in_region)[0]
            core_regions.append((float(t[idx[0]]), float(t[idx[-1]])))
        else:
            core_regions.append((float(a), float(b)))
    sources, region_scores = _smart_region_sources(t, regions, core_regions, sig_feat, ref_feat, shared, score)
    summary = (
        f"smart artifacts: n={len(regions)}, "
        f"core={float(np.mean(core)) * 100.0:.2f}%, padded={float(np.mean(padded)) * 100.0:.2f}%"
    )
    return ArtifactDetectionResult(
        mask=np.asarray(padded, bool),
        core_mask=np.asarray(core, bool),
        signal_core_mask=np.asarray(sig_core, bool),
        reference_core_mask=np.asarray(ref_core, bool),
        score=np.asarray(score, float),
        signal_score=np.asarray(sig_feat["score"], float),
        reference_score=np.asarray(ref_feat["score"], float),
        regions=regions,
        core_regions=core_regions,
        region_sources=sources,
        region_scores=region_scores,
        summary=summary,
    )


def _robust_corr(x: np.ndarray, y: np.ndarray) -> float:
    xx = np.asarray(x, float)
    yy = np.asarray(y, float)
    m = np.isfinite(xx) & np.isfinite(yy)
    if np.sum(m) < 10:
        return float("nan")
    if np.nanstd(xx[m]) <= 1e-12 or np.nanstd(yy[m]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xx[m], yy[m])[0, 1])


def _safe_quantile(x: np.ndarray, q: float, default: float = float("nan")) -> float:
    arr = np.asarray(x, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, float(q)))


def _clip_float(value: float, lo: float, hi: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        return float(lo)
    return float(max(lo, min(hi, v)))


def _rolling_correlation_summary(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    *,
    window_s: float = 30.0,
    step_s: float = 5.0,
) -> Dict[str, float]:
    tt = np.asarray(t, float)
    xx = np.asarray(x, float)
    yy = np.asarray(y, float)
    n = min(tt.size, xx.size, yy.size)
    if n < 20 or not np.isfinite(fs) or fs <= 0:
        return {
            "median": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
            "fraction_negative": 0.0,
            "fraction_positive": 0.0,
            "fraction_strong_negative": 0.0,
            "fraction_strong_positive": 0.0,
        }
    win = int(max(10, round(float(window_s) * float(fs))))
    step = int(max(1, round(float(step_s) * float(fs))))
    if win >= n:
        c = _robust_corr(xx[:n], yy[:n])
        vals = np.asarray([c], float) if np.isfinite(c) else np.asarray([], float)
    else:
        vals_list: List[float] = []
        for start in range(0, n - win + 1, step):
            c = _robust_corr(xx[start:start + win], yy[start:start + win])
            if np.isfinite(c):
                vals_list.append(float(c))
        vals = np.asarray(vals_list, float)
    if vals.size == 0:
        return {
            "median": float("nan"),
            "q10": float("nan"),
            "q90": float("nan"),
            "fraction_negative": 0.0,
            "fraction_positive": 0.0,
            "fraction_strong_negative": 0.0,
            "fraction_strong_positive": 0.0,
        }
    return {
        "median": float(np.nanmedian(vals)),
        "q10": float(np.nanquantile(vals, 0.10)),
        "q90": float(np.nanquantile(vals, 0.90)),
        "fraction_negative": float(np.mean(vals < 0.0)),
        "fraction_positive": float(np.mean(vals > 0.0)),
        "fraction_strong_negative": float(np.mean(vals < -0.30)),
        "fraction_strong_positive": float(np.mean(vals > 0.30)),
    }


def _format_metric(value: float, digits: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{int(digits)}g}"


def recommend_preprocessing_settings(
    trial: LoadedTrial,
    base_params: Optional[ProcessingParams] = None,
) -> PreprocessingRecommendation:
    """
    Recommend preprocessing settings from raw recording properties.

    The adviser is intentionally conservative: it proposes settings and explains
    them, but the GUI lets the user decide whether to apply them.
    """
    p = ProcessingParams.from_dict(base_params.to_dict()) if hasattr(base_params, "to_dict") else ProcessingParams()
    sensor = get_sensor(str(getattr(p, "sensor_id", SENSOR_UNKNOWN) or SENSOR_UNKNOWN))

    t = np.asarray(getattr(trial, "time", np.array([], float)), float)
    sig = np.asarray(getattr(trial, "signal_465", np.array([], float)), float)
    ref = np.asarray(getattr(trial, "reference_405", np.array([], float)), float)
    if ref.size == 0:
        ref = np.full_like(sig, np.nan, dtype=float)
    n = min(t.size, sig.size)
    if n < 20:
        p.output_mode = "dFF (non motion corrected)"
        sensor_adv = SectionAdvice(
            headline=f"{sensor.name}: select the sensor before interpreting this trace.",
            settings=[
                ("Target", sensor.target),
                ("Expected direction", sensor.direction),
                ("Isobestic", sensor.isobestic_nm),
                ("Paper", sensor.source),
            ],
            why=[
                "The recording is too short for a trace-shape check, but pyBer will still store the selected sensor in exports.",
                sensor.notes,
            ],
        )
        short_advice = {
            "sensor": sensor_adv,
            "artifacts": SectionAdvice(
                headline="Too few samples to judge the artifact burden.",
                why=["Load a longer recording and the adviser will measure it."],
            ),
            "filtering": SectionAdvice(
                headline="Too few samples to measure the sampling rate and noise.",
                why=["Load a longer recording and the adviser will measure it."],
            ),
            "baseline": SectionAdvice(
                headline="Keep the default baseline until a longer trace is loaded.",
                why=["Drift cannot be estimated from this many samples."],
            ),
            "output": SectionAdvice(
                headline="Use dFF (non motion corrected).",
                why=["The 405 reference cannot be evaluated on a trace this short."],
            ),
        }
        return PreprocessingRecommendation(
            params=p,
            confidence=0.0,
            summary="Recording is too short for automatic recommendations.",
            sections={key: adv.as_text() for key, adv in short_advice.items()},
            advice=short_advice,
            metrics={"n_samples": int(n), "sensor": {"id": sensor.sensor_id, "name": sensor.name}},
            warnings=["Too few samples for reliable automatic settings."],
        )
    t = t[:n]
    sig = sig[:n]
    if ref.size < n:
        ref = np.full(n, np.nan, dtype=float)
    else:
        ref = ref[:n]
    fs = float(getattr(trial, "sampling_rate", np.nan))
    if not np.isfinite(fs) or fs <= 0:
        fs = 1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
    if not np.isfinite(fs) or fs <= 0:
        fs = 10.0
    duration_s = float(np.nanmax(t) - np.nanmin(t)) if t.size else 0.0

    if np.any(~np.isfinite(sig)):
        sig = interpolate_nans(sig)
    if np.any(~np.isfinite(ref)):
        ref = interpolate_nans(ref)

    finite_ref = ref[np.isfinite(ref)]
    has_reference = bool(finite_ref.size >= 20 and np.nanstd(finite_ref) > 1e-12)

    hp_window_s = 60.0 if duration_s >= 180.0 else _clip_float(duration_s / 4.0, 10.0, 45.0)
    sig_hp, sig_trend, _ = _rolling_median_highpass(sig, fs, hp_window_s)
    if has_reference:
        ref_hp, ref_trend, _ = _rolling_median_highpass(ref, fs, hp_window_s)
    else:
        ref_hp = np.full_like(sig_hp, np.nan, dtype=float)
        ref_trend = np.full_like(sig_trend, np.nan, dtype=float)
    raw_corr = _robust_corr(ref, sig) if has_reference else float("nan")
    hp_corr = _robust_corr(ref_hp, sig_hp) if has_reference else float("nan")
    rolling = _rolling_correlation_summary(t, ref_hp, sig_hp, fs) if has_reference else {}

    sig_scale = max(_mad(sig_hp), 1e-12)
    ref_scale = max(_mad(ref_hp), 1e-12) if has_reference else 0.0
    sig_pos = _safe_quantile(sig_hp, 0.99, 0.0)
    sig_neg = -_safe_quantile(sig_hp, 0.01, 0.0)
    sig_median = float(np.nanmedian(sig[np.isfinite(sig)])) if np.isfinite(sig).any() else float("nan")
    sensor_check = assess_sensor_trace(sensor.sensor_id, t, sig)
    sensor_check_metrics = dict(sensor_check.get("metrics", {}) or {}) if isinstance(sensor_check, dict) else {}
    expected_down = sensor.direction.lower().startswith("decrease")
    sensor_pos_tail = float(sensor_check_metrics.get("positive_tail_z", sig_pos) or sig_pos)
    sensor_neg_tail = float(sensor_check_metrics.get("negative_tail_z", sig_neg) or sig_neg)
    sensor_direction_inverted = bool(
        sensor.sensor_id != SENSOR_UNKNOWN
        and (
            (expected_down and sensor_pos_tail > max(1.75 * sensor_neg_tail, 2.0))
            or ((not expected_down) and sensor_neg_tail > max(1.75 * sensor_pos_tail, 2.0))
        )
    )
    signal_inverted = bool(
        np.isfinite(sig_median)
        and sig_median < 0.0
        and sig_neg > max(1.25 * sig_pos, 3.0 * sig_scale)
    )
    p.invert_polarity = bool(signal_inverted or sensor_direction_inverted)

    # Artifact burden from the same smart detector used by preprocessing.
    smart_probe = detect_artifacts_smart(
        t,
        sig,
        ref if has_reference else None,
        k=8.0,
        window_s=float(_clip_float(max(5.0, min(30.0, duration_s / 30.0)), 5.0, 30.0)),
        pad_s=0.0,
        fs=fs,
    )
    mask_global = np.asarray(smart_probe.core_mask, bool)
    artifact_fraction = float(np.mean(mask_global)) if mask_global.size else 0.0
    drift_sig = float(np.nanstd(sig_trend) / max(np.nanstd(sig_hp), 1e-12))
    drift_ref = float(np.nanstd(ref_trend) / max(np.nanstd(ref_hp), 1e-12)) if has_reference else 0.0
    drift_score = max(drift_sig, drift_ref)

    p.artifact_detection_enabled = artifact_fraction > 0.0005
    p.artifact_mode = SMART_ARTIFACT_MODE
    p.mad_k = 7.0 if artifact_fraction < 0.02 else 8.5
    p.adaptive_window_s = float(_clip_float(max(5.0, min(30.0, duration_s / 30.0)), 5.0, 30.0))
    p.artifact_pad_s = float(_clip_float(max(0.25, 1.5 / fs), 0.10, 0.75))
    p.artifact_handling = "Interpolate" if artifact_fraction < 0.05 else "Strong local low-pass"

    if sensor.sensor_id != SENSOR_UNKNOWN:
        target_fs = min(float(fs), max(5.0, float(sensor.recommended_fs_hz)))
    else:
        target_fs = float(fs if fs <= 100.0 else 100.0)
    target_fs = float(max(1.0, round(target_fs, 1)))
    p.target_fs_hz = target_fs
    p.filter_order = 3
    lowpass_cap = float(sensor.recommended_lowpass_hz) if sensor.sensor_id != SENSOR_UNKNOWN else 12.0
    p.lowpass_hz = float(_clip_float(min(lowpass_cap, 0.40 * target_fs), 0.1, max(0.1, 0.45 * target_fs)))
    noise_ratio = float(_mad(np.diff(sig_hp)) / max(_mad(sig_hp), 1e-12)) if sig_hp.size > 5 else 0.0
    p.smoothing_enabled = bool(fs >= 20.0 and noise_ratio > 1.8)
    p.smoothing_method = "Savitzky-Golay"
    p.smoothing_window_s = float(_clip_float(0.20 if fs >= 20.0 else 1.0 / fs, 0.02, 0.50))
    p.smoothing_polyorder = 2

    p.baseline_method = "airpls"
    if n < 1500:
        lam_exp = 7
    elif n < 6000:
        lam_exp = 9
    else:
        lam_exp = 10 if fs > 100.0 else 9
    p.baseline_lambda = float(10.0 ** lam_exp)
    p.baseline_diff_order = 2
    p.baseline_max_iter = 50
    p.baseline_tol = 1e-3
    p.asls_p = 0.01

    frac_neg = float(rolling.get("fraction_negative", 0.0) or 0.0)
    frac_pos = float(rolling.get("fraction_positive", 0.0) or 0.0)
    strong_neg = float(rolling.get("fraction_strong_negative", 0.0) or 0.0)
    output_reason = ""
    if not has_reference:
        p.output_mode = "dFF (non motion corrected)"
        output_reason = (
            "There is no usable 405 reference in this file, so there is nothing to correct "
            "movement with."
        )
    elif np.isfinite(hp_corr) and hp_corr < -0.25 and frac_neg >= 0.60:
        mixed_drift = bool(np.isfinite(raw_corr) and raw_corr > 0.20 and abs(raw_corr - hp_corr) > 0.45)
        if mixed_drift:
            p.output_mode = BAND_LIMITED_INVERTED_ISO_MODE
            p.band_limited_reference_window_s = hp_window_s
            output_reason = (
                "The two channels bleach together but their fast fluctuations move in opposite "
                "directions, so only the band-limited inverted 405 component should be "
                "subtracted; the shared slow drift is left to the baseline."
            )
        else:
            p.output_mode = "dFF (motion corrected with inverted isobestic fit)"
            output_reason = (
                "Fast 405 and 465 fluctuations consistently move in opposite directions, so the "
                "isobestic has to be flipped before it is fitted and subtracted."
            )
    elif has_reference and ((np.isfinite(hp_corr) and hp_corr > 0.20 and frac_pos >= 0.50) or (np.isfinite(raw_corr) and raw_corr > 0.35)):
        p.output_mode = "dFF (motion corrected with fitted ref)"
        output_reason = (
            "The 405 channel follows the 465 channel with positive coupling, so it is a good "
            "movement estimate: fitting and subtracting it removes movement without removing "
            "your signal."
        )
    else:
        p.output_mode = "dFF (non motion corrected)"
        output_reason = (
            "The 405 reference is weak or inconsistent here, so forcing a correction would add "
            "noise instead of removing movement."
        )

    p.reference_fit = "RLM (HuberT)" if artifact_fraction > 0.02 else "OLS (recommended)"
    p.lasso_alpha = 1e-3
    p.rlm_huber_t = 1.345
    p.rlm_max_iter = 50
    p.rlm_tol = 1e-6

    conf_terms = [
        min(1.0, abs(hp_corr)) if np.isfinite(hp_corr) else 0.0,
        max(frac_neg, frac_pos),
        min(1.0, duration_s / 300.0),
        1.0 if has_reference else 0.25,
    ]
    confidence = float(_clip_float(0.15 + 0.25 * sum(conf_terms), 0.0, 1.0))
    if p.output_mode == "dFF (non motion corrected)" and has_reference:
        confidence = min(confidence, 0.55)

    if sensor_direction_inverted:
        expected_word = "downward" if expected_down else "upward"
        observed_word = "upward" if sensor_pos_tail > sensor_neg_tail else "downward"
        polarity_text = (
            f"{sensor.name} is expected to report {expected_word} biological events, but this "
            f"trace is dominated by {observed_word} excursions. The adviser switches full "
            f"465/405 polarity inversion on so the preview matches the selected sensor."
        )
    elif signal_inverted:
        polarity_text = (
            "Signal polarity looks electrically inverted (the trace sits below zero and its "
            "excursions point down), so switch the full 465/405 inversion on."
        )
    elif sensor.sensor_id != SENSOR_UNKNOWN:
        expected_word = "downward" if expected_down else "upward"
        polarity_text = (
            f"{sensor.name} expects {expected_word} biological events and the raw trace is "
            "broadly compatible with that direction, so full 465/405 inversion stays off."
        )
    else:
        polarity_text = (
            "Signal polarity looks normal (the trace sits above zero and its peaks point up), "
            "so keep the full 465/405 inversion off."
        )

    sensor_summary_prefix = f"{sensor.name}: " if sensor.sensor_id != SENSOR_UNKNOWN else ""
    summary = (
        f"{sensor_summary_prefix}Recommended: {p.output_mode}. "
        f"Fs {fs:.2f} Hz, raw corr={_format_metric(raw_corr)}, "
        f"detrended corr={_format_metric(hp_corr)}, confidence={confidence:.0%}."
    )

    # ---------------------------------------------------------------- #
    # Per-panel advice: one instruction, the values to dial in, and the
    # evidence behind them. The GUI renders this in the green
    # "Recommendations" frame at the bottom of each settings panel.
    # ---------------------------------------------------------------- #
    art_pct = artifact_fraction * 100.0
    if artifact_fraction <= 0.0005:
        art_headline = (f"This recording is clean - only {art_pct:.2f}% of samples look like "
                        f"artifact. Leave detection on as a safety net.")
    elif artifact_fraction < 0.05:
        art_headline = (f"About {art_pct:.2f}% of the samples look like artifact. Detect them and "
                        f"repair the gaps by interpolation.")
    else:
        art_headline = (f"About {art_pct:.2f}% of the samples look like artifact - too much to "
                        f"interpolate honestly. Smooth those stretches instead.")

    art_why = [
        "The smart detector only flags a sample when several kinds of evidence agree (a jump in "
        "amplitude, an impossible slope, a dropout, and the same event appearing in both 405 and "
        "465), so it catches hardware hits without eating real transients.",
        (f"{art_pct:.2f}% of samples are flagged. Interpolating is honest below about 5%; above "
         f"that it would invent more data than it repairs, which is why a local low-pass is the "
         f"gentler repair here."
         if artifact_fraction >= 0.05 else
         f"{art_pct:.2f}% of samples are flagged, well under the ~5% where interpolation starts "
         f"inventing more than it repairs."),
        f"The {p.adaptive_window_s:.0f} s window is roughly a thirtieth of the session, so the "
        f"threshold follows slow changes in noise instead of applying one global cutoff.",
        f"The {p.artifact_pad_s:.2g} s pad covers the filter ringing on each side of a hit, which "
        f"is otherwise left behind after the artifact itself is repaired.",
    ]

    filt_headline = (f"Resample to {p.target_fs_hz:.1f} Hz and low-pass at {p.lowpass_hz:.3g} Hz.")
    if sensor.sensor_id != SENSOR_UNKNOWN:
        bandwidth_text = (
            f"{sensor.name} reports {sensor.target} with {sensor.rise} rise and {sensor.decay} "
            f"decay. The recommendation therefore caps the target rate near "
            f"{sensor.recommended_fs_hz:.0f} Hz and the low-pass near "
            f"{sensor.recommended_lowpass_hz:.3g} Hz when the raw file allows it."
        )
    else:
        bandwidth_text = (
            f"{p.lowpass_hz:.3g} Hz sits below the {p.target_fs_hz / 2.0:.0f} Hz Nyquist limit "
            "of that target rate, so nothing aliases. Generic sensor transients pass through "
            "without assuming a specific reporter."
        )

    filt_why = [
        (f"The file is sampled at {fs:.2f} Hz. Coming down to {p.target_fs_hz:.1f} Hz keeps "
         f"everything the sensor can actually report and makes the rest of the pipeline faster."
         if fs > p.target_fs_hz + 0.5 else
         f"The file is sampled at {fs:.2f} Hz, so the native rate is kept."),
        bandwidth_text,
        (f"Sample-to-sample jitter is {noise_ratio:.1f}x the size of the trace's own fluctuations, "
         f"so a {p.smoothing_window_s:.2f} s Savitzky-Golay window is worth switching on."
         if p.smoothing_enabled else
         f"Sample-to-sample jitter is only {noise_ratio:.1f}x the trace's own fluctuations, so "
         f"extra smoothing would blur transients for no real gain - leave it off."),
        polarity_text,
    ]

    base_headline = (f"Use the {p.baseline_method} baseline with lambda = {p.baseline_lambda:.1e}.")
    base_why = [
        f"The session lasts {duration_s:.0f} s and its slow drift is {drift_score:.2g}x the size "
        f"of the fast fluctuations, so the baseline has to follow a large, slow curve.",
        f"Lambda sets how stiff that curve is. {p.baseline_lambda:.1e} is stiff enough to track "
        f"bleaching but too stiff to bend into a transient and subtract your signal away.",
        "airPLS re-weights toward the bottom of the trace on every pass, so upward transients pull "
        "the baseline much less than they would with a plain polynomial fit.",
    ]
    if n < 6000:
        base_why.append(
            f"Only {n} samples are available, so a softer lambda is used to avoid over-fitting a "
            f"short trace."
        )

    out_headline = f"Use {p.output_mode}."
    out_why = [output_reason]
    if has_reference:
        out_why.append(
            f"Across rolling 30 s windows the two channels correlate at "
            f"{_format_metric(rolling.get('median', float('nan')))}; {frac_neg * 100.0:.0f}% of "
            f"windows go negative and {strong_neg * 100.0:.0f}% go strongly negative, so the "
            f"coupling is consistent enough to trust."
        )
        out_why.append(
            f"{'A robust (Huber) fit is used because ' if p.reference_fit.startswith('RLM') else 'Ordinary least squares is enough because only '}"
            f"{art_pct:.2f}% of the samples are artifact"
            f"{'; robust fitting stops leftover spikes from tilting the fit.' if p.reference_fit.startswith('RLM') else '; above ~2% a robust fit would be the safer choice.'}"
        )
        if sensor.sensor_id != SENSOR_UNKNOWN and "not a standard 405" in sensor.isobestic_nm.lower():
            out_why.append(
                f"{sensor.name} is listed as {sensor.color} with '{sensor.isobestic_nm}' for the "
                "control wavelength. Treat any 405 correction as empirical unless your optical "
                "setup validated that wavelength for this reporter."
            )

    sensor_headline = (
        f"{sensor.name}: expected {sensor.direction} response for {sensor.target}."
        if sensor.sensor_id != SENSOR_UNKNOWN else
        "No sensor selected. pyBer is using generic photometry assumptions."
    )
    sensor_why = [
        sensor.notes,
        str(sensor_check.get("message", "Sensor check unavailable.") if isinstance(sensor_check, dict) else "Sensor check unavailable."),
        (
            f"Primary literature/source: {sensor.source}. The paper link is available in the "
            "Sensor table and is stored in export metadata."
        ),
    ]

    advice: Dict[str, SectionAdvice] = {
        "sensor": SectionAdvice(
            headline=sensor_headline,
            settings=[
                ("Family", sensor.family),
                ("Target", sensor.target),
                ("Direction", sensor.direction),
                ("Excitation", f"{sensor.excitation_nm} nm"),
                ("Isobestic", f"{sensor.isobestic_nm} nm"),
                ("Rise", sensor.rise),
                ("Decay", sensor.decay),
                ("Recommended Fs", f"{sensor.recommended_fs_hz:.0f} Hz"),
                ("Recommended LP", f"{sensor.recommended_lowpass_hz:.3g} Hz"),
            ],
            why=sensor_why,
        ),
        "artifacts": SectionAdvice(
            headline=art_headline,
            settings=[
                ("Detection", "on" if p.artifact_detection_enabled else "off"),
                ("Method", str(p.artifact_mode)),
                ("Handling", str(p.artifact_handling)),
                ("Sensitivity (k)", f"{p.mad_k:.1f}"),
                ("Window", f"{p.adaptive_window_s:.0f} s"),
                ("Pad", f"{p.artifact_pad_s:.2g} s"),
            ],
            why=art_why,
        ),
        "filtering": SectionAdvice(
            headline=filt_headline,
            settings=[
                ("Target Fs", f"{p.target_fs_hz:.1f} Hz"),
                ("Low-pass", f"{p.lowpass_hz:.3g} Hz"),
                ("Filter order", f"{p.filter_order}"),
                ("Smoothing", (f"{p.smoothing_method}, {p.smoothing_window_s:.2f} s"
                               if p.smoothing_enabled else "off")),
                ("Invert polarity", "on" if p.invert_polarity else "off"),
            ],
            why=filt_why,
        ),
        "baseline": SectionAdvice(
            headline=base_headline,
            settings=[
                ("Method", str(p.baseline_method)),
                ("Lambda", f"{p.baseline_lambda:.1e}"),
                ("diff_order", f"{p.baseline_diff_order}"),
                ("max_iter", f"{p.baseline_max_iter}"),
            ],
            why=base_why,
        ),
        "output": SectionAdvice(
            headline=out_headline,
            settings=(
                [("Output mode", str(p.output_mode)), ("Reference fit", str(p.reference_fit))]
                + ([("Band window", f"{p.band_limited_reference_window_s:.0f} s")]
                   if p.output_mode == BAND_LIMITED_INVERTED_ISO_MODE else [])
            ),
            why=out_why,
        ),
        "qc": SectionAdvice(
            headline="Check the result before you export a whole batch.",
            settings=[],
            why=[
                "Compare the recommended output against a plain non-motion dFF: if the transients "
                "survive the correction, the correction is doing its job.",
                "Look at an event-aligned average too - real responses stay put when the "
                "preprocessing changes, artifacts do not.",
            ],
        ),
    }
    sections = {key: adv.as_text() for key, adv in advice.items()}
    warnings: List[str] = []
    if has_reference and np.isfinite(raw_corr) and np.isfinite(hp_corr) and np.sign(raw_corr) != np.sign(hp_corr):
        warnings.append("Slow drift and fast reference coupling have opposite signs.")
    if isinstance(sensor_check, dict) and str(sensor_check.get("status", "")) == "warn":
        msg = str(sensor_check.get("message", "") or "").strip()
        if msg:
            warnings.append(msg)
    if confidence < 0.50:
        warnings.append("Recommendation confidence is low. Inspect the raw and corrected traces manually.")

    metrics: Dict[str, Any] = {
        "n_samples": int(n),
        "duration_s": duration_s,
        "fs_hz": float(fs),
        "has_reference": bool(has_reference),
        "raw_corr_405_465": raw_corr,
        "detrended_corr_405_465": hp_corr,
        "rolling_corr": dict(rolling),
        "artifact_fraction": artifact_fraction,
        "drift_score": drift_score,
        "signal_positive_q99": sig_pos,
        "signal_negative_q01_abs": sig_neg,
        "signal_median": sig_median,
        "signal_polarity_inverted": bool(signal_inverted),
        "reference_hp_mad": ref_scale,
        "signal_hp_mad": sig_scale,
        "noise_ratio": noise_ratio,
        "recommended_output": p.output_mode,
        "sensor": {
            "id": sensor.sensor_id,
            "name": sensor.name,
            "family": sensor.family,
            "target": sensor.target,
            "direction": sensor.direction,
            "trace_check": sensor_check if isinstance(sensor_check, dict) else {},
        },
    }
    return PreprocessingRecommendation(
        params=p,
        confidence=confidence,
        summary=summary,
        sections=sections,
        advice=advice,
        metrics=metrics,
        warnings=warnings,
    )


def zscore_median_std(x: np.ndarray) -> np.ndarray:
    """
    Z-score using median centering and standard deviation scaling.
    This is slightly more outlier-robust than mean-centering, while still using std.
    """
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.full_like(x, np.nan)
    return (x - med) / sd


def _trigger_rising_edges(time: np.ndarray, trigger: Optional[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Return event times from a thresholded trigger trace."""
    if trigger is None:
        return np.array([], float)
    t = np.asarray(time, float)
    y = np.asarray(trigger, float)
    if t.size < 2 or y.size != t.size:
        return np.array([], float)
    finite = np.isfinite(t) & np.isfinite(y)
    if np.sum(finite) < 2:
        return np.array([], float)
    tt = t[finite]
    yy = y[finite]
    high = yy > float(threshold)
    idx = np.where((~high[:-1]) & high[1:])[0] + 1
    if high.size and bool(high[0]):
        idx = np.concatenate(([0], idx))
    return np.asarray(tt[idx], float)


_HMS_RE = re.compile(r'^\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$')


def parse_hms_to_seconds(text: str) -> float:
    """Convert an 'HH:MM:SS.fraction' string to seconds (float).

    Returns np.nan if the string does not match the expected pattern.
    """
    m = _HMS_RE.match(str(text))
    if m is None:
        return np.nan
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def coerce_time_value(text: str) -> float:
    """Try to parse *text* as a numeric float; fall back to HH:MM:SS."""
    text = str(text or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return np.nan
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    return parse_hms_to_seconds(text)


def _baseline_mask_excluding_events(
    time: np.ndarray,
    event_times: np.ndarray,
    sec_before: float,
    sec_after: float,
) -> np.ndarray:
    """Build the MATLAB-style to_consider mask by dropping windows around events."""
    t = np.asarray(time, float)
    keep = np.isfinite(t)
    before = max(0.0, float(sec_before))
    after = max(0.0, float(sec_after))
    for event_t in np.asarray(event_times, float):
        if not np.isfinite(event_t):
            continue
        keep &= ~((t >= event_t - before) & (t <= event_t + after))
    return keep


def _load_event_times_csv(path: str) -> np.ndarray:
    """Load event timestamps (seconds) from a CSV / XLSX. Returns a flat sorted array."""
    if not path or not os.path.isfile(path):
        return np.array([], float)
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        elif ext == ".xlsx":
            import pandas as pd
            df = pd.read_excel(path, engine="openpyxl")
        else:
            return np.array([], float)
    except Exception:
        return np.array([], float)

    # Prefer a column that looks like an event-time column; fall back to the
    # first numeric column.
    name_hint = re.compile(r"(time|event|onset|start|sec)", re.IGNORECASE)
    chosen = None
    for col in df.columns:
        if name_hint.search(str(col)):
            chosen = col
            break
    if chosen is None:
        for col in df.columns:
            try:
                import pandas as pd
                arr = pd.to_numeric(df[col], errors="coerce")
                if np.isfinite(arr).any():
                    chosen = col
                    break
            except Exception:
                continue
    if chosen is None:
        return np.array([], float)
    try:
        import pandas as pd
        arr = pd.to_numeric(df[chosen], errors="coerce")
        out = np.asarray(arr, float)
        out = out[np.isfinite(out)]
        return np.sort(np.unique(out))
    except Exception:
        return np.array([], float)


def _baseline_intervals_from_mask(time: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """Convert a boolean baseline mask to a list of (start_s, end_s) intervals."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool)
    if t.size != m.size or t.size == 0:
        return []
    rising = np.where((~m[:-1]) & m[1:])[0] + 1
    falling = np.where(m[:-1] & (~m[1:]))[0]
    if m[0]:
        rising = np.concatenate([[0], rising])
    if m[-1]:
        falling = np.concatenate([falling, [m.size - 1]])
    out: List[Tuple[float, float]] = []
    for s, e in zip(rising, falling):
        if 0 <= s <= e < t.size:
            out.append((float(t[s]), float(t[e])))
    return out


# =============================================================================
# RWD CSV raw import
# =============================================================================

_RWD_REF_WAVELENGTHS = ("405", "410", "415")
_RWD_SIGNAL_WAVELENGTHS = ("465", "470", "475", "560", "565")


def _rwd_clean_csv_row(row: List[str]) -> List[str]:
    out = [str(cell or "").strip() for cell in row]
    while out and not out[-1]:
        out.pop()
    return out


def _rwd_read_csv_rows(path: str) -> List[List[str]]:
    """Read an RWD CSV with a few common Windows encodings."""
    import csv

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            with open(path, "r", newline="", encoding=encoding) as f:
                return [_rwd_clean_csv_row(row) for row in csv.reader(f)]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return []


def _rwd_norm(value: object) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _rwd_float(value: object) -> float:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return np.nan
    try:
        return float(text)
    except Exception:
        pass
    try:
        return float(text.replace(" ", "").replace(",", "."))
    except Exception:
        return coerce_time_value(text)


def _rwd_time_seconds(values: np.ndarray) -> np.ndarray:
    """RWD TimeStamp values are milliseconds in fluorescence and event CSVs."""
    return np.asarray(values, float) / 1000.0


def _rwd_find_table(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    cleaned = [r for r in (_rwd_clean_csv_row(row) for row in rows) if r and any(r)]
    for idx, row in enumerate(cleaned):
        norms = [_rwd_norm(cell) for cell in row]
        if "timestamp" in norms or "time" in norms:
            headers = [h.strip() or f"Column {i + 1}" for i, h in enumerate(row)]
            return headers, cleaned[idx + 1 :]
    return [], []


def _rwd_column_index(headers: List[str], *names: str) -> int:
    wanted = {_rwd_norm(name) for name in names}
    for idx, header in enumerate(headers):
        if _rwd_norm(header) in wanted:
            return idx
    return -1


def _rwd_parse_wavelength_column(header: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^\s*(.+?)[\s_\-]*([0-9]{3})\s*$", str(header or ""))
    if match is None:
        return None
    channel = match.group(1).strip(" _-") or "Channel"
    wave = match.group(2)
    return channel, wave


def _rwd_choose_ref_signal(waves: Dict[str, int]) -> Tuple[str, str]:
    ref = next((wave for wave in _RWD_REF_WAVELENGTHS if wave in waves), "")
    signal = next((wave for wave in _RWD_SIGNAL_WAVELENGTHS if wave in waves and wave != ref), "")
    if not signal:
        signal = next((wave for wave in waves.keys() if wave != ref), "")
    return ref, signal


def _rwd_unique_name(name: str, used: set) -> str:
    base = str(name or "Channel").strip() or "Channel"
    out = base
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def is_rwd_events_csv(path: str) -> bool:
    """Return True for sparse RWD `Events.csv` files."""
    try:
        if os.path.basename(str(path or "")).lower() != "events.csv":
            return False
        rows = _rwd_read_csv_rows(path)
        headers, _data = _rwd_find_table(rows)
        norms = {_rwd_norm(h) for h in headers}
        return {"timestamp", "name", "state"}.issubset(norms)
    except Exception:
        return False


def _rwd_find_events_csv(path: str) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    if not folder or not os.path.isdir(folder):
        return ""
    preferred = os.path.join(folder, "Events.csv")
    if is_rwd_events_csv(preferred):
        return preferred
    try:
        for name in os.listdir(folder):
            candidate = os.path.join(folder, name)
            if is_rwd_events_csv(candidate):
                return candidate
    except Exception:
        return ""
    return ""


def _rwd_event_traces(events_path: str, timebase: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert sparse RWD state changes into active-high trigger traces."""
    if not events_path or not os.path.isfile(events_path):
        return {}
    rows = _rwd_read_csv_rows(events_path)
    headers, data_rows = _rwd_find_table(rows)
    idx_time = _rwd_column_index(headers, "TimeStamp", "Time")
    idx_name = _rwd_column_index(headers, "Name", "Event", "Events")
    idx_state = _rwd_column_index(headers, "State", "Value")
    if min(idx_time, idx_name, idx_state) < 0:
        return {}

    grouped: Dict[str, List[Tuple[float, float]]] = {}
    for row in data_rows:
        if max(idx_time, idx_name, idx_state) >= len(row):
            continue
        name = str(row[idx_name] or "").strip()
        if not name:
            continue
        t_ms = _rwd_float(row[idx_time])
        state = _rwd_float(row[idx_state])
        if not np.isfinite(t_ms):
            continue
        if not np.isfinite(state):
            state = 1.0
        grouped.setdefault(name, []).append((float(t_ms) / 1000.0, float(state)))

    t = np.asarray(timebase, float)
    if t.size < 2:
        return {}

    out: Dict[str, np.ndarray] = {}
    used: set = set()
    for raw_name, transitions in grouped.items():
        transitions = sorted(transitions, key=lambda item: item[0])
        trace = np.zeros(t.size, dtype=float)
        cursor = 0
        current = 0.0
        active_state = bool(float(transitions[0][1]) > 0.5) if transitions else True
        for event_t, state in transitions:
            idx = int(np.searchsorted(t, float(event_t), side="left"))
            idx = max(0, min(idx, t.size))
            if idx > cursor:
                trace[cursor:idx] = current
            current = 1.0 if bool(float(state) > 0.5) == active_state else 0.0
            cursor = max(cursor, idx)
        if cursor < t.size:
            trace[cursor:] = current
        out[_rwd_unique_name(raw_name, used)] = trace
    return out


def _rwd_attach_events(path: str, loaded: LoadedDoricFile) -> LoadedDoricFile:
    if not loaded.channels:
        return loaded
    first_time = loaded.time_by_channel.get(loaded.channels[0], np.array([], float))
    events_path = _rwd_find_events_csv(path)
    triggers = _rwd_event_traces(events_path, first_time)
    if not triggers:
        return loaded
    trigger_time_by = dict(loaded.trigger_time_by_name or {})
    trigger_by = dict(loaded.trigger_by_name or {})
    digital_by = dict(loaded.digital_by_name or {})
    for name, values in triggers.items():
        trigger_by[name] = np.asarray(values, float)
        trigger_time_by[name] = np.asarray(first_time, float).copy()
        digital_by[name] = np.asarray(values, float)
    return LoadedDoricFile(
        path=loaded.path,
        channels=list(loaded.channels),
        time_by_channel=dict(loaded.time_by_channel),
        signal_by_channel=dict(loaded.signal_by_channel),
        reference_by_channel=dict(loaded.reference_by_channel),
        digital_time=np.asarray(first_time, float).copy(),
        digital_by_name=digital_by,
        trigger_time_by_name=trigger_time_by,
        trigger_by_name=trigger_by,
    )


def _rwd_load_aligned_csv(path: str, headers: List[str], data_rows: List[List[str]]) -> Optional[LoadedDoricFile]:
    idx_time = _rwd_column_index(headers, "TimeStamp", "Time")
    if idx_time < 0:
        return None

    groups: Dict[str, Dict[str, int]] = {}
    for idx, header in enumerate(headers):
        parsed = _rwd_parse_wavelength_column(header)
        if parsed is None:
            continue
        channel, wave = parsed
        groups.setdefault(channel, {})[wave] = idx
    if not groups:
        return None

    time_vals: List[float] = []
    by_col: Dict[int, List[float]] = {idx: [] for waves in groups.values() for idx in waves.values()}
    for row in data_rows:
        if idx_time >= len(row):
            continue
        tval = _rwd_float(row[idx_time])
        if not np.isfinite(tval):
            continue
        time_vals.append(tval)
        for col_idx in by_col:
            by_col[col_idx].append(_rwd_float(row[col_idx] if col_idx < len(row) else ""))
    if len(time_vals) < 2:
        return None

    t = _rwd_time_seconds(np.asarray(time_vals, float))
    order = np.argsort(t)
    t = t[order]
    finite_time = np.isfinite(t)
    t = t[finite_time]
    if t.size < 2:
        return None

    channels: List[str] = []
    time_by: Dict[str, np.ndarray] = {}
    signal_by: Dict[str, np.ndarray] = {}
    reference_by: Dict[str, np.ndarray] = {}
    used: set = set()
    for raw_channel, waves in groups.items():
        ref_wave, signal_wave = _rwd_choose_ref_signal(waves)
        if not ref_wave or not signal_wave:
            continue
        ref = np.asarray(by_col[waves[ref_wave]], float)[order][finite_time]
        signal = np.asarray(by_col[waves[signal_wave]], float)[order][finite_time]
        if signal.size != t.size or ref.size != t.size:
            continue
        if not np.isfinite(signal).any() or not np.isfinite(ref).any():
            continue
        channel = _rwd_unique_name(raw_channel, used)
        channels.append(channel)
        time_by[channel] = t.copy()
        signal_by[channel] = signal
        reference_by[channel] = ref

    if not channels:
        return None
    return _rwd_attach_events(
        path,
        LoadedDoricFile(
            path=path,
            channels=channels,
            time_by_channel=time_by,
            signal_by_channel=signal_by,
            reference_by_channel=reference_by,
            digital_time=None,
            digital_by_name={},
            trigger_time_by_name={},
            trigger_by_name={},
        ),
    )


def _rwd_load_unaligned_csv(path: str, headers: List[str], data_rows: List[List[str]]) -> Optional[LoadedDoricFile]:
    idx_time = _rwd_column_index(headers, "TimeStamp", "Time")
    idx_lights = _rwd_column_index(headers, "Lights", "Light", "Led")
    if min(idx_time, idx_lights) < 0:
        return None

    channel_cols = [
        idx for idx, header in enumerate(headers)
        if idx not in {idx_time, idx_lights} and str(header or "").strip()
    ]
    if not channel_cols:
        return None

    records: Dict[int, Dict[str, List[Tuple[float, float]]]] = {idx: {} for idx in channel_cols}
    for row in data_rows:
        if max(idx_time, idx_lights) >= len(row):
            continue
        t_ms = _rwd_float(row[idx_time])
        if not np.isfinite(t_ms):
            continue
        light_text = str(row[idx_lights] or "").strip()
        match = re.search(r"([0-9]{3})", light_text)
        if match is None:
            continue
        wave = match.group(1)
        for col_idx in channel_cols:
            value = _rwd_float(row[col_idx] if col_idx < len(row) else "")
            if np.isfinite(value):
                records[col_idx].setdefault(wave, []).append((float(t_ms), float(value)))

    channels: List[str] = []
    time_by: Dict[str, np.ndarray] = {}
    signal_by: Dict[str, np.ndarray] = {}
    reference_by: Dict[str, np.ndarray] = {}
    used: set = set()

    for col_idx in channel_cols:
        waves = records.get(col_idx, {})
        wave_cols = {wave: i for i, wave in enumerate(waves.keys())}
        ref_wave, signal_wave = _rwd_choose_ref_signal(wave_cols)
        if not ref_wave or not signal_wave:
            continue
        ref_pairs = np.asarray(waves.get(ref_wave, []), float)
        sig_pairs = np.asarray(waves.get(signal_wave, []), float)
        if ref_pairs.ndim != 2 or sig_pairs.ndim != 2 or ref_pairs.shape[0] < 2 or sig_pairs.shape[0] < 2:
            continue
        ref_order = np.argsort(ref_pairs[:, 0])
        sig_order = np.argsort(sig_pairs[:, 0])
        ref_t_ms = ref_pairs[ref_order, 0]
        ref_values = ref_pairs[ref_order, 1]
        sig_values = sig_pairs[sig_order, 1]

        n = min(ref_t_ms.size, sig_values.size)
        if n < 2:
            continue
        t = _rwd_time_seconds(ref_t_ms[:n])
        ref = ref_values[:n]
        signal = sig_values[:n]
        finite = np.isfinite(t) & np.isfinite(ref) & np.isfinite(signal)
        if int(np.sum(finite)) < 2:
            continue
        t = t[finite]
        signal = signal[finite]
        ref = ref[finite]
        channel = _rwd_unique_name(str(headers[col_idx] or f"Channel{col_idx + 1}"), used)
        channels.append(channel)
        time_by[channel] = t
        signal_by[channel] = signal
        reference_by[channel] = ref

    if not channels:
        return None
    return _rwd_attach_events(
        path,
        LoadedDoricFile(
            path=path,
            channels=channels,
            time_by_channel=time_by,
            signal_by_channel=signal_by,
            reference_by_channel=reference_by,
            digital_time=None,
            digital_by_name={},
            trigger_time_by_name={},
            trigger_by_name={},
        ),
    )


def load_rwd_csv(path: str) -> Optional[LoadedDoricFile]:
    """Load RWD fluorescence CSV exports as pyBer raw preprocessing inputs.

    Supported RWD styles:
    - aligned fluorescence tables with `CH1-410`, `CH1-470`, ...
    - unaligned alternating-light tables with `TimeStamp,Lights,Channel1,...`

    Sibling `Events.csv` files are attached as dense trigger traces on the
    fluorescence timebase, making them visible in preprocessing and selectable
    for postprocessing.
    """
    if not path or not os.path.isfile(path) or is_rwd_events_csv(path):
        return None
    try:
        rows = _rwd_read_csv_rows(path)
        headers, data_rows = _rwd_find_table(rows)
    except Exception:
        return None
    if not headers or not data_rows:
        return None

    if _rwd_column_index(headers, "Lights", "Light", "Led") >= 0:
        return _rwd_load_unaligned_csv(path, headers, data_rows)
    if any(_rwd_parse_wavelength_column(header) is not None for header in headers):
        return _rwd_load_aligned_csv(path, headers, data_rows)
    return None


def prominence_peaks_detection(
    temp: np.ndarray,
    percent_top: float,
    to_consider: np.ndarray,
    minpeak: float,
    maxpeak: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python equivalent of the MATLAB peaks_detection helper.

    Returns:
      top_prominences, top_indices, top_peak_values
    """
    y = np.asarray(temp, float)
    keep = np.asarray(to_consider, bool)
    if y.size == 0 or keep.size != y.size:
        return np.array([], float), np.array([], int), np.array([], float)

    finite = np.isfinite(y)
    if not np.any(finite):
        return np.array([], float), np.array([], int), np.array([], float)

    y_for_peaks = y.copy()
    nan_mask = ~finite
    if np.any(nan_mask) and not np.all(nan_mask):
        valid_idx = np.where(finite)[0]
        y_for_peaks[nan_mask] = np.interp(
            np.where(nan_mask)[0], valid_idx, y_for_peaks[valid_idx]
        )

    min_prom = max(0.0, float(minpeak))
    locs, props = find_peaks(y_for_peaks, prominence=min_prom)
    if locs.size == 0:
        return np.array([], float), np.array([], int), np.array([], float)
    proms = np.asarray(props.get("prominences", np.array([], float)), float)
    if proms.size != locs.size:
        return np.array([], float), np.array([], int), np.array([], float)

    max_prom = float(maxpeak)
    in_range = keep[locs] & finite[locs]
    if np.isfinite(max_prom):
        in_range &= proms <= max_prom
    locs_clean = locs[in_range]
    proms_clean = proms[in_range]
    peaks_clean = y[locs_clean]

    if proms_clean.size == 0:
        return np.array([], float), np.array([], int), np.array([], float)

    order = np.argsort(proms_clean)[::-1]
    fraction = min(1.0, max(0.0, float(percent_top)))
    num_peaks = int(np.ceil(proms_clean.size * fraction))
    if num_peaks <= 0:
        return np.array([], float), np.array([], int), np.array([], float)
    idx = order[:num_peaks]
    return proms_clean[idx], locs_clean[idx].astype(int), peaks_clean[idx]


def prominence_normalize(
    signal: np.ndarray,
    time: np.ndarray,
    event_times: np.ndarray,
    params: ProcessingParams,
    fs_used: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize a trace by baseline peak prominence.

    This mirrors the supplied MATLAB workflow:
    1) create a to_consider mask by excluding samples around events;
    2) detect peaks using MinPeakProminence, keep peaks in to_consider, drop
       peaks above maxpeak, and average the top percentage of prominences.

    The returned trace is centered by the baseline median and divided by the
    mean selected prominence, making it z-score-like but peak-scale based.
    """
    y = np.asarray(signal, float)
    t = np.asarray(time, float)
    if y.size == 0 or t.size != y.size:
        return np.full_like(y, np.nan), {
            "duration_s": 0.0,
            "mean_amplitude": np.nan,
            "sem_amplitude": np.nan,
            "n_peaks": 0.0,
            "baseline_median": np.nan,
        }

    source = str(getattr(params, "prominence_baseline_source", "events") or "events").lower()
    before = float(getattr(params, "prominence_exclude_before_s", 0.0))
    after = float(getattr(params, "prominence_exclude_after_s", 0.0))
    if source == "window":
        # Explicit baseline window [start, end]; end=0 means "to end of trace".
        start_s = float(getattr(params, "prominence_baseline_start_s", 0.0))
        end_s = float(getattr(params, "prominence_baseline_end_s", 0.0))
        finite_t = np.isfinite(t)
        if not np.any(finite_t):
            keep = np.zeros_like(t, dtype=bool)
        else:
            t_lo = float(np.nanmin(t[finite_t]))
            abs_start = t_lo + max(0.0, start_s)
            abs_end = t_lo + max(0.0, end_s) if end_s > 0 else float(np.nanmax(t[finite_t]))
            if abs_end <= abs_start:
                # User configured an empty window - fall back to whole trace.
                keep = finite_t.copy()
            else:
                keep = finite_t & (t >= abs_start) & (t <= abs_end)
    elif source == "file":
        file_events = _load_event_times_csv(str(getattr(params, "prominence_event_file_path", "") or ""))
        # If the file produced no events, fall back to whatever event_times the
        # caller supplied (typically DIO rising edges) so we never silently use
        # the whole trace.
        if file_events.size == 0:
            file_events = np.asarray(event_times, float)
        keep = _baseline_mask_excluding_events(t, file_events, before, after)
    elif source == "whole":
        keep = np.isfinite(t)
    else:  # "events" (legacy DIO-driven)
        keep = _baseline_mask_excluding_events(t, np.asarray(event_times, float), before, after)
    finite_keep = keep & np.isfinite(y)
    baseline_median = float(np.nanmedian(y[finite_keep])) if np.any(finite_keep) else float(np.nanmedian(y))

    # Center by baseline median BEFORE peak detection so prominences reflect
    # transient amplitude above baseline (matches MATLAB data_norm.data input).
    y_centered = y - baseline_median

    min_peak = float(getattr(params, "prominence_min_peak", 0.0))
    max_peak = float(getattr(params, "prominence_max_peak", 1e6))
    top_fraction = float(getattr(params, "prominence_percent_top", 0.10))

    amplitudes, idx_peak, _top_peaks = prominence_peaks_detection(
        y_centered,
        top_fraction,
        keep,
        min_peak,
        max_peak,
    )
    scale_source = "baseline_peak_prominence"
    n_peaks = int(amplitudes.size)
    effective_min_peak = float(min_peak)

    # Fallback 1: retry without the user's min-prominence floor in case the
    # configured threshold is too aggressive for this trace.
    if n_peaks == 0 and min_peak > 0.0:
        amplitudes_retry, idx_peak_retry, _t2 = prominence_peaks_detection(
            y_centered, top_fraction, keep, 0.0, max_peak,
        )
        if amplitudes_retry.size:
            amplitudes = amplitudes_retry
            idx_peak = idx_peak_retry
            n_peaks = int(amplitudes.size)
            scale_source = "baseline_peak_prominence_no_min_floor"
            effective_min_peak = 0.0

    mean_amp = float(np.nanmean(amplitudes)) if n_peaks else np.nan

    # Fallback 2: still nothing? Use a MAD-noise estimate of the baseline.
    # Better than emitting an all-NaN output that hides every transient.
    mad_sigma = float("nan")
    if (not np.isfinite(mean_amp) or mean_amp <= 1e-12):
        base_arr = y_centered[finite_keep] if np.any(finite_keep) else y_centered[np.isfinite(y_centered)]
        if base_arr.size >= 5:
            mad = float(np.nanmedian(np.abs(base_arr - np.nanmedian(base_arr))))
            if np.isfinite(mad) and mad > 1e-12:
                mad_sigma = 1.4826 * mad
                # Use a peak-equivalent scale: mean of |samples| above 2*sigma.
                # Falls back to mad_sigma itself if no such samples exist.
                tail = np.abs(base_arr)
                tail = tail[tail >= 2.0 * mad_sigma]
                mean_amp = float(np.nanmean(tail)) if tail.size else float(mad_sigma)
                scale_source = "mad_noise_fallback"

    if n_peaks > 1:
        sem_amp = float(np.nanstd(amplitudes, ddof=1) / np.sqrt(n_peaks))
    elif n_peaks == 1:
        sem_amp = 0.0
    else:
        sem_amp = float("nan")

    fs = float(fs_used)
    if not np.isfinite(fs) or fs <= 0:
        fs = 1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
    duration = float(np.sum(keep) / fs) if np.isfinite(fs) and fs > 0 else np.nan

    if not np.isfinite(mean_amp) or mean_amp <= 1e-12 or not np.isfinite(baseline_median):
        # Absolute last resort: leave the trace centered by baseline median
        # so the user at least sees the data, instead of a flat-line NaN.
        out = y - baseline_median
        scale_source = "centered_only_no_scale"
        mean_amp = float("nan")
    else:
        out = (y - baseline_median) / mean_amp

    # Peak times / values for the raw-signal overlay.
    if idx_peak is not None and len(idx_peak) > 0:
        idx_arr = np.asarray(idx_peak, int)
        idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < t.size)]
        peak_times = t[idx_arr].astype(float)
        peak_values = y[idx_arr].astype(float)
    else:
        peak_times = np.array([], float)
        peak_values = np.array([], float)

    baseline_intervals = _baseline_intervals_from_mask(t, keep)

    return out, {
        "duration_s": duration,
        "mean_amplitude": mean_amp,
        "sem_amplitude": sem_amp,
        "n_peaks": float(n_peaks),
        "baseline_median": baseline_median,
        "scale_source": scale_source,
        "mad_noise_sigma": mad_sigma,
        "peak_times": peak_times,
        "peak_values": peak_values,
        "baseline_intervals": baseline_intervals,
        "effective_min_peak": effective_min_peak,
        "baseline_source": source,
    }


def ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a*x + b with ordinary least squares (finite samples only)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10:
        return 1.0, 0.0
    X = np.vstack([x[m], np.ones(np.sum(m))]).T
    coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
    return float(coef[0]), float(coef[1])


def _rlm_huber_fit(
    x: np.ndarray,
    y: np.ndarray,
    huber_t: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    """
    Robust linear regression y ≈ a*x + b using IRLS with Huber weights.

    Weighting:
      r_i = y_i - (a*x_i + b)
      s   = robust scale estimate (MAD of residuals)
      u_i = r_i / s
      w_i = 1                    if |u_i| <= t
            t / |u_i|            if |u_i| >  t

    Returns:
      (a, b)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 10:
        return 1.0, 0.0

    xx = x[m]
    yy = y[m]

    # Initialize with OLS
    a, b = ols_fit(xx, yy)

    # IRLS loop
    for _ in range(int(max_iter)):
        yhat = a * xx + b
        r = yy - yhat

        s = _mad(r)
        if not np.isfinite(s) or s <= 1e-12:
            break

        u = r / s
        absu = np.abs(u)

        # Huber weights
        w = np.ones_like(absu, dtype=float)
        big = absu > float(huber_t)
        w[big] = float(huber_t) / absu[big]

        # Weighted least squares solve
        # Solve min || sqrt(w) * (yy - (a*xx + b)) ||^2
        W = np.sqrt(w)
        Xw = np.vstack([xx * W, W]).T
        yw = yy * W
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        a_new, b_new = float(coef[0]), float(coef[1])

        if (abs(a_new - a) + abs(b_new - b)) < float(tol):
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return float(a), float(b)


def fit_reference_to_signal(
    ref: np.ndarray,
    sig: np.ndarray,
    params: ProcessingParams,
    *,
    nonnegative_slope: bool = False,
) -> Tuple[float, float]:
    """
    Fit sig ≈ a*ref + b using the selected method in params.reference_fit.

    Notes:
    - If nonnegative_slope is True, a is constrained to be >= 0 so the model
      cannot silently undo the user-selected reference polarity.
    - For Lasso: if sklearn is unavailable, we fall back to OLS.
    - For RLM (HuberT): we use an internal IRLS implementation.
    """
    x = np.asarray(ref, float)
    y = np.asarray(sig, float)
    m = np.isfinite(x) & np.isfinite(y)

    if np.sum(m) < 10:
        return 1.0, 0.0

    method = str(params.reference_fit or "OLS (recommended)")

    def _constant_fit() -> Tuple[float, float]:
        yy = y[m]
        if yy.size == 0:
            return 0.0, 0.0
        if method.startswith("RLM"):
            return 0.0, float(np.nanmedian(yy))
        return 0.0, float(np.nanmean(yy))

    def _apply_slope_constraint(a: float, b: float) -> Tuple[float, float]:
        if bool(nonnegative_slope) and np.isfinite(a) and float(a) < 0.0:
            return _constant_fit()
        return float(a), float(b)

    # --- Lasso ---
    if method.startswith("Lasso"):
        if Lasso is None:
            return _apply_slope_constraint(*ols_fit(x, y))
        model = Lasso(
            alpha=float(params.lasso_alpha),
            fit_intercept=True,
            max_iter=5000,
            positive=bool(nonnegative_slope),
        )
        model.fit(x[m].reshape(-1, 1), y[m])
        a = float(model.coef_[0])
        b = float(model.intercept_)
        return _apply_slope_constraint(a, b)

    # --- Robust regression: Huber ---
    if method.startswith("RLM"):
        return _apply_slope_constraint(*_rlm_huber_fit(
            x,
            y,
            huber_t=float(params.rlm_huber_t),
            max_iter=int(params.rlm_max_iter),
            tol=float(params.rlm_tol),
        ))

    # --- Default: OLS ---
    return _apply_slope_constraint(*ols_fit(x, y))


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise division with protection against near-zero denominators."""
    num = np.asarray(num, float)
    den = np.asarray(den, float).copy()
    den[np.abs(den) < 1e-12] = np.nan
    return num / den


def _compute_fitted_reference_dff(
    ref: np.ndarray,
    sig: np.ndarray,
    params: ProcessingParams,
    *,
    invert_reference: bool = False,
    nonnegative_slope: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fit the isobestic trace onto the calcium trace, then compute dF/F.

    When invert_reference is True, the reference is multiplied by -1 before
    fitting. This documents and enforces the intended polarity for sessions
    where the isobestic artifact moves opposite to the 465 nm signal.
    """
    sig_arr = np.asarray(sig, float)
    ref_for_fit = np.asarray(ref, float)
    if invert_reference:
        ref_for_fit = -ref_for_fit
    a, b = fit_reference_to_signal(
        ref_for_fit,
        sig_arr,
        params,
        nonnegative_slope=nonnegative_slope,
    )
    fitted_ref = a * ref_for_fit + b
    dff_fit = safe_divide(sig_arr - fitted_ref, fitted_ref)
    return dff_fit, fitted_ref, a, b


def _rolling_median_highpass(x: np.ndarray, fs: float, window_s: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Split a trace into rolling-median trend and residual components.

    The residual is used for band-limited reference correction so slow bleaching
    or session drift does not dominate the reference fit.
    """
    y = np.asarray(x, float)
    if y.size == 0:
        return y.copy(), y.copy(), 0
    if np.any(~np.isfinite(y)):
        y = interpolate_nans(y)
    n = _window_samples_from_seconds(float(fs), float(window_s), minimum=5, require_odd=True)
    if n >= y.size:
        fallback = np.full_like(y, float(np.nanmedian(y)), dtype=float)
        return y - fallback, fallback, int(y.size)
    trend = np.asarray(median_filter(y, size=int(n), mode="nearest"), float)
    return y - trend, trend, int(n)


def _fit_nonnegative_no_intercept(x: np.ndarray, y: np.ndarray) -> float:
    """Fit y = beta*x with beta constrained to be nonnegative."""
    xx = np.asarray(x, float)
    yy = np.asarray(y, float)
    m = np.isfinite(xx) & np.isfinite(yy)
    if np.sum(m) < 10:
        return 0.0
    den = float(np.sum(xx[m] * xx[m]))
    if not np.isfinite(den) or den <= 1e-12:
        return 0.0
    beta = float(np.sum(xx[m] * yy[m]) / den)
    if not np.isfinite(beta) or beta < 0.0:
        return 0.0
    return beta


def _compute_band_limited_reference_dff(
    dff_sig: np.ndarray,
    dff_ref: np.ndarray,
    fs: float,
    params: ProcessingParams,
    *,
    invert_reference: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Correct dFF using only the short-timescale reference component.

    This is designed for recordings where slow 410/470 drift is positively
    shared but motion-like fluctuations are anti-correlated. A rolling median
    removes slow drift from both dFF traces, a nonnegative no-intercept beta is
    fit in the selected polarity, and only that band-limited reference component
    is subtracted from the original signal dFF.
    """
    sig_arr = np.asarray(dff_sig, float)
    ref_arr = np.asarray(dff_ref, float)
    window_s = float(getattr(params, "band_limited_reference_window_s", 60.0) or 60.0)
    if not np.isfinite(window_s) or window_s <= 0:
        window_s = 60.0
    sig_hp, _, win_n = _rolling_median_highpass(sig_arr, fs, window_s)
    ref_hp, _, _ = _rolling_median_highpass(ref_arr, fs, window_s)
    reg = -ref_hp if bool(invert_reference) else ref_hp
    beta = _fit_nonnegative_no_intercept(reg, sig_hp)
    corrected = sig_arr - beta * reg
    return np.asarray(corrected, float), float(beta), int(win_n)


# =============================================================================
# Worker task (stable)
# =============================================================================

class _TaskSignals(QtCore.QObject):
    finished = QtCore.Signal(object, int, float)  # (ProcessedTrial, job_id, elapsed_s)
    failed = QtCore.Signal(str, int)


class PreviewTask(QtCore.QRunnable):
    def __init__(
        self,
        processor: "PhotometryProcessor",
        trial: LoadedTrial,
        params: ProcessingParams,
        manual_regions_sec: List[Tuple[float, float]],
        manual_exclude_regions_sec: List[Tuple[float, float]],
        job_id: int,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.processor = processor
        self.trial = trial
        self.params = params
        self.manual = manual_regions_sec
        self.manual_exclude = manual_exclude_regions_sec
        self.job_id = job_id
        self.signals = _TaskSignals()

    def run(self) -> None:
        t0 = time.time()
        try:
            proc = self.processor.process_trial(
                self.trial,
                self.params,
                manual_regions_sec=self.manual,
                manual_exclude_regions_sec=self.manual_exclude,
                preview_mode=True,
            )
            self.signals.finished.emit(proc, self.job_id, time.time() - t0)
        except Exception as e:
            self.signals.failed.emit(str(e), self.job_id)


# =============================================================================
# Processor
# =============================================================================

class PhotometryProcessor:
    def load_file(self, path: str) -> LoadedDoricFile:
        with h5py.File(path, "r") as f:
            base = f["DataAcquisition"]["FPConsole"]["Signals"]["Series0001"]

            chans: List[str] = []
            if "LockInAOUT02" in base:
                for k in base["LockInAOUT02"].keys():
                    if k.startswith("AIN"):
                        chans.append(k)
            chans = sorted(chans) or ["AIN01"]

            def _read_time(folder: str) -> np.ndarray:
                if folder in base and "Time" in base[folder]:
                    return np.asarray(base[folder]["Time"][()], float)
                return np.array([], float)

            time_by: Dict[str, np.ndarray] = {}
            sig_by: Dict[str, np.ndarray] = {}
            ref_by: Dict[str, np.ndarray] = {}

            for ch in chans:
                sig = np.asarray(base["LockInAOUT02"][ch][()], float)
                t_sig = _read_time("LockInAOUT02")

                if "LockInAOUT01" in base and ch in base["LockInAOUT01"]:
                    ref = np.asarray(base["LockInAOUT01"][ch][()], float)
                    t_ref = _read_time("LockInAOUT01")
                else:
                    ref = np.full_like(sig, np.nan, dtype=float)
                    t_ref = np.array([], float)

                # Best-effort time vector selection (Doric exports can vary)
                if t_sig.size == sig.size:
                    t = t_sig
                elif t_ref.size == sig.size:
                    t = t_ref
                else:
                    dt = float(np.nanmedian(np.diff(t_sig))) if t_sig.size > 1 else 1.0 / 1000.0
                    t = np.arange(sig.size, dtype=float) * dt

                # Ensure reference matches signal length (interp if possible; otherwise resize)
                if ref.size != sig.size and t_ref.size == ref.size:
                    ref = np.interp(t, t_ref, ref)
                elif ref.size != sig.size:
                    ref = np.resize(ref, sig.size)

                time_by[ch] = t
                sig_by[ch] = sig
                ref_by[ch] = ref

            digital_time = None
            digital_by: Dict[str, np.ndarray] = {}
            if "DigitalIO" in base:
                dio = base["DigitalIO"]
                if "Time" in dio:
                    digital_time = np.asarray(dio["Time"][()], float)
                for k in dio.keys():
                    if k.startswith("DIO"):
                        digital_by[k] = np.asarray(dio[k][()], float)

            trigger_by: Dict[str, np.ndarray] = dict(digital_by)
            trigger_time_by: Dict[str, np.ndarray] = {}
            if digital_time is not None:
                for name in digital_by.keys():
                    trigger_time_by[name] = np.asarray(digital_time, float)

            if "AnalogOut" in base:
                aout = base["AnalogOut"]
                aout_time = np.asarray(aout["Time"][()], float) if "Time" in aout else None
                for k in aout.keys():
                    if k.startswith("AOUT"):
                        trigger_by[k] = np.asarray(aout[k][()], float)
                        if aout_time is not None:
                            trigger_time_by[k] = np.asarray(aout_time, float)

            return LoadedDoricFile(
                path=path,
                channels=chans,
                time_by_channel=time_by,
                signal_by_channel=sig_by,
                reference_by_channel=ref_by,
                digital_time=digital_time,
                digital_by_name=digital_by,
                trigger_time_by_name=trigger_time_by,
                trigger_by_name=trigger_by,
            )

    def make_preview_task(
        self,
        trial: LoadedTrial,
        params: ProcessingParams,
        manual_regions_sec: List[Tuple[float, float]],
        manual_exclude_regions_sec: List[Tuple[float, float]],
        job_id: int,
    ) -> PreviewTask:
        return PreviewTask(self, trial, params, manual_regions_sec, manual_exclude_regions_sec, job_id)

    def _baseline(self, t: np.ndarray, x: np.ndarray, params: ProcessingParams) -> np.ndarray:
        """
        Estimate baseline for a trace using pybaselines.

        Baseline is computed AFTER filtering and any resampling, so that:
        - artifacts are already removed/interpolated
        - bandwidth is controlled (less baseline leakage)
        - signals are aligned and at the final timebase
        """
        fitter = Baseline(x_data=t)
        method = (params.baseline_method or "airpls").lower()
        if method not in BASELINE_METHODS:
            method = "airpls"

        lam = float(params.baseline_lambda)
        diff_order = int(params.baseline_diff_order)
        max_iter = int(params.baseline_max_iter)
        tol = float(params.baseline_tol)

        if method == "asls":
            p = float(params.asls_p)
            b, _ = fitter.asls(x, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol)
            return np.asarray(b, float)
        if method == "arpls":
            b, _ = fitter.arpls(x, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
            return np.asarray(b, float)

        b, _ = fitter.airpls(x, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol)
        return np.asarray(b, float)

    def process_trial(
        self,
        trial: LoadedTrial,
        params: ProcessingParams,
        manual_regions_sec: Optional[List[Tuple[float, float]]] = None,
        manual_exclude_regions_sec: Optional[List[Tuple[float, float]]] = None,
        preview_mode: bool = False,
    ) -> ProcessedTrial:
        # ---------------------------------------------------------------------
        # 1) Load raw arrays
        # ---------------------------------------------------------------------
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float)
        ref = np.asarray(trial.reference_405, float)

        if bool(getattr(params, "invert_polarity", False)):
            sig = -sig
            ref = -ref

        fs = float(trial.sampling_rate) if np.isfinite(trial.sampling_rate) else (
            1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        )
        sensor = get_sensor(str(getattr(params, "sensor_id", SENSOR_UNKNOWN) or SENSOR_UNKNOWN))
        sensor_check = assess_sensor_trace(sensor.sensor_id, t, sig)

        # ---------------------------------------------------------------------
        # 2) Display envelope on raw 465 (computed pre-masking for user context)
        # ---------------------------------------------------------------------
        artifact_mode = _normalize_artifact_mode(getattr(params, "artifact_mode", SMART_ARTIFACT_MODE))
        envelope_mode = "Adaptive MAD (windowed)" if artifact_mode == SMART_ARTIFACT_MODE else artifact_mode
        hi_raw, lo_raw = _compute_signal_envelope(
            t,
            sig,
            float(params.mad_k),
            envelope_mode,
            float(params.adaptive_window_s),
        )
        hi_ref_raw, lo_ref_raw = _compute_signal_envelope(
            t,
            ref,
            float(params.mad_k),
            envelope_mode,
            float(params.adaptive_window_s),
        )

        raw_display_dio_time = None
        raw_display_dio = None
        if trial.trigger is not None and np.asarray(trial.trigger).size and trial.trigger_name:
            trig = np.asarray(trial.trigger, float)
            trig_t = getattr(trial, "trigger_time", None)
            if trig_t is not None and np.asarray(trig_t).size == trig.size:
                raw_display_dio_time = np.asarray(trig_t, float)
                raw_display_dio = trig
            elif trig.size == t.size:
                raw_display_dio_time = np.asarray(t, float)
                raw_display_dio = trig

        # ---------------------------------------------------------------------
        # 3) Artifact detection on BOTH the signal (465) and reference (405)
        #    channels, then OR the masks. A spike in either channel - even one
        #    that only shows up in the isosbestic reference - is a real motion
        #    artefact and must be cut from both channels together so the cut
        #    boundary stays aligned. The QC dialog already does this; the
        #    processing pipeline used to look only at the signal channel,
        #    leaving reference-side spikes untouched.
        # ---------------------------------------------------------------------
        mask_sig = np.zeros_like(t, dtype=bool)
        mask_ref = np.zeros_like(t, dtype=bool)
        core_sig = np.zeros_like(t, dtype=bool)
        core_ref = np.zeros_like(t, dtype=bool)
        auto_regions: List[Tuple[float, float]] = []
        auto_core_regions: List[Tuple[float, float]] = []
        auto_sources: List[str] = []
        smart_artifact_summary = ""
        if bool(getattr(params, "artifact_detection_enabled", True)):
            if artifact_mode == SMART_ARTIFACT_MODE:
                smart_result = detect_artifacts_smart(
                    t,
                    sig,
                    ref,
                    k=float(params.mad_k),
                    window_s=float(params.adaptive_window_s),
                    pad_s=float(params.artifact_pad_s),
                    fs=fs,
                )
                core_sig = np.asarray(smart_result.signal_core_mask, bool)
                core_ref = np.asarray(smart_result.reference_core_mask, bool)
                mask = np.asarray(smart_result.mask, bool)
                mask_sig = mask & (core_sig | np.asarray(smart_result.core_mask, bool))
                mask_ref = mask & (core_ref | np.asarray(smart_result.core_mask, bool))
                auto_regions = list(smart_result.regions)
                auto_core_regions = list(smart_result.core_regions)
                auto_sources = list(smart_result.region_sources)
                smart_artifact_summary = str(smart_result.summary or "")
            else:
                core_sig = _mask_outside_signal_envelope(t, sig, hi_raw, lo_raw, 0.0)
                core_ref = _mask_outside_signal_envelope(t, ref, hi_ref_raw, lo_ref_raw, 0.0)
                pad_s = float(params.artifact_pad_s)
                mask_sig = _pad_mask_by_seconds(t, core_sig, pad_s)
                mask_ref = _pad_mask_by_seconds(t, core_ref, pad_s)
                mask = np.asarray(mask_sig, bool) | np.asarray(mask_ref, bool)
        else:
            mask = np.zeros_like(t, dtype=bool)

        core_mask = np.asarray(core_sig, bool) | np.asarray(core_ref, bool)
        core_regions = regions_from_mask(t, core_mask)
        if artifact_mode != SMART_ARTIFACT_MODE:
            auto_regions = regions_from_mask(t, mask)
            auto_core_regions = []
            auto_sources = []
            for a, b in auto_regions:
                in_region = (t >= float(a)) & (t <= float(b))
                sig_hit = bool(np.any(np.asarray(core_sig, bool) & in_region))
                ref_hit = bool(np.any(np.asarray(core_ref, bool) & in_region))
                if sig_hit and ref_hit:
                    auto_sources.append("465 + 405")
                elif sig_hit:
                    auto_sources.append("465")
                elif ref_hit:
                    auto_sources.append("405")
                else:
                    auto_sources.append("")
                overlapping_core = [
                    (max(float(ca), float(a)), min(float(cb), float(b)))
                    for ca, cb in core_regions
                    if float(cb) >= float(a) and float(ca) <= float(b)
                ]
                if overlapping_core:
                    auto_core_regions.append((overlapping_core[0][0], overlapping_core[-1][1]))
                else:
                    auto_core_regions.append((float(a), float(b)))
        if manual_exclude_regions_sec:
            mask = remove_manual_regions(t, mask, manual_exclude_regions_sec or [])
        mask = apply_manual_regions(t, mask, manual_regions_sec or [])
        final_regions = regions_from_mask(t, mask)

        # ---------------------------------------------------------------------
        # 4) Apply selected artifact handling.
        #    Interpolate is the historical default and keeps the timebase intact.
        #    Cut is applied after resampling so all processed arrays stay aligned.
        # ---------------------------------------------------------------------
        sig_corr, ref_corr, artifact_handling = _apply_artifact_handling(sig, ref, mask, fs, params)

        raw_display_signal = np.asarray(sig, float).copy()
        raw_display_reference = np.asarray(ref, float).copy()
        raw_display_thr_hi = np.asarray(hi_raw, float).copy() if hi_raw is not None else None
        raw_display_thr_lo = np.asarray(lo_raw, float).copy() if lo_raw is not None else None
        raw_display_ref_thr_hi = np.asarray(hi_ref_raw, float).copy() if hi_ref_raw is not None else None
        raw_display_ref_thr_lo = np.asarray(lo_ref_raw, float).copy() if lo_ref_raw is not None else None
        if artifact_handling == "Cut" and np.any(mask):
            raw_display_signal[mask] = np.nan
            raw_display_reference[mask] = np.nan
            if raw_display_thr_hi is not None and raw_display_thr_hi.size == mask.size:
                raw_display_thr_hi[mask] = np.nan
            if raw_display_thr_lo is not None and raw_display_thr_lo.size == mask.size:
                raw_display_thr_lo[mask] = np.nan
            if raw_display_ref_thr_hi is not None and raw_display_ref_thr_hi.size == mask.size:
                raw_display_ref_thr_hi[mask] = np.nan
            if raw_display_ref_thr_lo is not None and raw_display_ref_thr_lo.size == mask.size:
                raw_display_ref_thr_lo[mask] = np.nan
            if raw_display_dio is not None and raw_display_dio_time is not None:
                dio_t = np.asarray(raw_display_dio_time, float)
                dio_m = np.zeros_like(dio_t, dtype=bool)
                for a, b in final_regions:
                    dio_m |= (dio_t >= float(a)) & (dio_t <= float(b))
                if np.asarray(raw_display_dio).size == dio_m.size:
                    raw_display_dio = np.asarray(raw_display_dio, float).copy()
                    raw_display_dio[dio_m] = np.nan

        # ---------------------------------------------------------------------
        # 5) Low-pass filter before decimation (anti-aliasing)
        # ---------------------------------------------------------------------
        target_fs = float(params.target_fs_hz)
        cutoff = float(params.lowpass_hz)
        if np.isfinite(fs) and np.isfinite(target_fs) and fs > target_fs * 1.05:
            cutoff = min(cutoff, 0.45 * target_fs)

        sig_f = _lowpass_sos(sig_corr, fs, cutoff, int(params.filter_order))
        ref_f = _lowpass_sos(ref_corr, fs, cutoff, int(params.filter_order))

        # ---------------------------------------------------------------------
        # 6) Resample signals together to target fs (only if true decimation)
        # ---------------------------------------------------------------------
        t2, sig2, ref2, fs_used = _resample_pair_to_target_fs(t, sig_f, ref_f, fs, target_fs)
        sig2 = _apply_optional_smoothing(sig2, fs_used, params)
        ref2 = _apply_optional_smoothing(ref2, fs_used, params)

        # Resample the envelope for display (same timebase as processed)
        _, hi2, lo2, _ = _resample_pair_to_target_fs(t, hi_raw, lo_raw, fs, target_fs)

        if artifact_handling == "Cut" and np.any(mask):
            mask2 = np.zeros_like(t2, dtype=bool)
            for a, b in final_regions:
                mask2 |= (t2 >= float(a)) & (t2 <= float(b))
            keep = ~mask2
            if np.sum(keep) >= 3:
                t2 = t2[keep]
                sig2 = sig2[keep]
                ref2 = ref2[keep]
                hi2 = hi2[keep]
                lo2 = lo2[keep]

        # ---------------------------------------------------------------------
        # 7) A/D overlay (if present): interpolate and binarize
        # ---------------------------------------------------------------------
        dio2 = None
        dio_name = ""
        if trial.trigger is not None and trial.trigger.size and trial.trigger_name:
            dio_name = trial.trigger_name
            dio_interp = np.interp(t2, t, np.asarray(trial.trigger, float))
            dio2 = (dio_interp > 0.5).astype(float)

        all_triggers2 = {}
        if hasattr(trial, "triggers") and trial.triggers:
            for name, val in trial.triggers.items():
                vt = trial.trigger_times.get(name)
                if vt is not None:
                    interp_val = np.interp(t2, vt, np.asarray(val, float))
                    all_triggers2[name] = (interp_val > 0.5).astype(float)

        # ---------------------------------------------------------------------
        # 8) Baseline estimation AFTER filtering/resampling (on final timebase)
        # ---------------------------------------------------------------------
        b_sig = self._baseline(t2, sig2, params)
        finite_ref2 = np.asarray(ref2, float)
        finite_ref2 = finite_ref2[np.isfinite(finite_ref2)]
        has_reference = bool(finite_ref2.size >= 5 and np.nanstd(finite_ref2) > 1e-12)
        b_ref = self._baseline(t2, ref2, params) if has_reference else np.full_like(ref2, np.nan, dtype=float)

        # ---------------------------------------------------------------------
        # 9) Compute requested output mode
        # ---------------------------------------------------------------------
        mode = params.output_mode if params.output_mode in OUTPUT_MODES else OUTPUT_MODES[0]
        out: Optional[np.ndarray] = None
        prominence_stats: Optional[Dict[str, float]] = None
        no_reference_fallback = False
        fit_slope: Optional[float] = None
        fit_intercept: Optional[float] = None
        band_ref_beta: Optional[float] = None
        band_ref_window_n: Optional[int] = None

        # --- Compute baseline-referenced dFFs (building blocks) ---
        # dFF_sig = (sig_filtered - baseline_sig) / baseline_sig
        # dFF_ref = (ref_filtered - baseline_ref) / baseline_ref
        dff_sig = safe_divide(sig2 - b_sig, b_sig)
        dff_ref = safe_divide(ref2 - b_ref, b_ref)
        dff_sub = dff_sig - dff_ref

        if mode == "dFF (non motion corrected)":
            # (1) dFF (non motion corrected)
            # dFF = (signal_filtered - signal_baseline) / signal_baseline
            out = dff_sig

        elif mode == "zscore (non motion corrected)":
            # (2) zscore (non motion corrected)
            # zscore(dFF_nonMC)
            out = zscore_median_std(dff_sig)

        elif mode == "dFF (motion corrected via subtraction)":
            # (3) dFF (motion corrected via subtraction)
            # dFF_mc = dFF_sig - dFF_ref
            if has_reference:
                out = dff_sub
            else:
                out = dff_sig
                no_reference_fallback = True

        elif mode == "zscore (motion corrected via subtraction)":
            # (4) zscore (motion corrected via subtraction)
            # zscore(dFF_sig - dFF_ref)
            if has_reference:
                out = zscore_median_std(dff_sub)
            else:
                out = zscore_median_std(dff_sig)
                no_reference_fallback = True

        elif mode == "zscore (subtractions)":
            # (5) zscore (subtractions)
            # zscore(dFF_sig) - zscore(dFF_ref)
            if has_reference:
                out = zscore_median_std(dff_sig) - zscore_median_std(dff_ref)
            else:
                out = zscore_median_std(dff_sig)
                no_reference_fallback = True

        elif mode == "dFF (motion corrected with fitted ref)":
            # (6) dFF (motion corrected with fitted ref)
            # 1) Fit reference (405) onto signal (465): fitted_ref = a*ref_filtered + b
            # 2) Compute dFF using fitted reference denominator:
            #    dFF = (sig_filtered - fitted_ref) / fitted_ref
            if has_reference:
                out, _, fit_slope, fit_intercept = _compute_fitted_reference_dff(ref2, sig2, params)
            else:
                out = dff_sig
                no_reference_fallback = True

        elif mode == "dFF (motion corrected with inverted isobestic fit)":
            # (7) dFF (motion corrected with inverted isobestic fit)
            # 1) Invert the isobestic before fitting: fitted_ref = a*(-ref_filtered) + b
            # 2) Compute dFF using the fitted reference in 465 signal units:
            #    dFF = (sig_filtered - fitted_ref) / fitted_ref
            if has_reference:
                out, _, fit_slope, fit_intercept = _compute_fitted_reference_dff(
                    ref2,
                    sig2,
                    params,
                    invert_reference=True,
                )
            else:
                out = dff_sig
                no_reference_fallback = True

        elif mode == BAND_LIMITED_INVERTED_ISO_MODE:
            # (8) band-limited inverted isobestic correction
            # 1) Compute dFF for signal and reference.
            # 2) High-pass both with a rolling median window.
            # 3) Fit beta >= 0 from -dFF_ref_hp to dFF_sig_hp.
            # 4) Subtract only that short-timescale inverted reference component.
            if has_reference:
                out, band_ref_beta, band_ref_window_n = _compute_band_limited_reference_dff(
                    dff_sig,
                    dff_ref,
                    fs_used,
                    params,
                    invert_reference=True,
                )
            else:
                out = dff_sig
                no_reference_fallback = True

        elif mode == "zscore (motion corrected with fitted ref)":
            # (9) zscore (motion corrected with fitted ref)
            # zscore( (sig_filtered - fitted_ref) / fitted_ref )
            if has_reference:
                dff_fit, _, fit_slope, fit_intercept = _compute_fitted_reference_dff(ref2, sig2, params)
                out = zscore_median_std(dff_fit)
            else:
                out = zscore_median_std(dff_sig)
                no_reference_fallback = True

        elif mode == "prominence normalized (motion corrected with fitted ref)":
            # (10) prominence-normalized fitted-ref dFF
            # 1) Fit reference and compute fitted-ref dFF.
            # 2) Exclude event windows from the selected trigger channel.
            # 3) Detect baseline peaks by prominence, average the top fraction,
            #    then scale like a z-score using peak prominence instead of std.
            if has_reference:
                dff_fit, _, fit_slope, fit_intercept = _compute_fitted_reference_dff(ref2, sig2, params)
            else:
                dff_fit = dff_sig
                no_reference_fallback = True
            event_times = _trigger_rising_edges(t2, dio2)
            out, prominence_stats = prominence_normalize(dff_fit, t2, event_times, params, fs_used)

        elif mode == "Raw signal (465)":
            # (11) raw signal (processed 465 trace)
            # Directly expose the filtered/resampled 465 channel.
            out = np.asarray(sig2, float)

        else:
            # Safety fallback (should not happen if OUTPUT_MODES is authoritative)
            out = dff_sig

        context_parts = []
        if sensor.sensor_id != SENSOR_UNKNOWN:
            context_parts.append(f"Sensor: {sensor.name} ({sensor.target})")
            if isinstance(sensor_check, dict) and str(sensor_check.get("status", "")) == "warn":
                msg = str(sensor_check.get("message", "") or "").strip()
                if msg:
                    context_parts.append(f"Sensor check: {msg}")
        if np.any(mask):
            context_parts.append(f"Artifacts: {artifact_handling}")
            if smart_artifact_summary:
                context_parts.append(smart_artifact_summary)
        if no_reference_fallback:
            context_parts.append("No 405 reference; using signal-only output")
        if mode == "Raw signal (465)":
            context_parts.append("Raw 465 after artifact handling, filtering, and resampling")
        else:
            baseline_desc = f"Baseline: {params.baseline_method} (lambda={float(params.baseline_lambda):.2e})"
            if mode in (
                "dFF (motion corrected with fitted ref)",
                "dFF (motion corrected with inverted isobestic fit)",
                BAND_LIMITED_INVERTED_ISO_MODE,
                "zscore (motion corrected with fitted ref)",
                "prominence normalized (motion corrected with fitted ref)",
            ):
                if mode == BAND_LIMITED_INVERTED_ISO_MODE and band_ref_beta is not None:
                    window_s = float(getattr(params, "band_limited_reference_window_s", 60.0) or 60.0)
                    context_parts.append(
                        f"Band-limited inverted 405: beta={float(band_ref_beta):.4g}, "
                        f"window={window_s:.4g}s, n={int(band_ref_window_n or 0)}"
                    )
                elif fit_slope is not None and fit_intercept is not None:
                    context_parts.append(
                        f"Fit: {params.reference_fit} "
                        f"(slope={float(fit_slope):.4g}, intercept={float(fit_intercept):.4g})"
                    )
                else:
                    context_parts.append(f"Fit: {params.reference_fit}")
                if mode in ("dFF (motion corrected with inverted isobestic fit)", BAND_LIMITED_INVERTED_ISO_MODE):
                    context_parts.append("Iso polarity: inverted before fit")
            context_parts.append(baseline_desc)
            if prominence_stats is not None:
                mean_amp = prominence_stats.get("mean_amplitude", np.nan)
                sem_amp = prominence_stats.get("sem_amplitude", np.nan)
                n_peaks = int(prominence_stats.get("n_peaks", 0.0))
                duration = prominence_stats.get("duration_s", np.nan)
                scale_source = str(prominence_stats.get("scale_source", "baseline_peak_prominence"))
                context_parts.append(
                    f"Prominence scale [{scale_source}]: "
                    f"mean={mean_amp:.3g}, sem={sem_amp:.3g}, n={n_peaks}, "
                    f"baseline={duration:.3g}s, top={float(getattr(params, 'prominence_percent_top', 0.10)):.3g}, "
                    f"exclude=-{float(getattr(params, 'prominence_exclude_before_s', 0.0)):.3g}/+"
                    f"{float(getattr(params, 'prominence_exclude_after_s', 0.0)):.3g}s, "
                    f"min={float(getattr(params, 'prominence_min_peak', 0.0)):.3g}, "
                    f"max={float(getattr(params, 'prominence_max_peak', 1e6)):.3g}"
                )
        if bool(getattr(params, "smoothing_enabled", False)):
            sm_method = str(getattr(params, "smoothing_method", "Savitzky-Golay") or "Savitzky-Golay")
            sm_win = float(getattr(params, "smoothing_window_s", 0.0))
            sm_desc = f"Smoothing: {sm_method} (window={sm_win:.3g}s"
            if sm_method.startswith("Savitzky"):
                sm_desc += f", poly={int(getattr(params, 'smoothing_polyorder', 2))}"
            sm_desc += ")"
            context_parts.append(sm_desc)
        output_context = " | ".join(context_parts)

        # ---------------------------------------------------------------------
        # 10) Package outputs
        # ---------------------------------------------------------------------
        return ProcessedTrial(
            path=trial.path,
            channel_id=trial.channel_id,
            time=t2,
            raw_signal=sig2,
            raw_reference=ref2,
            raw_thr_hi=hi2,
            raw_thr_lo=lo2,
            raw_display_time=np.asarray(t, float),
            raw_display_signal=raw_display_signal,
            raw_display_reference=raw_display_reference,
            raw_display_thr_hi=raw_display_thr_hi,
            raw_display_thr_lo=raw_display_thr_lo,
            raw_display_ref_thr_hi=raw_display_ref_thr_hi,
            raw_display_ref_thr_lo=raw_display_ref_thr_lo,
            raw_display_dio_time=raw_display_dio_time,
            raw_display_dio=raw_display_dio,
            dio=dio2,
            dio_name=dio_name,
            triggers=all_triggers2,
            sig_f=sig2,
            ref_f=ref2,
            baseline_sig=b_sig,
            baseline_ref=b_ref,
            output=out,
            output_label=mode,
            output_context=output_context,
            outputs={mode: np.asarray(out, float)} if out is not None else {},
            sensor_label=(sensor.name if sensor.sensor_id != SENSOR_UNKNOWN else ""),
            sensor_check=sensor_check if isinstance(sensor_check, dict) else {},
            artifact_regions_sec=final_regions,
            artifact_regions_auto_sec=auto_regions,
            artifact_regions_auto_core_sec=auto_core_regions,
            artifact_regions_auto_source=auto_sources,
            prominence_peak_times=(
                np.asarray(prominence_stats.get("peak_times", np.array([], float)), float)
                if prominence_stats is not None else None
            ),
            prominence_peak_values=(
                np.asarray(prominence_stats.get("peak_values", np.array([], float)), float)
                if prominence_stats is not None else None
            ),
            prominence_baseline_intervals=(
                list(prominence_stats.get("baseline_intervals", []) or [])
                if prominence_stats is not None else None
            ),
            prominence_threshold=(
                float(prominence_stats.get("effective_min_peak", float("nan")))
                if prominence_stats is not None else float("nan")
            ),
            prominence_baseline_source=(
                str(prominence_stats.get("baseline_source", ""))
                if prominence_stats is not None else ""
            ),
            fs_actual=float(fs),
            fs_target=float(target_fs),
            fs_used=float(fs_used),
        )
