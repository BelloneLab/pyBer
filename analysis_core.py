# analysis_core.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py

from scipy.signal import butter, sosfiltfilt, resample_poly
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

# Output modes (user requested: 7 clear options)
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
    # 7) zscore of that fitted-ref dFF
    "zscore (motion corrected with fitted ref)",
]


@dataclass
class ProcessingParams:
    # -------------------------
    # Artifact detection
    # -------------------------
    artifact_detection_enabled: bool = True
    artifact_mode: str = "Global MAD (dx)"  # or "Adaptive MAD (windowed)"
    mad_k: float = 8.0
    adaptive_window_s: float = 5.0
    artifact_pad_s: float = 0.25

    # -------------------------
    # Filtering
    # -------------------------
    lowpass_hz: float = 12.0
    filter_order: int = 3

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
    # Reference fit options (used by "fitted ref" output modes)
    # -------------------------
    reference_fit: str = "OLS (recommended)"  # OLS | Lasso | RLM (HuberT)

    # Lasso hyperparameter (only used if reference_fit == "Lasso")
    lasso_alpha: float = 1e-3

    # Robust regression (Huber) hyperparameters (only used if reference_fit == "RLM (HuberT)")
    rlm_huber_t: float = 1.345  # classic Huber threshold (in sigma units)
    rlm_max_iter: int = 50
    rlm_tol: float = 1e-6

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


@dataclass
class LoadedDoricFile:
    path: str
    channels: List[str]
    time_by_channel: Dict[str, np.ndarray]
    signal_by_channel: Dict[str, np.ndarray]
    reference_by_channel: Dict[str, np.ndarray]
    digital_time: Optional[np.ndarray]
    digital_by_name: Dict[str, np.ndarray]

    def make_trial(self, channel: str, trigger_name: Optional[str] = None) -> LoadedTrial:
        t = self.time_by_channel[channel]
        sig = self.signal_by_channel[channel]
        ref = self.reference_by_channel[channel]

        trig_t = None
        trig = None
        trig_name = trigger_name or ""

        if trigger_name:
            if self.digital_time is not None and trigger_name in self.digital_by_name:
                trig_t = self.digital_time
                trig = self.digital_by_name[trigger_name]

                # Align analog signals to DIO time base (so DIO overlays correctly).
                if trig_t is not None and trig_t.size and t.size and trig_t.size != t.size:
                    sig = np.interp(trig_t, t, sig)
                    ref = np.interp(trig_t, t, ref)
                    t = trig_t

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

    # Optional DIO aligned to processed time
    dio: Optional[np.ndarray] = None
    dio_name: str = ""

    # Processing intermediates
    sig_f: Optional[np.ndarray] = None
    ref_f: Optional[np.ndarray] = None
    baseline_sig: Optional[np.ndarray] = None
    baseline_ref: Optional[np.ndarray] = None

    # Final selected output
    output: Optional[np.ndarray] = None
    output_label: str = ""

    artifact_regions_sec: Optional[List[Tuple[float, float]]] = None

    fs_actual: float = np.nan
    fs_target: float = np.nan
    fs_used: float = np.nan


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

    if a and s and t:
        return f"{clean(a)}_{clean(s)}_{clean(t)}_{clean(channel)}"
    return f"{clean(base)}_{clean(channel)}"


def export_processed_csv(
    path: str,
    processed: ProcessedTrial,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    import csv

    t = np.asarray(processed.time, float)
    out = np.asarray(
        processed.output if processed.output is not None else np.full_like(t, np.nan),
        float,
    )

    dio = None
    if processed.dio is not None and processed.dio.size == t.size:
        dio = np.asarray(processed.dio, float)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if metadata:
            for k, v in metadata.items():
                w.writerow([f"# {k}: {v}"])
        if dio is None:
            w.writerow(["time", "output"])
            for i in range(t.size):
                w.writerow([float(t[i]), float(out[i]) if np.isfinite(out[i]) else np.nan])
        else:
            w.writerow(["time", "output", "dio"])
            for i in range(t.size):
                w.writerow([
                    float(t[i]),
                    float(out[i]) if np.isfinite(out[i]) else np.nan,
                    float(dio[i]) if np.isfinite(dio[i]) else np.nan,
                ])


def export_processed_h5(path: str, processed: ProcessedTrial, metadata: Optional[Dict[str, str]] = None) -> None:
    with h5py.File(path, "w") as f:
        g = f.create_group("data")
        g.create_dataset("time", data=np.asarray(processed.time, float), compression="gzip")
        g.create_dataset("output", data=np.asarray(processed.output, float), compression="gzip")
        g.attrs["output_label"] = str(processed.output_label)
        g.attrs["fs_actual"] = float(processed.fs_actual)
        g.attrs["fs_used"] = float(processed.fs_used)
        g.attrs["fs_target"] = float(processed.fs_target)

        g.create_dataset("raw_465", data=np.asarray(processed.raw_signal, float), compression="gzip")
        g.create_dataset("raw_405", data=np.asarray(processed.raw_reference, float), compression="gzip")

        if processed.dio is not None:
            g.create_dataset("dio", data=np.asarray(processed.dio, float), compression="gzip")
            g.attrs["dio_name"] = str(processed.dio_name)

        if processed.baseline_sig is not None:
            g.create_dataset("baseline_465", data=np.asarray(processed.baseline_sig, float), compression="gzip")
        if processed.baseline_ref is not None:
            g.create_dataset("baseline_405", data=np.asarray(processed.baseline_ref, float), compression="gzip")

        if metadata:
            mg = f.create_group("metadata")
            for k, v in metadata.items():
                mg.attrs[str(k)] = str(v)


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


def apply_manual_regions(time: np.ndarray, mask: np.ndarray, regions: List[Tuple[float, float]]) -> np.ndarray:
    """OR a user-provided list of regions (sec) into the artifact mask."""
    t = np.asarray(time, float)
    m = np.asarray(mask, bool).copy()
    for (a, b) in (regions or []):
        t0, t1 = (min(a, b), max(a, b))
        m |= (t >= t0) & (t <= t1)
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

    med = float(np.nanmedian(y))
    sp = float(k * _mad(y))
    hi = np.full_like(y, med + sp, dtype=float)
    lo = np.full_like(y, med - sp, dtype=float)
    return hi, lo


def detect_artifacts_global_dx(time: np.ndarray, x: np.ndarray, k: float, pad_s: float) -> np.ndarray:
    """
    Artifact detection via derivative thresholding:
    - Compute dx
    - Threshold |dx| > k * MAD(dx)
    - Optionally pad around detections (pad_s)
    """
    t = np.asarray(time, float)
    y = np.asarray(x, float)
    dx = np.diff(y, prepend=y[0])
    m = _mad(dx)
    thr = k * m if np.isfinite(m) else np.nan
    mask = np.zeros_like(y, dtype=bool)
    if np.isfinite(thr) and thr > 0:
        mask = np.abs(dx) > thr

    if pad_s > 0 and t.size > 1:
        dt = float(np.nanmedian(np.diff(t)))
        pad_n = int(max(0, round(pad_s / max(dt, 1e-12))))
        if pad_n > 0:
            idx = np.where(mask)[0]
            for i in idx:
                a = max(0, i - pad_n)
                b = min(mask.size, i + pad_n + 1)
                mask[a:b] = True
    return mask


def detect_artifacts_adaptive(time: np.ndarray, x: np.ndarray, k: float, window_s: float, pad_s: float) -> np.ndarray:
    """
    Artifact detection via windowed derivative thresholding:
    - For each window: threshold |dx| > k * MAD(dx_window)
    - Optionally pad around detections (pad_s)
    """
    t = np.asarray(time, float)
    y = np.asarray(x, float)
    dx = np.diff(y, prepend=y[0])

    if t.size < 3:
        return np.zeros_like(y, dtype=bool)

    dt = float(np.nanmedian(np.diff(t)))
    wN = int(max(10, round(window_s / max(dt, 1e-12))))
    stride = max(10, wN // 4)

    mask = np.zeros_like(y, dtype=bool)
    for start in range(0, y.size, stride):
        end = min(y.size, start + wN)
        m = _mad(dx[start:end])
        if not np.isfinite(m) or m <= 1e-12:
            continue
        thr = k * m
        mask[start:end] |= (np.abs(dx[start:end]) > thr)

    if pad_s > 0:
        pad_n = int(max(0, round(pad_s / max(dt, 1e-12))))
        if pad_n > 0:
            idx = np.where(mask)[0]
            for i in idx:
                a = max(0, i - pad_n)
                b = min(mask.size, i + pad_n + 1)
                mask[a:b] = True
    return mask


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
) -> Tuple[float, float]:
    """
    Fit sig ≈ a*ref + b using the selected method in params.reference_fit.

    Notes:
    - For Lasso: if sklearn is unavailable, we fall back to OLS.
    - For RLM (HuberT): we use an internal IRLS implementation.
    """
    x = np.asarray(ref, float)
    y = np.asarray(sig, float)
    m = np.isfinite(x) & np.isfinite(y)

    if np.sum(m) < 10:
        return 1.0, 0.0

    method = str(params.reference_fit or "OLS (recommended)")

    # --- Lasso ---
    if method.startswith("Lasso"):
        if Lasso is None:
            return ols_fit(x, y)
        model = Lasso(
            alpha=float(params.lasso_alpha),
            fit_intercept=True,
            max_iter=5000,
        )
        model.fit(x[m].reshape(-1, 1), y[m])
        a = float(model.coef_[0])
        b = float(model.intercept_)
        return a, b

    # --- Robust regression: Huber ---
    if method.startswith("RLM"):
        return _rlm_huber_fit(
            x,
            y,
            huber_t=float(params.rlm_huber_t),
            max_iter=int(params.rlm_max_iter),
            tol=float(params.rlm_tol),
        )

    # --- Default: OLS ---
    return ols_fit(x, y)


def safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise division with protection against near-zero denominators."""
    num = np.asarray(num, float)
    den = np.asarray(den, float).copy()
    den[np.abs(den) < 1e-12] = np.nan
    return num / den


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
        job_id: int,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self.processor = processor
        self.trial = trial
        self.params = params
        self.manual = manual_regions_sec
        self.job_id = job_id
        self.signals = _TaskSignals()

    def run(self) -> None:
        t0 = time.time()
        try:
            proc = self.processor.process_trial(self.trial, self.params, self.manual, preview_mode=True)
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

                ref = np.asarray(base["LockInAOUT01"][ch][()], float)
                t_ref = _read_time("LockInAOUT01")

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

            return LoadedDoricFile(
                path=path,
                channels=chans,
                time_by_channel=time_by,
                signal_by_channel=sig_by,
                reference_by_channel=ref_by,
                digital_time=digital_time,
                digital_by_name=digital_by,
            )

    def make_preview_task(
        self,
        trial: LoadedTrial,
        params: ProcessingParams,
        manual_regions_sec: List[Tuple[float, float]],
        job_id: int,
    ) -> PreviewTask:
        return PreviewTask(self, trial, params, manual_regions_sec, job_id)

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
        preview_mode: bool = False,
    ) -> ProcessedTrial:
        # ---------------------------------------------------------------------
        # 1) Load raw arrays
        # ---------------------------------------------------------------------
        t = np.asarray(trial.time, float)
        sig = np.asarray(trial.signal_465, float)
        ref = np.asarray(trial.reference_405, float)

        fs = float(trial.sampling_rate) if np.isfinite(trial.sampling_rate) else (
            1.0 / float(np.nanmedian(np.diff(t))) if t.size > 2 else np.nan
        )

        # ---------------------------------------------------------------------
        # 2) Display envelope on raw 465 (computed pre-masking for user context)
        # ---------------------------------------------------------------------
        hi_raw, lo_raw = _compute_signal_envelope(
            t,
            sig,
            float(params.mad_k),
            str(params.artifact_mode),
            float(params.adaptive_window_s),
        )

        # ---------------------------------------------------------------------
        # 3) Artifact detection on raw 465, then apply manual regions
        # ---------------------------------------------------------------------
        if bool(getattr(params, "artifact_detection_enabled", True)):
            if str(params.artifact_mode).startswith("Adaptive"):
                mask = detect_artifacts_adaptive(
                    t, sig, float(params.mad_k), float(params.adaptive_window_s), float(params.artifact_pad_s)
                )
            else:
                mask = detect_artifacts_global_dx(
                    t, sig, float(params.mad_k), float(params.artifact_pad_s)
                )
        else:
            mask = np.zeros_like(t, dtype=bool)

        mask = apply_manual_regions(t, mask, manual_regions_sec or [])

        # ---------------------------------------------------------------------
        # 4) Mask artifacts (set NaN) then interpolate (keeps timebase intact)
        # ---------------------------------------------------------------------
        sig_corr = sig.copy()
        ref_corr = ref.copy()
        sig_corr[mask] = np.nan
        ref_corr[mask] = np.nan
        sig_corr = interpolate_nans(sig_corr)
        ref_corr = interpolate_nans(ref_corr)

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

        # Resample the envelope for display (same timebase as processed)
        _, hi2, lo2, _ = _resample_pair_to_target_fs(t, hi_raw, lo_raw, fs, target_fs)

        # ---------------------------------------------------------------------
        # 7) DIO overlay (if present): interpolate and binarize
        # ---------------------------------------------------------------------
        dio2 = None
        dio_name = ""
        if trial.trigger is not None and trial.trigger.size and trial.trigger_name:
            dio_name = trial.trigger_name
            dio_interp = np.interp(t2, t, np.asarray(trial.trigger, float))
            dio2 = (dio_interp > 0.5).astype(float)

        # ---------------------------------------------------------------------
        # 8) Baseline estimation AFTER filtering/resampling (on final timebase)
        # ---------------------------------------------------------------------
        b_sig = self._baseline(t2, sig2, params)
        b_ref = self._baseline(t2, ref2, params)

        # ---------------------------------------------------------------------
        # 9) Compute requested output mode (7 user-defined options)
        # ---------------------------------------------------------------------
        mode = params.output_mode if params.output_mode in OUTPUT_MODES else OUTPUT_MODES[0]
        out: Optional[np.ndarray] = None

        # --- Compute baseline-referenced dFFs (building blocks) ---
        # dFF_sig = (sig_filtered - baseline_sig) / baseline_sig
        # dFF_ref = (ref_filtered - baseline_ref) / baseline_ref
        dff_sig = safe_divide(sig2 - b_sig, b_sig)
        dff_ref = safe_divide(ref2 - b_ref, b_ref)

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
            out = dff_sig - dff_ref

        elif mode == "zscore (motion corrected via subtraction)":
            # (4) zscore (motion corrected via subtraction)
            # zscore(dFF_sig - dFF_ref)
            out = zscore_median_std(dff_sig - dff_ref)

        elif mode == "zscore (subtractions)":
            # (5) zscore (subtractions)
            # zscore(dFF_sig) - zscore(dFF_ref)
            out = zscore_median_std(dff_sig) - zscore_median_std(dff_ref)

        elif mode == "dFF (motion corrected with fitted ref)":
            # (6) dFF (motion corrected with fitted ref)
            # 1) Fit reference (405) onto signal (465): fitted_ref = a*ref_filtered + b
            # 2) Compute dFF using fitted reference denominator:
            #    dFF = (sig_filtered - fitted_ref) / fitted_ref
            a, b = fit_reference_to_signal(ref2, sig2, params)
            fitted_ref = a * ref2 + b
            out = safe_divide(sig2 - fitted_ref, fitted_ref)

        elif mode == "zscore (motion corrected with fitted ref)":
            # (7) zscore (motion corrected with fitted ref)
            # zscore( (sig_filtered - fitted_ref) / fitted_ref )
            a, b = fit_reference_to_signal(ref2, sig2, params)
            fitted_ref = a * ref2 + b
            dff_fit = safe_divide(sig2 - fitted_ref, fitted_ref)
            out = zscore_median_std(dff_fit)

        else:
            # Safety fallback (should not happen if OUTPUT_MODES is authoritative)
            out = dff_sig

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
            dio=dio2,
            dio_name=dio_name,
            sig_f=sig2,
            ref_f=ref2,
            baseline_sig=b_sig,
            baseline_ref=b_ref,
            output=out,
            output_label=mode,
            artifact_regions_sec=regions_from_mask(t, mask),
            fs_actual=float(fs),
            fs_target=float(target_fs),
            fs_used=float(fs_used),
        )
