"""Time-synchronization helpers for pyBer postprocessing.

The GUI uses this module to align a photometry timebase to an external camera or
behavior timebase from shared TTL/barcode-like sync events.  The mapping is
defined as:

    camera_time = f(photometry_time)

so the returned aligned vector can be saved next to the processed photometry
trace and used directly for behavior/postprocessing analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SyncResult:
    method: str
    status: str
    aligned_time: np.ndarray
    camera_events: np.ndarray
    fiber_events: np.ndarray
    matched_camera_events: np.ndarray
    matched_fiber_events: np.ndarray
    fitted_camera_events: np.ndarray
    residuals: np.ndarray
    slope: float = float("nan")
    intercept: float = float("nan")
    rms_error_s: float = float("nan")
    median_error_s: float = float("nan")
    max_abs_error_s: float = float("nan")
    median_lag_s: float = float("nan")
    drift_ppm: float = float("nan")
    pair_offset: int = 0
    warnings: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, object]:
        """JSON-safe report without large arrays."""
        return {
            "method": str(self.method),
            "status": str(self.status),
            "n_camera_events": int(np.asarray(self.camera_events).size),
            "n_fiber_events": int(np.asarray(self.fiber_events).size),
            "n_matched": int(np.asarray(self.matched_camera_events).size),
            "slope": float(self.slope) if np.isfinite(self.slope) else float("nan"),
            "intercept": float(self.intercept) if np.isfinite(self.intercept) else float("nan"),
            "rms_error_s": float(self.rms_error_s) if np.isfinite(self.rms_error_s) else float("nan"),
            "median_error_s": float(self.median_error_s) if np.isfinite(self.median_error_s) else float("nan"),
            "max_abs_error_s": float(self.max_abs_error_s) if np.isfinite(self.max_abs_error_s) else float("nan"),
            "median_lag_s": float(self.median_lag_s) if np.isfinite(self.median_lag_s) else float("nan"),
            "drift_ppm": float(self.drift_ppm) if np.isfinite(self.drift_ppm) else float("nan"),
            "pair_offset": int(self.pair_offset),
            "warnings": [str(w) for w in self.warnings],
        }


def _finite_sorted(time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, float).reshape(-1)
    x = np.asarray(signal, float).reshape(-1)
    n = min(t.size, x.size)
    if n <= 0:
        return np.array([], float), np.array([], float)
    t = t[:n]
    x = x[:n]
    m = np.isfinite(t) & np.isfinite(x)
    if not np.any(m):
        return np.array([], float), np.array([], float)
    t = t[m]
    x = x[m]
    order = np.argsort(t)
    return t[order], x[order]


def _auto_threshold(x: np.ndarray) -> float:
    finite = np.asarray(x, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.5
    lo = float(np.nanpercentile(finite, 10))
    hi = float(np.nanpercentile(finite, 90))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return 0.5 * (lo + hi)
    return float(np.nanmedian(finite))


def _deduplicate_events(events: np.ndarray, min_interval_s: float) -> np.ndarray:
    ev = np.asarray(events, float).reshape(-1)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        return ev
    ev = np.sort(ev)
    keep = [float(ev[0])]
    min_dt = max(0.0, float(min_interval_s))
    for val in ev[1:]:
        if float(val) - keep[-1] >= min_dt:
            keep.append(float(val))
    return np.asarray(keep, float)


def extract_sync_events(
    time: np.ndarray,
    signal: Optional[np.ndarray] = None,
    *,
    mode: str = "ttl_rising",
    threshold: Optional[float] = None,
    polarity: str = "high",
    min_interval_s: float = 0.2,
) -> np.ndarray:
    """Extract event times from a sync trace or precomputed timestamp vector."""
    if signal is None:
        return _deduplicate_events(np.asarray(time, float), min_interval_s)

    t, x = _finite_sorted(time, signal)
    if t.size < 2:
        return np.array([], float)

    mode_l = str(mode or "").strip().lower()
    polarity_l = str(polarity or "").strip().lower()
    if "barcode" in mode_l or "change" in mode_l or "value" in mode_l:
        rounded = np.asarray(x, float)
        finite = rounded[np.isfinite(rounded)]
        if finite.size == 0:
            return np.array([], float)
        # Preserve integer/barcode transitions when possible; otherwise smooth
        # tiny float noise so value changes are not over-counted.
        if np.nanmax(finite) - np.nanmin(finite) <= 2.0:
            values = (rounded > _auto_threshold(rounded)).astype(int)
        else:
            values = np.round(rounded, decimals=3)
        idx = np.flatnonzero(values[1:] != values[:-1]) + 1
        return _deduplicate_events(t[idx], min_interval_s)

    thr = _auto_threshold(x) if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
    high = x > thr
    if "low" in polarity_l:
        high = ~high
    if "fall" in mode_l:
        idx = np.flatnonzero((~high[1:]) & high[:-1]) + 1
    else:
        idx = np.flatnonzero(high[1:] & (~high[:-1])) + 1
    return _deduplicate_events(t[idx], min_interval_s)


def _paired_by_offset(
    camera_events: np.ndarray,
    fiber_events: np.ndarray,
    offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cam = np.asarray(camera_events, float).reshape(-1)
    fib = np.asarray(fiber_events, float).reshape(-1)
    if offset >= 0:
        n = min(cam.size, fib.size - offset)
        if n <= 0:
            return np.array([], float), np.array([], float)
        return cam[:n], fib[offset:offset + n]
    start_cam = -offset
    n = min(cam.size - start_cam, fib.size)
    if n <= 0:
        return np.array([], float), np.array([], float)
    return cam[start_cam:start_cam + n], fib[:n]


def _fit_linear(camera: np.ndarray, fiber: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    cam = np.asarray(camera, float)
    fib = np.asarray(fiber, float)
    if cam.size >= 2 and np.nanmax(fib) > np.nanmin(fib):
        slope, intercept = np.polyfit(fib, cam, 1)
    elif cam.size == 1:
        slope = 1.0
        intercept = float(cam[0] - fib[0])
    else:
        slope = float("nan")
        intercept = float("nan")
    fitted = slope * fib + intercept if np.isfinite(slope) and np.isfinite(intercept) else np.full(cam.shape, np.nan)
    residuals = cam - fitted
    return float(slope), float(intercept), np.asarray(fitted, float), np.asarray(residuals, float)


def match_sync_events(
    camera_events: np.ndarray,
    fiber_events: np.ndarray,
    *,
    max_offset: int = 5,
    min_pairs: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """Pair camera and fiber sync events by order, allowing dropped leading pulses."""
    cam = _deduplicate_events(np.asarray(camera_events, float), 0.0)
    fib = _deduplicate_events(np.asarray(fiber_events, float), 0.0)
    warnings: List[str] = []
    if cam.size == 0 or fib.size == 0:
        return np.array([], float), np.array([], float), 0, ["Missing camera or fiber sync events."]
    if min(cam.size, fib.size) < max(1, min_pairs):
        n = min(cam.size, fib.size)
        return cam[:n], fib[:n], 0, [f"Only {n} sync pair(s); alignment is weak."]

    best: Optional[Tuple[float, int, int, np.ndarray, np.ndarray]] = None
    max_off = max(0, int(max_offset))
    max_pairs = min(cam.size, fib.size)
    for offset in range(-max_off, max_off + 1):
        c, f = _paired_by_offset(cam, fib, offset)
        if c.size < min_pairs:
            continue
        _, _, _, resid = _fit_linear(c, f)
        finite = resid[np.isfinite(resid)]
        if finite.size == 0:
            continue
        rms = float(np.sqrt(np.nanmean(finite ** 2)))
        # Dropping many pulses can always overfit a straight line. Penalize
        # unused pairs, but still allow offsets when the unshifted residual is
        # clearly wrong because a leading pulse was missed.
        score = rms + 0.25 * max(0, max_pairs - int(c.size)) + 0.001 * abs(offset)
        candidate = (score, -int(c.size), offset, c, f)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        n = min(cam.size, fib.size)
        warnings.append("Could not robustly offset-match sync pulses; paired by order.")
        return cam[:n], fib[:n], 0, warnings
    _, neg_n, offset, c_best, f_best = best
    if int(-neg_n) < min(cam.size, fib.size):
        warnings.append(f"Matched with pulse offset {offset}; check for dropped leading sync pulses.")
    return c_best, f_best, int(offset), warnings


def _interp_with_linear_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    xp = np.asarray(xp, float)
    fp = np.asarray(fp, float)
    if xp.size == 0:
        return np.full(x.shape, np.nan, dtype=float)
    if xp.size == 1:
        return x + (float(fp[0]) - float(xp[0]))
    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]
    out = np.interp(x, xp, fp)
    left = x < xp[0]
    right = x > xp[-1]
    left_slope = (fp[1] - fp[0]) / (xp[1] - xp[0]) if xp[1] != xp[0] else 1.0
    right_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]) if xp[-1] != xp[-2] else 1.0
    out[left] = fp[0] + (x[left] - xp[0]) * left_slope
    out[right] = fp[-1] + (x[right] - xp[-1]) * right_slope
    return out


def align_timebase(
    fiber_time: np.ndarray,
    camera_events: np.ndarray,
    fiber_events: np.ndarray,
    *,
    method: str = "linear",
    max_offset: int = 5,
    min_pairs: int = 2,
) -> SyncResult:
    """Return a camera-time vector for each photometry sample."""
    ft = np.asarray(fiber_time, float).reshape(-1)
    cam, fib, offset, warnings = match_sync_events(
        camera_events,
        fiber_events,
        max_offset=max_offset,
        min_pairs=min_pairs,
    )
    if cam.size == 0 or fib.size == 0:
        return SyncResult(
            method=str(method),
            status="failed",
            aligned_time=np.full(ft.shape, np.nan, dtype=float),
            camera_events=np.asarray(camera_events, float),
            fiber_events=np.asarray(fiber_events, float),
            matched_camera_events=cam,
            matched_fiber_events=fib,
            fitted_camera_events=np.array([], float),
            residuals=np.array([], float),
            pair_offset=offset,
            warnings=warnings,
        )

    method_l = str(method or "").strip().lower()
    slope, intercept, fitted, residuals = _fit_linear(cam, fib)
    if "interp" in method_l and cam.size >= 2:
        aligned = _interp_with_linear_extrapolation(ft, fib, cam)
        method_out = "interpolation"
        # Keep the linear fit and its residuals for diagnostics.  Interpolation
        # passes exactly through matched pulses, so interpolation residuals would
        # hide dropped-pulse or non-linear clock problems in the QC report.
    else:
        aligned = slope * ft + intercept if np.isfinite(slope) and np.isfinite(intercept) else np.full(ft.shape, np.nan)
        method_out = "linear_regression"

    finite_resid = residuals[np.isfinite(residuals)]
    if finite_resid.size:
        rms = float(np.sqrt(np.nanmean(finite_resid ** 2)))
        med = float(np.nanmedian(finite_resid))
        max_abs = float(np.nanmax(np.abs(finite_resid)))
    else:
        rms = med = max_abs = float("nan")
    if cam.size and fib.size:
        lag = cam - fib
        median_lag = float(np.nanmedian(lag[np.isfinite(lag)])) if np.any(np.isfinite(lag)) else float("nan")
    else:
        median_lag = float("nan")
    drift_ppm = float((slope - 1.0) * 1e6) if np.isfinite(slope) else float("nan")

    status = "ok"
    if cam.size < 3:
        status = "warning"
        warnings.append("Fewer than 3 matched sync events; inspect the residual plot.")
    if np.isfinite(rms) and rms > 0.2:
        status = "warning"
        warnings.append(f"High sync residual RMS ({rms * 1000:.1f} ms).")

    return SyncResult(
        method=method_out,
        status=status,
        aligned_time=np.asarray(aligned, float),
        camera_events=np.asarray(camera_events, float),
        fiber_events=np.asarray(fiber_events, float),
        matched_camera_events=cam,
        matched_fiber_events=fib,
        fitted_camera_events=np.asarray(fitted, float),
        residuals=np.asarray(residuals, float),
        slope=float(slope),
        intercept=float(intercept),
        rms_error_s=rms,
        median_error_s=med,
        max_abs_error_s=max_abs,
        median_lag_s=median_lag,
        drift_ppm=drift_ppm,
        pair_offset=offset,
        warnings=warnings,
    )
