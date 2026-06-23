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


DEFAULT_MIN_XCORR_R = 0.30


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


@dataclass(frozen=True)
class BarcodePacket:
    start_time: float
    end_time: float
    anchor_time: float
    code: Tuple[int, ...]
    n_transitions: int


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
    """Duty-cycle-aware threshold shared with the LED export path.

    Delegates to ``led_extract.compute_threshold`` (Otsu when the signal is
    balanced, Triangle when skewed/sparse) so the live preview, alignment and
    the "Binary (thresholded)" export all use the identical threshold. Falls
    back to a self-contained Otsu if that module cannot be imported.
    """
    finite = np.asarray(x, float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.5
    try:
        import led_extract

        thr = led_extract.compute_threshold(finite, "auto")
        if np.isfinite(thr):
            return float(thr)
    except Exception:
        pass

    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.nanmedian(finite))
    hist, edges = np.histogram(finite, bins=256, range=(lo, hi))
    hist = hist.astype(float)
    total = float(hist.sum())
    if total <= 0:
        return 0.5 * (lo + hi)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = total - w0
    s0 = np.cumsum(hist * centers)
    grand = float(s0[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = s0 / w0
        m1 = (grand - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -1.0
    idx = int(np.nanargmax(between))
    thr = float(centers[idx])
    if not np.isfinite(thr) or thr <= lo or thr >= hi:
        return 0.5 * (lo + hi)
    return thr


def _norm01(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / span


def _looks_discrete_value_trace(x: np.ndarray) -> bool:
    finite = np.asarray(x, float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return False
    rounded = np.round(finite, decimals=3)
    unique = np.unique(rounded)
    if unique.size <= 2:
        return False
    if unique.size > min(64, max(8, finite.size // 100)):
        return False
    nearest_int = np.rint(unique)
    return bool(np.nanmax(np.abs(unique - nearest_int)) <= 1e-3)


def edge_times(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    threshold_norm: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return sub-sample transition times and signed directions.

    The LED extractor aligns on all binary transitions, not only rising TTL
    events. This reproduces that behavior and estimates the threshold crossing
    by linear interpolation between neighboring samples.
    """
    t, x = _finite_sorted(time, signal)
    if t.size < 2:
        return np.array([], float), np.array([], np.int8)
    x_norm = _norm01(x)
    state = x_norm >= float(threshold_norm)
    changes = np.flatnonzero(np.diff(state.astype(np.int8)) != 0) + 1
    if changes.size == 0:
        return np.array([], float), np.array([], np.int8)
    before = x_norm[changes - 1]
    after = x_norm[changes]
    denom = after - before
    alpha = np.where(
        np.abs(denom) > 1e-12,
        (float(threshold_norm) - before) / denom,
        0.5,
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    edge_t = t[changes - 1] + alpha * (t[changes] - t[changes - 1])
    edge_dir = np.where(state[changes], 1, -1).astype(np.int8)
    return edge_t.astype(float), edge_dir


def _sample_rate_from_time(time: np.ndarray, fallback: float = 30.0) -> float:
    t = np.asarray(time, float).reshape(-1)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float(fallback)
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return float(fallback)
    med = float(np.nanmedian(dt))
    if not np.isfinite(med) or med <= 0.0:
        return float(fallback)
    return float(1.0 / med)


def pair_edges(
    fiber_edges: np.ndarray,
    fiber_dirs: np.ndarray,
    camera_edges: np.ndarray,
    camera_dirs: np.ndarray,
    offset: float,
    sample_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair fiber edge times to camera edge times after an xcorr offset seed.

    ``offset`` follows the cross-correlation convention used below:
    ``camera_time = fiber_time - offset``.
    """
    fib = np.asarray(fiber_edges, float).reshape(-1)
    cam = np.asarray(camera_edges, float).reshape(-1)
    fib_dirs = np.asarray(fiber_dirs, np.int8).reshape(-1)
    cam_dirs = np.asarray(camera_dirs, np.int8).reshape(-1)
    if fib.size == 0 or cam.size == 0:
        return np.array([], float), np.array([], float)

    cam_on_fiber_time = cam + float(offset)
    spacings: List[float] = []
    for arr in (fib, cam_on_fiber_time):
        if arr.size > 1:
            d = np.diff(arr)
            spacings.extend([float(v) for v in d[np.isfinite(d) & (d > 0.0)]])
    typical = float(np.nanmedian(spacings)) if spacings else 1.0
    if not np.isfinite(typical) or typical <= 0.0:
        typical = 1.0
    rate = max(float(sample_rate_hz), 1e-9)
    tol = max(3.0 / rate, min(1.0, 0.45 * typical))

    pairs_fiber: List[float] = []
    pairs_camera: List[float] = []
    j = 0
    for i, fe in enumerate(fib):
        fd = fib_dirs[i] if i < fib_dirs.size else 0
        while j < cam_on_fiber_time.size and cam_on_fiber_time[j] < fe - tol:
            j += 1
        candidates: List[int] = []
        for k in (j, j + 1):
            if k < cam_on_fiber_time.size and abs(float(cam_on_fiber_time[k] - fe)) <= tol:
                if cam_dirs.size <= k or int(cam_dirs[k]) == int(fd):
                    candidates.append(int(k))
        if candidates:
            k_best = min(candidates, key=lambda idx: abs(float(cam_on_fiber_time[idx] - fe)))
            pairs_fiber.append(float(fe))
            pairs_camera.append(float(cam[k_best]))
            j = k_best + 1

    if len(pairs_fiber) < 2:
        diff = abs(int(fib.size) - int(cam.size))
        slack = max(2, int(0.15 * min(fib.size, cam.size)))
        if diff <= slack:
            n = min(fib.size, cam.size)
            pairs_fiber = fib[:n].astype(float).tolist()
            pairs_camera = cam[:n].astype(float).tolist()

    return np.asarray(pairs_fiber, float), np.asarray(pairs_camera, float)


def estimate_xcorr_offset(
    fiber_time: np.ndarray,
    fiber_signal: np.ndarray,
    camera_time: np.ndarray,
    camera_signal: np.ndarray,
    *,
    max_lag_s: float = 5.0,
) -> Optional[Dict[str, object]]:
    """Estimate the global offset between fiber and camera sync traces.

    The sign convention is ``camera_time = fiber_time - offset``. This is the
    same convention used by the standalone barcode extractor.
    """
    ft, fx = _finite_sorted(fiber_time, fiber_signal)
    ct, cx = _finite_sorted(camera_time, camera_signal)
    if ft.size < 20 or ct.size < 20:
        return None

    dt_f = np.diff(ft)
    dt_f = dt_f[np.isfinite(dt_f) & (dt_f > 0.0)]
    dt_c = np.diff(ct)
    dt_c = dt_c[np.isfinite(dt_c) & (dt_c > 0.0)]
    if dt_f.size == 0 and dt_c.size == 0:
        return None
    dt = float(np.nanmedian(dt_f if dt_f.size else dt_c))
    if dt_c.size:
        dt = max(dt, float(np.nanmedian(dt_c)))
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    lag = max(0.0, float(max_lag_s))
    t_lo = max(float(ft[0]), float(ct[0]) - lag)
    t_hi = min(float(ft[-1]), float(ct[-1]) + lag)
    if (t_hi - t_lo) < max(1.0, 20.0 * dt):
        return None
    grid = np.arange(t_lo, t_hi + 0.5 * dt, dt, dtype=np.float64)
    if grid.size < 20:
        return None

    a = np.interp(grid, ft, fx, left=np.nan, right=np.nan)
    b = np.interp(grid, ct, cx, left=np.nan, right=np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 20:
        return None
    a = _norm01(a[mask])
    b = _norm01(b[mask])
    grid = grid[mask]

    a_c = a - float(np.nanmean(a))
    b_c = b - float(np.nanmean(b))
    a_std = float(np.nanstd(a_c))
    b_std = float(np.nanstd(b_c))
    if a_std < 1e-12 or b_std < 1e-12:
        return None

    try:
        from scipy.signal import fftconvolve
        corr = fftconvolve(a_c, b_c[::-1], mode="full")
    except Exception:
        corr = np.correlate(a_c, b_c, mode="full")
    n = int(a_c.size)
    lags = np.arange(-(n - 1), n, dtype=np.float64) * dt
    corr_norm = corr / max(a_std * b_std * n, 1e-12)
    keep = np.abs(lags) <= lag
    if not np.any(keep):
        return None
    lags = lags[keep]
    corr_norm = corr_norm[keep]
    if corr_norm.size == 0 or not np.any(np.isfinite(corr_norm)):
        return None
    best_idx = int(np.nanargmax(corr_norm))
    best_lag = float(lags[best_idx])
    return {
        "offset": best_lag,
        "lag": best_lag,
        "peak": float(corr_norm[best_idx]),
        "lags": lags,
        "corr_norm": corr_norm,
        "window_s": (float(grid[0]), float(grid[-1])),
        "samples": int(n),
    }


def _robust_filter_edge_pairs(
    pairs_fiber: np.ndarray,
    pairs_camera: np.ndarray,
    *,
    min_pairs: int = 3,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    fib = np.asarray(pairs_fiber, float).reshape(-1)
    cam = np.asarray(pairs_camera, float).reshape(-1)
    n = min(fib.size, cam.size)
    fib = fib[:n]
    cam = cam[:n]
    if n < max(1, int(min_pairs)):
        return fib, cam, n, n
    keep = np.isfinite(fib) & np.isfinite(cam)
    for _ in range(3):
        if int(np.sum(keep)) < max(2, int(min_pairs)):
            break
        slope, intercept, _fitted, resid = _fit_linear(cam[keep], fib[keep])
        if not np.isfinite(slope) or not np.isfinite(intercept):
            break
        r = resid[np.isfinite(resid)]
        if r.size < max(2, int(min_pairs)):
            break
        med = float(np.nanmedian(r))
        mad = float(np.nanmedian(np.abs(r - med)))
        limit = max(0.05, 6.0 * 1.4826 * mad)
        local_keep = np.zeros_like(keep, dtype=bool)
        local_keep[keep] = np.abs(resid - med) <= limit
        if int(np.sum(local_keep)) == int(np.sum(keep)):
            break
        if int(np.sum(local_keep)) < max(2, int(min_pairs)):
            break
        keep = local_keep
    return fib[keep], cam[keep], int(np.sum(keep)), int(n)


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


def _is_barcode_mode(mode: str) -> bool:
    mode_l = str(mode or "").strip().lower()
    return "barcode" in mode_l or "change" in mode_l or "value" in mode_l


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
    if _is_barcode_mode(mode_l):
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            return np.array([], float)

        # A true decoded barcode/value trace is discrete. Preserve changes in
        # those integer-like values. A raw LED ROI trace is continuous, so it
        # must be thresholded to a binary state before edge extraction.
        if _looks_discrete_value_trace(x):
            values = np.round(x, decimals=3)
            idx = np.flatnonzero(values[1:] != values[:-1]) + 1
            return _deduplicate_events(t[idx], min_interval_s)

        try:
            packets = decode_barcode_packets(t, x, threshold=threshold)
            if len(packets) >= 2:
                return _deduplicate_events(
                    np.asarray([pkt.anchor_time for pkt in packets], float),
                    min_interval_s,
                )
        except Exception:
            pass

        thr = _auto_threshold(x) if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
        high = x > thr
        if "low" in polarity_l:
            high = ~high
        changes = np.flatnonzero(high[1:] != high[:-1]) + 1
        if changes.size == 0:
            return np.array([], float)
        before = x[changes - 1]
        after = x[changes]
        denom = after - before
        alpha = np.where(np.abs(denom) > 1e-12, (thr - before) / denom, 0.5)
        alpha = np.clip(alpha, 0.0, 1.0)
        edge_t = t[changes - 1] + alpha * (t[changes] - t[changes - 1])
        return _deduplicate_events(edge_t, 0.0)

    thr = _auto_threshold(x) if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
    high = x > thr
    if "low" in polarity_l:
        high = ~high
    if "change" in mode_l or "both" in mode_l:
        idx = np.flatnonzero(high[1:] != high[:-1]) + 1
    elif "fall" in mode_l:
        idx = np.flatnonzero((~high[1:]) & high[:-1]) + 1
    else:
        idx = np.flatnonzero(high[1:] & (~high[:-1])) + 1
    return _deduplicate_events(t[idx], min_interval_s)


def decode_barcode_packets(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    threshold: Optional[float] = None,
    min_transitions: int = 4,
) -> List[BarcodePacket]:
    """Decode binary barcode-like bursts into packet anchors and bit patterns."""
    t, x = _finite_sorted(time, signal)
    if t.size < 4:
        return []
    finite = x[np.isfinite(x)]
    if finite.size < 4:
        return []
    value_span = float(np.nanmax(finite) - np.nanmin(finite))
    if not np.isfinite(value_span) or value_span <= 0.0:
        return []
    thr = _auto_threshold(x) if threshold is None or not np.isfinite(float(threshold)) else float(threshold)
    values = (x > thr).astype(int)
    if np.unique(values).size < 2:
        return []
    change_idx = np.flatnonzero(values[1:] != values[:-1]) + 1
    if change_idx.size < max(2, int(min_transitions)):
        return []

    change_t = t[change_idx]
    intervals = np.diff(change_t)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if intervals.size < 2:
        return []
    short_dt = float(np.nanpercentile(intervals, 35))
    med_dt = float(np.nanmedian(intervals))
    if not np.isfinite(short_dt) or short_dt <= 0.0:
        short_dt = med_dt
    if not np.isfinite(med_dt) or med_dt <= 0.0:
        return []
    gap_threshold = max(0.35, 4.0 * short_dt, 2.5 * med_dt)
    split_after = np.flatnonzero(np.diff(change_t) > gap_threshold)
    if split_after.size == 0:
        return []

    starts = np.r_[0, split_after + 1]
    stops = np.r_[split_after + 1, change_idx.size]
    packets: List[BarcodePacket] = []
    for start, stop in zip(starts, stops):
        idx = change_idx[int(start):int(stop)]
        if idx.size < max(2, int(min_transitions)):
            continue
        trans_t = t[idx]
        trans_values = values[idx]
        local_dt = np.diff(trans_t)
        local_dt = local_dt[np.isfinite(local_dt) & (local_dt > 0.0)]
        if local_dt.size < 2:
            continue
        unit = float(np.nanpercentile(local_dt, 35))
        if not np.isfinite(unit) or unit <= 0.0:
            unit = float(np.nanmedian(local_dt))
        if not np.isfinite(unit) or unit <= 0.0:
            continue
        units = np.clip(np.rint(np.diff(trans_t) / unit).astype(int), 1, 16)
        if units.size == 0:
            continue
        bits: List[int] = []
        for val, reps in zip(trans_values[:-1], units):
            bits.extend([int(val)] * int(reps))
        bits.append(int(trans_values[-1]))
        if len(bits) < 3 or len(set(bits)) < 2:
            continue
        packets.append(
            BarcodePacket(
                start_time=float(trans_t[0]),
                end_time=float(trans_t[-1]),
                anchor_time=float(trans_t[0]),
                code=tuple(bits),
                n_transitions=int(idx.size),
            )
        )
    return packets


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


def _overlap_candidate_offsets(
    camera_events: np.ndarray,
    fiber_events: np.ndarray,
    *,
    max_candidates: int = 200,
) -> List[int]:
    """Infer plausible pulse offsets when both event vectors share a time range."""
    cam = np.asarray(camera_events, float).reshape(-1)
    fib = np.asarray(fiber_events, float).reshape(-1)
    if cam.size < 2 or fib.size < 2:
        return []

    overlap_start = max(float(cam[0]), float(fib[0]))
    overlap_end = min(float(cam[-1]), float(fib[-1]))
    cam_span = max(0.0, float(cam[-1] - cam[0]))
    fib_span = max(0.0, float(fib[-1] - fib[0]))
    overlap_span = overlap_end - overlap_start
    min_span = min(cam_span, fib_span)
    if overlap_span <= 0.0 or min_span <= 0.0:
        return []
    if overlap_span < max(1.0, 0.02 * min_span):
        return []

    offsets: set[int] = set()

    def _sample_indices(indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, int)
        if idx.size <= 64:
            return idx
        picks = np.linspace(0, idx.size - 1, 64)
        return np.unique(idx[np.round(picks).astype(int)])

    cam_idx = _sample_indices(np.flatnonzero((cam >= overlap_start) & (cam <= overlap_end)))
    fib_idx = _sample_indices(np.flatnonzero((fib >= overlap_start) & (fib <= overlap_end)))

    for i_raw in cam_idx:
        i = int(i_raw)
        j = int(np.searchsorted(fib, cam[i], side="left"))
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < fib.size:
                offsets.add(int(jj) - i)

    for j_raw in fib_idx:
        j = int(j_raw)
        i = int(np.searchsorted(cam, fib[j], side="left"))
        for ii in (i - 1, i, i + 1):
            if 0 <= ii < cam.size:
                offsets.add(j - int(ii))

    ranked = sorted(offsets, key=lambda val: (abs(int(val)), int(val)))
    return [int(val) for val in ranked[:max(1, int(max_candidates))]]


def _paired_packets_by_offset(
    camera_packets: List[BarcodePacket],
    fiber_packets: List[BarcodePacket],
    offset: int,
) -> Tuple[List[BarcodePacket], List[BarcodePacket]]:
    if offset >= 0:
        n = min(len(camera_packets), len(fiber_packets) - offset)
        if n <= 0:
            return [], []
        return camera_packets[:n], fiber_packets[offset:offset + n]
    start_cam = -offset
    n = min(len(camera_packets) - start_cam, len(fiber_packets))
    if n <= 0:
        return [], []
    return camera_packets[start_cam:start_cam + n], fiber_packets[:n]


def _barcode_code_distance(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, aval in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, bval in enumerate(b, start=1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if aval == bval else 1),
            )
        prev = cur
    return float(prev[-1]) / float(max(len(a), len(b), 1))


def match_barcode_packets(
    camera_packets: List[BarcodePacket],
    fiber_packets: List[BarcodePacket],
    *,
    max_offset: int = 5,
    min_pairs: int = 2,
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """Match barcode packets by decoded code, then return their anchor times."""
    warnings: List[str] = []
    if len(camera_packets) < max(1, int(min_pairs)) or len(fiber_packets) < max(1, int(min_pairs)):
        return np.array([], float), np.array([], float), 0, ["Not enough decoded barcode packets."]

    cam_anchors = np.asarray([pkt.anchor_time for pkt in camera_packets], float)
    fib_anchors = np.asarray([pkt.anchor_time for pkt in fiber_packets], float)

    cam_step = float(np.nanmedian(np.diff(cam_anchors))) if cam_anchors.size > 1 else 1.0
    fib_step = float(np.nanmedian(np.diff(fib_anchors))) if fib_anchors.size > 1 else 1.0
    typical_step = float(np.nanmedian([cam_step, fib_step]))
    if not np.isfinite(typical_step) or typical_step <= 0.0:
        typical_step = 1.0
    time_tolerance = max(0.35, min(2.0, 0.45 * typical_step))
    bin_width = max(0.05, min(0.5, 0.12 * typical_step))
    max_code_distance = 0.35

    pair_rows: List[Tuple[int, int, float, float]] = []
    for ci, c_pkt in enumerate(camera_packets):
        for fi, f_pkt in enumerate(fiber_packets):
            dist = _barcode_code_distance(c_pkt.code, f_pkt.code)
            if dist <= max_code_distance:
                pair_rows.append((ci, fi, float(c_pkt.anchor_time - f_pkt.anchor_time), float(dist)))
    if not pair_rows:
        return np.array([], float), np.array([], float), 0, ["No decoded barcode packet identities matched."]

    bins: Dict[int, List[Tuple[int, int, float, float]]] = {}
    for row in pair_rows:
        key = int(np.round(row[2] / bin_width))
        bins.setdefault(key, []).append(row)
    ranked_bins = sorted(
        bins.items(),
        key=lambda item: (-len(item[1]), float(np.nanmean([row[3] for row in item[1]])), abs(item[0])),
    )[:30]

    best: Optional[Tuple[float, int, int, np.ndarray, np.ndarray, float, float]] = None
    for _bin_key, rows in ranked_bins:
        lag0 = float(np.nanmedian([row[2] for row in rows]))
        used_camera: set[int] = set()
        matched: List[Tuple[int, int, float, float, float]] = []
        for fi, f_pkt in enumerate(fiber_packets):
            predicted = float(f_pkt.anchor_time + lag0)
            left = int(np.searchsorted(cam_anchors, predicted - time_tolerance, side="left"))
            right = int(np.searchsorted(cam_anchors, predicted + time_tolerance, side="right"))
            best_local: Optional[Tuple[float, int, float, float]] = None
            for ci in range(left, right):
                if ci in used_camera:
                    continue
                dist = _barcode_code_distance(camera_packets[ci].code, f_pkt.code)
                if dist > max_code_distance:
                    continue
                dt = abs(float(cam_anchors[ci] - predicted))
                local_score = dt + time_tolerance * dist
                candidate = (local_score, ci, dt, dist)
                if best_local is None or candidate < best_local:
                    best_local = candidate
            if best_local is None:
                continue
            _local_score, ci, dt, dist = best_local
            used_camera.add(int(ci))
            matched.append((int(ci), int(fi), float(cam_anchors[ci]), float(f_pkt.anchor_time), float(dist)))
        if len(matched) < min_pairs:
            continue
        c = np.asarray([row[2] for row in matched], float)
        f = np.asarray([row[3] for row in matched], float)
        dists = np.asarray([row[4] for row in matched], float)
        slope, intercept, fitted, resid = _fit_linear(c, f)
        finite = resid[np.isfinite(resid)]
        if finite.size == 0:
            continue
        rms = float(np.sqrt(np.nanmean(finite ** 2)))
        if finite.size >= 4:
            keep = np.abs(finite - float(np.nanmedian(finite))) <= max(0.15, 0.5 * time_tolerance)
            if int(np.sum(keep)) >= min_pairs and int(np.sum(keep)) < finite.size:
                c = c[keep]
                f = f[keep]
                dists = dists[keep]
                slope, intercept, fitted, resid = _fit_linear(c, f)
                finite = resid[np.isfinite(resid)]
                if finite.size == 0:
                    continue
                rms = float(np.sqrt(np.nanmean(finite ** 2)))
        mean_dist = float(np.nanmean(dists)) if dists.size else 1.0
        median_offset = int(np.round(np.nanmedian([row[1] - row[0] for row in matched])))
        score = rms + 0.05 * mean_dist + 0.5 / np.sqrt(max(1, int(c.size))) + 0.0001 * abs(lag0)
        candidate = (score, -int(c.size), median_offset, c, f, rms, mean_dist)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        return np.array([], float), np.array([], float), 0, ["No decoded barcode packet identities matched."]

    _score, neg_n, offset, c_best, f_best, _rms, mean_dist = best
    if int(-neg_n) < min(len(camera_packets), len(fiber_packets)):
        warnings.append(f"Matched barcode packets with median packet offset {offset}.")
    if mean_dist > 0.0:
        warnings.append(f"Mean barcode identity edit distance: {mean_dist:.3g}.")
    warnings.append(f"Decoded barcode packets: camera {len(camera_packets)}, photometry {len(fiber_packets)}.")
    return np.asarray(c_best, float), np.asarray(f_best, float), int(offset), warnings


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
    inferred_offsets = set(_overlap_candidate_offsets(cam, fib))
    candidate_offsets = set(range(-max_off, max_off + 1))
    candidate_offsets.update(inferred_offsets)
    for offset in sorted(candidate_offsets, key=lambda val: (abs(int(val)), int(val))):
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
    if offset in inferred_offsets and abs(int(offset)) > max_off:
        warnings.append(
            f"Inferred pulse offset {offset} from overlapping timestamps; check for unmatched leading sync pulses."
        )
    elif int(-neg_n) < min(cam.size, fib.size):
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


def _sync_result_from_matches(
    fiber_time: np.ndarray,
    camera_events: np.ndarray,
    fiber_events: np.ndarray,
    matched_camera_events: np.ndarray,
    matched_fiber_events: np.ndarray,
    *,
    method: str,
    pair_offset: int,
    warnings: List[str],
    method_prefix: str = "",
) -> SyncResult:
    ft = np.asarray(fiber_time, float).reshape(-1)
    cam = np.asarray(matched_camera_events, float).reshape(-1)
    fib = np.asarray(matched_fiber_events, float).reshape(-1)
    if cam.size == 0 or fib.size == 0:
        return SyncResult(
            method=str(method_prefix or method),
            status="failed",
            aligned_time=np.full(ft.shape, np.nan, dtype=float),
            camera_events=np.asarray(camera_events, float),
            fiber_events=np.asarray(fiber_events, float),
            matched_camera_events=cam,
            matched_fiber_events=fib,
            fitted_camera_events=np.array([], float),
            residuals=np.array([], float),
            pair_offset=int(pair_offset),
            warnings=list(warnings),
        )

    method_l = str(method or "").strip().lower()
    slope, intercept, fitted, residuals = _fit_linear(cam, fib)
    if "interp" in method_l and cam.size >= 2:
        aligned = _interp_with_linear_extrapolation(ft, fib, cam)
        base_method = "interpolation"
    else:
        aligned = slope * ft + intercept if np.isfinite(slope) and np.isfinite(intercept) else np.full(ft.shape, np.nan)
        base_method = "linear_regression"
    method_out = f"{method_prefix}_{base_method}" if method_prefix else base_method

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
    result_warnings = list(warnings)
    if cam.size < 3:
        status = "warning"
        result_warnings.append("Fewer than 3 matched sync events; inspect the residual plot.")
    if np.isfinite(rms) and rms > 0.2:
        status = "warning"
        result_warnings.append(f"High sync residual RMS ({rms * 1000:.1f} ms).")

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
        pair_offset=int(pair_offset),
        warnings=result_warnings,
    )


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
    cam, fib, offset, warnings = match_sync_events(
        camera_events,
        fiber_events,
        max_offset=max_offset,
        min_pairs=min_pairs,
    )
    return _sync_result_from_matches(
        fiber_time,
        camera_events,
        fiber_events,
        cam,
        fib,
        method=method,
        pair_offset=offset,
        warnings=warnings,
    )


def align_sync_traces_xcorr(
    fiber_time: np.ndarray,
    camera_time: np.ndarray,
    camera_signal: np.ndarray,
    fiber_signal: np.ndarray,
    *,
    camera_mode: str = "barcode",
    fiber_mode: str = "barcode",
    threshold: Optional[float] = None,
    min_interval_s: float = 0.2,
    method: str = "cross_correlation",
    max_lag_s: float = 5.0,
    min_pairs: int = 2,
) -> SyncResult:
    """Align traces with the standalone extractor's waveform correlation model.

    The returned mapping follows the module convention:
    ``camera_time = f(photometry_time)``.
    """
    ft = np.asarray(fiber_time, float).reshape(-1)
    xcorr = estimate_xcorr_offset(
        ft,
        np.asarray(fiber_signal, float),
        np.asarray(camera_time, float),
        np.asarray(camera_signal, float),
        max_lag_s=float(max_lag_s),
    )
    if xcorr is None:
        return SyncResult(
            method="cross_correlation",
            status="failed",
            aligned_time=np.full(ft.shape, np.nan, dtype=float),
            camera_events=np.array([], float),
            fiber_events=np.array([], float),
            matched_camera_events=np.array([], float),
            matched_fiber_events=np.array([], float),
            fitted_camera_events=np.array([], float),
            residuals=np.array([], float),
            warnings=["Cross-correlation could not estimate a reliable lag."],
        )

    offset = float(xcorr.get("offset", 0.0))
    camera_events = extract_sync_events(
        camera_time,
        camera_signal,
        mode=camera_mode,
        threshold=threshold,
        min_interval_s=min_interval_s,
    )
    fiber_events = extract_sync_events(
        fiber_time,
        fiber_signal,
        mode=fiber_mode,
        threshold=threshold,
        min_interval_s=min_interval_s,
    )
    rate = _sample_rate_from_time(ft, fallback=_sample_rate_from_time(camera_time, 30.0))
    pairs_fiber, pairs_camera = pair_edges(
        fiber_events,
        np.array([], np.int8),
        camera_events,
        np.array([], np.int8),
        offset,
        rate,
    )
    pairs_fiber, pairs_camera, n_inlier, n_total = _robust_filter_edge_pairs(
        pairs_fiber,
        pairs_camera,
        min_pairs=max(2, int(min_pairs)),
    )

    method_l = str(method or "").lower()
    warnings: List[str] = []
    peak = float(xcorr.get("peak", float("nan")))
    if np.isfinite(peak) and peak < DEFAULT_MIN_XCORR_R:
        warnings.append(
            f"Low cross-correlation peak r={peak:.3f}; inspect the waveform overlay."
        )
    if n_total and n_inlier < n_total:
        warnings.append(f"Rejected {n_total - n_inlier} inconsistent edge pair(s).")

    if ("linear" in method_l or "interp" in method_l) and pairs_fiber.size >= max(2, int(min_pairs)):
        warnings.insert(0, f"Cross-correlation seed offset: {offset:.6g} s.")
        return _sync_result_from_matches(
            ft,
            camera_events,
            fiber_events,
            pairs_camera,
            pairs_fiber,
            method=("interpolation" if "interp" in method_l else "linear"),
            pair_offset=0,
            warnings=warnings,
            method_prefix="xcorr_seeded",
        )

    aligned = ft - offset
    fitted = pairs_fiber - offset if pairs_fiber.size else np.array([], float)
    residuals = pairs_camera - fitted if pairs_camera.size and fitted.size else np.array([], float)
    finite_resid = residuals[np.isfinite(residuals)]
    if finite_resid.size:
        rms = float(np.sqrt(np.nanmean(finite_resid ** 2)))
        med = float(np.nanmedian(finite_resid))
        max_abs = float(np.nanmax(np.abs(finite_resid)))
        median_lag = float(np.nanmedian(pairs_camera - pairs_fiber))
    else:
        rms = med = max_abs = float("nan")
        median_lag = float(-offset)

    status = "ok"
    if not np.isfinite(peak) or peak < DEFAULT_MIN_XCORR_R:
        status = "warning"
    if pairs_fiber.size < max(2, int(min_pairs)):
        warnings.append(
            "Waveform lag was estimated, but too few thresholded edges were paired for residual diagnostics."
        )

    return SyncResult(
        method="cross_correlation",
        status=status,
        aligned_time=np.asarray(aligned, float),
        camera_events=np.asarray(camera_events, float),
        fiber_events=np.asarray(fiber_events, float),
        matched_camera_events=np.asarray(pairs_camera, float),
        matched_fiber_events=np.asarray(pairs_fiber, float),
        fitted_camera_events=np.asarray(fitted, float),
        residuals=np.asarray(residuals, float),
        slope=1.0,
        intercept=-offset,
        rms_error_s=rms,
        median_error_s=med,
        max_abs_error_s=max_abs,
        median_lag_s=median_lag,
        drift_ppm=0.0,
        pair_offset=0,
        warnings=warnings,
    )


def align_sync_traces(
    fiber_time: np.ndarray,
    camera_time: np.ndarray,
    camera_signal: np.ndarray,
    fiber_signal: np.ndarray,
    *,
    camera_mode: str = "ttl_rising",
    fiber_mode: str = "ttl_rising",
    threshold: Optional[float] = None,
    min_interval_s: float = 0.2,
    method: str = "linear",
    max_offset: int = 5,
    min_pairs: int = 2,
) -> SyncResult:
    """Align raw sync traces, using decoded barcode packets when available."""
    method_l = str(method or "").lower()
    if "cross" in method_l or "xcorr" in method_l:
        return align_sync_traces_xcorr(
            fiber_time,
            camera_time,
            camera_signal,
            fiber_signal,
            camera_mode=camera_mode,
            fiber_mode=fiber_mode,
            threshold=threshold,
            min_interval_s=min_interval_s,
            method=method,
            max_lag_s=float(max_offset),
            min_pairs=min_pairs,
        )

    barcode_requested = _is_barcode_mode(camera_mode) or _is_barcode_mode(fiber_mode)
    camera_packets = decode_barcode_packets(camera_time, camera_signal, threshold=threshold)
    fiber_packets = decode_barcode_packets(fiber_time, fiber_signal, threshold=threshold)
    if len(camera_packets) >= min_pairs and len(fiber_packets) >= min_pairs:
        cam, fib, offset, warnings = match_barcode_packets(
            camera_packets,
            fiber_packets,
            max_offset=max_offset,
            min_pairs=min_pairs,
        )
        if cam.size >= min_pairs and fib.size >= min_pairs:
            if not barcode_requested:
                warnings.insert(0, "Auto-detected barcode packets in both sync traces.")
            return _sync_result_from_matches(
                fiber_time,
                np.asarray([pkt.anchor_time for pkt in camera_packets], float),
                np.asarray([pkt.anchor_time for pkt in fiber_packets], float),
                cam,
                fib,
                method=method,
                pair_offset=offset,
                warnings=warnings,
                method_prefix="barcode_packets",
            )

    camera_events = extract_sync_events(
        camera_time,
        camera_signal,
        mode=camera_mode,
        threshold=threshold,
        min_interval_s=min_interval_s,
    )
    fiber_events = extract_sync_events(
        fiber_time,
        fiber_signal,
        mode=fiber_mode,
        threshold=threshold,
        min_interval_s=min_interval_s,
    )
    result = align_timebase(
        fiber_time,
        camera_events,
        fiber_events,
        method=method,
        max_offset=max_offset,
        min_pairs=min_pairs,
    )
    if barcode_requested:
        result.warnings.insert(0, "No reliable barcode packets decoded; fell back to edge-train matching.")
    return result
