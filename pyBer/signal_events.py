"""Gap-preserving numerical helpers for signal-event analysis.

Inputs are never mutated. All indices refer to the original recording, and
filters, prominence bases, widths and integration stop at missing intervals.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths


def continuous_segments(t, y):
    """Return slices separated by NaNs, reversed time, or >3 nominal time steps."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t.ndim != 1 or y.ndim != 1 or t.shape != y.shape:
        raise ValueError("Time and signal must be one-dimensional arrays of equal length.")
    if t.size == 0:
        return []
    good = np.isfinite(t) & np.isfinite(y)
    delta = np.diff(t)
    adjacent = good[:-1] & good[1:]
    positive = delta[adjacent & (delta > 0)]
    nominal = float(np.median(positive)) if positive.size else np.inf
    linked = adjacent & (delta > 0) & (delta <= 3.0 * nominal)
    starts = np.flatnonzero(good & ~np.r_[False, linked])
    stops = np.flatnonzero(good & ~np.r_[linked, False]) + 1
    return [slice(int(a), int(b)) for a, b in zip(starts, stops)]


def observed_intervals(t, y):
    """Return observed start/end times, excluding unobserved time across cuts."""
    t = np.asarray(t, float)
    return np.asarray([(t[s.start], t[s.stop - 1]) for s in continuous_segments(t, y)], float).reshape(-1, 2)


def _window(t, seconds):
    """Translate a duration to an odd sample window on one continuous segment."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    window = max(3, int(round(max(0.1, seconds) / dt)))
    return dt, window + (window % 2 == 0)


def _rolling_baseline(y, window):
    """Compute a rolling median with bounded cost for high-rate recordings.

    Long windows use medians of short blocks before the rolling median. The
    resulting smooth baseline is interpolated only within this valid segment.
    """
    stride = max(1, int(np.ceil(window / 301)))
    if stride == 1:
        return median_filter(y, size=window, mode="nearest")
    starts = np.arange(0, len(y), stride)
    centers = np.minimum(starts + (stride - 1) / 2.0, len(y) - 1)
    medians = np.asarray([np.median(y[a:a + stride]) for a in starts])
    coarse_window = max(3, int(round(window / stride))) | 1
    baseline = median_filter(medians, size=coarse_window, mode="nearest")
    return np.interp(np.arange(len(y)), centers, baseline)


def preprocess_trace(t, y, baseline_mode="as-is", baseline_window_sec=10.0, smooth_sigma_sec=0.0):
    """Detrend and smooth each valid segment, retaining missing sample positions."""
    t, original = np.array(t, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    processed = np.full(original.shape, np.nan)
    for segment in continuous_segments(t, original):
        values = original[segment].copy()
        dt, window = _window(t[segment], baseline_window_sec)
        if baseline_mode.endswith("rolling median"):
            values -= _rolling_baseline(values, window)
        elif baseline_mode.endswith("rolling mean"):
            values -= uniform_filter1d(values, size=window, mode="nearest")
        if smooth_sigma_sec > 0:
            values = gaussian_filter1d(values, sigma=smooth_sigma_sec / dt, mode="nearest")
        processed[segment] = values
    return t, processed, original


def estimate_noise(t, y, baseline_window_sec=10.0, baseline_mask=None):
    """Estimate robust residual noise after removal of slow baseline drift.

    Sigma is Gaussian-consistent MAD of residuals, trimmed once at 3 sigma to
    reduce sparse-transient contamination. It is an empirical noise scale,
    not a confidence interval or a peak false-positive probability. A selected
    baseline supplies the samples for scale estimation; empty selections and
    constant traces deliberately return an unavailable sigma.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    baseline = np.full(y.shape, np.nan)
    for segment in continuous_segments(t, y):
        _, window = _window(t[segment], baseline_window_sec)
        baseline[segment] = _rolling_baseline(y[segment], window)
    residual = y - baseline
    keep = np.isfinite(residual)
    if baseline_mask is not None:
        selected = np.asarray(baseline_mask, bool)
        if selected.shape != y.shape:
            raise ValueError("Baseline mask must match the signal shape.")
        keep &= selected
    values = residual[keep]
    center = float(np.median(y[keep])) if values.size else np.nan
    mad, sigma = np.nan, np.nan
    if values.size >= 5:
        residual_center = float(np.median(values))
        deviation = np.abs(values - residual_center)
        mad = float(np.median(deviation))
        sigma = 1.4826 * mad
        if sigma > 1e-12:
            core = values[deviation <= 3.0 * sigma]
            if core.size >= 5:
                mad = float(np.median(np.abs(core - np.median(core))))
                sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 1e-12:
            sigma = np.nan
    return dict(center=center, mad=mad, noise_sigma=sigma, n_samples=float(values.size),
                baseline=baseline, residual=residual, estimator="rolling_median_residual_mad")


def detect_peaks(t, y, prominence, min_height=0.0, min_distance_sec=0.5, auc_half_window_sec=1.0):
    """Detect positive peaks independently per segment and measure in real time.

    Distance suppression prefers higher peaks and compares actual timestamps.
    AUC integrates the detection signal only when the complete requested
    window is observed; otherwise it is NaN. Widths interpolate crossings in
    actual time instead of multiplying sample widths by a global sample rate.
    ``min_height`` may be a scalar (nonpositive disables it) or a same-length
    array of local absolute height thresholds, for example baseline + k sigma.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    height_array = np.asarray(min_height, float)
    if height_array.ndim and height_array.shape != y.shape:
        raise ValueError("Local minimum-height array must match the signal shape.")
    indices, prominences, widths, areas, intervals = [], [], [], [], []
    duration = 0.0
    for segment in continuous_segments(t, y):
        ts, values = t[segment], y[segment]
        duration += float(ts[-1] - ts[0])
        if values.size < 3:
            continue
        height = height_array[segment] if height_array.ndim else (float(height_array) if height_array > 0 else None)
        peaks, props = find_peaks(values, prominence=max(0.0, float(prominence)), height=height)
        # SciPy distance is in samples; enforce seconds for irregular recordings.
        accepted = []
        suppressed = np.zeros(peaks.size, dtype=bool)
        candidate_times = ts[peaks]
        for index in sorted(range(peaks.size), key=lambda i: (-values[peaks[i]], peaks[i])):
            if suppressed[index]:
                continue
            accepted.append(index)
            peak_time = candidate_times[index]
            lo = np.searchsorted(candidate_times, peak_time - min_distance_sec, side="right")
            hi = np.searchsorted(candidate_times, peak_time + min_distance_sec, side="left")
            suppressed[lo:hi] = True
        chosen = np.asarray(sorted(accepted, key=lambda i: peaks[i]), dtype=int)
        peaks = peaks[chosen]
        if not peaks.size:
            continue
        _, _, left, right = peak_widths(values, peaks, rel_height=0.5)
        positions = np.arange(values.size)
        widths.extend((np.interp(right, positions, ts) - np.interp(left, positions, ts)).tolist())
        prominences.extend(props["prominences"][chosen].tolist())
        indices.extend((peaks + segment.start).tolist())
        intervals.extend(np.diff(ts[peaks]).tolist())
        for peak in peaks:
            start, stop = ts[peak] - auc_half_window_sec, ts[peak] + auc_half_window_sec
            integrate = getattr(np, "trapezoid", None) or np.trapz
            complete = start >= ts[0] and stop <= ts[-1]
            if complete and auc_half_window_sec > 0:
                # Interpolate only the exact integration endpoints, within the
                # current observed segment, so sub-sample windows retain width.
                lo = np.searchsorted(ts, start, side="right")
                hi = np.searchsorted(ts, stop, side="left")
                window_time = np.r_[start, ts[lo:hi], stop]
                window_values = np.r_[np.interp(start, ts, values), values[lo:hi], np.interp(stop, ts, values)]
                areas.append(float(integrate(window_values, window_time)))
            else:
                areas.append(np.nan)
    return dict(indices=np.asarray(indices, int), prominences=np.asarray(prominences, float),
                widths_sec=np.asarray(widths, float), auc=np.asarray(areas, float),
                inter_peak_intervals_sec=np.asarray(intervals, float), duration_s=duration)
