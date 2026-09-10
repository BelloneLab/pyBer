"""Recording summaries that preserve removed samples and discontinuous clocks.

Peak amplitude and frequency retain the historical global detector's threshold.
They are descriptive summaries, independent of the Signal Events detector.
"""

from __future__ import annotations

import numpy as np


# This registry is shared by the selection menu, plot labels, and export metadata.
GLOBAL_SIGNAL_METRICS = {
    "amp": {"label": "Mean peak amplitude", "unit": "signal units", "description": "Mean signal value at detected local maxima; zero if no peaks."},
    "freq": {"label": "Transient frequency", "unit": "Hz", "description": "Detected peaks divided by observed time, excluding gaps."},
    "peaks": {"label": "Transient count", "unit": "peaks", "description": "Number of detected local maxima in the selected range."},
    "median": {"label": "Median signal", "unit": "signal units", "description": "Median of finite observed signal samples."},
    "iqr": {"label": "Signal interquartile range", "unit": "signal units", "description": "75th minus 25th percentile of observed signal samples."},
    "std": {"label": "Signal variability", "unit": "signal units", "description": "Sample standard deviation of observed signal samples."},
    "rms": {"label": "Root mean square", "unit": "signal units", "description": "Square root of mean squared signal; includes its baseline offset."},
    "dynamic_range": {"label": "Robust signal range", "unit": "signal units", "description": "95th minus 5th percentile, reducing sensitivity to extreme samples."},
    "auc": {"label": "Integrated signal", "unit": "signal units · s", "description": "Signed trapezoidal signal integral, never bridging missing samples or clock gaps."},
    "ibi": {"label": "Median inter-peak interval", "unit": "s", "description": "Median interval between successive peaks within uninterrupted segments; unavailable without a pair."},
}


def compute_global_signal_metrics(t, y, start_s=0.0, end_s=0.0):
    """Return recording summaries or ``None`` when insufficient data are available.

    A valid increasing start/end pair selects an inclusive range; otherwise the
    full recording is used, preserving the existing UI's 0/0 convention. NaNs
    and clock jumps larger than three median native sample intervals split the
    recording. Time integration, peak neighborhoods, and peak intervals all
    respect those boundaries. Distribution summaries use the finite samples.
    """
    tt = np.asarray(t, dtype=float)
    yy = np.asarray(y, dtype=float)
    if tt.ndim != 1 or yy.ndim != 1 or tt.shape != yy.shape:
        raise ValueError("Time and signal must be aligned one-dimensional arrays.")
    finite_times = tt[np.isfinite(tt)]
    if finite_times.size > 1 and np.any(np.diff(finite_times) <= 0):
        raise ValueError("Time must increase strictly without duplicate samples.")

    # Estimate the native interval before range selection so a short selected
    # range cannot redefine a large recording gap as a normal sample interval.
    dt = np.diff(tt)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    native_dt = float(np.median(positive_dt)) if positive_dt.size else np.nan
    valid = np.isfinite(tt) & np.isfinite(yy)
    if np.isfinite(start_s) and np.isfinite(end_s) and end_s > start_s:
        valid &= (tt >= start_s) & (tt <= end_s)
    indices = np.flatnonzero(valid)
    if indices.size < 3 or not np.isfinite(native_dt):
        return None
    breaks = (np.diff(indices) != 1) | (np.diff(tt[indices]) > 3.0 * native_dt)
    segments = np.split(indices, np.flatnonzero(breaks) + 1)
    values = yy[indices]

    # Preserve the previous detector's threshold exactly, but identify local
    # maxima separately in each observed segment rather than joining cuts.
    local_peaks = [s[1:-1][(yy[s[1:-1]] > yy[s[:-2]]) & (yy[s[1:-1]] > yy[s[2:]])]
                   for s in segments if s.size >= 3]
    peak_indices = np.concatenate(local_peaks) if local_peaks else np.array([], dtype=int)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    excluded = peak_indices[yy[peak_indices] > median + 2.0 * mad]
    threshold_samples = yy[np.setdiff1d(indices, excluded, assume_unique=True)]
    threshold = 3.0 * float(np.median(threshold_samples))
    peaks = peak_indices[yy[peak_indices] >= threshold]

    # Durations and integrals are sums of observed within-segment edges. A
    # singleton sample contributes to the distribution but no elapsed time.
    duration = float(sum(tt[s[-1]] - tt[s[0]] for s in segments if s.size >= 2))
    integral = float(sum(np.sum((yy[s[:-1]] + yy[s[1:]]) * .5 * np.diff(tt[s]))
                         for s in segments if s.size >= 2))
    intervals = []
    for s in segments:
        segment_peaks = peaks[(peaks >= s[0]) & (peaks <= s[-1])]
        intervals.extend(np.diff(tt[segment_peaks]))
    q05, q25, q75, q95 = np.percentile(values, [5, 25, 75, 95])
    return {
        "amp": float(np.mean(yy[peaks])) if peaks.size else 0.0,
        "freq": float(peaks.size / duration) if duration > 0 else 0.0,
        "thr": threshold,
        "peaks": float(peaks.size),
        "duration": duration,
        "median": median,
        "iqr": float(q75 - q25),
        "std": float(np.std(values, ddof=1)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "dynamic_range": float(q95 - q05),
        "auc": integral,
        "ibi": float(np.median(intervals)) if intervals else float("nan"),
    }
