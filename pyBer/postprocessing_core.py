"""Numerical postprocessing helpers independent of the graphical interface.

Missing observations remain missing. Event arrays are always transformed together
so that an onset, offset and duration cannot silently become unrelated.
"""

from typing import Dict, Optional, Tuple

import numpy as np


# Explicit numerical policies shared by the GUI and reproducible tests.
MIN_BASELINE_SAMPLES = 5
MIN_WINDOW_SAMPLES = 5
MAX_INTERPOLATION_GAP_FACTOR = 3.0
BASELINE_STD_EPSILON = 1e-12


def _time_signal(time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate a sampled recording without sorting or altering observations."""
    t, y = np.asarray(time, float), np.asarray(signal, float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("Time and signal must be one-dimensional arrays of equal length.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Recording times must be finite and strictly increasing (no duplicates).")
    return t, y


def normalize_events(
    onsets: np.ndarray,
    offsets: Optional[np.ndarray] = None,
    durations: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort and deduplicate event rows while preserving their associations.

Rows lacking an onset are removed. An unavailable offset stays NaN, rather
than borrowing the next event's offset. Point events have unknown duration.
Complete onset/offset pairs define their own duration, ensuring consistency.
"""
    on = np.asarray(onsets, float).reshape(-1)
    off = on.copy() if offsets is None else np.asarray(offsets, float).reshape(-1)
    dur = np.full(on.shape, np.nan) if durations is None else np.asarray(durations, float).reshape(-1)
    if off.size != on.size:
        raise ValueError("Event onset and offset arrays must have equal length.")
    if dur.size not in (0, on.size):
        raise ValueError("Event durations must match the number of onsets.")
    if dur.size == 0:
        dur = np.full(on.shape, np.nan)
    keep = np.isfinite(on)
    on, off, dur = on[keep], off[keep].copy(), dur[keep].copy()
    invalid_offset = ~np.isfinite(off) | (off < on)
    off[invalid_offset] = np.nan
    complete = np.isfinite(off) & (off > on)
    dur[complete] = off[complete] - on[complete]
    dur[~np.isfinite(dur) | (dur < 0) | invalid_offset] = np.nan
    order = np.argsort(on, kind="stable")
    on, off, dur = on[order], off[order], dur[order]
    unique = np.r_[True, np.diff(on) != 0] if on.size else np.array([], bool)
    return on[unique], off[unique], dur[unique]


def extract_complete_events(
    time: np.ndarray, signal: np.ndarray, threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract observed low-to-high-to-low bouts using first-low offsets.

    Boundary-truncated bouts and bouts interrupted by missing samples or large
    timestamp gaps are excluded because neither has a fully observed duration.
    This retains the binary-input boundary policy and avoids inventing an
    offset at a NaN sample.
"""
    t, x = _time_signal(time, signal)
    if not np.isfinite(threshold):
        raise ValueError("The event threshold must be finite.")
    if t.size < 2:
        empty = np.array([], float)
        return empty.copy(), empty.copy(), empty.copy()
    finite = np.isfinite(x)
    high = finite & (x > threshold)
    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    ends = np.flatnonzero(high & ~np.r_[high[1:], False]) + 1
    # A complete bout needs observed low samples immediately on both sides.
    complete = (starts > 0) & (ends < t.size)
    starts, ends = starts[complete], ends[complete]
    observed = finite[starts - 1] & finite[ends]
    intervals = np.diff(t)
    gap_limit = MAX_INTERPOLATION_GAP_FACTOR * float(np.median(intervals))
    gap_counts = np.r_[0, np.cumsum(intervals > gap_limit)]
    observed &= (gap_counts[ends] - gap_counts[starts - 1]) == 0
    on, off = t[starts[observed]], t[ends[observed]]
    return on, off, off - on


def group_close_events(
    times: np.ndarray, durations: np.ndarray, window_s: float,
    alignment: str = "onset",
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge chains of nearby alignment times with correct bout boundaries.

Onset alignment retains the first onset; offset alignment retains the last
offset. Merged duration spans the earliest onset to the latest offset. If a
duration is unavailable, the merged duration remains unknown.
"""
    t = np.asarray(times, float).reshape(-1)
    d = np.asarray(durations, float).reshape(-1)
    if d.size != t.size:
        d = np.full(t.shape, np.nan)
    if alignment not in {"onset", "offset"}:
        raise ValueError("Event grouping alignment must be onset or offset.")
    if not np.isfinite(window_s) or window_s < 0:
        raise ValueError("The grouping window must be finite and nonnegative.")
    finite = np.isfinite(t)
    t, d = t[finite], d[finite]
    order = np.argsort(t, kind="stable")
    t, d = t[order], d[order]
    if t.size < 2 or window_s == 0:
        return t, d
    boundaries = np.r_[0, np.flatnonzero(np.diff(t) > window_s) + 1, t.size]
    grouped_t, grouped_d = [], []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        tt, dd = t[start:stop], d[start:stop]
        grouped_t.append(tt[-1] if alignment == "offset" else tt[0])
        if not np.all(np.isfinite(dd) & (dd >= 0)):
            grouped_d.append(np.nan)
        elif alignment == "offset":
            grouped_d.append(float(tt[-1] - np.min(tt - dd)))
        else:
            grouped_d.append(float(np.max(tt + dd) - tt[0]))
    return np.asarray(grouped_t), np.asarray(grouped_d)


def mean_sem(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite mean, sample SEM and contributing count at each time bin.

SEM is undefined for fewer than two observations and is represented by NaN.
No warning is emitted for empty columns or an empty matrix.
"""
    values = np.asarray(matrix, float)
    if values.ndim != 2:
        raise ValueError("PSTH summary requires a two-dimensional matrix.")
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    means = np.divide(np.where(valid, values, 0.0).sum(axis=0), counts,
                      out=np.full(values.shape[1], np.nan), where=counts > 0)
    deviations = np.where(valid, values - means, 0.0)
    variance = np.divide(np.sum(deviations ** 2, axis=0), counts - 1,
                         out=np.full(means.shape, np.nan), where=counts > 1)
    sem = np.sqrt(np.divide(variance, counts, out=np.full(means.shape, np.nan), where=counts > 1))
    return means, sem, counts


def compute_psth_matrix(
    t: np.ndarray, y: np.ndarray, event_times: np.ndarray,
    window: Tuple[float, float], baseline_win: Tuple[float, float],
    resample_hz: float, smooth_sigma_s: float = 0.0,
    normalization: str = "zscore",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute baseline-standardized trials without extrapolating missing data.

    At least five finite source samples are required for baseline and output.
    ``normalization`` selects baseline z-score, baseline subtraction, or none.
    A flat baseline cannot define a z-score, so that trial remains unavailable.
Interpolation spans adjacent observations only, never NaNs or timestamp gaps
longer than three median sample intervals. Gaussian smoothing is performed
separately within each observed segment, preserving all missing bins.
"""
    t, y = _time_signal(t, y)
    if not np.isfinite(resample_hz) or resample_hz <= 0:
        raise ValueError("PSTH sampling rate must be finite and greater than zero.")
    if not np.isfinite(smooth_sigma_s) or smooth_sigma_s < 0:
        raise ValueError("Smoothing must be finite and nonnegative.")
    if normalization not in {"zscore", "subtract", "none"}:
        raise ValueError("PSTH normalization must be zscore, subtract, or none.")
    if len(window) != 2 or not np.all(np.isfinite(window)) or window[1] <= window[0]:
        raise ValueError("The PSTH window must have a finite start earlier than its end.")
    if normalization != "none" and (len(baseline_win) != 2 or not np.all(np.isfinite(baseline_win)) or baseline_win[1] <= baseline_win[0]):
        raise ValueError("The baseline window must have a finite start earlier than its end.")
    events = np.asarray(event_times, float).reshape(-1)
    events = events[np.isfinite(events)]
    if events.size == 0:
        return np.array([], float), np.zeros((0, 0), float)
    dt = 1.0 / resample_hz
    n_samples = int(np.floor((window[1] - window[0]) * resample_hz + 1e-9)) + 1
    relative = window[0] + np.arange(n_samples) * dt
    matrix = np.full((events.size, n_samples), np.nan)
    if t.size < 2:
        return relative, matrix
    finite_y = np.where(np.isfinite(y), y, np.nan)
    gap_limit = MAX_INTERPOLATION_GAP_FACTOR * float(np.median(np.diff(t)))
    for row, event in enumerate(events):
        baseline_mean, baseline_std = 0.0, 1.0
        if normalization != "none":
            b0 = np.searchsorted(t, event + baseline_win[0], side="left")
            b1 = np.searchsorted(t, event + baseline_win[1], side="right")
            baseline = finite_y[b0:b1]
            baseline = baseline[np.isfinite(baseline)]
            if baseline.size < MIN_BASELINE_SAMPLES:
                continue
            baseline_mean = float(baseline.mean())
            if normalization == "zscore":
                baseline_std = float(baseline.std())
                if baseline_std <= BASELINE_STD_EPSILON:
                    continue
        w0 = np.searchsorted(t, event + window[0], side="left")
        w1 = np.searchsorted(t, event + window[1], side="right")
        if np.count_nonzero(np.isfinite(finite_y[w0:w1])) < MIN_WINDOW_SAMPLES:
            continue
        query = event + relative
        interpolated = np.interp(query, t, finite_y, left=np.nan, right=np.nan)
        # Exact observations remain usable even at the edge of a large gap.
        right = np.searchsorted(t, query, side="left")
        interior = (right > 0) & (right < t.size)
        idx = np.flatnonzero(interior)
        r = right[idx]
        spans_gap = (t[r] - t[r - 1] > gap_limit) & (query[idx] != t[r])
        interpolated[idx[spans_gap]] = np.nan
        matrix[row] = (interpolated - baseline_mean) / baseline_std
    if smooth_sigma_s > 0:
        from scipy.ndimage import gaussian_filter1d
        sigma = smooth_sigma_s * resample_hz
        for row in matrix:
            valid = np.isfinite(row)
            starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1]])
            stops = np.flatnonzero(valid & ~np.r_[valid[1:], False]) + 1
            for start, stop in zip(starts, stops):
                row[start:stop] = gaussian_filter1d(row[start:stop], sigma=sigma, mode="nearest")
    return relative, matrix


def window_metrics(
    matrix: np.ndarray, time: np.ndarray, start: float, stop: float,
    metric: str = "mean",
) -> np.ndarray:
    """Reduce trial windows consistently for screen and exported metrics.

Mean uses available finite bins. AUC uses the trapezoidal integral with exact
window endpoints interpolated from adjacent bins; incomplete coverage or a
missing observation leaves that trial's AUC undefined. Thus a missing section
cannot silently be integrated as zero or stretched over the full window.
"""
    values, t = np.asarray(matrix, float), np.asarray(time, float)
    if values.ndim != 2 or t.ndim != 1 or values.shape[1] != t.size:
        raise ValueError("Metric matrix columns must match the time vector.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Metric times must be finite and strictly increasing.")
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("Metric window start must be earlier than its end.")
    if metric.lower() not in {"mean", "auc"}:
        raise ValueError("Metric must be mean or auc.")
    result = np.full(values.shape[0], np.nan)
    if t.size == 0:
        return result
    mask = (t >= start) & (t <= stop)
    if metric.lower() == "mean":
        selected = values[:, mask]
        valid = np.isfinite(selected)
        counts = valid.sum(axis=1)
        return np.divide(np.where(valid, selected, 0.0).sum(axis=1), counts,
                         out=result, where=counts > 0)
    if start < t[0] or stop > t[-1]:
        return result
    integration_times = np.r_[start, t[(t > start) & (t < stop)], stop]
    for idx, row in enumerate(values):
        samples = np.interp(integration_times, t, np.where(np.isfinite(row), row, np.nan))
        if np.all(np.isfinite(samples)):
            result[idx] = np.sum((samples[:-1] + samples[1:]) * 0.5 * np.diff(integration_times))
    return result


def paired_summary(
    pre: np.ndarray, post: np.ndarray, independent_units: bool = True,
) -> Dict[str, object]:
    """Summarize a two-sided exact paired sign test with explicit assumptions.

Only finite pairs are used. Zero differences are ties and do not enter the
binomial test. Under the null, positive and negative nonzero differences are
equally likely across independent units. This does not assume normality or
symmetric difference magnitudes, but ignores effect magnitude and may have
low power. Repeated trials pooled across animals must not be declared
independent; callers can suppress inference with ``independent_units=False``.
"""
    before, after = np.asarray(pre, float).reshape(-1), np.asarray(post, float).reshape(-1)
    if before.size != after.size:
        raise ValueError("Paired metrics must have equal numbers of pre and post values.")
    paired = np.isfinite(before) & np.isfinite(after)
    before, after = before[paired], after[paired]
    positive = int(np.count_nonzero(after > before))
    negative = int(np.count_nonzero(after < before))
    nonzero = positive + negative
    count = int(before.size)
    result: Dict[str, object] = {
        "paired_n": count,
        "paired_nonzero_n": nonzero,
        "paired_ties_n": count - nonzero,
        "paired_positive_n": positive,
        "paired_negative_n": negative,
        "paired_p": float("nan"),
        "method": "Exact paired sign test (two-sided)",
        "normality_p": float("nan"),
        "assumption_note": (
            "Assumes independent paired units and equal probabilities of positive and "
            "negative differences under the null. No normality assumption; exact ties "
            "are excluded. Effect magnitude is not tested."
        ),
    }
    if not independent_units:
        result["method"] = "Descriptive only"
        result["assumption_note"] = (
            "Inference is suppressed because pooled trials are not independent animal-level "
            "units. Use one summary per independent animal for group inference."
        )
        return result
    if count == 0:
        result["assumption_note"] = "No finite paired observations are available."
        return result
    if nonzero == 0:
        result["paired_p"] = 1.0
        result["assumption_note"] += " All observed differences are zero; there is no directional evidence."
        return result
    from scipy.stats import binomtest
    result["paired_p"] = float(binomtest(positive, nonzero, p=0.5, alternative="two-sided").pvalue)
    if nonzero < 6:
        result["assumption_note"] += " Fewer than six nonzero pairs cannot reach p < 0.05 in this two-sided test."
    return result
