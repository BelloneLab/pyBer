"""Fast, descriptive baseline choices for the inline PSTH settings panel.

These fit scores rank pre-event reference windows; they are not probabilities,
significance tests, or proof of a biologically neutral baseline. Unlike the
separate strict advisor, this helper can show a limited-evidence choice. Every
limitation is retained, and applying a choice does not mask or select trials.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from baseline_advisor import BaselineRecording


@dataclass(frozen=True)
class BaselineSuggestionConfig:
    """Bound the automatic search and expose its reproducible scoring policy."""

    pre_window_s: float = 5.0
    guard_s: float = 0.1
    recovery_s: float = 0.5
    min_duration_s: float = 0.25
    max_sampled_events: int = 32
    max_segment_samples: int = 256


def _merge(intervals):
    """Merge complete bouts without discarding point events or overlapping states."""
    result = []
    for left, right in sorted(np.asarray(intervals, float).reshape(-1, 2).tolist()):
        if result and left <= result[-1][1]:
            result[-1][1] = max(right, result[-1][1])
        else:
            result.append([left, right])
    return np.asarray(result, float).reshape(-1, 2)


def _overlap(intervals, starts, ends):
    """Check all event windows efficiently, including events not sampled for statistics."""
    if not len(intervals):
        return np.zeros(len(starts), bool)
    index = np.searchsorted(intervals[:, 0], ends, side="right") - 1
    return (index >= 0) & (intervals[np.maximum(index, 0), 1] >= starts)


def _sample_indices(count, cap):
    """Use evenly distributed deterministic observations, never favorable trials."""
    return np.unique(np.linspace(0, count - 1, min(count, cap), dtype=int)) if count else np.array([], int)


def _prepare(recording, config):
    """Cache native continuity and all-bout protection without reading post-event values."""
    t, y = np.asarray(recording.time, float), np.asarray(recording.signal, float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 6:
        raise ValueError(f"{recording.label}: at least six aligned signal samples are required.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{recording.label}: times must be finite and strictly increasing.")
    events = np.asarray(recording.events, float).reshape(-1)
    events = np.unique(events[np.isfinite(events)])
    events = events[(events >= t[0]) & (events <= t[-1])]
    raw = np.asarray(recording.exclusion_intervals, float).reshape(-1, 2) if recording.exclusion_intervals is not None else np.empty((0, 2))
    if not np.all(np.isfinite(raw)) or np.any(raw[:, 1] < raw[:, 0]):
        raise ValueError(f"{recording.label}: event bouts require finite onset <= offset.")
    raw = _merge(np.vstack((raw, np.column_stack((events, events)))))
    intervals = _merge(raw + [-config.guard_s, config.recovery_s])
    dt = float(np.median(np.diff(t)))
    finite = np.isfinite(y)
    adjacent = finite[:-1] & finite[1:] & (np.diff(t) <= 3 * dt)
    start = np.flatnonzero(finite & ~np.r_[False, adjacent])
    stop = np.flatnonzero(finite & ~np.r_[adjacent, False])
    prepared = dict(label=str(recording.label), time=t, signal=y, events=events,
                    intervals=intervals, segment_starts=t[start], segment_ends=t[stop],
                    dt=dt, reference_sd=None)
    # A shared pre-event reference assesses scale representativeness. It prevents
    # the scoring rule from rewarding an accidentally quiet, low-SD denominator.
    reference_scales = []
    for index in _sample_indices(len(events), config.max_sampled_events):
        event = events[index]
        left = np.searchsorted(t, event - config.pre_window_s, side="left")
        right = np.searchsorted(t, event - config.guard_s, side="left")
        native = np.arange(left, right)
        native = native[_sample_indices(len(native), config.max_segment_samples)]
        if not native.size:
            continue
        keep = np.isfinite(y[native]) & ~_overlap(intervals, t[native], t[native])
        values = y[native[keep]]
        if len(values) >= 6:
            sd = float(np.std(values))
            if sd > 1e-12:
                reference_scales.append(sd)
    if reference_scales:
        prepared["reference_sd"] = float(np.median(reference_scales))
    return prepared


def _effective_count(values):
    """Estimate information in mean and variance using short, local positive ACFs.

    This is a descriptive approximation on at most 256 native observations. It
    is deliberately capped by the number sampled and is never a confidence level.
    """
    values = np.asarray(values, float)
    n = len(values)
    center = values - np.mean(values)
    sd = float(np.std(center))
    if n < 6 or not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    center /= sd
    counts = []
    for series in (center, center * center):
        series = series - np.mean(series)
        fft = np.fft.rfft(series, n=2 * n)
        covariance = np.fft.irfft(fft * np.conj(fft))[: max(2, n // 4)]
        if covariance[0] <= np.finfo(float).eps:
            counts.append(float(n))
            continue
        rho = covariance / covariance[0]
        positive = rho[1:]
        nonpositive = np.flatnonzero(positive <= 0)
        if nonpositive.size:
            positive = positive[: nonpositive[0]]
        tau = max(1.0, 1 + 2 * float(np.sum(positive)))
        counts.append(float(np.clip(n / tau, 1, n)))
    return min(counts)


def _window_statistics(t, y):
    """Measure information, distribution shape and scale stability within one window."""
    sd = float(np.std(y))
    if len(y) < 6 or not np.isfinite(sd) or sd <= 1e-12:
        return None
    half = len(y) // 2
    left, right = y[:half], y[half:]
    split_shift = abs(float(np.mean(left) - np.mean(right))) / sd
    halves = [float(np.std(left)), float(np.std(right))]
    split_ratio = max(halves) / max(min(halves), sd * 1e-6)
    robust_sd = float(1.4826 * np.median(np.abs(y - np.median(y))))
    shape_ratio = sd / max(robust_sd, sd * 1e-6)
    time_center = t - np.mean(t)
    correlation = abs(float(np.dot(time_center, y - np.mean(y)))) / max(
        float(np.linalg.norm(time_center)) * sd * np.sqrt(len(y)), np.finfo(float).tiny
    )
    return dict(sd=sd, effective_samples=_effective_count(y), shift=split_shift,
                split_ratio=split_ratio, shape_ratio=shape_ratio,
                monotonic=bool(correlation > 0.95 and split_shift > 1.5))


def _batch_window_statistics(samples):
    """Evaluate equal-length windows together, preserving the scalar formulas.

    Grouping by native sample count avoids padding or interpolation. FFTs and
    robust reductions operate along each window independently; no samples from
    separate events are joined. This bounds GUI latency without reducing the
    number of events inspected or weakening the scoring criteria.
    """
    groups = {}
    for t, y in samples:
        groups.setdefault(len(y), []).append((t, y))
    results = []
    for n, windows in groups.items():
        if n < 6:
            continue
        values = np.stack([row[1] for row in windows])
        times = np.stack([row[0] for row in windows])
        sd = np.std(values, axis=1)
        valid = np.isfinite(sd) & (sd > 1e-12)
        values, times, sd = values[valid], times[valid], sd[valid]
        if not len(sd):
            continue
        centered = values - values.mean(axis=1, keepdims=True)
        standardized = centered / sd[:, None]
        # Value and squared-value dependence both constrain effective count.
        series = np.stack((standardized, standardized ** 2), axis=1)
        series -= series.mean(axis=2, keepdims=True)
        transform = np.fft.rfft(series, n=2 * n, axis=2)
        covariance = np.fft.irfft(transform * transform.conj(), axis=2)[..., :max(2, n // 4)]
        variance = covariance[..., :1]
        rho = np.divide(covariance, variance, out=np.zeros_like(covariance),
                        where=variance > np.finfo(float).eps)
        positive = rho[..., 1:]
        # Keep the initial positive run, exactly as the scalar estimator does.
        prefix = np.logical_and.accumulate(positive > 0, axis=2)
        tau = np.maximum(1., 1 + 2 * np.sum(np.where(prefix, positive, 0.), axis=2))
        effective = np.min(np.clip(n / tau, 1, n), axis=1)
        half = n // 2
        left, right = values[:, :half], values[:, half:]
        shifts = np.abs(left.mean(axis=1) - right.mean(axis=1)) / sd
        halves = np.stack((left.std(axis=1), right.std(axis=1)), axis=1)
        ratios = halves.max(axis=1) / np.maximum(halves.min(axis=1), sd * 1e-6)
        robust_sd = 1.4826 * np.median(np.abs(values - np.median(values, axis=1)[:, None]), axis=1)
        shapes = sd / np.maximum(robust_sd, sd * 1e-6)
        time_center = times - times.mean(axis=1, keepdims=True)
        correlation = np.abs(np.sum(time_center * centered, axis=1)) / np.maximum(
            np.linalg.norm(time_center, axis=1) * sd * np.sqrt(n), np.finfo(float).tiny)
        for i in range(len(sd)):
            results.append(dict(sd=float(sd[i]), effective_samples=float(effective[i]),
                                shift=float(shifts[i]), split_ratio=float(ratios[i]),
                                shape_ratio=float(shapes[i]),
                                monotonic=bool(correlation[i] > .95 and shifts[i] > 1.5)))
    return results


def _evaluate_recording(recording, window, config):
    """Check every target for data/bout coverage, then sample signal quality evenly."""
    t, y, events = recording["time"], recording["signal"], recording["events"]
    start, end = events + window[0], events + window[1]
    overlap = _overlap(recording["intervals"], start, end)
    left, right = np.searchsorted(t, start), np.searchsorted(t, end, side="right")
    segments = recording["segment_starts"]
    segment_index = np.searchsorted(segments, start, side="right") - 1
    observed = np.zeros(len(events), bool)
    if len(segments):
        observed = (segment_index >= 0) & (recording["segment_ends"][np.maximum(segment_index, 0)] >= end) & (right - left >= 6)
    # Earlier-event responses must not determine whether a baseline's signal
    # distribution looks attractive. Their full overlap count remains visible
    # in the score, while only event-free windows supply quality statistics.
    valid_events = np.flatnonzero(observed & ~overlap)
    samples = []
    for position in _sample_indices(len(valid_events), config.max_sampled_events):
        index = valid_events[position]
        native = left[index] + _sample_indices(right[index] - left[index], config.max_segment_samples)
        samples.append((t[native], y[native]))
    statistics = _batch_window_statistics(samples)
    count = len(events)
    row = dict(label=recording["label"], events=count, observed_events=int(np.sum(observed)),
               event_free_observed_events=int(np.sum(observed & ~overlap)),
               observed_coverage=float(np.mean(observed)) if count else 0.0,
               event_free_fraction=float(np.mean(~overlap)) if count else 0.0,
               overlap_events=int(np.sum(overlap)), sampled_events=len(statistics),
               reference_sd=recording["reference_sd"], unusable=not statistics)
    if not statistics:
        return row
    sd = np.array([stat["sd"] for stat in statistics])
    candidate_scale = float(np.median(sd))
    reference = recording["reference_sd"]
    scale_ratio = max(candidate_scale / reference, reference / candidate_scale) if reference else 1.0
    p10, p90 = np.quantile(sd, [.1, .9])
    row.update(
        baseline_sd=candidate_scale, scale_reference_ratio=float(scale_ratio),
        between_event_scale_ratio=float(p90 / max(p10, candidate_scale * 1e-6)),
        effective_samples=float(np.quantile([stat["effective_samples"] for stat in statistics], .25)),
        split_shift=float(np.median([stat["shift"] for stat in statistics])),
        split_scale_ratio=float(np.median([stat["split_ratio"] for stat in statistics])),
        shape_ratio=float(np.median([stat["shape_ratio"] for stat in statistics])),
        monotonic_fraction=float(np.mean([stat["monotonic"] for stat in statistics])),
    )
    return row


def _candidate_windows(config):
    """Search distinct durations and offsets wholly inside the displayed pre-event span."""
    pre = min(60.0, config.pre_window_s)
    # A strictly negative endpoint prevents a sample at time zero entering the
    # baseline. Centisecond rounding matches the editable PSTH input controls.
    guard = max(0.01, config.guard_s + 0.01)
    available = pre - guard
    if available < config.min_duration_s:
        return []
    durations = {config.min_duration_s, available}
    durations.update(available * fraction for fraction in (.18, .30, .45, .65, .82))
    durations.update(value for value in (.5, 1., 2., 3.) if value < available)
    ends = {guard, max(guard, .10 * pre), max(guard, .25 * pre), max(guard, .45 * pre)}
    windows = set()
    for duration in durations:
        if duration < config.min_duration_s:
            continue
        for end in ends:
            if duration + end <= pre + 1e-9:
                start_rounded = round(-duration - end, 2)
                end_rounded = round(-end, 2)
                if start_rounded >= -pre and start_rounded < end_rounded < 0:
                    windows.add((start_rounded, end_rounded))
    return sorted(windows)


def _score(rows, window, config):
    """Combine explanatory fit terms without optimizing response size or low variance."""
    usable = [row for row in rows if not row["unusable"]]
    if not usable:
        return None
    # A near-deterministic slope cannot supply an estimate of baseline noise.
    if all(row["monotonic_fraction"] >= .8 for row in usable):
        return None
    per_file_scores = []
    for row in rows:
        if row["unusable"]:
            per_file_scores.append(0.)
            continue
        info = min(1., row["effective_samples"] / 20)
        drift = 1 / (1 + (row["split_shift"] / .9) ** 2)
        scale = float(np.exp(-(np.log(row["scale_reference_ratio"]) / np.log(3)) ** 2))
        stable = 1 / (1 + (np.log(max(row["between_event_scale_ratio"], row["split_scale_ratio"])) / np.log(3)) ** 2)
        distribution = min(1., 2. / max(1., row["shape_ratio"]))
        recency = max(0., 1 - abs(window[1]) / config.pre_window_s)
        score = 100 * (.22 * info + .17 * drift + .17 * scale + .12 * stable +
                       .08 * distribution + .19 * row["observed_coverage"] + .05 * recency)
        per_file_scores.append(score)
    value = .6 * float(np.mean(per_file_scores)) + .4 * min(per_file_scores)
    coverage = min(row["observed_coverage"] for row in rows)
    clean = min(row["event_free_fraction"] for row in rows)
    information = min(row["effective_samples"] for row in usable)
    evidence = min(row["sampled_events"] for row in rows)
    cautions = []
    if clean < 1:
        value *= .5 + .5 * clean
        value = min(value, 69 if clean > .75 else 49 if clean > .4 else 35)
        cautions.append(f"Earlier bout overlap: {100 * (1 - clean):.0f}% in the most affected file")
    if coverage < 1:
        value *= .5 + .5 * coverage
        value = min(value, 69 if coverage >= .8 else 49)
        cautions.append(f"Complete data coverage: {100 * coverage:.0f}% in the least covered file")
    if evidence < 8:
        value = min(value, 55 if evidence < 4 else 69)
        cautions.append(f"Limited event evidence: {evidence} sampled usable events in the smallest file")
    if information < 20:
        value = min(value, 69)
        cautions.append("Slow signal or limited effective information")
    if any(row.get("scale_reference_ratio", 1) > 3 for row in usable):
        value = min(value, 69)
        cautions.append("Short-window scale differs from the wider pre-event reference")
    if any(row["reference_sd"] is None for row in rows):
        value = min(value, 69)
        cautions.append("No clean pre-event scale reference in at least one file")
    if any(row.get("split_shift", 0) > 1.25 for row in usable):
        value = min(value, 49)
        cautions.append("Strong trend within the baseline")
    if any(row.get("between_event_scale_ratio", 0) > 4 for row in usable):
        value = min(value, 69)
        cautions.append("Baseline scale varies across events")
    score = int(np.clip(round(value), 0, 100))
    quality = "Supported" if score >= 70 and not cautions else "Limited"
    summary = "; ".join(cautions) if cautions else "Good pre-event coverage, information and scale stability"
    return dict(start=window[0], end=window[1], window=list(window), score=score,
                quality=quality, summary=summary,
                diagnostics=dict(observed_coverage=coverage, event_free_fraction=clean,
                                 effective_samples=information, sampled_event_evidence=evidence,
                                 per_recording=rows, cautions=cautions))


def suggest_baselines(recordings, config=None):
    """Return up to three bounded-cost choices without changing any analysis inputs.

    Coverage and bout overlap inspect every selected event. Signal statistics use
    evenly spaced events and bounded native samples, retaining all loaded files.
    A limited choice exposes its limitations instead of hiding affected trials.
    """
    config = config or BaselineSuggestionConfig()
    values = asdict(config)
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Suggestion settings must be finite.")
    if not 0 < config.pre_window_s <= 60 or min(config.guard_s, config.recovery_s) < 0:
        raise ValueError("Use a positive pre-event span up to 60 s and nonnegative protection.")
    if config.min_duration_s <= 0 or config.max_sampled_events < 1 or config.max_segment_samples < 6:
        raise ValueError("Positive duration, event count and at least six sampled points are required.")
    prepared = [_prepare(recording, config) for recording in recordings]
    result = dict(status="unavailable", choices=[], summary="Load a signal and select events for baseline suggestions.",
                  config=values, evaluated_candidates=0,
                  recordings=[dict(label=row["label"], events=len(row["events"])) for row in prepared])
    if not prepared or not any(len(row["events"]) for row in prepared):
        return result
    candidates = []
    observed_windows, clean_windows = 0, 0
    for window in _candidate_windows(config):
        rows = [_evaluate_recording(recording, window, config) for recording in prepared]
        observed_windows += sum(row["observed_events"] for row in rows)
        clean_windows += sum(row["event_free_observed_events"] for row in rows)
        candidate = _score(rows, window, config)
        result["evaluated_candidates"] += 1
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda row: (row["score"], row["diagnostics"]["event_free_fraction"], row["end"]), reverse=True)
    choices = []
    for candidate in candidates:
        left, right = candidate["window"]
        distinct = True
        for chosen in choices:
            start, end = chosen["window"]
            intersection = max(0., min(right, end) - max(left, start))
            union = max(right, end) - min(left, start)
            if intersection / union > .8:
                distinct = False
                break
        if distinct:
            choices.append(candidate)
        if len(choices) == 3:
            break
    if not choices:
        result["summary"] = (
            "No event-free baseline in this pre-event span. Increase Pre or review event grouping."
            if observed_windows and not clean_windows else
            "No usable pre-event baseline: signal is flat, strongly drifting, missing or too short."
        )
        return result
    result.update(status="ready", choices=choices,
                  summary="Fit scores are not confidence probabilities. " +
                          ("Limited choices need review before applying." if any(choice["quality"] == "Limited" for choice in choices)
                           else "Click a window to apply it."))
    return result
