"""Conservative, baseline-only recommendations for event-aligned normalization.

The advisor ranks reference windows, not neural responses. Its thresholds are
explicit engineering heuristics, not a test of biological independence or a
guarantee of unbiased inference. Source samples and PSTH inclusion stay intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class BaselineAdvisorConfig:
    """User-tunable search limits and reproducible reliability policies."""

    lookback_s: float = 30.0
    event_guard_s: float = 0.25
    recovery_s: float = 1.0
    min_window_s: float = 1.0
    min_samples: int = 20
    min_effective_samples: float = 20.0
    min_coverage: float = 0.8
    max_split_shift: float = 1.0
    max_split_scale_ratio: float = 3.0
    max_between_trial_sd_ratio: float = 4.0
    min_train_events: int = 8
    min_validation_events: int = 4
    max_events_per_split: int = 100
    max_acf_rate_hz: float = 100.0


@dataclass(frozen=True)
class BaselineRecording:
    """Native processed signal, analysis events and all known excluded bouts.

    ``events`` are the alignment timestamps used for the planned PSTH.
    ``exclusion_intervals`` are absolute onset/offset pairs, including source
    events omitted by PSTH filters. Unknown durations can be point intervals.
    """

    label: str
    time: np.ndarray
    signal: np.ndarray
    events: np.ndarray
    exclusion_intervals: np.ndarray | None = None


def _validate_config(config):
    """Reject impossible policies before inspecting user recordings."""
    values = asdict(config)
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Baseline advisor settings must be finite.")
    if not 0 < config.min_window_s <= config.lookback_s <= 600:
        raise ValueError("Choose a positive minimum duration and a search limit up to 600 s.")
    if min(config.event_guard_s, config.recovery_s) < 0:
        raise ValueError("Event guards cannot be negative.")
    if not 0 < config.min_coverage <= 1:
        raise ValueError("Coverage must be between zero and one.")
    if min(config.min_samples, config.min_train_events, config.min_validation_events) < 4:
        raise ValueError("At least four samples/events are required by each minimum.")
    if config.max_events_per_split < max(config.min_train_events, config.min_validation_events):
        raise ValueError("The event cap must exceed the minimum event counts.")
    if min(config.min_effective_samples, config.max_acf_rate_hz, config.max_split_shift) <= 0:
        raise ValueError("Information, analysis rate and stability limits must be positive.")
    if min(config.max_split_scale_ratio, config.max_between_trial_sd_ratio) < 1:
        raise ValueError("Scale ratios must be at least one.")


def _merge_intervals(intervals):
    """Merge overlapping exclusions so interval lookups stay deterministic."""
    merged = []
    for start, end in sorted(intervals.tolist()):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return np.asarray(merged, float).reshape(-1, 2)


def _overlaps(intervals, start, end):
    """Test closed intervals, including instantaneous event markers."""
    if not len(intervals):
        return False
    index = np.searchsorted(intervals[:, 0], end, side="right") - 1
    return index >= 0 and intervals[index, 1] >= start


def _thin(events, count):
    """Cap computation with deterministic, evenly spaced event selection."""
    if len(events) <= count:
        return events
    return events[np.linspace(0, len(events) - 1, count, dtype=int)]


def _quantile(values, q, fallback=0.0):
    """Return a finite scalar suitable for portable report serialization."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float(fallback)


def _prepare(recording, config):
    """Validate clocks and create nonoverlapping chronological evaluation sets."""
    t, y = np.asarray(recording.time, float), np.asarray(recording.signal, float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < config.min_samples:
        raise ValueError(f"{recording.label}: insufficient one-dimensional signal samples.")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{recording.label}: timestamps must be finite and strictly increasing.")
    events = np.asarray(recording.events, float).reshape(-1)
    events = np.unique(events[np.isfinite(events)])
    events = events[(events >= t[0]) & (events <= t[-1])]
    if recording.exclusion_intervals is None:
        raw = np.column_stack((events, events))
    else:
        raw = np.asarray(recording.exclusion_intervals, float).reshape(-1, 2)
        if not np.all(np.isfinite(raw)) or np.any(raw[:, 1] < raw[:, 0]):
            raise ValueError(f"{recording.label}: exclusions require finite onset <= offset pairs.")
    # Partial exclusion metadata must never erase a known alignment marker.
    raw = _merge_intervals(np.vstack((raw, np.column_stack((events, events)))))
    excluded = _merge_intervals(raw + np.array([-config.event_guard_s, config.recovery_s]))
    split = max(1, int(np.floor(len(events) * 2 / 3)))
    train = events[:split]
    boundary = train[-1] if len(train) else t[0]
    # Every validation search interval begins after the last training event.
    # Purging the full lookback avoids shared native samples across the split.
    validation = events[split:]
    validation = validation[validation - config.lookback_s > boundary]
    dt = float(np.median(np.diff(t)))
    return {"label": str(recording.label), "time": t, "signal": y, "events": events,
            "excluded": excluded, "raw_intervals": raw, "dt": dt,
            "train": _thin(train, config.max_events_per_split),
            "validation": _thin(validation, config.max_events_per_split),
            "train_total": len(train), "validation_total": len(validation), "split_boundary": boundary,
            "purged_events": len(events) - len(train) - len(validation)}


def _context(recording, events, config, window=None):
    """Estimate information timescales from contiguous, eligible baseline data.

    Autocorrelation of both values and centered squared values is used. The
    latter responds to clustered variance and is relevant to SD estimation.
    Initial-positive ACF sums are descriptive approximations, not confidence
    intervals. Binning limits work at high acquisition rates without treating
    interpolated samples as new observations.
    """
    t, y, dt = recording["time"], recording["signal"], recording["dt"]
    selected = np.zeros(len(t), bool)
    window = window or (-config.lookback_s, -config.event_guard_s)
    for event in events:
        lo = np.searchsorted(t, event + window[0])
        hi = np.searchsorted(t, event + window[1], side="right")
        selected[lo:hi] = True
    for start, end in recording["excluded"]:
        selected[np.searchsorted(t, start):np.searchsorted(t, end, side="right")] = False
    selected &= np.isfinite(y)
    amplitude = max(float(np.max(np.abs(y[selected]))) if np.any(selected) else 0.0,
                    np.finfo(float).tiny)
    breaks = np.r_[True, np.diff(t) > 3 * dt]
    starts = np.flatnonzero(selected & (~np.r_[False, selected[:-1]] | breaks))
    stops = np.flatnonzero(selected & (~np.r_[selected[1:], False] | np.r_[breaks[1:], True])) + 1
    stride = max(1, int(np.ceil(1 / (dt * config.max_acf_rate_hz))))
    analysis_dt = dt * stride
    max_lag = max(2, int(min(config.lookback_s / 2, 10.0) / analysis_dt))
    covariance = np.zeros((2, max_lag + 1))
    counts = np.zeros(max_lag + 1)
    kept = []
    for start, stop in zip(starts, stops):
        # Scale before squaring/FFT to avoid fourth-power overflow/underflow.
        part = y[start:stop] / amplitude
        n = len(part) // stride
        if n < 8:
            continue
        # Bin means are only formed inside a contiguous observed segment.
        part = part[:n * stride].reshape(n, stride).mean(axis=1)
        kept.append(part)
        centered = part - np.mean(part)
        squared = centered ** 2
        squared -= squared.mean()
        lag_count = min(n // 2, max_lag) + 1
        counts[:lag_count] += np.arange(n, n - lag_count, -1)
        for axis, vector in enumerate((centered, squared)):
            covariance[axis, :lag_count] += fftconvolve(vector, vector[::-1], mode="full")[n - 1:n - 1 + lag_count]
    taus = []
    truncated = False
    for vector in covariance:
        if vector[0] <= 0 or counts[0] == 0:
            taus.append(config.lookback_s)
            truncated = True
            continue
        acf = np.divide(vector, counts, out=np.zeros_like(vector), where=counts > 0)
        acf /= acf[0]
        positive = []
        for lag in range(1, len(acf)):
            if counts[lag] < 20:
                truncated = True
                break
            if acf[lag] <= 0:
                break
            positive.append(float(acf[lag]))
        else:
            truncated = True
        taus.append(max(analysis_dt, analysis_dt * (1 + 2 * sum(positive))))
    values = np.concatenate(kept) if kept else np.array([], float)
    if values.size and np.std(values) > 0:
        standardized = (values - values.mean()) / values.std()
        skew = float(np.mean(standardized ** 3))
        tail = float(np.mean(np.abs(standardized) > 3))
    else:
        skew, tail = 0.0, 0.0
    return {"mean_information_s": taus[0], "variance_information_s": taus[1],
            "information_s": max(taus), "acf_truncated": truncated,
            "context_samples": int(values.size), "skewness": skew, "tail_fraction": tail}


def _evaluate(recording, events, context, window, config):
    """Measure reliability without evaluating the post-event response."""
    t, y, dt = recording["time"], recording["signal"], recording["dt"]
    # A long search context can hide a locally smooth candidate in noisy data.
    # Use the slower of local and broad reference timescales for eligibility.
    local = _context(recording, events, config, window)
    information_s = max(context["information_s"], local["information_s"])
    shifts, scale_ratios, scales, effective = [], [], [], []
    rejected = {"event_overlap": 0, "missing_or_cut": 0, "flat": 0}
    for event in events:
        start, end = event + window[0], event + window[1]
        if _overlaps(recording["excluded"], start, end):
            rejected["event_overlap"] += 1
            continue
        lo, hi = np.searchsorted(t, start), np.searchsorted(t, end, side="right")
        part, times = y[lo:hi], t[lo:hi]
        if (start < t[0] or end > t[-1] or len(part) < config.min_samples
                or not np.all(np.isfinite(part)) or np.any(np.diff(times) > 3 * dt)
                or times[-1] - times[0] < .95 * (end - start)):
            rejected["missing_or_cut"] += 1
            continue
        amplitude = max(float(np.max(np.abs(part))), np.finfo(float).tiny)
        part = part / amplitude
        sd = float(np.std(part))
        # Relative floating-point tolerance preserves invariance to signal units.
        tolerance = np.finfo(float).eps * max(float(np.max(np.abs(part))), np.finfo(float).tiny) * 32
        if sd <= tolerance:
            rejected["flat"] += 1
            continue
        first, second = np.array_split(part, 2)
        a, b = float(np.std(first)), float(np.std(second))
        shift = abs(float(np.mean(first) - np.mean(second))) / sd
        ratio = max(a, b) / max(min(a, b), tolerance)
        shifts.append(shift)
        scale_ratios.append(ratio)
        scales.append(sd * amplitude)
        effective.append(min(len(part), (times[-1] - times[0] + dt) / information_s))
    coverage = len(scales) / max(1, len(events))
    shift = _quantile(shifts, .9, 1e6)
    ratio = _quantile(scale_ratios, .9, 1e6)
    sd_ratio = _quantile(scales, .9) / max(_quantile(scales, .1), np.finfo(float).tiny)
    neff = _quantile(effective, .1)
    failures = []
    if coverage < config.min_coverage:
        failures.append("insufficient event-free observed coverage")
    if neff < config.min_effective_samples or context["acf_truncated"] or local["acf_truncated"]:
        failures.append("insufficient independent information")
    if shift > config.max_split_shift:
        failures.append("baseline level changes within the window")
    if ratio > config.max_split_scale_ratio or sd_ratio > config.max_between_trial_sd_ratio:
        failures.append("baseline scale is unstable")
    return {"label": recording["label"], "events": int(len(events)), "usable_events": len(scales), "coverage": coverage,
            "effective_samples_p10": neff, "split_shift_p90": shift,
            "information_s": information_s, "local_information_s": local["information_s"],
            "split_scale_ratio_p90": ratio, "between_trial_sd_ratio": float(sd_ratio),
            "median_sd": _quantile(scales, .5), "rejected": rejected,
            "eligible": not failures, "reasons": failures}


def _aggregate(parts, window, config):
    """Require every recording to pass; larger recordings cannot mask failures."""
    coverage = min((part["coverage"] for part in parts), default=0.0)
    information = min((part["effective_samples_p10"] for part in parts), default=0.0)
    shift = max((part["split_shift_p90"] for part in parts), default=1e6)
    ratio = max((part["split_scale_ratio_p90"] for part in parts), default=1e6)
    eligible = bool(parts) and all(part["eligible"] for part in parts)
    # No term rewards a small absolute SD, a high z-score, or a pre/post contrast.
    score = (40 * coverage + 25 * min(1.0, information / (2 * config.min_effective_samples))
             + 20 / (1 + shift) + 10 / max(1, ratio)
             + 5 * (1 - abs(window[1]) / config.lookback_s))
    return {"eligible": eligible, "score": round(float(score), 3), "coverage": coverage,
            "effective_samples_p10": information, "split_shift_p90": shift,
            "split_scale_ratio_p90": ratio, "per_recording": parts}


def _windows(config):
    """Enumerate a compact shared grid with endpoints matching GUI precision."""
    durations = np.geomspace(config.min_window_s, config.lookback_s, 12)
    ends = -np.unique(np.round(np.r_[config.event_guard_s + .01,
                                    np.geomspace(max(.1, config.event_guard_s + .01), config.lookback_s / 2, 7)], 2))
    return sorted({(round(float(end - duration), 2), round(float(end), 2))
                   for end in ends for duration in durations
                   if end - duration >= -config.lookback_s and end < 0})


def recommend_baseline(recordings, config=None, current_window=None):
    """Rank on training references, check one candidate on purged later events.

    A failed holdout causes abstention, never a search for another candidate on
    that holdout. Validation describes reference stability; it does not provide
    a p-value or validate downstream response significance. Repeated user runs
    on the same data do not create fresh independent validation.
    """
    config = config or BaselineAdvisorConfig()
    _validate_config(config)
    prepared = [_prepare(recording, config) for recording in recordings]
    report = {"method": "baseline_advisor_v1", "status": "unavailable", "window": None,
              "summary": "No reliable common baseline window found.", "reasons": [],
              "config": asdict(config), "candidates": [], "recordings": [], "current": None}
    if not prepared:
        report["reasons"] = ["Load a processed signal and select alignment events."]
        return report
    contexts = []
    for recording in prepared:
        training = _context(recording, recording["train"], config)
        # Separate blocks by an additional training-derived dependence buffer.
        # This limits nearby dependence without claiming independent events.
        buffer_s = max(config.recovery_s, training["information_s"])
        validation_all = recording["events"]
        validation_all = validation_all[validation_all - config.lookback_s > recording["split_boundary"] + buffer_s]
        recording["validation"] = _thin(validation_all, config.max_events_per_split)
        recording["validation_total"] = len(validation_all)
        recording["purged_events"] = len(recording["events"]) - recording["train_total"] - len(validation_all)
        validation = _context(recording, recording["validation"], config)
        contexts.append((training, validation))
        raw = recording["raw_intervals"]
        ibi = raw[1:, 0] - raw[:-1, 1] if len(raw) > 1 else np.array([])
        info = {"label": recording["label"], "events": len(recording["events"]),
                "training_events": len(recording["train"]), "validation_events": len(recording["validation"]),
                "purged_events": recording["purged_events"], "training_available": recording["train_total"],
                "validation_available": recording["validation_total"],
                "validation_buffer_s": buffer_s,
                "ibi_median_s": _quantile(ibi, .5), "ibi_p10_s": _quantile(ibi, .1),
                "point_intervals": int(np.count_nonzero(raw[:, 0] == raw[:, 1])) if len(raw) else 0,
                "training_context": training, "validation_context": validation}
        report["recordings"].append(info)
        if len(recording["train"]) < config.min_train_events or len(recording["validation"]) < config.min_validation_events:
            report["reasons"].append(f"{recording['label']}: too few separated events for training and validation.")
    for window in _windows(config):
        parts = [_evaluate(rec, rec["train"], ctx[0], window, config) for rec, ctx in zip(prepared, contexts)]
        train = _aggregate(parts, window, config)
        report["candidates"].append({"start": window[0], "end": window[1], "score": train["score"],
                                     "eligible": train["eligible"], "diagnostics": {"training": train}})
    report["candidates"].sort(key=lambda row: (row["eligible"], row["score"]), reverse=True)
    if current_window is not None:
        if len(current_window) != 2 or not np.all(np.isfinite(current_window)) or current_window[0] >= current_window[1]:
            raise ValueError("Current baseline must have a finite start earlier than its end.")
        parts = [_evaluate(rec, rec["train"], ctx[0], current_window, config) for rec, ctx in zip(prepared, contexts)]
        report["current"] = {"start": float(current_window[0]), "end": float(current_window[1]),
                             **_aggregate(parts, current_window, config)}
    eligible = [row for row in report["candidates"] if row["eligible"]]
    if report["reasons"]:
        return report
    if not eligible:
        reasons = {reason for row in report["candidates"][:3]
                   for part in row["diagnostics"]["training"]["per_recording"] for reason in part["reasons"]}
        report["reasons"] = sorted(reasons) or ["Search limits leave no candidate windows."]
        return report
    winner = eligible[0]
    window = (winner["start"], winner["end"])
    parts = [_evaluate(rec, rec["validation"], ctx[1], window, config) for rec, ctx in zip(prepared, contexts)]
    validation = _aggregate(parts, window, config)
    # Also reject a change of reference scale between chronological blocks.
    for train, check in zip(winner["diagnostics"]["training"]["per_recording"], parts):
        a, b = train["median_sd"], check["median_sd"]
        ratio = max(a, b) / max(min(a, b), np.finfo(float).tiny)
        check["train_validation_sd_ratio"] = float(ratio)
        if ratio > config.max_split_scale_ratio:
            validation["eligible"] = False
            check["eligible"] = False
            check["reasons"].append("baseline scale changes in the later validation block")
    winner["diagnostics"]["validation"] = validation
    if not validation["eligible"]:
        report["reasons"] = ["The first-choice training window failed the later-event check."]
        report["reasons"] += sorted({reason for part in parts for reason in part["reasons"]})
        return report
    # Sampling keeps the search bounded, but cannot hide a problematic event in
    # the final suggestion. Audit every selected event once, without re-ranking.
    parts = [_evaluate(rec, rec["events"], ctx[0], window, config) for rec, ctx in zip(prepared, contexts)]
    complete = _aggregate(parts, window, config)
    winner["diagnostics"]["all_events"] = complete
    if not complete["eligible"]:
        report["reasons"] = ["The selected window failed the final check of all selected events."]
        report["reasons"] += sorted({reason for part in parts for reason in part["reasons"]})
        return report
    report.update(status="recommended", window=list(window),
                  summary=f"Suggested baseline: {window[0]:.2f} to {window[1]:.2f} s relative to alignment.")
    report["reasons"] = [f"All recordings passed both blocks; minimum usable reference coverage across files is {complete['coverage']:.0%}.",
                         "Event protection uses the supplied bouts and recovery setting; biological recovery is not inferred.",
                         "This is a baseline reliability check, not a test of event-response significance."]
    return report


def export_baseline_report(report, prefix):
    """Write the exact policy and all candidate diagnostics as JSON and CSV."""
    prefix = Path(prefix)
    json_path, csv_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".csv")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["start", "end", "score", "eligible", "coverage", "effective_samples_p10",
                   "split_shift_p90", "split_scale_ratio_p90", "validation_passed"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for candidate in report["candidates"]:
            train = candidate["diagnostics"]["training"]
            writer.writerow({**{key: candidate[key] for key in columns[:4]},
                             **{key: train[key] for key in columns[4:-1]},
                             "validation_passed": candidate["diagnostics"].get("validation", {}).get("eligible", "")})
    return str(json_path), str(csv_path)
