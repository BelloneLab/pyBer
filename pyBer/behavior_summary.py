"""Binned behavior summaries with explicit observation time and recording scope.

Inputs are the selected event bouts, not inferred continuous neural events.
Unknown bout ends stay unknown. No interval is formed across recordings or an
unobserved gap; cumulative duration integrates the union of observed bout time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


MAX_TIME_BINS = 2000
MAX_AUTO_HISTOGRAM_BINS = 60
METRICS = {
    "Bout duration": ("histogram", "Bout duration", "Duration (s)", "Selected bouts"),
    "Frequency over time": ("timeseries", "Behavior frequency", "Time from recording start (s)", "Selected bouts / observed min"),
    "Inter-bout interval": ("histogram", "Inter-bout intervals", "IBI (s)", "Intervals"),
    "Cumulative duration": ("timeseries", "Cumulative behavior duration", "Time from recording start (s)", "Cumulative observed duration (s)"),
}


def _union(intervals):
    """Merge occupied or observed intervals to prevent double counting."""
    intervals = np.asarray(intervals, float).reshape(-1, 2)
    intervals = intervals[np.all(np.isfinite(intervals), axis=1)]
    result = []
    for start, end in sorted(intervals.tolist()):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return np.asarray(result, float).reshape(-1, 2)


def _overlap_duration(intervals, edges):
    """Integrate intervals exactly over time bins, including partial end bins."""
    duration = np.zeros(len(edges) - 1)
    for start, end in intervals:
        left = max(0, np.searchsorted(edges, start, side="right") - 1)
        right = min(len(duration), np.searchsorted(edges, end, side="left") + 1)
        duration[left:right] += np.maximum(0, np.minimum(edges[left + 1:right + 1], end)
                                         - np.maximum(edges[left:right], start))
    return duration


def _intersect(first, second):
    """Intersect sorted disjoint intervals without bridging missing observations."""
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        start, end = max(first[i, 0], second[j, 0]), min(first[i, 1], second[j, 1])
        if end > start:
            result.append([start, end])
        if first[i, 1] <= second[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(result, float).reshape(-1, 2)


def _prepare(recording):
    """Validate one source and move its times to a shared elapsed-time origin."""
    start, end = float(recording["start"]), float(recording["end"])
    if not np.isfinite(start + end) or end <= start:
        raise ValueError("Behavior summary needs a finite recording start earlier than its end.")
    onsets = np.asarray(recording.get("onsets", []), float).reshape(-1)
    offsets = np.asarray(recording.get("offsets", []), float).reshape(-1)
    if len(offsets) != len(onsets):
        raise ValueError("Bout onset and offset arrays must have equal length.")
    valid = np.isfinite(onsets)
    onsets, offsets = onsets[valid].copy(), offsets[valid].copy()
    offsets[~np.isfinite(offsets) | (offsets < onsets)] = np.nan
    order = np.argsort(onsets, kind="stable")
    onsets, offsets = onsets[order] - start, offsets[order] - start
    observed = np.asarray(recording.get("observed_intervals", [[start, end]]), float).reshape(-1, 2)
    observed = _union(np.clip(observed - start, 0, end - start))
    return {"file_id": str(recording.get("file_id", "recording")), "span": end - start,
            "onsets": onsets, "offsets": offsets, "observed": observed}


def _observed_points(points, intervals):
    """Keep an event if its onset lies within an observed recording segment."""
    keep = np.zeros(len(points), bool)
    for start, end in intervals:
        keep |= (points >= start) & (points <= end)
    return keep


def _histogram_values(recording, metric):
    """Extract complete durations or genuine offset-to-next-onset intervals."""
    on, off, observed = recording["onsets"], recording["offsets"], recording["observed"]
    values = []
    if metric == "Bout duration":
        for start, end in zip(on, off):
            if np.isfinite(end) and end > start and any(a <= start and end <= b for a, b in observed):
                values.append(end - start)
    else:
        # Only adjacent selected bouts count. An unknown intervening end cannot
        # be skipped to construct an interval between different neighbors.
        last_end = None
        last_segment = None
        for start, end in zip(on, off):
            segment = next((i for i, (a, b) in enumerate(observed) if a <= start <= b), None)
            if last_end is not None and segment is not None and segment == last_segment:
                if start >= last_end:
                    values.append(start - last_end)
            if np.isfinite(end) and segment is not None and end <= observed[segment, 1]:
                # Overlapping selected bouts are one occupied period for IBI.
                last_end = max(last_end, end) if last_segment == segment and last_end is not None and start < last_end else end
                last_segment = segment
            else:
                last_end, last_segment = None, None
    return np.asarray(values, float)


def _mean_sem(matrix):
    """Give each observed recording equal weight; missing bins are not zeros."""
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0).sum(axis=0)
    mean = np.divide(sums, counts, out=np.full(matrix.shape[1], np.nan), where=counts > 0)
    deviations = np.where(finite, matrix - mean, 0)
    variance = np.divide((deviations ** 2).sum(axis=0), counts - 1,
                         out=np.zeros(matrix.shape[1]), where=counts > 1)
    sem = np.divide(np.sqrt(variance), np.sqrt(counts), out=np.zeros(matrix.shape[1]), where=counts > 1)
    sem[counts == 0] = np.nan
    return mean, sem, counts


def summarize_behavior(recordings, metric="Bout duration", bin_s=30.0, auto_bins=True):
    """Build plot/export data for a selected behavior metric.

    Histograms pool complete selected bouts across files. Frequency is counted
    at bout onset and divided by observed seconds within each bin. Time curves
    use elapsed recording time and equal file weights. Cumulative duration sums
    observed occupied time only and never extrapolates through missing periods.
    """
    metric = {"duration": "Bout duration", "frequency": "Frequency over time",
              "ibi": "Inter-bout interval", "cumulative": "Cumulative duration"}.get(metric, metric)
    if metric not in METRICS:
        raise ValueError("Unknown behavior summary metric.")
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("Behavior bin width must be positive and finite.")
    prepared = [_prepare(recording) for recording in recordings]
    kind, title, x_label, y_label = METRICS[metric]
    result = {"metric": metric, "kind": kind, "title": title, "x_label": x_label,
              "y_label": y_label, "edges": np.array([], float), "values": np.array([], float),
              "sem": np.array([], float), "counts": np.array([], int),
              "per_file_values": np.empty((len(prepared), 0)), "file_ids": [r["file_id"] for r in prepared],
              "notes": [], "has_data": False, "bin_s": float(bin_s), "auto_bins": bool(auto_bins)}
    if not prepared:
        return result
    result["notes"].append("Uses selected PSTH events; event filters and baseline validity define the selected bouts.")
    unknown = sum(np.count_nonzero(~np.isfinite(r["offsets"])) for r in prepared)
    if unknown:
        result["notes"].append(f"{unknown} selected events have unknown duration; their occupied time cannot be inferred.")
    if kind == "histogram":
        per_file = [_histogram_values(recording, metric) for recording in prepared]
        values = np.concatenate(per_file)
        if not values.size:
            return result
        if auto_bins:
            edges = np.histogram_bin_edges(values, bins="fd")
            if len(edges) > MAX_AUTO_HISTOGRAM_BINS + 1:
                edges = np.histogram_bin_edges(values, bins=MAX_AUTO_HISTOGRAM_BINS)
            # Keep duration and interval axes nonnegative even for constants.
            if edges[0] < 0:
                edges = np.linspace(0, max(float(edges[-1]), .1), len(edges))
        else:
            count = max(1, int(np.ceil(float(np.max(values)) / bin_s)))
            if count > MAX_TIME_BINS:
                raise ValueError("Too many behavior bins. Increase the bin width.")
            edges = np.arange(count + 1) * bin_s
        matrix = np.asarray([np.histogram(values, bins=edges)[0] for values in per_file])
        counts = matrix.sum(axis=0)
        result.update(edges=edges, values=counts.astype(float), sem=np.zeros(len(counts)),
                      per_file_values=matrix, counts=counts, has_data=True,
                      bin_s=float(edges[1] - edges[0]))
        result["notes"].append("Histograms pool selected bouts across recordings; intervals never span recordings or gaps.")
        return result
    span = max(recording["span"] for recording in prepared)
    count = max(1, int(np.ceil(span / bin_s)))
    if count > MAX_TIME_BINS:
        raise ValueError("Too many time bins. Increase the bin width.")
    edges = np.minimum(np.arange(count + 1) * bin_s, span)
    matrix = np.full((len(prepared), count), np.nan)
    exposure_matrix = np.zeros_like(matrix)
    for index, recording in enumerate(prepared):
        observed = recording["observed"]
        exposure = _overlap_duration(observed, edges)
        exposure_matrix[index] = exposure
        if metric == "Frequency over time":
            onsets = recording["onsets"]
            counts = np.histogram(onsets[_observed_points(onsets, observed)], bins=edges)[0]
            matrix[index] = np.divide(60 * counts, exposure, out=np.full(count, np.nan), where=exposure > 0)
        else:
            onsets, offsets = recording["onsets"], recording["offsets"]
            known = np.isfinite(offsets) & (offsets > onsets)
            # Point-only event lists cannot imply zero cumulative occupied time.
            if len(onsets) and not np.any(known):
                continue
            bouts = _union(np.column_stack((onsets[known], offsets[known])))
            occupied = _intersect(bouts, observed)
            cumulative = np.cumsum(_overlap_duration(occupied, edges))
            if np.any(exposure > 0):
                # Observed accumulated duration is known even in later empty
                # bins: carry its last total, not an inferred activity rate.
                # A fixed file cohort prevents falling group cumulative means
                # when shorter recordings leave the observation window.
                matrix[index] = cumulative
    values, sem, counts = _mean_sem(matrix)
    result.update(edges=edges, values=values, sem=sem, counts=counts, per_file_values=matrix,
                  observed_seconds=exposure_matrix, has_data=bool(np.any(np.isfinite(values))))
    result["notes"].append("Time is relative to each recording start; group curves are equal-recording means, with SEM where at least two files contribute.")
    if metric == "Cumulative duration":
        result["notes"].append("Duration is integrated across bin boundaries; overlapping bouts count once. Last observed totals are carried through gaps and past recording end, preserving a fixed group cohort without inferring unobserved behavior.")
    return result


def export_behavior_summary(summary, prefix, *, write_csv=True, write_h5=False):
    """Export precisely the selected plot's bins, units and per-recording data."""
    paths = []
    metadata = {key: summary[key] for key in ("metric", "kind", "title", "x_label", "y_label",
                                             "bin_s", "auto_bins", "file_ids", "notes")}
    edges, values = summary["edges"], summary["values"]
    if write_csv:
        path = Path(str(prefix) + ".csv")
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["bin_start_s", "bin_end_s", "value", "sem", *summary["file_ids"]])
            for index, value in enumerate(values):
                writer.writerow([edges[index], edges[index + 1], value, summary["sem"][index],
                                 *summary["per_file_values"][:, index]])
        paths.append(str(path))
    if write_h5:
        import h5py
        path = Path(str(prefix) + ".h5")
        with h5py.File(path, "w") as output:
            for key in ("edges", "values", "sem", "counts", "per_file_values"):
                output.create_dataset(key, data=summary[key])
            if "observed_seconds" in summary:
                output.create_dataset("observed_seconds", data=summary["observed_seconds"])
            output.attrs["metadata_json"] = json.dumps(metadata)
        paths.append(str(path))
    if write_csv or write_h5:
        path = Path(str(prefix) + ".json")
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths
