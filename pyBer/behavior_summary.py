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
    "Occupancy over time": ("timeseries", "Behavior occupancy", "Time from recording start (s)", "Observed time occupied (%)"),
    "Cumulative bout count": ("timeseries", "Cumulative bout count", "Time from recording start (s)", "Observed bout onsets"),
    "Onset-to-onset interval": ("histogram", "Onset-to-onset intervals", "Onset interval (s)", "Intervals"),
    "Bout duration over time": ("timeseries", "Bout duration over time", "Time from recording start (s)", "Median bout duration (s)"),
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


def _onset_counts(onsets, observed, edges):
    """Assign observed segment-end events to the final observed bin.

    Most bins are left-closed/right-open. A bout at the last observed sample
    before a cut belongs to that observed segment, not to the following empty
    bin. A one-ULP adjustment implements this endpoint convention only.
    """
    points = onsets[_observed_points(onsets, observed)].copy()
    for start, end in observed:
        at_end = (points == end) & (end > start)
        points[at_end] = np.nextafter(points[at_end], -np.inf)
    return np.histogram(points, bins=edges)[0]


def _histogram_values(recording, metric):
    """Extract complete durations or genuine offset-to-next-onset intervals."""
    on, off, observed = recording["onsets"], recording["offsets"], recording["observed"]
    values = []
    if metric == "Bout duration":
        for start, end in zip(on, off):
            if np.isfinite(end) and end > start and any(a <= start and end <= b for a, b in observed):
                values.append(end - start)
    elif metric == "Onset-to-onset interval":
        # Adjacent onset intervals do not require measured bout offsets. Never
        # connect across an unobserved segment or skip an intervening event.
        for first, second in zip(on[:-1], on[1:]):
            if second >= first and any(a <= first <= second <= b for a, b in observed):
                values.append(second - first)
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


def _distribution_statistics(values, unit, definition):
    """Report exact unbinned descriptive statistics with their sampling unit."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    quartiles = np.quantile(values, [.25, .5, .75]) if values.size else [None] * 3
    return {"q1": None if quartiles[0] is None else float(quartiles[0]),
            "median": None if quartiles[1] is None else float(quartiles[1]),
            "q3": None if quartiles[2] is None else float(quartiles[2]),
            "n": int(values.size), "unit": unit, "definition": definition}


def summarize_behavior(recordings, metric="Bout duration", bin_s=30.0, auto_bins=True):
    """Build plot/export data for a selected behavior metric.

    Histograms pool complete selected bouts across files. Frequency is counted
    at bout onset and divided by observed seconds within each bin. Time curves
    use elapsed recording time and equal file weights. Cumulative duration sums
    observed occupied time only and never extrapolates through missing periods.
    """
    metric = {"duration": "Bout duration", "frequency": "Frequency over time",
              "ibi": "Inter-bout interval", "cumulative": "Cumulative duration",
              "occupancy": "Occupancy over time", "cumulative_count": "Cumulative bout count",
              "onset_interval": "Onset-to-onset interval", "duration_time": "Bout duration over time"}.get(metric, metric)
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
              "notes": [], "has_data": False, "bin_s": float(bin_s), "auto_bins": bool(auto_bins),
              "statistics": _distribution_statistics([], "", "No observations"), "per_file_statistics": [],
              "aggregation": "pooled counts" if kind == "histogram" else "equal-recording mean +/- SEM"}
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
        result["statistics"] = _distribution_statistics(values, "s", "Pooled complete selected bouts or intervals; median and quartiles use unbinned observations")
        result["per_file_statistics"] = [dict(file_id=recording["file_id"], **_distribution_statistics(samples, "s", "Complete selected bouts or intervals in this recording"))
                                         for recording, samples in zip(prepared, per_file)]
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
        result["notes"].append("Median and IQR use exact unbinned observations, not histogram centers.")
        return result
    span = max(recording["span"] for recording in prepared)
    count = max(1, int(np.ceil(span / bin_s)))
    if count > MAX_TIME_BINS:
        raise ValueError("Too many time bins. Increase the bin width.")
    edges = np.minimum(np.arange(count + 1) * bin_s, span)
    matrix = np.full((len(prepared), count), np.nan)
    exposure_matrix = np.zeros_like(matrix)
    lower_matrix = np.full_like(matrix, np.nan)
    upper_matrix = np.full_like(matrix, np.nan)
    overall = []
    for index, recording in enumerate(prepared):
        observed = recording["observed"]
        exposure = _overlap_duration(observed, edges)
        exposure_matrix[index] = exposure
        onsets = recording["onsets"]
        onset_counts = _onset_counts(onsets, observed, edges)
        if metric in ("Frequency over time", "Cumulative bout count"):
            if metric == "Frequency over time":
                matrix[index] = np.divide(60 * onset_counts, exposure, out=np.full(count, np.nan), where=exposure > 0)
                overall.append(60 * onset_counts.sum() / exposure.sum() if exposure.sum() > 0 else np.nan)
            else:
                if exposure.sum() > 0:
                    matrix[index] = np.cumsum(onset_counts)
                overall.append(float(onset_counts.sum()) if exposure.sum() > 0 else np.nan)
        elif metric == "Bout duration over time":
            # A complete bout is assigned to its onset bin. Bin edges do not
            # truncate duration; incomplete bouts and unknown ends stay absent.
            complete_on, complete_duration = [], []
            for start, end in zip(onsets, recording["offsets"]):
                if np.isfinite(end) and end > start and any(a <= start < end <= b for a, b in observed):
                    complete_on.append(start)
                    complete_duration.append(end - start)
            complete_on, complete_duration = np.asarray(complete_on), np.asarray(complete_duration)
            positions = np.clip(np.searchsorted(edges, complete_on, side="right") - 1, 0, count - 1)
            for bin_index in np.unique(positions):
                q1, median, q3 = np.quantile(complete_duration[positions == bin_index], [.25, .5, .75])
                lower_matrix[index, bin_index], matrix[index, bin_index], upper_matrix[index, bin_index] = q1, median, q3
            overall.append(float(np.median(complete_duration)) if complete_duration.size else np.nan)
        else:
            onsets, offsets = recording["onsets"], recording["offsets"]
            known = np.isfinite(offsets) & (offsets > onsets)
            # Point-only event lists cannot imply zero cumulative occupied time.
            if len(onsets) and not np.any(known):
                overall.append(np.nan)
                continue
            bouts = _union(np.column_stack((onsets[known], offsets[known])))
            occupied = _intersect(bouts, observed)
            occupied_seconds = _overlap_duration(occupied, edges)
            cumulative = np.cumsum(occupied_seconds)
            if metric == "Occupancy over time":
                matrix[index] = np.divide(100 * occupied_seconds, exposure, out=np.full(count, np.nan), where=exposure > 0)
                overall.append(100 * occupied_seconds.sum() / exposure.sum() if exposure.sum() > 0 else np.nan)
            elif np.any(exposure > 0):
                # Observed accumulated duration is known even in later empty
                # bins: carry its last total, not an inferred activity rate.
                # A fixed file cohort prevents falling group cumulative means
                # when shorter recordings leave the observation window.
                matrix[index] = cumulative
                overall.append(float(cumulative[-1]))
            else:
                overall.append(np.nan)
    values, sem, counts = _mean_sem(matrix)
    if metric == "Bout duration over time":
        lower, upper = np.full(count, np.nan), np.full(count, np.nan)
        for bin_index in range(count):
            finite = matrix[:, bin_index][np.isfinite(matrix[:, bin_index])]
            if finite.size:
                lower[bin_index], values[bin_index], upper[bin_index] = np.quantile(finite, [.25, .5, .75])
                if len(prepared) == 1:
                    lower[bin_index], upper[bin_index] = lower_matrix[0, bin_index], upper_matrix[0, bin_index]
        sem[:] = np.nan
        result.update(lower=lower, upper=upper, per_file_q1=lower_matrix, per_file_q3=upper_matrix,
                      aggregation="median of recording medians; between-recording IQR (one recording: within-bout IQR)")
    result.update(edges=edges, values=values, sem=sem, counts=counts, per_file_values=matrix,
                  observed_seconds=exposure_matrix, has_data=bool(np.any(np.isfinite(values))))
    definition, unit = {
        "Frequency over time": ("Across-recording distribution of total observed onset count / total observed minutes", "bouts/min"),
        "Cumulative bout count": ("Across-recording distribution of final observed onset counts", "bouts"),
        "Occupancy over time": ("Across-recording distribution of total known occupied time / total observed time", "%"),
        "Cumulative duration": ("Across-recording distribution of final known occupied duration", "s"),
        "Bout duration over time": ("Across-recording distribution of complete-bout duration medians", "s"),
    }[metric]
    result["statistics"] = _distribution_statistics(overall, unit, definition)
    result["notes"].append("Annotated median and IQR: " + definition + "; independent of the displayed bin width.")
    result["per_file_statistics"] = [{"file_id": recording["file_id"], "value": float(value) if np.isfinite(value) else None,
                                      "unit": unit, "definition": definition.replace("Across-recording distribution of ", "")}
                                     for recording, value in zip(prepared, overall)]
    result["notes"].append("Time is relative to each recording start. " + result["aggregation"] + ".")
    if metric == "Occupancy over time":
        result["notes"].append("Occupancy is the union of known bouts divided by observed time; overlap counts once. Unknown bout ends make this a lower bound on true occupancy.")
    if metric == "Cumulative bout count":
        result["notes"].append("Counts observed onsets, including point events. Last count is carried through gaps and past recording end without inventing new bouts.")
    if metric == "Cumulative duration":
        result["notes"].append("Duration is integrated across bin boundaries; overlapping bouts count once. Last observed totals are carried through gaps and past recording end, preserving a fixed group cohort without inferring unobserved behavior.")
    return result


def export_behavior_summary(summary, prefix, *, write_csv=True, write_h5=False):
    """Export precisely the selected plot's bins, units and per-recording data."""
    paths = []
    metadata = {key: summary[key] for key in ("metric", "kind", "title", "x_label", "y_label",
                                             "bin_s", "auto_bins", "file_ids", "notes")}
    metadata.update({key: summary[key] for key in ("statistics", "per_file_statistics", "aggregation") if key in summary})
    edges, values = summary["edges"], summary["values"]
    if write_csv:
        path = Path(str(prefix) + ".csv")
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            extra_columns = [key for key in ("lower", "upper") if key in summary]
            writer.writerow(["bin_start_s", "bin_end_s", "value", "sem", *summary["file_ids"], *extra_columns])
            for index, value in enumerate(values):
                writer.writerow([edges[index], edges[index + 1], value, summary["sem"][index],
                                 *summary["per_file_values"][:, index], *[summary[key][index] for key in extra_columns]])
        paths.append(str(path))
    if write_h5:
        import h5py
        path = Path(str(prefix) + ".h5")
        with h5py.File(path, "w") as output:
            for key in ("edges", "values", "sem", "counts", "per_file_values"):
                output.create_dataset(key, data=summary[key])
            if "observed_seconds" in summary:
                output.create_dataset("observed_seconds", data=summary["observed_seconds"])
            for key in ("lower", "upper", "per_file_q1", "per_file_q3"):
                if key in summary:
                    output.create_dataset(key, data=summary[key])
            output.attrs["metadata_json"] = json.dumps(metadata)
        paths.append(str(path))
    if write_csv or write_h5:
        path = Path(str(prefix) + ".json")
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths
