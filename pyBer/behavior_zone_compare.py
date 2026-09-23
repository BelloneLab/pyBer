"""Event-window photometry comparisons for behavior and zone imports.

Each recording contributes one mean to a group result, regardless of its bout
count. Missing signal and incomplete windows are excluded, never filled with
zeros. This module does not alter the processed trace or behavior source.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def _window_mean(time: np.ndarray, signal: np.ndarray, start: float,
                 stop: float, typical_step: float) -> float:
    if not np.isfinite(start + stop) or stop <= start:
        return float("nan")
    # Sampling tolerance must not accept windows outside the recording. The
    # final sample represents one nominal sample interval, like every other bin.
    if start < time[0] - 1e-9 or stop > time[-1] + typical_step + 1e-9:
        return float("nan")
    left = int(np.searchsorted(time, start, side="left"))
    right = int(np.searchsorted(time, stop, side="left"))
    segment_time, segment_signal = time[left:right], signal[left:right]
    if segment_time.size < 2 or not np.all(np.isfinite(segment_signal)):
        return float("nan")
    tolerance = max(typical_step * 2.5, 1e-9)
    if segment_time[0] - start > tolerance or stop - segment_time[-1] > tolerance:
        return float("nan")
    if np.any(np.diff(segment_time) > tolerance):
        return float("nan")
    return float(np.mean(segment_signal))


def compare_recording(file_id: str, time: np.ndarray, signal: np.ndarray,
                      events: Mapping[str, Mapping[str, np.ndarray]],
                      labels: Iterable[str], *, pre_s: float, post_s: float,
                      offset_s: float = 0.0) -> list[dict[str, object]]:
    """Compute before/during/after means from complete windows around bouts."""
    if not np.all(np.isfinite([pre_s, post_s, offset_s])) or pre_s <= 0 or post_s <= 0:
        raise ValueError("Before and after windows must be positive.")
    time, signal = np.asarray(time, float), np.asarray(signal, float)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 3:
        raise ValueError("A comparison needs matching one-dimensional signal and time arrays.")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Photometry time must be finite and strictly increasing.")
    typical_step = float(np.median(np.diff(time)))
    rows = []
    for label in labels:
        item = events.get(label) or {}
        starts = np.asarray(item.get("on", []), float).reshape(-1) + offset_s
        stops = np.asarray(item.get("off", []), float).reshape(-1) + offset_s
        if starts.size != stops.size:
            raise ValueError(f"{label}: onset and offset counts differ.")
        accepted = []
        for onset, offset in zip(starts, stops):
            if not np.isfinite(onset + offset) or offset <= onset:
                continue
            before = _window_mean(time, signal, onset - pre_s, onset, typical_step)
            during = _window_mean(time, signal, onset, offset, typical_step)
            after = _window_mean(time, signal, offset, offset + post_s, typical_step)
            if np.all(np.isfinite([before, during, after])):
                accepted.append((before, during, after))
        if not accepted:
            rows.append({"file_id": file_id, "label": label, "events": 0,
                         "rejected": int(starts.size), "before": np.nan,
                         "during": np.nan, "after": np.nan,
                         "during_minus_before": np.nan, "after_minus_before": np.nan})
            continue
        means = np.asarray(accepted, float).mean(axis=0)
        rows.append({"file_id": file_id, "label": label,
                     "events": len(accepted), "rejected": int(starts.size - len(accepted)),
                     "before": float(means[0]), "during": float(means[1]),
                     "after": float(means[2]),
                     "during_minus_before": float(means[1] - means[0]),
                     "after_minus_before": float(means[2] - means[0])})
    return rows


def group_recordings(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Summarize per-recording means so long recordings do not dominate."""
    by_label: dict[str, list[Mapping[str, object]]] = {}
    seen = set()
    for row in rows:
        key = (str(row["label"]), str(row["file_id"]))
        if key in seen:
            raise ValueError(f"Duplicate recording for {key[0]}: {key[1]}. Each recording may contribute only once.")
        seen.add(key)
        by_label.setdefault(str(row["label"]), []).append(row)
    output = []
    metrics = ("before", "during", "after", "during_minus_before", "after_minus_before")
    for label, candidates in by_label.items():
        valid = [row for row in candidates if int(row["events"]) > 0
                 and all(np.isfinite(float(row[metric])) for metric in metrics)]
        result: dict[str, object] = {"label": label, "recordings": len(valid),
                                     "recordings_available": len(candidates),
                                     "recordings_excluded": len(candidates) - len(valid),
                                     "recording_ids": [str(row["file_id"]) for row in valid],
                                     "events": sum(int(row["events"]) for row in valid),
                                     "rejected": sum(int(row.get("rejected", 0)) for row in candidates)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in valid], float)
            result[metric] = float(np.mean(values)) if values.size else np.nan
            result[metric + "_sem"] = (float(np.std(values, ddof=1) / np.sqrt(values.size))
                                        if values.size > 1 else np.nan)
        output.append(result)
    return output
