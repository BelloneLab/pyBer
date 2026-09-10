"""Shared numerical definitions for selectable PSTH metric panels and exports.

Metrics summarize each displayed matrix row. A grouped row is an averaged file
trace, so its peak/median/SD is a metric of that average, not the average of the
trial-level metric. This distinction is retained in every exported summary.
"""
from __future__ import annotations

import numpy as np

from postprocessing_core import paired_summary, window_metrics

# Stable identifiers keep project files independent of presentation wording.
METRICS = {
    "auc": "AUC", "mean": "Mean signal", "median": "Median signal",
    "peak": "Peak signal", "trough": "Minimum signal", "std": "Signal SD",
    "peak_latency": "Peak delay",
}
DURATION_SENSITIVE = {"auc", "peak", "trough", "peak_latency"}


def metric_id(value):
    """Accept stored identifiers and legacy display names without losing defaults."""
    if value in METRICS:
        return value
    if value == "Mean z":
        return "mean"
    return next((key for key, label in METRICS.items() if label == value), "auc")


def metric_units(key, signal_units):
    """Expose dimensional differences, especially integrated signal and delay."""
    if key == "auc":
        return f"{signal_units} · s"
    if key == "peak_latency":
        return "s from window start"
    return signal_units


def metric_values(matrix, time, start, stop, key):
    """Reduce one time window per row using documented missing-data rules.

    Existing mean and AUC calculations remain unchanged. Mean uses available
    bins; AUC requires complete coverage and interpolates exact boundaries.
    New distribution/extremum metrics require a fully observed requested
    window. Peak delay is measured from that window's start; the first tied
    maximum wins, and a flat trace has no defined peak delay.
    """
    if key not in METRICS:
        raise ValueError(f"Unknown PSTH metric: {key}")
    values, t = np.asarray(matrix, float), np.asarray(time, float)
    if key in {"mean", "auc"}:
        return window_metrics(values, t, start, stop, key)
    # Reuse core shape, clock, and window validation without changing its rules.
    coverage = window_metrics(values, t, start, stop, "auc")
    result = np.full(values.shape[0], np.nan)
    mask = (t >= start) & (t <= stop)
    segment, segment_t = values[:, mask], t[mask]
    if not len(segment_t):
        return result
    for index, row in enumerate(segment):
        if not np.isfinite(coverage[index]) or not np.all(np.isfinite(row)):
            continue
        if key == "median":
            result[index] = np.median(row)
        elif key == "peak":
            result[index] = np.max(row)
        elif key == "trough":
            result[index] = np.min(row)
        elif key == "std" and len(row) >= 2:
            result[index] = np.std(row, ddof=1)
        elif key == "peak_latency" and np.ptp(row) > 0:
            result[index] = segment_t[int(np.argmax(row))] - start
    return result


def _describe(values):
    """Keep sample counts, median, mean and descriptive SEM explicit."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": np.nan, "median": np.nan, "sem": np.nan, "n": 0}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
            "n": int(len(values))}


def summarize_metrics(matrix, time, selected, pre_window, post_window, signal_units,
                      independent_units=True, row_unit="trials"):
    """Compute matched panels and Holm-adjust their finite paired sign-test p-values.

    The selected family is explicit. Changing the family is exploratory and is
    not a correction for repeatedly inspecting different windows or metrics.
    Duration-sensitive metrics remain descriptive when window lengths differ.
    """
    keys = list(dict.fromkeys(metric_id(key) for key in selected))
    results = {}
    same_duration = np.isclose(pre_window[1] - pre_window[0], post_window[1] - post_window[0])
    for key in keys:
        pre = metric_values(matrix, time, *pre_window, key)
        post = metric_values(matrix, time, *post_window, key)
        before, after = _describe(pre), _describe(post)
        comparable = same_duration or key not in DURATION_SENSITIVE
        test = paired_summary(pre, post, independent_units=independent_units and comparable)
        summary = {
            "metric": METRICS[key], "metric_id": key, "units": metric_units(key, signal_units),
            "pre": before["mean"], "post": after["mean"],
            "pre_median": before["median"], "post_median": after["median"],
            "pre_sem": before["sem"], "post_sem": after["sem"],
            "pre_n": before["n"], "post_n": after["n"], **test,
            "pre_start_s": pre_window[0], "pre_end_s": pre_window[1],
            "post_start_s": post_window[0], "post_end_s": post_window[1],
            "row_unit": row_unit, "reduction_level": "metric of each displayed row",
            "selected_family": ",".join(keys), "paired_p_holm": np.nan,
            "multiplicity_method": "Holm across finite tests in selected metric family",
            "family_size": 0,
        }
        if row_unit == "files":
            summary["reduction_level"] = "metric of each file's averaged trial waveform"
        if not comparable:
            summary["assumption_note"] += " Unequal window lengths: this duration-sensitive metric is descriptive only."
        summary["assumption_note"] += " SEM is descriptive. Selection across repeated analyses is not corrected."
        results[key] = {"summary": summary, "pre_values": pre, "post_values": post}
    finite = [(key, result["summary"]["paired_p"]) for key, result in results.items()
              if np.isfinite(result["summary"]["paired_p"])]
    previous = 0.0
    for rank, (key, pvalue) in enumerate(sorted(finite, key=lambda item: item[1])):
        previous = min(1.0, max(previous, float(pvalue) * (len(finite) - rank)))
        results[key]["summary"]["paired_p_holm"] = previous
    for result in results.values():
        result["summary"]["family_size"] = len(finite)
    return results


def draw_metric_matplotlib(axis, result, title=None):
    """Render export panels with the same numerical summaries as the live view."""
    summary = result["summary"]
    pre, post = result["pre_values"], result["post_values"]
    pairs = np.isfinite(pre) & np.isfinite(post)
    positions = np.linspace(-0.12, 0.12, len(pre))
    for index in np.flatnonzero(pairs):
        axis.plot([positions[index], 1 + positions[index]], [pre[index], post[index]],
                  color="#A7B1BD", alpha=0.35, linewidth=0.6, zorder=1)
    for x, name, values, color in ((0, "pre", pre, "#4D93BD"), (1, "post", post, "#D28A64")):
        valid = np.isfinite(values)
        axis.scatter(x + positions[valid], values[valid], s=11, color=color, alpha=0.7,
                     edgecolors="none", zorder=2)
        median, mean, sem = (summary[name + suffix] for suffix in ("_median", "", "_sem"))
        if np.isfinite(median):
            axis.plot([x - 0.23, x + 0.23], [median, median], color=color, linewidth=2.2, zorder=4)
        if np.isfinite(mean):
            axis.errorbar(x + 0.28, mean, yerr=sem if np.isfinite(sem) else None,
                          fmt="D", ms=3, color="#526371", capsize=2, linewidth=0.9, zorder=5)
    adjusted = summary["paired_p_holm"]
    note = f"Holm p={adjusted:.3g}" if np.isfinite(adjusted) else "Descriptive only"
    axis.set_title(title or summary["metric"], fontsize=10, fontweight="bold")
    axis.text(0.5, 0.99, f"{note}\nline: median  ·  diamond: mean ± SEM", transform=axis.transAxes,
              ha="center", va="top", fontsize=6.5, color="#546274")
    axis.margins(y=0.28)
    axis.set_xticks([0, 1], ["Pre", "Post"])
    axis.set_xlim(-0.45, 1.5)
    axis.set_ylabel(summary["units"], fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)


def export_selected_metrics(results, prefix, write_csv=True, write_h5=True):
    """Export complete metric summaries and per-row values without truncating labels."""
    import csv
    import h5py
    from pathlib import Path
    base = str(Path(prefix))
    if not results:
        return
    if write_csv:
        summaries = [result["summary"] for result in results.values()]
        with open(base + "_selected.csv", "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        with open(base + "_rows.csv", "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric_id", "row", "row_unit", "units", "pre", "post", "difference"])
            for key, result in results.items():
                summary = result["summary"]
                for row, (pre, post) in enumerate(zip(result["pre_values"], result["post_values"]), 1):
                    writer.writerow([key, row, summary["row_unit"], summary["units"], pre, post, post - pre])
    if write_h5:
        with h5py.File(base + "_selected.h5", "w") as file:
            for key, result in results.items():
                group = file.create_group(key)
                for name, value in result["summary"].items():
                    group.create_dataset(name, data=value,
                                         dtype=h5py.string_dtype("utf-8") if isinstance(value, str) else None)
                group.create_dataset("pre_values", data=result["pre_values"])
                group.create_dataset("post_values", data=result["post_values"])
