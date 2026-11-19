"""Display handling of missing samples in heatmaps.

Heatmap rows (trials or animals) can contain missing samples (NaN):

* interior gaps, where artifacts were cut out of a recording;
* leading / trailing gaps, where a recording starts later or ends earlier than
  the others around the alignment event.

Drawn as they are, these samples are transparent and show the page background
(white holes in a light figure), and rows look as if they had different
lengths. This module prepares the matrix that is *drawn*. It never changes the
analysis: means, SEM, metrics and exports are computed from the original
matrix with NaN-aware statistics.

Two independent steps, both chosen in Preferences:

1. Trim to the common span: keep only the columns between the latest row start
   and the earliest row end, so every row has the same length and none ends in
   a blank. The trim is skipped when it would discard too much (one very short
   recording must not hide everyone else's data).
2. Interior gaps: fill each one by linear interpolation between its two
   neighbouring samples of the same row (display only), or leave it and draw
   it in a neutral "no data" gray, or leave it blank.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

#: Preference values for interior gaps.
MODE_INTERPOLATE = "interpolate"   # fill gaps by linear interpolation (display only)
MODE_GRAY = "gray"                 # leave gaps, draw them in neutral gray
MODE_BLANK = "blank"               # leave gaps transparent (previous behavior)
MISSING_MODES: Tuple[Tuple[str, str], ...] = (
    (MODE_INTERPOLATE, "Interpolate gaps (display only)"),
    (MODE_GRAY, "Show gaps in neutral gray"),
    (MODE_BLANK, "Leave gaps blank"),
)
DEFAULT_MODE = MODE_INTERPOLATE
DEFAULT_TRIM = True

#: QSettings keys shared by the Preferences dialog and the plots.
SETTINGS_KEY_MODE = "display/missing_data_mode"
SETTINGS_KEY_TRIM = "display/missing_trim_common_span"

#: The common span is used only if it keeps at least this fraction of the
#: longest row. Below it, rows are left untrimmed and their ends drawn as gaps.
MIN_COMMON_FRACTION = 0.5
#: Color of samples that are still missing after the chosen handling.
NO_DATA_COLOR = "#8b909a"


def common_span(matrix: np.ndarray) -> Tuple[int, int, float]:
    """Columns shared by every row that has data.

    Returns ``(start, stop, kept_fraction)``: the half-open column range
    between the latest first sample and the earliest last sample, and the
    fraction of the longest row it keeps. Rows without any finite value are
    ignored. ``(0, n_columns, 1.0)`` when there is nothing to trim.
    """
    mat = np.asarray(matrix, float)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return 0, int(mat.shape[1]) if mat.ndim == 2 else 0, 1.0
    finite = np.isfinite(mat)
    rows = finite.any(axis=1)
    if not rows.any():
        return 0, mat.shape[1], 1.0
    first = finite[rows].argmax(axis=1)
    last = mat.shape[1] - 1 - finite[rows][:, ::-1].argmax(axis=1)
    start, stop = int(first.max()), int(last.min()) + 1
    longest = int((last - first + 1).max())
    if stop - start < 2:
        return 0, mat.shape[1], 0.0
    return start, stop, float(stop - start) / float(max(1, longest))


def interpolate_interior_gaps(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """Fill NaN runs that have a finite sample on both sides, row by row.

    Linear interpolation between the two neighbours. Leading and trailing NaN
    are never extrapolated. Returns the filled copy and the number of samples
    filled.
    """
    mat = np.array(matrix, dtype=float, copy=True)
    filled = 0
    if mat.ndim != 2:
        return mat, filled
    columns = np.arange(mat.shape[1])
    for row in mat:
        ok = np.isfinite(row)
        if ok.all() or ok.sum() < 2:
            continue
        first = int(ok.argmax())
        last = int(row.size - 1 - ok[::-1].argmax())
        inside = ~ok
        inside[:first] = False
        inside[last + 1:] = False
        if inside.any():
            row[inside] = np.interp(columns[inside], columns[ok], row[ok])
            filled += int(inside.sum())
    return mat, filled


def prepare_heatmap(matrix: np.ndarray, tvec: np.ndarray, mode: str = DEFAULT_MODE,
                    trim: bool = DEFAULT_TRIM) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Matrix and time axis to draw, plus a report of what was done.

    The input arrays are not modified. The report holds: ``trimmed`` (bool),
    ``trim_skipped`` (bool, the common span was too short), ``kept_fraction``,
    ``span_s`` (start, end of the drawn window), ``filled`` (samples
    interpolated) and ``missing`` (samples still missing in the drawn matrix).
    """
    mat = np.asarray(matrix, float)
    t = np.asarray(tvec, float)
    report: Dict[str, object] = {"trimmed": False, "trim_skipped": False, "kept_fraction": 1.0,
                                 "filled": 0, "missing": 0, "span_s": None}
    if mat.ndim != 2 or mat.size == 0 or t.size != mat.shape[1]:
        return mat, t, report
    if trim:
        start, stop, kept = common_span(mat)
        report["kept_fraction"] = kept
        if (start, stop) != (0, mat.shape[1]):
            if kept >= MIN_COMMON_FRACTION:
                mat, t = mat[:, start:stop], t[start:stop]
                report["trimmed"] = True
            else:
                report["trim_skipped"] = True
    if str(mode) == MODE_INTERPOLATE:
        mat, report["filled"] = interpolate_interior_gaps(mat)
    report["missing"] = int(np.count_nonzero(~np.isfinite(mat)))
    if t.size:
        report["span_s"] = (float(t[0]), float(t[-1]))
    return mat, t, report


def describe(report: Dict[str, object]) -> str:
    """One sentence for a tooltip: what the drawn heatmap differs by."""
    parts = []
    if report.get("trimmed") and report.get("span_s"):
        a, b = report["span_s"]
        parts.append(f"trimmed to the span shared by all rows ({a:.4g} to {b:.4g} s)")
    if report.get("trim_skipped"):
        parts.append(f"not trimmed: the shared span is only {100 * float(report['kept_fraction']):.0f}% "
                     "of the longest row")
    if report.get("filled"):
        parts.append(f"{int(report['filled'])} missing samples interpolated for display")
    if report.get("missing"):
        parts.append(f"{int(report['missing'])} samples without data")
    if not parts:
        return ""
    return "Display only (statistics use the original data): " + "; ".join(parts) + "."
