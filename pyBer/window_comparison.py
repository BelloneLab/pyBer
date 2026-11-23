"""Compare one PSTH metric across all metric windows at once (Pre, Post, Late, ...).

With only Pre and Post there is one comparison. As soon as a third window is
added ("Late"), three questions exist: Pre vs Post, Pre vs Late and Post vs
Late. This module answers them together, in one panel per metric:

==========  ==================================================================
omnibus     Friedman test across all windows, on the rows that have a value
            in every window (ranks within each row, no normality assumption).
            Effect size: Kendall's W (0 = no agreement, 1 = same order in
            every row).
pairwise    Exact two-sided paired sign test of every pair of windows, the
            same test as the standard Pre vs Post panel. All pairs of all the
            selected metrics form one Holm family.
==========  ==================================================================

Rules kept from the two-window panels: pooled trials of several animals are
not independent units (descriptive only), and a duration-sensitive metric
(AUC, peak, minimum, peak delay) is only tested between windows of equal
length.

The module is pure NumPy / SciPy. ``build_window_view`` returns the same plain
"view" description as ``group_metric_views``, so the on-screen (pyqtgraph) and
exported (matplotlib) panels are drawn by the same code and cannot disagree.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np

from group_metric_views import ALPHA, _column, bracket_levels, significance_stars
from group_stats import MIN_TEST_N, _holm
from postprocessing_core import paired_summary

# =============================================================================
# Configuration
# =============================================================================

#: Comparison code -> label shown in the "Compare windows" selector (menu order).
WINDOW_COMPARISONS: Dict[str, str] = {
    "all": "All windows together",
    "pre": "Each window with Pre",
}
DEFAULT_WINDOW_COMPARISON = "all"

#: Column colors. The extra windows use the color of their shading on the PSTH.
PRE_COLOR = "#4D93BD"
POST_COLOR = "#D28A64"
EXTRA_COLORS = ("#1BAF7A", "#EDA100", "#E87BA4", "#6250D6", "#008300", "#E34948")
LINK_COLOR = "#8893A1"          # lines joining the values of one row across windows

#: Up to this many windows every tested pair gets a bracket ("ns" included);
#: with more windows only the significant pairs are drawn.
MAX_WINDOWS_ALL_BRACKETS = 3
MAX_BRACKETS = 6                # brackets drawn at most (most significant first)
BRACKET_INSET = 0.08            # keeps neighbouring brackets from touching
SMALL_SAMPLE_N = 10             # below: the Friedman chi-square p-value is approximate


def window_color(index: int) -> str:
    """Color of the window at ``index`` (0 = Pre, 1 = Post, 2... = extra windows)."""
    if index == 0:
        return PRE_COLOR
    if index == 1:
        return POST_COLOR
    return EXTRA_COLORS[(index - 2) % len(EXTRA_COLORS)]


# =============================================================================
# Statistics
# =============================================================================

def friedman_test(values: Sequence[np.ndarray], independent_units: bool = True,
                  comparable: bool = True) -> Dict[str, object]:
    """Friedman test across windows, on the rows that are finite in every window.

    ``values``: one array per window, same length and row order. Returns the
    statistic, degrees of freedom, p-value, the number of complete rows and
    Kendall's W. The result is "Descriptive only" (p = NaN) when a requirement
    is not met; ``note`` says which one.
    """
    stack = np.vstack([np.asarray(v, float).reshape(-1) for v in values])
    count = stack.shape[0]
    data = stack[:, np.all(np.isfinite(stack), axis=0)]
    rows = int(data.shape[1])
    result: Dict[str, object] = {"test": "Descriptive only", "statistic": np.nan, "df": count - 1,
                                 "p": np.nan, "p_holm": np.nan, "n": rows, "kendall_w": np.nan, "note": ""}
    if count < 3:
        result["note"] = "An omnibus test needs at least three windows."
    elif not independent_units:
        result["note"] = ("Inference is suppressed because pooled trials are not independent "
                          "animal-level units.")
    elif not comparable:
        result["note"] = "Unequal window lengths: this duration-sensitive metric is descriptive only."
    elif rows < MIN_TEST_N:
        result["note"] = f"Fewer than {MIN_TEST_N} rows have a value in every window: descriptive only."
    else:
        from scipy.stats import friedmanchisquare
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore")
            statistic, pvalue = friedmanchisquare(*data)
        if not (np.isfinite(statistic) and np.isfinite(pvalue)):
            result["note"] = "The windows have identical values in every row: no test."
        else:
            note = ("Friedman test on the rows that have a value in every window. Values are ranked "
                    "within each row: no normality assumption; rows are assumed independent.")
            if rows < SMALL_SAMPLE_N:
                note += (f" Fewer than {SMALL_SAMPLE_N} complete rows: the chi-square p-value "
                         "is approximate.")
            result.update(test="Friedman test", statistic=float(statistic), p=float(pvalue),
                          kendall_w=float(statistic) / (rows * (count - 1)), note=note)
    return result


def compare_windows(windows: Sequence[Dict[str, object]], independent_units: bool = True,
                    duration_sensitive: bool = False) -> Dict[str, object]:
    """Omnibus and pairwise comparison of one metric across ``windows``.

    ``windows``: dicts with ``id``, ``label``, ``start``, ``end`` and ``values``
    (one value per row, same row order in every window). Returns
    ``{"omnibus": {...}, "pairs": [...]}``. The Holm adjustment (``p_holm``)
    is left to the caller, because the family spans all the selected metrics.
    """
    values = [np.asarray(window["values"], float).reshape(-1) for window in windows]
    lengths = [float(window["end"]) - float(window["start"]) for window in windows]

    def same_length(first: int, second: int) -> bool:
        return bool(np.isclose(lengths[first], lengths[second]))

    pairs: List[Dict[str, object]] = []
    for first in range(len(windows)):
        for second in range(first + 1, len(windows)):
            comparable = not duration_sensitive or same_length(first, second)
            test = paired_summary(values[first], values[second],
                                  independent_units=independent_units and comparable)
            both = np.isfinite(values[first]) & np.isfinite(values[second])
            difference = values[second][both] - values[first][both]
            note = str(test["assumption_note"])
            if not comparable:
                note += " Unequal window lengths: this duration-sensitive metric is descriptive only."
            pairs.append({
                "a": str(windows[first]["id"]), "b": str(windows[second]["id"]),
                "a_label": str(windows[first]["label"]), "b_label": str(windows[second]["label"]),
                "a_index": first, "b_index": second,
                "n_pairs": int(test["paired_n"]), "positive_n": int(test["paired_positive_n"]),
                "negative_n": int(test["paired_negative_n"]), "ties_n": int(test["paired_ties_n"]),
                "median_difference": float(np.median(difference)) if difference.size else np.nan,
                "mean_difference": float(np.mean(difference)) if difference.size else np.nan,
                "method": str(test["method"]), "p": float(test["paired_p"]), "p_holm": np.nan,
                "note": note,
            })
    all_comparable = not duration_sensitive or all(same_length(0, index) for index in range(1, len(windows)))
    return {"omnibus": friedman_test(values, independent_units, all_comparable), "pairs": pairs}


def adjust_families(comparisons: Sequence[Dict[str, object]]) -> int:
    """Holm-adjust in place: all pairs of all metrics together, and the omnibus tests together.

    Returns the size of the pairwise family (number of finite pairwise p-values).
    """
    pairs = [pair for comparison in comparisons for pair in comparison["pairs"]]
    for pair, adjusted in zip(pairs, _holm([pair["p"] for pair in pairs])):
        pair["p_holm"] = adjusted
    omnibus = [comparison["omnibus"] for comparison in comparisons]
    for item, adjusted in zip(omnibus, _holm([item["p"] for item in omnibus])):
        item["p_holm"] = adjusted
    family = int(sum(1 for item in omnibus if np.isfinite(item["p"])))
    for item in omnibus:
        item["family_size"] = family
    return int(sum(1 for pair in pairs if np.isfinite(pair["p"])))


# =============================================================================
# View description (drawn by group_metric_views)
# =============================================================================

def build_window_view(metric_label: str, windows: Sequence[Dict[str, object]],
                      comparison: Dict[str, object], row_unit: str = "trials") -> Dict[str, object]:
    """Describe the all-windows panel: one column per window, rows joined by lines.

    The result has the layout of ``group_metric_views.build_group_view`` and is
    drawn by ``draw_group_view`` (screen) and ``draw_group_view_matplotlib``
    (export) in the chosen summary style.
    """
    view: Dict[str, object] = {"mode": "windows", "quantity": "window value", "within": [], "columns": [],
                               "links": [], "ticks": [], "brackets": [], "zero_line": False}
    for index, window in enumerate(windows):
        column = _column(float(index), window["values"], window_color(index), str(window["label"]),
                         str(window["label"]), "window")
        view["columns"].append(column)
        view["ticks"].append((float(index), str(window["label"])))
    for index in range(len(windows) - 1):
        view["links"].append({"x0": float(index), "x1": float(index + 1),
                              "y0": view["columns"][index]["values"], "y1": view["columns"][index + 1]["values"],
                              "jitter": view["columns"][index]["jitter"], "color": LINK_COLOR})
    tested = [pair for pair in comparison["pairs"] if np.isfinite(pair["p_holm"])]
    if len(windows) > MAX_WINDOWS_ALL_BRACKETS:
        tested = [pair for pair in tested if pair["p_holm"] < ALPHA]
    shown = sorted(tested, key=lambda pair: pair["p_holm"])[:MAX_BRACKETS]
    spans = [(pair["a_index"] + BRACKET_INSET, pair["b_index"] - BRACKET_INSET) for pair in shown]
    for pair, (x0, x1), level in zip(shown, spans, bracket_levels(spans)):
        view["brackets"].append({"x0": x0, "x1": x1, "level": level,
                                 "text": f"{significance_stars(pair['p_holm'])} p={pair['p_holm']:.2g}"})
    labels = [str(window["label"]) for window in windows]
    suffix = " vs ".join(labels) if len(labels) <= 3 else f"{len(labels)} windows"
    view["title"] = f"{metric_label}: {suffix}" if metric_label else suffix
    omnibus = comparison["omnibus"]
    any_pair = any(np.isfinite(pair["p"]) for pair in comparison["pairs"])
    lines = []
    if np.isfinite(omnibus["p"]):
        corrected = int(omnibus.get("family_size", 1)) > 1
        pvalue = omnibus["p_holm"] if corrected else omnibus["p"]
        lines.append(f"Friedman χ²({omnibus['df']})={omnibus['statistic']:.3g}, "
                     f"{'Holm p' if corrected else 'p'}={pvalue:.3g}, n={omnibus['n']}")
    if any_pair:
        lines.append("Pairs: sign test, Holm · {style}")
    view["note"] = "\n".join(lines)
    tooltip = [f"Each line follows one of the {row_unit} across the windows."]
    if any_pair:
        tooltip.append("Brackets: exact paired sign test of two windows, Holm-adjusted across all window "
                       "pairs and selected metrics"
                       + ("." if len(windows) <= MAX_WINDOWS_ALL_BRACKETS
                          else f"; only pairs with p < {ALPHA:g} are drawn."))
    if omnibus["note"]:
        tooltip.append(str(omnibus["note"]))
    if comparison["pairs"]:
        tooltip.append(str(comparison["pairs"][0]["note"]))
    view["tooltip"] = " ".join(tooltip)
    return view


# =============================================================================
# Export rows
# =============================================================================

def window_rows(metric_key: str, result: Dict[str, object]) -> List[Dict[str, object]]:
    """One CSV row per window: label, limits and descriptive statistics."""
    units = result["summary"]["units"]
    return [{"metric_id": metric_key, "window_id": window["id"], "window_label": window["label"],
             "start_s": window["start"], "end_s": window["end"], "units": units, "n": window["n"],
             "mean": window["mean"], "median": window["median"], "sem": window["sem"]}
            for window in result.get("windows", [])]


def pair_rows(metric_key: str, result: Dict[str, object]) -> List[Dict[str, object]]:
    """One CSV row per pair of windows: difference (b minus a), test and Holm p-value."""
    comparison = result.get("window_comparison") or {"pairs": []}
    family = result["summary"].get("family_size", 0)
    return [{"metric_id": metric_key, "window_a": pair["a_label"], "window_b": pair["b_label"],
             "window_a_id": pair["a"], "window_b_id": pair["b"], "n_pairs": pair["n_pairs"],
             "positive_n": pair["positive_n"], "negative_n": pair["negative_n"], "ties_n": pair["ties_n"],
             "median_difference_b_minus_a": pair["median_difference"],
             "mean_difference_b_minus_a": pair["mean_difference"], "method": pair["method"],
             "p": pair["p"], "p_holm": pair["p_holm"], "family_size": family, "note": pair["note"]}
            for pair in comparison["pairs"]]


def value_rows(metric_key: str, result: Dict[str, object]) -> List[Dict[str, object]]:
    """Long-format CSV rows: the metric of every displayed row in every window."""
    summary = result["summary"]
    return [{"metric_id": metric_key, "row": row, "row_unit": summary["row_unit"], "units": summary["units"],
             "window_id": window["id"], "window_label": window["label"], "value": value}
            for window in result.get("windows", [])
            for row, value in enumerate(np.asarray(window["values"], float), 1)]
