"""Selectable representations of a pre/post metric across recording groups.

The group view of a PSTH metric can answer different questions. Each
representation ("view") picks the quantity, the statistics that match it and
the on-plot annotation:

=========  ============================================  ==========================================
view       what is drawn (one point = one animal)        statistics
=========  ============================================  ==========================================
delta      Post - Pre per group                          between groups (on the change)
post       Post per group                                between groups (on Post)
pre        Pre per group                                 between groups (on Pre, baseline check)
paired     Pre next to Post per group, same animal       within group: paired Pre vs Post (Holm
           joined by a line                              across groups); between groups: on change
overlay    groups share the window ticks: bold mean      group x window interaction (mixed ANOVA,
           +/- SEM line per group, thin animal lines     Greenhouse-Geisser beyond two windows)
=========  ============================================  ==========================================

Between-group tests are chosen by ``group_stats.compare_groups`` (Welch t /
Mann-Whitney U, ANOVA / Kruskal-Wallis with Holm pairwise). The paired test is
a paired t-test when the differences pass Shapiro-Wilk, otherwise the Wilcoxon
signed-rank test. The unit of analysis is always the animal / recording.

Three summary styles draw the same numbers: points with median and mean +/-
SEM, bars (mean +/- SEM) with points, or box plots with points. One view
description (a plain dict) feeds both the on-screen (pyqtgraph) and the
exported (matplotlib) panel, so they cannot disagree.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from group_stats import MIN_TEST_N, _holm, compare_groups, describe, stats_label, window_interaction

# =============================================================================
# Configuration
# =============================================================================

#: Representation code -> label shown in menus (order = menu order).
GROUP_VIEWS: Dict[str, str] = {
    "delta": "Post − Pre (change) per group",
    "post": "Post only per group",
    "pre": "Pre only per group",
    "paired": "Pre and Post side by side, paired lines",
    "overlay": "Groups overlaid, mean ± SEM line per group",
}
DEFAULT_GROUP_VIEW = "delta"
#: Summary style code -> label.
SUMMARY_STYLES: Dict[str, str] = {
    "points": "Points, median line, mean ± SEM",
    "bars": "Bars (mean ± SEM) with points",
    "box": "Box plot with points",
}
DEFAULT_SUMMARY_STYLE = "points"

ALPHA = 0.05                 # significance level for assumption checks and stars
MAX_BRACKETS = 3             # pairwise brackets drawn (most significant first)
JITTER = 0.14                # half-width of the point cloud of one column
PAIR_GAP = 2.6               # x distance between two groups in the paired view
GROUP_WINDOW_GAP = 0.8       # free space between two groups in the all-windows view (in columns)
OVERLAY_DODGE = 0.07         # x offset between neighbouring groups in the overlaid view
OVERLAY_LINE_WIDTH = 2.6     # group mean line of the overlaid view
OVERLAY_MARKER_SIZE = 9      # group mean marker (pixels on screen)
#: Short quantity name written to the CSV exports.
QUANTITY = {"delta": "post_minus_pre", "post": "post", "pre": "pre", "paired": "post_minus_pre",
            "overlay": "post_minus_pre"}


# =============================================================================
# Statistics
# =============================================================================

def significance_stars(p: float) -> str:
    """'***' p<0.001, '**' p<0.01, '*' p<0.05, 'ns' otherwise, '' when undefined."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < ALPHA else "ns"


def paired_test(pre: Sequence[float], post: Sequence[float], alpha: float = ALPHA) -> Dict[str, object]:
    """Paired Pre vs Post test of one group (animals with both values).

    Paired t-test when the differences pass Shapiro-Wilk (p > alpha), otherwise
    Wilcoxon signed-rank. Effect size: Cohen's dz (t) or the matched-pairs
    rank-biserial correlation (Wilcoxon). Fewer than ``MIN_TEST_N`` pairs, or
    identical differences, are descriptive only.
    """
    from scipy import stats
    a, b = np.asarray(pre, float), np.asarray(post, float)
    keep = np.isfinite(a) & np.isfinite(b)
    diff = b[keep] - a[keep]
    result: Dict[str, object] = {"n_pairs": int(diff.size), "test": "Descriptive only", "statistic": np.nan,
                                 "p": np.nan, "p_holm": np.nan, "effect_name": "", "effect": np.nan,
                                 "mean_difference": float(np.mean(diff)) if diff.size else np.nan,
                                 "normality": "", "note": ""}
    if diff.size < MIN_TEST_N:
        result["note"] = f"Fewer than {MIN_TEST_N} paired animals: descriptive only."
        return result
    if np.ptp(diff) == 0:
        result["note"] = "All paired differences are identical: no test."
        return result
    shapiro_p = float(stats.shapiro(diff).pvalue)
    result["normality"] = f"Shapiro-Wilk p (differences)={shapiro_p:.3g}"
    if shapiro_p > alpha:
        test = stats.ttest_rel(b[keep], a[keep])
        result.update(test="Paired t-test", statistic=float(test.statistic), p=float(test.pvalue),
                      effect_name="Cohen dz", effect=float(np.mean(diff) / np.std(diff, ddof=1)),
                      note="Differences normal: paired t-test.")
        return result
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return result
    test = stats.wilcoxon(nonzero)
    ranks = stats.rankdata(np.abs(nonzero))
    positive, negative = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    result.update(test="Wilcoxon signed-rank", statistic=float(test.statistic), p=float(test.pvalue),
                  effect_name="rank-biserial r", effect=(positive - negative) / (positive + negative),
                  note="Normality of differences rejected: Wilcoxon signed-rank.")
    return result


def _was_tested(comparison: Dict[str, object]) -> bool:
    """True when a between-group test produced a p-value (not descriptive only)."""
    return bool(np.isfinite(float(comparison.get("p", np.nan))))


def _column(x: float, values: np.ndarray, color: str, label: str, group: str, role: str,
            faded: bool = False) -> Dict[str, object]:
    """One drawn column: per-animal values (NaN kept for alignment) and its summaries."""
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    summary = describe(v)
    quartiles = np.percentile(finite, [25, 75]) if finite.size else (np.nan, np.nan)
    jitter = np.linspace(-JITTER, JITTER, v.size) if v.size > 1 else np.zeros(v.size)
    return {"x": float(x), "values": v, "jitter": jitter, "color": color, "label": label, "group": group,
            "role": role, "faded": faded, "q1": float(quartiles[0]), "q3": float(quartiles[1]),
            # Side of the mean +/- SEM marker: outside the pair so it never sits on the joining lines.
            "side": -1.0 if faded else 1.0, **summary}


def _pairwise_brackets(comparison: Dict[str, object], x_of: Dict[str, float]) -> List[Dict[str, object]]:
    """Brackets for the significant Holm-adjusted pairs (2 groups: the single test)."""
    brackets = []
    pairs = comparison.get("pairwise") or []
    if pairs:
        for pair in sorted(pairs, key=lambda item: item.get("p_holm", np.nan)):
            p = float(pair.get("p_holm", np.nan))
            if np.isfinite(p) and p < ALPHA and pair["a"] in x_of and pair["b"] in x_of:
                brackets.append({"x0": x_of[pair["a"]], "x1": x_of[pair["b"]],
                                 "text": f"{significance_stars(p)} p={p:.2g}"})
    elif np.isfinite(float(comparison.get("p", np.nan))) and float(comparison["p"]) < ALPHA:
        tested = [g["name"] for g in comparison["groups"] if g["tested"] and g["n"] > 0]
        if len(tested) == 2:
            p = float(comparison["p"])
            brackets.append({"x0": x_of[tested[0]], "x1": x_of[tested[1]],
                             "text": f"{significance_stars(p)} p={p:.2g}"})
    return brackets[:MAX_BRACKETS]


def bracket_levels(spans: Sequence[tuple]) -> List[int]:
    """Lowest free level of each (x0, x1) bracket; short brackets are placed first."""
    order = sorted(range(len(spans)), key=lambda index: (spans[index][1] - spans[index][0], spans[index][0]))
    taken: List[List[tuple]] = []
    levels = [0] * len(spans)
    for index in order:
        x0, x1 = spans[index]
        level = 0
        while level < len(taken) and any(x0 < other1 and other0 < x1 for other0, other1 in taken[level]):
            level += 1
        if level == len(taken):
            taken.append([])
        taken[level].append((x0, x1))
        levels[index] = level
    return levels


def build_group_window_view(groups: Sequence[Dict[str, object]], labels: Sequence[str],
                            metric_label: str = "", overlay: bool = False,
                            limits: Optional[Sequence[tuple]] = None) -> Dict[str, object]:
    """Every metric window for each group (Pre, Post, Late, ...), in one of two layouts.

    * side by side (default): the paired Pre / Post view extended to any number
      of windows. One cluster of columns per group, one tick per window, each
      animal joined by a line across its windows, in the chosen summary style.
    * ``overlay``: the groups share one set of window ticks. Each group is a
      bold mean +/- SEM line across the windows; the animals are thin faint
      lines behind it. The groups are named in a colored legend.

    ``groups``: dicts with ``name``, ``color``, ``windows`` (one array per
    window, one value per animal, same animal order), optional ``ids`` and
    ``tested``. ``labels``: the window names, Pre first. ``limits``: optional
    (start, end) in seconds of each window, written under its tick in the
    overlaid layout.

    Statistics (unit = animal): within each group every pair of windows gets
    the paired test of ``paired_test`` (t-test or Wilcoxon), Holm-adjusted
    across all pairs and groups (brackets in the side-by-side layout). Between
    groups, the change from the first window (Post - Pre, Late - Pre, ...) is
    compared with ``compare_groups``, Holm-adjusted across windows, and the
    group x window interaction is tested by ``group_stats.window_interaction``
    (annotated in the overlaid layout, exported in both).
    """
    labels = [str(label) for label in labels]
    count = len(labels)
    gap = count + GROUP_WINDOW_GAP
    view: Dict[str, object] = {"mode": "overlay" if overlay else "group_windows",
                               "quantity": QUANTITY["paired"], "within": [], "columns": [], "links": [],
                               "ticks": [], "brackets": [], "zero_line": False}
    within: List[Dict[str, object]] = []
    if overlay:
        view["mean_lines"], view["legend"] = [], []
        for w in range(count):
            span = limits[w] if limits is not None and w < len(limits) else None
            view["ticks"].append((float(w), f"{labels[w]}\n{span[0]:g} to {span[1]:g} s" if span else labels[w]))
    for index, group in enumerate(groups):
        color = group.get("color", "#888888")
        values = [np.asarray(window, float) for window in group["windows"]]
        complete = int(np.count_nonzero(np.all(np.isfinite(np.vstack(values)), axis=0))) if values[0].size else 0
        if overlay:
            # Groups are dodged slightly around each window so their error bars do not hide each other.
            base = (index - (len(groups) - 1) / 2.0) * OVERLAY_DODGE
            columns = [_column(base + w, values[w], color, labels[w], group["name"], "window", faded=True)
                       for w in range(count)]
            for column in columns:
                column["jitter"] = np.zeros(column["values"].size)     # one animal = one straight thin line
                column["summary"] = False                              # the group mean line is the summary
            view["mean_lines"].append({"x": [column["x"] for column in columns],
                                       "mean": [column["mean"] for column in columns],
                                       "sem": [column["sem"] for column in columns],
                                       "color": color, "name": str(group["name"])})
            view["legend"].append({"name": f"{group['name']}, n={complete}", "color": color})
        else:
            base = index * gap
            columns = [_column(base + w, values[w], color, labels[w], group["name"], "window", faded=(w == 0))
                       for w in range(count)]
            caption = f"{group['name']}\nn={complete}"
            middle = (count - 1) // 2 if count % 2 else None
            for w in range(count):
                # The group caption goes under the middle window (odd count) or between the two middle ones.
                view["ticks"].append((base + w, f"{labels[w]}\n{caption}" if w == middle else labels[w]))
            if middle is None:
                view["ticks"].append((base + (count - 1) / 2.0, "\n" + caption))
        view["columns"].extend(columns)
        for w in range(count - 1):
            view["links"].append({"x0": base + w, "x1": base + w + 1.0, "y0": values[w], "y1": values[w + 1],
                                  "jitter": columns[w]["jitter"], "color": color, "faint": overlay})
        for first in range(count):
            for second in range(first + 1, count):
                test = paired_test(values[first], values[second])
                test.update(group=f"{group['name']}: {labels[first]} vs {labels[second]}",
                            tested=bool(group.get("tested", True)), x0=base + first, x1=base + second)
                within.append(test)
    tested = [item for item in within if item["tested"]]
    for item, adjusted in zip(tested, _holm([item["p"] for item in tested])):
        item["p_holm"] = adjusted
    # Overlaid groups share the ticks: a bracket could not say which group it belongs to.
    shown = [] if overlay else [item for item in tested if np.isfinite(float(item["p_holm"]))]
    if count > 3:   # many windows: only the significant pairs, to keep the panel readable
        shown = [item for item in shown if item["p_holm"] < ALPHA]
    spans = [(item["x0"] + 0.08, item["x1"] - 0.08) for item in shown]
    for item, (x0, x1), level in zip(shown, spans, bracket_levels(spans)):
        # Stars only: neighbouring brackets are one column wide, a p-value would overlap the next
        # label. The exact values are in the tooltip-free CSV export (..._group_paired.csv).
        view["brackets"].append({"x0": x0, "x1": x1, "level": level,
                                 "text": significance_stars(item["p_holm"])})
    # Group x window interaction (tested groups, animals with every window). Stored with the
    # within-group tests so it is exported with them; it never gets a bracket.
    interaction = window_interaction([np.column_stack([np.asarray(window, float) for window in group["windows"]])
                                      for group in groups if group.get("tested", True)])
    view["interaction"] = interaction
    within.append({"group": "Group x window interaction", "tested": True, "n_pairs": interaction["n"],
                   "mean_difference": np.nan, "test": interaction["test"], "statistic": interaction["statistic"],
                   "p": interaction["p"], "p_holm": np.nan, "effect_name": interaction["effect_name"],
                   "effect": interaction["effect"], "normality": interaction["normality"],
                   "note": interaction["note"]})
    view["within"] = within
    # Between groups: the change from the first window, one test per later window.
    between = []
    for w in range(1, count):
        between.append(compare_groups([
            {"name": g["name"], "color": g.get("color", "#888888"), "ids": g.get("ids", []),
             "tested": g.get("tested", True),
             "values": np.asarray(g["windows"][w], float) - np.asarray(g["windows"][0], float)} for g in groups]))
    adjusted = _holm([float(item.get("p", np.nan)) for item in between])
    view["between"] = between[0]
    view["between_windows"] = [{"window": labels[w + 1], "comparison": item, "p_holm": p}
                               for w, (item, p) in enumerate(zip(between, adjusted))]
    suffix = " vs ".join(labels) if count <= 3 else f"{count} windows"
    view["title"] = f"{metric_label}: {suffix}" if metric_label else suffix
    lines = []
    families = sorted({str(item["test"]) for item in tested if np.isfinite(float(item["p"]))})
    changes = [f"{entry['window']} p={entry['p_holm']:.2g}" for entry in view["between_windows"]
               if np.isfinite(entry["p_holm"])]
    interaction_text = ""
    if np.isfinite(float(interaction["p"])):
        interaction_text = (f"Group × window: F({interaction['df1']:.3g}, {interaction['df2']:.3g})="
                            f"{interaction['statistic']:.3g}, p={interaction['p']:.3g}")
    if overlay:
        if interaction_text:
            lines.append(interaction_text)
        view["tooltip"] = ("Bold lines: group mean ± SEM in each window. Thin lines: one animal each (trials "
                           "averaged per recording). Group × window tests whether the groups follow different "
                           "courses across the windows. " + str(interaction["note"])
                           + (" " + str(interaction["normality"]) if interaction["normality"] else ""))
    else:
        if families:
            lines.append("Within group: " + " / ".join(families) + ", Holm")
        if changes:
            lines.append(f"Change from {labels[0]} between groups (Holm): " + ", ".join(changes))
        view["tooltip"] = ("Each line joins the values of one animal across the windows (trials averaged per "
                           "recording). Brackets: paired test of two windows within a group, Holm-adjusted "
                           "across all pairs and groups (no bracket: not tested, fewer than 3 animals or "
                           f"ungrouped). Between groups: the change from {labels[0]} is compared for each later "
                           "window, Holm-adjusted across windows. " + str(between[0]["note"])
                           + (" " + interaction_text + "." if interaction_text else ""))
    view["note"] = "\n".join(lines)
    return view


def build_group_view(mode: str, groups: Sequence[Dict[str, object]], metric_label: str = "",
                     pre_label: str = "Pre", post_label: str = "Post") -> Dict[str, object]:
    """Describe one representation: columns, paired lines, statistics and annotations.

    ``groups``: dicts with ``name``, ``color``, ``pre`` and ``post`` (one value
    per animal, same order), optional ``ids`` and ``tested`` (False for
    "Ungrouped": summarized, never tested). ``pre_label`` / ``post_label``:
    the window names shown on the axis and in the title (renamed or extra
    windows).
    """
    mode = mode if mode in GROUP_VIEWS else DEFAULT_GROUP_VIEW
    if mode == "overlay":   # the overlaid layout with the two standard windows
        return build_group_window_view([{**group, "windows": [group["pre"], group["post"]]} for group in groups],
                                       [pre_label, post_label], metric_label, overlay=True)
    prepared = []
    for group in groups:
        pre, post = np.asarray(group["pre"], float), np.asarray(group["post"], float)
        prepared.append({**group, "pre": pre, "post": post, "delta": post - pre})
    quantity = "delta" if mode == "paired" else mode
    between = compare_groups([{"name": g["name"], "color": g.get("color", "#888888"), "values": g[quantity],
                               "ids": g.get("ids", []), "tested": g.get("tested", True)} for g in prepared])
    view: Dict[str, object] = {"mode": mode, "quantity": QUANTITY[mode], "between": between, "within": [],
                               "columns": [], "links": [], "ticks": [], "brackets": [], "zero_line": False}
    suffix = {"delta": f"{post_label} − {pre_label}", "post": post_label, "pre": pre_label,
              "paired": f"{pre_label} vs {post_label}"}[mode]
    view["title"] = f"{metric_label}: {suffix}" if metric_label else suffix
    unit_note = "animals · "
    if mode == "paired":
        within = []
        for index, group in enumerate(prepared):
            x_pre, x_post = index * PAIR_GAP, index * PAIR_GAP + 1.0
            pre_col = _column(x_pre, group["pre"], group.get("color", "#888888"), pre_label, group["name"], "pre", True)
            post_col = _column(x_post, group["post"], group.get("color", "#888888"), post_label, group["name"], "post")
            view["columns"].extend([pre_col, post_col])
            view["links"].append({"x0": x_pre, "x1": x_post, "y0": group["pre"], "y1": group["post"],
                                  "jitter": pre_col["jitter"], "color": group.get("color", "#888888")})
            n_pairs = int(np.count_nonzero(np.isfinite(group["pre"]) & np.isfinite(group["post"])))
            view["ticks"].extend([(x_pre, pre_label), (x_post, post_label),
                                  ((x_pre + x_post) / 2, f"\n{group['name']}\nn={n_pairs}")])
            test = paired_test(group["pre"], group["post"])
            test.update(group=group["name"], tested=bool(group.get("tested", True)))
            within.append(test)
        tested = [item for item in within if item["tested"]]
        for item, adjusted in zip(tested, _holm([item["p"] for item in tested])):
            item["p_holm"] = adjusted
        for index, item in enumerate(within):
            p = float(item["p_holm"]) if item["tested"] else np.nan
            if np.isfinite(p):   # untested groups (fewer than 3 pairs, ungrouped) get no label
                view["brackets"].append({"x0": index * PAIR_GAP, "x1": index * PAIR_GAP + 1.0,
                                         "text": f"{significance_stars(p)} p={p:.2g}", "level": 0})
        view["within"] = within
        families = sorted({str(item["test"]) for item in tested if np.isfinite(float(item["p"]))})
        # Only tests that were actually run are written on the plot: a
        # "descriptive only" line carries no information.
        lines = []
        if families:
            lines.append(f"{pre_label} vs {post_label} within group: " + " / ".join(families)
                         + ", Holm across groups")
        if _was_tested(between):
            lines.append(f"Change between groups: {stats_label(between)}")
        view["note"] = "\n".join(lines)
        view["tooltip"] = ("Each line joins the Pre and Post value of one animal (trials averaged per recording). "
                           "Above each pair: paired Pre vs Post test of that group, Holm-adjusted across groups "
                           "(no label: not tested, fewer than 3 pairs or ungrouped). "
                           "Change between groups tests Post − Pre across groups. " + str(between["note"]))
    else:
        x_of = {}
        for index, group in enumerate(prepared):
            column = _column(index, group[mode], group.get("color", "#888888"), group["name"], group["name"], mode)
            view["columns"].append(column)
            view["ticks"].append((float(index), f"{group['name']}\nn={column['n']}"))
            x_of[group["name"]] = float(index)
        view["brackets"] = _pairwise_brackets(between, x_of)
        view["zero_line"] = mode == "delta"
        view["note"] = (stats_label(between) + "\n" + unit_note + "{style}") if _was_tested(between) else ""
        what = {"delta": "Post − Pre", "post": "the Post window value", "pre": "the Pre window value"}[mode]
        view["tooltip"] = (f"Each point is one animal (trials averaged per recording): {what}. "
                           + str(between["note"])
                           + (" " + between["normality"] if between["normality"] else "")
                           + (" " + between["variance"] if between["variance"] else "")
                           + (" Brackets: Holm-adjusted pairwise tests with p < 0.05." if view["brackets"] else ""))
    return view


def view_from_comparison(comparison: Dict[str, object], title: str = "") -> Dict[str, object]:
    """Wrap an existing between-group comparison (one value per recording) as a drawable view."""
    view: Dict[str, object] = {"mode": "values", "quantity": "value", "between": comparison, "within": [],
                               "columns": [], "links": [], "ticks": [], "zero_line": False, "title": title}
    x_of = {}
    for index, group in enumerate(comparison.get("groups", [])):
        column = _column(index, group["values"], group["color"], group["name"], group["name"], "values")
        view["columns"].append(column)
        view["ticks"].append((float(index), f"{group['name']}\nn={column['n']}"))
        x_of[group["name"]] = float(index)
    view["brackets"] = _pairwise_brackets(comparison, x_of)
    view["note"] = (stats_label(comparison) + "\nrecordings · {style}") if _was_tested(comparison) else ""
    view["tooltip"] = str(comparison.get("note", ""))
    return view


def style_legend(style: str) -> str:
    """Short reading key of a summary style for the on-plot note."""
    return {"points": "median line | mean ± SEM", "bars": "bar: mean ± SEM",
            "box": "box: median, quartiles, 1.5 IQR whiskers"}.get(style, "")


def view_note(view: Dict[str, object], style: str) -> str:
    """On-plot annotation of a view in the chosen style."""
    return str(view.get("note", "")).replace("{style}", style_legend(style))


def view_bounds(view: Dict[str, object], style: str):
    """(low, high) of everything drawn for the data, before annotations; None when empty."""
    values: List[float] = []
    for column in view["columns"]:
        finite = column["values"][np.isfinite(column["values"])]
        values.extend(finite.tolist())
        if np.isfinite(column["mean"]):
            sem = column["sem"] if np.isfinite(column["sem"]) else 0.0
            values.extend([column["mean"] - sem, column["mean"] + sem])
    if not values:
        return None
    low, high = min(values), max(values)
    if style == "bars" or view.get("zero_line"):
        low, high = min(low, 0.0), max(high, 0.0)   # bars start at zero
    return low, high


def bracket_layout(view: Dict[str, object], low: float, high: float):
    """Vertical placement of the brackets above the data.

    Returns (list of (x0, x1, y, text), new high). Paired brackets share one
    level; pairwise brackets are stacked, most significant lowest.
    """
    span = high - low or max(abs(high) * 0.1, 1.0)
    placed = []
    level_count = 0
    for index, bracket in enumerate(view.get("brackets", [])):
        level = int(bracket.get("level", index))
        level_count = max(level_count, level + 1)
        placed.append((bracket["x0"], bracket["x1"], high + span * (0.07 + 0.24 * level), bracket["text"]))
    return placed, high + span * 0.24 * level_count


def whiskers(column: Dict[str, object]):
    """Tukey whisker ends: most extreme values within 1.5 IQR of the quartiles."""
    finite = column["values"][np.isfinite(column["values"])]
    if not finite.size:
        return np.nan, np.nan
    iqr = column["q3"] - column["q1"]
    inside = finite[(finite >= column["q1"] - 1.5 * iqr) & (finite <= column["q3"] + 1.5 * iqr)]
    return (float(inside.min()), float(inside.max())) if inside.size else (column["q1"], column["q3"])


def within_rows(view: Dict[str, object], **fixed) -> List[Dict[str, object]]:
    """Flat CSV rows of the paired within-group tests (empty for other views)."""
    return [{**fixed, "group": item["group"], "tested": item["tested"], "n_pairs": item["n_pairs"],
             "mean_difference": item["mean_difference"], "test": item["test"], "statistic": item["statistic"],
             "p": item["p"], "p_holm": item["p_holm"], "effect_name": item["effect_name"],
             "effect": item["effect"], "normality": item["normality"], "note": item["note"]}
            for item in view.get("within", [])]


# =============================================================================
# On-screen drawing (pyqtgraph)
# =============================================================================

def draw_group_view(plot, view: Dict[str, object], text_color: str, style: str = DEFAULT_SUMMARY_STYLE) -> list:
    """Draw ``view`` on a pyqtgraph PlotWidget; returns the created items.

    The x axis is labelled from the view, the y range is fitted to the data
    and its brackets. The caller places the note (it owns the headroom).
    """
    import pyqtgraph as pg
    from PySide6 import QtGui
    style = style if style in SUMMARY_STYLES else DEFAULT_SUMMARY_STYLE
    if view.get("mean_lines"):
        style = DEFAULT_SUMMARY_STYLE          # the overlaid view has its own summary: the mean line
    items: list = []

    def tint(color: str, alpha: int) -> QtGui.QColor:
        shade = QtGui.QColor(color)
        shade.setAlpha(alpha)
        return shade

    for link in view["links"]:
        xs, ys = [], []
        for jitter, a, b in zip(link["jitter"], link["y0"], link["y1"]):
            if np.isfinite(a) and np.isfinite(b):
                xs.extend([link["x0"] + jitter, link["x1"] + jitter, np.nan])
                ys.extend([a, b, np.nan])
        if xs:
            alpha = 70 if link.get("faint") else 110
            items.append(pg.PlotDataItem(xs, ys, connect="finite", pen=pg.mkPen(tint(link["color"], alpha), width=1.0)))
    for column in view["columns"]:
        x, values = column["x"], column["values"]
        keep = np.isfinite(values)
        if not keep.any():
            continue
        color = column["color"]
        if column.get("summary") is False:
            # Overlaid view: the animals are small open markers behind the group mean line.
            items.append(pg.ScatterPlotItem(x=x + column["jitter"][keep], y=values[keep], size=5,
                                            pen=pg.mkPen(tint(color, 120), width=1), brush=pg.mkBrush(tint(color, 30))))
            continue
        point_alpha = 150 if column["faded"] else 255
        mean, sem = column["mean"], column["sem"] if np.isfinite(column["sem"]) else 0.0
        if style == "bars":
            bar = pg.BarGraphItem(x=[x], height=[mean], width=0.62, brush=pg.mkBrush(tint(color, 70)),
                                  pen=pg.mkPen(tint(color, 220), width=1.2))
            error = pg.ErrorBarItem(x=np.array([x]), y=np.array([mean]), top=np.array([sem]),
                                    bottom=np.array([sem]), beam=0.16, pen=pg.mkPen(text_color, width=1.2))
            items.extend([bar, error])
        elif style == "box":
            low, high = whiskers(column)
            box = pg.BarGraphItem(x=[x], y0=[column["q1"]], height=[column["q3"] - column["q1"]], width=0.56,
                                  brush=pg.mkBrush(tint(color, 55)), pen=pg.mkPen(tint(color, 220), width=1.2))
            median = pg.PlotDataItem([x - 0.28, x + 0.28], [column["median"]] * 2, pen=pg.mkPen(color, width=2.5))
            stems = pg.PlotDataItem([x, x, np.nan, x, x], [low, column["q1"], np.nan, column["q3"], high],
                                    connect="finite", pen=pg.mkPen(tint(color, 220), width=1.2))
            items.extend([box, stems, median])
        # Optional one color per value (rows tinted by recording group in the windows view).
        point_colors = column.get("point_colors")
        if point_colors is not None and len(point_colors) == len(values):
            brush = [pg.mkBrush(tint(pg.mkColor(shade), point_alpha))
                     for shade, kept in zip(point_colors, keep) if kept]
        else:
            brush = pg.mkBrush(tint(color, point_alpha))
        scatter = pg.ScatterPlotItem(x=x + column["jitter"][keep], y=values[keep], size=6, pen=None, brush=brush)
        items.append(scatter)
        if style == "points":
            items.append(pg.PlotDataItem([x - 0.24, x + 0.24], [column["median"]] * 2,
                                         pen=pg.mkPen(tint(color, point_alpha), width=2.5)))
            if np.isfinite(mean):
                offset = 0.32 * column["side"]
                items.append(pg.ScatterPlotItem(x=[x + offset], y=[mean], size=8, symbol="d", pen=None,
                                                brush=pg.mkBrush(text_color)))
                items.append(pg.ErrorBarItem(x=np.array([x + offset]), y=np.array([mean]), top=np.array([sem]),
                                             bottom=np.array([sem]), beam=0.1, pen=pg.mkPen(text_color, width=1)))
    for line in view.get("mean_lines", []):
        # Group mean +/- SEM across the windows (overlaid view), drawn above the animals.
        xs, means = np.asarray(line["x"], float), np.asarray(line["mean"], float)
        sems = np.nan_to_num(np.asarray(line["sem"], float), nan=0.0)
        ok = np.isfinite(means)
        if not ok.any():
            continue
        curve = pg.PlotDataItem(xs[ok], means[ok], pen=pg.mkPen(line["color"], width=OVERLAY_LINE_WIDTH),
                                symbol="o", symbolSize=OVERLAY_MARKER_SIZE, symbolBrush=pg.mkBrush(line["color"]),
                                symbolPen=pg.mkPen(line["color"]))
        error = pg.ErrorBarItem(x=xs[ok], y=means[ok], top=sems[ok], bottom=sems[ok], beam=0.09,
                                pen=pg.mkPen(line["color"], width=2))
        for item in (error, curve):
            item.setZValue(5)
            items.append(item)
    bounds = view_bounds(view, style)
    if bounds is not None:
        low, high = bounds
        placed, top = bracket_layout(view, low, high)
        span = high - low or max(abs(high) * 0.1, 1.0)
        for x0, x1, y, text in placed:
            tick = span * 0.025
            items.append(pg.PlotDataItem([x0, x0, x1, x1], [y - tick, y, y, y - tick],
                                         pen=pg.mkPen(text_color, width=1)))
            label = pg.TextItem(text, color=text_color, anchor=(0.5, 1.0))
            label.setFont(QtGui.QFont("Segoe UI", 7))
            label.setPos((x0 + x1) / 2, y)
            items.append(label)
        if view.get("zero_line") or style == "bars":
            items.append(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(tint(text_color, 90), width=1)))
        view["_drawn_bounds"] = (low, top + (span * 0.08 if placed else 0.0))
        plot.setYRange(low - span * 0.08, top + span * 0.45, padding=0)
    for item in items:
        plot.addItem(item)
    axis = plot.getAxis("bottom")
    axis.setTicks([view["ticks"]])
    # Multi-line tick labels (group name, n) need the axis to be tall enough:
    # the automatic height only counts one line and clipped "n=...".
    lines = max((str(text).count("\n") + 1 for _x, text in view["ticks"]), default=1)
    metrics = QtGui.QFontMetrics(axis.style.get("tickFont") or plot.font())
    axis.setHeight(int(lines * metrics.lineSpacing() + axis.style.get("tickTextOffset", [5, 2])[1] + 12))
    xs = [column["x"] for column in view["columns"]] or [0.0]
    plot.setXRange(min(xs) - 0.6, max(xs) + 0.6, padding=0)
    return items


# =============================================================================
# Export drawing (matplotlib)
# =============================================================================

def draw_group_view_matplotlib(axis, view: Dict[str, object], style: str = DEFAULT_SUMMARY_STYLE,
                               title: Optional[str] = None, units: str = "") -> None:
    """Publication panel of ``view`` with the same numbers and annotations as on screen."""
    style = style if style in SUMMARY_STYLES else DEFAULT_SUMMARY_STYLE
    if view.get("mean_lines"):
        style = DEFAULT_SUMMARY_STYLE          # the overlaid view has its own summary: the mean line
    neutral = "#526371"
    for link in view["links"]:
        for jitter, a, b in zip(link["jitter"], link["y0"], link["y1"]):
            if np.isfinite(a) and np.isfinite(b):
                axis.plot([link["x0"] + jitter, link["x1"] + jitter], [a, b], color=link["color"],
                          alpha=0.28 if link.get("faint") else 0.4, linewidth=0.7, zorder=1)
    for line, entry in zip(view.get("mean_lines", []), view.get("legend", [])):
        # Group mean +/- SEM across the windows (overlaid view), with its legend entry.
        xs, means = np.asarray(line["x"], float), np.asarray(line["mean"], float)
        sems = np.nan_to_num(np.asarray(line["sem"], float), nan=0.0)
        ok = np.isfinite(means)
        if ok.any():
            axis.errorbar(xs[ok], means[ok], yerr=sems[ok], color=line["color"], linewidth=1.8, marker="o",
                          markersize=5, capsize=2.5, elinewidth=1.4, zorder=6, label=entry["name"])
    for column in view["columns"]:
        x, values = column["x"], column["values"]
        keep = np.isfinite(values)
        if not keep.any():
            continue
        color, alpha = column["color"], (0.55 if column["faded"] else 0.85)
        if column.get("summary") is False:
            # Overlaid view: the animals are small open markers behind the group mean line.
            axis.scatter(x + column["jitter"][keep], values[keep], s=9, facecolors="none", edgecolors=color,
                         alpha=0.45, linewidths=0.6, zorder=2)
            continue
        mean, sem = column["mean"], column["sem"] if np.isfinite(column["sem"]) else None
        if style == "bars":
            axis.bar(x, mean, width=0.62, color=color, alpha=0.28, edgecolor=color, linewidth=0.9, zorder=1)
            axis.errorbar(x, mean, yerr=sem, fmt="none", ecolor=neutral, capsize=2.5, linewidth=0.9, zorder=4)
        elif style == "box":
            low, high = whiskers(column)
            axis.bar(x, column["q3"] - column["q1"], bottom=column["q1"], width=0.56, color=color, alpha=0.22,
                     edgecolor=color, linewidth=0.9, zorder=1)
            axis.plot([x - 0.28, x + 0.28], [column["median"]] * 2, color=color, linewidth=2.0, zorder=4)
            axis.plot([x, x], [low, column["q1"]], color=color, linewidth=0.9, zorder=2)
            axis.plot([x, x], [column["q3"], high], color=color, linewidth=0.9, zorder=2)
        axis.scatter(x + column["jitter"][keep], values[keep], s=12, color=color, alpha=alpha,
                     edgecolors="none", zorder=3)
        if style == "points":
            axis.plot([x - 0.24, x + 0.24], [column["median"]] * 2, color=color, alpha=alpha,
                      linewidth=2.2, zorder=4)
            if np.isfinite(mean):
                axis.errorbar(x + 0.32 * column["side"], mean, yerr=sem, fmt="D", ms=3, color=neutral,
                              capsize=2, linewidth=0.9, zorder=5)
    bounds = view_bounds(view, style)
    if bounds is not None:
        low, high = bounds
        placed, top = bracket_layout(view, low, high)
        span = high - low or max(abs(high) * 0.1, 1.0)
        for x0, x1, y, text in placed:
            tick = span * 0.025
            axis.plot([x0, x0, x1, x1], [y - tick, y, y, y - tick], color=neutral, linewidth=0.8, zorder=5)
            axis.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=6, color=neutral)
        axis.set_ylim(low - span * 0.08, top + span * (0.16 if placed else 0.08))
    if view.get("zero_line") or style == "bars":
        axis.axhline(0, color="#A7B1BD", linewidth=0.6, zorder=0)
    note = view_note(view, style)
    axis.set_title(title or view.get("title", ""), fontsize=10, fontweight="bold", pad=30 if note else 8)
    if note:
        axis.text(0.5, 1.015, note, transform=axis.transAxes, ha="center", va="bottom",
                  fontsize=6.5, color="#546274", clip_on=False)
    if view.get("mean_lines") and axis.get_legend_handles_labels()[0]:
        axis.legend(loc="best", frameon=False, fontsize=7, handlelength=1.6)
    positions = [tick[0] for tick in view["ticks"]]
    axis.set_xticks(positions, [tick[1] for tick in view["ticks"]])
    xs = [column["x"] for column in view["columns"]] or [0.0]
    axis.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    if units:
        axis.set_ylabel(units, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
