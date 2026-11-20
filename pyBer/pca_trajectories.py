"""PCA state-space trajectories of event-aligned photometry responses.

A dashboard panel ("Change to > State-space (PCA) > PCA trajectories") that
draws how the event-aligned signal moves through a low-dimensional state space,
for one or several behaviors and one or several groups. Options are chosen from
the panel's right-click menu.

Why two methods
---------------
A PCA trajectory needs several dimensions per time point. A fiber recording
gives one signal, so the dimensions have to come from somewhere, and the two
honest choices answer different questions:

1. Delay embedding (default; works with any number of recordings, even one).
   Each condition (group x behavior) is reduced to its mean event-aligned
   trace y(t). Every time point is described by the trace around it,
   [y(t), y(t+dt), ..., y(t+(m-1)dt)] (Takens / singular-spectrum embedding).
   All conditions are embedded the same way and one PCA is fitted on all of
   them, so the trajectories share one space and can be compared. For smooth
   traces PC1 follows the signal level and PC2 its rate of change: the
   trajectory is a phase portrait of the response. It describes the shape and
   dynamics of the mean response, not a population code.

2. Recordings as dimensions (pseudo-population; needs at least two recordings
   that have every selected behavior). Each recording is one dimension, each
   time point of each behavior one sample. One PCA is fitted on all behaviors
   together and each behavior is one trajectory. Recordings of the selected
   groups are pooled: groups cannot be separate trajectories here, because
   they are the dimensions. This is the classical "neural trajectory" with
   animals in place of neurons, and it only shows what the recordings share.

The PCA is descriptive. No test is attached to it; the axis labels give the
variance each component explains. Nothing in the recordings is modified.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from postprocessing_core import compute_psth_matrix
from recording_groups import UNGROUPED_COLOR, UNGROUPED_LABEL, color_dot_icon


# =============================================================================
# Configuration
# =============================================================================

FAMILY = "trajectory"
CODE = "pca"
SPEC = f"{FAMILY}:{CODE}"
PANEL_LABEL = "PCA trajectories"

METHOD_DELAY = "delay"
METHOD_RECORDINGS = "recordings"
METHODS: Tuple[Tuple[str, str], ...] = (
    (METHOD_DELAY, "Delay embedding (shape and dynamics of each mean response)"),
    (METHOD_RECORDINGS, "Recordings as dimensions (pseudo-population)"),
)
#: Length of the delay window, in seconds, offered in the menu.
DELAY_SPANS_S: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
DEFAULT_DELAY_SPAN_S = 1.0
#: The delay window never has fewer samples than this, nor more than this
#: fraction of the trace.
MIN_DELAY_SAMPLES = 3
MAX_DELAY_FRACTION = 0.5
#: Component pairs offered for the two plot axes (0-based).
AXES_CHOICES: Tuple[Tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
N_COMPONENTS = 3
MIN_RECORDINGS_AS_DIMENSIONS = 2

#: Group keys used in the options: pooled recordings and ungrouped recordings.
POOLED = "*"
UNGROUPED = ""
SETTINGS_KEY = "pca_trajectories_options"

# Figure styling.
LINE_WIDTH = 1.8
MARKER_SIZE = 9
#: Width allowed per legend column (pixels) and the plot layout row the legend
#: uses (below the bottom axis, which sits in row 3).
LEGEND_COLUMN_WIDTH = 135
LEGEND_LAYOUT_ROW = 4
#: Categorical colors (fixed order) for behaviors when one group is shown.
COLORS_ON_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
COLORS_ON_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767")
#: Line styles (fixed order) for behaviors when several groups are shown.
LINE_STYLES = (QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.DashLine,
               QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashDotLine)


# =============================================================================
# Numerics (no GUI)
# =============================================================================

def fit_pca(samples: np.ndarray, n_components: int = N_COMPONENTS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA by SVD of the centered samples (rows = samples, columns = dimensions).

    Returns ``(mean, components, explained_ratio)``. Components are rows, at
    most ``n_components`` of them. Each component's sign is fixed so that its
    largest loading is positive, which makes results reproducible.
    """
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError("PCA needs at least two finite samples.")
    mean = x.mean(axis=0)
    _u, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    variance = singular ** 2
    total = float(variance.sum())
    ratio = variance / total if total > 0 else np.zeros_like(variance)
    count = min(int(n_components), vt.shape[0])
    components = vt[:count].copy()
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1.0
    return mean, components, ratio[:count]


def delay_embed(trace: np.ndarray, n_delays: int) -> Tuple[np.ndarray, np.ndarray]:
    """Delay vectors of a trace and the index of each vector's center sample.

    Row i is ``[y[i], y[i+1], ..., y[i+n_delays-1]]``. Rows that contain a
    missing sample are dropped, so nothing is interpolated.
    """
    y = np.asarray(trace, float).reshape(-1)
    m = int(n_delays)
    if m < 1 or y.size < m:
        return np.empty((0, max(m, 1))), np.empty(0, dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(y, m)
    keep = np.all(np.isfinite(windows), axis=1)
    centers = np.arange(windows.shape[0]) + (m - 1) // 2
    return np.array(windows[keep]), centers[keep]


def delay_sample_count(span_s: float, time: np.ndarray) -> int:
    """Number of samples in a delay window of ``span_s`` on this time axis."""
    t = np.asarray(time, float)
    if t.size < 2:
        return MIN_DELAY_SAMPLES
    step = float(np.nanmedian(np.diff(t)))
    wanted = int(round(float(span_s) / step)) + 1 if step > 0 else MIN_DELAY_SAMPLES
    return int(max(MIN_DELAY_SAMPLES, min(wanted, int(t.size * MAX_DELAY_FRACTION))))


def delay_trajectories(traces: Dict[object, Tuple[np.ndarray, np.ndarray]], span_s: float,
                       n_components: int = N_COMPONENTS) -> Tuple[Dict[object, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Trajectories of several mean traces in one shared delay-embedding PCA space.

    ``traces``: key -> (time, trace). Returns key -> (time of each point,
    scores with one column per component) and the explained variance ratios.
    Keys whose trace is too short or entirely missing are left out.
    """
    embedded = {}
    for key, (time, trace) in traces.items():
        t = np.asarray(time, float)
        vectors, centers = delay_embed(trace, delay_sample_count(span_s, t))
        if vectors.shape[0] >= 2:
            embedded[key] = (t[centers], vectors)
    if not embedded:
        return {}, np.empty(0)
    widths = {vectors.shape[1] for _t, vectors in embedded.values()}
    if len(widths) != 1:
        raise ValueError("Traces must share one time step to be embedded together.")
    mean, components, ratio = fit_pca(np.vstack([vectors for _t, vectors in embedded.values()]), n_components)
    return {key: (t, (vectors - mean) @ components.T) for key, (t, vectors) in embedded.items()}, ratio


def population_trajectories(responses: Dict[object, Tuple[np.ndarray, np.ndarray]],
                            n_components: int = N_COMPONENTS) -> Tuple[Dict[object, Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Trajectories of several conditions with recordings as dimensions.

    ``responses``: key -> (time, matrix with one row per time point and one
    column per recording; the same recordings, in the same order, for every
    key). Time points with a missing value in any recording are dropped.
    """
    kept = {}
    for key, (time, matrix) in responses.items():
        mat = np.asarray(matrix, float)
        rows = np.all(np.isfinite(mat), axis=1) if mat.ndim == 2 else np.zeros(0, bool)
        if mat.ndim == 2 and rows.sum() >= 2:
            kept[key] = (np.asarray(time, float)[rows], mat[rows])
    if not kept:
        return {}, np.empty(0)
    if len({mat.shape[1] for _t, mat in kept.values()}) != 1:
        raise ValueError("Every condition must use the same recordings.")
    mean, components, ratio = fit_pca(np.vstack([mat for _t, mat in kept.values()]), n_components)
    return {key: (t, (mat - mean) @ components.T) for key, (t, mat) in kept.items()}, ratio


# =============================================================================
# Panel
# =============================================================================

class PcaTrajectoryView(QtCore.QObject):
    """Options, data collection and drawing of the PCA trajectories panel."""

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        # behaviors: [] = follow the PSTH behavior. groups: None = automatic
        # (the groups shown by Hide/show groups, or all recordings pooled).
        self.options: Dict[str, object] = {"behaviors": [], "groups": None, "method": METHOD_DELAY,
                                           "span_s": DEFAULT_DELAY_SPAN_S, "axes": [0, 1]}
        self._legends: Dict[int, pg.LegendItem] = {}
        self.last_report: Dict[str, object] = {}
        self._load_options()

    # ---------------------------------------------------------------- options
    def _load_options(self) -> None:
        try:
            stored = json.loads(str(self.panel._settings.value(SETTINGS_KEY, "") or "{}"))
        except Exception:
            stored = {}
        if isinstance(stored, dict):
            for key in self.options:
                if key in stored:
                    self.options[key] = stored[key]
        if self.options["method"] not in dict(METHODS):
            self.options["method"] = METHOD_DELAY
        if tuple(self.options.get("axes") or ()) not in AXES_CHOICES:
            self.options["axes"] = [0, 1]

    def set_option(self, key: str, value) -> None:
        self.options[key] = value
        try:
            self.panel._settings.setValue(SETTINGS_KEY, json.dumps(self.options))
        except Exception:
            pass
        self.refresh()

    def refresh(self) -> None:
        panels = getattr(self.panel, "dashboard_panels", None)
        if panels is not None:
            panels.render_substitutes((FAMILY,))

    # ------------------------------------------------------- what can be shown
    def behaviors_apply(self) -> bool:
        """Behaviors can be chosen only when the PSTH is aligned to one scored behavior."""
        panel = self.panel
        return (panel.combo_align.currentText().startswith("Behavior")
                and not panel.combo_behavior_align.currentText().startswith("Transition")
                and bool(self.available_behaviors()))

    def available_behaviors(self) -> List[str]:
        combo = self.panel.combo_behavior_name
        return [combo.itemText(index) for index in range(combo.count()) if combo.itemText(index).strip()]

    def selected_behaviors(self) -> List[str]:
        """Behaviors drawn: the chosen ones still loaded, else the PSTH behavior."""
        if not self.behaviors_apply():
            return [self._current_event_label()]
        available = self.available_behaviors()
        chosen = [name for name in (self.options.get("behaviors") or []) if name in available]
        if chosen:
            return chosen
        current = self.panel.combo_behavior_name.currentText().strip()
        return [current] if current else available[:1]

    def _current_event_label(self) -> str:
        panel = self.panel
        align = panel.combo_align.currentText()
        if align == "Event marker":
            return str(panel._marker_label()) if hasattr(panel, "_marker_label") else "Event marker"
        if align.startswith("Behavior"):
            if panel.combo_behavior_align.currentText().startswith("Transition"):
                return f"{panel.combo_behavior_from.currentText()} → {panel.combo_behavior_to.currentText()}"
            return panel.combo_behavior_name.currentText().strip() or "Behavior"
        return panel.combo_dio.currentText().strip() or "Events"

    def group_entries(self) -> List[Tuple[str, str, str, List[str]]]:
        """(key, name, color, recording ids) for each group that has loaded recordings."""
        panel = self.panel
        file_ids = [panel._file_id_for_proc(proc) for proc in panel._processed]
        entries = []
        for group in panel.recording_groups.groups():
            members = [fid for fid in file_ids if panel.recording_groups.group_id_of(fid) == group["id"]]
            if members:
                entries.append((group["id"], group["name"], group["color"], members))
        known = {group["id"] for group in panel.recording_groups.groups()}
        loose = [fid for fid in file_ids if panel.recording_groups.group_id_of(fid) not in known]
        if loose and entries:
            entries.append((UNGROUPED, UNGROUPED_LABEL, UNGROUPED_COLOR, loose))
        return entries

    def selected_groups(self) -> List[Tuple[str, str, Optional[str], List[str]]]:
        """Groups drawn as separate trajectories (color None = use the behavior colors)."""
        panel = self.panel
        all_ids = [panel._file_id_for_proc(proc) for proc in panel._processed]
        pooled = (POOLED, "All recordings", None, all_ids)
        entries = self.group_entries()
        chosen = self.options.get("groups")
        if chosen is None:   # automatic: the groups that Hide/show groups leaves visible
            hidden = panel._hidden_groups() if hasattr(panel, "_hidden_groups") else set()
            visible = [entry for entry in entries if entry[0] not in hidden]
            return visible if entries else [pooled]
        out = [pooled] if POOLED in chosen else []
        out += [entry for entry in entries if entry[0] in chosen]
        return out or [pooled]

    # ------------------------------------------------------------------ data
    def _mean_responses(self, behaviors: Sequence[str]) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[np.ndarray]]:
        """Mean event-aligned trace of every recording for every behavior.

        Uses the panel's own event extraction (clock, alignment edge, event
        filters, hidden events) and the current PSTH window, baseline and
        normalization, so a trajectory is built from the same responses as the
        PSTH. The behavior picker is switched silently and put back.
        """
        panel = self.panel
        window = (-float(panel.spin_pre.value()), float(panel.spin_post.value()))
        baseline = (float(panel.spin_b0.value()), float(panel.spin_b1.value()))
        rate = float(panel.spin_resample.value())
        smooth = float(panel.spin_smooth.value())
        normalization = panel._psth_normalization()
        marker_alignment = panel.combo_align.currentText() == "Event marker"
        combo = panel.combo_behavior_name
        switch = self.behaviors_apply()
        previous = combo.currentIndex()
        responses: Dict[str, Dict[str, np.ndarray]] = {}
        tvec: Optional[np.ndarray] = None
        blocker = QtCore.QSignalBlocker(combo)
        try:
            for behavior in behaviors:
                if switch:
                    index = combo.findText(behavior)
                    if index < 0:
                        continue
                    combo.setCurrentIndex(index)
                per_file: Dict[str, np.ndarray] = {}
                for proc in panel._processed:
                    _label, signal = panel._psth_input(proc)
                    if signal is None:
                        continue
                    events, durations = panel._get_events_for_proc(proc)
                    if not marker_alignment:
                        events, durations = panel._filter_events(events, durations)
                    if np.asarray(events).size == 0:
                        continue
                    try:
                        time, matrix = compute_psth_matrix(panel._proc_time(proc), signal, events, window, baseline,
                                                           rate, smooth_sigma_s=smooth, normalization=normalization)
                    except ValueError:
                        continue
                    matrix = matrix[np.any(np.isfinite(matrix), axis=1)] if matrix.ndim == 2 else matrix[:0]
                    if matrix.size == 0:
                        continue
                    with np.errstate(all="ignore"):
                        mean = np.nanmean(matrix, axis=0)
                    per_file[panel._file_id_for_proc(proc)] = mean
                    tvec = time
                responses[behavior] = per_file
        finally:
            if switch:
                combo.setCurrentIndex(previous)
            del blocker
        return responses, tvec

    def compute(self) -> Dict[str, object]:
        """Trajectories for the current options, with what was left out and why."""
        behaviors = self.selected_behaviors()
        groups = self.selected_groups()
        method = str(self.options.get("method") or METHOD_DELAY)
        report: Dict[str, object] = {"method": method, "behaviors": list(behaviors), "curves": [],
                                     "ratio": np.empty(0), "note": "", "skipped": []}
        if not self.panel._processed or not behaviors:
            report["note"] = "Load recordings and events to draw trajectories."
            return report
        responses, tvec = self._mean_responses(behaviors)
        if tvec is None:
            report["note"] = "No event-aligned response for the selected behaviors."
            return report

        if method == METHOD_RECORDINGS:
            wanted: List[str] = []
            for _key, _name, _color, members in groups:
                wanted.extend(fid for fid in members if fid not in wanted)
            usable = [fid for fid in wanted if all(fid in responses.get(b, {}) for b in behaviors)]
            report["skipped"] = [fid for fid in wanted if fid not in usable]
            report["recordings"] = usable
            if len(usable) < MIN_RECORDINGS_AS_DIMENSIONS:
                report["note"] = (f"Recordings as dimensions needs at least {MIN_RECORDINGS_AS_DIMENSIONS} "
                                  f"recordings with every selected behavior ({len(usable)} available). "
                                  "Use Delay embedding instead.")
                return report
            stacked = {b: (tvec, np.column_stack([responses[b][fid] for fid in usable])) for b in behaviors}
            trajectories, ratio = population_trajectories(stacked)
            for index, behavior in enumerate(behaviors):
                if behavior in trajectories:
                    time, scores = trajectories[behavior]
                    report["curves"].append({"label": f"{behavior} (n={len(usable)} rec.)", "behavior": behavior,
                                             "behavior_index": index, "group": None, "group_color": None,
                                             "time": time, "scores": scores, "n": len(usable)})
            report["ratio"] = ratio
            return report

        traces = {}
        meta = {}
        for group_index, (key, name, color, members) in enumerate(groups):
            for behavior_index, behavior in enumerate(behaviors):
                rows = [responses.get(behavior, {}).get(fid) for fid in members]
                rows = [row for row in rows if row is not None]
                if not rows:
                    report["skipped"].append(f"{name} / {behavior}")
                    continue
                with np.errstate(all="ignore"):
                    mean = np.nanmean(np.vstack(rows), axis=0)
                traces[(key, behavior)] = (tvec, mean)
                meta[(key, behavior)] = (name, color, behavior, behavior_index, len(rows), group_index)
        try:
            trajectories, ratio = delay_trajectories(traces, float(self.options.get("span_s") or DEFAULT_DELAY_SPAN_S))
        except ValueError as exc:
            report["note"] = str(exc)
            return report
        several_groups = len(groups) > 1
        several_behaviors = len(behaviors) > 1
        for key, (time, scores) in trajectories.items():
            name, color, behavior, behavior_index, count, _group_index = meta[key]
            parts = ([behavior] if several_behaviors or not several_groups else []) + \
                    ([name] if several_groups or not several_behaviors else [])
            report["curves"].append({"label": f"{' · '.join(parts)} (n={count})", "behavior": behavior,
                                     "behavior_index": behavior_index, "group": name,
                                     "group_color": color if several_groups else None,
                                     "time": time, "scores": scores, "n": count})
        report["ratio"] = ratio
        if not report["curves"]:
            report["note"] = "The selected responses are too short or incomplete for the delay window."
        return report

    # --------------------------------------------------------------- drawing
    def _behavior_colors(self, palette: Dict[str, object]) -> Sequence[str]:
        text = QtGui.QColor(str(palette.get("text", "#000000")))
        return COLORS_ON_DARK if text.lightness() > 128 else COLORS_ON_LIGHT

    def render(self, widget: pg.PlotWidget) -> None:
        """Draw the trajectories of the current options in a dashboard plot."""
        panel = self.panel
        palette = panel._current_palette()
        text_color = str(palette.get("text", "#000000"))
        widget.clear()
        plot_item = widget.getPlotItem()
        legend = self._legends.pop(id(widget), None)
        if legend is not None:
            try:
                plot_item.layout.removeItem(legend)
                legend.setParentItem(None)
                if legend.scene() is not None:
                    legend.scene().removeItem(legend)
            except Exception:
                pass
        try:
            report = self.compute()
        except Exception as exc:   # a display problem must not break the analysis
            report = {"curves": [], "note": f"Could not compute trajectories: {exc}", "ratio": np.empty(0),
                      "method": self.options.get("method"), "skipped": []}
        self.last_report = report
        widget.setProperty("hasPlotData", True)
        plot_item.setVisible(True)        # stays reachable by right-click even when empty
        method_name = "delay embedding" if report.get("method") == METHOD_DELAY else "recordings as dimensions"
        widget.setTitle(f"{PANEL_LABEL} · {method_name}", color=text_color, size="10pt")
        curves = report["curves"]
        ax, ay = (int(value) for value in (self.options.get("axes") or [0, 1]))
        ratio = np.asarray(report.get("ratio", np.empty(0)), float)
        if not curves or ratio.size <= max(ax, ay):
            note = report.get("note") or "This component pair is not available for the current selection."
            message = pg.TextItem(str(note), color=text_color, anchor=(0.5, 0.5))
            widget.addItem(message)
            message.setPos(0.5, 0.5)
            plot_item.setAspectLocked(False)
            widget.setXRange(0, 1, padding=0)
            widget.setYRange(0, 1, padding=0)
            for axis in ("bottom", "left"):
                widget.setLabel(axis, "")
            widget.setToolTip(str(note))
            return

        colors = self._behavior_colors(palette)
        # A trajectory is a parametric curve (x is not time): the time-series
        # shortcuts of the other plots (peak downsampling, clipping to the
        # view) would redraw it as a staircase.
        plot_item.setDownsampling(ds=1, auto=False)
        plot_item.setClipToView(False)
        # Legend under the x axis, outside the data area, in as many columns as fit.
        columns = max(1, min(len(curves), widget.width() // LEGEND_COLUMN_WIDTH))
        legend = pg.LegendItem(labelTextColor=text_color, labelTextSize="8pt", colCount=columns)
        legend.setParentItem(plot_item)
        plot_item.layout.addItem(legend, LEGEND_LAYOUT_ROW, 1)
        self._legends[id(widget)] = legend
        for curve in curves:
            color = curve["group_color"] or colors[curve["behavior_index"] % len(colors)]
            style = LINE_STYLES[curve["behavior_index"] % len(LINE_STYLES)] if curve["group_color"] else LINE_STYLES[0]
            x, y = curve["scores"][:, ax], curve["scores"][:, ay]
            line = widget.plot(x, y, pen=pg.mkPen(color, width=LINE_WIDTH, style=style), antialias=True)
            legend.addItem(line, curve["label"])
            # Start (hollow circle), event time (diamond), end (filled triangle).
            marks = [(0, "o", None), (int(np.argmin(np.abs(curve["time"]))), "d", color), (x.size - 1, "t1", color)]
            if not (curve["time"][0] <= 0.0 <= curve["time"][-1]):
                marks.pop(1)
            for index, symbol, fill in marks:
                widget.addItem(pg.ScatterPlotItem(
                    [x[index]], [y[index]], symbol=symbol, size=MARKER_SIZE,
                    pen=pg.mkPen(color, width=1.5), brush=pg.mkBrush(fill) if fill else pg.mkBrush(None)))
        widget.setLabel("bottom", f"PC{ax + 1} ({100 * ratio[ax]:.0f}% of variance)")
        widget.setLabel("left", f"PC{ay + 1} ({100 * ratio[ay]:.0f}% of variance)")
        plot_item.setAspectLocked(True)            # both axes share the signal's units
        plot_item.enableAutoRange()
        skipped = report.get("skipped") or []
        tips = ["Hollow circle: start of the window. Diamond: event time. Triangle: end.",
                "Descriptive only; statistics use the PSTH panels."]
        if report.get("method") == METHOD_DELAY:
            tips.insert(0, "Each trajectory is a mean response in delay-embedding space: "
                           "PC1 mostly follows its level, PC2 its rate of change.")
        else:
            tips.insert(0, f"Dimensions: {len(report.get('recordings', []))} recordings (selected groups pooled).")
        if skipped:
            tips.append("Left out (no response): " + ", ".join(str(item) for item in skipped[:6])
                        + (" ..." if len(skipped) > 6 else ""))
        widget.setToolTip("\n".join(tips))

    # ------------------------------------------------------------------ menu
    def build_menu(self, parent: QtWidgets.QMenu) -> QtWidgets.QMenu:
        """Right-click options of the panel: behaviors, groups, method, axes."""
        menu = QtWidgets.QMenu("PCA trajectory options", parent)

        behaviors = menu.addMenu("Behaviors")
        if self.behaviors_apply():
            chosen = [name for name in (self.options.get("behaviors") or []) if name in self.available_behaviors()]
            follow = behaviors.addAction("Follow the PSTH behavior")
            follow.setCheckable(True)
            follow.setChecked(not chosen)
            follow.triggered.connect(lambda: self.set_option("behaviors", []))
            behaviors.addSeparator()
            for name in self.available_behaviors():
                action = behaviors.addAction(name)
                action.setCheckable(True)
                action.setChecked(name in chosen)
                action.triggered.connect(lambda checked, n=name: self._toggle_behavior(n, checked))
        else:
            note = behaviors.addAction("Align the PSTH to a behavior to choose several")
            note.setEnabled(False)

        groups = menu.addMenu("Groups")
        chosen_groups = self.options.get("groups")
        automatic = groups.addAction("Automatic (groups shown by Hide/show groups)")
        automatic.setCheckable(True)
        automatic.setChecked(chosen_groups is None)
        automatic.triggered.connect(lambda: self.set_option("groups", None))
        groups.addSeparator()
        pooled = groups.addAction("All recordings pooled")
        pooled.setCheckable(True)
        pooled.setChecked(chosen_groups is not None and POOLED in chosen_groups)
        pooled.triggered.connect(lambda checked: self._toggle_group(POOLED, checked))
        for key, name, color, members in self.group_entries():
            action = groups.addAction(color_dot_icon(color), f"{name} ({len(members)})")
            action.setCheckable(True)
            action.setChecked(chosen_groups is not None and key in chosen_groups)
            action.triggered.connect(lambda checked, k=key: self._toggle_group(k, checked))

        methods = menu.addMenu("Method")
        for value, label in METHODS:
            action = methods.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.options.get("method") == value)
            action.triggered.connect(lambda _=False, v=value: self.set_option("method", v))

        spans = menu.addMenu("Delay window")
        spans.setEnabled(self.options.get("method") == METHOD_DELAY)
        for span in DELAY_SPANS_S:
            action = spans.addAction(f"{span:g} s")
            action.setCheckable(True)
            action.setChecked(abs(float(self.options.get("span_s") or 0) - span) < 1e-9)
            action.triggered.connect(lambda _=False, s=span: self.set_option("span_s", float(s)))

        axes = menu.addMenu("Axes")
        for first, second in AXES_CHOICES:
            action = axes.addAction(f"PC{first + 1} vs PC{second + 1}")
            action.setCheckable(True)
            action.setChecked(list(self.options.get("axes") or [0, 1]) == [first, second])
            action.triggered.connect(lambda _=False, a=(first, second): self.set_option("axes", list(a)))
        return menu

    def _toggle_behavior(self, name: str, checked: bool) -> None:
        available = self.available_behaviors()
        chosen = [item for item in (self.options.get("behaviors") or []) if item in available]
        if checked and name not in chosen:
            chosen.append(name)
        elif not checked and name in chosen:
            chosen.remove(name)
        self.set_option("behaviors", [item for item in available if item in chosen])

    def _toggle_group(self, key: str, checked: bool) -> None:
        chosen = list(self.options.get("groups") or [])
        if checked and key not in chosen:
            chosen.append(key)
        elif not checked and key in chosen:
            chosen.remove(key)
        self.set_option("groups", chosen if chosen else None)
