"""Named PSTH metric windows: renamable Pre / Post and extra post windows.

The PSTH metrics compare a reference window (Pre) with a response window
(Post). A response is not always one event: a bimodal response has an early
and a late component that one Post window would average together. This module
lets the user

* rename the two standard windows by double-clicking their label
  (for example "Post" -> "Early"), and
* add further windows with the "+" button below the last window, each with
  its own label, start and end ("Late", 2 to 5 s, ...). There is no limit to
  the number of windows.

With at least one extra window, "Compare windows" (shown below the "+" button
and in the right-click menu of a metric panel) chooses between

* "All windows together" (default): one panel per metric with Pre, Post and
  every extra window side by side, an omnibus test and every pair of windows
  tested (see ``window_comparison``), and
* "Each window with Pre": one panel per selected metric and window, each
  compared with the same Pre window exactly like Post.

Both use one Holm correction across everything that is shown together. When
recording groups are compared ("All groups", two or more groups) every window
keeps its own panel, because the group views are built on one Pre / Post pair.

The controller owns its widgets and plots; the postprocessing panel only
places them and asks for the current windows when it computes the metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from numeric_controls import DurationSpinBox
from window_comparison import DEFAULT_WINDOW_COMPARISON, EXTRA_COLORS, WINDOW_COMPARISONS


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PRE_LABEL = "Pre"
DEFAULT_POST_LABEL = "Post"
#: Label proposed for the first extra window; later ones are "Window 3", ...
FIRST_EXTRA_LABEL = "Late"
#: Separator between a metric id and a window id in result keys ("auc@w1").
KEY_SEPARATOR = "@"
LABEL_MAX_LENGTH = 24
REGION_ALPHA = 26
#: Translucent shading of each extra window on the mean PSTH (fixed order). The
#: colors are the ones of the window columns in the all-windows metric panel.
REGION_BRUSHES: Tuple[Tuple[int, int, int, int], ...] = tuple(
    (*QtGui.QColor(color).getRgb()[:3], REGION_ALPHA) for color in EXTRA_COLORS)
SPIN_MIN_WIDTH = 60
CAPTION_MIN_WIDTH = 35      # "Start:" / "End:" captions, as in the Pre / Post rows


def result_key(metric: str, window_id: str) -> str:
    """Key of one metric of one extra window in the metric results."""
    return f"{metric}{KEY_SEPARATOR}{window_id}"


def is_window_key(key: str) -> bool:
    return KEY_SEPARATOR in str(key)


# =============================================================================
# Widgets
# =============================================================================

class RenamableLabel(QtWidgets.QLabel):
    """A label whose name is edited in place on double-click.

    ``suffix`` is shown after the name but is not part of it ("Post window").
    Enter or leaving the field keeps the new name; Escape or an empty name
    keeps the old one.
    """

    renamed = QtCore.Signal(str)

    def __init__(self, name: str, suffix: str = "", parent=None) -> None:
        super().__init__(parent)
        self._name = str(name)
        self._suffix = str(suffix)
        self._editor: Optional[QtWidgets.QLineEdit] = None
        self.setToolTip("Double-click to rename")
        self.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        self._refresh()

    def name(self) -> str:
        return self._name

    def set_name(self, name: str, emit: bool = False) -> None:
        name = str(name or "").strip()[:LABEL_MAX_LENGTH]
        if not name or name == self._name:
            return
        self._name = name
        self._refresh()
        if emit:
            self.renamed.emit(name)

    def _refresh(self) -> None:
        self.setText(self._name + self._suffix)

    def mouseDoubleClickEvent(self, event) -> None:
        self.start_editing()
        event.accept()

    def start_editing(self) -> None:
        if self._editor is not None or not self.isEnabled():
            return
        editor = QtWidgets.QLineEdit(self._name, self)
        editor.setMaxLength(LABEL_MAX_LENGTH)
        editor.setFrame(False)
        editor.setGeometry(self.rect().adjusted(0, 0, max(0, 110 - self.width()), 0))
        editor.selectAll()
        editor.installEventFilter(self)
        editor.editingFinished.connect(self._commit)
        editor.show()
        editor.raise_()
        editor.setFocus()
        self._editor = editor

    def _commit(self) -> None:
        editor, self._editor = self._editor, None
        if editor is None:
            return
        text = editor.text()
        editor.hide()
        editor.deleteLater()
        self.set_name(text, emit=True)

    def _cancel(self) -> None:
        editor, self._editor = self._editor, None
        if editor is not None:
            editor.blockSignals(True)
            editor.hide()
            editor.deleteLater()

    def eventFilter(self, watched, event) -> bool:
        if (watched is self._editor and event.type() == QtCore.QEvent.Type.KeyPress
                and event.key() == QtCore.Qt.Key.Key_Escape):
            self._cancel()
            return True
        return False


class _WindowRow(QtWidgets.QWidget):
    """One extra window: renamable label, start, end, remove.

    The row is the fields widget (start, end, remove); ``label`` is a separate
    widget so both can sit in the two columns of the metrics form, aligned
    with the Pre and Post rows.
    """

    changed = QtCore.Signal()
    removeRequested = QtCore.Signal(object)

    def __init__(self, window_id: str, label: str, start: float, end: float, unit: str, parent=None) -> None:
        super().__init__(parent)
        self.window_id = window_id
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        self.label = RenamableLabel(label, " window")
        self.start = DurationSpinBox(-1e9, 1e9, start)
        self.end = DurationSpinBox(-1e9, 1e9, end)
        for spin in (self.start, self.end):
            spin.setMinimumWidth(SPIN_MIN_WIDTH)
            spin.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
            spin.set_unit(unit)
            spin.valueChanged.connect(lambda _value: self.changed.emit())
        self.remove_button = QtWidgets.QToolButton()
        self.remove_button.setText("×")
        self.remove_button.setToolTip("Remove this window")
        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self))
        self.label.renamed.connect(lambda _name: self.changed.emit())
        # Same grid as the Pre / Post rows: caption, field, caption, field, action.
        start_caption, end_caption = QtWidgets.QLabel("Start:"), QtWidgets.QLabel("End:")
        for caption in (start_caption, end_caption):
            caption.setMinimumWidth(CAPTION_MIN_WIDTH)
        layout.addWidget(start_caption, 0, 0)
        layout.addWidget(self.start, 0, 1)
        layout.addWidget(end_caption, 0, 2)
        layout.addWidget(self.end, 0, 3)
        layout.addWidget(self.remove_button, 0, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.setMinimumWidth(layout.minimumSize().width())

    def set_row_enabled(self, enabled: bool) -> None:
        self.setEnabled(bool(enabled))
        self.label.setEnabled(bool(enabled))

    def set_row_visible(self, visible: bool) -> None:
        self.setVisible(bool(visible))
        self.label.setVisible(bool(visible))

    def state(self) -> Dict[str, object]:
        return {"id": self.window_id, "label": self.label.name(),
                "start": float(self.start.value()), "end": float(self.end.value())}


# =============================================================================
# Controller
# =============================================================================

class MetricWindows(QtCore.QObject):
    """Labels of Pre / Post, the extra windows, their plots and their shading."""

    changed = QtCore.Signal()

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        self.pre_label_widget = RenamableLabel(DEFAULT_PRE_LABEL, " window")
        self.post_label_widget = RenamableLabel(DEFAULT_POST_LABEL, " window")
        self.add_button = QtWidgets.QToolButton()
        self.add_button.setText("+  Add window")
        self.add_button.setAccessibleName("Add a metric window")
        self.add_button.setToolTip(
            "Add a response window (for example Early and Late for a bimodal response).\n"
            "Each window is compared with the Pre window. Add as many as you need.\n"
            "Double-click a window label to rename it.")
        self.add_button.clicked.connect(lambda: self.add_window())
        # The button has its own form row, kept below the last window.
        self.add_row = QtWidgets.QWidget()
        add_layout = QtWidgets.QHBoxLayout(self.add_row)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_layout.addWidget(self.add_button)
        add_layout.addStretch(1)
        # How the windows are compared; only shown once there is an extra window.
        self.compare_label = QtWidgets.QLabel("Compare windows")
        self.compare_combo = QtWidgets.QComboBox()
        for code, label in WINDOW_COMPARISONS.items():
            self.compare_combo.addItem(label, code)
        self.compare_combo.setAccessibleName("Compare metric windows")
        self.compare_combo.setToolTip(
            "All windows together: one panel per metric with every window side by side.\n"
            "Friedman test across the windows and an exact paired sign test of every pair of\n"
            "windows (Pre vs Post, Pre vs Late, Post vs Late, ...), Holm-adjusted.\n"
            "Each window with Pre: one panel per window, each compared with Pre only.\n"
            "When recording groups are compared (All groups), every window keeps its own panel.")
        self.compare_combo.setCurrentIndex(max(0, self.compare_combo.findData(DEFAULT_WINDOW_COMPARISON)))
        self.compare_combo.currentIndexChanged.connect(lambda _index: self._on_changed())
        # The metrics form and the widget of its Post row: extra windows are
        # inserted as form rows right below it (see attach_form).
        self._form: Optional[QtWidgets.QFormLayout] = None
        self._anchor: Optional[QtWidgets.QWidget] = None
        self._rows: List[_WindowRow] = []
        self._regions: List[pg.LinearRegionItem] = []
        self._counter = 0
        self._restoring = False
        self._hidden = False
        for label in (self.pre_label_widget, self.post_label_widget):
            label.renamed.connect(lambda _name: self._on_changed())

    def attach_form(self, form: QtWidgets.QFormLayout, post_row_widget: QtWidgets.QWidget) -> None:
        """Tell where the extra windows go: below the Post row of the metrics form.

        The "+" row is placed right below Post; every new window is inserted
        above it, so the button always follows the last window. The "Compare
        windows" row comes right after the button.
        """
        self._form = form
        self._anchor = post_row_widget
        row, _role = form.getWidgetPosition(post_row_widget)
        row = (row if row >= 0 else form.rowCount() - 1) + 1
        # Spanning row: the button starts at the left edge, like the window rows when they wrap.
        form.insertRow(row, self.add_row)
        form.insertRow(row + 1, self.compare_label, self.compare_combo)
        self._sync_compare_row()

    def _sync_compare_row(self) -> None:
        """Offer the comparison choice only when there is something to choose."""
        visible = bool(self._rows) and not self._hidden
        if self._form is not None:
            self._form.setRowVisible(self.compare_combo, visible)
        else:
            self.compare_label.setVisible(visible)
            self.compare_combo.setVisible(visible)

    def _insert_position(self) -> int:
        row, _role = self._form.getWidgetPosition(self._anchor)
        return (row if row >= 0 else self._form.rowCount() - 1) + 1 + len(self._rows)

    # ----------------------------------------------------------------- state
    def pre_label(self) -> str:
        return self.pre_label_widget.name()

    def post_label(self) -> str:
        return self.post_label_widget.name()

    def windows(self) -> List[Dict[str, object]]:
        return [row.state() for row in self._rows]

    def valid_windows(self) -> List[Tuple[str, str, Tuple[float, float]]]:
        """(id, label, (start, end)) of the extra windows whose end is after their start."""
        return [(w["id"], w["label"], (w["start"], w["end"])) for w in self.windows() if w["end"] > w["start"]]

    def compare_mode(self) -> str:
        """Selected comparison: "all" (every window in one panel) or "pre" (each window with Pre)."""
        return str(self.compare_combo.currentData() or DEFAULT_WINDOW_COMPARISON)

    def set_compare_mode(self, mode: str) -> None:
        self.compare_combo.setCurrentIndex(max(0, self.compare_combo.findData(str(mode))))

    def summary_arguments(self, group_layout: bool = False) -> Dict[str, object]:
        """Keyword arguments for ``psth_metrics.summarize_metrics``.

        ``group_layout``: recording groups are compared in the metric panels.
        The group views need one Pre / Post pair per panel, so each window is
        then compared with Pre whatever the selected comparison.
        """
        return {"pre_label": self.pre_label(), "post_label": self.post_label(),
                "extra_windows": self.valid_windows(),
                "compare": "pre" if group_layout else self.compare_mode()}

    def state(self) -> Dict[str, object]:
        return {"pre_label": self.pre_label(), "post_label": self.post_label(), "windows": self.windows(),
                "compare": self.compare_mode()}

    def set_state(self, state: object) -> None:
        """Restore labels and windows from saved settings (missing = defaults)."""
        state = state if isinstance(state, dict) else {}
        self._restoring = True
        try:
            self.pre_label_widget.set_name(str(state.get("pre_label") or DEFAULT_PRE_LABEL))
            self.post_label_widget.set_name(str(state.get("post_label") or DEFAULT_POST_LABEL))
            self.set_compare_mode(str(state.get("compare") or DEFAULT_WINDOW_COMPARISON))
            for row in list(self._rows):
                self._remove_row(row)
            for entry in (state.get("windows") or []):
                try:
                    self.add_window(str(entry.get("label") or ""), float(entry["start"]), float(entry["end"]),
                                    window_id=str(entry.get("id") or ""))
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
        finally:
            self._restoring = False

    # --------------------------------------------------------------- editing
    def add_window(self, label: str = "", start: Optional[float] = None, end: Optional[float] = None,
                   window_id: str = "") -> Optional[_WindowRow]:
        """Add a window. By default it starts where the last one ends, with the Post length."""
        panel = self.panel
        post0, post1 = float(panel.spin_metric_post0.value()), float(panel.spin_metric_post1.value())
        last_end = self._rows[-1].end.value() if self._rows else post1
        if start is None:
            start = float(last_end)
        if end is None:
            end = float(start) + max(post1 - post0, 0.1)
        used = {row.window_id for row in self._rows}
        if not window_id or window_id in used:
            self._counter += 1
            while f"w{self._counter}" in used:
                self._counter += 1
            window_id = f"w{self._counter}"
        if not label:
            label = FIRST_EXTRA_LABEL if not self._rows else f"Window {len(self._rows) + 2}"
        unit = panel.combo_window_unit.currentData() if hasattr(panel, "combo_window_unit") else "s"
        row = _WindowRow(window_id, label, float(start), float(end), str(unit or "s"))
        row.set_row_enabled(panel.cb_metrics.isChecked() if hasattr(panel, "cb_metrics") else True)
        row.changed.connect(self._on_changed)
        row.removeRequested.connect(self.remove_window)
        if self._form is not None and self._anchor is not None:
            self._form.insertRow(self._insert_position(), row.label, row)
        self._rows.append(row)
        row.set_row_visible(not self._hidden)
        self._sync_compare_row()
        self._on_changed()
        if not self._restoring and self.compare_mode() == "all" and hasattr(panel, "statusUpdate"):
            names = " vs ".join([self.pre_label(), self.post_label()] + [str(w["label"]) for w in self.windows()])
            panel.statusUpdate.emit(f"Metric panels now compare {names}. Change it with Compare windows.", 7000)
        return row

    def remove_window(self, row: _WindowRow) -> None:
        self._remove_row(row)
        self._on_changed()

    def _remove_row(self, row: _WindowRow) -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        if self._form is not None and self._form.getWidgetPosition(row)[0] >= 0:
            self._form.removeRow(row)          # deletes the label and the fields
        else:
            for widget in (row.label, row):
                widget.hide()
                widget.deleteLater()
        self._drop_plots(row.window_id)
        self._sync_compare_row()

    def _on_changed(self) -> None:
        self.update_regions()
        if self._restoring:
            return
        self.changed.emit()
        panel = self.panel
        for name in ("_schedule_psth", "_queue_settings_save"):
            callback = getattr(panel, name, None)
            if callback is not None:
                callback()
        refresh = getattr(panel, "_refresh_section_scroll", None)
        if refresh is not None:
            refresh("psth")

    # ----------------------------------------------- panel-driven adjustments
    def set_unit(self, unit: str) -> None:
        for row in self._rows:
            row.start.set_unit(unit)
            row.end.set_unit(unit)

    def set_enabled(self, enabled: bool) -> None:
        self.add_button.setEnabled(bool(enabled))
        self.compare_combo.setEnabled(bool(enabled))
        for row in self._rows:
            row.set_row_enabled(enabled)
        self.update_regions()

    def set_hidden(self, hidden: bool) -> None:
        """Follow the panel's "hide metric settings" toggle."""
        self._hidden = bool(hidden)
        self.add_row.setVisible(not hidden)
        for row in self._rows:
            row.set_row_visible(not hidden)
        self._sync_compare_row()

    # ------------------------------------------------------------------ plots
    def plot_for(self, key: str):
        """Metric panel of one extra-window result, created on first use."""
        panel = self.panel
        plot = panel._extra_metric_plots.get(key)
        if plot is None:
            from gui_postprocessing import _opt_plot
            from psth_metric_panels import MetricPanel
            plot = MetricPanel(panel.metric_panels_widget)
            _opt_plot(plot)
            # Look exactly like the metric panels it sits beside.
            plot.setBackground(panel.plot_metrics.backgroundBrush())
            plot.set_colors(panel._current_palette())
            panel._install_metric_view_menu(plot)
            plot.hide()
            panel._extra_metric_plots[key] = plot
        return plot

    def visible_plots(self) -> List[QtWidgets.QWidget]:
        """Panels of the extra windows that currently hold a result, in result order."""
        panel = self.panel
        results = getattr(panel, "_last_metric_panels", None) or {}
        return [panel._extra_metric_plots[key] for key in results
                if is_window_key(key) and key in panel._extra_metric_plots]

    def _drop_plots(self, window_id: str) -> None:
        panel = self.panel
        plots = getattr(panel, "_extra_metric_plots", {})
        for key in [key for key in plots if is_window_key(key) and key.split(KEY_SEPARATOR, 1)[1] == window_id]:
            plot = plots.pop(key)
            grid = panel.metric_panels_widget.layout() if hasattr(panel, "metric_panels_widget") else None
            if grid is not None:
                grid.removeWidget(plot)
            plot.hide()
            plot.setParent(None)
            plot.deleteLater()

    # ---------------------------------------------------------------- shading
    def clear_regions(self) -> None:
        plot = getattr(self.panel, "plot_avg", None)
        for region in self._regions:
            try:
                if plot is not None:
                    plot.removeItem(region)
            except Exception:
                pass
        self._regions = []

    def update_regions(self) -> None:
        """Shade each extra window on the mean PSTH, like Pre and Post."""
        self.clear_regions()
        panel = self.panel
        plot = getattr(panel, "plot_avg", None)
        if plot is None or not hasattr(panel, "cb_metrics") or not panel.cb_metrics.isChecked():
            return
        for index, (_wid, _label, (start, end)) in enumerate(self.valid_windows()):
            region = pg.LinearRegionItem(values=(start, end), brush=REGION_BRUSHES[index % len(REGION_BRUSHES)],
                                         pen=pg.mkPen(None), movable=False)
            region.setZValue(-5)
            plot.addItem(region)
            self._regions.append(region)
