"""Named PSTH metric windows: renamable Pre / Post and extra post windows.

The PSTH metrics compare a reference window (Pre) with a response window
(Post). A response is not always one event: a bimodal response has an early
and a late component that one Post window would average together. This module
lets the user

* rename the two standard windows by double-clicking their label
  (for example "Post" -> "Early"), and
* add further windows with the "+" beside the metric selector, each with its
  own label, start and end ("Late", 2 to 5 s, ...).

Every extra window is compared with the same Pre window, exactly like Post:
one panel per selected metric and window, the same paired test, and one Holm
correction across all the windows and metrics that are shown together. Windows
are compared with Pre, not with each other.

The controller owns its widgets and plots; the postprocessing panel only
places them and asks for the current windows when it computes the metrics.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from numeric_controls import DurationSpinBox


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PRE_LABEL = "Pre"
DEFAULT_POST_LABEL = "Post"
#: Label proposed for the first extra window; later ones are "Window 3", ...
FIRST_EXTRA_LABEL = "Late"
#: Separator between a metric id and a window id in result keys ("auc@w1").
KEY_SEPARATOR = "@"
MAX_EXTRA_WINDOWS = 6
LABEL_MAX_LENGTH = 24
#: Translucent shading of each extra window on the mean PSTH (fixed order).
REGION_BRUSHES: Tuple[Tuple[int, int, int, int], ...] = (
    (27, 175, 122, 26), (237, 161, 0, 26), (232, 123, 164, 26),
    (98, 80, 214, 26), (0, 131, 0, 26), (227, 73, 72, 26),
)
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
        self.add_button.setText("+")
        self.add_button.setAccessibleName("Add a metric window")
        self.add_button.setToolTip(
            "Add a response window (for example Early and Late for a bimodal response).\n"
            "Each window is compared with the Pre window. Double-click a window label to rename it.")
        self.add_button.clicked.connect(lambda: self.add_window())
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
        """Tell where the extra windows go: below the Post row of the metrics form."""
        self._form = form
        self._anchor = post_row_widget

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

    def summary_arguments(self) -> Dict[str, object]:
        """Keyword arguments for ``psth_metrics.summarize_metrics``."""
        return {"pre_label": self.pre_label(), "post_label": self.post_label(),
                "extra_windows": self.valid_windows()}

    def state(self) -> Dict[str, object]:
        return {"pre_label": self.pre_label(), "post_label": self.post_label(), "windows": self.windows()}

    def set_state(self, state: object) -> None:
        """Restore labels and windows from saved settings (missing = defaults)."""
        state = state if isinstance(state, dict) else {}
        self._restoring = True
        try:
            self.pre_label_widget.set_name(str(state.get("pre_label") or DEFAULT_PRE_LABEL))
            self.post_label_widget.set_name(str(state.get("post_label") or DEFAULT_POST_LABEL))
            for row in list(self._rows):
                self._remove_row(row)
            for entry in (state.get("windows") or [])[:MAX_EXTRA_WINDOWS]:
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
        if len(self._rows) >= MAX_EXTRA_WINDOWS:
            self.panel.statusUpdate.emit(f"At most {MAX_EXTRA_WINDOWS} extra metric windows.", 5000)
            return None
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
        self._on_changed()
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
        for row in self._rows:
            row.set_row_enabled(enabled)
        self.update_regions()

    def set_hidden(self, hidden: bool) -> None:
        """Follow the panel's "hide metric settings" toggle."""
        self._hidden = bool(hidden)
        self.add_button.setVisible(not hidden)
        for row in self._rows:
            row.set_row_visible(not hidden)

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
