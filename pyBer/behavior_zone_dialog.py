"""Optional bout comparison added to the existing postprocessing dashboard."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from behavior_zone_compare import compare_recording, group_recordings
from numeric_controls import DurationSpinBox
from mamir_import import legacy_import_warning


_COLUMNS = ("Scope", "Label", "Recordings", "Events", "Rejected", "Before",
            "During", "After", "During - before", "After - before")
# Behavior chip grid cell (pixels) and how many chip rows show before scrolling.
_CHIP_WIDTH = 190
_CHIP_HEIGHT = 32
_CHIP_MAX_ROWS = 2
# Room kept for the status text before the toolbar folds onto two rows.
_STATUS_MIN_WIDTH = 90
# Smallest height given to the comparison plot, its legend and the table.
_RESULTS_MIN_HEIGHT = 200
_COLORS = ((89, 174, 255), (245, 153, 92), (124, 207, 151),
           (202, 153, 239), (240, 203, 99), (104, 210, 213))


class BehaviorPsthBar(QtWidgets.QWidget):
    """Visible mirrors of the existing PSTH controls, not a second analysis."""

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        self.import_warning = QtWidgets.QLabel()
        self.import_warning.setWordWrap(True)
        self.import_warning.setStyleSheet("color: #d97706; font-weight: 600;")
        self.import_warning.hide()
        root.addWidget(self.import_warning)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("PSTH / HEATMAP"))
        self.name = QtWidgets.QComboBox()
        self.name.setMinimumWidth(150)
        self.name.setAccessibleName("Event displayed in PSTH and heatmap")
        self.name.setToolTip("Choose a scored behavior or the recording’s event marker.")
        self.align = QtWidgets.QComboBox()
        self.pre = DurationSpinBox()
        self.post = DurationSpinBox()
        panel.combo_window_unit.currentIndexChanged.connect(self.sync_controls)
        for target, source in ((self.pre, panel.spin_pre), (self.post, panel.spin_post)):
            target.setMaximumWidth(110)
            target.valueChanged.connect(source.setValue)
            source.valueChanged.connect(self.sync_controls)
        self.name.activated.connect(self._choose_name)
        panel.combo_align.currentIndexChanged.connect(self.sync_controls)
        self.align.activated.connect(panel.combo_behavior_align.setCurrentIndex)
        for source in (panel.combo_behavior_name, panel.combo_behavior_align):
            source.currentIndexChanged.connect(self.sync_controls)
            source.model().rowsInserted.connect(self.sync_controls)
            source.model().rowsRemoved.connect(self.sync_controls)
        row.addWidget(self.name, 1)
        row.addWidget(self.align)
        row.addWidget(QtWidgets.QLabel("Pre"))
        row.addWidget(self.pre)
        row.addWidget(QtWidgets.QLabel("Post"))
        row.addWidget(self.post)
        self.collapse_button = QtWidgets.QPushButton("Hide comparison")
        self.collapse_button.setCheckable(True)
        self.collapse_button.toggled.connect(self._toggle_comparison)
        row.addWidget(self.collapse_button)
        root.addLayout(row)
        # Second row: what the plots show (shortened to one line, complete in
        # its tooltip) with the plot shortcuts beside it.
        self.summary = ElidedLabel()
        navigation = QtWidgets.QHBoxLayout()
        navigation.setSpacing(4)
        self.heat_button = QtWidgets.QPushButton("Heatmap")
        self.heat_button.setToolTip("Scroll to the heatmap.")
        self.mean_button = QtWidgets.QPushButton("Mean PSTH")
        self.mean_button.setToolTip("Scroll to the mean PSTH.")
        self.settings_button = QtWidgets.QPushButton("Filters…")
        self.settings_button.setToolTip("PSTH filters and normalization.")
        self.heat_button.clicked.connect(lambda: self._show_plot(panel.plot_heat))
        self.mean_button.clicked.connect(lambda: self._show_plot(panel.plot_avg))
        self.settings_button.clicked.connect(lambda: panel._toggle_section_popup("psth", True))
        navigation.addWidget(self.summary, 1)
        navigation.addWidget(self.heat_button)
        navigation.addWidget(self.mean_button)
        navigation.addWidget(self.settings_button)
        root.addLayout(navigation)
        self.sync_controls()

    def _choose_name(self, index) -> None:
        if self.name.itemData(index) == "marker":
            self.panel.combo_align.setCurrentText("Event marker")
            return
        name = self.name.itemText(index)
        self.panel.combo_align.setCurrentIndex(1)
        source = self.panel.combo_behavior_name
        source.setCurrentText(name)
        source.activated.emit(source.currentIndex())

    def sync_controls(self, *_args) -> None:
        panel = self.panel
        marker = panel.combo_align.currentText() == "Event marker"
        for target, source in ((self.name, panel.combo_behavior_name),
                               (self.align, panel.combo_behavior_align)):
            names = [source.itemText(index) for index in range(source.count())]
            if target is self.name:
                names.append(f"Event marker · {panel._marker_label()}" if panel.marker_name.text().strip() else "Event marker")
            with QtCore.QSignalBlocker(target):
                if names != [target.itemText(index) for index in range(target.count())]:
                    target.clear()
                    target.addItems(names)
                if target is self.name:
                    target.setItemData(len(names) - 1, "marker")
                target.setCurrentIndex(len(names) - 1 if marker and target is self.name else source.currentIndex())
        self.align.setEnabled(not marker)
        for target, source in ((self.pre, panel.spin_pre), (self.post, panel.spin_post)):
            with QtCore.QSignalBlocker(target):
                target.set_unit(source.unit)
                target.setValue(source.value())

    def _toggle_comparison(self, collapsed) -> None:
        panel = self.panel
        if not hasattr(panel, "_behavior_comparison_scroll"):
            return
        panel._behavior_comparison_scroll.setVisible(not collapsed and panel._event_category() == "behavior")
        self.collapse_button.setText("Show comparison" if collapsed else "Hide comparison")
        if not collapsed and panel.behavior_zone_panel is not None:
            panel.behavior_zone_panel.sync_workspace()

    def _show_plot(self, plot) -> None:
        panel = self.panel
        card = panel._plot_card_by_widget.get(plot, plot)
        if card.isHidden():
            panel.combo_view_layout.setCurrentText("Standard")
        # Include the heatmap's display controls, which sit above its image.
        target = card if plot is panel.plot_heat else plot
        y = target.mapTo(panel._results_splitter, QtCore.QPoint(0, 0)).y()
        panel._results_scroll.verticalScrollBar().setValue(max(0, y - 8))

    def update_summary(self) -> None:
        panel = self.panel
        warning = legacy_import_warning(panel._behavior_sources)
        self.import_warning.setText(warning)
        self.import_warning.setVisible(bool(warning))
        if panel._event_category() != "behavior":
            return
        name = panel.combo_behavior_name.currentText() or "No behavior selected"
        align = panel.combo_behavior_align.currentText()
        edge = "end" if align.endswith("offset") else "start"
        if align.startswith("Transition"):
            name = f"{panel.combo_behavior_from.currentText()} → {panel.combo_behavior_to.currentText()}"
            edge = "B start after A"
        rows = ("recording averages" if getattr(panel, "_last_psth_display_level", "trials") == "animals"
                else "individual bouts")
        units = panel._psth_units()
        self.summary.setText(
            f"Showing {name} only · zero = behavior {edge} · rows = {rows} · color = {units}. "
            "Real seconds, not stretched bouts. Pre/Post here are separate from Before/During/After above.")
        if panel.combo_align.currentText() == "Event marker":
            label = escape(panel._marker_label())
            self.summary.setText(f"PSTH zero = {label} for each recording.")
            panel.plot_heat.setTitle(f"{label} · Heatmap ({rows})")
            panel.plot_avg.setTitle(f"{label} · Mean PSTH ± SEM")
            for plot in (panel.plot_heat, panel.plot_avg):
                plot.setLabel("bottom", f"Seconds from {label}")
            return
        if panel.combo_align.currentIndex() == 0:
            self.summary.setText("PSTH source is a Doric channel. Choose Behavior in Setup to plot these scored behaviors.")
            return
        # Presentation only: never modify the stored heatmap, event times or windows.
        safe_name = str(name).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        panel.plot_heat.setTitle(f"{safe_name} · Heatmap ({rows})")
        panel.plot_avg.setTitle(f"{safe_name} · Mean PSTH ± SEM")
        panel.plot_heat.setLabel("bottom", f"Seconds from behavior {edge}")
        panel.plot_heat.getAxis("bottom").setStyle(showValues=True)
        panel.plot_heat.getAxis("bottom").setHeight(42)
        panel.plot_avg.setLabel("bottom", f"Seconds from behavior {edge}")


_EXPANDED_ROLE = QtCore.Qt.ItemDataRole.UserRole + 7
# Comparison table: the Events cell opens that row's single-event list.
_EVENTS_COLUMN = _COLUMNS.index("Events")
_EXPANDABLE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 8
_EVENTS_GLYPH_WIDTH = 20      # room kept at the right of the count for the + / - glyph


class EventsCellDelegate(QtWidgets.QStyledItemDelegate):
    """Events count followed by a small "+" ("-" while its list is open)."""

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QtCore.QSize(size.width() + _EVENTS_GLYPH_WIDTH, size.height())

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        if not index.data(_EXPANDABLE_ROLE):
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(option.palette.color(QtGui.QPalette.ColorRole.Text), 1.6))
        center = QtCore.QPoint(option.rect.right() - _EVENTS_GLYPH_WIDTH // 2, option.rect.center().y())
        painter.drawLine(center.x() - 4, center.y(), center.x() + 4, center.y())
        if not index.data(_EXPANDED_ROLE):
            painter.drawLine(center.x(), center.y() - 4, center.x(), center.y() + 4)
        painter.restore()


def _plus_rect(chip: QtCore.QRect) -> QtCore.QRect:
    """Click target of the small '+' inside a behavior chip (right end)."""
    return QtCore.QRect(chip.right() - 26, chip.center().y() - 10, 20, 20)


class BehaviorChoiceDelegate(QtWidgets.QStyledItemDelegate):
    """Large, theme-aware check targets without changing the item's check role."""

    def sizeHint(self, option, index):
        return QtCore.QSize(_CHIP_WIDTH, _CHIP_HEIGHT)

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        checked = index.data(QtCore.Qt.ItemDataRole.CheckStateRole) == QtCore.Qt.CheckState.Checked.value
        rect = option.rect.adjusted(2, 2, -2, -2)
        accent = option.palette.color(QtGui.QPalette.ColorRole.Highlight)
        fill = QtGui.QColor(accent)
        fill.setAlpha(65 if checked else 12)
        border = accent if checked else option.palette.color(QtGui.QPalette.ColorRole.Mid)
        painter.setPen(QtGui.QPen(border, 2 if option.state & QtWidgets.QStyle.StateFlag.State_HasFocus else 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6, 6)
        box = QtCore.QRect(rect.left() + 7, rect.center().y() - 7, 14, 14)
        painter.setPen(QtGui.QPen(border, 1.5))
        painter.setBrush(accent if checked else option.palette.color(QtGui.QPalette.ColorRole.Base))
        painter.drawRoundedRect(box, 3, 3)
        if checked:
            painter.setPen(QtGui.QPen(option.palette.color(QtGui.QPalette.ColorRole.HighlightedText), 2))
            painter.drawLine(box.left() + 3, box.top() + 7, box.left() + 6, box.top() + 10)
            painter.drawLine(box.left() + 6, box.top() + 10, box.left() + 11, box.top() + 4)
        # Small "+" (or "-" when open) at the right: opens this behavior's event list.
        expanded = bool(index.data(_EXPANDED_ROLE))
        plus = _plus_rect(option.rect)
        painter.setPen(QtGui.QPen(option.palette.color(QtGui.QPalette.ColorRole.Text), 1.6))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        center = plus.center()
        painter.drawLine(center.x() - 4, center.y(), center.x() + 4, center.y())
        if not expanded:
            painter.drawLine(center.x(), center.y() - 4, center.x(), center.y() + 4)
        painter.setPen(option.palette.color(QtGui.QPalette.ColorRole.Text))
        text_rect = rect.adjusted(28, 0, -(plus.width() + 8), 0)
        text = option.fontMetrics.elidedText(str(index.data()), QtCore.Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()


class BehaviorCheckList(QtWidgets.QListWidget):
    """Make the whole behavior row a check target, including its text."""

    heightChanged = QtCore.Signal()
    eventsRequested = QtCore.Signal(str)

    def sizeHint(self):
        return QtCore.QSize(660, self._wanted_height())

    def _wanted_height(self) -> int:
        """Height of the chip rows in use: one or two rows, never empty space."""
        shown = sum(not self.item(index).isHidden() for index in range(self.count()))
        per_row = max(1, self.viewport().width() // max(1, self.gridSize().width()))
        rows = max(1, min(_CHIP_MAX_ROWS, -(-shown // per_row)))
        return rows * max(1, self.gridSize().height()) + 2 * self.frameWidth() + 2

    def fit_height(self) -> None:
        """Resize to the rows in use and disappear while there is no behavior."""
        try:
            has_items = self.count() > 0
        except RuntimeError:
            return  # deferred call arrived after the list was deleted
        height = self._wanted_height()
        changed = self.isHidden() == has_items or self.maximumHeight() != height
        self.setFixedHeight(height)
        self.setVisible(has_items)
        if changed:
            self.heightChanged.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Chips per row depend on the width, so the row count can change.
        QtCore.QTimer.singleShot(0, self.fit_height)

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        if (event.button() == QtCore.Qt.MouseButton.LeftButton and item is not None
                and _plus_rect(self.visualItemRect(item)).contains(event.position().toPoint())):
            self.eventsRequested.emit(item.text())
            event.accept()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton and item is not None:
            self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
            self.setCurrentItem(item)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked
                               if item.checkState() == QtCore.Qt.CheckState.Checked
                               else QtCore.Qt.CheckState.Checked)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ElidedLabel(QtWidgets.QLabel):
    """One-line status: shortened with an ellipsis on screen, complete in the tooltip."""

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setToolTip(text)
        self.update()

    def minimumSizeHint(self):
        return QtCore.QSize(60, self.fontMetrics().height() + 2)

    def sizeHint(self):
        return QtCore.QSize(200, self.fontMetrics().height() + 2)

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
        text = self.fontMetrics().elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight, self.width())
        painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, text)


class ComparisonPlotWidget(pg.PlotWidget):
    """Stable preferred size: an expanding plot must not inflate its scroll area."""

    def sizeHint(self):
        return QtCore.QSize(400, 160)


class BehaviorZonePanel(QtWidgets.QWidget):
    """Compare behavior bouts; zone analysis uses the established PSTH plots."""

    def sizeHint(self):
        return QtCore.QSize(900, self.compact_height() if self.results_hidden() else 420)

    def results_hidden(self) -> bool:
        return not hasattr(self, "results_splitter") or self.results_splitter.isHidden()

    def compact_height(self) -> int:
        """Height of the toolbar and behavior chips alone (no plot, no table)."""
        layout = self.layout()
        if layout is None:
            return 60
        height = int(layout.sizeHint().height())
        if not self.results_hidden():
            height -= int(self.results_splitter.sizeHint().height()) + layout.spacing()
        return max(30, height)

    def _arrange_toolbar(self, folded: bool) -> None:
        """Lay the toolbar out on one row, or on two when one is too narrow.

        One row:  [selection] status ........ [windows and actions]
        Two rows: [selection] status / [windows and actions]
        """
        if folded == self._toolbar_folded:
            return
        self._toolbar_folded = folded
        first, second = self._toolbar_rows
        for row in (first, second):
            while row.count():
                row.takeAt(0)
        first.addWidget(self._selection_group)
        first.addWidget(self.status, 1)
        if folded:
            second.addWidget(self._action_group)
            second.addStretch(1)
        else:
            first.addWidget(self._action_group)

    def _refold_toolbar(self) -> None:
        """Use one toolbar row whenever the current width allows it."""
        needed = (self._selection_group.sizeHint().width()
                  + self._action_group.sizeHint().width() + _STATUS_MIN_WIDTH)
        folded = self.width() < needed
        if folded != self._toolbar_folded:
            self._arrange_toolbar(folded)
            QtCore.QTimer.singleShot(0, self._layout_changed)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refold_toolbar()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self.sync_workspace)

    def _layout_changed(self, *_args) -> None:
        try:
            self.updateGeometry()
        except RuntimeError:
            return  # deferred call arrived after the panel was closed
        self.sync_workspace()

    def _set_results_visible(self, visible: bool) -> None:
        """Show the plot and table only while a result exists."""
        if self.results_splitter.isHidden() != visible:
            return
        self.results_splitter.setVisible(visible)
        self._layout_changed()

    def sync_workspace(self) -> None:
        """Give the comparison only the height it needs; plots get the rest.

        Without results the comparison shrinks to its toolbar and chips. With
        results it takes the height the user last chose with the divider.
        """
        panel = self.panel
        workspace = getattr(panel, "_behavior_workspace", None)
        scroll = getattr(panel, "_behavior_comparison_scroll", None)
        try:
            if workspace is None or scroll is None or scroll.isHidden():
                return
        except RuntimeError:
            return  # deferred call arrived after the panel was closed
        total = sum(workspace.sizes()) or 1000
        if self.results_hidden():
            top = self.compact_height() + 4
        else:
            saved = panel._behavior_workspace_sizes
            top = int(round(total * saved[0] / max(1, sum(saved))))
            top = max(top, self.compact_height() + _RESULTS_MIN_HEIGHT)
        top = min(top, max(1, total - 120))
        workspace.setSizes([top, max(1, total - top)])

    def hasHeightForWidth(self):
        return False

    def __init__(self, panel: QtWidgets.QWidget) -> None:
        super().__init__(panel)
        self.panel = panel
        self._last_rows: list[dict[str, object]] = []
        self._last_group_rows: list[dict[str, object]] = []
        self._last_export_context: dict[str, object] = {}
        self._pending_arena_path = ""
        self._pending_arena_category = ""
        self._pending_arena_options: list[dict[str, str]] = []
        self._selection_touched = False
        self._compare_timer = QtCore.QTimer(self)
        self._compare_timer.setSingleShot(True)
        self._compare_timer.setInterval(180)
        self._compare_timer.timeout.connect(lambda: self._compute())

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)

        # Arena selection belongs beside the existing Setup file loader.
        self.arena_combo = panel.combo_arena
        self.arena_hint = panel.lbl_arena_hint
        self.arena_combo.activated.connect(lambda _index: self._use_arena())

        # ------------------------------------------------------------------
        # One toolbar: [file | filter | All None Combine] status [windows | actions]
        # It folds into stacked rows when the panel is too narrow for one line.
        # ------------------------------------------------------------------
        self.selectors: dict[str, QtWidgets.QListWidget] = {}
        self.searches: dict[str, QtWidgets.QLineEdit] = {}
        self._selection_group = QtWidgets.QWidget()
        selection_row = QtWidgets.QHBoxLayout(self._selection_group)
        selection_row.setContentsMargins(0, 0, 0, 0)
        selection_row.setSpacing(4)
        self.add_file_button = QtWidgets.QPushButton("Add events…")
        self.add_file_button.clicked.connect(lambda _checked=False: self._load("behavior"))
        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Filter behaviors")
        search.setClearButtonEnabled(True)
        search.setMinimumWidth(110)
        search.setMaximumWidth(260)
        search.textChanged.connect(lambda value: self._filter("behavior", value))
        self.select_visible_button = QtWidgets.QPushButton("All")
        self.select_visible_button.setToolTip("Select every behavior shown by the filter.")
        self.clear_selection_button = QtWidgets.QPushButton("None")
        self.clear_selection_button.setToolTip("Clear the selection.")
        self.select_visible_button.clicked.connect(lambda: self._set_visible_selection(True))
        self.clear_selection_button.clicked.connect(lambda: self._set_visible_selection(False, all_items=True))
        self.combine_button = QtWidgets.QPushButton("Combine…")
        self.combine_button.setToolTip("Combine checked labels into one OR event; keep the originals.")
        self.combine_button.clicked.connect(panel._open_combine_events)
        for widget in (self.add_file_button, search, self.select_visible_button,
                       self.clear_selection_button, self.combine_button):
            selection_row.addWidget(widget)

        # Internal mirrors use the existing Individual / Group and file pickers.
        self.scope = QtWidgets.QComboBox(self)
        self.scope.addItems(["Selected recording", "Group (all matched recordings)"])
        self.scope.hide()
        self.recording = QtWidgets.QComboBox(self)
        self.recording.hide()
        self.scope.currentIndexChanged.connect(self._scope_changed)
        self.recording.currentIndexChanged.connect(self._recording_changed)
        self.before = QtWidgets.QDoubleSpinBox()
        self.before.setRange(0.05, 300.0)
        self.before.setValue(2.0)
        self.before.setSuffix(" s")
        self.before.setToolTip("Seconds averaged before each bout starts.")
        self.after = QtWidgets.QDoubleSpinBox()
        self.after.setRange(0.05, 300.0)
        self.after.setValue(2.0)
        self.after.setSuffix(" s")
        self.after.setToolTip("Seconds averaged after each bout ends.")
        self.offset = QtWidgets.QDoubleSpinBox()
        self.offset.setRange(-86400.0, 86400.0)
        self.offset.setDecimals(3)
        self.offset.setSuffix(" s")
        self.offset.setToolTip("Adds seconds to bout times in this comparison only, not the PSTH/heatmap. "
                               "Use Sync for a clock correction shared by all analyses.")
        self.offset.valueChanged.connect(self._update_pairing)
        self._action_group = QtWidgets.QWidget()
        action_row = QtWidgets.QHBoxLayout(self._action_group)
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(4)
        for label, widget, width in (("Before", self.before, 84), ("After", self.after, 84),
                                     ("Offset", self.offset, 104)):
            caption = QtWidgets.QLabel(label)
            caption.setToolTip(widget.toolTip())
            widget.setMinimumWidth(width)
            widget.setMaximumWidth(width + 16)
            action_row.addWidget(caption)
            action_row.addWidget(widget)
        self.details_button = QtWidgets.QToolButton()
        self.details_button.setText("Details")
        self.details_button.setToolTip("Show how recordings are paired and which seconds each window averages.")
        self.details_button.setCheckable(True)
        self.export_button = QtWidgets.QPushButton("Export CSV…")
        self.export_button.setToolTip("Export the comparison table as CSV.")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        self.compute_button = QtWidgets.QPushButton("Refresh")
        self.compute_button.setToolTip("Results update automatically; click to refresh immediately.")
        self.compute_button.clicked.connect(self._compute)
        for widget in (self.details_button, self.export_button, self.compute_button):
            action_row.addWidget(widget)

        self.status = ElidedLabel("")
        self.status.setAccessibleName("Behavior comparison status")
        self._toolbar_rows = (QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout())
        for toolbar_row in self._toolbar_rows:
            toolbar_row.setSpacing(8)
            root.addLayout(toolbar_row)
        self._toolbar_folded: bool | None = None
        self._arrange_toolbar(folded=False)

        # Behavior chips: as tall as their rows need (two at most, then scroll),
        # and absent while no behavior file is loaded.
        listing = BehaviorCheckList()
        listing.setItemDelegate(BehaviorChoiceDelegate(listing))
        listing.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        listing.setWrapping(True)
        listing.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        listing.setGridSize(QtCore.QSize(_CHIP_WIDTH, _CHIP_HEIGHT))
        listing.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        listing.setAccessibleName("Behaviors included in the comparison")
        listing.setToolTip("Before / During / After comparison. Click a behavior, or press Space, "
                           "to add or remove it.")
        listing.itemChanged.connect(self._selection_changed)
        listing.heightChanged.connect(self._layout_changed)
        listing.eventsRequested.connect(self._toggle_event_list)
        root.addWidget(listing)
        self.selectors["behavior"], self.searches["behavior"] = listing, search
        # Inline event list of one behavior (opened by the chip's "+").
        from event_list_dialog import EventListPanel
        self.event_list = EventListPanel(panel)
        self.event_list.closed.connect(lambda: self._toggle_event_list(""))
        self.event_list.hide()
        root.addWidget(self.event_list)

        self.timing_help = QtWidgets.QLabel()
        self.timing_help.setWordWrap(True)
        self.timing_help.setToolTip(
            "Example: a bout starts at 10 s and ends at 13 s. With Before = 2 s and "
            "After = 2 s, the comparison averages 8–10 s, 10–13 s, and 13–15 s. "
            "These are three summaries, not equally long time segments. Before/after "
            "windows can contain other behaviors. PSTH Pre = 2 s, Post = 5 s aligned "
            "to onset instead shows 8–15 s, with zero at 10 s. Bouts are not stretched. "
            "Individual heatmap rows are bouts; Group rows follow the existing animal/trial "
            "display option. Colors show signal values, not behavior probability. "
            "PSTH filters/normalization do not apply to the complete-bout comparison.")
        root.addWidget(self.timing_help)
        self.timing_help.hide()
        self.pairing = QtWidgets.QLabel("")
        self.pairing.setWordWrap(True)
        self.pairing.setVisible(False)
        self.details_button.toggled.connect(self.pairing.setVisible)
        self.details_button.toggled.connect(self.timing_help.setVisible)
        self.details_button.toggled.connect(self._layout_changed)
        root.addWidget(self.pairing)

        self.response = ComparisonPlotWidget(title="Mean fiber signal per behavior")
        self.response.setToolTip(
            "Means in the loaded processed signal's units. PSTH normalization and "
            "event filters apply to the PSTH plots; this comparison uses complete bouts.")
        # The Before / During / After ticks name the axis; a label would only
        # take height from the data.
        self.response.setMinimumHeight(120)
        # Keep names outside the data rectangle, even when many behaviors are
        # selected or the comparison panel is made narrow.
        self.response_container = QtWidgets.QWidget()
        response_layout = QtWidgets.QVBoxLayout(self.response_container)
        response_layout.setContentsMargins(0, 0, 0, 0)
        response_layout.setSpacing(0)
        self.response_legend = QtWidgets.QScrollArea()
        self.response_legend.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.response_legend.setWidgetResizable(True)
        self.response_legend.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.response_legend.setFixedHeight(30)
        self.response_legend.setAccessibleName("Behavior comparison legend")
        legend_contents = QtWidgets.QWidget()
        self.response_legend_layout = QtWidgets.QHBoxLayout(legend_contents)
        self.response_legend_layout.setContentsMargins(6, 0, 6, 0)
        self.response_legend_layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinimumSize)
        self.response_legend.setWidget(legend_contents)
        self.response_legend.hide()
        response_layout.addWidget(self.response_legend)
        response_layout.addWidget(self.response, 1)
        self.table = QtWidgets.QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(24)
        # Events cell: "+" opens the row's single-event list inside the table.
        self.table.setItemDelegateForColumn(_EVENTS_COLUMN, EventsCellDelegate(self.table))
        self.table.cellClicked.connect(self._table_cell_clicked)
        self._table_row_keys: list = []          # (file id or "", label) per table row; None = list row
        self._table_events_open = None           # key of the row whose event list is open
        self._table_events_show_removed = False
        self.results_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.results_splitter.setChildrenCollapsible(False)
        self.results_splitter.setHandleWidth(8)
        self.results_splitter.addWidget(self.response_container)
        self.results_splitter.addWidget(self.table)
        self.results_splitter.setStretchFactor(0, 1)
        self.results_splitter.setStretchFactor(1, 2)
        self.results_splitter.setSizes([420, 800])
        # The plot and table take room only once there is a result to show.
        self.results_splitter.hide()
        root.addWidget(self.results_splitter, 1)
        for control in (self.before, self.after, self.offset):
            control.setKeyboardTracking(False)
            control.valueChanged.connect(self._queue_comparison)
            control.valueChanged.connect(panel._queue_settings_save)
        for control in (panel.spin_pre, panel.spin_post):
            control.valueChanged.connect(self._update_timing_help)
        for control in (panel.combo_behavior_name, panel.combo_behavior_align,
                        panel.combo_behavior_from, panel.combo_behavior_to,
                        panel.combo_psth_normalization, panel.combo_align):
            control.currentTextChanged.connect(self._update_timing_help)
        panel.cb_sync_use_aligned.toggled.connect(self._queue_comparison)
        self.reload()

    def _category(self) -> str:
        return "behavior"

    def settings_state(self) -> dict:
        return {"labels": self._selected_labels(), "before_s": self.before.value(),
                "after_s": self.after.value(), "offset_s": self.offset.value(),
                "selection_touched": self._selection_touched,
                "search": self.searches["behavior"].text(),
                "collapsed": self.panel.behavior_psth_bar.collapse_button.isChecked()}

    def restore_settings(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self._selection_touched = bool(state.get("selection_touched", True))
        for control, key, default in ((self.before, "before_s", 2.), (self.after, "after_s", 2.),
                                      (self.offset, "offset_s", 0.)):
            value = float(state.get(key, default))
            if np.isfinite(value):
                with QtCore.QSignalBlocker(control):
                    control.setValue(value)
        selected = set(state.get("labels", []))
        listing = self.selectors["behavior"]
        with QtCore.QSignalBlocker(listing):
            for index in range(listing.count()):
                item = listing.item(index)
                item.setCheckState(QtCore.Qt.CheckState.Checked if item.text() in selected
                                   else QtCore.Qt.CheckState.Unchecked)
        self.searches["behavior"].setText(str(state.get("search", "")))
        self.panel.behavior_psth_bar.collapse_button.setChecked(bool(state.get("collapsed", False)))
        self._update_pairing()
        self._queue_comparison()

    def sync_context(self, *_args) -> None:
        """Follow the main toolbar, including project restores and group changes."""
        self.scope.setCurrentIndex(self.panel.tab_visual_mode.currentIndex())
        selected = self.panel.combo_individual_file.currentText().strip()
        index = next((i for i, proc in enumerate(self.panel._processed)
                      if self.panel._file_id_for_proc(proc) == selected), 0)
        match = self.recording.findData(index)
        if match >= 0:
            self.recording.setCurrentIndex(match)
        self._update_arena_options()
        self._update_pairing()
        self._queue_comparison()

    def _load(self, category: str) -> None:
        self.panel._load_behavior_zone_files(category)
        self.reload()

    def choose_workbook_arena(self, path: str, category: str,
                              options: list[dict[str, str]]) -> None:
        """Expose detected arena sheets before importing an ambiguous workbook."""
        self._pending_arena_path = path
        self._pending_arena_category = category
        self._pending_arena_options = options
        self._update_arena_options()
        self.status.setText("Choose the arena containing the fiber-associated animal.")

    def _update_arena_options(self) -> None:
        if not hasattr(self, "arena_combo") or not hasattr(self, "recording"):
            return
        if self._pending_arena_path:
            options = self._pending_arena_options
            chosen = ""
            path = self._pending_arena_path
        else:
            selected = self._selected_recordings()
            info = self.panel._match_behavior_source(selected[0]) if len(selected) == 1 else None
            report = (info or {}).get("import_report") or {}
            options = report.get("arena_options", []) if isinstance(report, dict) else []
            chosen = str((info or {}).get("sheet", "") or "")
            path = str(report.get("arena_workbook", "") or "") if isinstance(report, dict) else ""
        self.arena_combo.blockSignals(True)
        self.arena_combo.clear()
        if self._pending_arena_path:
            self.arena_combo.addItem("Choose arena…", "")
        for option in options:
            self.arena_combo.addItem(option.get("label", option["sheet"]), option["sheet"])
            self.arena_combo.setItemData(self.arena_combo.count() - 1, option["sheet"],
                                         QtCore.Qt.ItemDataRole.ToolTipRole)
        index = self.arena_combo.findData(chosen)
        if index >= 0:
            self.arena_combo.setCurrentIndex(index)
        self.arena_combo.blockSignals(False)
        source_available = bool(path) and Path(path).is_file()
        for widget in (self.panel.lbl_arena, self.arena_combo, self.arena_hint):
            widget.setVisible(bool(options))
        self.arena_combo.setEnabled(bool(options) and self.scope.currentIndex() == 0
                                    and source_available)
        if self._pending_arena_path:
            self.arena_hint.setText(f"Detected {len(options)} arena sheet(s) in {Path(path).name}.")
        elif options:
            self.arena_hint.setText(
                "Select another arena to replace this workbook’s active sheet."
                if source_available else "Workbook unavailable; saved arena data remain usable.")
        else:
            self.arena_hint.setText("Add an EthoVision workbook to find its arenas.")

    def _use_arena(self) -> None:
        sheet = self.arena_combo.currentData()
        if not sheet:
            return
        if self._pending_arena_path:
            path = self._pending_arena_path
            category = self._pending_arena_category
            options = self._pending_arena_options
            target_index = None
            selected = self._selected_recordings()
            if len(selected) == 1:
                current = self.panel._match_behavior_source(selected[0]) or {}
                current_report = current.get("import_report") or {}
                if str(current_report.get("arena_workbook", "")) == path:
                    target_index = self.recording.currentData()
        else:
            recordings = self._selected_recordings()
            if len(recordings) != 1:
                return
            info = self.panel._match_behavior_source(recordings[0]) or {}
            report = info.get("import_report") or {}
            path = str(report.get("arena_workbook", "") or "")
            category = str(report.get("category", "zone") or "zone")
            options = report.get("arena_options", [])
            target_index = self.recording.currentData()
            if sheet == info.get("sheet"):
                self.status.setText(f"{sheet} is already the active arena.")
                return
        if not Path(path).is_file():
            self.status.setText("The workbook is unavailable. The saved arena data remain unchanged.")
            return
        try:
            changed = self.panel._add_generic_behavior_zone_file(
                path, category, sheet_name=str(sheet), arena_options=options,
                replace_arena=target_index is not None, target_index=target_index)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Arena selection failed", str(exc))
            return
        if changed:
            self._pending_arena_path = ""
            self._pending_arena_category = ""
            self._pending_arena_options = []
            if changed in {"behavior", "zone"}:
                self.panel._set_event_category(changed)
            self.reload()
            self.status.setText(f"Using {sheet} for the selected fiber recording.")

    def _filter(self, category: str, value: str) -> None:
        for index in range(self.selectors[category].count()):
            item = self.selectors[category].item(index)
            item.setHidden(value.casefold() not in item.text().casefold())
        self.selectors[category].fit_height()

    def _set_visible_selection(self, checked: bool, *, all_items: bool = False) -> None:
        listing = self.selectors["behavior"]
        self._selection_touched = True
        with QtCore.QSignalBlocker(listing):
            for index in range(listing.count()):
                item = listing.item(index)
                if all_items or not item.isHidden():
                    item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        selected = self._selected_labels()
        if selected and self.panel.combo_behavior_name.currentText() not in selected:
            self.panel.combo_behavior_name.setCurrentText(selected[0])
        self._queue_comparison()

    def _scope_changed(self) -> None:
        self.recording.setEnabled(self.scope.currentIndex() == 0)
        self._update_arena_options()
        self._update_pairing()
        self._queue_comparison()

    def _selection_changed(self, item=None) -> None:
        self._selection_touched = True
        self.panel._queue_settings_save()
        # Two or more checked behaviors are drawn side by side (and colored in the trace).
        self.panel.queue_multi_behavior()
        if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
            name = item.text()  # the list may be rebuilt below
            # Checking a behavior means "show its PSTH": leave Event-marker
            # alignment, which otherwise keeps the PSTH empty without markers.
            if self.panel.combo_align.currentText() == "Event marker":
                self.panel.combo_align.setCurrentText("Behavior (CSV/XLSX)")
            self.panel.combo_behavior_name.setCurrentText(name)
            self._queue_comparison()
            return
        elif item is not None and item.text() == self.panel.combo_behavior_name.currentText():
            remaining = self._selected_labels()
            if remaining:
                self.panel.combo_behavior_name.setCurrentText(remaining[-1])
        self._queue_comparison()

    def follow_psth_selection(self, name: str) -> None:
        """Use the PSTH behavior for a single comparison; retain multi-selection."""
        if self.panel._event_category() != "behavior":
            return
        keep_multiple = len(self._selected_labels()) > 1
        listing = self.selectors["behavior"]
        matches = listing.findItems(name, QtCore.Qt.MatchFlag.MatchExactly)
        if not matches:
            return
        with QtCore.QSignalBlocker(listing):
            for index in range(listing.count()):
                item = listing.item(index)
                if item is matches[0]:
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
                elif not keep_multiple:
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        listing.scrollToItem(matches[0])
        self._queue_comparison()

    def _recording_changed(self) -> None:
        self._update_arena_options()
        self._update_pairing()
        self._queue_comparison()

    def _update_pairing(self) -> None:
        if not hasattr(self, "pairing"):
            return
        if self.scope.currentIndex() == 1:
            matched = sum(self.panel._match_behavior_source(proc) is not None
                          for proc in self.panel._processed)
            self.pairing.setText(
                f"Pairing: {matched} of {len(self.panel._processed)} fiber recordings have a behavior source. "
                "Each recording is weighted once; mouse identities are not automatically grouped across sessions.")
            return
        selected = self._selected_recordings()
        if not selected:
            self.pairing.setText("Load a processed fiber recording first.")
            return
        proc = selected[0]
        info = self.panel._match_behavior_source(proc)
        if info is None:
            self.pairing.setText(f"No behavior source paired with {Path(proc.path).name}.")
            return
        report = info.get("import_report") or {}
        imports = report.get("mamir_imports", []) if isinstance(report, dict) else []
        details = [report] if isinstance(report, dict) and report.get("format") == "mamir" else []
        details.extend(item for item in imports if isinstance(item, dict))
        identity_text = ", ".join(sorted({str(item.get("identity")) for item in details
                                           if item.get("identity")}))
        subject = str(report.get("subject", "") or "") if isinstance(report, dict) else ""
        subject_text = (f"animal {identity_text}" if identity_text else
                        f"subject {subject}" if subject else "generic import")
        fiber_time = np.asarray(self.panel._proc_time(proc), float)
        fiber_span = (f"{fiber_time[0]:.2f}–{fiber_time[-1]:.2f} s"
                      if fiber_time.size else "no time samples")
        category = self._category()
        event_spans = []
        names = set(info.get("behaviors") or {}) | set(info.get("event_behaviors") or {})
        for name in names:
            is_zone = str(name).startswith("Zone: ")
            if (category == "zone") != is_zone:
                continue
            try:
                on, off, _duration = self.panel._extract_behavior_events(info, name)
            except ValueError as exc:
                self.pairing.setText(f"{Path(proc.path).name}: invalid behavior timeline ({exc}). "
                                     "Re-import the behavior file and select its animal/arena ID.")
                return
            if on.size and off.size:
                event_spans.append((float(np.nanmin(on)) + self.offset.value(),
                                    float(np.nanmax(off)) + self.offset.value()))
        event_span = "no bouts in this category"
        if event_spans:
            low = min(start for start, _ in event_spans)
            high = max(stop for _, stop in event_spans)
            overlap = fiber_time.size and high > fiber_time[0] and low < fiber_time[-1]
            event_span = f"{low:.2f}–{high:.2f} s"
            if not overlap:
                event_span += " (NO TIME OVERLAP)"
        active_path = report.get("arena_workbook") if isinstance(report, dict) and info.get("sheet") else ""
        source_name = Path(str(active_path or info.get("source_path", "") or "")).name or "embedded source"
        if info.get("sheet"):
            source_name += f" [{info['sheet']}]"
        self.pairing.setText(
            f"Pairing: {Path(proc.path).name} ↔ {source_name}; {subject_text}. "
            f"Fiber clock: {fiber_span}. {category.title()} bouts: {event_span}. "
            f"Behavior offset: {self.offset.value():+.3f} s. "
            "Check synchronization before interpreting event responses.")

    def _clear_results(self) -> None:
        self._compare_timer.stop()
        self._last_rows = []
        self._last_group_rows = []
        self._last_export_context = {}
        self.export_button.setEnabled(False)
        if hasattr(self, "table"):
            self.table.clearSpans()
            self.table.setRowCount(0)
            self._table_row_keys = []
            self.response.clear()
            self._clear_legend()

    def _clear_legend(self) -> None:
        while self.response_legend_layout.count():
            item = self.response_legend_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()
        self.response_legend.hide()

    def _add_legend_entry(self, name: str, color) -> None:
        entry = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(entry)
        layout.setContentsMargins(0, 0, 8, 0)
        swatch = QtWidgets.QLabel("━ ●")
        swatch.setStyleSheet(f"color: {pg.mkColor(color).name()};")
        label = QtWidgets.QLabel(name)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        entry.setToolTip(name)
        layout.addWidget(swatch)
        layout.addWidget(label)
        self.response_legend_layout.addWidget(entry)
        self.response_legend.show()

    def _queue_comparison(self, *_args) -> None:
        """Combine rapid clicks into one update using data already in memory."""
        if not hasattr(self, "export_button"):
            return
        self._update_timing_help()
        self._clear_results()
        if self.panel._event_category() != "behavior":
            return
        if not self._selected_labels():
            self.status.setText("Click a behavior to compare its fiber signal."
                                if self.selectors["behavior"].count() else
                                "Add a behavior file to compare the signal before, during and after bouts.")
            self._set_results_visible(False)
            return
        if not self._selected_recordings():
            self.status.setText("Load a processed fiber recording first.")
            self._set_results_visible(False)
            return
        if self._pending_arena_path:
            self._set_results_visible(False)
            return
        self.status.setText("Updating comparison…")
        self._compare_timer.start()

    def _update_timing_help(self, *_args) -> None:
        if not hasattr(self, "timing_help"):
            return
        panel = self.panel
        edge = panel.combo_behavior_align.currentText()
        zero = ("behavior end" if edge.endswith("offset") else
                "B onset after A" if edge.startswith("Transition") else "behavior start")
        name = panel.combo_behavior_name.currentText() or "no behavior selected"
        if edge.startswith("Transition"):
            name = f"{panel.combo_behavior_from.currentText()} → {panel.combo_behavior_to.currentText()}"
        heatmap = (f"PSTH / heatmap: {name}; 0 = {zero}; "
                   f"−{panel.spin_pre.value():g} to +{panel.spin_post.value():g} s; "
                   f"colors = {panel.combo_psth_normalization.currentText()}.")
        if panel.combo_align.currentIndex() == 0:
            heatmap = "PSTH / heatmap uses the selected Doric channel, not these behavior checkboxes."
        if panel.combo_align.currentText() == "Event marker":
            heatmap = f"PSTH / heatmap: zero = {escape(panel._marker_label())}."
        shift = (f" Comparison-only shift: {self.offset.value():+g} s; heatmap is NOT shifted."
                 if self.offset.value() else "")
        self.timing_help.setText(
            f"Comparison: {self.before.value():g} s before start → whole bout → "
            f"{self.after.value():g} s after end (processed units).{shift}\n"
            + heatmap + " Hover here for an example.")

    def reload(self) -> None:
        had_choices = any(listing.count() for listing in self.selectors.values())
        previous = {category: {self.selectors[category].item(i).text()
                               for i in range(self.selectors[category].count())
                               if self.selectors[category].item(i).checkState() == QtCore.Qt.CheckState.Checked}
                    for category in self.selectors}
        names: dict[str, set[str]] = {"behavior": set(), "zone": set()}
        for info in self.panel._behavior_sources.values():
            for name in set(info.get("behaviors") or {}) | set(info.get("event_behaviors") or {}):
                category = "zone" if str(name).startswith("Zone: ") else "behavior"
                names[category].add(str(name))
        for category, listing in self.selectors.items():
            listing.blockSignals(True)
            listing.clear()
            for name in sorted(names[category], key=str.casefold):
                item = QtWidgets.QListWidgetItem(name)
                item.setToolTip(name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if name in previous[category]
                                   else QtCore.Qt.CheckState.Unchecked)
                listing.addItem(item)
            listing.blockSignals(False)
            self._filter(category, self.searches[category].text())
        # Keep an open event list in sync with the rebuilt chips (or close it).
        if not self.event_list.isHidden():
            open_label = self.event_list.label
            if open_label in names["behavior"] | names["zone"]:
                listing = self.selectors["behavior"]
                for index in range(listing.count()):
                    listing.item(index).setData(_EXPANDED_ROLE, listing.item(index).text() == open_label)
                self.event_list.reload()
            else:
                self.event_list.hide()
        # Filter and selection tools are only useful once behaviors exist.
        has_behaviors = self.selectors["behavior"].count() > 0
        for widget in (self.searches["behavior"], self.select_visible_button,
                       self.clear_selection_button, self.combine_button):
            widget.setVisible(has_behaviors)
        self._selection_group.adjustSize()
        self._refold_toolbar()
        if (not had_choices or not self._selection_touched) and not self._selected_labels():
            self.follow_psth_selection(self.panel.combo_behavior_name.currentText())
        previous_recording = self.recording.currentData()
        self.recording.clear()
        for index, proc in enumerate(self.panel._processed):
            label = Path(proc.path).name if proc.path else f"Recording {index + 1}"
            self.recording.addItem(label, index)
        if previous_recording is not None:
            match = self.recording.findData(previous_recording)
            if match >= 0:
                self.recording.setCurrentIndex(match)
        self.sync_context()
        self._scope_changed()
        self.status.setText(f"{len(self.panel._processed)} fiber recording(s), "
                            f"{len(names['behavior'])} behaviors, {len(names['zone'])} zones loaded.")
        if self._pending_arena_path:
            self.status.setText("Choose the fiber-associated arena in Setup to load its data.")
        self._queue_comparison()

    # -- single events inside the comparison table ---------------------------
    def _source_for_file(self, file_id: str):
        """Behavior source paired with a recording ('' = all recordings)."""
        if not file_id:
            return None
        for proc in self.panel._processed:
            if (Path(proc.path).stem if proc.path else "recording") == file_id:
                return self.panel._match_behavior_source(proc)
        return None

    def _insert_table_event_list(self, key) -> None:
        """Add, under the last table row, a full-width row holding that row's events."""
        from event_list_dialog import EventListPanel
        file_id, label = key
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._table_row_keys.append(None)
        events = EventListPanel(self.panel, self.table)
        events.show_removed.setChecked(self._table_events_show_removed)
        events.show_removed.toggled.connect(
            lambda checked: setattr(self, "_table_events_show_removed", bool(checked)))
        # Closing rebuilds the table, so let the click finish first.
        events.closed.connect(lambda: QtCore.QTimer.singleShot(0, lambda: self._toggle_table_event_list(key)))
        events.set_label(label, only_source=self._source_for_file(file_id))
        self.table.setSpan(row, 0, 1, self.table.columnCount())
        self.table.setCellWidget(row, 0, events)

        def fit() -> None:
            try:
                self.table.setRowHeight(row, events.sizeHint().height() + 6)
            except RuntimeError:
                pass  # the table was rebuilt meanwhile
        events.sizeChanged.connect(fit)
        fit()
        self.table_event_list = events

    def _table_cell_clicked(self, row: int, column: int) -> None:
        if column != _EVENTS_COLUMN or not 0 <= row < len(self._table_row_keys):
            return
        key = self._table_row_keys[row]
        if key is not None:
            self._toggle_table_event_list(key)

    def _toggle_table_event_list(self, key) -> None:
        """Open the event list under a table row; the same row (or its close) shuts it."""
        self._table_events_open = None if key == self._table_events_open else key
        if self._last_rows or self._last_group_rows:
            self._show_results(self._last_rows, self._last_group_rows)

    def _toggle_event_list(self, label: str) -> None:
        """Open the chip's event list under the chips; same chip (or close) hides it."""
        listing = self.selectors["behavior"]
        current = self.event_list.label if not self.event_list.isHidden() else ""
        target = "" if not label or label == current else label
        for index in range(listing.count()):
            item = listing.item(index)
            item.setData(_EXPANDED_ROLE, item.text() == target)
        if target:
            self.event_list.set_label(target)
            self.event_list.show()
        else:
            self.event_list.hide()
        listing.viewport().update()
        self._layout_changed()

    def _selected_labels(self) -> list[str]:
        listing = self.selectors[self._category()]
        return [listing.item(i).text() for i in range(listing.count())
                if listing.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def _selected_recordings(self) -> list[object]:
        if self.scope.currentIndex() == 1:
            return list(self.panel._processed)
        index = self.recording.currentData()
        return [self.panel._processed[index]] if isinstance(index, int) and \
            0 <= index < len(self.panel._processed) else []

    def _events_for(self, info: dict, labels: list[str]) -> dict[str, dict[str, np.ndarray]]:
        events = {}
        names = set(info.get("behaviors") or {}) | set(info.get("event_behaviors") or {})
        for label in labels:
            if label in names:
                on, off, dur = self.panel._extract_behavior_events(info, label)
                events[label] = {"on": on, "off": off, "dur": dur}
        return events

    def _compute(self) -> None:
        self._clear_results()
        labels = self._selected_labels()
        if not labels:
            self.status.setText("Select at least one behavior.")
            self._set_results_visible(False)
            return
        matched = [proc for proc in self._selected_recordings()
                   if self.panel._match_behavior_source(proc) is not None]
        identifiers = [self.panel._file_id_for_proc(proc) for proc in self._selected_recordings()]
        if len(identifiers) != len(set(identifiers)):
            self.status.setText("Recording names must be unique. Remove duplicate recordings or rename "
                                "same-named files before group comparison.")
            self._set_results_visible(False)
            return
        if self.scope.currentIndex() == 1 and len({proc.output_label for proc in matched}) > 1:
            self.status.setText("Cannot compare different signal units in a group. "
                                "Load matching processed outputs or use Individual view.")
            self._set_results_visible(False)
            return
        self.response.setLabel("left", matched[0].output_label if matched else "Processed units")
        rows: list[dict[str, object]] = []
        clocks = {}
        unmatched = []
        has_bouts = False
        for proc in self._selected_recordings():
            info = self.panel._match_behavior_source(proc)
            file_id = Path(proc.path).stem if proc.path else "recording"
            if info is None:
                unmatched.append(file_id)
                continue
            try:
                events = self._events_for(info, labels)
            except ValueError as exc:
                unmatched.append(f"{file_id} (invalid behavior timeline: {exc}; "
                                 "re-import and select the animal/arena ID)")
                continue
            if not events:
                unmatched.append(file_id)
                continue
            has_bouts = has_bouts or any(np.any(event["dur"] > 0)
                                         for event in events.values())
            time = self.panel._proc_time(proc)
            aligned = getattr(proc, "sync_aligned_time", None)
            uses_aligned = (self.panel.cb_sync_use_aligned.isChecked() and aligned is not None
                            and np.asarray(aligned).size == np.asarray(proc.time).size
                            and np.asarray(aligned).size > 0 and np.any(np.isfinite(aligned)))
            clocks[file_id] = "sync_aligned" if uses_aligned else "original"
            try:
                rows.extend(compare_recording(
                    file_id, time, proc.output, events, events.keys(),
                    pre_s=self.before.value(), post_s=self.after.value(),
                    offset_s=self.offset.value()))
            except ValueError as exc:
                unmatched.append(f"{file_id} ({exc})")
        self._last_rows = rows
        self._last_group_rows = group_recordings(rows)
        self._last_export_context = {
            "combined_labels": json.dumps(self.panel._combined_event_definitions(), sort_keys=True),
            "scope": "group" if self.scope.currentIndex() == 1 else "individual",
            "before_window_s": self.before.value(), "after_window_s": self.after.value(),
            "comparison_offset_s": self.offset.value(),
            "signal_units": matched[0].output_label if matched else "",
            "photometry_clock": next(iter(set(clocks.values()))) if len(set(clocks.values())) == 1 else "mixed",
            "recording_clocks": json.dumps(clocks, sort_keys=True),
            "weighting": "equal recording means; equal bouts within recording",
            "normalization": "processed units; no additional baseline normalization",
            "event_selection": "complete bouts; independent of PSTH filters",
            "import_warning": legacy_import_warning(self.panel._behavior_sources),
        }
        self._show_results(rows, self._last_group_rows)
        included = sum(int(row["events"]) for row in rows)
        contributors = {row['file_id'] for row in rows if int(row["events"]) > 0}
        message = f"{included} events across {len(contributors)} contributing animal/file(s)."
        if self.scope.currentIndex() == 1:
            message += " Group: each animal/file weighted once, as in main pyBer; N is per behavior."
        if self.panel.cb_sync_use_aligned.isChecked() and "original" in clocks.values():
            message += " Warning: some recordings have no aligned clock; their original times are used."
        if unmatched:
            message += " No matching data: " + ", ".join(unmatched[:4])
        if included == 0:
            message += (" Check animal identity, recording pairing, and clock offset."
                        if has_bouts or not rows else
                        " During requires bout start/end times. For timestamp-only events, "
                        "use the existing PSTH pre/post analysis.")
        self.status.setText(message)
        self.export_button.setEnabled(bool(rows))

    def _show_results(self, rows: list[dict[str, object]], group: list[dict[str, object]]) -> None:
        show_group = self.scope.currentIndex() == 1
        displayed = group if show_group else rows
        self.table.clearSpans()
        self.table.setRowCount(0)
        self._table_row_keys = []
        open_key = self._table_events_open
        keys = [("" if show_group else str(row["file_id"]), str(row["label"])) for row in displayed]
        if displayed and open_key not in keys:
            open_key = self._table_events_open = None
        for row, key in zip(displayed, keys):
            index = self.table.rowCount()
            self.table.insertRow(index)
            self._table_row_keys.append(key)
            values = (("Group" if show_group else row["file_id"]), row["label"],
                      row.get("recordings", 1), row["events"], row.get("rejected", ""),
                      row["before"], row["during"], row["after"],
                      row["during_minus_before"], row["after_minus_before"])
            for column, value in enumerate(values):
                text = f"{value:.4g}" if isinstance(value, float) and np.isfinite(value) else str(value)
                item = QtWidgets.QTableWidgetItem(text)
                if column == _EVENTS_COLUMN:
                    item.setData(_EXPANDABLE_ROLE, True)
                    item.setData(_EXPANDED_ROLE, key == open_key)
                    item.setToolTip("Click to list this row's single events: hide, remove or restore them.")
                self.table.setItem(index, column, item)
            if key == open_key:
                self._insert_table_event_list(key)
        self.response.clear()
        self.response.setTitle("Group: mean ± SEM across animals/files" if show_group
                               else "Mean fiber signal per behavior")
        self.response.setToolTip(
            "Each recording contributes once per behavior. SEM uses recording means and requires at least two "
            "recordings. Repeated sessions are not automatically combined by mouse. PSTH filters and baseline "
            "normalization do not apply here." if show_group else
            "Mean across complete bouts in processed signal units; PSTH normalization and filters are separate.")
        self._clear_legend()
        self.response.getAxis("bottom").setTicks([[(0, "Before"), (1, "During"), (2, "After")]])
        for index, row in enumerate(displayed):
            if not int(row["events"]):
                continue
            values = [row["before"], row["during"], row["after"]]
            color = _COLORS[index % len(_COLORS)]
            self._add_legend_entry(str(row["label"]), color)
            self.response.plot([0, 1, 2], values, pen=pg.mkPen(color, width=2),
                               symbol="o", symbolBrush=color, name=str(row["label"]))
            if show_group and int(row["recordings"]) > 1:
                sem = np.array([row[key + "_sem"] for key in ("before", "during", "after")], float)
                self.response.addItem(pg.ErrorBarItem(x=np.arange(3.), y=np.asarray(values),
                                                      top=sem, bottom=sem, beam=.1, pen=pg.mkPen(color)))
        self.response.setXRange(-.2, 2.2, padding=0)
        self.response_legend_layout.addStretch(1)
        self.table.resizeColumnsToContents()
        self._set_results_visible(bool(displayed))

    def _export(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export comparison", "behavior_comparison.csv", "CSV (*.csv)")
        if not path:
            return
        rows = self._last_group_rows if self.scope.currentIndex() == 1 else self._last_rows
        if not rows:
            return
        rows = [{**row, **self._last_export_context} for row in rows]
        for row in rows:
            if "recording_ids" in row:
                row["recording_ids"] = json.dumps(row["recording_ids"])
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.status.setText(f"Exported {len(rows)} row(s) to {path}")
