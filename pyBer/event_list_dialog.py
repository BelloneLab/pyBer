"""Inline list of one behavior's events, with hide / remove / restore.

Opened by the small "+" inside a behavior chip of the Events panel, or by the
"+" in an Events cell of the comparison table (then limited to that row's
recording in Individual scope). Rows:
recording, event number, onset, offset, duration (seconds, as loaded) and
status. Right-click hides selected events from plots and metrics, removes them
from the list, or restores them. Nothing in the loaded arrays or source files
is modified: exclusions are stored per source in its import report (saved with
projects) and applied when events are extracted.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

COLUMNS = ["Recording", "#", "Onset (s)", "Offset (s)", "Duration (s)", "Status"]
MAX_VISIBLE_ROWS = 8


def _fmt(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "—"


class _NumericItem(QtWidgets.QTableWidgetItem):
    """Sort numerically (NaN last) instead of by text."""

    def __lt__(self, other):
        a = self.data(QtCore.Qt.ItemDataRole.UserRole)
        b = other.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(a, float) and isinstance(b, float):
            return (np.isnan(a), a) < (np.isnan(b), b)
        return super().__lt__(other)


class EventListPanel(QtWidgets.QFrame):
    """Compact table of the events of one behavior across loaded recordings."""

    closed = QtCore.Signal()
    sizeChanged = QtCore.Signal()    # the row count, hence the height, changed

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel
        self.label = ""
        # When set, only the events of this behavior source (one recording).
        self.only_source = None
        self.setObjectName("eventListPanel")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(3)
        top = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-weight: 600;")
        self.summary = QtWidgets.QLabel()
        self.summary.setProperty("class", "hint")
        self.show_removed = QtWidgets.QCheckBox("Show removed")
        self.show_removed.toggled.connect(self.reload)
        restore_all = QtWidgets.QToolButton()
        restore_all.setText("Restore all")
        restore_all.clicked.connect(self._restore_all)
        close = QtWidgets.QToolButton()
        close.setText("×")
        close.setToolTip("Close the event list")
        close.clicked.connect(self.closed.emit)
        for widget in (self.title, self.summary):
            top.addWidget(widget)
        top.addStretch(1)
        for widget in (self.show_removed, restore_all, close):
            top.addWidget(widget)
        root.addLayout(top)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.setToolTip("Right-click events to hide them from plots and metrics, "
                              "remove them, or restore them. Source files are never changed.")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table)

    # -- data ---------------------------------------------------------------
    def set_label(self, label: str, only_source=None) -> None:
        """Show ``label``'s events, for every recording or for one behavior source."""
        self.label = str(label)
        self.only_source = only_source
        self.reload()

    def _events(self) -> List[Tuple[str, int, float, float, float, Optional[str]]]:
        rows = []
        for key, info in (self.panel._behavior_sources or {}).items():
            if self.only_source is not None and info is not self.only_source:
                continue
            if self.label not in (info.get("behaviors") or {}) and \
                    self.label not in (info.get("event_behaviors") or {}):
                continue
            excluded = self.panel._event_exclusions(info)
            try:
                on, off, dur = self.panel._extract_behavior_events_raw(info, self.label)
            except Exception:
                continue
            for index, (start, stop, length) in enumerate(zip(on, off, dur), 1):
                state = excluded.get((self.label, round(float(start), 6)))
                bout = np.isfinite(stop) and stop > start
                if not np.isfinite(length) and bout:
                    length = stop - start
                rows.append((str(key), index, float(start), float(stop) if bout else np.nan,
                             float(length), state))
        return rows

    def reload(self, *_args) -> None:
        if not self.label:
            return
        rows = self._events()
        visible = [row for row in rows if row[5] != "removed" or self.show_removed.isChecked()]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(visible))
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        for r, (key, index, onset, offset, length, state) in enumerate(visible):
            for c, value in enumerate([key, float(index), onset, offset, length, state or "used"]):
                if isinstance(value, float):
                    item = _NumericItem(str(int(value)) if c == 1 else _fmt(value))
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, (key, onset))
                if state:
                    item.setForeground(muted)
                    font = item.font()
                    font.setStrikeOut(state == "removed")
                    item.setFont(font)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        used = sum(1 for row in rows if not row[5])
        hidden = sum(1 for row in rows if row[5] == "hidden")
        removed = sum(1 for row in rows if row[5] == "removed")
        self.title.setText(f"{self.label} · events")
        self.summary.setText(f"{used} used · {hidden} hidden · {removed} removed")
        # Height follows the rows (up to MAX_VISIBLE_ROWS, then scroll).
        row_h = self.table.verticalHeader().defaultSectionSize()
        shown = max(1, min(MAX_VISIBLE_ROWS, len(visible)))
        self.table.setFixedHeight(self.table.horizontalHeader().height() + shown * row_h + 2 * self.table.frameWidth() + 2)
        self.sizeChanged.emit()

    # -- actions ------------------------------------------------------------
    def _selected_keys(self) -> List[Tuple[str, float]]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole + 1) for r in rows]

    def _set_selected(self, state: Optional[str]) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        for key, onset in keys:
            info = self.panel._behavior_sources.get(key)
            if info is not None:
                self.panel._set_event_exclusion(info, self.label, onset, state)
        self.reload()
        self.panel._on_event_exclusions_changed()

    def _restore_all(self) -> None:
        for info in (self.panel._behavior_sources or {}).values():
            if self.only_source is not None and info is not self.only_source:
                continue
            report = info.get("import_report")
            if isinstance(report, dict) and report.get("event_exclusions"):
                report["event_exclusions"] = [entry for entry in report["event_exclusions"]
                                              if str(entry[0]) != self.label]
        self.reload()
        self.panel._on_event_exclusions_changed()

    def _context_menu(self, position: QtCore.QPoint) -> None:
        if not self._selected_keys():
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Hide from plots and metrics", lambda: self._set_selected("hidden"))
        menu.addAction("Remove", lambda: self._set_selected("removed"))
        menu.addAction("Restore", lambda: self._set_selected(None))
        menu.exec(self.table.viewport().mapToGlobal(position))
