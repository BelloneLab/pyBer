"""Non-modal list of every loaded behavior event, with hide / remove / restore.

Rows: recording, behavior, onset, offset, duration (seconds, as loaded) and
status. Right-click (or the buttons) hides selected events from plots and
metrics, removes them from the session list, or restores them. Nothing in the
loaded arrays or source files is modified: exclusions are stored per source in
its import report (saved with projects) and applied when events are extracted.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

COLUMNS = ["Recording", "Behavior", "#", "Onset (s)", "Offset (s)", "Duration (s)", "Status"]


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


class EventListWindow(QtWidgets.QWidget):
    """One table for all events of all loaded behavior sources."""

    def __init__(self, panel):
        super().__init__(panel, QtCore.Qt.WindowType.Window)
        self.panel = panel
        self.setWindowTitle("Events · timing and duration")
        self.resize(760, 520)
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Filter by recording or behavior")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)
        self.show_removed = QtWidgets.QCheckBox("Show removed")
        self.show_removed.toggled.connect(self.reload)
        self.summary = QtWidgets.QLabel()
        top.addWidget(self.search, 1)
        top.addWidget(self.show_removed)
        top.addWidget(self.summary)
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
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)
        buttons = QtWidgets.QHBoxLayout()
        for text, state in (("Hide from plots and metrics", "hidden"), ("Remove", "removed"),
                            ("Restore", None)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _checked=False, value=state: self._set_selected(value))
            buttons.addWidget(button)
        restore_all = QtWidgets.QPushButton("Restore all")
        restore_all.clicked.connect(self._restore_all)
        buttons.addStretch(1)
        buttons.addWidget(restore_all)
        root.addLayout(buttons)
        hint = QtWidgets.QLabel("Right-click events to hide or remove them. Hidden events stay listed "
                                "(greyed); removed events are listed only with “Show removed”. "
                                "Source files are never changed.")
        hint.setWordWrap(True)
        hint.setProperty("class", "hint")
        root.addWidget(hint)

    # -- data ---------------------------------------------------------------
    def _events(self) -> List[Tuple[str, str, int, float, float, float, Optional[str]]]:
        rows = []
        for key, info in (self.panel._behavior_sources or {}).items():
            labels = list(dict.fromkeys(list((info.get("behaviors") or {}).keys()) +
                                        list((info.get("event_behaviors") or {}).keys())))
            excluded = self.panel._event_exclusions(info)
            for label in labels:
                try:
                    on, off, dur = self.panel._extract_behavior_events_raw(info, label)
                except Exception:
                    continue
                for index, (start, stop, length) in enumerate(zip(on, off, dur), 1):
                    state = excluded.get((str(label), round(float(start), 6)))
                    if not np.isfinite(length) and np.isfinite(stop) and stop > start:
                        length = stop - start
                    rows.append((str(key), str(label), index, float(start),
                                 float(stop) if np.isfinite(stop) and stop > start else np.nan,
                                 float(length), state))
        return rows

    def reload(self, *_args) -> None:
        rows = self._events()
        visible = [row for row in rows if row[6] != "removed" or self.show_removed.isChecked()]
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(visible))
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText)
        for r, (key, label, index, onset, offset, length, state) in enumerate(visible):
            values = [key, label, float(index), onset, offset, length, state or "used"]
            for c, value in enumerate(values):
                if isinstance(value, float):
                    item = _NumericItem(str(int(value)) if c == 2 else _fmt(value))
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, value)
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                else:
                    item = QtWidgets.QTableWidgetItem(str(value))
                if c == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, (key, label, onset))
                if state:
                    item.setForeground(muted)
                    font = item.font()
                    font.setStrikeOut(state == "removed")
                    item.setFont(font)
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        used = sum(1 for row in rows if not row[6])
        hidden = sum(1 for row in rows if row[6] == "hidden")
        removed = sum(1 for row in rows if row[6] == "removed")
        self.summary.setText(f"{used} used · {hidden} hidden · {removed} removed")
        self._apply_filter(self.search.text())

    def _apply_filter(self, text: str) -> None:
        needle = str(text).strip().lower()
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c).text().lower() for c in (0, 1)]
            self.table.setRowHidden(r, bool(needle) and not any(needle in cell for cell in cells))

    # -- actions ------------------------------------------------------------
    def _selected_keys(self) -> List[Tuple[str, str, float]]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        return [self.table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole + 1) for r in rows]

    def _set_selected(self, state: Optional[str]) -> None:
        keys = self._selected_keys()
        if not keys:
            return
        for key, label, onset in keys:
            info = self.panel._behavior_sources.get(key)
            if info is not None:
                self.panel._set_event_exclusion(info, label, onset, state)
        self.reload()
        self.panel._on_event_exclusions_changed()

    def _restore_all(self) -> None:
        for info in (self.panel._behavior_sources or {}).values():
            report = info.get("import_report")
            if isinstance(report, dict):
                report.pop("event_exclusions", None)
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
