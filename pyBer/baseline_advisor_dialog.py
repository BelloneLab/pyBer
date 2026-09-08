"""Optional, asynchronous review of event-relative baseline recommendations.

The dialog never changes analysis settings itself. Its owner applies the returned
window only after the user chooses Apply. Worker inputs are detached snapshots,
so closing the dialog or continuing to use the application cannot mutate them.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from PySide6 import QtCore, QtWidgets

from baseline_advisor import (
    BaselineAdvisorConfig, BaselineRecording, export_baseline_report,
    recommend_baseline,
)


# Retain jobs until their queued completion has reached the GUI thread. A closed
# dialog may discard its callbacks while the bounded numerical work finishes.
_ACTIVE_JOBS: set = set()


class _WorkerSignals(QtCore.QObject):
    """Deliver numerical results across threads without touching GUI widgets."""

    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    @QtCore.Slot()
    def release_job(self) -> None:
        """Release the Python job on the owning GUI thread after completion."""
        _ACTIVE_JOBS.discard(self.job)
        self.job = None
        self.deleteLater()


class _RecommendationJob(QtCore.QRunnable):
    """Run one immutable recommendation request in Qt's managed thread pool."""

    def __init__(self, recordings, config, current_window):
        super().__init__()
        self.setAutoDelete(False)
        self.recordings = recordings
        self.config = config
        self.current_window = current_window
        self.signals = _WorkerSignals()
        self.signals.job = self
        self.signals.finished.connect(self.signals.release_job)

    def run(self) -> None:
        """Convert computation failures into a recoverable dialog message."""
        try:
            result = recommend_baseline(
                self.recordings, config=self.config,
                current_window=self.current_window,
            )
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


def _display_value(value: Any) -> str:
    """Keep scalar diagnostics compact without hiding their numerical precision."""
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.5g}"
    return str(value)


def _populate_tree(parent, key: str, value: Any) -> None:
    """Show nested report fields with readable labels and inspectable children."""
    item = QtWidgets.QTreeWidgetItem(parent, [key.replace("_", " ").capitalize(), ""])
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _populate_tree(item, str(child_key), child_value)
    elif isinstance(value, (list, tuple)):
        if all(not isinstance(child, (dict, list, tuple)) for child in value):
            item.setText(1, ", ".join(_display_value(child) for child in value))
        else:
            for index, child in enumerate(value):
                name = str(child.get("label", child.get("recording", index + 1))) if isinstance(child, dict) else str(index + 1)
                _populate_tree(item, name, child)
    else:
        item.setText(1, _display_value(value))


class BaselineAdvisorDialog(QtWidgets.QDialog):
    """Review candidate quality, export diagnostics, and explicitly accept a window."""

    def __init__(
        self, recordings: Sequence[BaselineRecording], *,
        current_window: tuple[float, float], scope: str, parent=None,
        auto_start: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Recommend PSTH baseline")
        self.resize(960, 740)
        self.setMinimumSize(700, 540)
        self.recordings = list(recordings)
        self.current_window = tuple(current_window)
        self.report: dict | None = None
        self.proposed_window: tuple[float, float] | None = None
        self._job = None
        self._closed = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title = QtWidgets.QLabel("Find a stable baseline before the event")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)
        context = QtWidgets.QLabel(scope)
        context.setWordWrap(True)
        context.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        context.setMaximumHeight(64)
        context.setToolTip(scope)
        layout.addWidget(context)

        controls = QtWidgets.QGridLayout()
        self.lookback = self._spin(1, 60, 30)
        self.guard = self._spin(0, 30, 0.25)
        self.recovery = self._spin(0, 60, 1)
        self.minimum = self._spin(0.1, 60, 1)
        self.lookback.setToolTip("Search this far before each event, independently of the displayed PSTH window.")
        self.guard.setToolTip("Keep this much time before every known event out of candidate baselines.")
        self.recovery.setToolTip("Exclude each complete event bout and this additional recovery period afterward. Set this from your indicator and protocol.")
        self.minimum.setToolTip("Shortest candidate to consider. Passing this duration alone does not establish baseline reliability.")
        for column, (label, control) in enumerate((
            ("Search before event", self.lookback), ("Pre-event guard", self.guard),
            ("Post-bout recovery", self.recovery), ("Minimum duration", self.minimum),
        )):
            controls.addWidget(QtWidgets.QLabel(label), 0, column)
            controls.addWidget(control, 1, column)
        layout.addLayout(controls)
        explanation = QtWidgets.QLabel(
            "Uses pre-event signal quality, temporal correlation and event spacing. "
            "Checks the selected candidate on held-out events. A recommendation is "
            "not statistical validation or proof of a biologically neutral baseline. "
            "Post-event responses and significance are never selection targets."
        )
        explanation.setWordWrap(True)
        explanation.setProperty("class", "hint")
        layout.addWidget(explanation)

        row = QtWidgets.QHBoxLayout()
        self.current_label = QtWidgets.QLabel(
            f"Current: {current_window[0]:.2f} to {current_window[1]:.2f} s"
        )
        self.proposal_label = QtWidgets.QLabel("Suggested: pending")
        self.proposal_label.setStyleSheet("font-weight: 600;")
        row.addWidget(self.current_label)
        row.addStretch(1)
        row.addWidget(self.proposal_label)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("Ready to assess the selected recordings.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)

        tabs = QtWidgets.QTabWidget()
        candidate_page = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_page)
        candidate_layout.setContentsMargins(0, 6, 0, 0)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.candidates = QtWidgets.QTableWidget(0, 4)
        self.candidates.setHorizontalHeaderLabels(["Window (s)", "Duration (s)", "Search score", "Search checks"])
        self.candidates.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidates.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidates.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.candidates.verticalHeader().hide()
        self.candidates.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.candidates.itemSelectionChanged.connect(self._show_candidate)
        self.candidates.setToolTip("Candidate scores rank the search only. They are not probabilities, confidence levels or a test of an event response.")
        self.details = self._tree()
        splitter.addWidget(self.candidates)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        candidate_layout.addWidget(splitter)
        tabs.addTab(candidate_page, "Candidates and diagnostics")
        self.recording_details = self._tree()
        tabs.addTab(self.recording_details, "Recordings and current baseline")
        layout.addWidget(tabs, 1)

        actions = QtWidgets.QHBoxLayout()
        self.refresh_button = QtWidgets.QPushButton("Estimate baseline")
        self.refresh_button.clicked.connect(self._start)
        self.export_button = QtWidgets.QPushButton("Export report")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.export_button)
        actions.addStretch(1)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.reject)
        self.apply_button = QtWidgets.QPushButton("Apply suggested window")
        self.apply_button.setEnabled(False)
        self.apply_button.setDefault(False)
        self.apply_button.setToolTip(
            "Event selections are preserved. Usable PSTH trial counts may change "
            "when baseline validity changes. This action can be undone."
        )
        self.apply_button.clicked.connect(self._apply)
        actions.addWidget(close_button)
        actions.addWidget(self.apply_button)
        layout.addLayout(actions)
        for control in (self.lookback, self.guard, self.recovery, self.minimum):
            control.valueChanged.connect(self._settings_changed)
        if auto_start:
            QtCore.QTimer.singleShot(0, self._start)

    @staticmethod
    def _spin(low: float, high: float, value: float) -> QtWidgets.QDoubleSpinBox:
        """Offer keyboard-first duration entry with ordinary incremental stepping."""
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setSingleStep(0.25)
        spin.setSuffix(" s")
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _tree() -> QtWidgets.QTreeWidget:
        """Create a consistent, selectable diagnostic inspection panel."""
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["Diagnostic", "Value"])
        tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setStretchLastSection(True)
        tree.setAlternatingRowColors(True)
        return tree

    @QtCore.Slot()
    def _settings_changed(self) -> None:
        """An earlier recommendation cannot be applied after its settings change."""
        self.proposed_window = None
        self.apply_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.proposal_label.setText("Suggested: settings changed")
        self.status.setText("Estimate again to assess these settings.")

    @QtCore.Slot()
    def _start(self) -> None:
        """Schedule CPU work while leaving painting, closing and navigation responsive."""
        if self._closed or self._job is not None:
            return
        if self.minimum.value() + self.guard.value() > self.lookback.value():
            self.status.setText("Search duration must include the minimum baseline duration and pre-event guard.")
            return
        self.proposed_window = None
        self.report = None
        self.apply_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        for control in (self.lookback, self.guard, self.recovery, self.minimum):
            control.setEnabled(False)
        self.proposal_label.setText("Suggested: assessing")
        self.status.setText("Assessing baseline candidates and held-out events...")
        config = BaselineAdvisorConfig(
            lookback_s=self.lookback.value(), event_guard_s=self.guard.value(),
            recovery_s=self.recovery.value(), min_window_s=self.minimum.value(),
        )
        # A malformed current baseline should not prevent the advisor from
        # finding a replacement. It simply has no current-window comparison.
        current = self.current_window
        if not all(math.isfinite(value) for value in current) or current[0] >= current[1]:
            current = None
        job = _RecommendationJob(self.recordings, config, current)
        self._job = job
        job.signals.result.connect(self._receive_result)
        job.signals.error.connect(self._receive_error)
        _ACTIVE_JOBS.add(job)
        QtCore.QThreadPool.globalInstance().start(job)

    def _finish(self) -> None:
        """Restore editable controls after success or recoverable failure."""
        self._job = None
        self.refresh_button.setEnabled(True)
        for control in (self.lookback, self.guard, self.recovery, self.minimum):
            control.setEnabled(True)

    @QtCore.Slot(object)
    def _receive_result(self, report: dict) -> None:
        """Render a completed report without changing the owner's analysis."""
        if self._closed:
            return
        self._finish()
        self.report = report
        window = report.get("window")
        if report.get("status") == "recommended" and window and len(window) == 2:
            self.proposed_window = tuple(map(float, window))
            self.proposal_label.setText(f"Suggested: {window[0]:.2f} to {window[1]:.2f} s")
            self.apply_button.setEnabled(True)
        else:
            self.proposed_window = None
            self.proposal_label.setText("Suggested: no reliable window")
            self.apply_button.setEnabled(False)
        summary = str(report.get("summary", "Assessment completed."))
        reasons = [str(reason) for reason in report.get("reasons", [])]
        visible_reasons = " ".join(reasons[:3])
        if len(reasons) > 3:
            visible_reasons += f" {len(reasons) - 3} further reasons are in the report details."
        self.status.setText(summary + ("\n" + visible_reasons if reasons else ""))
        self.status.setToolTip(summary + "\n" + "\n".join(reasons))
        self.export_button.setEnabled(True)
        candidates = report.get("candidates", [])
        self.candidates.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            start, end = candidate.get("start", 0), candidate.get("end", 0)
            values = [f"{start:.2f} to {end:.2f}", f"{end - start:.2f}",
                      _display_value(candidate.get("score")),
                      "Passed" if candidate.get("eligible") else "Not passed"]
            for col, value in enumerate(values):
                self.candidates.setItem(row, col, QtWidgets.QTableWidgetItem(value))
        self.recording_details.clear()
        _populate_tree(self.recording_details, "Assessment reasons", reasons)
        _populate_tree(self.recording_details, "Current baseline", report.get("current"))
        _populate_tree(self.recording_details, "Recordings", report.get("recordings", []))
        _populate_tree(self.recording_details, "Assessment settings", report.get("config", {}))
        self.recording_details.expandToDepth(1)
        if candidates:
            self.candidates.selectRow(0)
        else:
            self.details.clear()

    @QtCore.Slot(str)
    def _receive_error(self, message: str) -> None:
        """Keep the dialog usable when an input cannot be assessed."""
        if self._closed:
            return
        self._finish()
        self.proposal_label.setText("Suggested: unavailable")
        self.status.setText(f"Unable to assess this input: {message}")

    @QtCore.Slot()
    def _show_candidate(self) -> None:
        """Selecting a row reveals diagnostics; it cannot replace the recommendation."""
        self.details.clear()
        rows = (self.report or {}).get("candidates", [])
        index = self.candidates.currentRow()
        if 0 <= index < len(rows):
            for name, value in rows[index].items():
                _populate_tree(self.details, name, value)
            self.details.expandToDepth(1)

    @QtCore.Slot()
    def _apply(self) -> None:
        """Return an explicit acceptance only for an available, current proposal."""
        if self.proposed_window is not None and self.apply_button.isEnabled():
            self.accept()

    @QtCore.Slot()
    def _export(self) -> None:
        """Save both machine-readable settings and a tabular candidate audit."""
        if self.report is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export baseline recommendation", "baseline_recommendation.json", "JSON report (*.json)"
        )
        if not path:
            return
        try:
            prefix = str(Path(path).with_suffix(""))
            paths = export_baseline_report(self.report, prefix)
            self.status.setText("Saved report: " + ", ".join(map(str, paths)))
        except Exception as exc:
            self.status.setText(f"Could not export the report: {exc}")

    def done(self, result: int) -> None:
        """Closing never waits for computation or destroys a running Qt thread."""
        self._closed = True
        if self._job is not None:
            for signal, callback in ((self._job.signals.result, self._receive_result),
                                     (self._job.signals.error, self._receive_error)):
                try:
                    signal.disconnect(callback)
                except (RuntimeError, TypeError):
                    pass
            self._job = None
        super().done(result)
