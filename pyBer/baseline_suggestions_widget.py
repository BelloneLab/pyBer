"""Compact, automatically refreshed baseline choices with a bounded worker queue."""
from __future__ import annotations

from collections import OrderedDict
import hashlib

import numpy as np
from PySide6 import QtCore, QtWidgets

from baseline_suggestions import suggest_baselines


_ACTIVE_JOBS = set()


class _Signals(QtCore.QObject):
    """Keep worker lifetime independent of the parent settings panel."""

    result = QtCore.Signal(str, object)
    error = QtCore.Signal(str, str)
    finished = QtCore.Signal()

    @QtCore.Slot()
    def release(self):
        """Dispose of worker references on the GUI thread after queued delivery."""
        _ACTIVE_JOBS.discard(self.job)
        self.job = None
        self.deleteLater()


class _Job(QtCore.QRunnable):
    """One detached signal snapshot evaluated outside the GUI event loop."""

    def __init__(self, key, recordings, config):
        super().__init__()
        self.setAutoDelete(False)
        self.key, self.recordings, self.config = key, recordings, config
        self.signals = _Signals()
        self.signals.job = self
        self.signals.finished.connect(self.signals.release)

    def run(self):
        """Deliver results or a short recoverable failure, never touching widgets."""
        try:
            self.signals.result.emit(self.key, suggest_baselines(self.recordings, self.config))
        except Exception as exc:
            self.signals.error.emit(self.key, str(exc))
        finally:
            self.signals.finished.emit()


def _snapshot_key(recordings, config):
    """Cache the actual immutable inputs, so baseline edits do not repeat analysis."""
    digest = hashlib.blake2b(digest_size=20)
    digest.update(repr(config).encode("utf8"))
    for recording in recordings:
        digest.update(recording.label.encode("utf8"))
        for array in (recording.time, recording.signal, recording.events, recording.exclusion_intervals):
            values = np.ascontiguousarray(array if array is not None else [], dtype=np.float64)
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
    return digest.hexdigest()


class BaselineSuggestionsWidget(QtWidgets.QWidget):
    """Show three scored windows inline; only an explicit click changes a baseline."""

    selected = QtCore.Signal(object)

    def __init__(self, snapshot, config, parent=None):
        super().__init__(parent)
        self._snapshot, self._config = snapshot, config
        self._cache = OrderedDict()
        self._wanted_key = None
        self._job = None
        self._pending = None
        self._closed = False
        self.report = None
        self.scope = ""
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(240)
        self.timer.timeout.connect(self._refresh)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(4)
        self.title = QtWidgets.QLabel("Suggested baseline")
        self.title.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.title)
        self.buttons = []
        for index in range(3):
            button = QtWidgets.QPushButton()
            button.setMaximumHeight(31)
            button.setMinimumHeight(27)
            button.setStyleSheet("QPushButton { padding: 2px 8px; font-size: 12px; }")
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            button.setVisible(False)
            button.clicked.connect(lambda _checked=False, number=index: self._apply(number))
            self.buttons.append(button)
            layout.addWidget(button)
        self.status = QtWidgets.QLabel("Select a signal and events.")
        self.status.setProperty("class", "hint")
        self.status.setWordWrap(True)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.status)

    def queue(self):
        """Debounce rapid edits and disable stale choices before the next snapshot."""
        if self._closed:
            return
        self._wanted_key = None
        self._pending = None
        for button in self.buttons:
            button.setEnabled(False)
        self.timer.start()

    @QtCore.Slot()
    def _refresh(self):
        """Use cached inputs or retain only the newest pending background request."""
        if self._closed:
            return
        try:
            recordings, scope = self._snapshot()
            config = self._config()
            key = _snapshot_key(recordings, config)
        except Exception as exc:
            self._wanted_key = None
            self._pending = None
            self._show_unavailable(f"Suggestions unavailable: {exc}")
            return
        self.scope = scope
        self.title.setToolTip(scope)
        self._wanted_key = key
        if key in self._cache:
            self._cache.move_to_end(key)
            self._show_report(self._cache[key])
            self._pending = None
            return
        if not recordings:
            self._show_unavailable("Select a signal and events.")
            self._pending = None
            return
        payload = (key, recordings, config)
        self.status.setText("Updating suggestions...")
        if self._job is not None:
            self._pending = payload if self._job.key != key else None
            return
        self._start(payload)

    def _start(self, payload):
        """Allow only one numerical job at once per panel, even during rapid edits."""
        job = _Job(*payload)
        self._job = job
        job.signals.result.connect(self._result)
        job.signals.error.connect(self._error)
        _ACTIVE_JOBS.add(job)
        QtCore.QThreadPool.globalInstance().start(job)

    @QtCore.Slot(str, object)
    def _result(self, key, report):
        """A superseded result is cached but can never replace current suggestions."""
        if self._closed:
            return
        self._job = None
        self._cache[key] = report
        self._cache.move_to_end(key)
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        if key == self._wanted_key and not self.timer.isActive():
            self._show_report(report)
        self._start_pending()

    @QtCore.Slot(str, str)
    def _error(self, key, message):
        """Surface current errors without opening modal interruptions."""
        if self._closed:
            return
        self._job = None
        if key == self._wanted_key and not self.timer.isActive():
            self._show_unavailable(f"Suggestions unavailable: {message}")
        self._start_pending()

    def _start_pending(self):
        """Discard intermediate requests and resume only the newest complete snapshot."""
        pending, self._pending = self._pending, None
        if (pending is not None and pending[0] == self._wanted_key and
                not self.timer.isActive() and not self._closed):
            if pending[0] in self._cache:
                self._show_report(self._cache[pending[0]])
            else:
                self._start(pending)

    def _show_unavailable(self, message):
        """Empty or unusable signals produce a compact status, not made-up choices."""
        self.report = None
        for button in self.buttons:
            button.setVisible(False)
            button.setEnabled(False)
        self.status.setText(message)
        self.status.setToolTip(message)

    def _show_report(self, report):
        """Present fit scores with explicit limitations in every choice tooltip."""
        self.report = report
        choices = report.get("choices", [])
        for index, button in enumerate(self.buttons):
            visible = index < len(choices)
            button.setVisible(visible)
            button.setEnabled(visible)
            if not visible:
                continue
            choice = choices[index]
            button.setText(f"{choice['start']:.2f} to {choice['end']:.2f} s  ·  {choice['score']}%  ·  Apply")
            diagnostic = choice.get("diagnostics", {})
            tooltip = (
                f"{choice['quality']} fit. Score {choice['score']}/100 is a descriptive ranking, not confidence.\n"
                f"{choice['summary']}\n"
                f"Complete data: {100 * diagnostic.get('observed_coverage', 0):.0f}% minimum across files. "
                f"Event-free windows: {100 * diagnostic.get('event_free_fraction', 0):.0f}% minimum.\n"
                f"Estimated effective samples: {diagnostic.get('effective_samples', 0):.1f}. "
                f"Quality samples at most {report.get('config', {}).get('max_sampled_events', 32)} events per file.\n"
                "Click to change only the baseline window. Event selections are preserved; usable PSTH trial counts may change.\n"
                + self.scope
            )
            button.setToolTip(tooltip)
            button.setAccessibleName(f"Apply baseline {choice['start']:.2f} to {choice['end']:.2f} seconds, {choice['score']} percent fit score, {choice['quality'].lower()}")
        if not choices:
            self.status.setText(report.get("summary", "No usable pre-event baseline."))
        else:
            best = choices[0]
            if best["quality"] == "Limited":
                cautions = best.get("diagnostics", {}).get("cautions", [])
                short = cautions[0].split(":")[0].lower() if cautions else "limited signal information"
                self.status.setText(f"Limited: {short}. Hover for details.")
            else:
                self.status.setText("Fit score, not confidence. Click to apply.")
        self.status.setToolTip(report.get("summary", ""))

    def _apply(self, index):
        """Only enabled current choices can be applied, never an outdated result."""
        choices = (self.report or {}).get("choices", [])
        if 0 <= index < len(choices) and self.buttons[index].isEnabled():
            self.selected.emit(tuple(choices[index]["window"]))

    def stop(self):
        """Panel closure drops callbacks while Qt safely owns any finishing work."""
        self._closed = True
        self.timer.stop()
        self._pending = None
        if self._job is not None:
            for signal, callback in ((self._job.signals.result, self._result), (self._job.signals.error, self._error)):
                try:
                    signal.disconnect(callback)
                except (RuntimeError, TypeError):
                    pass
            self._job = None

    def resume(self):
        """A reopened settings panel can resume after an ordinary close event."""
        self._closed = False
        self.queue()
