"""Compact "Auto" baseline button: scored windows computed on request.

Nothing is computed or changed until the user clicks the button. The click
evaluates the current signal and events in a background worker, then opens a
small menu of scored windows; choosing one applies it. Manual entries in the
baseline fields are never touched.
"""
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
    """One compact button; only an explicit menu selection changes a baseline."""

    selected = QtCore.Signal(object)
    #: Short progress / outcome text for the status bar of the application.
    message = QtCore.Signal(str)

    BUTTON_TEXT = "Auto"
    BUSY_TEXT = "…"
    HELP = ("Suggest a baseline window from the signal and the selected events.\n"
            "Click to compute and choose a window. Nothing changes until you pick one.")

    def __init__(self, snapshot, config, parent=None):
        super().__init__(parent)
        self._snapshot, self._config = snapshot, config
        self._cache = OrderedDict()
        self._wanted_key = None
        self._job = None
        self._pending = None
        self._closed = False
        # True between a button click and the display of its result.
        self._popup_when_ready = False
        self.report = None
        self.scope = ""
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self._refresh)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.menu_button = QtWidgets.QPushButton(self.BUTTON_TEXT)
        self.menu_button.setProperty("class", "compactSmall")
        self.menu_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.menu_button.setAccessibleName("Suggest a baseline window")
        self.menu_button.setToolTip(self.HELP)
        self.menu_button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.menu_button.clicked.connect(self.request_menu)
        self.menu = QtWidgets.QMenu(self.menu_button)
        self.menu.setToolTipsVisible(True)
        self.actions = []
        for index in range(3):
            action = self.menu.addAction("")
            action.setVisible(False)
            action.setEnabled(False)
            action.triggered.connect(lambda _checked=False, number=index: self._apply(number))
            self.actions.append(action)
        layout.addWidget(self.menu_button)
        # Outcome text (mirrored in the button tooltip). Kept as a label for
        # callers that read it, but not shown: the control is one button.
        self.status = QtWidgets.QLabel("Select a signal and events.", self)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.status.hide()

    # --------------------------------------------------------------- requests
    def request_menu(self):
        """Button click: compute for the current inputs, then open the choices."""
        if self._closed:
            return
        self._popup_when_ready = True
        self.menu_button.setText(self.BUSY_TEXT)
        self.menu_button.setEnabled(False)
        self.message.emit("Evaluating baseline windows…")
        self.queue()

    def invalidate(self):
        """Inputs changed: drop the displayed choices without computing anything."""
        if self._closed:
            return
        if self._popup_when_ready:
            # The user asked for suggestions and is waiting: a recompute that
            # lands meanwhile must not cancel the request silently. Evaluate
            # the newest inputs instead.
            self.queue()
            return
        self.timer.stop()
        self._wanted_key = None
        self._pending = None
        self.report = None
        self.menu.close()
        for action in self.actions:
            action.setEnabled(False)
        self._idle()

    def _idle(self):
        """Return the button to its clickable resting state."""
        self.menu_button.setText(self.BUTTON_TEXT)
        self.menu_button.setEnabled(True)

    def queue(self):
        """Evaluate the current snapshot (cached inputs are not recomputed)."""
        if self._closed:
            return
        self._wanted_key = None
        self._pending = None
        self.menu.close()
        for button in self.actions:
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
        self._set_status("Updating suggestions...")
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

    # ---------------------------------------------------------------- display
    def _set_status(self, text, tooltip=None):
        """Keep the outcome readable from the button tooltip."""
        self.status.setText(text)
        self.status.setToolTip(tooltip if tooltip is not None else text)
        self.menu_button.setToolTip(self.HELP + "\n\n" + text)

    def _show_unavailable(self, message):
        """Empty or unusable signals produce a short status, not made-up choices."""
        self.report = None
        for button in self.actions:
            button.setVisible(False)
            button.setEnabled(False)
        self._set_status(message)
        self._idle()
        if self._popup_when_ready:
            self._popup_when_ready = False
            self.message.emit(message)

    def _show_report(self, report):
        """Present fit scores with explicit limitations in every choice tooltip."""
        self.report = report
        choices = report.get("choices", [])
        for index, button in enumerate(self.actions):
            visible = index < len(choices)
            button.setVisible(visible)
            button.setEnabled(visible)
            if not visible:
                continue
            choice = choices[index]
            button.setText(f"{choice['start']:.0f} to {choice['end']:.0f} s  ·  {choice['score']}%  ·  Apply")
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
        if not choices:
            text = report.get("summary", "No usable pre-event baseline.")
        else:
            best = choices[0]
            if best["quality"] == "Limited":
                cautions = best.get("diagnostics", {}).get("cautions", [])
                short = cautions[0].split(":")[0].lower() if cautions else "limited signal information"
                text = f"Limited: {short}. Hover a choice for details."
            else:
                text = "Fit score, not confidence. Select a window to apply."
        self._set_status(text, report.get("summary", ""))
        self._idle()
        if self._popup_when_ready:
            self._popup_when_ready = False
            self.message.emit(text if choices else f"No suggested baseline window: {text}")
            if choices and self.menu_button.isVisible():
                self.menu.popup(self.menu_button.mapToGlobal(QtCore.QPoint(0, self.menu_button.height())))

    def _apply(self, index):
        """Only enabled current choices can be applied, never an outdated result."""
        choices = (self.report or {}).get("choices", [])
        if 0 <= index < len(choices) and self.actions[index].isEnabled():
            self.selected.emit(tuple(choices[index]["window"]))

    # -------------------------------------------------------------- lifecycle
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
        """A reopened settings panel starts idle: suggestions are computed on request only."""
        self._closed = False
        self.invalidate()
