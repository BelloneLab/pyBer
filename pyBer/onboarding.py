# onboarding.py
"""
User-experience polish layer for pyBer:

* TutorialOverlay  - first-run guided walkthrough with highlight cutout, callouts,
                     Next/Back/Skip controls. Reopen via F1 / Help menu.
* ToastManager     - lightweight non-blocking notifications (info/warn/error)
                     stacked top-right, click to dismiss.
* PreferencesDialog- consolidates theme, autosave, default kernel window, behavior
                     defaults, and shows the keyboard cheat sheet.
* register_shortcuts(window) - installs a wide set of global keyboard shortcuts.
* attach_dirty_title(window) - shows "*" in window title while project is dirty.

This module is intentionally self-contained: it touches no analysis code
and never raises if the host window lacks an optional attribute.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


# ============================================================================
# Toast notifications
# ============================================================================

_TOAST_QSS = {
    "info":  "background: #1f2a3a; color: #e9f0fb; border: 1px solid #355080;",
    "warn":  "background: #3a2d1d; color: #fde6c8; border: 1px solid #8a6a3a;",
    "error": "background: #3b1f25; color: #ffd6dc; border: 1px solid #8a3949;",
    "ok":    "background: #1c2e22; color: #d4f4dc; border: 1px solid #2f7a4a;",
}


class _Toast(QtWidgets.QFrame):
    closed = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget, text: str, severity: str, timeout_ms: int):
        super().__init__(parent)
        self.setObjectName("pyberToast")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setStyleSheet(
            "QFrame#pyberToast { %s border-radius: 8px; padding: 8px 12px; } "
            "QFrame#pyberToast QLabel { background: transparent; }"
            % _TOAST_QSS.get(severity, _TOAST_QSS["info"])
        )
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(8)
        icon = {"info": "i", "warn": "!", "error": "x", "ok": "v"}.get(severity, "i")
        badge = QtWidgets.QLabel(icon)
        badge.setFixedSize(20, 20)
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet("background: rgba(255,255,255,0.08); border-radius: 10px; font-weight: 700;")
        lay.addWidget(badge)
        self._label = QtWidgets.QLabel(str(text))
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(360)
        lay.addWidget(self._label, 1)
        self.setMinimumWidth(220)
        self.setMaximumWidth(420)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if timeout_ms > 0:
            QtCore.QTimer.singleShot(int(timeout_ms), self._dismiss)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._dismiss()
        super().mousePressEvent(event)

    def _dismiss(self) -> None:
        try:
            self.closed.emit(self)
        finally:
            self.close()


class ToastManager(QtCore.QObject):
    """
    Stacked top-right toast queue that follows the host window. Up to
    `max_visible` toasts; older ones drop off when more arrive.
    """

    def __init__(self, window: QtWidgets.QMainWindow, max_visible: int = 4):
        super().__init__(window)
        self._window = window
        self._max_visible = int(max_visible)
        self._toasts: List[_Toast] = []
        self._margin = 14
        self._spacing = 8
        window.installEventFilter(self)

    def post(self, text: str, severity: str = "info", timeout_ms: int = 5000) -> None:
        if not text:
            return
        toast = _Toast(self._window, text, severity, timeout_ms)
        toast.closed.connect(self._on_closed)
        self._toasts.append(toast)
        # Cap visible count.
        while len(self._toasts) > self._max_visible:
            old = self._toasts.pop(0)
            old.close()
        toast.show()
        toast.adjustSize()
        self._reflow()

    # Convenience wrappers.
    def info(self, text: str, timeout_ms: int = 4000) -> None:
        self.post(text, "info", timeout_ms)

    def ok(self, text: str, timeout_ms: int = 3500) -> None:
        self.post(text, "ok", timeout_ms)

    def warn(self, text: str, timeout_ms: int = 6500) -> None:
        self.post(text, "warn", timeout_ms)

    def error(self, text: str, timeout_ms: int = 9000) -> None:
        self.post(text, "error", timeout_ms)

    def _on_closed(self, toast: _Toast) -> None:
        try:
            self._toasts.remove(toast)
        except ValueError:
            pass
        self._reflow()

    def _reflow(self) -> None:
        if not self._window.isVisible():
            return
        rect = self._window.rect()
        x_right = rect.right() - self._margin
        y = rect.top() + self._margin
        # Status bar/toolbar offset
        try:
            mw = self._window
            if isinstance(mw, QtWidgets.QMainWindow) and mw.menuBar() is not None:
                y += mw.menuBar().height() + 4
        except Exception:
            pass
        for toast in self._toasts:
            toast.adjustSize()
            w = toast.width()
            toast.move(x_right - w, y)
            toast.raise_()
            y += toast.height() + self._spacing

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self._reflow()
        return False


# ============================================================================
# Tutorial overlay
# ============================================================================


class TutorialStep:
    """One step of the guided tutorial."""

    def __init__(
        self,
        title: str,
        body: str,
        target_resolver: Optional[Callable[[QtWidgets.QWidget], Optional[QtWidgets.QWidget]]] = None,
        before: Optional[Callable[[QtWidgets.QWidget], None]] = None,
    ):
        self.title = title
        self.body = body
        self.target_resolver = target_resolver
        self.before = before  # called before showing the step (e.g. switch tab)


class TutorialOverlay(QtWidgets.QWidget):
    """
    Full-window translucent overlay. A "spotlight" cutout illuminates the
    target widget, and a styled callout near it shows step text and
    Back / Next / Skip controls. Esc skips. Arrow keys page through.
    """

    finished = QtCore.Signal()

    def __init__(self, host: QtWidgets.QMainWindow, steps: List[TutorialStep]):
        super().__init__(host)
        self._host = host
        self._steps = list(steps)
        self._index = 0
        self._dont_show_again = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._target_rect = QtCore.QRect()
        self._build_callout()
        host.installEventFilter(self)

    # --- internal UI ---

    def _build_callout(self) -> None:
        self._callout = QtWidgets.QFrame(self)
        self._callout.setObjectName("tutorialCallout")
        self._callout.setStyleSheet(
            "QFrame#tutorialCallout { background: #14202f; color: #e9f0fb; "
            "border: 1px solid #2f8cff; border-radius: 12px; }"
            "QFrame#tutorialCallout QLabel { background: transparent; color: #e9f0fb; }"
            "QFrame#tutorialCallout QLabel#tutTitle { font-weight: 700; font-size: 12pt; }"
            "QFrame#tutorialCallout QLabel#tutStep { color: #95a5c2; font-size: 8.5pt; }"
            "QFrame#tutorialCallout QPushButton { background: #1c2a3e; color: #e9f0fb; "
            "border: 1px solid #355080; border-radius: 6px; padding: 5px 12px; }"
            "QFrame#tutorialCallout QPushButton:hover { background: #233553; }"
            "QFrame#tutorialCallout QPushButton#tutPrimary { background: #2f8cff; "
            "border: 1px solid #46a0ff; color: white; font-weight: 700; }"
            "QFrame#tutorialCallout QPushButton#tutPrimary:hover { background: #46a0ff; }"
            "QFrame#tutorialCallout QPushButton#tutSkip { color: #95a5c2; border-color: transparent; }"
        )
        lay = QtWidgets.QVBoxLayout(self._callout)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(8)

        self._lbl_step = QtWidgets.QLabel("Step 1 / N")
        self._lbl_step.setObjectName("tutStep")
        lay.addWidget(self._lbl_step)
        self._lbl_title = QtWidgets.QLabel("Title")
        self._lbl_title.setObjectName("tutTitle")
        self._lbl_title.setWordWrap(True)
        lay.addWidget(self._lbl_title)
        self._lbl_body = QtWidgets.QLabel("Body")
        self._lbl_body.setWordWrap(True)
        self._lbl_body.setMinimumWidth(340)
        self._lbl_body.setMaximumWidth(420)
        lay.addWidget(self._lbl_body)

        self._chk_dont_show = QtWidgets.QCheckBox("Don't show this tutorial automatically again")
        self._chk_dont_show.setToolTip("You can always replay it later with F1.")
        self._chk_dont_show.setStyleSheet(
            "QCheckBox { background: transparent; color: #c5d2e3; spacing: 8px; }"
            "QCheckBox::indicator { width: 15px; height: 15px; }"
        )
        lay.addWidget(self._chk_dont_show)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_skip = QtWidgets.QPushButton("Skip tutorial")
        self._btn_skip.setObjectName("tutSkip")
        self._btn_skip.clicked.connect(self._end)
        btn_row.addWidget(self._btn_skip)
        btn_row.addStretch(1)
        self._btn_back = QtWidgets.QPushButton("◀ Back")
        self._btn_back.setText("< Back")
        self._btn_back.clicked.connect(self._prev)
        btn_row.addWidget(self._btn_back)
        self._btn_next = QtWidgets.QPushButton("Next ▶")
        self._btn_next.setText("Next >")
        self._btn_next.setObjectName("tutPrimary")
        self._btn_next.setDefault(True)
        self._btn_next.clicked.connect(self._next)
        btn_row.addWidget(self._btn_next)
        lay.addLayout(btn_row)
        self._callout.adjustSize()

    # --- lifecycle ---

    def start(self) -> None:
        self.setGeometry(self._host.rect())
        self.show()
        self.raise_()
        self.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
        self._render_step()

    def _end(self) -> None:
        try:
            self._dont_show_again = bool(self._chk_dont_show.isChecked())
        except Exception:
            self._dont_show_again = False
        self._host.removeEventFilter(self)
        self.finished.emit()
        self.close()
        self.deleteLater()

    def dont_show_again(self) -> bool:
        return bool(self._dont_show_again)

    def _next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._end()
            return
        self._index += 1
        self._render_step()

    def _prev(self) -> None:
        if self._index <= 0:
            return
        self._index -= 1
        self._render_step()

    # --- rendering ---

    def _render_step(self) -> None:
        step = self._steps[self._index]
        if step.before is not None:
            try:
                step.before(self._host)
            except Exception:
                pass
        n = len(self._steps)
        self._lbl_step.setText(f"Step {self._index + 1} / {n}")
        self._lbl_title.setText(step.title)
        self._lbl_body.setText(step.body)
        self._btn_back.setEnabled(self._index > 0)
        self._btn_next.setText("Got it!" if self._index == n - 1 else "Next ▶")

        if self._index < n - 1:
            self._btn_next.setText("Next >")
        target = None
        if step.target_resolver is not None:
            try:
                target = step.target_resolver(self._host)
            except Exception:
                target = None
        self._target_rect = self._compute_target_rect(target)
        self._position_callout()
        self.update()

    def _compute_target_rect(self, target: Optional[QtWidgets.QWidget]) -> QtCore.QRect:
        if target is None or not target.isVisible():
            return QtCore.QRect()
        # Map the target widget rect into the overlay's coordinate space.
        top_left = target.mapTo(self._host, QtCore.QPoint(0, 0))
        rect = QtCore.QRect(top_left, target.size())
        rect = rect.adjusted(-6, -6, 6, 6)
        return rect

    def _position_callout(self) -> None:
        margin = 16
        host_rect = self.rect()
        self._callout.adjustSize()
        co_w = self._callout.width()
        co_h = self._callout.height()
        if self._target_rect.isEmpty():
            x = (host_rect.width() - co_w) // 2
            y = (host_rect.height() - co_h) // 2
            self._callout.move(x, y)
            return
        # Try to place to the right of the target, fallback below, then left, then above.
        candidates = [
            QtCore.QPoint(self._target_rect.right() + margin, self._target_rect.top()),
            QtCore.QPoint(self._target_rect.left(), self._target_rect.bottom() + margin),
            QtCore.QPoint(self._target_rect.left() - co_w - margin, self._target_rect.top()),
            QtCore.QPoint(self._target_rect.left(), self._target_rect.top() - co_h - margin),
        ]
        chosen = candidates[0]
        for cand in candidates:
            if (
                cand.x() >= margin and cand.y() >= margin
                and cand.x() + co_w + margin <= host_rect.width()
                and cand.y() + co_h + margin <= host_rect.height()
            ):
                chosen = cand
                break
        chosen.setX(max(margin, min(host_rect.width() - co_w - margin, chosen.x())))
        chosen.setY(max(margin, min(host_rect.height() - co_h - margin, chosen.y())))
        self._callout.move(chosen)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        # Backdrop
        painter.fillRect(self.rect(), QtGui.QColor(8, 12, 22, 190))
        if not self._target_rect.isEmpty():
            # Punch a soft "spotlight" on the target.
            path = QtGui.QPainterPath()
            path.addRect(QtCore.QRectF(self.rect()))
            spot = QtGui.QPainterPath()
            spot.addRoundedRect(QtCore.QRectF(self._target_rect), 8, 8)
            path = path.subtracted(spot)
            painter.fillPath(path, QtGui.QColor(8, 12, 22, 215))
            # Outline the target.
            pen = QtGui.QPen(QtGui.QColor("#2f8cff"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self._target_rect, 8, 8)
            # Optional: dashed glow inside.
            glow = QtGui.QPen(QtGui.QColor(47, 140, 255, 120))
            glow.setWidth(1)
            glow.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(glow)
            painter.drawRoundedRect(self._target_rect.adjusted(3, 3, -3, -3), 6, 6)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Escape,):
            self._end()
        elif key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_PageDown, QtCore.Qt.Key.Key_Space, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._next()
        elif key in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_PageUp, QtCore.Qt.Key.Key_Backspace):
            self._prev()
        else:
            super().keyPressEvent(event)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._host and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
        ):
            self.setGeometry(self._host.rect())
            self._render_step()
        return False


def _build_legacy_default_tutorial(window: QtWidgets.QMainWindow) -> List[TutorialStep]:
    """Legacy compact tutorial kept for reference; build_default_tutorial overrides it below."""

    def _switch_pre(_w: QtWidgets.QMainWindow) -> None:
        try:
            window.tabs.setCurrentIndex(0)
        except Exception:
            pass

    def _switch_post(_w: QtWidgets.QMainWindow) -> None:
        try:
            window.tabs.setCurrentIndex(1)
        except Exception:
            pass

    def _resolve_tabs(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "tabs", None)

    def _resolve_pre_files(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "file_panel", None)

    def _resolve_status(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "_status_bar", None)

    def _resolve_post(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "post_tab", None)

    def _resolve_temporal(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        post = getattr(window, "post_tab", None)
        if post is None:
            return None
        return getattr(post, "section_temporal", None)

    return [
        TutorialStep(
            "Welcome to pyBer",
            "pyBer takes raw fiber-photometry recordings (Doric / CSV) all the way to "
            "PSTH, behavior alignment, and GLM/FLMM analysis.\n\n"
            "Use ◀ ▶ or arrow keys to step through. Press F1 anytime to reopen this tour.",
            target_resolver=None,
        ),
        TutorialStep(
            "Two main tabs: Preprocessing -> Postprocessing",
            "Preprocessing handles raw signal cleanup, artifact removal and export to "
            "processed traces. Postprocessing consumes those traces for PSTH, peak/event "
            "metrics, behavior alignment, and modeling. The flow is left to right.",
            target_resolver=_resolve_tabs,
            before=_switch_pre,
        ),
        TutorialStep(
            "Step 1 - Drop your files here",
            "Drag .doric, .csv or .h5 files (or a whole folder) onto the file queue. "
            "Use Ctrl+O to browse, Ctrl+Shift+O for a folder, Delete to remove a selection.",
            target_resolver=_resolve_pre_files,
            before=_switch_pre,
        ),
        TutorialStep(
            "Step 2 - Run quality check + export",
            "Ctrl+Q runs QC on the active recording, Ctrl+Shift+Q does a batch QC, "
            "Ctrl+E exports the current selection. Ctrl+Z / Ctrl+Y undo/redo "
            "preprocessing actions.",
            target_resolver=_resolve_status,
            before=_switch_pre,
        ),
        TutorialStep(
            "Step 3 - Switch to Postprocessing",
            "Once you've exported processed traces, hop to the Postprocessing tab. "
            "Drag the .csv/.h5 outputs onto its file dropzone, or open a saved project.",
            target_resolver=_resolve_post,
            before=_switch_post,
        ),
        TutorialStep(
            "Active file vs Group",
            "Postprocessing shows a single recording in 'Individual' mode or a group of "
            "animals in 'Group' mode. Use Ctrl+G to toggle. Ctrl+Left / Ctrl+Right step "
            "through loaded animals.",
            target_resolver=_resolve_post,
            before=_switch_post,
        ),
        TutorialStep(
            "Temporal Modeling",
            "The 'T' panel fits a Continuous GLM or trial-level FLMM. Choose a Scope:\n"
            "- Active file (single)\n"
            "- All loaded (concatenated)\n"
            "- Per-file batch + group\n\n"
            "Press Ctrl+Shift+F to fit. The Group tab aggregates per-file kernels "
            "with mean +/- SEM across animals.",
            target_resolver=_resolve_temporal,
            before=_switch_post,
        ),
        TutorialStep(
            "Keyboard shortcuts",
            "Press Ctrl+/ anytime for the full cheat sheet. Highlights:\n"
            "- F1 Help / replay tour    Ctrl+, Preferences\n"
            "- Ctrl+S Save project      Ctrl+Shift+S Save as\n"
            "- Ctrl+1 Preprocessing     Ctrl+2 Postprocessing\n"
            "- Ctrl+0 Reset focused plot view\n"
            "- Esc Cancel current operation",
            target_resolver=None,
        ),
        TutorialStep(
            "You're set",
            "That's the whirlwind tour. Tooltips on every input fill in the rest. "
            "If something fails the toast in the corner shows what happened - click "
            "it to dismiss.\n\nHappy analyzing.",
            target_resolver=None,
        ),
    ]


def build_default_tutorial(window: QtWidgets.QMainWindow) -> List[TutorialStep]:
    """Expanded first-run tutorial covering the workflow and every rail panel."""

    def _switch_pre(_w: QtWidgets.QMainWindow) -> None:
        try:
            window.tabs.setCurrentIndex(0)
        except Exception:
            pass

    def _switch_post(_w: QtWidgets.QMainWindow) -> None:
        try:
            window.tabs.setCurrentIndex(1)
        except Exception:
            pass

    def _resolve_tabs(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "tabs", None)

    def _resolve_pre_files(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "file_panel", None)

    def _resolve_post(_w: QtWidgets.QMainWindow) -> Optional[QtWidgets.QWidget]:
        return getattr(window, "post_tab", None)

    def _pre_button(key: str) -> Callable[[QtWidgets.QWidget], Optional[QtWidgets.QWidget]]:
        return lambda _w: getattr(window, "_section_buttons", {}).get(key)

    def _post_button(key: str) -> Callable[[QtWidgets.QWidget], Optional[QtWidgets.QWidget]]:
        def _resolver(_w: QtWidgets.QWidget) -> Optional[QtWidgets.QWidget]:
            post = getattr(window, "post_tab", None)
            return getattr(post, "_section_buttons", {}).get(key) if post is not None else None
        return _resolver

    def _open_pre_panel(key: str) -> Callable[[QtWidgets.QWidget], None]:
        def _before(_w: QtWidgets.QWidget) -> None:
            _switch_pre(window)
            try:
                if hasattr(window, "_toggle_section_popup"):
                    window._toggle_section_popup(key, True)
            except Exception:
                pass
        return _before

    def _open_post_panel(key: str) -> Callable[[QtWidgets.QWidget], None]:
        def _before(_w: QtWidgets.QWidget) -> None:
            _switch_post(window)
            try:
                post = getattr(window, "post_tab", None)
                if post is not None and hasattr(post, "_toggle_section_popup"):
                    post._toggle_section_popup(key, True)
            except Exception:
                pass
        return _before

    steps: List[TutorialStep] = [
        TutorialStep(
            "Welcome to pyBer",
            "pyBer takes raw fiber-photometry recordings all the way to PSTH, "
            "behavior alignment, peak/event metrics, spatial maps, and GLM/FLMM analysis.\n\n"
            "Use < > or arrow keys to step through. Press F1 anytime to replay this tour.",
            target_resolver=None,
        ),
        TutorialStep(
            "Two main tabs",
            "Preprocessing cleans raw recordings and exports processed traces. "
            "Postprocessing consumes those traces for group analysis, visualization, and modeling.",
            target_resolver=_resolve_tabs,
            before=_switch_pre,
        ),
        TutorialStep(
            "Load raw files",
            "Drag .doric, .csv, .h5 files, or a whole folder onto the file queue. "
            "Use Ctrl+O to browse, Ctrl+Shift+O for a folder, Delete to remove a selection.",
            target_resolver=_resolve_pre_files,
            before=_switch_pre,
        ),
    ]

    for key, title, body in [
        ("artifacts_list", "Artifact list", "Inspect detected and manual artifact windows, jump to them, and remove entries."),
        ("artifacts", "Artifact setup", "Choose artifact thresholds and how artifacts are handled: interpolate, cut, low-pass locally, or leave unchanged."),
        ("filtering", "Filtering", "Set low-pass and smoothing options for signal and reference traces."),
        ("baseline", "Baseline", "Configure baseline estimation before computing dF/F, dF, or z-score outputs."),
        ("output", "Output", "Choose the processed trace formula and preview the resulting channel."),
        ("qc", "Quality control", "Run per-recording or batch QC before exporting processed data."),
        ("export", "Preprocessing export", "Write selected or batch-processed traces to CSV/H5 for downstream analysis."),
        ("config", "Preprocessing configuration", "Save, reload, or reset preprocessing parameters for reproducible batches."),
    ]:
        steps.append(TutorialStep(title, body, target_resolver=_pre_button(key), before=_open_pre_panel(key)))

    steps.extend([
        TutorialStep(
            "Switch to Postprocessing",
            "Once processed traces are exported, load them here from the File menu, by drag/drop, "
            "or through a saved postprocessing project.",
            target_resolver=_resolve_post,
            before=_switch_post,
        ),
        TutorialStep(
            "Individual and Group views",
            "Individual mode shows one recording at a time; Group mode combines loaded animals. "
            "Use Ctrl+G to toggle and Ctrl+Left / Ctrl+Right to step through files.",
            target_resolver=_resolve_post,
            before=_switch_post,
        ),
    ])

    for key, title, body in [
        ("setup", "Postprocessing setup", "Load processed traces, behavior files, and alignment sources."),
        ("psth", "PSTH panel", "Set event windows, baselines, filtering, rate bins, and pre/post metrics."),
        ("spatial", "Spatial panel", "Build occupancy, activity, and velocity maps from behavior trajectories."),
        ("signal", "Signal events", "Detect peaks/transients, compare amplitude metrics, and overlay detected events/noise."),
        ("behavior", "Behavior panel", "Analyze behavior bouts, durations, starts, and aligned behavior rate."),
        ("temporal", "Temporal Modeling", "Fit Continuous GLM or trial-level FLMM models, including per-file batch and group summaries."),
        ("export", "Postprocessing export", "Export PSTH matrices, metrics, peaks, behavior tables, images, and project files."),
    ]:
        steps.append(TutorialStep(title, body, target_resolver=_post_button(key), before=_open_post_panel(key)))

    steps.extend([
        TutorialStep(
            "Keyboard shortcuts",
            "Press Ctrl+/ anytime for the full cheat sheet. Highlights:\n"
            "- F1 Help / replay tour    Ctrl+, Preferences\n"
            "- Ctrl+S Save project      Ctrl+Shift+S Save as\n"
            "- Ctrl+1 Preprocessing     Ctrl+2 Postprocessing\n"
            "- Ctrl+0 Reset focused plot view\n"
            "- Esc Cancel current operation",
            target_resolver=None,
        ),
        TutorialStep(
            "You're set",
            "Tooltips on every input fill in the rest. If something fails, the toast "
            "in the corner shows what happened; click it to dismiss.",
            target_resolver=None,
        ),
    ])
    return steps


# ============================================================================
# Preferences dialog
# ============================================================================


class PreferencesDialog(QtWidgets.QDialog):
    """
    Compact preferences dialog. Reads/writes via QSettings so the values
    survive restarts. Apply emits no signal; consumers (theme button, etc.)
    pick up changes the next time they read the corresponding key.
    """

    KEYS = {
        "theme": "app/theme",                      # "dark" | "light"
        "autosave": "app/autosave_enabled",        # bool
        "autosave_min": "app/autosave_minutes",    # int
        "kernel_pre": "temporal_modeling/kernel_pre",
        "kernel_post": "temporal_modeling/kernel_post",
        "show_tutorial": "onboarding/replay_next_launch",
        "toast_timeout": "ui/toast_timeout_ms",
    }

    def __init__(self, parent: QtWidgets.QWidget, settings: QtCore.QSettings):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self._settings = settings
        self.resize(540, 420)

        tabs = QtWidgets.QTabWidget(self)

        # ---- Appearance ----
        appearance = QtWidgets.QWidget()
        a = QtWidgets.QFormLayout(appearance)
        a.setContentsMargins(16, 16, 16, 16)
        self.combo_theme = QtWidgets.QComboBox()
        self.combo_theme.addItem("Dark", "dark")
        self.combo_theme.addItem("Light", "light")
        a.addRow("Theme", self.combo_theme)
        self.spin_toast = QtWidgets.QSpinBox()
        self.spin_toast.setRange(1500, 30000)
        self.spin_toast.setSingleStep(500)
        self.spin_toast.setSuffix(" ms")
        a.addRow("Toast default duration", self.spin_toast)
        self.chk_show_tutorial = QtWidgets.QCheckBox(
            "Replay tutorial on next launch"
        )
        a.addRow("Onboarding", self.chk_show_tutorial)
        tabs.addTab(appearance, "Appearance")

        # ---- Defaults ----
        defaults = QtWidgets.QWidget()
        d = QtWidgets.QFormLayout(defaults)
        d.setContentsMargins(16, 16, 16, 16)
        self.spin_kernel_pre = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_pre.setRange(-30.0, 0.0)
        self.spin_kernel_pre.setDecimals(2)
        self.spin_kernel_pre.setSuffix(" s")
        d.addRow("Default GLM kernel pre", self.spin_kernel_pre)
        self.spin_kernel_post = QtWidgets.QDoubleSpinBox()
        self.spin_kernel_post.setRange(0.1, 60.0)
        self.spin_kernel_post.setDecimals(2)
        self.spin_kernel_post.setSuffix(" s")
        d.addRow("Default GLM kernel post", self.spin_kernel_post)
        self.chk_autosave = QtWidgets.QCheckBox("Enable project autosave")
        d.addRow("Autosave", self.chk_autosave)
        self.spin_autosave = QtWidgets.QSpinBox()
        self.spin_autosave.setRange(1, 60)
        self.spin_autosave.setSuffix(" min")
        d.addRow("Autosave interval", self.spin_autosave)
        tabs.addTab(defaults, "Defaults")

        # ---- Keyboard cheat sheet ----
        keyboard = QtWidgets.QWidget()
        k = QtWidgets.QVBoxLayout(keyboard)
        k.setContentsMargins(16, 16, 16, 16)
        cheat = QtWidgets.QTextBrowser()
        cheat.setReadOnly(True)
        cheat.setOpenExternalLinks(False)
        cheat.setHtml(_keyboard_cheatsheet_html())
        k.addWidget(cheat, 1)
        tabs.addTab(keyboard, "Keyboard")

        # ---- About ----
        about = QtWidgets.QWidget()
        ab = QtWidgets.QVBoxLayout(about)
        ab.setContentsMargins(16, 16, 16, 16)
        about_lbl = QtWidgets.QLabel(
            "<h3>pyBer - Fiber Photometry</h3>"
            "<p>Pipeline for raw photometry preprocessing, PSTH, behavior alignment, "
            "and GLM/FLMM modeling.</p>"
            "<p>Bellone Lab toolkit.</p>"
        )
        about_lbl.setWordWrap(True)
        ab.addWidget(about_lbl)
        ab.addStretch(1)
        tabs.addTab(about, "About")

        # ---- Buttons ----
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(tabs)
        layout.addWidget(button_box)

        self._load()

    # --- internals ---

    def _load(self) -> None:
        s = self._settings
        theme = str(s.value(self.KEYS["theme"], "dark") or "dark").lower()
        idx = self.combo_theme.findData(theme)
        if idx >= 0:
            self.combo_theme.setCurrentIndex(idx)
        self.spin_toast.setValue(int(s.value(self.KEYS["toast_timeout"], 5000) or 5000))
        self.chk_show_tutorial.setChecked(_to_bool(s.value(self.KEYS["show_tutorial"], False), False))
        self.spin_kernel_pre.setValue(float(s.value(self.KEYS["kernel_pre"], -1.0) or -1.0))
        self.spin_kernel_post.setValue(float(s.value(self.KEYS["kernel_post"], 3.0) or 3.0))
        self.chk_autosave.setChecked(_to_bool(s.value(self.KEYS["autosave"], True), True))
        self.spin_autosave.setValue(int(s.value(self.KEYS["autosave_min"], 5) or 5))

    def _apply(self) -> None:
        s = self._settings
        s.setValue(self.KEYS["theme"], self.combo_theme.currentData())
        s.setValue(self.KEYS["toast_timeout"], int(self.spin_toast.value()))
        s.setValue(self.KEYS["show_tutorial"], bool(self.chk_show_tutorial.isChecked()))
        s.setValue(self.KEYS["kernel_pre"], float(self.spin_kernel_pre.value()))
        s.setValue(self.KEYS["kernel_post"], float(self.spin_kernel_post.value()))
        s.setValue(self.KEYS["autosave"], bool(self.chk_autosave.isChecked()))
        s.setValue(self.KEYS["autosave_min"], int(self.spin_autosave.value()))

    def _on_accept(self) -> None:
        self._apply()
        self.accept()


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


# ============================================================================
# Keyboard shortcuts
# ============================================================================


_GLOBAL_SHORTCUTS: List[Tuple[str, str, str]] = [
    # (sequence, callback_attr_or_method, description)
    ("F1",            "_show_tutorial_again",      "Show / replay the tutorial"),
    ("Ctrl+/",        "_show_keyboard_cheatsheet", "Open the keyboard cheat sheet"),
    ("Ctrl+,",        "_open_preferences",         "Open Preferences"),
    ("Ctrl+1",        "_focus_pre_tab",            "Switch to Preprocessing tab"),
    ("Ctrl+2",        "_focus_post_tab",           "Switch to Postprocessing tab"),
    ("Ctrl+Tab",      "_cycle_main_tab",           "Cycle main tabs"),
    ("Ctrl+0",        "_reset_focused_plot_view",  "Reset focused plot view"),
    ("Ctrl+Right",    "_step_active_file_next",    "Next loaded file"),
    ("Ctrl+Left",     "_step_active_file_prev",    "Previous loaded file"),
    ("Ctrl+G",        "_toggle_individual_group",  "Toggle Individual / Group"),
    ("Ctrl+Shift+F",  "_fit_temporal_model",       "Fit temporal model"),
    ("Ctrl+Shift+B",  "_fit_temporal_all_files",   "Fit GLM on every file (batch)"),
    ("F5",            "_recompute_psth",           "Recompute PSTH"),
    ("Ctrl+Shift+E",  "_run_postprocess_export",   "Run Postprocessing export"),
    ("Esc",           "_cancel_current_operation", "Cancel current operation"),
]


def register_global_shortcuts(window: QtWidgets.QMainWindow) -> Dict[str, str]:
    """
    Bind every shortcut in `_GLOBAL_SHORTCUTS` to `window` with application-wide
    context. Resolves the callback by attribute name; if the host doesn't define
    the attribute, the shortcut becomes a no-op (so partial wiring is safe).

    Returns a {sequence: description} dict so the cheat sheet can stay accurate.
    """
    descriptions: Dict[str, str] = {}
    for seq, attr, desc in _GLOBAL_SHORTCUTS:
        descriptions[seq] = desc
        sc = QtGui.QShortcut(QtGui.QKeySequence(seq), window)
        sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)

        def _make_handler(attr_name: str) -> Callable[[], None]:
            def _handler() -> None:
                # Don't fire when typing in a line edit / spin box.
                w = QtWidgets.QApplication.focusWidget()
                if isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
                    return
                # QSpinBox / QDoubleSpinBox: allow Esc, Ctrl+Right, etc. to pass.
                fn = getattr(window, attr_name, None)
                if not callable(fn):
                    return
                try:
                    fn()
                except Exception:
                    pass

            return _handler

        sc.activated.connect(_make_handler(attr))
        # Keep a reference so it isn't GC'd.
        if not hasattr(window, "_pyber_global_shortcuts"):
            window._pyber_global_shortcuts = []
        window._pyber_global_shortcuts.append(sc)
    return descriptions


def _keyboard_cheatsheet_html() -> str:
    rows = [
        ("Application", []),
        (None, [
            ("F1", "Help / replay tutorial"),
            ("Ctrl+/", "Open this cheat sheet"),
            ("Ctrl+,", "Preferences"),
            ("Ctrl+1 / Ctrl+2", "Switch to Preprocessing / Postprocessing"),
            ("Ctrl+Tab", "Cycle main tabs"),
            ("Esc", "Cancel current operation"),
        ]),
        ("Files / project", []),
        (None, [
            ("Ctrl+O", "Open files (Preprocessing)"),
            ("Ctrl+Shift+O", "Open folder (Preprocessing)"),
            ("Ctrl+S", "Save project / config"),
            ("Ctrl+Shift+S", "Save as"),
            ("Ctrl+L", "Load preprocessing config"),
            ("Delete", "Remove selected files"),
            ("Ctrl+Right / Ctrl+Left", "Next / previous loaded file"),
            ("Ctrl+G", "Toggle Individual / Group"),
        ]),
        ("Preprocessing", []),
        (None, [
            ("Ctrl+Q", "Run QC on active file"),
            ("Ctrl+Shift+Q", "Batch QC"),
            ("Ctrl+E", "Export current selection"),
            ("Ctrl+K", "Toggle Artifacts panel"),
            ("Ctrl+F", "Toggle Filtering panel"),
            ("Ctrl+B", "Toggle Baseline panel"),
            ("Ctrl+M", "Toggle Output panel"),
            ("Ctrl+D", "Toggle Data panel"),
            ("Ctrl+P", "Toggle parameter popups"),
            ("Ctrl+Enter", "Trigger preview"),
            ("A / C / S", "Assign pending box -> Artifact / Cut / Section"),
        ]),
        ("Postprocessing / Modeling", []),
        (None, [
            ("F5", "Recompute PSTH"),
            ("Ctrl+Shift+E", "Run postprocessing export"),
            ("Ctrl+Shift+F", "Fit temporal model (current scope)"),
            ("Ctrl+Shift+B", "Fit GLM on every file (batch)"),
            ("Ctrl+0", "Reset focused plot view"),
        ]),
    ]
    parts = ["<style>td{padding:3px 14px 3px 0;} kbd{background:#1b2230;color:#e9f0fb;"
            "border:1px solid #355080;border-radius:4px;padding:1px 6px;font-family:Consolas,monospace;}"
            "h4{color:#2f8cff;margin-top:14px;margin-bottom:6px;}</style>"]
    for header, items in rows:
        if header is not None:
            parts.append(f"<h4>{header}</h4>")
        if items:
            parts.append("<table>")
            for keys, desc in items:
                key_html = " ".join(f"<kbd>{k.strip()}</kbd>" for k in keys.split("/"))
                parts.append(f"<tr><td>{key_html}</td><td>{desc}</td></tr>")
            parts.append("</table>")
    return "".join(parts)


# ============================================================================
# Panel header (per-section badge + title + subtitle)
# ============================================================================

# (badge_letter, badge_color, title, subtitle)
_POST_SECTION_META: Dict[str, Tuple[str, str, str, str]] = {
    "setup":    ("S", "#4b9df8", "Setup",         "Load processed traces and behavior / events"),
    "psth":     ("P", "#7d4df2", "PSTH analysis", "Trial windows, baselines and metrics"),
    "spatial":  ("X", "#5dd39e", "Spatial maps",  "Occupancy, activity and velocity heatmaps"),
    "temporal": ("T", "#2d8cff", "Temporal Modeling", "Continuous GLM and trial-level FLMM"),
    "signal":   ("E", "#f5a97f", "Signal events", "Peak detection and amplitude / rate metrics"),
    "behavior": ("B", "#ee99a0", "Behavior",       "Behavior alignment and per-state analysis"),
    "export":   ("D", "#94e2d5", "Export",         "Export PSTH, peaks, behavior and project files"),
}

_PRE_SECTION_META: Dict[str, Tuple[str, str, str, str]] = {
    "artifacts_list": ("L", "#f5c542", "Artifact list",   "Inspect and edit detected / manual artifacts"),
    "artifacts":      ("A", "#ee6471", "Artifact setup",  "Detection thresholds and manual selection"),
    "filtering":      ("F", "#7d4df2", "Filtering",       "Low-pass + smoothing for the photometry trace"),
    "baseline":       ("B", "#4b9df8", "Baseline",        "Baseline estimation across the recording"),
    "output":         ("O", "#5dd39e", "Output",          "Choose dFF / dF / z-score formula"),
    "qc":             ("Q", "#94e2d5", "Quality control", "Per-recording diagnostic checks"),
    "export":         ("D", "#f5a97f", "Export",          "Export processed traces"),
    "config":         ("C", "#aab4c5", "Configuration",   "Save / load preprocessing parameter sets"),
}


class PanelHeader(QtWidgets.QFrame):
    """
    Reusable per-panel header. Shows:
        [coloured badge with letter]  Title (large, bold)
                                      Subtitle (muted, single line)
    Use `set_section(key, meta_dict)` to swap the displayed section.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("pyberPanelHeader")
        self.setStyleSheet("")
        self.setMinimumHeight(54)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 10)
        lay.setSpacing(12)

        self._badge = QtWidgets.QLabel("")
        self._badge.setObjectName("pyberPanelBadge")
        self._badge.setFixedSize(36, 36)
        self._badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._badge)

        col = QtWidgets.QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self._title = QtWidgets.QLabel("")
        self._title.setObjectName("pyberPanelHeaderTitle")
        self._subtitle = QtWidgets.QLabel("")
        self._subtitle.setObjectName("pyberPanelHeaderSubtitle")
        self._subtitle.setWordWrap(True)
        col.addWidget(self._title)
        col.addWidget(self._subtitle)
        lay.addLayout(col, 1)

    def set_from_meta(self, meta: Optional[Tuple[str, str, str, str]]) -> None:
        """meta = (badge_letter, badge_color, title, subtitle); falsy hides everything."""
        if not meta:
            self._badge.setText("")
            self._title.setText("")
            self._subtitle.setText("")
            self._badge.setStyleSheet("background: transparent;")
            return
        letter, color, title, subtitle = meta
        self._badge.setText(letter)
        self._badge.setStyleSheet(
            f"background: {color}; color: #ffffff; border-radius: 18px; "
            f"font-weight: 800; font-size: 14pt;"
        )
        self._title.setText(title)
        self._subtitle.setText(subtitle)

    def set_postprocess_section(self, key: str) -> None:
        self.set_from_meta(_POST_SECTION_META.get(str(key)))

    def set_preprocess_section(self, key: str) -> None:
        self.set_from_meta(_PRE_SECTION_META.get(str(key)))


# ============================================================================
# Reusable in-panel building blocks
#   - Subsection: collapsible card with title + hint + chevron toggle
#   - FooterActions: right-pinned primary button strip pinned to panel bottom
#   - InlineStatus: small banner with severity (info / ok / warn / error)
# ============================================================================


class Subsection(QtWidgets.QFrame):
    """
    A lightweight section inside a panel:
        [▾ ]  Title                                            [optional toggle]
              one-line hint
        ┌───────────────────────────────────────────────────────────────┐
        │  body content (whatever the caller adds)                      │
        └───────────────────────────────────────────────────────────────┘

    Use `add(widget)` or `add_layout(layout)` to populate the body.
    Set `collapsible=True` to expose a chevron that hides the body.
    """

    toggled = QtCore.Signal(bool)  # emitted when collapsed/expanded

    def __init__(
        self,
        title: str,
        hint: Optional[str] = None,
        *,
        collapsible: bool = False,
        start_collapsed: bool = False,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setProperty("class", "subsection")
        self.setStyleSheet("")  # let global QSS style the subsection class

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 8)
        outer.setSpacing(4)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(8)

        self._chevron = QtWidgets.QToolButton()
        self._chevron.setProperty("class", "chevron")
        self._chevron.setText("▾")
        self._chevron.setVisible(bool(collapsible))
        self._chevron.clicked.connect(self._toggle)
        head.addWidget(self._chevron)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)
        self._title = QtWidgets.QLabel(title)
        self._title.setProperty("class", "subsectionTitle")
        text_col.addWidget(self._title)
        if hint:
            self._hint = QtWidgets.QLabel(hint)
            self._hint.setProperty("class", "subsectionHint")
            self._hint.setWordWrap(True)
            text_col.addWidget(self._hint)
        else:
            self._hint = None
        head.addLayout(text_col, 1)

        # Caller can attach an action widget (e.g., "Reset to defaults" link).
        self._head_extra_holder = QtWidgets.QWidget()
        self._head_extra_holder.setStyleSheet("background: transparent;")
        self._head_extra_layout = QtWidgets.QHBoxLayout(self._head_extra_holder)
        self._head_extra_layout.setContentsMargins(0, 0, 0, 0)
        self._head_extra_layout.setSpacing(6)
        head.addWidget(self._head_extra_holder)
        outer.addLayout(head)

        self._body = QtWidgets.QWidget()
        self._body.setStyleSheet("background: transparent;")
        self._body_layout = QtWidgets.QVBoxLayout(self._body)
        # Small left indent so subsection body sits under the title text but
        # leaves the full drawer width available for inputs.
        self._body_layout.setContentsMargins(6, 4, 0, 4)
        self._body_layout.setSpacing(6)
        outer.addWidget(self._body)

        if start_collapsed and collapsible:
            self._chevron.setText("▸")
            self._body.setVisible(False)

    def add(self, widget: QtWidgets.QWidget) -> None:
        self._body_layout.addWidget(widget)

    def add_layout(self, layout: QtWidgets.QLayout) -> None:
        self._body_layout.addLayout(layout)

    def add_head_action(self, widget: QtWidgets.QWidget) -> None:
        """Pin a small action widget (button / link) to the right of the title."""
        self._head_extra_layout.addWidget(widget)

    def _toggle(self) -> None:
        is_open = self._body.isVisible()
        self._body.setVisible(not is_open)
        self._chevron.setText("▸" if is_open else "▾")
        self.toggled.emit(not is_open)


class FooterActions(QtWidgets.QFrame):
    """
    A pinned footer strip for primary panel actions:
        [optional status]                                [Cancel] [Primary]
    Add buttons via `add_primary` (right side) or `add_secondary` (right side, ghost).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "footerActions")
        self.setStyleSheet("")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)

        self._status = QtWidgets.QLabel("")
        self._status.setProperty("class", "muted")
        self._status.setWordWrap(True)
        lay.addWidget(self._status, 1)

        self._right = QtWidgets.QHBoxLayout()
        self._right.setSpacing(6)
        lay.addLayout(self._right)

    def set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))

    def add_primary(self, button: QtWidgets.QPushButton) -> None:
        button.setProperty("class", "primary")
        self._right.addWidget(button)

    def add_secondary(self, button: QtWidgets.QPushButton) -> None:
        button.setProperty("class", "ghost")
        self._right.addWidget(button)

    def add_widget(self, widget: QtWidgets.QWidget) -> None:
        self._right.addWidget(widget)


class InlineStatus(QtWidgets.QFrame):
    """
    Inline status banner: [icon] short-message
    severity = "info" | "ok" | "warn" | "error"
    """

    def __init__(self, text: str = "", severity: str = "info",
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setProperty("class", "inlineStatus")
        self.setProperty("severity", severity)
        self.setStyleSheet("")

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(8)

        self._icon = QtWidgets.QLabel("")
        self._icon.setFixedWidth(18)
        self._icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._icon)

        self._label = QtWidgets.QLabel("")
        self._label.setWordWrap(True)
        lay.addWidget(self._label, 1)

        self.setVisible(bool(text))
        self.set(text, severity)

    def set(self, text: str, severity: str = "info") -> None:
        glyph = {"info": "i", "ok": "v", "warn": "!", "error": "x"}.get(severity, "i")
        self._icon.setText(glyph)
        self._label.setText(str(text or ""))
        self.setProperty("severity", severity)
        self.style().unpolish(self); self.style().polish(self)
        self.setVisible(bool(text))


# ============================================================================
# Top app bar (workflow header)
# ============================================================================


class TopAppBar(QtWidgets.QFrame):
    """
    Persistent strip above the main tab widget. Shows:
        [logo]  pyBer        Preprocessing > Postprocessing      [Project: name *]   [Help]
    The workflow step is highlighted based on the active main tab.
    """

    helpRequested = QtCore.Signal()
    preferencesRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setObjectName("pyberTopBar")
        self.setFixedHeight(54)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(12)

        self._mark = QtWidgets.QLabel("p")
        self._mark.setObjectName("pyberAppMark")
        self._mark.setFixedSize(30, 30)
        lay.addWidget(self._mark)

        self._app_name = QtWidgets.QLabel("pyBer")
        self._app_name.setObjectName("pyberAppName")
        lay.addWidget(self._app_name)

        # Workflow steps
        self._steps_holder = QtWidgets.QFrame()
        sl = QtWidgets.QHBoxLayout(self._steps_holder)
        sl.setContentsMargins(12, 0, 12, 0)
        sl.setSpacing(6)
        self._step_labels: List[QtWidgets.QLabel] = []
        for i, name in enumerate(("Preprocessing", "Postprocessing")):
            if i > 0:
                sep = QtWidgets.QLabel("›")
                sep.setObjectName("pyberWorkflowSep")
                sl.addWidget(sep)
            lbl = QtWidgets.QLabel(name)
            lbl.setObjectName("pyberWorkflowStep")
            lbl.setProperty("active", "true" if i == 0 else "false")
            self._step_labels.append(lbl)
            sl.addWidget(lbl)
        lay.addWidget(self._steps_holder)
        lay.addStretch(1)

        self._project_lbl = QtWidgets.QLabel("Untitled project")
        self._project_lbl.setObjectName("pyberProjectName")
        self._project_lbl.setProperty("dirty", "false")
        lay.addWidget(self._project_lbl)

        self._btn_prefs = QtWidgets.QPushButton("Preferences")
        self._btn_prefs.setProperty("class", "ghost")
        self._btn_prefs.setToolTip("Open preferences (Ctrl+,)")
        self._btn_prefs.clicked.connect(self.preferencesRequested.emit)
        lay.addWidget(self._btn_prefs)

        self._btn_help = QtWidgets.QPushButton("?")
        self._btn_help.setProperty("class", "help")
        self._btn_help.setToolTip("Replay tutorial / help (F1)")
        self._btn_help.clicked.connect(self.helpRequested.emit)
        lay.addWidget(self._btn_help)

    def set_active_step(self, index: int) -> None:
        for i, lbl in enumerate(self._step_labels):
            lbl.setProperty("active", "true" if i == int(index) else "false")
            # Force style recompute.
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
            lbl.update()

    def set_project_name(self, name: str, dirty: bool = False) -> None:
        text = (str(name).strip() or "Untitled project")
        if dirty:
            text = text + " *"
        self._project_lbl.setText(text)
        self._project_lbl.setProperty("dirty", "true" if dirty else "false")
        self._project_lbl.style().unpolish(self._project_lbl)
        self._project_lbl.style().polish(self._project_lbl)
        self._project_lbl.update()


# ============================================================================
# Window helpers
# ============================================================================


def attach_dirty_title(
    window: QtWidgets.QMainWindow,
    base_title: str,
    is_dirty_callback: Callable[[], bool],
) -> Callable[[], None]:
    """
    Returns a `refresh()` function. Call it whenever the dirty state may have
    changed and the title bar will gain/lose its trailing '*'.
    """

    def refresh() -> None:
        try:
            dirty = bool(is_dirty_callback())
        except Exception:
            dirty = False
        suffix = " *" if dirty else ""
        window.setWindowTitle(f"{base_title}{suffix}")

    refresh()
    return refresh


def install_close_confirmation(
    window: QtWidgets.QMainWindow,
    is_dirty_callback: Callable[[], bool],
    save_callback: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Wraps the window's closeEvent to prompt when there is unsaved work.
    `save_callback` (if provided) should perform the save and return True
    on success.
    """
    original = window.closeEvent

    def closeEvent(event: QtGui.QCloseEvent) -> None:
        try:
            dirty = bool(is_dirty_callback())
        except Exception:
            dirty = False
        if not dirty:
            original(event)
            return
        choice = QtWidgets.QMessageBox.question(
            window,
            "Unsaved changes",
            "You have unsaved postprocessing changes. Save before exiting?",
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save,
        )
        if choice == QtWidgets.QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        if choice == QtWidgets.QMessageBox.StandardButton.Save and save_callback is not None:
            try:
                if not save_callback():
                    event.ignore()
                    return
            except Exception:
                event.ignore()
                return
        original(event)

    window.closeEvent = closeEvent  # type: ignore[assignment]


# ============================================================================
# Plot helpers
# ============================================================================


def reset_focused_plot_view(window: QtWidgets.QWidget) -> None:
    """
    Walk up from the focus widget looking for a pyqtgraph PlotWidget;
    if found, autorange.
    """
    try:
        import pyqtgraph as pg
    except Exception:
        return
    candidate = QtWidgets.QApplication.focusWidget() or window
    while candidate is not None:
        if isinstance(candidate, pg.PlotWidget):
            try:
                candidate.getPlotItem().enableAutoRange()
            except Exception:
                pass
            return
        candidate = candidate.parent() if hasattr(candidate, "parent") else None
    # Fallback: autorange all visible PlotWidgets on the window.
    for pw in window.findChildren(pg.PlotWidget):
        try:
            pw.getPlotItem().enableAutoRange()
        except Exception:
            pass


def add_empty_state_hint(
    plot,  # type: ignore[no-untyped-def]  pg.PlotWidget
    text: str,
    color: str = "#7d8aa1",
) -> Optional[Any]:
    """
    Add a non-interactive TextItem at (0,0) on the plot with the given hint
    text. Returns the item so callers can hide() / setVisible(False) once
    real data is plotted. Safe no-op if pyqtgraph isn't importable.
    """
    try:
        import pyqtgraph as pg
    except Exception:
        return None
    try:
        item = pg.TextItem(str(text), color=color, anchor=(0.5, 0.5))
        item.setZValue(100)
        plot.addItem(item)
        item.setPos(0, 0)
        return item
    except Exception:
        return None
