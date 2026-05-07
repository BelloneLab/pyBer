from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


def _event_global_pos(event: QtCore.QEvent) -> QtCore.QPoint:
    try:
        return event.globalPosition().toPoint()  # Qt 6
    except Exception:
        try:
            return event.globalPos()  # Qt 5 compatibility for older shims
        except Exception:
            return QtCore.QPoint()


def _event_local_pos(event: QtCore.QEvent) -> QtCore.QPoint:
    try:
        return event.position().toPoint()  # Qt 6
    except Exception:
        try:
            return event.pos()
        except Exception:
            return QtCore.QPoint()


class SpinBoxScrubber(QtCore.QObject):
    """Turn spin boxes into arrowless, draggable numeric controls.

    Users can still click into the field and type exact values. Dragging left/right
    changes the value using the spin box's native stepping, so existing signal
    wiring and validation continue to work.
    """

    _CONFIGURED_PROP = "_pyber_spin_scrubber_configured"

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._press_spin: Optional[QtWidgets.QAbstractSpinBox] = None
        self._press_pos = QtCore.QPoint()
        self._press_local_pos = QtCore.QPoint()
        self._last_steps = 0
        self._dragging = False
        self._override_cursor = False

    def scan(self, root: QtCore.QObject) -> None:
        if isinstance(root, QtWidgets.QAbstractSpinBox):
            self._configure_spinbox(root)
        if isinstance(root, QtWidgets.QWidget):
            for spin in root.findChildren(QtWidgets.QAbstractSpinBox):
                self._configure_spinbox(spin)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        etype = event.type()
        if etype == QtCore.QEvent.Type.Show:
            self._configure_object_tree(obj)

        spin = self._spinbox_for_object(obj)
        if spin is None:
            return False
        if etype in (
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.Wheel,
            QtCore.QEvent.Type.FocusIn,
            QtCore.QEvent.Type.KeyPress,
            QtCore.QEvent.Type.Show,
        ):
            self._configure_spinbox(spin)
        elif not bool(spin.property(self._CONFIGURED_PROP)):
            return False

        if etype == QtCore.QEvent.Type.MouseButtonPress:
            if not self._left_button_event(event) or not spin.isEnabled():
                return False
            self._press_spin = spin
            self._press_pos = _event_global_pos(event)
            self._press_local_pos = _event_local_pos(event)
            self._last_steps = 0
            self._dragging = False
            return False

        if etype == QtCore.QEvent.Type.MouseMove and self._press_spin is spin:
            if not self._left_button_held(event):
                return False
            dx_global = _event_global_pos(event).x() - self._press_pos.x()
            dx_local = _event_local_pos(event).x() - self._press_local_pos.x()
            dx = dx_global if abs(dx_global) >= abs(dx_local) else dx_local
            if not self._dragging:
                if abs(dx) < 5:
                    return False
                self._dragging = True
                spin.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
                self._set_override_cursor(QtCore.Qt.CursorShape.SizeHorCursor)

            steps = self._steps_from_drag(dx, event)
            delta = steps - self._last_steps
            if delta:
                spin.stepBy(delta)
                self._last_steps = steps
            return True

        if etype == QtCore.QEvent.Type.MouseButtonRelease and self._press_spin is spin:
            was_dragging = self._dragging
            self._press_spin = None
            self._last_steps = 0
            self._dragging = False
            self._restore_override_cursor()
            return was_dragging

        if etype == QtCore.QEvent.Type.Leave and self._press_spin is spin and not self._left_button_held(event):
            self._press_spin = None
            self._last_steps = 0
            self._dragging = False
            self._restore_override_cursor()

        return False

    def _configure_object_tree(self, obj: QtCore.QObject) -> None:
        spin = self._spinbox_for_object(obj)
        if spin is not None:
            self._configure_spinbox(spin)
            return
        if isinstance(obj, QtWidgets.QWidget):
            for child in obj.findChildren(QtWidgets.QAbstractSpinBox):
                self._configure_spinbox(child)

    def _configure_spinbox(self, spin: QtWidgets.QAbstractSpinBox) -> None:
        if bool(spin.property(self._CONFIGURED_PROP)):
            return
        spin.setProperty(self._CONFIGURED_PROP, True)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setKeyboardTracking(False)
        spin.setAccelerated(True)
        spin.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
        line_edit = spin.lineEdit()
        if line_edit is not None:
            line_edit.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            line_edit.setTextMargins(1, 0, 1, 0)
        if isinstance(spin, QtWidgets.QDoubleSpinBox):
            try:
                spin.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            except Exception:
                pass
        tip = spin.toolTip().strip()
        scrub_tip = "Drag left/right to adjust. Type for an exact value. Shift = faster, Ctrl = finer."
        if scrub_tip not in tip:
            spin.setToolTip(f"{tip}\n{scrub_tip}" if tip else scrub_tip)

    def _spinbox_for_object(self, obj: QtCore.QObject) -> Optional[QtWidgets.QAbstractSpinBox]:
        if isinstance(obj, QtWidgets.QAbstractSpinBox):
            return obj
        parent = obj.parent() if isinstance(obj, QtCore.QObject) else None
        while parent is not None:
            if isinstance(parent, QtWidgets.QAbstractSpinBox):
                return parent
            parent = parent.parent()
        return None

    def _steps_from_drag(self, dx: int, event: QtCore.QEvent) -> int:
        pixels_per_step = 12.0
        try:
            mods = event.modifiers()
        except Exception:
            mods = QtCore.Qt.KeyboardModifier.NoModifier
        if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
            pixels_per_step = 5.0
        elif mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            pixels_per_step = 28.0
        return int(dx / pixels_per_step)

    def _left_button_event(self, event: QtCore.QEvent) -> bool:
        try:
            return event.button() == QtCore.Qt.MouseButton.LeftButton
        except Exception:
            return False

    def _left_button_held(self, event: QtCore.QEvent) -> bool:
        try:
            return bool(event.buttons() & QtCore.Qt.MouseButton.LeftButton)
        except Exception:
            return False

    def _set_override_cursor(self, cursor: QtCore.Qt.CursorShape) -> None:
        if self._override_cursor:
            return
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(cursor))
        self._override_cursor = True

    def _restore_override_cursor(self) -> None:
        if not self._override_cursor:
            return
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self._override_cursor = False


def install_spinbox_scrubbers(app: QtWidgets.QApplication) -> SpinBoxScrubber:
    existing = getattr(app, "_pyber_spinbox_scrubber", None)
    if isinstance(existing, SpinBoxScrubber):
        return existing
    scrubber = SpinBoxScrubber(app)
    app.installEventFilter(scrubber)
    setattr(app, "_pyber_spinbox_scrubber", scrubber)
    return scrubber
