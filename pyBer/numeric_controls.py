from __future__ import annotations

import math
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid


def _is_alive(obj: object) -> bool:
    """Return whether a PySide wrapper still owns a live C++ object.

    Qt can deliver a final queued event while tearing down a dialog. During
    that narrow window ``isinstance`` still succeeds for a Python wrapper, but
    any Qt method call, including ``parent()``, raises ``RuntimeError`` because
    the underlying C++ widget has already gone away.
    """
    try:
        return isinstance(obj, QtCore.QObject) and bool(isValid(obj))
    except (RuntimeError, TypeError):
        return False


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


# Editor inset matching the app stylesheet (1 px border + 8 px padding).
_EDITOR_INSET_X = 9
_MIN_EDITOR_HEIGHT = 18


def ensure_editor_visible(spin: QtWidgets.QAbstractSpinBox) -> bool:
    """Re-place a spin box's text editor when the Qt style collapsed it.

    Qt 6.11.3 lays out the editor of a style-sheeted spin box without arrow
    buttons as a 1 px wide strip, so the value is invisible although it is
    still stored and editable. When that happens the editor gets the full
    inner box (minus border and padding, vertically centred). Healthy layouts
    are left untouched. Returns True when a correction was applied.
    """
    if not _is_alive(spin):
        return False
    editor = spin.lineEdit()
    if editor is None or not _is_alive(editor):
        return False
    rect = spin.rect()
    inner_width = rect.width() - 2 * _EDITOR_INSET_X
    if inner_width <= 0 or editor.width() >= inner_width // 2:
        return False
    height = min(rect.height() - 2, max(_MIN_EDITOR_HEIGHT, editor.sizeHint().height()))
    top = max(0, (rect.height() - height) // 2)
    editor.setGeometry(_EDITOR_INSET_X, top, inner_width, height)
    return True


class SpinBoxScrubber(QtCore.QObject):
    """Keep numeric fields editable, with optional Alt-drag adjustment.

    Users can still click into the field and type exact values. Dragging left/right
    changes the value using the spin box's native stepping, so existing signal
    wiring and validation continue to work.
    """

    _CONFIGURED_PROP = "_pyber_spin_scrubber_configured"
    _SLIDER_PROP = "_pyber_slider_checked"
    _GEOMETRY_EVENTS = (
        QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Resize,
        QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.FontChange,
    )

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._press_spin: Optional[QtWidgets.QAbstractSpinBox] = None
        self._press_pos = QtCore.QPoint()
        self._press_local_pos = QtCore.QPoint()
        self._last_steps = 0
        self._dragging = False
        self._override_cursor = False

    def scan(self, root: QtCore.QObject) -> None:
        if not _is_alive(root):
            return
        if isinstance(root, QtWidgets.QAbstractSpinBox):
            self._configure_spinbox(root)
        if isinstance(root, QtWidgets.QWidget):
            for spin in root.findChildren(QtWidgets.QAbstractSpinBox):
                self._configure_spinbox(spin)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        # A dialog's spin-box editor can be destroyed before its last queued
        # event reaches this application-wide filter. Never dereference such a
        # stale PySide wrapper. This is particularly visible when QColorDialog
        # closes from the postprocessing Plot Styling dialog.
        if not _is_alive(obj):
            self._clear_deleted_press_target()
            return False

        self._clear_deleted_press_target()
        etype = event.type()
        if etype == QtCore.QEvent.Type.Show:
            self._configure_object_tree(obj)

        spin = self._spinbox_for_object(obj)
        if spin is None:
            return False
        if obj is spin and etype in self._GEOMETRY_EVENTS:
            # Run after Qt has placed the editor for this event.
            QtCore.QTimer.singleShot(0, lambda s=spin: ensure_editor_visible(s))
        if obj is spin and etype == QtCore.QEvent.Type.Show and not spin.property(self._SLIDER_PROP):
            # Every standalone numeric setting gets a slider the first time it
            # is shown, wherever it lives (drawers, dialogs built later).
            spin.setProperty(self._SLIDER_PROP, True)
            QtCore.QTimer.singleShot(0, lambda s=spin: wrap_form_spinbox(s))
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
            # Ordinary drags belong to the text editor (selection, copy, paste).
            # Require Alt for scrubbing so a small mouse movement cannot edit data.
            if not event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier:
                return False
            self._press_spin = spin
            self._press_pos = _event_global_pos(event)
            self._press_local_pos = _event_local_pos(event)
            self._last_steps = 0
            self._dragging = False
            return False

        if etype == QtCore.QEvent.Type.Wheel and not spin.hasFocus():
            # Let the surrounding settings drawer scroll without editing values.
            event.ignore()
            return True

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
        if not _is_alive(obj):
            return
        spin = self._spinbox_for_object(obj)
        if spin is not None:
            self._configure_spinbox(spin)
            return
        if isinstance(obj, QtWidgets.QWidget):
            for child in obj.findChildren(QtWidgets.QAbstractSpinBox):
                self._configure_spinbox(child)

    def _configure_spinbox(self, spin: QtWidgets.QAbstractSpinBox) -> None:
        if not _is_alive(spin):
            return
        if bool(spin.property(self._CONFIGURED_PROP)):
            return
        spin.setProperty(self._CONFIGURED_PROP, True)
        spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setKeyboardTracking(False)
        spin.setAccelerated(True)
        spin.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        spin.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        line_edit = spin.lineEdit()
        if line_edit is not None:
            line_edit.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
            line_edit.setTextMargins(1, 0, 1, 0)
        tip = spin.toolTip().strip()
        scrub_tip = "Type an exact value; Enter or Tab applies it. Alt-drag adjusts; Shift = faster, Ctrl = finer."
        if scrub_tip not in tip:
            spin.setToolTip(f"{tip}\n{scrub_tip}" if tip else scrub_tip)

    def _spinbox_for_object(self, obj: QtCore.QObject) -> Optional[QtWidgets.QAbstractSpinBox]:
        if not _is_alive(obj):
            return None
        if isinstance(obj, QtWidgets.QAbstractSpinBox):
            return obj
        parent = obj.parent()
        while _is_alive(parent):
            if isinstance(parent, QtWidgets.QAbstractSpinBox):
                return parent
            parent = parent.parent()
        return None

    def _clear_deleted_press_target(self) -> None:
        """Release drag state if its spin box vanished with a closed dialog."""
        if self._press_spin is None or _is_alive(self._press_spin):
            return
        self._press_spin = None
        self._last_steps = 0
        self._dragging = False
        self._restore_override_cursor()

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


class NumericSlider(QtWidgets.QWidget):
    """Pair a compact slider with the existing, authoritative numeric editor.

    Slider position is an approximation for exploration. Typing retains the
    editor's full precision, range, units, validation and original signal wiring.
    Logarithmic mapping uses log1p to include a genuine zero endpoint.
    """

    RESOLUTION = 1000
    STYLE = """
        QSlider::groove:horizontal { height: 4px; background: #566078;
                                    border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #9272ff; border-radius: 2px; }
        QSlider::handle:horizontal { background: #b9a5ff; border: 1px solid #8060df;
                                    width: 12px; margin: -5px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #ded3ff; }
        QSlider::handle:horizontal:focus { border: 2px solid #ede7ff; }
        QSlider::sub-page:horizontal:disabled { background: #677087; }
        QSlider::handle:horizontal:disabled { background: #7a8191; border-color: #677087; }
    """

    def __init__(self, spin, *, logarithmic: bool = False) -> None:
        super().__init__()
        self.spin = spin
        self.logarithmic = logarithmic
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, self.RESOLUTION)
        self.slider.setMinimumWidth(48)
        self.slider.setFixedHeight(22)
        self.slider.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.slider.setStyleSheet(self.STYLE)
        self.slider.setAccessibleName("Adjust numeric value")
        self.setFocusProxy(spin)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.slider, 1)
        layout.addWidget(spin, 1)
        spin.setKeyboardTracking(False)
        spin.valueChanged.connect(self._sync_from_spin)
        self.slider.valueChanged.connect(self._set_from_slider)
        spin.installEventFilter(self)
        self.slider.installEventFilter(self)
        self._sync_from_spin()

    def _mapping(self):
        """Read current limits, including changes made after construction."""
        low, high = self.spin.minimum(), self.spin.maximum()
        scale = max(abs(low), 10 ** (-self.spin.decimals()) if isinstance(
            self.spin, QtWidgets.QDoubleSpinBox) else 1)
        return low, high, scale

    def _sync_from_spin(self, *_args) -> None:
        """Move the thumb without rounding or re-emitting the typed value."""
        low, high, scale = self._mapping()
        fraction = (self.spin.value() - low) / (high - low) if high > low else 0
        if self.logarithmic and high > low:
            fraction = math.log1p((self.spin.value() - low) / scale) / math.log1p((high - low) / scale)
        blocker = QtCore.QSignalBlocker(self.slider)
        self.slider.setValue(round(fraction * self.RESOLUTION))
        del blocker
        self.slider.setEnabled(self.spin.isEnabled() and not self.spin.isReadOnly() and high > low)
        self.slider.setToolTip(f"{self.spin.text()}\nDrag to adjust, or type an exact value in the field.")

    def _set_from_slider(self, position: int) -> None:
        """Apply slider edits through the same validated spin-box value path."""
        low, high, scale = self._mapping()
        fraction = position / self.RESOLUTION
        value = low + fraction * (high - low)
        if self.logarithmic:
            value = low + scale * math.expm1(fraction * math.log1p((high - low) / scale))
        if isinstance(self.spin, QtWidgets.QSpinBox):
            value = round(value)
        self.spin.setValue(value)

    def eventFilter(self, obj, event) -> bool:
        """Follow availability changes and protect scrolling from stray edits."""
        if obj is self.spin and event.type() in (
            QtCore.QEvent.Type.EnabledChange, QtCore.QEvent.Type.Show,
            QtCore.QEvent.Type.ReadOnlyChange,
        ):
            self._sync_from_spin()
        # Code that hides/shows the numeric field (conditional settings rows)
        # must hide/show the whole slider row with it.
        if obj is self.spin and event.type() == QtCore.QEvent.Type.HideToParent:
            self.setVisible(False)
        elif obj is self.spin and event.type() == QtCore.QEvent.Type.ShowToParent:
            self.setVisible(True)
        if obj is self.slider and event.type() == QtCore.QEvent.Type.Wheel and not self.slider.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def with_slider(spin, *, logarithmic: bool = False) -> NumericSlider:
    """Use in a settings layout while retaining the original spin-box attribute."""
    return NumericSlider(spin, logarithmic=logarithmic)


def _wants_log_scale(spin: QtWidgets.QAbstractSpinBox) -> bool:
    """Log mapping when the range spans more than two decades above its floor.

    A linear thumb over e.g. 0-1e6 would make every useful value sit in the
    first pixel; log1p mapping keeps small values reachable and still includes 0.
    """
    low, high = spin.minimum(), spin.maximum()
    if low < 0 or high <= low:
        return False
    step = 10 ** (-spin.decimals()) if isinstance(spin, QtWidgets.QDoubleSpinBox) else 1
    return (high - low) / max(abs(low), step, 1e-12) > 100 and (high - low) > 100 * step


NO_SLIDER_PROP = "_pyber_no_slider"


def _form_slot(spin: QtWidgets.QAbstractSpinBox):
    """Return ``(form, row)`` when ``spin`` is directly the field of a form row."""
    parent = spin.parentWidget()
    if parent is None or isinstance(parent, NumericSlider):
        return None
    forms = parent.findChildren(QtWidgets.QFormLayout)
    if isinstance(parent.layout(), QtWidgets.QFormLayout):
        forms.insert(0, parent.layout())
    for form in forms:
        if form.indexOf(spin) < 0:
            continue
        row, role = form.getWidgetPosition(spin)
        if row >= 0 and role == QtWidgets.QFormLayout.ItemRole.FieldRole:
            return form, row
    return None


def wrap_form_spinbox(spin: QtWidgets.QAbstractSpinBox) -> bool:
    """Pair a standalone form-row spin box with a slider; True when converted.

    Only rows whose *field* is the bare spin box are converted, so paired rows
    (Start/End, Pre/Post) and custom composite rows keep their layout. The spin
    box object, its signals and attribute names are unchanged, so existing
    wiring keeps working. Set the ``_pyber_no_slider`` property to opt out.
    """
    if not _is_alive(spin) or isinstance(spin, QtWidgets.QDateTimeEdit):
        return False
    if spin.isReadOnly() or bool(spin.property(NO_SLIDER_PROP)):
        return False
    if spin.minimum() < 0 and spin.maximum() - spin.minimum() >= 1e5:
        # Effectively unbounded signed fields (e.g. exact colour limits): a
        # thumb position would carry no meaning, keep the typed field only.
        return False
    slot = _form_slot(spin)
    if slot is None:
        return False
    form, row = slot
    explicitly_hidden = spin.isHidden()
    form.removeWidget(spin)
    wrapper = NumericSlider(spin, logarithmic=_wants_log_scale(spin))
    form.setWidget(row, QtWidgets.QFormLayout.ItemRole.FieldRole, wrapper)
    spin.setVisible(not explicitly_hidden)
    wrapper.setVisible(not explicitly_hidden)
    return True


def add_sliders_to_forms(root: QtWidgets.QWidget) -> int:
    """Pair every standalone numeric form field under ``root`` with a slider."""
    return sum(wrap_form_spinbox(spin) for spin in root.findChildren(QtWidgets.QAbstractSpinBox))


def install_spinbox_scrubbers(app: QtWidgets.QApplication) -> SpinBoxScrubber:
    existing = getattr(app, "_pyber_spinbox_scrubber", None)
    if isinstance(existing, SpinBoxScrubber):
        return existing
    scrubber = SpinBoxScrubber(app)
    app.installEventFilter(scrubber)
    setattr(app, "_pyber_spinbox_scrubber", scrubber)
    return scrubber
