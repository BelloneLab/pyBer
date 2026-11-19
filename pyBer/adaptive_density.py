"""Tool panels that shrink their content instead of clipping it.

A drawer page lives in a scroll area. When the user makes the drawer narrower
than the page's minimum width, Qt keeps the page at that minimum and cuts off
whatever lies to the right. ``AdaptiveDensity`` watches the available width and
steps the page through denser presentations until it fits:

* slightly smaller text, never below ``MIN_FONT_PT``,
* tighter button / field padding,
* tighter layout spacing and margins,
* proportionally smaller explicit minimum and fixed widths.

Before any of that, rows marked ``density_keep_row`` (side-by-side lists that
must stay aligned) let their captions and lists get narrower: the mildest step.

If the densest level is still too wide, rows are reflowed, the way a web page
does on a phone:

* a horizontal row that cannot fit is stacked vertically (innermost first);
* form rows put their label above their field;
* kept rows are never stacked.

Widening the drawer undoes these steps in reverse order, back to the normal
presentation. As a last resort the scroll area shows a horizontal scroll bar,
so no option is ever unreachable.

Usage::

    scroll = QtWidgets.QScrollArea(); scroll.setWidget(page)
    AdaptiveDensity(scroll)          # keeps itself alive as a child of scroll
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6 import QtCore, QtWidgets


# =============================================================================
# Configuration
# =============================================================================

#: Smallest text a panel may use (points). The application's normal control
#: text is about 9 to 10 pt. Below this floor a panel stops shrinking its text
#: and saves width by tightening padding and spacing, then by stacking rows.
MIN_FONT_PT = 8.4
#: Density levels, from normal to densest:
#: (geometry scale, font size in pt, vertical padding px, horizontal padding px,
#:  minimum control height px). Level 0 leaves the application style untouched.
DENSITY_LEVELS = (
    (1.00, None, None, None, None),
    (0.92, max(MIN_FONT_PT, 8.9), 4, 8, 20),
    (0.84, max(MIN_FONT_PT, 8.4), 3, 6, 19),
    (0.74, MIN_FONT_PT, 2, 4, 18),
    (0.64, MIN_FONT_PT, 1, 3, 18),
)
#: Extra pixels required before returning to a less dense level (no flicker
#: while the divider rests on a threshold).
HYSTERESIS_PX = 14
#: Delay after a resize before the density is re-evaluated (milliseconds).
DEBOUNCE_MS = 40

#: Layout property: keep this row (and the rows inside it) horizontal.
KEEP_ROW_PROPERTY = "density_keep_row"
#: Narrowest caption / list inside a kept row once it is squeezed (pixels).
SQUEEZED_LABEL_WIDTH = 18
SQUEEZED_VIEW_WIDTH = 56
#: Ancestor margins assumed around a row when testing whether it fits (pixels).
ROW_FIT_SLACK_PX = 12
#: When no single row is wider than the drawer, the widest row is still stacked
#: if it uses at least this fraction of the width (nested margins add the rest).
WIDE_ROW_FRACTION = 0.6
#: Layout property: stacking this row did not make the page narrower.
_P_NO_GAIN = "_density_no_gain"

_SHEET_BEGIN = "/* adaptive-density:begin */"
_SHEET_END = "/* adaptive-density:end */"
_QWIDGETSIZE_MAX = 16777215
# Dynamic property names used to remember each object's normal geometry.
_P_MIN_W = "_density_min_width"
_P_MAX_W = "_density_max_width"
_P_SPACING = "_density_spacing"
_P_MARGINS = "_density_margins"


def _density_sheet(level: int) -> str:
    """Style sheet block for one density level ('' for the normal level)."""
    _scale, font_pt, pad_v, pad_h, min_h = DENSITY_LEVELS[level]
    if font_pt is None:
        return ""
    return (
        f"{_SHEET_BEGIN}\n"
        f"* {{ font-size: {font_pt:.1f}pt; }}\n"
        # Spin boxes keep the application's own padding: their text field is
        # sensitive to it. Their text still shrinks with the rule above.
        f"QPushButton, QToolButton, QComboBox, QLineEdit {{ padding: {pad_v}px {pad_h}px; "
        f"min-height: {min_h}px; }}\n"
        f"QAbstractSpinBox {{ min-height: {min_h}px; }}\n"
        f"{_SHEET_END}"
    )


def _strip_density_sheet(sheet: str) -> str:
    """Remove a previously added density block, keeping the widget's own rules."""
    start = sheet.find(_SHEET_BEGIN)
    if start < 0:
        return sheet
    end = sheet.find(_SHEET_END, start)
    end = len(sheet) if end < 0 else end + len(_SHEET_END)
    return (sheet[:start] + sheet[end:]).strip()


class AdaptiveDensity(QtCore.QObject):
    """Fit a scroll area's page to the viewport width by changing its density."""

    levelChanged = QtCore.Signal(int)

    def __init__(self, scroll: QtWidgets.QScrollArea) -> None:
        super().__init__(scroll)
        self._scroll = scroll
        self._level = 0
        # Minimum page width last measured at each level.
        self._needed: Dict[int, int] = {}
        self._applying = False
        # Rows stacked vertically, most recent last: (layout, restore value,
        # page width that was needed before stacking it).
        self._reflowed: list = []
        # Widgets narrowed inside kept rows: (widget, normal minimum width, tooltip).
        self._squeezed: list = []
        self._squeeze_needed = 0
        # Row stacked on the previous tick, kept only if it made the page narrower.
        self._trial_row = None
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._evaluate)
        # Last resort at the densest level: scroll sideways rather than clip.
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.viewport().installEventFilter(self)
        scroll.installEventFilter(self)
        page = scroll.widget()
        if page is not None:
            page.installEventFilter(self)

    # ------------------------------------------------------------------ state
    @property
    def level(self) -> int:
        return self._level

    def refresh(self) -> None:
        """Re-evaluate soon (after a resize, a show, or a content change)."""
        if not self._timer.isActive():
            self._timer.start(DEBOUNCE_MS)

    # ----------------------------------------------------------------- events
    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        if kind in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            self.refresh()
        elif kind == QtCore.QEvent.Type.LayoutRequest and not self._applying:
            self.refresh()
        return False

    # -------------------------------------------------------------- algorithm
    def _evaluate(self) -> None:
        page = self._scroll.widget()
        if page is None or not self._scroll.isVisible():
            return
        available = self._scroll.viewport().width()
        if available <= 0:
            return
        # A theme change may have replaced the page's style sheet.
        if self._level and _SHEET_BEGIN not in page.styleSheet():
            self._apply(self._level)
        needed = int(page.minimumSizeHint().width())
        self._needed[self._level] = needed
        if self._trial_row is not None:
            # Keep the last stacked row only if the page became narrower.
            row, self._trial_row = self._trial_row, None
            if self._reflowed and self._reflowed[-1][0] is row and needed >= self._reflowed[-1][2]:
                self._unreflow_one()
                row.setProperty(_P_NO_GAIN, True)
                self._timer.start(0)
                return
        if needed > available:
            # Tighten one step, mildest first, then measure again on the next tick:
            # 1) narrow the captions and lists of kept rows, 2) denser text and
            # controls, 3) stack rows that still cannot fit.
            if not self._squeezed and self._squeeze(page, needed):
                pass
            elif self._level < len(DENSITY_LEVELS) - 1:
                self._apply(self._level + 1)
            elif not self._reflow_one(page, available, needed):
                return  # nothing left to tighten: the horizontal scroll bar takes over
            self._timer.start(0)
            return
        # Relax in the reverse order, one step at a time.
        if self._reflowed:
            if self._reflowed[-1][2] + HYSTERESIS_PX <= available:
                self._unreflow_one()
                self._timer.start(0)
        elif self._level > 0:
            previous = self._needed.get(self._level - 1)
            if previous is None or previous + HYSTERESIS_PX <= available:
                self._apply(self._level - 1)
                self._timer.start(0)
        elif self._squeezed and self._squeeze_needed + HYSTERESIS_PX <= available:
            self._unsqueeze()
            self._timer.start(0)

    # ----------------------------------------------------------------- reflow
    @staticmethod
    def _child_layouts(layout: QtWidgets.QLayout):
        """Layouts directly below a layout, through nested layouts and widgets."""
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None:
                continue
            if item.layout() is not None:
                yield item.layout()
                continue
            widget = item.widget()
            if widget is None or widget.isHidden() or isinstance(widget, QtWidgets.QAbstractScrollArea):
                continue
            if widget.layout() is not None:
                yield widget.layout()
            else:  # containers without their own layout, e.g. QTabWidget
                for child in widget.findChildren(
                        QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
                    if not child.isHidden() and child.layout() is not None:
                        yield child.layout()

    @staticmethod
    def _is_row(layout: QtWidgets.QLayout) -> bool:
        if isinstance(layout, QtWidgets.QBoxLayout):
            return layout.count() > 1 and layout.direction() in (
                QtWidgets.QBoxLayout.Direction.LeftToRight, QtWidgets.QBoxLayout.Direction.RightToLeft)
        if isinstance(layout, QtWidgets.QFormLayout):
            return layout.rowWrapPolicy() != QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows
        return False

    def _innermost_overflowing_row(self, layout: QtWidgets.QLayout, limit: int):
        """Deepest reflowable row whose minimum width exceeds ``limit``."""
        if layout is None or bool(layout.property(KEEP_ROW_PROPERTY)):
            return None
        for child in self._child_layouts(layout):
            found = self._innermost_overflowing_row(child, limit)
            if found is not None:
                return found
        if (self._is_row(layout) and not layout.property(_P_NO_GAIN)
                and layout.minimumSize().width() > limit):
            return layout
        return None

    def _widest_row(self, layout: QtWidgets.QLayout, best=None):
        """Widest reflowable row of the page (innermost on equal widths)."""
        if layout is None or bool(layout.property(KEEP_ROW_PROPERTY)):
            return best
        if self._is_row(layout) and not layout.property(_P_NO_GAIN):
            if best is None or layout.minimumSize().width() >= best.minimumSize().width():
                best = layout
        for child in self._child_layouts(layout):
            best = self._widest_row(child, best)
        return best

    def _reflow_one(self, page: QtWidgets.QWidget, available: int, needed: int) -> bool:
        """Stack the innermost row that cannot fit. Returns False when none is left."""
        row = self._innermost_overflowing_row(page.layout(), available - ROW_FIT_SLACK_PX)
        if row is None:
            # No single row is too wide: nested margins push the widest one out.
            row = self._widest_row(page.layout())
            if row is None or row.minimumSize().width() < WIDE_ROW_FRACTION * available:
                return False
        self._trial_row = row
        self._applying = True
        try:
            if isinstance(row, QtWidgets.QBoxLayout):
                self._reflowed.append((row, row.direction(), needed))
                row.setDirection(QtWidgets.QBoxLayout.Direction.TopToBottom)
            else:
                self._reflowed.append((row, row.rowWrapPolicy(), needed))
                row.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
            page.updateGeometry()
        finally:
            self._applying = False
        return True

    def _unreflow_one(self) -> None:
        row, restore, _needed = self._reflowed.pop()
        self._applying = True
        try:
            if isinstance(row, QtWidgets.QBoxLayout):
                row.setDirection(restore)
            else:
                row.setRowWrapPolicy(restore)
        except RuntimeError:
            pass  # the row was deleted with its panel
        finally:
            self._applying = False

    # ---------------------------------------------------------------- squeeze
    def _kept_row_widgets(self, layout: QtWidgets.QLayout, inside: bool = False):
        """Captions and lists inside rows marked to stay horizontal."""
        if layout is None:
            return
        inside = inside or bool(layout.property(KEEP_ROW_PROPERTY))
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is None:
                continue
            if item.layout() is not None:
                yield from self._kept_row_widgets(item.layout(), inside)
                continue
            widget = item.widget()
            if widget is None or widget.isHidden():
                continue
            if inside and isinstance(widget, (QtWidgets.QLabel, QtWidgets.QAbstractItemView)):
                yield widget
            elif widget.layout() is not None and not isinstance(widget, QtWidgets.QAbstractScrollArea):
                yield from self._kept_row_widgets(widget.layout(), inside)

    def _squeeze(self, page: QtWidgets.QWidget, needed: int) -> bool:
        """Let captions and lists in kept rows get narrower. False if already done."""
        if self._squeezed:
            return False
        self._applying = True
        try:
            for widget in self._kept_row_widgets(page.layout()):
                is_view = isinstance(widget, QtWidgets.QAbstractItemView)
                floor = SQUEEZED_VIEW_WIDTH if is_view else SQUEEZED_LABEL_WIDTH
                current = widget.minimumWidth()
                if 0 < current <= floor:
                    continue
                self._squeezed.append((widget, current, widget.toolTip()))
                if not is_view and not widget.toolTip():
                    widget.setToolTip(widget.text())  # the full caption stays readable
                widget.setMinimumWidth(floor)
            page.updateGeometry()
        finally:
            self._applying = False
        self._squeeze_needed = needed
        return bool(self._squeezed)

    def _unsqueeze(self) -> None:
        self._applying = True
        try:
            for widget, minimum, tooltip in self._squeezed:
                try:
                    widget.setMinimumWidth(minimum)
                    widget.setToolTip(tooltip)
                except RuntimeError:
                    pass
        finally:
            self._squeezed = []
            self._applying = False

    def _apply(self, level: int) -> None:
        page = self._scroll.widget()
        if page is None:
            return
        self._applying = True
        try:
            scale = DENSITY_LEVELS[level][0]
            own_rules = _strip_density_sheet(page.styleSheet())
            block = _density_sheet(level)
            page.setStyleSheet(f"{own_rules}\n{block}".strip() if block else own_rules)
            self._scale_geometry(page, scale)
            changed = level != self._level
            self._level = level
            page.updateGeometry()
        finally:
            self._applying = False
        if changed:
            self.levelChanged.emit(level)

    # --------------------------------------------------------------- geometry
    @staticmethod
    def _remembered(obj: QtCore.QObject, name: str, current, applied_name: str):
        """Return the object's normal value for a property we rescale.

        The normal value is stored the first time. If the program has changed
        the live value since we last set it, that new value becomes the normal
        one.
        """
        stored = obj.property(name)
        applied = obj.property(applied_name)
        if stored is None or (applied is not None and applied != current):
            obj.setProperty(name, current)
            stored = current
        return stored

    def _scale_geometry(self, page: QtWidgets.QWidget, scale: float) -> None:
        """Scale explicit widths, layout spacing and margins from their normal values."""
        for layout in [page.layout(), *page.findChildren(QtWidgets.QLayout)]:
            if layout is None:
                continue
            spacing = layout.spacing()
            if spacing > 0 or layout.property(_P_SPACING) is not None:
                normal = int(self._remembered(layout, _P_SPACING, spacing, _P_SPACING + "_applied"))
                if normal > 0:
                    value = max(1, round(normal * scale))
                    layout.setSpacing(value)
                    layout.setProperty(_P_SPACING + "_applied", value)
            margins = layout.contentsMargins()
            current = [margins.left(), margins.top(), margins.right(), margins.bottom()]
            normal_margins = self._remembered(layout, _P_MARGINS, current, _P_MARGINS + "_applied")
            if any(normal_margins):
                value = [round(m * scale) for m in normal_margins]
                layout.setContentsMargins(*value)
                layout.setProperty(_P_MARGINS + "_applied", value)

        for widget in page.findChildren(QtWidgets.QWidget):
            if widget.isWindow():
                continue
            # Numeric fields keep their width: a value must never be cut off.
            if isinstance(widget, QtWidgets.QAbstractSpinBox) or isinstance(
                    widget.parentWidget(), QtWidgets.QAbstractSpinBox):
                continue
            minimum = widget.minimumWidth()
            if minimum > 0 or widget.property(_P_MIN_W) is not None:
                normal = int(self._remembered(widget, _P_MIN_W, minimum, _P_MIN_W + "_applied"))
                if normal > 0:
                    value = max(1, round(normal * scale))
                    # Lower the minimum before the maximum so a fixed width
                    # never passes through an invalid min > max state.
                    maximum = widget.maximumWidth()
                    if maximum < _QWIDGETSIZE_MAX or widget.property(_P_MAX_W) is not None:
                        normal_max = int(self._remembered(widget, _P_MAX_W, maximum, _P_MAX_W + "_applied"))
                        max_value = max(value, round(normal_max * scale)) if normal_max < _QWIDGETSIZE_MAX else normal_max
                        if max_value >= widget.minimumWidth():
                            widget.setMaximumWidth(max_value)
                            widget.setMinimumWidth(value)
                        else:
                            widget.setMinimumWidth(value)
                            widget.setMaximumWidth(max_value)
                        widget.setProperty(_P_MAX_W + "_applied", max_value)
                    else:
                        widget.setMinimumWidth(value)
                    widget.setProperty(_P_MIN_W + "_applied", value)
                    continue
            maximum = widget.maximumWidth()
            if maximum < _QWIDGETSIZE_MAX or widget.property(_P_MAX_W) is not None:
                normal_max = int(self._remembered(widget, _P_MAX_W, maximum, _P_MAX_W + "_applied"))
                if 0 < normal_max < _QWIDGETSIZE_MAX:
                    value = max(1, round(normal_max * scale))
                    widget.setMaximumWidth(value)
                    widget.setProperty(_P_MAX_W + "_applied", value)


def install_adaptive_density(scroll: QtWidgets.QScrollArea) -> Optional[AdaptiveDensity]:
    """Attach an ``AdaptiveDensity`` controller to a scroll area (once)."""
    existing = scroll.findChild(AdaptiveDensity)
    if existing is not None:
        return existing
    return AdaptiveDensity(scroll)
