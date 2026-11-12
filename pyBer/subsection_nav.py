# subsection_nav.py
"""Vertical subsection navigator for long settings panels.

A long settings panel built from many stacked group boxes forces the user to
scroll through every control to reach the one they want. ``VerticalSubsections``
replaces that stack with a narrow vertical rail of rotated subsection names
on the left and a single page on the right: only the selected subsection is shown, so
the panel height is bounded by its largest subsection instead of the sum of all
of them.

Design notes
------------
* Pages are plain widgets (typically the existing ``QGroupBox`` cards). Their
  titles are cleared because the rail already names them; the card styling is
  kept through the ``subsectionPage`` dynamic property (see ``styles.py``).
* The page area follows the *current* page height. Hidden pages get an
  ``Ignored`` vertical size policy so a tall hidden page never stretches the
  panel or pushes the action buttons below it down.
* Widgets on hidden pages keep their state and signal connections; switching
  subsection never recreates or resets a control.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


class VerticalTabButton(QtWidgets.QAbstractButton):
    """Checkable tab whose label is rotated 90 degrees (reads bottom to top).

    Rotating the label makes the rail as narrow as one line of text (about
    30 px) instead of as wide as the longest subsection name. The button
    paints itself from its palette, so it follows the dark/light app theme
    (``WindowText`` for the label, ``Highlight`` for the accent).
    """

    PAD_ALONG = 12   # px of padding before/after the label, along the rail
    PAD_ACROSS = 7   # px of padding on each side of the text line
    BAR = 3          # px width of the accent bar of the selected tab

    def __init__(self, text: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        font = self.font()
        font.setWeight(QtGui.QFont.Weight.DemiBold)
        self.setFont(font)

    def _bold_font(self) -> QtGui.QFont:
        """Font of the selected tab; also used for sizing so selection never resizes."""
        font = QtGui.QFont(self.font())
        font.setWeight(QtGui.QFont.Weight.Bold)
        return font

    def sizeHint(self) -> QtCore.QSize:
        """Width = one text line (+padding, accent bar); height = text length."""
        metrics = QtGui.QFontMetrics(self._bold_font())
        across = metrics.height() + 2 * self.PAD_ACROSS + self.BAR
        along = metrics.horizontalAdvance(self.text()) + 2 * self.PAD_ALONG
        return QtCore.QSize(across, along)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Draw hover/selected fill, accent bar and the rotated label."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        accent = pal.color(QtGui.QPalette.ColorRole.Highlight)
        text = pal.color(QtGui.QPalette.ColorRole.WindowText)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        checked, hovered = self.isChecked(), self.underMouse()

        # Background: soft accent wash when selected, faint text wash on hover.
        if checked or hovered:
            fill = QtGui.QColor(accent if checked else text)
            fill.setAlphaF(0.22 if checked else 0.08)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 7, 7)
        if checked:
            bar = QtCore.QRectF(rect.left(), rect.top() + 6, self.BAR, rect.height() - 12)
            painter.setBrush(accent)
            painter.drawRoundedRect(bar, 1.5, 1.5)
        if self.hasFocus() and self.focusPolicy() != QtCore.Qt.FocusPolicy.NoFocus:
            ring = QtGui.QColor(accent)
            painter.setPen(QtGui.QPen(ring, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 7, 7)

        # Label: rotate -90 deg about the bottom-left so text reads upward.
        label = QtGui.QColor(text)
        if not checked:
            label.setAlphaF(0.95 if hovered else 0.68)
        painter.setPen(label)
        painter.setFont(self._bold_font() if checked else self.font())
        painter.translate(0, self.height())
        painter.rotate(-90)
        text_rect = QtCore.QRectF(0, self.BAR, self.height(), self.width() - self.BAR)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class VerticalSubsections(QtWidgets.QWidget):
    """Rail of subsection buttons (left) driving a page stack (right).

    Signals
    -------
    currentChanged(int)
        Emitted with the new page index whenever the visible subsection changes.
    """

    currentChanged = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Build the empty navigator.

        Parameters
        ----------
        parent : QWidget, optional
            Qt parent widget.
        """
        super().__init__(parent)
        self.setObjectName("verticalSubsections")

        # -- Left rail: one checkable button per subsection, exclusive group --
        self._rail = QtWidgets.QFrame(self)
        self._rail.setProperty("class", "subsectionRail")
        # Width follows the rotated tabs (one text line), see VerticalTabButton.
        self._rail.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        self._rail_layout = QtWidgets.QVBoxLayout(self._rail)
        self._rail_layout.setContentsMargins(3, 4, 3, 4)
        self._rail_layout.setSpacing(1)
        self._rail_layout.addStretch(1)  # buttons are inserted above this stretch
        self._buttons = QtWidgets.QButtonGroup(self)
        self._buttons.setExclusive(True)
        self._buttons.idClicked.connect(self.setCurrentIndex)

        # -- Right: page stack showing exactly one subsection --
        self._stack = QtWidgets.QStackedWidget(self)
        self._stack.currentChanged.connect(self._on_stack_changed)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._rail, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._stack, 1, QtCore.Qt.AlignmentFlag.AlignTop)

        self._titles: List[str] = []

    # ------------------------------------------------------------------ API
    def addSection(self, title: str, page: QtWidgets.QWidget, tooltip: str = "") -> int:
        """Append a subsection and return its index.

        If ``page`` is a ``QGroupBox`` its title is cleared (the rail already
        shows it) and it is flagged so the stylesheet drops the title margin.
        """
        if isinstance(page, QtWidgets.QGroupBox):
            page.setTitle("")
            page.setProperty("subsectionPage", True)
        index = self._stack.addWidget(page)
        self._titles.append(str(title))

        # Rotated label keeps the rail one text line wide.
        button = VerticalTabButton(str(title), self._rail)
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        button.setToolTip(tooltip or str(title))
        button.setAccessibleName(f"{title} subsection")
        self._buttons.addButton(button, index)
        # Insert before the trailing stretch so buttons stay packed at the top.
        self._rail_layout.insertWidget(self._rail_layout.count() - 1, button)

        if index == 0:
            button.setChecked(True)
        self._sync_page_policies()
        return index

    def count(self) -> int:
        """Number of subsections."""
        return self._stack.count()

    def currentIndex(self) -> int:
        """Index of the visible subsection (-1 when empty)."""
        return self._stack.currentIndex()

    def currentWidget(self) -> Optional[QtWidgets.QWidget]:
        """The visible page widget."""
        return self._stack.currentWidget()

    def widget(self, index: int) -> Optional[QtWidgets.QWidget]:
        """Page widget at ``index``."""
        return self._stack.widget(index)

    def indexOf(self, page: QtWidgets.QWidget) -> int:
        """Index of ``page`` (-1 if absent)."""
        return self._stack.indexOf(page)

    def sectionTitle(self, index: int) -> str:
        """Rail label of subsection ``index``."""
        return self._titles[index] if 0 <= index < len(self._titles) else ""

    def indexOfTitle(self, title: str) -> int:
        """Index of the subsection labelled ``title`` (case-insensitive), or -1."""
        wanted = str(title or "").strip().lower()
        for i, existing in enumerate(self._titles):
            if existing.lower() == wanted:
                return i
        return -1

    def button(self, index: int) -> Optional[QtWidgets.QAbstractButton]:
        """Rail button of subsection ``index`` (useful for tours and tests)."""
        return self._buttons.button(index)

    def setCurrentIndex(self, index: int) -> None:
        """Show subsection ``index`` (out-of-range values are ignored)."""
        if not 0 <= int(index) < self._stack.count():
            return
        self._stack.setCurrentIndex(int(index))
        button = self._buttons.button(int(index))
        if button is not None and not button.isChecked():
            button.setChecked(True)

    def revealWidget(self, child: QtWidgets.QWidget) -> bool:
        """Switch to the subsection containing ``child``; True when found."""
        for i in range(self._stack.count()):
            page = self._stack.widget(i)
            if page is child or (page is not None and page.isAncestorOf(child)):
                self.setCurrentIndex(i)
                return True
        return False

    # ------------------------------------------------------------ internals
    def _on_stack_changed(self, index: int) -> None:
        """Resize to the newly visible page, then notify listeners."""
        self._sync_page_policies()
        self.currentChanged.emit(int(index))

    def _sync_page_policies(self) -> None:
        """Let only the current page contribute to the stack's size hints.

        QStackedWidget normally reports the largest page's size hint and
        minimum size. Giving hidden pages an ``Ignored`` policy in both
        directions makes the stack hug the visible page: a tall hidden page no
        longer adds empty space and a wide one no longer forces the panel wider
        than the drawer (which would clip the visible form on the right).
        """
        current = self._stack.currentIndex()
        preferred = QtWidgets.QSizePolicy.Policy.Preferred
        ignored = QtWidgets.QSizePolicy.Policy.Ignored
        for i in range(self._stack.count()):
            page = self._stack.widget(i)
            if page is None:
                continue
            policy = preferred if i == current else ignored
            page.setSizePolicy(policy, policy)
        self._stack.updateGeometry()
        self.updateGeometry()
