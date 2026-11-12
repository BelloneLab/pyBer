# subsection_nav.py
"""Vertical subsection navigator for long settings panels.

A long settings panel built from many stacked group boxes forces the user to
scroll through every control to reach the one they want. ``VerticalSubsections``
replaces that stack with a compact vertical rail of subsection names on the
left and a single page on the right: only the selected subsection is shown, so
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

from PySide6 import QtCore, QtWidgets


class VerticalSubsections(QtWidgets.QWidget):
    """Rail of subsection buttons (left) driving a page stack (right).

    Signals
    -------
    currentChanged(int)
        Emitted with the new page index whenever the visible subsection changes.
    """

    currentChanged = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, rail_width: int = 104) -> None:
        """Build the empty navigator.

        Parameters
        ----------
        parent : QWidget, optional
            Qt parent widget.
        rail_width : int
            Fixed width (px) of the left navigation rail. Kept narrow so the
            page keeps most of the panel width for forms.
        """
        super().__init__(parent)
        self.setObjectName("verticalSubsections")

        # -- Left rail: one checkable button per subsection, exclusive group --
        self._rail = QtWidgets.QFrame(self)
        self._rail.setProperty("class", "subsectionRail")
        self._rail.setFixedWidth(int(rail_width))
        self._rail_layout = QtWidgets.QVBoxLayout(self._rail)
        self._rail_layout.setContentsMargins(4, 4, 4, 4)
        self._rail_layout.setSpacing(2)
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

        # QPushButton (not QToolButton) so the stylesheet can left-align text.
        button = QtWidgets.QPushButton(str(title), self._rail)
        button.setCheckable(True)
        button.setProperty("class", "subsectionTab")
        button.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)
        button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
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
