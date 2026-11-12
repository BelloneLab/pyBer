"""Recording groups: model, list outlines, group chips and a detachable list window.

A *group* is an experimental label (genotype, treatment, cohort...) with a
user-chosen name and color. Recordings are attributed to at most one group by
their file id (the processed file stem, identical to PSTH row labels), so the
assignment survives reordering, reloading and project restoration.

Nothing here touches signals or analysis results: the model only stores names,
colors and memberships and emits ``changed`` so the panel can redraw.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------------
# Palette: validated categorical order (dataviz reference palette).
# Dark steps are used on the dark plot themes, light steps on light themes.
# Assigned in fixed order and never cycled to a generated hue.
# ---------------------------------------------------------------------------
GROUP_COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                     "#d55181", "#008300", "#9085e9", "#e66767"]
GROUP_COLORS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                      "#e87ba4", "#008300", "#6250d6", "#e34948"]
UNGROUPED_COLOR = "#8a94a6"
UNGROUPED_LABEL = "Ungrouped"
OUTLINE_WIDTH = 1.6     # px, recording outline in the file list
TAG_DOT = 8             # px, colored dot drawn beside the group name
ROW_HEIGHT = 28         # px, identical in both lists so rows stay line-aligned
GRIP_W = 18             # px, drag grip at the left of each row
CLOSE_W = 24            # px, remove cross at the right of each row


def default_group_name(index: int) -> str:
    """Group A, Group B, ..., Group Z, Group 27, ..."""
    return f"Group {chr(ord('A') + index)}" if index < 26 else f"Group {index + 1}"


class RecordingGroups(QtCore.QObject):
    """Ordered groups and file-id assignments, serializable to JSON."""

    changed = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._groups: List[Dict[str, str]] = []
        self._assign: Dict[str, str] = {}
        self._next_id = 1
        self.light_theme = False

    # -- queries -------------------------------------------------------------
    def groups(self) -> List[Dict[str, str]]:
        """Copies of the groups in display order."""
        return [dict(group) for group in self._groups]

    def group(self, group_id: Optional[str]) -> Optional[Dict[str, str]]:
        for group in self._groups:
            if group["id"] == group_id:
                return dict(group)
        return None

    def group_id_of(self, file_id: str) -> Optional[str]:
        group_id = self._assign.get(str(file_id))
        return group_id if self.group(group_id) is not None else None

    def group_of(self, file_id: str) -> Optional[Dict[str, str]]:
        return self.group(self.group_id_of(file_id))

    def members(self, group_id: Optional[str], file_ids: Optional[Iterable[str]] = None) -> List[str]:
        """Members in ``file_ids`` order (or assignment order); None = ungrouped."""
        if file_ids is None:
            file_ids = list(self._assign) if group_id is not None else []
        return [fid for fid in file_ids if self.group_id_of(fid) == group_id]

    def assignments(self) -> Dict[str, str]:
        return {fid: gid for fid, gid in self._assign.items() if self.group(gid) is not None}

    def __len__(self) -> int:
        return len(self._groups)

    # -- edits ---------------------------------------------------------------
    def _palette(self) -> List[str]:
        return GROUP_COLORS_LIGHT if self.light_theme else GROUP_COLORS_DARK

    def next_color(self) -> str:
        """First palette slot not used yet, so colors follow the entity."""
        used = {group["color"].lower() for group in self._groups}
        for color in self._palette():
            if color.lower() not in used:
                return color
        return self._palette()[len(self._groups) % len(self._palette())]

    def add_group(self, name: Optional[str] = None, color: Optional[str] = None) -> str:
        group_id = f"g{self._next_id}"
        self._next_id += 1
        names = {group["name"] for group in self._groups}
        if not name:
            index = len(self._groups)
            name = default_group_name(index)
            while name in names:
                index += 1
                name = default_group_name(index)
        qcolor = QtGui.QColor(color or self.next_color())
        if not qcolor.isValid():
            qcolor = QtGui.QColor(self.next_color())
        self._groups.append({"id": group_id, "name": str(name).strip() or group_id,
                             "color": qcolor.name()})
        self.changed.emit()
        return group_id

    def update_group(self, group_id: str, name: Optional[str] = None, color: Optional[str] = None) -> bool:
        for group in self._groups:
            if group["id"] != group_id:
                continue
            changed = False
            if name is not None and str(name).strip() and str(name).strip() != group["name"]:
                group["name"] = str(name).strip()
                changed = True
            if color is not None and QtGui.QColor(color).isValid():
                hex_color = QtGui.QColor(color).name()
                if hex_color != group["color"]:
                    group["color"] = hex_color
                    changed = True
            if changed:
                self.changed.emit()
            return changed
        return False

    def remove_group(self, group_id: str) -> None:
        before = len(self._groups)
        self._groups = [group for group in self._groups if group["id"] != group_id]
        self._assign = {fid: gid for fid, gid in self._assign.items() if gid != group_id}
        if len(self._groups) != before:
            self.changed.emit()

    def move_group(self, group_id: str, offset: int) -> None:
        index = next((i for i, g in enumerate(self._groups) if g["id"] == group_id), -1)
        target = index + int(offset)
        if index < 0 or not 0 <= target < len(self._groups):
            return
        self._groups.insert(target, self._groups.pop(index))
        self.changed.emit()

    def assign(self, file_ids: Iterable[str], group_id: Optional[str]) -> int:
        """Attribute recordings to a group (None or '' removes them). Returns count."""
        if group_id and self.group(group_id) is None:
            return 0
        count = 0
        for fid in file_ids:
            fid = str(fid)
            if not fid:
                continue
            if group_id:
                if self._assign.get(fid) != group_id:
                    self._assign[fid] = group_id
                    count += 1
            elif fid in self._assign:
                del self._assign[fid]
                count += 1
        if count:
            self.changed.emit()
        return count

    def clear(self) -> None:
        if self._groups or self._assign:
            self._groups, self._assign = [], {}
            self.changed.emit()

    # -- persistence ---------------------------------------------------------
    def to_state(self) -> Dict[str, object]:
        return {"groups": copy.deepcopy(self._groups), "assignments": self.assignments(),
                "next_id": self._next_id}

    def from_state(self, state: object, emit: bool = True) -> None:
        """Restore validated state; malformed entries are ignored, never guessed."""
        groups: List[Dict[str, str]] = []
        assign: Dict[str, str] = {}
        next_id = 1
        if isinstance(state, dict):
            seen = set()
            for raw in state.get("groups", []) or []:
                if not isinstance(raw, dict):
                    continue
                gid = str(raw.get("id", "")).strip()
                name = str(raw.get("name", "")).strip()
                color = QtGui.QColor(str(raw.get("color", "")))
                if not gid or gid in seen or not name:
                    continue
                seen.add(gid)
                groups.append({"id": gid, "name": name,
                               "color": color.name() if color.isValid() else UNGROUPED_COLOR})
                digits = gid[1:] if gid.startswith("g") else ""
                if digits.isdigit():
                    next_id = max(next_id, int(digits) + 1)
            raw_assign = state.get("assignments", {}) or {}
            if isinstance(raw_assign, dict):
                assign = {str(fid): str(gid) for fid, gid in raw_assign.items() if str(gid) in seen}
            try:
                next_id = max(next_id, int(state.get("next_id", next_id)))
            except (TypeError, ValueError):
                pass
        changed = groups != self._groups or assign != self._assign
        self._groups, self._assign, self._next_id = groups, assign, next_id
        if changed and emit:
            self.changed.emit()


# ---------------------------------------------------------------------------
# Painting helpers
# ---------------------------------------------------------------------------
def color_dot_icon(color: str, size: int = 12) -> QtGui.QIcon:
    """A crisp filled circle icon (rendered at 2x for high-DPI screens)."""
    pixmap = QtGui.QPixmap(size * 2, size * 2)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawEllipse(QtCore.QRectF(2, 2, size * 2 - 4, size * 2 - 4))
    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return QtGui.QIcon(pixmap)


class FileRowDelegate(QtWidgets.QStyledItemDelegate):
    """One file row: drag grip, name, pairing note, group outline/tag, remove cross.

    * ``resolver(index)`` -> ``(color, name)`` of the row's group, or None.
    * ``note_resolver(index)`` -> ``(text, color)`` shown before the tag, or None
      (used to flag rows whose behavior pairing is not the same line).
    * Clicking the cross emits ``removeRequested(row)``; the click never
      toggles the selection.
    Rows have a fixed height so the two lists stay line-aligned.
    """

    removeRequested = QtCore.Signal(int)

    def __init__(self, resolver: Callable[[QtCore.QModelIndex], Optional[Tuple[str, str]]],
                 parent: Optional[QtCore.QObject] = None,
                 note_resolver: Optional[Callable[[QtCore.QModelIndex], Optional[Tuple[str, str]]]] = None) -> None:
        super().__init__(parent)
        self.resolver = resolver
        self.note_resolver = note_resolver

    # -- geometry ----------------------------------------------------------------
    @staticmethod
    def close_rect(rect: QtCore.QRect) -> QtCore.QRectF:
        r = QtCore.QRectF(rect)
        return QtCore.QRectF(r.right() - CLOSE_W + 1, r.top(), CLOSE_W - 2, r.height())

    @staticmethod
    def grip_rect(rect: QtCore.QRect) -> QtCore.QRectF:
        r = QtCore.QRectF(rect)
        return QtCore.QRectF(r.left(), r.top(), GRIP_W, r.height())

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QtCore.QSize(size.width() + GRIP_W + CLOSE_W, ROW_HEIGHT)

    # -- interaction ---------------------------------------------------------------
    def editorEvent(self, event, model, option, index):
        kind = event.type()
        if kind in (QtCore.QEvent.Type.MouseButtonPress, QtCore.QEvent.Type.MouseButtonRelease,
                    QtCore.QEvent.Type.MouseButtonDblClick):
            if event.button() == QtCore.Qt.MouseButton.LeftButton and \
                    self.close_rect(option.rect).contains(event.position()):
                if kind == QtCore.QEvent.Type.MouseButtonRelease:
                    self.removeRequested.emit(index.row())
                return True
        return super().editorEvent(event, model, option, index)

    # -- painting ------------------------------------------------------------------
    def paint(self, painter: QtGui.QPainter, option, index) -> None:
        info = self.resolver(index)
        note = self.note_resolver(index) if self.note_resolver is not None else None
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else QtWidgets.QApplication.style()
        metrics = QtGui.QFontMetrics(opt.font)
        tag_width = note_width = 0
        tag_name = note_text = ""
        if info is not None:
            tag_name = metrics.elidedText(info[1], QtCore.Qt.TextElideMode.ElideRight, 110)
            tag_width = metrics.horizontalAdvance(tag_name) + TAG_DOT + 14
        if note is not None:
            note_text = metrics.elidedText(note[0], QtCore.Qt.TextElideMode.ElideRight, 130)
            note_width = metrics.horizontalAdvance(note_text) + 10
        # Background (hover/selection) across the whole row, grip included.
        style.drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_PanelItemViewItem, opt, painter, widget)
        text_opt = QtWidgets.QStyleOptionViewItem(opt)
        text_opt.rect = opt.rect.adjusted(GRIP_W, 0, -(CLOSE_W + tag_width + note_width), 0)
        text_opt.text = metrics.elidedText(opt.text, QtCore.Qt.TextElideMode.ElideMiddle,
                                           max(20, text_opt.rect.width() - 8))
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, text_opt, painter, widget)
        selected = bool(opt.state & QtWidgets.QStyle.StateFlag.State_Selected)
        hovered = bool(opt.state & QtWidgets.QStyle.StateFlag.State_MouseOver)
        role = QtGui.QPalette.ColorRole.HighlightedText if selected else QtGui.QPalette.ColorRole.Text
        ink = QtGui.QColor(opt.palette.color(role))
        rect = QtCore.QRectF(opt.rect)
        painter.save()
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            # Drag grip: two columns of three dots.
            grip = QtGui.QColor(ink)
            grip.setAlpha(150 if hovered or selected else 85)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(grip)
            cx, cy = rect.left() + GRIP_W / 2.0, rect.center().y()
            for dx in (-2.5, 2.5):
                for dy in (-4.5, 0.0, 4.5):
                    painter.drawEllipse(QtCore.QPointF(cx + dx, cy + dy), 1.3, 1.3)
            if info is not None:
                color = QtGui.QColor(info[0])
                painter.setPen(QtGui.QPen(color, OUTLINE_WIDTH))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                inset = OUTLINE_WIDTH / 2.0 + 1.0
                painter.drawRoundedRect(rect.adjusted(inset, inset, -inset, -inset), 5, 5)
            painter.setFont(opt.font)
            x = rect.right() - CLOSE_W - tag_width - note_width
            if note is not None:
                painter.setPen(QtGui.QColor(note[1]))
                painter.drawText(QtCore.QRectF(x + 4, rect.top(), note_width, rect.height()),
                                 int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                                 note_text)
                x += note_width
            if info is not None:
                dot = QtCore.QRectF(x + 4, rect.center().y() - TAG_DOT / 2, TAG_DOT, TAG_DOT)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(QtGui.QColor(info[0]))
                painter.drawEllipse(dot)
                painter.setPen(ink)
                painter.drawText(QtCore.QRectF(dot.right() + 5, rect.top(), tag_width, rect.height()),
                                 int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                                 tag_name)
            # Remove cross: quiet until the row is hovered or selected.
            cross = QtGui.QColor(ink)
            cross.setAlpha(230 if hovered else 140 if selected else 70)
            box = self.close_rect(opt.rect)
            c = box.center()
            painter.setPen(QtGui.QPen(cross, 1.4, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap))
            painter.drawLine(QtCore.QPointF(c.x() - 3.5, c.y() - 3.5), QtCore.QPointF(c.x() + 3.5, c.y() + 3.5))
            painter.drawLine(QtCore.QPointF(c.x() - 3.5, c.y() + 3.5), QtCore.QPointF(c.x() + 3.5, c.y() - 3.5))
        finally:
            painter.restore()


# Backward-compatible name.
GroupOutlineDelegate = FileRowDelegate


# ---------------------------------------------------------------------------
# Wrapping layout for group chips
# ---------------------------------------------------------------------------
class FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps rows (Qt's classic flow layout)."""

    def __init__(self, parent=None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QtCore.QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        spacing = self.spacing()
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > area.right() + 1 and line_height > 0:
                x = area.x()
                y += line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class GroupChipBar(QtWidgets.QWidget):
    """Clickable group chips: a click attributes the selected recordings.

    Right-click a chip to rename it, change its color, reorder or delete it.
    """

    assignRequested = QtCore.Signal(str)   # group id, '' = remove from group
    statusMessage = QtCore.Signal(str)

    def __init__(self, model: RecordingGroups, count_provider: Callable[[Optional[str]], int],
                 parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.count_provider = count_provider
        self.setObjectName("groupChipBar")
        self._layout = FlowLayout(self, spacing=6)
        self._buttons: List[QtWidgets.QAbstractButton] = []
        self.setToolTip("Select recordings, then click a group to attribute them.\n"
                        "Right-click a group to rename it, change its color or delete it.")
        model.changed.connect(self.rebuild)
        self.rebuild()

    def _chip(self, text: str, color: Optional[str], tooltip: str) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setText(text)
        button.setToolTip(tooltip)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setAutoRaise(False)
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if color:
            button.setIcon(color_dot_icon(color))
            button.setIconSize(QtCore.QSize(10, 10))
            button.setStyleSheet(
                "QToolButton { border: 1px solid %s; border-radius: 11px; padding: 2px 9px 2px 6px; }"
                "QToolButton:hover { background: %s; }" % (color, QtGui.QColor(color).name() + "33"))
        else:
            button.setStyleSheet("QToolButton { border-radius: 11px; padding: 2px 9px; }")
        button.setFixedHeight(24)
        return button

    def rebuild(self) -> None:
        for button in self._buttons:
            self._layout.removeWidget(button)
            button.hide()            # hidden at once; deletion happens later
            button.setParent(None)
            button.deleteLater()
        self._buttons = []
        for group in self.model.groups():
            count = self.count_provider(group["id"])
            chip = self._chip(f"{group['name']}  {count}", group["color"],
                              f"Attribute selected recordings to {group['name']} ({count} assigned).\n"
                              "Right-click for rename, color, order and delete.")
            chip.clicked.connect(lambda _=False, gid=group["id"]: self.assignRequested.emit(gid))
            chip.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            chip.customContextMenuRequested.connect(
                lambda pos, gid=group["id"], btn=chip: self._chip_menu(gid, btn.mapToGlobal(pos)))
            self._add(chip)
        ungroup = self._chip("Ungroup", None, "Remove the selected recordings from their group.")
        ungroup.clicked.connect(lambda: self.assignRequested.emit(""))
        ungroup.setEnabled(len(self.model) > 0)
        self._add(ungroup)
        new = self._chip("+ New group", None, "Create a group; the selected recordings join it.")
        new.clicked.connect(self.create_group_interactive)
        self._add(new)

    def _add(self, button: QtWidgets.QAbstractButton) -> None:
        # Polish first so the style-sheet padding is part of the size hint;
        # otherwise the flow layout places chips on top of each other.
        button.ensurePolished()
        button.adjustSize()
        self._layout.addWidget(button)
        self._buttons.append(button)
        button.show()
        self._layout.invalidate()
        self.updateGeometry()

    def create_group_interactive(self) -> Optional[str]:
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New group", "Group label (e.g. WT, Het, Saline):",
            text=default_group_name(len(self.model)))
        if not ok or not name.strip():
            return None
        group_id = self.model.add_group(name.strip())
        self.assignRequested.emit(group_id)
        return group_id

    def _chip_menu(self, group_id: str, global_pos: QtCore.QPoint) -> None:
        group = self.model.group(group_id)
        if group is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Attribute selected recordings", lambda: self.assignRequested.emit(group_id))
        menu.addSeparator()
        menu.addAction("Rename...", lambda: rename_group_interactive(self, self.model, group_id))
        menu.addAction("Change color...", lambda: recolor_group_interactive(self, self.model, group_id))
        menu.addAction("Move left", lambda: self.model.move_group(group_id, -1))
        menu.addAction("Move right", lambda: self.model.move_group(group_id, 1))
        menu.addSeparator()
        menu.addAction(f"Delete '{group['name']}'", lambda: delete_group_interactive(self, self.model, group_id))
        menu.exec(global_pos)


def rename_group_interactive(parent, model: RecordingGroups, group_id: str) -> None:
    group = model.group(group_id)
    if group is None:
        return
    name, ok = QtWidgets.QInputDialog.getText(parent, "Rename group", "Group label:", text=group["name"])
    if ok and name.strip():
        model.update_group(group_id, name=name.strip())


def recolor_group_interactive(parent, model: RecordingGroups, group_id: str) -> None:
    group = model.group(group_id)
    if group is None:
        return
    color = QtWidgets.QColorDialog.getColor(QtGui.QColor(group["color"]), parent,
                                            f"Color for {group['name']}")
    if color.isValid():
        model.update_group(group_id, color=color.name())


def delete_group_interactive(parent, model: RecordingGroups, group_id: str) -> None:
    group = model.group(group_id)
    if group is None:
        return
    answer = QtWidgets.QMessageBox.question(
        parent, "Delete group",
        f"Delete group '{group['name']}'? Its recordings become ungrouped; no data is removed.")
    if answer == QtWidgets.QMessageBox.StandardButton.Yes:
        model.remove_group(group_id)


def build_assign_menu(menu: QtWidgets.QMenu, model: RecordingGroups, assign: Callable[[Optional[str]], None],
                      create: Callable[[], None]) -> None:
    """Populate a list context menu with group attribution actions."""
    submenu = menu.addMenu("Attribute to group")
    for group in model.groups():
        action = submenu.addAction(color_dot_icon(group["color"]), group["name"])
        action.triggered.connect(lambda _=False, gid=group["id"]: assign(gid))
    if model.groups():
        submenu.addSeparator()
    submenu.addAction("New group from selection...", create)
    remove = menu.addAction("Remove from group")
    remove.triggered.connect(lambda: assign(None))
    remove.setEnabled(len(model) > 0)


# ---------------------------------------------------------------------------
# Detachable host: the same widget lives in the drawer or in its own window
# ---------------------------------------------------------------------------
class DetachableSection(QtWidgets.QWidget):
    """Wrap ``content`` so it can move between its drawer slot and a window.

    The content widget is reparented, never recreated, so every existing
    signal connection (drag-reorder, selection, drop targets) keeps working.
    Closing the window docks the content back automatically.
    """

    detachedChanged = QtCore.Signal(bool)

    def __init__(self, content: QtWidgets.QWidget, title: str, settings: Optional[QtCore.QSettings] = None,
                 settings_key: str = "recordings_window_geometry", parent=None) -> None:
        super().__init__(parent)
        self.content = content
        self.title = title
        self.settings = settings
        self.settings_key = settings_key
        self.window_widget: Optional[QtWidgets.QWidget] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("detachableSectionTitle")
        font = self.title_label.font()
        font.setBold(True)
        self.title_label.setFont(font)
        header.addWidget(self.title_label, 1)
        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setText("Open in window")
        self.btn_toggle.setToolTip("Detach the recordings list into its own resizable window "
                                   "(longer list, group editing). Close the window to dock it back.")
        self.btn_toggle.clicked.connect(self.toggle)
        header.addWidget(self.btn_toggle, 0)
        layout.addLayout(header)
        self.placeholder = QtWidgets.QFrame()
        placeholder_layout = QtWidgets.QHBoxLayout(self.placeholder)
        placeholder_layout.setContentsMargins(8, 10, 8, 10)
        note = QtWidgets.QLabel("The recordings list is open in a separate window.")
        note.setProperty("class", "hint")
        note.setWordWrap(True)
        placeholder_layout.addWidget(note, 1)
        show = QtWidgets.QPushButton("Show window")
        show.setProperty("class", "compactSmall")
        show.clicked.connect(self._raise_window)
        placeholder_layout.addWidget(show)
        self.placeholder.hide()
        layout.addWidget(self.placeholder)
        layout.addWidget(content, 1)

    def is_detached(self) -> bool:
        return self.window_widget is not None and self.content.parent() is self.window_widget

    def toggle(self) -> None:
        if self.is_detached():
            self.attach()
        else:
            self.detach()

    def _ensure_window(self) -> QtWidgets.QWidget:
        if self.window_widget is None:
            window = _SectionWindow(self, self.window())
            window.setWindowTitle(f"pyBer - {self.title}")
            layout = QtWidgets.QVBoxLayout(window)
            layout.setContentsMargins(12, 10, 12, 12)
            layout.setSpacing(6)
            bar = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel(self.title)
            font = title.font()
            font.setBold(True)
            font.setPointSizeF(font.pointSizeF() + 1.5)
            title.setFont(font)
            bar.addWidget(title, 1)
            dock = QtWidgets.QPushButton("Dock back")
            dock.setProperty("class", "compactSmall")
            dock.setToolTip("Return the list to the Setup drawer")
            dock.clicked.connect(self.attach)
            bar.addWidget(dock)
            layout.addLayout(bar)
            self.window_widget = window
        return self.window_widget

    def detach(self) -> None:
        if self.is_detached():
            self._raise_window()
            return
        window = self._ensure_window()
        self.layout().removeWidget(self.content)
        window.layout().addWidget(self.content, 1)
        self.content.show()
        self.placeholder.show()
        self.btn_toggle.setText("Dock back")
        geometry = self.settings.value(self.settings_key) if self.settings is not None else None
        restored = False
        if isinstance(geometry, (QtCore.QByteArray, bytes)):
            restored = window.restoreGeometry(QtCore.QByteArray(geometry))
        if not restored:
            window.resize(980, 760)
        window.show()
        self._raise_window()
        self.detachedChanged.emit(True)

    def attach(self) -> None:
        if not self.is_detached():
            return
        window = self.window_widget
        if self.settings is not None:
            self.settings.setValue(self.settings_key, window.saveGeometry())
        window.layout().removeWidget(self.content)
        self.layout().addWidget(self.content, 1)
        self.content.show()
        self.placeholder.hide()
        self.btn_toggle.setText("Open in window")
        window._docking = True
        try:
            window.hide()
        finally:
            window._docking = False
        self.detachedChanged.emit(False)

    def _raise_window(self) -> None:
        if self.window_widget is not None:
            self.window_widget.showNormal()
            self.window_widget.raise_()
            self.window_widget.activateWindow()


class _SectionWindow(QtWidgets.QWidget):
    """Top-level window that docks its content back when closed."""

    def __init__(self, section: DetachableSection, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.section = section
        self._docking = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._docking:
            self.section.attach()
        event.accept()
