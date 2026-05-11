# styles.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# Modern flat icons painted programmatically — used by the side-rail buttons
# in main.py and gui_postprocessing.py. No external assets needed.
# ---------------------------------------------------------------------------

def _make_icon(painter_fn, size: int = 40, color: str = "#cdd6f4"):
    from PySide6 import QtCore, QtGui
    pix = QtGui.QPixmap(size, size)
    pix.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter_fn(p, QtCore.QRect(6, 6, size - 12, size - 12), QtGui.QColor(color))
    p.end()
    return QtGui.QIcon(pix)


def _pen(c, w=2.0):
    from PySide6 import QtCore, QtGui
    return QtGui.QPen(c, w, QtCore.Qt.PenStyle.SolidLine,
                      QtCore.Qt.PenCapStyle.RoundCap,
                      QtCore.Qt.PenJoinStyle.RoundJoin)


def _paint_database(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 1.9)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx = r.center().x(); ry = max(2, r.height() // 8)
    p.drawEllipse(QtCore.QPoint(cx, r.top() + ry), r.width() // 2 - 1, ry)
    p.drawLine(r.left() + 1, r.top() + ry, r.left() + 1, r.bottom() - ry)
    p.drawLine(r.right() - 1, r.top() + ry, r.right() - 1, r.bottom() - ry)
    p.drawArc(QtCore.QRect(r.left() + 1, r.bottom() - 2 * ry, r.width() - 2, 2 * ry),
              200 * 16, 140 * 16)
    p.drawArc(QtCore.QRect(r.left() + 1, r.center().y() - ry, r.width() - 2, 2 * ry),
              200 * 16, 140 * 16)


def _paint_list(p, r, c):
    p.setPen(_pen(c, 2.0))
    for i in range(3):
        y = r.top() + 3 + i * (r.height() // 3)
        p.drawLine(r.left() + 5, y, r.left() + 5, y)
        p.drawLine(r.left() + 9, y, r.right() - 1, y)


def _paint_sliders(p, r, c):
    from PySide6 import QtCore, QtGui
    p.setPen(_pen(c, 2.0)); p.setBrush(QtGui.QColor(c))
    rows = [(0.25, 0.4), (0.55, 0.65), (0.8, 0.3)]
    for frac_y, knob_x in rows:
        y = r.top() + int(r.height() * frac_y)
        p.drawLine(r.left() + 1, y, r.right() - 1, y)
        kx = r.left() + int(r.width() * knob_x)
        p.drawEllipse(QtCore.QPoint(kx, y), 2, 2)


def _paint_filter(p, r, c):
    from PySide6 import QtCore, QtGui
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    pts = [QtCore.QPoint(r.left() + 1, r.top() + 2),
           QtCore.QPoint(r.right() - 1, r.top() + 2),
           QtCore.QPoint(r.center().x() + r.width() // 5, r.center().y()),
           QtCore.QPoint(r.center().x() + r.width() // 5, r.bottom() - 2),
           QtCore.QPoint(r.center().x() - r.width() // 5, r.bottom() - 2),
           QtCore.QPoint(r.center().x() - r.width() // 5, r.center().y())]
    p.drawPolygon(QtGui.QPolygon(pts))


def _paint_wave(p, r, c):
    from PySide6 import QtCore, QtGui
    import math
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    path = QtGui.QPainterPath()
    cy = r.center().y()
    path.moveTo(r.left(), cy)
    w = r.width()
    for i in range(w + 1):
        x = r.left() + i
        y = cy - math.sin(i / w * 2 * math.pi * 1.4) * (r.height() / 2 - 2)
        path.lineTo(x, y)
    p.drawPath(path)


def _paint_chart(p, r, c):
    from PySide6 import QtGui
    p.setPen(_pen(c, 1.6)); p.setBrush(QtGui.QColor(c))
    bar_w = max(3, r.width() // 5)
    gap = max(2, (r.width() - bar_w * 3) // 4)
    heights = [0.5, 0.85, 0.65]
    x = r.left() + gap
    for h in heights:
        bh = int(r.height() * h)
        p.drawRect(x, r.bottom() - bh, bar_w, bh)
        x += bar_w + gap


def _paint_badge(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx, cy = r.center().x(), r.center().y()
    rad = min(r.width(), r.height()) // 2 - 1
    p.drawEllipse(QtCore.QPoint(cx, cy), rad, rad)
    p.drawLine(cx - rad // 2, cy, cx - 2, cy + rad // 2)
    p.drawLine(cx - 2, cy + rad // 2, cx + rad // 2, cy - rad // 3)


def _paint_export(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx = r.center().x()
    p.drawLine(cx, r.top() + 1, cx, r.bottom() - r.height() // 3)
    p.drawLine(cx, r.bottom() - r.height() // 3,
               cx - r.width() // 4, r.bottom() - r.height() // 3 - r.width() // 4)
    p.drawLine(cx, r.bottom() - r.height() // 3,
               cx + r.width() // 4, r.bottom() - r.height() // 3 - r.width() // 4)
    p.drawLine(r.left() + 1, r.bottom() - 1, r.right() - 1, r.bottom() - 1)


def _paint_gear(p, r, c):
    from PySide6 import QtCore
    import math
    p.setPen(_pen(c, 1.8)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx, cy = r.center().x(), r.center().y()
    rad = min(r.width(), r.height()) // 2 - 2
    p.drawEllipse(QtCore.QPoint(cx, cy), rad - 2, rad - 2)
    p.drawEllipse(QtCore.QPoint(cx, cy), max(1, rad // 3), max(1, rad // 3))
    for k in range(8):
        a = k * math.pi / 4
        x1 = cx + (rad - 1) * math.cos(a); y1 = cy + (rad - 1) * math.sin(a)
        x2 = cx + (rad + 2) * math.cos(a); y2 = cy + (rad + 2) * math.sin(a)
        p.drawLine(int(x1), int(y1), int(x2), int(y2))


def _paint_grid(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 1.6)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    p.drawRect(r)
    p.drawLine(r.left(), r.center().y(), r.right(), r.center().y())
    p.drawLine(r.center().x(), r.top(), r.center().x(), r.bottom())


def _paint_target(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cx, cy = r.center().x(), r.center().y()
    rad = min(r.width(), r.height()) // 2 - 1
    p.drawEllipse(QtCore.QPoint(cx, cy), rad, rad)
    p.drawEllipse(QtCore.QPoint(cx, cy), rad // 2, rad // 2)
    p.drawLine(cx - rad - 2, cy, cx - 2, cy)
    p.drawLine(cx + 2, cy, cx + rad + 2, cy)
    p.drawLine(cx, cy - rad - 2, cx, cy - 2)
    p.drawLine(cx, cy + 2, cx, cy + rad + 2)


def _paint_pulse(p, r, c):
    from PySide6 import QtCore
    p.setPen(_pen(c, 2.0)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    cy = r.center().y()
    x0 = r.left()
    p.drawLine(x0, cy, x0 + r.width() // 4, cy)
    p.drawLine(x0 + r.width() // 4, cy, x0 + r.width() // 4, r.top() + 1)
    p.drawLine(x0 + r.width() // 4, r.top() + 1, x0 + r.width() // 2, r.top() + 1)
    p.drawLine(x0 + r.width() // 2, r.top() + 1, x0 + r.width() // 2, r.bottom() - 1)
    p.drawLine(x0 + r.width() // 2, r.bottom() - 1, x0 + 3 * r.width() // 4, r.bottom() - 1)
    p.drawLine(x0 + 3 * r.width() // 4, r.bottom() - 1, x0 + 3 * r.width() // 4, cy)
    p.drawLine(x0 + 3 * r.width() // 4, cy, r.right(), cy)


def _paint_paw(p, r, c):
    from PySide6 import QtCore, QtGui
    p.setPen(_pen(c, 1.4)); p.setBrush(QtGui.QColor(c))
    cx, cy = r.center().x(), r.center().y()
    rw = r.width(); rh = r.height()
    # Pad
    p.drawEllipse(QtCore.QPoint(cx, cy + rh // 6), rw // 3, rh // 4)
    # Toes
    for dx in (-rw // 3, -rw // 9, rw // 9, rw // 3):
        p.drawEllipse(QtCore.QPoint(cx + dx, cy - rh // 4), max(2, rw // 10), max(2, rh // 8))


def _paint_temporal(p, r, c):
    """Temporal modeling icon — sine wave over a grid with a regression line."""
    from PySide6 import QtCore, QtGui
    import math
    p.setPen(_pen(c, 1.6)); p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    # Horizontal axis
    cy = r.top() + int(r.height() * 0.6)
    p.drawLine(r.left(), cy, r.right(), cy)
    # Sine-like curve
    path = QtGui.QPainterPath()
    path.moveTo(r.left(), cy)
    w = r.width()
    for i in range(w + 1):
        x = r.left() + i
        y = cy - math.sin(i / w * 2.5 * math.pi) * (r.height() * 0.35)
        path.lineTo(x, y)
    p.setPen(_pen(c, 2.0))
    p.drawPath(path)
    # Regression trend line (dashed)
    p.setPen(_pen(QtGui.QColor(c).lighter(140), 1.4))
    p.drawLine(r.left() + 2, cy + int(r.height() * 0.15),
               r.right() - 2, cy - int(r.height() * 0.25))




APP_QSS = r"""
/* ============================================================================
   pyBer dark theme - design tokens (kept readable for future tweaks)

   Surfaces           Text                  Accents / status
   ----------------   ------------------    --------------------------------
   #14171f base       #e9ecf3 primary       #7d4df2 primary accent
   #1a1d26 panel      #aab4c5 secondary     #8d60ff accent hover
   #20242d card       #6f7a8e muted         #4a3678 accent soft
   #262b35 raised                           #378ef0 info / focus
   #2b303b chip                             #5dd39e success
                                            #f5c542 warn
   Borders                                  #ee6471 error
   #2c3240 subtle
   #3a4050 default
   #4a5163 strong

   Fonts: 12.5pt panel title, 9.5pt body, 8.7pt hint, 8.2pt micro
   Radii: 6 inputs/buttons, 8 cards, 12 outer panels
============================================================================ */

QMainWindow, QWidget {
    background: #1a1d26;
    color: #e9ecf3;
    font-family: "Segoe UI", "Inter", "Arial", sans-serif;
    font-size: 9.5pt;
}

QLabel {
    color: #e9ecf3;
}

QLabel[class="hint"], QLabel[class="muted"] {
    color: #aab4c5;
    font-size: 8.7pt;
}

QLabel[class="title"] {
    color: #f3f5f8;
    font-size: 12.5pt;
    font-weight: 700;
    letter-spacing: 0.2px;
}

QLabel[class="badge"] {
    background: #4a3678;
    color: #ffffff;
    border-radius: 9px;
    padding: 2px 9px;
    font-weight: 700;
    font-size: 8.2pt;
}

/* Panels: card-like group boxes with a small accent stripe on the left */
QGroupBox {
    border: 1px solid #2c3240;
    border-left: 2px solid #4a3678;
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px 14px 12px 14px;
    background: #1f242e;
}

/* Keep helper row-container widgets inside group boxes transparent. */
QGroupBox > QWidget {
    background: transparent;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 1px 10px;
    margin-left: 10px;
    color: #b9c4d6;
    background: #14171f;
    border: 1px solid #2c3240;
    border-radius: 6px;
    font-weight: 700;
    font-size: 8.5pt;
    letter-spacing: 0.7px;
    text-transform: uppercase;
}

/* Generic in-panel section header (used by the Subsection helper). */
QFrame[class="subsection"] {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #2c3240;
}

QLabel[class="subsectionTitle"] {
    color: #e9ecf3;
    font-size: 9.5pt;
    font-weight: 700;
    letter-spacing: 0.2px;
    padding: 2px 0;
}

QLabel[class="subsectionHint"] {
    color: #aab4c5;
    font-size: 8.5pt;
    padding: 0;
}

/* Footer action strip pinned to the bottom of a panel. */
QFrame[class="footerActions"] {
    background: #1a1d26;
    border: 1px solid #2c3240;
    border-radius: 10px;
    padding: 8px 12px;
}

/* Inline status banner (used by InlineStatus helper). */
QFrame[class="inlineStatus"] {
    background: #1f242e;
    border: 1px solid #2c3240;
    border-radius: 8px;
    padding: 6px 10px;
}

QFrame[class="inlineStatus"][severity="ok"]    { border: 1px solid #2f7a4a; background: #1c2e22; }
QFrame[class="inlineStatus"][severity="warn"]  { border: 1px solid #8a6a3a; background: #2e2918; }
QFrame[class="inlineStatus"][severity="error"] { border: 1px solid #8a3949; background: #2e1c20; }
QFrame[class="inlineStatus"] QLabel { background: transparent; color: #cfd8e6; font-size: 8.7pt; }

/* Collapsible chevron button (rotated when expanded). */
QToolButton[class="chevron"] {
    background: transparent;
    border: 0;
    color: #aab4c5;
    padding: 0 4px;
    font-size: 11pt;
    font-weight: 700;
}

QToolButton[class="chevron"]:hover {
    color: #ffffff;
}

QFrame {
    border-color: #2c3240;
}

/* Lists and tables */
QListWidget, QTableWidget {
    background: #1b1f28;
    border: 1px solid #2c3240;
    border-radius: 8px;
    padding: 4px;
    gridline-color: #2c3240;
    alternate-background-color: #1f242e;
}

QListWidget::item, QTableWidget::item {
    padding: 4px 6px;
    border-radius: 4px;
}

QListWidget::item:hover, QTableWidget::item:hover {
    background: #232834;
}

QListWidget::item:selected, QTableWidget::item:selected {
    background: #4a3678;
    color: #ffffff;
    border-radius: 4px;
}

QHeaderView::section {
    background: #232834;
    color: #cfd8e6;
    border: 0;
    border-bottom: 1px solid #2c3240;
    padding: 5px 8px;
    font-weight: 600;
    font-size: 8.7pt;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #1b1f28;
    color: #e9ecf3;
    border: 1px solid #3a4050;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #7d4df2;
    selection-color: #ffffff;
    min-height: 22px;
}

QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {
    border: 1px solid #4a5163;
}

QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 1px solid #8d60ff;
    background: #1f242e;
}

QComboBox::drop-down {
    border: 0;
    width: 22px;
    background: transparent;
}

QComboBox::down-arrow {
    image: none;
    width: 8px;
    height: 8px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #aab4c5;
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background: #1f242e;
    color: #e9ecf3;
    border: 1px solid #3a4050;
    border-radius: 8px;
    padding: 4px;
    selection-background-color: #4a3678;
    outline: 0;
}

QCheckBox {
    spacing: 7px;
    color: #e9ecf3;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1px solid #4a5163;
    background: #1b1f28;
}

QCheckBox::indicator:hover {
    border: 1px solid #8d60ff;
}

QCheckBox::indicator:checked {
    background: #7d4df2;
    border: 1px solid #9064ff;
}

QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border-radius: 8px;
    border: 1px solid #4a5163;
    background: #1b1f28;
}

QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 #ffffff, stop:0.45 #ffffff, stop:0.5 #7d4df2, stop:1 #7d4df2);
    border: 1px solid #9064ff;
}

/* Buttons */
QPushButton {
    background: #262b35;
    border: 1px solid #3a4050;
    border-radius: 7px;
    padding: 7px 14px;
    font-weight: 600;
    color: #e9ecf3;
}

QPushButton:hover {
    background: #2c3340;
    border: 1px solid #4a5163;
}

QPushButton:pressed {
    background: #1f242e;
}

QPushButton:disabled {
    color: #6f7a8e;
    background: #1f242e;
    border: 1px solid #2c3240;
}

QPushButton[class="compact"] {
    padding: 5px 10px;
    border-radius: 6px;
    font-weight: 600;
}

QPushButton[class="compactSmall"] {
    padding: 3px 9px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.4pt;
    min-height: 20px;
    color: #cfd8e6;
}

QPushButton[class="compactSmall"]:hover {
    color: #ffffff;
}

/* Primary purple action */
QPushButton[class="primary"],
QPushButton[class="compactPrimary"],
QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    background: #7d4df2;
    border: 1px solid #9064ff;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[class="primary"] {
    padding: 8px 18px;
    border-radius: 8px;
}

QPushButton[class="compactPrimary"] {
    padding: 5px 12px;
    border-radius: 7px;
}

QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    padding: 3px 11px;
    border-radius: 6px;
    font-size: 8.4pt;
    min-height: 20px;
}

QPushButton[class="primary"]:hover,
QPushButton[class="compactPrimary"]:hover,
QPushButton[class="compactPrimarySmall"]:hover,
QPushButton[class="bluePrimarySmall"]:hover {
    background: #8d60ff;
    border: 1px solid #9f79ff;
}

QPushButton[class="primary"]:pressed,
QPushButton[class="compactPrimary"]:pressed,
QPushButton[class="compactPrimarySmall"]:pressed,
QPushButton[class="bluePrimarySmall"]:pressed {
    background: #6f3de9;
    border: 1px solid #8253f3;
}

QPushButton[class="primary"]:disabled,
QPushButton[class="compactPrimary"]:disabled,
QPushButton[class="compactPrimarySmall"]:disabled {
    background: #4a3678;
    border: 1px solid #5a4690;
    color: #c0b1ee;
}

QPushButton[class="blueSecondarySmall"] {
    padding: 3px 11px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.4pt;
    min-height: 20px;
    background: #2c2640;
    border: 1px solid #4a3678;
    color: #e9e1f8;
}

QPushButton[class="blueSecondarySmall"]:hover {
    background: #382f50;
    border: 1px solid #5e4690;
}

QPushButton[class="blueSecondarySmall"]:pressed {
    background: #251e36;
}

QPushButton[class="blueSecondarySmall"]:checked,
QPushButton[class="sectionButton"]:checked {
    background: #7d4df2;
    border: 1px solid #9064ff;
    color: #ffffff;
}

/* Ghost button: no fill, used for non-primary toolbar actions (Undo/Redo). */
QPushButton[class="ghost"] {
    background: transparent;
    border: 1px solid transparent;
    color: #aab4c5;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 6px;
}

QPushButton[class="ghost"]:hover {
    background: #232834;
    color: #ffffff;
    border: 1px solid #2c3240;
}

QPushButton[class="ghost"]:pressed {
    background: #1f242e;
}

QPushButton[class="sectionButton"] {
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.4pt;
    background: #232834;
    border: 1px solid #2c3240;
    text-align: left;
    color: #cfd8e6;
}

QPushButton[class="sectionButton"]:hover {
    background: #2c3340;
    color: #ffffff;
}

QPushButton[class="help"] {
    padding: 0;
    border-radius: 11px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    font-weight: 700;
    background: #232834;
    border: 1px solid #3a4050;
    color: #aab4c5;
}

QPushButton[class="help"]:hover {
    background: #2c3340;
    color: #ffffff;
}

/* Toggleable rail-like icon button. */
QToolButton[class="iconRail"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px;
}

QToolButton[class="iconRail"]:hover {
    background: #232834;
    border: 1px solid #2c3240;
}

QToolButton[class="iconRail"]:checked {
    background: #4a3678;
    border: 1px solid #7d4df2;
}

/* Tabs (clean underline indicator) */
QTabWidget::pane {
    border: 0;
    border-top: 1px solid #2c3240;
    background: #1a1d26;
    padding: 4px;
}

QTabBar::tab {
    background: transparent;
    border: 0;
    padding: 9px 18px;
    margin: 0 2px 0 0;
    font-weight: 600;
    color: #aab4c5;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:hover {
    color: #ffffff;
    background: #1f242e;
    border-bottom: 2px solid #4a3678;
}

QTabBar::tab:selected {
    background: transparent;
    color: #ffffff;
    border-bottom: 2px solid #7d4df2;
}

/* Menus and tool buttons */
QMenu {
    background: #1f242e;
    color: #e9ecf3;
    border: 1px solid #3a4050;
    border-radius: 8px;
    padding: 5px;
}

QMenu::item {
    padding: 6px 18px;
    border-radius: 5px;
}

QMenu::item:selected {
    background: #4a3678;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background: #2c3240;
    margin: 4px 8px;
}

QMenuBar {
    background: #1a1d26;
    color: #e9ecf3;
    border-bottom: 1px solid #2c3240;
    padding: 2px;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 11px;
    border-radius: 5px;
}

QMenuBar::item:selected {
    background: #232834;
}

QToolButton {
    background: #232834;
    border: 1px solid #2c3240;
    border-radius: 6px;
    padding: 4px 8px;
    color: #e9ecf3;
}

QToolButton:hover {
    background: #2c3340;
    border: 1px solid #3a4050;
}

QToolButton:pressed {
    background: #1f242e;
}

/* Docking and splitters */
QDockWidget {
    background: #1f242e;
    color: #e9ecf3;
    border: 1px solid #2c3240;
}

QDockWidget::title {
    background: #232834;
    border-bottom: 1px solid #2c3240;
    padding: 7px 10px;
    text-align: left;
    font-weight: 700;
    font-size: 8.7pt;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    color: #aab4c5;
}

QDockWidget > QWidget {
    background: #1f242e;
}

QMainWindow::separator {
    background: #2c3240;
    width: 4px;
    height: 4px;
}

QMainWindow::separator:hover {
    background: #7d4df2;
}

QSplitter::handle {
    background: #2c3240;
}

QSplitter::handle:hover {
    background: #7d4df2;
}

QScrollArea {
    border: none;
    background: transparent;
}

/* Plot frame support */
QGraphicsView {
    background: #121722;
    border: 1px solid #2c3240;
    border-radius: 8px;
}

QToolTip {
    background: #1f242e;
    color: #e9ecf3;
    border: 1px solid #4a3678;
    border-radius: 6px;
    padding: 5px 8px;
}

/* Status bar */
QStatusBar {
    background: #14171f;
    color: #aab4c5;
    border-top: 1px solid #2c3240;
}

QStatusBar::item {
    border: 0;
}

QStatusBar QLabel {
    color: #aab4c5;
    padding: 0 4px;
}

/* Scroll bars (slim, themed) */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #2c3340;
    min-height: 24px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #4a3678;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    background: transparent;
    height: 0;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background: #2c3340;
    min-width: 24px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background: #4a3678;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    background: transparent;
    width: 0;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* Progress bar (used in temporal modeling, top-bar busy indicator) */
QProgressBar {
    background: #1b1f28;
    border: 1px solid #2c3240;
    border-radius: 6px;
    text-align: center;
    color: #e9ecf3;
    height: 18px;
    font-size: 8.4pt;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7d4df2, stop:1 #8d60ff);
    border-radius: 5px;
    margin: 1px;
}

/* ---------- Modern shell: side rail, drawers, transport bar ---------- */
QFrame#sideRail {
    background: #14171f;
    border: 1px solid #2c3240;
    border-radius: 14px;
}

QFrame#drawerPanel, QFrame#centerPanel {
    background: #1a1d26;
    border: 1px solid #2c3240;
    border-radius: 14px;
}

QFrame#transportBar {
    background: #14171f;
    border: 1px solid #2c3240;
    border-radius: 12px;
}

QFrame#railSeparator {
    background: #2c3240;
    max-height: 1px;
    min-height: 1px;
    border: none;
    margin: 6px 10px;
}

QLabel#panelTitle {
    color: #f3f5f8;
    font-size: 13pt;
    font-weight: 700;
    letter-spacing: 0.4px;
    padding: 2px 4px;
}

QFrame#pyberPanelHeader {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #2c3240;
    padding: 0;
}

QFrame#pyberPanelHeader QLabel {
    background: transparent;
}

QLabel#pyberPanelHeaderTitle {
    color: #f3f5f8;
    font-size: 13pt;
    font-weight: 800;
    letter-spacing: 0.3px;
}

QLabel#pyberPanelHeaderSubtitle {
    color: #aab4c5;
    font-size: 8.7pt;
}

QLabel#transportStatus {
    color: #aab4c5;
    font-size: 9pt;
    font-weight: 600;
    padding: 0 6px;
}

QFrame#pyberBusyWidget {
    background: #2a3045;
    border: 1px solid #46527a;
    border-radius: 6px;
    padding: 0 8px;
}

QFrame#pyberBusyWidget QLabel {
    background: transparent;
    color: #d7e0ee;
}

QFrame#pyberBusyWidget QPushButton {
    background: #543035;
    color: #ffd6dc;
    border: 1px solid #8a3949;
    border-radius: 4px;
    padding: 1px 8px;
}

QFrame#pyberBusyWidget QPushButton:hover {
    background: #6b3a40;
}

QPushButton#railButton, QPushButton#railToggleButton,
QToolButton#railButton, QToolButton#railToggleButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 10px;
    min-width: 0;
    padding: 8px;
    text-align: center;
    font-weight: 600;
    color: #cfd8e6;
}

QPushButton#railButton:hover,
QPushButton#railToggleButton:hover,
QToolButton#railButton:hover,
QToolButton#railToggleButton:hover {
    background: #1f242e;
    border: 1px solid #2c3240;
    color: #ffffff;
}

QPushButton#railButton:checked,
QPushButton#railToggleButton:checked,
QToolButton#railButton:checked,
QToolButton#railToggleButton:checked {
    background: #4a3678;
    border: 1px solid #7d4df2;
    color: #ffffff;
}

QPushButton#railButton:disabled,
QPushButton#railToggleButton:disabled,
QToolButton#railButton:disabled,
QToolButton#railToggleButton:disabled {
    background: transparent;
    color: #4a5266;
    border: 1px solid transparent;
}

QTabBar#visualModeBar::tab {
    min-width: 90px;
    padding: 5px 14px;
    border-radius: 6px;
    margin-right: 4px;
    font-weight: 700;
    background: #232834;
    color: #aab4c5;
}

QTabBar#visualModeBar::tab:hover:!selected {
    background: #2c3340;
    color: #ffffff;
}

QTabBar#visualModeBar::tab:selected {
    background: #4a3678;
    color: #ffffff;
    border: 1px solid #7d4df2;
}

/* ---------- Top app bar (workflow header) ---------- */
QFrame#pyberTopBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #1f242e, stop:1 #161a23);
    border: 0;
    border-bottom: 1px solid #2c3240;
}

QLabel#pyberAppName {
    color: #ffffff;
    font-size: 12.5pt;
    font-weight: 800;
    letter-spacing: 0.4px;
    padding: 0 6px;
}

QLabel#pyberAppMark {
    background: #4a3678;
    color: #ffffff;
    border-radius: 9px;
    font-size: 11pt;
    font-weight: 800;
    padding: 0;
    qproperty-alignment: AlignCenter;
}

QLabel#pyberWorkflowStep {
    color: #aab4c5;
    font-size: 9pt;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 8px;
}

QLabel#pyberWorkflowStep[active="true"] {
    color: #ffffff;
    background: #4a3678;
}

QLabel#pyberWorkflowSep {
    color: #4a5266;
    font-size: 11pt;
    font-weight: 700;
}

QLabel#pyberProjectName {
    color: #aab4c5;
    font-size: 9pt;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    background: #1a1d26;
    border: 1px solid #2c3240;
}

QLabel#pyberProjectName[dirty="true"] {
    color: #f5c542;
    border: 1px solid #5d4a14;
}

/* ---------- Reusable polish primitives ---------- */
QFrame[class="card"] {
    background: #20242d;
    border: 1px solid #2c3240;
    border-radius: 10px;
}

QFrame[class="callout"] {
    background: #1f242e;
    border: 1px solid #4a3678;
    border-radius: 10px;
}

QFrame[class="hairline"] {
    background: #2c3240;
    max-height: 1px;
    min-height: 1px;
    border: 0;
}

/* Empty-state hint label inside otherwise-blank panels. */
QLabel[class="emptyState"] {
    color: #6f7a8e;
    font-size: 10pt;
    font-weight: 500;
    padding: 24px;
}

QLabel[class="emptyStateTitle"] {
    color: #aab4c5;
    font-size: 13pt;
    font-weight: 700;
    padding: 6px;
}
"""


_LIGHT_COLOR_MAP = {
    "#121722": "#ffffff",
    "#14171f": "#eef1f7",
    "#161a23": "#eef1f7",
    "#1473e6": "#1f6fce",
    "#1a1d26": "#f4f6fb",
    "#1b1f28": "#ffffff",
    "#1b2029": "#ffffff",
    "#1d4f80": "#2f7fd8",
    "#1f242e": "#ffffff",
    "#1f2229": "#f4f6fb",
    "#181a22": "#eef1f7",
    "#20242d": "#ffffff",
    "#232833": "#d7dfeb",
    "#232834": "#eef2f8",
    "#242a34": "#f6f8fc",
    "#251e36": "#ddd3f3",
    "#252a33": "#eef2f8",
    "#262b35": "#ffffff",
    "#2680eb": "#2f7fd8",
    "#2b303b": "#e8eef6",
    "#2c2640": "#ebe4fa",
    "#2c3240": "#c8d2e0",
    "#2c3340": "#dde6f1",
    "#2d3340": "#e1e8f2",
    "#2d3442": "#f3f6fb",
    "#2f3543": "#eef3fa",
    "#313746": "#c8d2e0",
    "#343a47": "#d2dae6",
    "#343a48": "#dde6f1",
    "#363c4a": "#cad3e1",
    "#373d4a": "#d4dce8",
    "#378ef0": "#2f7fd8",
    "#382f50": "#ddd3f3",
    "#3a4050": "#c2ccda",
    "#3b3251": "#ddd3f3",
    "#3b4150": "#c5cedc",
    "#3f4758": "#bcc8d8",
    "#41495a": "#bcc8da",
    "#4a3678": "#9d7fe0",
    "#4a5163": "#adb8c9",
    "#4a5266": "#aab6c8",
    "#4b9df8": "#5fa8f5",
    "#4d5568": "#aab6c8",
    "#5a4690": "#b9a9df",
    "#5a6274": "#95a3b7",
    "#5b6480": "#9eadc4",
    "#5d4a14": "#d6c478",
    "#5e4690": "#b9a9df",
    "#5f4a89": "#b9a9df",
    "#5dd39e": "#37a673",
    "#67dba0": "#37a673",
    "#6f3de9": "#8f69d9",
    "#6f7a8e": "#8c98ab",
    "#7059a0": "#b8a8da",
    "#7d4df2": "#8f69d9",
    "#8253f3": "#9d7fe0",
    "#858f9f": "#8c98ab",
    "#8d60ff": "#9d7fe0",
    "#9064ff": "#9d7fe0",
    "#9f79ff": "#b09ae8",
    "#aab4c5": "#5d6a7d",
    "#aeb6c5": "#5d6a7d",
    "#bcc5d6": "#667489",
    "#c0b1ee": "#b09ae8",
    "#c5cedc": "#cdd5e2",
    "#cfd8e6": "#243247",
    "#cfd8e6": "#243247",
    "#e9e1f8": "#3a2a78",
    "#e9ecf3": "#1f2a37",
    "#ecf0f6": "#243247",
    "#ee6471": "#c44456",
    "#eef2f7": "#1f2a37",
    "#f3f5f8": "#1f2a37",
    "#f5c542": "#a78415",
    "#ffffff": "#ffffff",
}


def _build_light_qss(dark_qss: str) -> str:
    light = str(dark_qss)
    # Replace longer tokens first to avoid accidental partial replacement collisions.
    for dark, bright in sorted(_LIGHT_COLOR_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        light = light.replace(dark, bright)
    return light


# The find/replace mapping above can leave white-on-white text in :hover /
# :selected / :checked states (because `color: #ffffff` stays white). These
# overrides come last and win against the auto-mapped rules.
_LIGHT_OVERRIDES = r"""
/* ---------- Light-mode contrast fixes ---------- */

QMainWindow, QWidget {
    background: #f4f6fb;
    color: #1f2a37;
}

/* Tabs: dark text on hover / selected, accent-tinted background */
QTabWidget::pane { background: #f4f6fb; border-top: 1px solid #c8d2e0; }
QTabBar::tab { color: #4a5568; background: transparent; }
QTabBar::tab:hover {
    color: #1f2a37;
    background: #e6e9f1;
    border-bottom: 2px solid #b9a9df;
}
QTabBar::tab:selected {
    color: #1f2a37;
    background: transparent;
    border-bottom: 2px solid #7d4df2;
}

/* Group boxes as soft cards on the light surface */
QGroupBox {
    border: 1px solid #d6dde9;
    border-left: 2px solid #b9a9df;
    background: #ffffff;
}
QGroupBox::title {
    background: #f0f2f7;
    border: 1px solid #d6dde9;
    color: #4a5568;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #ffffff;
    color: #1f2a37;
    border: 1px solid #c2ccda;
    selection-background-color: #7d4df2;
    selection-color: #ffffff;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 1px solid #7d4df2;
    background: #ffffff;
}
QComboBox::down-arrow { border-top-color: #5d6a7d; }
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1f2a37;
    border: 1px solid #c2ccda;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}

/* Lists / tables */
QListWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid #d6dde9;
    color: #1f2a37;
    alternate-background-color: #f6f8fc;
}
QListWidget::item:hover, QTableWidget::item:hover { background: #eff3fa; }
QListWidget::item:selected, QTableWidget::item:selected {
    background: #2563eb; color: #ffffff;
}
QHeaderView::section {
    background: #f0f2f7; color: #1f2a37; border: 0;
    border-bottom: 1px solid #d6dde9;
}

/* Buttons: dark text everywhere; primary purple keeps white text */
QPushButton {
    background: #ffffff; color: #1f2a37; border: 1px solid #c2ccda;
}
QPushButton:hover { background: #eff3fa; border: 1px solid #b9a9df; }
QPushButton:pressed { background: #e6e9f1; }
QPushButton:disabled { color: #98a3b5; background: #f4f6fb; border: 1px solid #e0e6ee; }

QPushButton[class="ghost"] { color: #4a5568; }
QPushButton[class="ghost"]:hover { background: #e6e9f1; color: #1f2a37; border: 1px solid #d6dde9; }

QPushButton[class="primary"], QPushButton[class="compactPrimary"],
QPushButton[class="compactPrimarySmall"], QPushButton[class="bluePrimarySmall"] {
    background: #7d4df2; border: 1px solid #8d60ff; color: #ffffff;
}
QPushButton[class="primary"]:hover, QPushButton[class="compactPrimary"]:hover,
QPushButton[class="compactPrimarySmall"]:hover, QPushButton[class="bluePrimarySmall"]:hover {
    background: #8d60ff; border: 1px solid #9f79ff; color: #ffffff;
}
QPushButton[class="compactSmall"] { color: #4a5568; }
QPushButton[class="compactSmall"]:hover { color: #1f2a37; }

QPushButton[class="sectionButton"] {
    background: #ffffff; border: 1px solid #d6dde9; color: #1f2a37;
}
QPushButton[class="sectionButton"]:hover { background: #eff3fa; }
QPushButton[class="blueSecondarySmall"] {
    background: #f3edff; border: 1px solid #c0a8ee; color: #4a3678;
}
QPushButton[class="blueSecondarySmall"]:checked,
QPushButton[class="sectionButton"]:checked {
    background: #2563eb; border: 1px solid #1d4ed8; color: #ffffff;
}

/* Side rail (icon + label under) */
QFrame#sideRail { background: #eef1f7; border: 1px solid #d6dde9; }
QToolButton#railButton, QToolButton#railToggleButton {
    background: transparent; border: 1px solid transparent; color: #4a5568;
}
QToolButton#railButton:hover, QToolButton#railToggleButton:hover {
    background: #dbeafe; border: 1px solid #93c5fd; color: #172033;
}
QToolButton#railButton:checked, QToolButton#railToggleButton:checked {
    background: #2563eb; border: 1px solid #1d4ed8; color: #ffffff;
}
QPushButton#railButton:hover, QPushButton#railToggleButton:hover {
    background: #dbeafe; border: 1px solid #93c5fd; color: #172033;
}
QPushButton#railButton:checked, QPushButton#railToggleButton:checked {
    background: #2563eb; border: 1px solid #1d4ed8; color: #ffffff;
}

/* Drawer panels */
QFrame#drawerPanel, QFrame#centerPanel { background: #ffffff; border: 1px solid #d6dde9; }
QFrame#transportBar { background: #eef1f7; border: 1px solid #d6dde9; }

/* Status bar */
QStatusBar { background: #eef1f7; color: #4a5568; border-top: 1px solid #d6dde9; }
QStatusBar QLabel { color: #4a5568; }

/* Scroll bars */
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #c8d2e0; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #7d4df2; }

/* Plot frame stays a soft near-white so the dark plot lines pop */
QGraphicsView { background: #fbfcfe; border: 1px solid #d6dde9; }

/* Reusable in-panel building blocks */
QFrame#pyberPanelHeader {
    border-bottom: 1px solid #d6dde9;
}
QLabel#pyberPanelHeaderTitle { color: #172033; }
QLabel#pyberPanelHeaderSubtitle { color: #4c5a6f; }

QFrame#pyberBusyWidget {
    background: #fff7ed;
    border: 1px solid #fb923c;
    border-radius: 6px;
}
QFrame#pyberBusyWidget QLabel {
    color: #7c2d12;
}
QFrame#pyberBusyWidget QPushButton {
    background: #dc2626;
    color: #ffffff;
    border: 1px solid #b91c1c;
}
QFrame#pyberBusyWidget QPushButton:hover {
    background: #b91c1c;
}

QTabBar#visualModeBar::tab {
    background: #eef2f8;
    color: #4a5568;
    border: 1px solid #d6dde9;
}
QTabBar#visualModeBar::tab:hover:!selected {
    background: #dbeafe;
    color: #172033;
    border: 1px solid #93c5fd;
}
QTabBar#visualModeBar::tab:selected {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #1d4ed8;
}

QFrame[class="subsection"] { border-bottom: 1px solid #e0e6ee; }
QLabel[class="subsectionTitle"] { color: #1f2a37; }
QLabel[class="subsectionHint"] { color: #5d6a7d; }
QLabel[class="hint"], QLabel[class="muted"] { color: #5d6a7d; }
QLabel[class="emptyState"] { color: #98a3b5; }
QLabel[class="emptyStateTitle"] { color: #4a5568; }
QFrame[class="footerActions"] {
    background: #eef1f7; border: 1px solid #d6dde9;
}
QFrame[class="inlineStatus"] {
    background: #f6f8fc; border: 1px solid #d6dde9;
}
QFrame[class="inlineStatus"][severity="ok"]    { border: 1px solid #2f7a4a; background: #e3f3eb; color: #1f3d2c; }
QFrame[class="inlineStatus"][severity="warn"]  { border: 1px solid #d6a36c; background: #fff5e6; color: #5a3d18; }
QFrame[class="inlineStatus"][severity="error"] { border: 1px solid #c44456; background: #fdecee; color: #5a1c25; }
QFrame[class="inlineStatus"] QLabel { color: #1f2a37; }

/* Dock widgets */
QDockWidget { background: #ffffff; color: #1f2a37; border: 1px solid #d6dde9; }
QDockWidget::title { background: #eef1f7; color: #4a5568; border-bottom: 1px solid #d6dde9; }
QDockWidget > QWidget { background: #ffffff; }

/* Menus */
QMenu { background: #ffffff; color: #1f2a37; border: 1px solid #c2ccda; }
QMenu::item:selected { background: #2563eb; color: #ffffff; }
QMenuBar { background: #eef1f7; color: #1f2a37; border-bottom: 1px solid #d6dde9; }
QMenuBar::item:selected { background: #e6e9f1; }

QToolButton {
    background: #ffffff; color: #1f2a37; border: 1px solid #d6dde9;
}
QToolButton:hover { background: #eff3fa; border: 1px solid #c2ccda; }

QCheckBox::indicator {
    border: 1px solid #c2ccda; background: #ffffff;
}
QCheckBox::indicator:checked { background: #7d4df2; border: 1px solid #8d60ff; }

QToolTip {
    background: #ffffff; color: #1f2a37;
    border: 1px solid #b9a9df;
}
"""


APP_QSS_LIGHT = _build_light_qss(APP_QSS) + _LIGHT_OVERRIDES


def app_qss(theme_mode: object) -> str:
    mode = str(theme_mode or "").strip().lower()
    if mode in {"light", "white", "l", "w"}:
        return APP_QSS_LIGHT
    return APP_QSS


def apply_app_palette(app, theme_mode: object) -> None:
    """Force a consistent Fusion palette before applying app QSS.

    Some Windows/native Qt styles partially ignore dark QSS for menus, popup
    views, disabled controls, or newly-created widgets. Fusion plus an explicit
    palette keeps the app theme independent from the host OS theme.
    """
    from PySide6 import QtGui, QtWidgets

    if app is None:
        return

    mode = str(theme_mode or "").strip().lower()
    light = mode in {"light", "white", "l", "w"}
    try:
        QtWidgets.QApplication.setStyle("Fusion")
    except Exception:
        pass

    palette = QtGui.QPalette()
    if light:
        colors = {
            QtGui.QPalette.ColorRole.Window: "#f6f8fc",
            QtGui.QPalette.ColorRole.WindowText: "#1f2a37",
            QtGui.QPalette.ColorRole.Base: "#ffffff",
            QtGui.QPalette.ColorRole.AlternateBase: "#edf1f7",
            QtGui.QPalette.ColorRole.ToolTipBase: "#ffffff",
            QtGui.QPalette.ColorRole.ToolTipText: "#1f2a37",
            QtGui.QPalette.ColorRole.Text: "#1f2a37",
            QtGui.QPalette.ColorRole.Button: "#e7ecf4",
            QtGui.QPalette.ColorRole.ButtonText: "#1f2a37",
            QtGui.QPalette.ColorRole.BrightText: "#ffffff",
            QtGui.QPalette.ColorRole.Highlight: "#378ef0",
            QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
        }
    else:
        colors = {
            QtGui.QPalette.ColorRole.Window: "#1a1d26",
            QtGui.QPalette.ColorRole.WindowText: "#e9ecf3",
            QtGui.QPalette.ColorRole.Base: "#1b1f28",
            QtGui.QPalette.ColorRole.AlternateBase: "#20242d",
            QtGui.QPalette.ColorRole.ToolTipBase: "#1f242e",
            QtGui.QPalette.ColorRole.ToolTipText: "#e9ecf3",
            QtGui.QPalette.ColorRole.Text: "#e9ecf3",
            QtGui.QPalette.ColorRole.Button: "#262b35",
            QtGui.QPalette.ColorRole.ButtonText: "#e9ecf3",
            QtGui.QPalette.ColorRole.BrightText: "#ffffff",
            QtGui.QPalette.ColorRole.Highlight: "#7d4df2",
            QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
        }
    for role, color in colors.items():
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)
