# styles.py
from __future__ import annotations

import os

# Tool drawings are code-native vectors shared by both workflow rails.
from tool_icons import (
    _make_icon, _paint_database, _paint_list, _paint_sliders, _paint_artifacts,
    _paint_filter, _paint_wave, _paint_chart, _paint_output, _paint_badge,
    _paint_export, _paint_gear, _paint_grid, _paint_target, _paint_sync,
    _paint_pulse, _paint_paw, _paint_temporal,
)


# ===========================================================================
# pyBer "Obsidian" design system
#
#   DARK (primary identity)                  Accents / status
#   -------------------------------------    -----------------------------
#   #0c0f16 plot void (data glows)           #7c5cff accent (iris violet)
#   #10131c shell (rail, bars, chrome)       #916dff accent hover
#   #131722 well (footer strips)             #6a4bf0 accent pressed
#   #141824 inset (inputs, lists)            #372f66 accent soft fill
#   #161a26 panel (drawers, dialogs)         #4d3fa3 accent soft border
#   #1a1f2e card (frames, menus)             #43d9a3 success
#   #1c2132 -> #191e2c card gradient         #f6c453 warn
#   #1d2333 item hover                       #f26d7e error
#   #222840 raised (buttons)                 #4da3ff info
#   #293049 raised hover
#
#   Borders: #232a3d subtle / #333c56 default / #465073 strong
#   Text:    #f7f9fd display / #edf0f8 body / #a9b3c9 secondary
#            #8f99b3 caps-label / #707b93 muted / #4b5470 faint
#
#   Type: Segoe UI Variable (Display for titles, Text for body), 9.5pt body,
#         8.2pt letter-spaced uppercase micro-labels.
#   Radii: 7 inputs / 8 buttons / 12 cards / 14 outer shell panels.
#
# The light theme is derived via _LIGHT_COLOR_MAP + _LIGHT_OVERRIDES below.
# PLOT_THEME is the single source of truth for pyqtgraph surfaces so the
# preprocessing / postprocessing / QC / temporal plots finally agree.
# ===========================================================================

PLOT_THEME = {
    "dark": {
        "bg": (12, 15, 22),
        "axis": (86, 96, 121),
        "text": (150, 160, 184),
        "title": "#d9dfec",
        "grid_alpha": 0.22,
    },
    "light": {
        "bg": (250, 251, 254),
        "axis": (71, 81, 104),
        "text": (49, 60, 80),
        "title": "#172033",
        "grid_alpha": 0.24,
    },
}


_QSS_TEMPLATE = r"""
QMainWindow, QDialog, QWidget {
    background: #161a26;
    color: #edf0f8;
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", "Arial", sans-serif;
    font-size: 9.8pt;
}

QLabel {
    color: #edf0f8;
    background: transparent;
}

QLabel[class="hint"], QLabel[class="muted"] {
    color: #a9b3c9;
    font-size: 9.0pt;
}

QLabel[class="title"] {
    color: #f7f9fd;
    font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
    font-size: 12.5pt;
    font-weight: 700;
    letter-spacing: 0.2px;
}

QLabel[class="badge"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #8a63ff, stop:1 #6f4df2);
    color: #ffffff;
    border-radius: 9px;
    padding: 2px 9px;
    font-weight: 700;
    font-size: 8.2pt;
    letter-spacing: 0.4px;
}

/* Panels: soft-gradient cards, lit faintly from above */
QGroupBox {
    border: 1px solid #232a3d;
    border-radius: 12px;
    margin-top: 18px;
    padding: 14px 14px 12px 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1c2132, stop:1 #191e2c);
}

/* Keep helper row-container widgets inside group boxes transparent. */
QGroupBox > QWidget {
    background: transparent;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 10px;
    margin-left: 10px;
    color: #8f99b3;
    background: #10131c;
    border: 1px solid #232a3d;
    border-radius: 7px;
    font-weight: 800;
    font-size: 8.5pt;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

/* Generic in-panel section header (used by the Subsection helper). */
QFrame[class="subsection"] {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #232a3d;
}

QLabel[class="subsectionTitle"] {
    color: #edf0f8;
    font-size: 9.9pt;
    font-weight: 700;
    letter-spacing: 0.2px;
    padding: 2px 0;
}

QLabel[class="subsectionHint"] {
    color: #a9b3c9;
    font-size: 8.8pt;
    padding: 0;
}

/* Footer action strip pinned to the bottom of a panel. */
QFrame[class="footerActions"] {
    background: #131722;
    border: 1px solid #232a3d;
    border-radius: 10px;
    padding: 8px 12px;
}

/* Inline status banner (used by InlineStatus helper). */
QFrame[class="inlineStatus"] {
    background: #1a1f2e;
    border: 1px solid #232a3d;
    border-radius: 9px;
    padding: 6px 10px;
}

QFrame[class="inlineStatus"][severity="ok"]    { border: 1px solid #2a6b51; background: #142a24; }
QFrame[class="inlineStatus"][severity="warn"]  { border: 1px solid #94713a; background: #2b2517; }
QFrame[class="inlineStatus"][severity="error"] { border: 1px solid #8f3d4d; background: #2c1a20; }
QFrame[class="inlineStatus"] QLabel { background: transparent; color: #ccd4e4; font-size: 9.0pt; }

/* Collapsible chevron button (rotated when expanded). */
QToolButton[class="chevron"] {
    background: transparent;
    border: 0;
    color: #a9b3c9;
    padding: 0 4px;
    font-size: 11pt;
    font-weight: 700;
}

QToolButton[class="chevron"]:hover {
    color: #ffffff;
}

QFrame {
    border-color: #232a3d;
}

/* Lists and tables */
QListWidget, QTableWidget, QTreeWidget {
    background: #141824;
    border: 1px solid #232a3d;
    border-radius: 9px;
    padding: 4px;
    gridline-color: #232a3d;
    alternate-background-color: #171c2b;
}

QListWidget::item, QTableWidget::item {
    padding: 4px 6px;
    border-radius: 5px;
}

QListWidget::item:hover, QTableWidget::item:hover {
    background: #1d2333;
}

QListWidget::item:selected, QTableWidget::item:selected {
    background: #372f66;
    color: #ffffff;
    border-radius: 5px;
}

QHeaderView::section {
    background: #1d2333;
    color: #a9b3c9;
    border: 0;
    border-bottom: 1px solid #232a3d;
    padding: 6px 8px;
    font-weight: 700;
    font-size: 8.6pt;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #141824;
    color: #edf0f8;
    border: 1px solid #333c56;
    border-radius: 7px;
    padding: 5px 8px;
    selection-background-color: #7c5cff;
    selection-color: #ffffff;
    min-height: 22px;
}

QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {
    border: 1px solid #465073;
    background: #171c2b;
}

QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 1px solid #7c5cff;
    background: #171c2b;
}

QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    color: #707b93;
    border: 1px solid #232a3d;
    background: #131722;
}

QComboBox::drop-down {
    border: 0;
    width: 24px;
    background: transparent;
}

QComboBox::down-arrow {
    %CHEVRON_RULE%
    margin-right: 6px;
}

QComboBox QAbstractItemView {
    background: #1a1f2e;
    color: #edf0f8;
    border: 1px solid #333c56;
    border-radius: 9px;
    padding: 4px;
    selection-background-color: #372f66;
    selection-color: #ffffff;
    outline: 0;
}

QCheckBox {
    spacing: 7px;
    color: #edf0f8;
    background: transparent;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 5px;
    border: 1px solid #465073;
    background: #141824;
}

QCheckBox::indicator:hover {
    border: 1px solid #916dff;
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8a63ff, stop:1 #6f4df2);
    border: 1px solid #8f74ff;
    %CHECK_RULE%
}

QRadioButton {
    background: transparent;
}

QRadioButton::indicator {
    width: 15px;
    height: 15px;
    border-radius: 8px;
    border: 1px solid #465073;
    background: #141824;
}

QRadioButton::indicator:hover {
    border: 1px solid #916dff;
}

QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 #ffffff, stop:0.45 #ffffff, stop:0.5 #7c5cff, stop:1 #7c5cff);
    border: 1px solid #8f74ff;
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #262c46, stop:1 #222840);
    border: 1px solid #333c56;
    border-radius: 8px;
    padding: 7px 14px;
    font-weight: 600;
    color: #edf0f8;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2d3450, stop:1 #293049);
    border: 1px solid #465073;
}

QPushButton:pressed {
    background: #1d2333;
    border: 1px solid #333c56;
}

QPushButton:disabled {
    color: #707b93;
    background: #1a1f2e;
    border: 1px solid #232a3d;
}

QPushButton[class="compact"] {
    padding: 5px 10px;
    border-radius: 7px;
    font-weight: 600;
}

QPushButton[class="compactSmall"] {
    padding: 4px 11px;
    border-radius: 7px;
    font-weight: 600;
    font-size: 8.9pt;
    min-height: 22px;
    color: #ccd4e4;
}

QPushButton[class="compactSmall"]:hover {
    color: #ffffff;
}

/* Primary action: gradient-lit iris violet */
QPushButton[class="primary"],
QPushButton[class="compactPrimary"],
QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8a63ff, stop:1 #6f4df2);
    border: 1px solid #8f74ff;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[class="primary"] {
    padding: 8px 18px;
    border-radius: 9px;
}

QPushButton[class="compactPrimary"] {
    padding: 5px 12px;
    border-radius: 8px;
}

QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    padding: 4px 13px;
    border-radius: 7px;
    font-size: 8.9pt;
    min-height: 22px;
}

QPushButton[class="primary"]:hover,
QPushButton[class="compactPrimary"]:hover,
QPushButton[class="compactPrimarySmall"]:hover,
QPushButton[class="bluePrimarySmall"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9a77ff, stop:1 #7d5cff);
    border: 1px solid #a68cff;
}

QPushButton[class="primary"]:pressed,
QPushButton[class="compactPrimary"]:pressed,
QPushButton[class="compactPrimarySmall"]:pressed,
QPushButton[class="bluePrimarySmall"]:pressed {
    background: #6146e0;
    border: 1px solid #7c5cff;
}

QPushButton[class="primary"]:disabled,
QPushButton[class="compactPrimary"]:disabled,
QPushButton[class="compactPrimarySmall"]:disabled {
    background: #2a2450;
    border: 1px solid #3c3474;
    color: #b9a9f2;
}

QPushButton[class="blueSecondarySmall"] {
    padding: 4px 13px;
    border-radius: 7px;
    font-weight: 600;
    font-size: 8.9pt;
    min-height: 22px;
    background: #241f42;
    border: 1px solid #4d3fa3;
    color: #e6def8;
}

QPushButton[class="blueSecondarySmall"]:hover {
    background: #2e2854;
    border: 1px solid #5f4fc2;
}

QPushButton[class="blueSecondarySmall"]:pressed {
    background: #1e1a37;
}

QPushButton[class="blueSecondarySmall"]:checked,
QPushButton[class="sectionButton"]:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8a63ff, stop:1 #6f4df2);
    border: 1px solid #8f74ff;
    color: #ffffff;
}

/* Ghost button: no fill, used for non-primary toolbar actions (Undo/Redo). */
QPushButton[class="ghost"] {
    background: transparent;
    border: 1px solid transparent;
    color: #a9b3c9;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 7px;
}

QPushButton[class="ghost"]:hover {
    background: rgba(124, 92, 255, 0.10);
    color: #ffffff;
    border: 1px solid rgba(124, 92, 255, 0.35);
}

QPushButton[class="ghost"]:pressed {
    background: rgba(124, 92, 255, 0.18);
}

QPushButton[class="sectionButton"] {
    padding: 5px 11px;
    border-radius: 7px;
    font-weight: 600;
    font-size: 8.8pt;
    background: #1d2333;
    border: 1px solid #232a3d;
    text-align: left;
    color: #ccd4e4;
}

QPushButton[class="sectionButton"]:hover {
    background: #293049;
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
    background: #1d2333;
    border: 1px solid #333c56;
    color: #a9b3c9;
}

QPushButton[class="help"]:hover {
    background: #293049;
    border: 1px solid #7c5cff;
    color: #ffffff;
}

/* Toggleable rail-like icon button. */
QToolButton[class="iconRail"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 6px;
}

QToolButton[class="iconRail"]:hover {
    background: rgba(124, 92, 255, 0.10);
    border: 1px solid rgba(124, 92, 255, 0.30);
}

QToolButton[class="iconRail"]:checked {
    background: #372f66;
    border: 1px solid #7c5cff;
}

/* Tabs (clean underline indicator) */
QTabWidget::pane {
    border: 0;
    border-top: 1px solid #232a3d;
    background: #161a26;
    padding: 4px;
}

QTabBar::tab {
    background: transparent;
    border: 0;
    padding: 9px 18px;
    margin: 0 2px 0 0;
    font-weight: 600;
    color: #a9b3c9;
    border-bottom: 2px solid transparent;
}

QTabBar::tab:hover {
    color: #ffffff;
    background: rgba(124, 92, 255, 0.08);
    border-bottom: 2px solid #4d3fa3;
}

QTabBar::tab:selected {
    background: transparent;
    color: #ffffff;
    border-bottom: 2px solid #7c5cff;
}

/* Menus and tool buttons */
QMenu {
    background: #1a1f2e;
    color: #edf0f8;
    border: 1px solid #333c56;
    border-radius: 10px;
    padding: 5px;
}

QMenu::item {
    padding: 6px 20px;
    border-radius: 6px;
}

QMenu::item:selected {
    background: #372f66;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background: #232a3d;
    margin: 4px 8px;
}

QMenuBar {
    background: #161a26;
    color: #edf0f8;
    border-bottom: 1px solid #232a3d;
    padding: 2px;
}

QMenuBar::item {
    background: transparent;
    padding: 5px 11px;
    border-radius: 6px;
}

QMenuBar::item:selected {
    background: #1d2333;
}

QToolButton {
    background: #1d2333;
    border: 1px solid #232a3d;
    border-radius: 7px;
    padding: 4px 8px;
    color: #edf0f8;
}

QToolButton:hover {
    background: #293049;
    border: 1px solid #333c56;
}

QToolButton:pressed {
    background: #1a1f2e;
}

/* Docking and splitters */
QDockWidget {
    background: #1a1f2e;
    color: #edf0f8;
    border: 1px solid #232a3d;
}

QDockWidget::title {
    background: #1d2333;
    border-bottom: 1px solid #232a3d;
    padding: 7px 10px;
    text-align: left;
    font-weight: 700;
    font-size: 8.7pt;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #a9b3c9;
}

QDockWidget > QWidget {
    background: #1a1f2e;
}

QMainWindow::separator {
    background: #232a3d;
    width: 4px;
    height: 4px;
}

QMainWindow::separator:hover {
    background: #7c5cff;
}

QSplitter::handle {
    background: #232a3d;
}

QSplitter::handle:hover {
    background: #7c5cff;
}

QScrollArea {
    border: none;
    background: transparent;
}

/* Plot frame support */
QGraphicsView {
    background: #0c0f16;
    border: 1px solid #232a3d;
    border-radius: 10px;
}

QToolTip {
    background: #10131c;
    color: #edf0f8;
    border: 1px solid #4d3fa3;
    border-radius: 6px;
    padding: 5px 9px;
}

/* Status bar */
QStatusBar {
    background: #10131c;
    color: #a9b3c9;
    border-top: 1px solid #232a3d;
}

QStatusBar::item {
    border: 0;
}

QStatusBar QLabel {
    color: #a9b3c9;
    padding: 0 4px;
}

/* Scroll bars (slim, themed) */
QScrollBar:vertical {
    background: transparent;
    width: 9px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2b3450;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #4d3fa3;
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
    height: 9px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #2b3450;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #4d3fa3;
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
    background: #141824;
    border: 1px solid #232a3d;
    border-radius: 7px;
    text-align: center;
    color: #edf0f8;
    height: 18px;
    font-size: 8.4pt;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c5cff, stop:1 #9d7bff);
    border-radius: 6px;
    margin: 1px;
}

QFrame#pyberExportProgressWidget {
    background: transparent;
    border: 0;
}

QProgressBar#pyberExportProgressBar {
    background: #222840;
    border: 0;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
}

QProgressBar#pyberExportProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #43d9a3, stop:1 #7c5cff);
    border-radius: 3px;
    margin: 0;
}

/* ---------- Modern shell: side rail, drawers, transport bar ---------- */
QFrame#sideRail {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10131c, stop:1 #11141e);
    border: 1px solid #232a3d;
    border-radius: 14px;
}

QFrame#drawerPanel, QFrame#centerPanel {
    background: #161a26;
    border: 1px solid #232a3d;
    border-radius: 14px;
}

QFrame#transportBar {
    background: #10131c;
    border: 1px solid #232a3d;
    border-radius: 12px;
}

QFrame#SidePanel, QFrame#syncPanel {
    background: #1a1f2e;
    border: 1px solid #2b3450;
    border-radius: 9px;
}

QFrame#SidePanel QListWidget {
    background: #141824;
    border: 1px solid #333c56;
    border-radius: 9px;
    padding: 4px;
}

QFrame#SidePanel QListWidget::item {
    padding: 8px;
    border-radius: 7px;
    margin: 2px 0;
}

QFrame#SidePanel QListWidget::item:selected {
    background: #372f66;
    color: #ffffff;
}

QLabel#BadgeLabel {
    background: #1b2a3d;
    border: 1px solid #365a80;
    border-radius: 10px;
    color: #9ed9f4;
    padding: 4px 8px;
    font-weight: 700;
}

QToolButton#toolbarIconButton {
    background: #1a1f2e;
    border: 1px solid #2b3450;
    border-radius: 8px;
    color: #edf0f8;
    min-width: 30px;
    min-height: 26px;
    padding: 0;
    font-size: 13pt;
    font-weight: 800;
}

QToolButton#toolbarIconButton:hover {
    background: #232840;
    border: 1px solid #7c5cff;
    color: #ffffff;
}

QToolButton#toolbarIconButton:pressed {
    background: #7c5cff;
    border: 1px solid #916dff;
    color: #ffffff;
}

QToolButton#toolbarIconButton:disabled {
    background: #131722;
    border: 1px solid #232a3d;
    color: #4b5470;
}

QFrame#railSeparator {
    background: #232a3d;
    max-height: 1px;
    min-height: 1px;
    border: none;
    margin: 6px 10px;
}

/* Thin vertical divider between toolbar button groups. */
QFrame#toolbarSeparator {
    background: #2b3450;
    min-width: 1px;
    max-width: 1px;
    border: none;
    margin: 5px 3px;
}

QLabel#panelTitle {
    color: #f7f9fd;
    font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
    font-size: 13pt;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 2px 4px;
}

QFrame#pyberPanelHeader {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #232a3d;
    padding: 0;
}

QFrame#plotEmptyWorkspace {
    background: #141824;
    border: 1px solid #232a3d;
    border-radius: 12px;
}
QLabel#plotEmptyTitle {
    background: transparent;
    border: 0;
    color: #a9b3c9;
    font-family: "Segoe UI Variable Display", "Segoe UI", sans-serif;
    font-size: 12pt;
    font-weight: 500;
}
QLabel#plotEmptyHint {
    background: transparent;
    border: 0;
    color: #707b93;
    font-size: 9pt;
    font-weight: 400;
}

QFrame#pyberPanelHeader QLabel {
    background: transparent;
}

QLabel#pyberPanelHeaderTitle {
    color: #f7f9fd;
    font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: 0px;
}

QLabel#pyberPanelHeaderSubtitle {
    color: #a9b3c9;
    font-size: 9.0pt;
}

QLabel#transportStatus {
    color: #a9b3c9;
    font-size: 9pt;
    font-weight: 600;
    padding: 0 6px;
}

QFrame#pyberBusyWidget {
    background: #232746;
    border: 1px solid #45509a;
    border-radius: 7px;
    padding: 0 8px;
}

QFrame#pyberBusyWidget QLabel {
    background: transparent;
    color: #d5ddf0;
}

QFrame#pyberBusyWidget QPushButton {
    background: #4a2b33;
    color: #ffd3da;
    border: 1px solid #8f3d4d;
    border-radius: 5px;
    padding: 1px 8px;
}

QFrame#pyberBusyWidget QPushButton:hover {
    background: #5e343e;
}

QPushButton#railButton, QPushButton#railToggleButton,
QToolButton#railButton, QToolButton#railToggleButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 11px;
    min-width: 0;
    padding: 8px;
    text-align: center;
    font-weight: 600;
    font-size: 8.1pt;
    color: #a9b3c9;
}

QPushButton#railButton:hover,
QPushButton#railToggleButton:hover,
QToolButton#railButton:hover,
QToolButton#railToggleButton:hover {
    background: rgba(124, 92, 255, 0.10);
    border: 1px solid rgba(124, 92, 255, 0.30);
    color: #ffffff;
}

QPushButton#railButton:checked,
QPushButton#railToggleButton:checked,
QToolButton#railButton:checked,
QToolButton#railToggleButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #43397e, stop:1 #372f66);
    border: 1px solid #7c5cff;
    color: #ffffff;
}

QPushButton#railButton:disabled,
QPushButton#railToggleButton:disabled,
QToolButton#railButton:disabled,
QToolButton#railToggleButton:disabled {
    background: transparent;
    color: #4b5470;
    border: 1px solid transparent;
}

QTabBar#visualModeBar::tab {
    min-width: 90px;
    padding: 5px 14px;
    border-radius: 7px;
    margin-right: 4px;
    font-weight: 700;
    background: #1d2333;
    color: #a9b3c9;
    border-bottom: 2px solid transparent;
}

QTabBar#visualModeBar::tab:hover:!selected {
    background: #293049;
    color: #ffffff;
}

QTabBar#visualModeBar::tab:selected {
    background: #372f66;
    color: #ffffff;
    border: 1px solid #7c5cff;
}

/* ---------- Top app bar (workflow header) ---------- */
QFrame#pyberTopBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                stop:0 #1a1f2e, stop:1 #12151f);
    border: 0;
    border-bottom: 1px solid #232a3d;
}

QLabel#pyberAppName {
    color: #ffffff;
    font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
    font-size: 12.5pt;
    font-weight: 800;
    letter-spacing: 0.4px;
    padding: 0 6px;
}

QLabel#pyberWorkflowStep {
    color: #a9b3c9;
    font-size: 9pt;
    font-weight: 600;
    padding: 5px 10px;
    border-radius: 9px;
}

QLabel#pyberWorkflowStep[active="true"] {
    color: #ffffff;
    background: #372f66;
}

QLabel#pyberWorkflowSep {
    color: #4b5470;
    font-size: 11pt;
    font-weight: 700;
}

QLabel#pyberProjectName {
    color: #a9b3c9;
    font-size: 9pt;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9px;
    background: #161a26;
    border: 1px solid #232a3d;
}

QLabel#pyberProjectName[dirty="true"] {
    color: #f6c453;
    border: 1px solid #5c4a17;
}

/* ---------- Recommendation panel (preprocessing advice card) ---------- */
QFrame#recoPanel {
    background: rgba(67, 217, 163, 8%);
    border: 1px solid #43d9a3;
    border-radius: 10px;
}

QFrame#recoPanel QLabel {
    background: transparent;
    border: 0;
}

QLabel[reco="title"] {
    color: #43d9a3;
    font-size: 9.8pt;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

QLabel[reco="headline"] {
    color: #edf0f8;
    font-size: 9.6pt;
    font-weight: 600;
}

QLabel[reco="key"] {
    color: #8f99b3;
    font-size: 8.8pt;
}

QLabel[reco="value"] {
    color: #43d9a3;
    font-size: 8.8pt;
    font-weight: 700;
}

QLabel[reco="why"] {
    color: #a9b3c9;
    font-size: 8.5pt;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding-top: 4px;
}

QLabel[reco="body"] {
    color: #ccd4e4;
    font-size: 8.8pt;
}

/* ---------- Reusable polish primitives ---------- */
QFrame[class="card"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1c2132, stop:1 #191e2c);
    border: 1px solid #232a3d;
    border-radius: 11px;
}

QFrame[class="callout"] {
    background: #1a1f2e;
    border: 1px solid #4d3fa3;
    border-radius: 11px;
}

QFrame[class="hairline"] {
    background: #232a3d;
    max-height: 1px;
    min-height: 1px;
    border: 0;
}

/* Empty-state hint label inside otherwise-blank panels. */
QLabel[class="emptyState"] {
    color: #707b93;
    font-size: 10pt;
    font-weight: 500;
    padding: 24px;
}

QLabel[class="emptyStateTitle"] {
    color: #a9b3c9;
    font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
    font-size: 13pt;
    font-weight: 700;
    padding: 6px;
}
"""


# ---------------------------------------------------------------------------
# Light ("Porcelain") theme: derived from the dark QSS via color mapping,
# then patched with explicit overrides for states the mapping cannot express.
# Every 6-digit hex appearing in _QSS_TEMPLATE MUST have an entry here
# (tests/scripts can verify by extracting hexes from the template).
# ---------------------------------------------------------------------------

_LIGHT_COLOR_MAP = {
    # surfaces
    "#0c0f16": "#fafbfe",
    "#10131c": "#e9edf5",
    "#11141e": "#eef1f8",
    "#12151f": "#eef1f7",
    "#131722": "#eef1f7",
    "#141824": "#ffffff",
    "#161a26": "#f2f4f9",
    "#171c2b": "#f6f8fc",
    "#191e2c": "#fdfdfe",
    "#1a1f2e": "#ffffff",
    "#1c2132": "#ffffff",
    "#1d2333": "#e9edf5",
    "#222840": "#e7ecf5",
    "#232746": "#fff4e2",
    "#232840": "#dde6f4",
    "#262c46": "#ffffff",
    "#293049": "#dde3f0",
    "#2b3450": "#c9d2e2",
    "#2d3450": "#f3f5fa",
    # borders
    "#232a3d": "#d9dfec",
    "#333c56": "#c3ccdd",
    "#465073": "#a8b3c9",
    "#45509a": "#f0b660",
    # text
    "#f7f9fd": "#141b28",
    "#edf0f8": "#1b2434",
    "#ccd4e4": "#3c4a60",
    "#a9b3c9": "#506078",
    "#8f99b3": "#5d6a82",
    "#707b93": "#8792a8",
    "#4b5470": "#a6b0c2",
    "#d5ddf0": "#7a4b12",
    # accent family
    "#7c5cff": "#6d4de0",
    "#916dff": "#7d5df0",
    "#9a77ff": "#8d70f2",
    "#9d7bff": "#8d70f2",
    "#8a63ff": "#7a5ae8",
    "#7d5cff": "#6d4de0",
    "#6f4df2": "#5f41d6",
    "#6146e0": "#5539c8",
    "#6a4bf0": "#5f41d6",
    "#8f74ff": "#8a6cee",
    "#a68cff": "#a48eef",
    "#372f66": "#e2dbf8",
    "#43397e": "#eae4fb",
    "#4d3fa3": "#c4b4f0",
    "#5f4fc2": "#b39ff0",
    "#241f42": "#efeafc",
    "#2e2854": "#e2d8fa",
    "#1e1a37": "#d8ccf6",
    "#2a2450": "#cabcf0",
    "#3c3474": "#baa8ea",
    "#b9a9f2": "#8f7ad8",
    "#e6def8": "#43307e",
    # status
    "#43d9a3": "#1d9e6e",
    "#f6c453": "#b07d18",
    "#f26d7e": "#d14459",
    "#2a6b51": "#7fcfae",
    "#142a24": "#e2f5ec",
    "#94713a": "#e2bc7a",
    "#2b2517": "#fdf3dd",
    "#8f3d4d": "#eba4af",
    "#2c1a20": "#fdeaed",
    "#5c4a17": "#d9c584",
    "#4a2b33": "#d14459",
    "#ffd3da": "#ffffff",
    "#5e343e": "#b93348",
    # cyan badge
    "#1b2a3d": "#ddedf8",
    "#365a80": "#8cc3e4",
    "#9ed9f4": "#155a80",
    # white stays white
    "#ffffff": "#ffffff",
}


def _build_light_qss(dark_qss: str) -> str:
    light = str(dark_qss)
    # Replace longer tokens first to avoid accidental partial replacement collisions.
    for dark, bright in sorted(_LIGHT_COLOR_MAP.items(), key=lambda kv: len(kv[0]), reverse=True):
        light = light.replace(dark, bright)
    return light


# The find/replace mapping above can leave low-contrast text in :hover /
# :selected / :checked states (because `color: #ffffff` stays white). These
# overrides come last and win against the auto-mapped rules.
_LIGHT_OVERRIDES = r"""
/* ---------- Porcelain light-mode contrast fixes ---------- */

QMainWindow, QDialog, QWidget {
    background: #f2f4f9;
    color: #1b2434;
}

/* Tabs: dark text on hover / selected, accent underline */
QTabWidget::pane { background: #f2f4f9; border-top: 1px solid #d9dfec; }
QTabBar::tab { color: #506078; background: transparent; }
QTabBar::tab:hover {
    color: #1b2434;
    background: rgba(109, 77, 224, 0.07);
    border-bottom: 2px solid #c4b4f0;
}
QTabBar::tab:selected {
    color: #1b2434;
    background: transparent;
    border-bottom: 2px solid #6d4de0;
}

/* Group boxes as soft white cards */
QGroupBox {
    border: 1px solid #dde3ee;
    background: #ffffff;
}
QGroupBox::title {
    background: #eef1f7;
    border: 1px solid #dde3ee;
    color: #5d6a82;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #ffffff;
    color: #1b2434;
    border: 1px solid #c3ccdd;
    selection-background-color: #6d4de0;
    selection-color: #ffffff;
}
QLineEdit:hover, QAbstractSpinBox:hover, QComboBox:hover {
    border: 1px solid #a8b3c9;
    background: #ffffff;
}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 1px solid #6d4de0;
    background: #ffffff;
}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {
    color: #8792a8;
    border: 1px solid #e0e6ef;
    background: #f4f6fa;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #1b2434;
    border: 1px solid #c3ccdd;
    selection-background-color: #e2dbf8;
    selection-color: #2a1f5e;
}

/* Lists / tables */
QListWidget, QTableWidget, QTreeWidget {
    background: #ffffff;
    border: 1px solid #dde3ee;
    color: #1b2434;
    alternate-background-color: #f6f8fc;
}
QListWidget::item:hover, QTableWidget::item:hover { background: #eff2f9; }
QListWidget::item:selected, QTableWidget::item:selected {
    background: #e2dbf8; color: #2a1f5e;
}
QHeaderView::section {
    background: #eef1f7; color: #506078; border: 0;
    border-bottom: 1px solid #dde3ee;
}

/* Buttons: dark text everywhere; primary violet keeps white text */
QPushButton {
    background: #ffffff; color: #1b2434; border: 1px solid #c3ccdd;
}
QPushButton:hover { background: #f3f1fc; border: 1px solid #c4b4f0; }
QPushButton:pressed { background: #e9e4f8; }
QPushButton:disabled { color: #a0aabc; background: #f4f6fa; border: 1px solid #e0e6ef; }

QPushButton[class="ghost"] { color: #506078; background: transparent; border: 1px solid transparent; }
QPushButton[class="ghost"]:hover {
    background: rgba(109, 77, 224, 0.08);
    color: #1b2434;
    border: 1px solid rgba(109, 77, 224, 0.30);
}

QPushButton[class="primary"], QPushButton[class="compactPrimary"],
QPushButton[class="compactPrimarySmall"], QPushButton[class="bluePrimarySmall"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7a5ae8, stop:1 #5f41d6);
    border: 1px solid #8a6cee; color: #ffffff;
}
QPushButton[class="primary"]:hover, QPushButton[class="compactPrimary"]:hover,
QPushButton[class="compactPrimarySmall"]:hover, QPushButton[class="bluePrimarySmall"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8d70f2, stop:1 #6d4de0);
    border: 1px solid #a48eef; color: #ffffff;
}
QPushButton[class="compactSmall"] { color: #506078; }
QPushButton[class="compactSmall"]:hover { color: #1b2434; }

QPushButton[class="sectionButton"] {
    background: #ffffff; border: 1px solid #dde3ee; color: #1b2434;
}
QPushButton[class="sectionButton"]:hover { background: #f3f1fc; }
QPushButton[class="blueSecondarySmall"] {
    background: #f1edfd; border: 1px solid #c4b4f0; color: #43307e;
}
QPushButton[class="blueSecondarySmall"]:hover {
    background: #e6defa; border: 1px solid #b39ff0;
}
QPushButton[class="blueSecondarySmall"]:checked,
QPushButton[class="sectionButton"]:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7a5ae8, stop:1 #5f41d6);
    border: 1px solid #8a6cee; color: #ffffff;
}

/* Side rail (icon + label under) */
QFrame#sideRail { background: #e9edf5; border: 1px solid #d9dfec; }
QToolButton#railButton, QToolButton#railToggleButton,
QPushButton#railButton, QPushButton#railToggleButton {
    background: transparent; border: 1px solid transparent; color: #506078;
}
QToolButton#railButton:hover, QToolButton#railToggleButton:hover,
QPushButton#railButton:hover, QPushButton#railToggleButton:hover {
    background: rgba(109, 77, 224, 0.09);
    border: 1px solid rgba(109, 77, 224, 0.30);
    color: #1b2434;
}
QToolButton#railButton:checked, QToolButton#railToggleButton:checked,
QPushButton#railButton:checked, QPushButton#railToggleButton:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7a5ae8, stop:1 #5f41d6);
    border: 1px solid #8a6cee; color: #ffffff;
}

/* Drawer panels */
QFrame#drawerPanel, QFrame#centerPanel { background: #ffffff; border: 1px solid #d9dfec; }
QFrame#transportBar { background: #e9edf5; border: 1px solid #d9dfec; }
QFrame#SidePanel, QFrame#syncPanel { background: #ffffff; border: 1px solid #d9dfec; border-radius: 9px; }
QFrame#SidePanel QListWidget { background: #f7f9fc; border: 1px solid #d9dfec; border-radius: 9px; padding: 4px; }
QFrame#SidePanel QListWidget::item { padding: 8px; border-radius: 7px; margin: 2px 0; }
QFrame#SidePanel QListWidget::item:selected { background: #e2dbf8; color: #2a1f5e; }
QLabel#BadgeLabel {
    background: #ddedf8;
    border: 1px solid #8cc3e4;
    border-radius: 10px;
    color: #155a80;
    padding: 4px 8px;
    font-weight: 700;
}
QToolButton#toolbarIconButton {
    background: #ffffff;
    border: 1px solid #c3ccdd;
    border-radius: 8px;
    color: #172033;
    min-width: 30px;
    min-height: 26px;
    padding: 0;
    font-size: 13pt;
    font-weight: 800;
}
QToolButton#toolbarIconButton:hover {
    background: #f3f1fc;
    border: 1px solid #8a6cee;
    color: #0f172a;
}
QToolButton#toolbarIconButton:pressed {
    background: #6d4de0;
    border: 1px solid #5f41d6;
    color: #ffffff;
}
QToolButton#toolbarIconButton:disabled {
    background: #f4f6fa;
    border: 1px solid #e0e6ef;
    color: #a0aabc;
}

/* Status bar */
QStatusBar { background: #e9edf5; color: #506078; border-top: 1px solid #d9dfec; }
QStatusBar QLabel { color: #506078; }

/* Scroll bars */
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: #c9d2e2; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background: #6d4de0; }

/* Plot frame stays a soft near-white so the dark plot lines pop */
QGraphicsView { background: #fafbfe; border: 1px solid #d9dfec; }

/* Reusable in-panel building blocks */
QFrame#pyberPanelHeader {
    border-bottom: 1px solid #d9dfec;
}
QLabel#pyberPanelHeaderTitle { color: #141b28; }
QLabel#pyberPanelHeaderSubtitle { color: #506078; }

QFrame#pyberBusyWidget {
    background: #fff7ed;
    border: 1px solid #fb923c;
    border-radius: 7px;
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

QFrame#pyberExportProgressWidget {
    background: transparent;
    border: 0;
}
QProgressBar#pyberExportProgressBar {
    background: #dbe3ef;
    border: 0;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
}
QProgressBar#pyberExportProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1d9e6e, stop:1 #6d4de0);
    border-radius: 3px;
    margin: 0;
}

QTabBar#visualModeBar::tab {
    background: #eef1f7;
    color: #506078;
    border: 1px solid #d9dfec;
}
QTabBar#visualModeBar::tab:hover:!selected {
    background: #f3f1fc;
    color: #1b2434;
    border: 1px solid #c4b4f0;
}
QTabBar#visualModeBar::tab:selected {
    background: #6d4de0;
    color: #ffffff;
    border: 1px solid #5f41d6;
}

/* Top app bar */
QFrame#pyberTopBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #eef1f7);
    border: 0;
    border-bottom: 1px solid #d9dfec;
}
QLabel#pyberAppName { color: #141b28; }
QLabel#pyberWorkflowStep { color: #506078; }
QLabel#pyberWorkflowStep[active="true"] { color: #ffffff; background: #6d4de0; }
QLabel#pyberProjectName { color: #506078; background: #f2f4f9; border: 1px solid #d9dfec; }
QLabel#pyberProjectName[dirty="true"] { color: #b07d18; border: 1px solid #d9c584; }

QFrame[class="subsection"] { border-bottom: 1px solid #e3e8f1; }
QLabel[class="subsectionTitle"] { color: #1b2434; }
QLabel[class="subsectionHint"] { color: #506078; }
QLabel[class="hint"], QLabel[class="muted"] { color: #506078; }
QLabel[class="emptyState"] { color: #a0aabc; }
QLabel[class="emptyStateTitle"] { color: #506078; }
QFrame[class="card"] {
    background: #ffffff; border: 1px solid #dde3ee;
}
QFrame[class="callout"] {
    background: #faf9ff; border: 1px solid #c4b4f0;
}
QFrame[class="footerActions"] {
    background: #eef1f7; border: 1px solid #d9dfec;
}
QFrame[class="inlineStatus"] {
    background: #f6f8fc; border: 1px solid #d9dfec;
}
QFrame[class="inlineStatus"][severity="ok"]    { border: 1px solid #7fcfae; background: #e2f5ec; }
QFrame[class="inlineStatus"][severity="warn"]  { border: 1px solid #e2bc7a; background: #fdf3dd; }
QFrame[class="inlineStatus"][severity="error"] { border: 1px solid #eba4af; background: #fdeaed; }
QFrame[class="inlineStatus"] QLabel { color: #1b2434; }

/* Dock widgets */
QDockWidget { background: #ffffff; color: #1b2434; border: 1px solid #d9dfec; }
QDockWidget::title { background: #eef1f7; color: #506078; border-bottom: 1px solid #d9dfec; }
QDockWidget > QWidget { background: #ffffff; }

/* Menus */
QMenu { background: #ffffff; color: #1b2434; border: 1px solid #c3ccdd; }
QMenu::item:selected { background: #e2dbf8; color: #2a1f5e; }
QMenuBar { background: #e9edf5; color: #1b2434; border-bottom: 1px solid #d9dfec; }
QMenuBar::item:selected { background: #dde3f0; }

QToolButton {
    background: #ffffff; color: #1b2434; border: 1px solid #dde3ee;
}
QToolButton:hover { background: #f3f1fc; border: 1px solid #c4b4f0; }

QCheckBox::indicator {
    border: 1px solid #c3ccdd; background: #ffffff;
}
QCheckBox::indicator:hover { border: 1px solid #8a6cee; }
QRadioButton::indicator {
    border: 1px solid #c3ccdd; background: #ffffff;
}

QToolTip {
    background: #ffffff; color: #1b2434;
    border: 1px solid #c4b4f0;
}
"""


# ---------------------------------------------------------------------------
# Crisp glyph assets (checkmark, combo chevron) generated at runtime into the
# temp dir so QSS can reference real images instead of CSS border triangles.
# Generation needs a QGuiApplication; before one exists we fall back to
# glyph-free rules that still look correct.
# ---------------------------------------------------------------------------

_ASSET_PATHS: dict = {}


def _style_asset(name: str, painter) -> str | None:
    """Render a small PNG glyph and return its forward-slash path, or None."""
    if name in _ASSET_PATHS:
        return _ASSET_PATHS[name]
    try:
        from PySide6 import QtCore, QtGui
        if QtGui.QGuiApplication.instance() is None:
            return None
        import tempfile
        folder = os.path.join(tempfile.gettempdir(), "pyber_qss_assets")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name}.png")
        pix = QtGui.QPixmap(10, 10)
        pix.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter(p)
        p.end()
        if not pix.save(path, "PNG"):
            return None
        result = path.replace("\\", "/")
        _ASSET_PATHS[name] = result
        return result
    except Exception:
        return None


def _checkmark_asset() -> str | None:
    def _draw(p):
        from PySide6 import QtCore, QtGui
        pen = QtGui.QPen(QtGui.QColor("#ffffff"), 1.7,
                         QtCore.Qt.PenStyle.SolidLine,
                         QtCore.Qt.PenCapStyle.RoundCap,
                         QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(2.0, 5.4)
        path.lineTo(4.1, 7.6)
        path.lineTo(8.2, 2.6)
        p.drawPath(path)
    return _style_asset("check_white", _draw)


def _chevron_asset(color: str) -> str | None:
    key = "chevron_" + color.lstrip("#")

    def _draw(p):
        from PySide6 import QtCore, QtGui
        pen = QtGui.QPen(QtGui.QColor(color), 1.7,
                         QtCore.Qt.PenStyle.SolidLine,
                         QtCore.Qt.PenCapStyle.RoundCap,
                         QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(2.2, 3.8)
        path.lineTo(5.0, 6.8)
        path.lineTo(7.8, 3.8)
        p.drawPath(path)
    return _style_asset(key, _draw)


def _chevron_rule(mode: str) -> str:
    color = "#5d6a7d" if mode == "light" else "#a9b3c9"
    path = _chevron_asset(color)
    if path:
        return f'image: url("{path}"); width: 10px; height: 10px;'
    # Fallback: CSS border triangle (no image support before QApplication).
    return ("image: none; width: 8px; height: 8px;"
            " border-left: 5px solid transparent;"
            " border-right: 5px solid transparent;"
            f" border-top: 5px solid {color};")


def _check_rule() -> str:
    path = _checkmark_asset()
    if path:
        return f'image: url("{path}");'
    return ""


_QSS_CACHE: dict = {}


def app_qss(theme_mode: object) -> str:
    mode = str(theme_mode or "").strip().lower()
    mode = "light" if mode in {"light", "white", "l", "w"} else "dark"

    assets_ready = _checkmark_asset() is not None
    cache_key = (mode, assets_ready)
    cached = _QSS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if mode == "light":
        qss = _build_light_qss(_QSS_TEMPLATE) + _LIGHT_OVERRIDES
    else:
        qss = _QSS_TEMPLATE
    qss = qss.replace("%CHEVRON_RULE%", _chevron_rule(mode))
    qss = qss.replace("%CHECK_RULE%", _check_rule())

    _QSS_CACHE[cache_key] = qss
    return qss


# Backwards-compatible module constants (glyph-free fallback variants).
APP_QSS = (_QSS_TEMPLATE
           .replace("%CHEVRON_RULE%",
                    "image: none; width: 8px; height: 8px;"
                    " border-left: 5px solid transparent;"
                    " border-right: 5px solid transparent;"
                    " border-top: 5px solid #a9b3c9;")
           .replace("%CHECK_RULE%", ""))

APP_QSS_LIGHT = ((_build_light_qss(_QSS_TEMPLATE) + _LIGHT_OVERRIDES)
                 .replace("%CHEVRON_RULE%",
                          "image: none; width: 8px; height: 8px;"
                          " border-left: 5px solid transparent;"
                          " border-right: 5px solid transparent;"
                          " border-top: 5px solid #5d6a7d;")
                 .replace("%CHECK_RULE%", ""))


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
        # setStyle() repolishes every live widget (~2 s on a built window),
        # so only pay for it when the style actually needs changing.
        current = app.style()
        if current is None or current.objectName().lower() != "fusion":
            QtWidgets.QApplication.setStyle("Fusion")
    except Exception:
        pass

    palette = QtGui.QPalette()
    if light:
        colors = {
            QtGui.QPalette.ColorRole.Window: "#f2f4f9",
            QtGui.QPalette.ColorRole.WindowText: "#1b2434",
            QtGui.QPalette.ColorRole.Base: "#ffffff",
            QtGui.QPalette.ColorRole.AlternateBase: "#f0f3f9",
            QtGui.QPalette.ColorRole.ToolTipBase: "#ffffff",
            QtGui.QPalette.ColorRole.ToolTipText: "#1b2434",
            QtGui.QPalette.ColorRole.Text: "#1b2434",
            QtGui.QPalette.ColorRole.Button: "#e7ecf5",
            QtGui.QPalette.ColorRole.ButtonText: "#1b2434",
            QtGui.QPalette.ColorRole.BrightText: "#ffffff",
            QtGui.QPalette.ColorRole.Highlight: "#6d4de0",
            QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
            QtGui.QPalette.ColorRole.Link: "#5a3fd0",
        }
    else:
        colors = {
            QtGui.QPalette.ColorRole.Window: "#161a26",
            QtGui.QPalette.ColorRole.WindowText: "#edf0f8",
            QtGui.QPalette.ColorRole.Base: "#141824",
            QtGui.QPalette.ColorRole.AlternateBase: "#1a1f2e",
            QtGui.QPalette.ColorRole.ToolTipBase: "#10131c",
            QtGui.QPalette.ColorRole.ToolTipText: "#edf0f8",
            QtGui.QPalette.ColorRole.Text: "#edf0f8",
            QtGui.QPalette.ColorRole.Button: "#222840",
            QtGui.QPalette.ColorRole.ButtonText: "#edf0f8",
            QtGui.QPalette.ColorRole.BrightText: "#ffffff",
            QtGui.QPalette.ColorRole.Highlight: "#7c5cff",
            QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
            QtGui.QPalette.ColorRole.Link: "#9d7bff",
        }
    for role, color in colors.items():
        palette.setColor(role, QtGui.QColor(color))
    app.setPalette(palette)


# ---------------------------------------------------------------------------
# Native Windows title bar theming (DWM). Makes the OS window chrome match
# the app instead of shipping a white title bar over a dark instrument.
# Silently does nothing on non-Windows or older Windows builds.
# ---------------------------------------------------------------------------

_TITLEBAR_COLORS = {
    "dark": {"caption": "#10131c", "text": "#edf0f8", "border": "#232a3d"},
    "light": {"caption": "#e9edf5", "text": "#1b2434", "border": "#d9dfec"},
}


def _colorref(hex_color: str) -> int:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def apply_native_titlebar(widget, theme_mode: object) -> None:
    if os.name != "nt" or widget is None:
        return
    from PySide6 import QtGui
    if QtGui.QGuiApplication.platformName() != "windows":
        return
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = int(widget.winId())
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi
        # HWND is pointer-sized on 64-bit Windows. Explicit signatures also
        # prevent offscreen/test handles from reaching the native API above.
        dwm.DwmSetWindowAttribute.argtypes = [
            wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ]
        dwm.DwmSetWindowAttribute.restype = ctypes.HRESULT

        mode = str(theme_mode or "").strip().lower()
        light = mode in {"light", "white", "l", "w"}
        colors = _TITLEBAR_COLORS["light" if light else "dark"]

        # DWMWA_USE_IMMERSIVE_DARK_MODE: 20 (19 on pre-2004 Windows 10).
        dark_flag = ctypes.c_int(0 if light else 1)
        for attr in (20, 19):
            if dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(dark_flag),
                                         ctypes.sizeof(dark_flag)) == 0:
                break

        # Windows 11 only; harmless failures elsewhere.
        for attr, hex_color in ((35, colors["caption"]),   # DWMWA_CAPTION_COLOR
                                (36, colors["text"]),      # DWMWA_TEXT_COLOR
                                (34, colors["border"])):   # DWMWA_BORDER_COLOR
            value = ctypes.c_int(_colorref(hex_color))
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value),
                                      ctypes.sizeof(value))
    except Exception:
        pass


class _TitlebarThemer:
    """App-level event filter: themes every top-level window as it shows."""

    _instance = None

    def __init__(self, mode: str):
        from PySide6 import QtCore, QtWidgets

        class _Filter(QtCore.QObject):
            def __init__(self, outer):
                super().__init__()
                self._outer = outer

            def eventFilter(self, obj, event):
                try:
                    if (event.type() == QtCore.QEvent.Type.Show
                            and isinstance(obj, QtWidgets.QWidget)
                            and obj.isWindow()):
                        apply_native_titlebar(obj, self._outer.mode)
                except Exception:
                    pass
                return False

        self.mode = mode
        self._filter = _Filter(self)


def install_native_titlebar(app, theme_mode: object) -> None:
    """Install (or retarget) the title bar themer and restyle open windows."""
    if app is None or os.name != "nt":
        return
    from PySide6 import QtGui
    if QtGui.QGuiApplication.platformName() != "windows":
        return
    try:
        mode = str(theme_mode or "").strip().lower()
        mode = "light" if mode in {"light", "white", "l", "w"} else "dark"
        themer = _TitlebarThemer._instance
        if themer is None:
            themer = _TitlebarThemer(mode)
            _TitlebarThemer._instance = themer
            app.installEventFilter(themer._filter)
        else:
            themer.mode = mode
        for w in app.topLevelWidgets():
            if w.isWindow() and w.isVisible():
                apply_native_titlebar(w, mode)
    except Exception:
        pass
