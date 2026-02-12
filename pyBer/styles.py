# styles.py

APP_QSS = r"""
/* Adobe-like dark palette */
QMainWindow, QWidget {
    background: #1f2229;
    color: #f3f5f8;
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: 9.5pt;
}

QLabel {
    color: #f3f5f8;
}

QLabel[class="hint"] {
    color: #aeb6c5;
    font-size: 8.9pt;
}

/* Panels */
QGroupBox {
    border: 1px solid #3a4050;
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px;
    background: #262b35;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 7px;
    color: #f3f5f8;
    font-weight: 700;
}

QFrame {
    border-color: #3a4050;
}

/* Lists and tables */
QListWidget, QTableWidget {
    background: #20242d;
    border: 1px solid #3a4050;
    border-radius: 7px;
    padding: 4px;
    gridline-color: #343a47;
}

QListWidget::item, QTableWidget::item {
    padding: 2px 4px;
}

QListWidget::item:selected, QTableWidget::item:selected {
    background: #1d4f80;
    color: #ffffff;
    border-radius: 4px;
}

QHeaderView::section {
    background: #2b303b;
    color: #ecf0f6;
    border: 1px solid #3a4050;
    padding: 4px 6px;
    font-weight: 600;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #1b2029;
    color: #f3f5f8;
    border: 1px solid #4a5163;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #2680eb;
    selection-color: #ffffff;
    min-height: 20px;
}

QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus {
    border: 1px solid #378ef0;
}

QComboBox::drop-down {
    border-left: 1px solid #4a5163;
    width: 20px;
}

QComboBox QAbstractItemView {
    background: #1f242e;
    color: #f3f5f8;
    border: 1px solid #4a5163;
    selection-background-color: #1d4f80;
}

QCheckBox {
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #5a6274;
    background: #1b2029;
}

QCheckBox::indicator:checked {
    background: #2680eb;
    border: 1px solid #378ef0;
}

/* Buttons */
QPushButton {
    background: #2b303b;
    border: 1px solid #4a5163;
    border-radius: 7px;
    padding: 8px 12px;
    font-weight: 600;
    color: #f3f5f8;
}

QPushButton:hover {
    background: #343a48;
    border: 1px solid #5b6480;
}

QPushButton:pressed {
    background: #232833;
}

QPushButton:disabled {
    color: #858f9f;
    background: #252a33;
    border: 1px solid #373d4a;
}

QPushButton[class="compact"] {
    padding: 5px 10px;
    border-radius: 7px;
    font-weight: 600;
}

QPushButton[class="compactSmall"] {
    padding: 2px 7px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.2pt;
    min-height: 18px;
}

QPushButton[class="compactPrimary"],
QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    background: #2680eb;
    border: 1px solid #378ef0;
    color: #ffffff;
    font-weight: 700;
}

QPushButton[class="compactPrimary"] {
    padding: 5px 10px;
    border-radius: 7px;
}

QPushButton[class="compactPrimarySmall"],
QPushButton[class="bluePrimarySmall"] {
    padding: 2px 7px;
    border-radius: 6px;
    font-size: 8.2pt;
    min-height: 18px;
}

QPushButton[class="compactPrimary"]:hover,
QPushButton[class="compactPrimarySmall"]:hover,
QPushButton[class="bluePrimarySmall"]:hover {
    background: #378ef0;
    border: 1px solid #4b9df8;
}

QPushButton[class="compactPrimary"]:pressed,
QPushButton[class="compactPrimarySmall"]:pressed,
QPushButton[class="bluePrimarySmall"]:pressed {
    background: #1473e6;
    border: 1px solid #2680eb;
}

QPushButton[class="blueSecondarySmall"] {
    padding: 2px 7px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.2pt;
    min-height: 18px;
    background: #2b303b;
    border: 1px solid #4a5163;
    color: #eef2f7;
}

QPushButton[class="blueSecondarySmall"]:hover {
    background: #343a48;
    border: 1px solid #5b6480;
}

QPushButton[class="blueSecondarySmall"]:pressed {
    background: #232833;
}

QPushButton[class="blueSecondarySmall"]:checked,
QPushButton[class="sectionButton"]:checked {
    background: #2680eb;
    border: 1px solid #378ef0;
    color: #ffffff;
}

QPushButton[class="sectionButton"] {
    padding: 3px 9px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 8.4pt;
    background: #2b303b;
    border: 1px solid #4a5163;
    text-align: left;
}

QPushButton[class="sectionButton"]:hover {
    background: #343a48;
}

QPushButton[class="help"] {
    padding: 0;
    border-radius: 10px;
    min-width: 20px;
    max-width: 20px;
    min-height: 20px;
    max-height: 20px;
    font-weight: 700;
    background: #2f3543;
    border: 1px solid #4d5568;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #363c4a;
    border-radius: 8px;
    background: #1f2229;
    padding: 2px;
}

QTabBar::tab {
    background: #2b303b;
    border: 1px solid #3b4150;
    border-bottom: none;
    padding: 8px 12px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
    color: #bcc5d6;
}

QTabBar::tab:hover {
    background: #343a48;
}

QTabBar::tab:selected {
    background: #1f2229;
    color: #ffffff;
    border-color: #4a5163;
}

/* Menus and tool buttons */
QMenu {
    background: #2b303b;
    color: #f3f5f8;
    border: 1px solid #41495a;
}

QMenu::item {
    padding: 6px 16px;
}

QMenu::item:selected {
    background: #1d4f80;
}

QToolButton {
    background: #2b303b;
    border: 1px solid #4a5163;
    border-radius: 6px;
    padding: 3px 7px;
}

QToolButton:hover {
    background: #343a48;
}

/* Docking and splitters */
QDockWidget {
    background: #242a34;
    color: #f3f5f8;
    border: 1px solid #3b4150;
}

QDockWidget::title {
    background: #2d3340;
    border-bottom: 1px solid #3b4150;
    padding: 6px 8px;
    text-align: left;
    font-weight: 700;
}

QDockWidget QWidget {
    background: #242a34;
}

QMainWindow::separator {
    background: #3f4758;
    width: 2px;
    height: 2px;
}

QMainWindow::separator:hover {
    background: #2680eb;
}

QSplitter::handle {
    background: #3f4758;
}

QSplitter::handle:hover {
    background: #2680eb;
}

QScrollArea {
    border: none;
}

/* Plot frame support */
QGraphicsView {
    background: #121722;
    border: 1px solid #313746;
    border-radius: 6px;
}

QToolTip {
    background: #2d3442;
    color: #f3f5f8;
    border: 1px solid #4a5163;
    padding: 4px 6px;
}
"""
