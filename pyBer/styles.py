# styles.py

APP_QSS = r"""
/* Base */
QWidget {
    background: #0f141b;
    color: #e6eefc;
    font-family: Segoe UI, Arial, sans-serif;
    font-size: 9.5pt;
}

QGroupBox {
    border: 1px solid #1f2a38;
    border-radius: 10px;
    margin-top: 10px;
    padding: 10px;
    background: #0f141b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #cfe0ff;
    font-weight: 700;
}

/* Lists */
QListWidget {
    background: #0b1016;
    border: 1px solid #1f2a38;
    border-radius: 10px;
    padding: 6px;
}
QListWidget::item:selected {
    background: #1b3558;
    color: #ffffff;
    border-radius: 6px;
}

/* Inputs */
QLineEdit, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #0b1016;
    color: #e6eefc;
    border: 1px solid #26364a;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #2a66b3;
    selection-color: #ffffff;
}

QComboBox::drop-down {
    border-left: 1px solid #26364a;
    width: 24px;
}
QComboBox QAbstractItemView {
    background: #0b1016;
    color: #e6eefc;
    border: 1px solid #26364a;
    selection-background-color: #1b3558;
}

/* Buttons */
QPushButton {
    background: #162030;
    border: 1px solid #2a3c55;
    border-radius: 10px;
    padding: 10px 12px;
    font-weight: 600;
}
QPushButton:hover {
    background: #1b2a3f;
}
QPushButton:pressed {
    background: #122033;
}

QPushButton[class="compact"] {
    padding: 6px 10px;
    border-radius: 9px;
    font-weight: 600;
}
QPushButton[class="compactSmall"] {
    padding: 2px 6px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 8.0pt;
    min-height: 18px;
}
QPushButton[class="compactPrimary"] {
    padding: 6px 10px;
    border-radius: 9px;
    font-weight: 800;
    background: #254a7a;
    border: 1px solid #2f6bb3;
}
QPushButton[class="compactPrimarySmall"] {
    padding: 2px 6px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 8.0pt;
    background: #254a7a;
    border: 1px solid #2f6bb3;
}
QPushButton[class="compactPrimarySmall"]::text {
    text-align: left;
}
QPushButton[class="compactPrimarySmall"]:hover {
    background: #2b5a96;
}
QPushButton[class="compactPrimary"]:hover {
    background: #2b5a96;
}
QPushButton[class="help"] {
    padding: 0;
    border-radius: 11px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    font-weight: 800;
    background: #1b2a3f;
    border: 1px solid #2a3c55;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #1f2a38;
    border-radius: 10px;
    padding: 2px;
}
QTabBar::tab {
    background: #121a25;
    border: 1px solid #1f2a38;
    border-bottom: none;
    padding: 10px 14px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    margin-right: 2px;
    font-weight: 700;
    color: #b8c9e8;
}
QTabBar::tab:selected {
    background: #0f141b;
    color: #ffffff;
}

/* Scroll */
QScrollArea {
    border: none;
}

/* Dock */
QDockWidget {
    background: #0f141b;
    color: #e6eefc;
}
QDockWidget::title {
    background: #121a25;
    padding: 6px;
    font-weight: 700;
}

/* Hints/log */
QLabel[class="hint"] {
    color: #aab4c4;
    font-size: 9.0pt;
}
"""
