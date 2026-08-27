# gui_sensors.py
from __future__ import annotations

from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from sensor_registry import SENSOR_UNKNOWN, SensorInfo, all_sensors, get_sensor


class SensorDialog(QtWidgets.QDialog):
    """Sensor selector backed by the curated pyBer sensor registry."""

    COLUMNS = [
        "Sensor",
        "Family",
        "Target",
        "Color",
        "Direction",
        "Excitation",
        "Isobestic",
        "Emission",
        "Rise",
        "Decay",
        "Kinetics basis",
        "Affinity",
        "Dynamic range",
        "Rec Fs",
        "Rec LP",
        "Source",
        "Paper",
    ]

    def __init__(self, current_sensor_id: str = SENSOR_UNKNOWN, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fiber photometry sensors")
        self.resize(1320, 720)
        self._sensors: List[SensorInfo] = all_sensors()
        self._selected_sensor_id = str(current_sensor_id or SENSOR_UNKNOWN)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Sensor Library")
        title.setObjectName("sensorTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search sensor, target, family, or source")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(320)
        header.addWidget(self.search)
        root.addLayout(header)

        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(False)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().resizeSection(0, 180)
        root.addWidget(self.table, stretch=1)

        self.details = QtWidgets.QTextBrowser()
        self.details.setObjectName("sensorDetails")
        self.details.setOpenExternalLinks(True)
        self.details.setMinimumHeight(110)
        self.details.setMaximumHeight(170)
        root.addWidget(self.details)

        footer = QtWidgets.QHBoxLayout()
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_paper = QtWidgets.QPushButton("Open paper")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_select = QtWidgets.QPushButton("Select sensor")
        self.btn_select.setDefault(True)
        self.btn_select.setProperty("class", "bluePrimarySmall")
        footer.addWidget(self.btn_clear)
        footer.addStretch(1)
        footer.addWidget(self.btn_paper)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_select)
        root.addLayout(footer)

        self.setStyleSheet(
            """
            QDialog {
                background: #161a26;
                color: #edf0f8;
            }
            QLabel#sensorTitle {
                color: #f7f9fd;
                font-family: "Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif;
                font-size: 18pt;
                font-weight: 800;
            }
            QLineEdit {
                background: #141824;
                border: 1px solid #333c56;
                border-radius: 7px;
                color: #edf0f8;
                padding: 7px 10px;
            }
            QLineEdit:focus {
                border: 1px solid #7c5cff;
            }
            QTableWidget {
                background: #141824;
                alternate-background-color: #171c2b;
                border: 1px solid #232a3d;
                border-radius: 9px;
                color: #edf0f8;
                selection-background-color: #372f66;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #1d2333;
                color: #a9b3c9;
                border: 0px;
                border-right: 1px solid #232a3d;
                padding: 7px 8px;
                font-weight: 700;
            }
            QTextBrowser#sensorDetails {
                background: #141824;
                border: 1px solid #232a3d;
                border-radius: 9px;
                color: #ccd4e4;
                padding: 8px;
            }
            QPushButton {
                background: #241f42;
                border: 1px solid #4d3fa3;
                border-radius: 8px;
                color: #e6def8;
                padding: 7px 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #2e2854;
                border: 1px solid #5f4fc2;
            }
            QPushButton[class="bluePrimarySmall"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8a63ff, stop:1 #6f4df2);
                border: 1px solid #8f74ff;
                color: #ffffff;
            }
            QPushButton[class="bluePrimarySmall"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #9a77ff, stop:1 #7d5cff);
                border: 1px solid #a68cff;
            }
            """
        )

        self._populate()
        self.table.setSortingEnabled(True)
        self._select_sensor_id(self._selected_sensor_id)
        self.search.textChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._update_selection_from_table)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.btn_select.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_clear.clicked.connect(self._clear_sensor)
        self.btn_paper.clicked.connect(self._open_paper)

    def selected_sensor_id(self) -> str:
        return str(self._selected_sensor_id or SENSOR_UNKNOWN)

    def _row_values(self, sensor: SensorInfo) -> List[str]:
        return [
            sensor.name,
            sensor.family,
            sensor.target,
            sensor.color,
            sensor.direction,
            sensor.excitation_nm,
            sensor.isobestic_nm,
            sensor.emission_nm,
            sensor.rise,
            sensor.decay,
            sensor.kinetics_context,
            sensor.affinity,
            sensor.dynamic_range,
            f"{sensor.recommended_fs_hz:g} Hz",
            f"{sensor.recommended_lowpass_hz:g} Hz",
            sensor.source,
            "Open paper",
        ]

    def _populate(self) -> None:
        self.table.setRowCount(len(self._sensors))
        for row, sensor in enumerate(self._sensors):
            values = self._row_values(sensor)
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(text))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, sensor.sensor_id)
                if col == 16:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#a68cff")))
                self.table.setItem(row, col, item)

    def _selected_row_sensor(self) -> SensorInfo:
        indexes = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if indexes:
            row = int(indexes[0].row())
            item = self.table.item(row, 0)
            if item is not None:
                return get_sensor(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or SENSOR_UNKNOWN))
        return get_sensor(self._selected_sensor_id)

    def _select_sensor_id(self, sensor_id: str) -> None:
        target = str(sensor_id or SENSOR_UNKNOWN)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            if str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "") == target:
                self.table.selectRow(row)
                self.table.scrollToItem(item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter)
                self._selected_sensor_id = target
                self._set_details(get_sensor(target))
                return
        self._selected_sensor_id = SENSOR_UNKNOWN
        self._set_details(get_sensor(SENSOR_UNKNOWN))

    def _update_selection_from_table(self) -> None:
        sensor = self._selected_row_sensor()
        self._selected_sensor_id = sensor.sensor_id
        self._set_details(sensor)

    def _set_details(self, sensor: SensorInfo) -> None:
        link = sensor.paper_url
        html = (
            f"<b>{sensor.name}</b> | {sensor.family} | target: {sensor.target} | "
            f"direction: {sensor.direction}<br>"
            f"Excitation: {sensor.excitation_nm} nm | Isobestic/control: {sensor.isobestic_nm} nm | "
            f"Emission: {sensor.emission_nm} nm<br>"
            f"Rise: {sensor.rise} | Decay: {sensor.decay} | Affinity: {sensor.affinity}<br>"
            f"Kinetics basis: {sensor.kinetics_context or sensor.source}<br>"
            f"{sensor.notes}<br>"
            f"<a href=\"{link}\">{sensor.source}</a>"
        )
        self.details.setHtml(html)

    def _apply_filter(self, text: str) -> None:
        query = str(text or "").strip().lower()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            sensor = get_sensor(str(item.data(QtCore.Qt.ItemDataRole.UserRole) or SENSOR_UNKNOWN)) if item else get_sensor(SENSOR_UNKNOWN)
            haystack = " ".join(self._row_values(sensor)).lower()
            self.table.setRowHidden(row, bool(query and query not in haystack))

    def _on_item_double_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() == 16:
            self._open_paper()
            return
        self.accept()

    def _open_paper(self) -> None:
        sensor = self._selected_row_sensor()
        if sensor.paper_url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(sensor.paper_url))

    def _clear_sensor(self) -> None:
        self._selected_sensor_id = SENSOR_UNKNOWN
        self.accept()
