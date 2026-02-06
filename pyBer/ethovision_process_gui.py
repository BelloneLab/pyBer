from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QCheckBox,
    QSplitter,
    QGroupBox,
    QFormLayout,
    QPlainTextEdit,
)

import pyqtgraph as pg


MISSING_MARKERS = {"", " ", "-", "NaN", "nan", "NAN", "n/a", "N/A", None}


def find_header_row(excel_path: Path, sheet_name: str) -> int:
    """Find 0-based row index containing 'Trial time' (case-insensitive) in any cell."""
    preview = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
        nrows=300,
    )
    target = re.compile(r"^\s*Trial time\s*$", flags=re.IGNORECASE)
    for i in range(preview.shape[0]):
        row = preview.iloc[i, :]
        def _cell_matches(val) -> bool:
            if pd.isna(val):
                return False
            return bool(target.match(str(val).strip()))
        if row.apply(_cell_matches).any():
            return i
    raise ValueError(f"Could not find header row (cell 'Trial time') in sheet '{sheet_name}'.")


def extract_metadata(excel_path: Path, sheet_name: str) -> Dict[str, str]:
    """
    Extract metadata from rows before the data header.

    EthoVision exports typically have a variable number of metadata rows above the
    actual data header. This function reads those rows and tries to parse key/value
    pairs. It is intentionally tolerant (skips blank/nan rows).

    Returns:
        Dict[str, str]: metadata entries.
    """
    try:
        header_row = find_header_row(excel_path, sheet_name)
    except ValueError:
        return {}

    if header_row <= 0:
        return {}

    preview = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
        nrows=header_row,
    )

    metadata: Dict[str, str] = {}

    for i in range(preview.shape[0]):
        row = preview.iloc[i, :]
        # find first non-empty cell as key
        key = None
        key_col = None
        for col_idx in range(preview.shape[1]):
            v = row.iloc[col_idx]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s or s.lower() in {"nan", "none"}:
                continue
            key = s
            key_col = col_idx
            break

        if key is None or key_col is None:
            continue

        # find the next non-empty cell after key as value
        value = None
        for col_idx in range(key_col + 1, preview.shape[1]):
            v = row.iloc[col_idx]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if not s or s.lower() in {"nan", "none"}:
                continue
            value = s
            break

        if value is None:
            continue

        # avoid overwriting if duplicate keys appear; keep first occurrence
        if key not in metadata:
            metadata[key] = value

    return metadata


def coerce_decimal_comma_to_float(s: pd.Series) -> pd.Series:
    """Convert series with decimal commas to floats; non-convertible -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s2 = s.copy()
    s2 = s2.replace(list(MISSING_MARKERS), np.nan)
    s2 = s2.astype(str).str.strip()

    s2 = (
        s2.str.replace("\u00A0", " ", regex=False)  # NBSP
          .str.replace("'", "", regex=False)
          .str.replace(" ", "", regex=False)
          .str.replace(",", ".", regex=False)
    )
    s2 = s2.replace({"nan": np.nan, "NaN": np.nan, "NAN": np.nan, "-": np.nan})
    return pd.to_numeric(s2, errors="coerce")


def clean_sheet(
    excel_path: Path,
    sheet_name: str,
    interpolate: bool = True,
) -> pd.DataFrame:
    """
    Read, drop units row, clean markers, coerce numeric, interpolate numeric columns.

    Assumes:
    - Header row contains 'Trial time'
    - Units row is the first row after header
    - Decimal commas present in numeric fields
    """
    header_row = find_header_row(excel_path, sheet_name)

    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name,
        header=header_row,
        engine="openpyxl",
    ).dropna(axis=1, how="all")

    if df.empty:
        return df

    # Drop units row
    df = df.iloc[1:, :].reset_index(drop=True)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Replace missing markers globally
    df = df.replace(list(MISSING_MARKERS), np.nan)

    # Choose time column
    time_col = None
    for c in ["Trial time", "Recording time", "Trial Time", "Recording Time"]:
        if c in df.columns:
            time_col = c
            break

    # Coerce numeric columns
    numeric_cols: List[str] = []
    for col in df.columns:
        converted = coerce_decimal_comma_to_float(df[col])
        non_na_ratio = converted.notna().mean()
        if non_na_ratio >= 0.3:
            df[col] = converted
            numeric_cols.append(col)
        else:
            df[col] = (
                df[col]
                .astype(str)
                .replace({"nan": np.nan, "NaN": np.nan, "NAN": np.nan})
                .str.strip()
            )

    # Drop rows without time if we can identify time
    if time_col is not None:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df = df.loc[df[time_col].notna()].copy()
        df = df.sort_values(time_col).reset_index(drop=True)

    # Interpolate numeric columns by time index
    if interpolate and time_col is not None and len(df) > 1:
        df = df.set_index(time_col)

        cols_to_interp = [c for c in numeric_cols if c != time_col]
        if cols_to_interp:
            df[cols_to_interp] = df[cols_to_interp].interpolate(
                method="index",
                limit_direction="both",
            )
            df[cols_to_interp] = df[cols_to_interp].ffill().bfill()

        df = df.reset_index()

    df = df.dropna(how="all").reset_index(drop=True)
    return df


@dataclass
class WorkbookData:
    path: Path
    sheets: List[str]
    cleaned: Dict[str, pd.DataFrame]
    metadata: Dict[str, Dict[str, str]]


class PlotPanel(QWidget):
    """
    PySide6-native plotting using pyqtgraph (fast, stable, avoids matplotlib Qt backend issues).
    """
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        pg.setConfigOptions(antialias=True)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(10, 10))

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self.setLayout(layout)

        self._palette = [
            (31, 119, 180),
            (255, 127, 14),
            (44, 160, 44),
            (214, 39, 40),
            (148, 103, 189),
            (140, 86, 75),
            (227, 119, 194),
            (127, 127, 127),
            (188, 189, 34),
            (23, 190, 207),
        ]

    def plot_lines(self, df: pd.DataFrame, x_col: str, y_cols: List[str]):
        self.plot.clear()
        self.plot.addLegend(offset=(10, 10))

        if df.empty:
            self.plot.setTitle("No data loaded")
            return

        if x_col not in df.columns:
            self.plot.setTitle(f"X column '{x_col}' not found")
            return

        x = df[x_col].to_numpy()
        self.plot.setLabel("bottom", x_col)

        any_plotted = False
        color_i = 0

        for yc in y_cols:
            if yc not in df.columns:
                continue
            if not np.issubdtype(df[yc].dtype, np.number):
                continue

            y = df[yc].to_numpy()

            # Optional: downsample for very large traces (keeps UI responsive)
            # If you want no downsampling, remove this block.
            if len(x) > 200_000:
                step = max(1, len(x) // 200_000)
                x_plot = x[::step]
                y_plot = y[::step]
            else:
                x_plot, y_plot = x, y

            col = self._palette[color_i % len(self._palette)]
            pen = pg.mkPen(color=col, width=1.2)
            self.plot.plot(x_plot, y_plot, pen=pen, name=yc)
            color_i += 1
            any_plotted = True

        if any_plotted:
            self.plot.setTitle("")
        else:
            self.plot.setTitle("No numeric Y columns selected (or available)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking XLSX Viewer / Cleaner")
        self.setAcceptDrops(True)

        self.workbook: Optional[WorkbookData] = None

        # Top controls
        open_btn = QPushButton("Open XLSX…")
        open_btn.clicked.connect(self.open_file)

        self.file_label = QLabel("Drop an .xlsx file here, or click Open XLSX…")
        self.file_label.setWordWrap(True)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #444;")

        top_bar = QHBoxLayout()
        top_bar.addWidget(open_btn)
        top_bar.addWidget(self.file_label, 1)

        top_wrap = QVBoxLayout()
        top_wrap.addLayout(top_bar)
        top_wrap.addWidget(self.status_label)

        # Left panel
        self.sheet_combo = QComboBox()
        self.sheet_combo.currentTextChanged.connect(self.on_sheet_changed)

        self.interpolate_cb = QCheckBox("Interpolate numeric columns")
        self.interpolate_cb.setChecked(True)
        self.interpolate_cb.stateChanged.connect(self.reclean_current_sheet)

        self.x_combo = QComboBox()
        self.x_combo.currentTextChanged.connect(self.update_plot)

        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QListWidget.NoSelection)
        self.y_list.itemChanged.connect(self.update_plot)

        # Metadata display
        self.meta_text = QPlainTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setPlaceholderText("Metadata (header rows above 'Trial time') will appear here")

        export_selected_btn = QPushButton("Export selected columns to CSV…")
        export_selected_btn.clicked.connect(self.export_selected_columns_csv)

        export_sheet_btn = QPushButton("Export full cleaned sheet to CSV…")
        export_sheet_btn.clicked.connect(self.export_full_sheet_csv)

        export_meta_btn = QPushButton("Export metadata (current sheet)…")
        export_meta_btn.clicked.connect(self.export_metadata)

        export_sheet_with_meta_btn = QPushButton("Export cleaned sheet + metadata…")
        export_sheet_with_meta_btn.clicked.connect(self.export_full_sheet_with_metadata)

        left_box = QGroupBox("Controls")
        form = QFormLayout()
        form.addRow("Sheet:", self.sheet_combo)
        form.addRow("", self.interpolate_cb)
        form.addRow("X axis:", self.x_combo)
        form.addRow(QLabel("Y columns (check to plot):"))
        form.addRow(self.y_list)
        form.addRow(QLabel("Metadata:"))
        form.addRow(self.meta_text)
        form.addRow(export_selected_btn)
        form.addRow(export_sheet_btn)
        form.addRow(export_meta_btn)
        form.addRow(export_sheet_with_meta_btn)
        left_box.setLayout(form)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(left_box)
        left_layout.addStretch(1)
        left_widget.setLayout(left_layout)

        # Right panel
        self.plot_panel = PlotPanel()

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(self.plot_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addLayout(top_wrap)
        layout.addWidget(splitter, 1)
        central.setLayout(layout)
        self.setCentralWidget(central)

        # Menu
        file_menu = self.menuBar().addMenu("File")
        act_open = QAction("Open…", self)
        act_open.triggered.connect(self.open_file)
        file_menu.addAction(act_open)

        act_export_meta = QAction("Export metadata (current sheet)…", self)
        act_export_meta.triggered.connect(self.export_metadata)
        file_menu.addAction(act_export_meta)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    # Drag & drop
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(".xlsx"):
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = Path(urls[0].toLocalFile())
        if path.suffix.lower() == ".xlsx":
            self.load_workbook(path)

    # File handling
    def open_file(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open XLSX file",
            str(Path.home()),
            "Excel Files (*.xlsx)",
        )
        if not path_str:
            return
        self.load_workbook(Path(path_str))

    def load_workbook(self, path: Path):
        try:
            self.status_label.setText("Reading workbook...")
            xls = pd.ExcelFile(path, engine="openpyxl")
            sheets = list(xls.sheet_names)

            cleaned: Dict[str, pd.DataFrame] = {}
            metadata: Dict[str, Dict[str, str]] = {}
            interp = self.interpolate_cb.isChecked()

            for sh in sheets:
                try:
                    metadata[sh] = extract_metadata(path, sh)
                except Exception:
                    metadata[sh] = {}

                try:
                    cleaned[sh] = clean_sheet(path, sh, interpolate=interp)
                except Exception as e:
                    cleaned[sh] = pd.DataFrame()
                    QMessageBox.warning(self, "Sheet parse warning", f"Failed to clean '{sh}':\n{e}")

            self.workbook = WorkbookData(path=path, sheets=sheets, cleaned=cleaned, metadata=metadata)
            self.file_label.setText(str(path))
            self.status_label.setText(f"Loaded {len(sheets)} sheet(s).")

            self.sheet_combo.blockSignals(True)
            self.sheet_combo.clear()
            self.sheet_combo.addItems(sheets)
            self.sheet_combo.blockSignals(False)

            if sheets:
                self.sheet_combo.setCurrentIndex(0)
                self.on_sheet_changed(sheets[0])

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open workbook:\n{e}")
            self.status_label.setText("")

    def reclean_current_sheet(self):
        if not self.workbook:
            return
        sh = self.sheet_combo.currentText()
        if not sh:
            return
        try:
            self.status_label.setText(f"Cleaning '{sh}'...")
            self.workbook.cleaned[sh] = clean_sheet(
                self.workbook.path,
                sh,
                interpolate=self.interpolate_cb.isChecked(),
            )
            self.on_sheet_changed(sh)
            self.status_label.setText(f"Cleaned '{sh}'.")
        except Exception as e:
            QMessageBox.warning(self, "Re-clean failed", str(e))
            self.status_label.setText("")

    # UI updates
    def on_sheet_changed(self, sheet_name: str):
        if not self.workbook or sheet_name not in self.workbook.cleaned:
            return

        df = self.workbook.cleaned[sheet_name]

        # Metadata display
        md = self.workbook.metadata.get(sheet_name, {}) if self.workbook else {}
        if not md:
            self.meta_text.setPlainText("(No metadata found above the header in this sheet)")
        else:
            lines = [f"{k}: {v}" for k, v in sorted(md.items(), key=lambda kv: kv[0].lower())]
            self.meta_text.setPlainText("\n".join(lines))

        # X combo
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        if not df.empty:
            self.x_combo.addItems(list(df.columns))
            if "Trial time" in df.columns:
                self.x_combo.setCurrentText("Trial time")
            elif "Recording time" in df.columns:
                self.x_combo.setCurrentText("Recording time")
        self.x_combo.blockSignals(False)

        # Y checklist
        self.y_list.blockSignals(True)
        self.y_list.clear()
        if not df.empty:
            for c in df.columns:
                item = QListWidgetItem(c)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

                default_checked = c in {"X center", "Y center", "DistanceMoved", "Moving"}
                item.setCheckState(Qt.Checked if default_checked else Qt.Unchecked)

                self.y_list.addItem(item)
        self.y_list.blockSignals(False)

        self.update_plot()

    def selected_y_columns(self) -> List[str]:
        cols: List[str] = []
        for i in range(self.y_list.count()):
            it = self.y_list.item(i)
            if it.checkState() == Qt.Checked:
                cols.append(it.text())
        return cols

    def update_plot(self):
        if not self.workbook:
            self.plot_panel.plot_lines(pd.DataFrame(), "Trial time", [])
            return

        sh = self.sheet_combo.currentText()
        if not sh:
            return

        df = self.workbook.cleaned.get(sh, pd.DataFrame())
        x_col = self.x_combo.currentText() if self.x_combo.count() else "Trial time"
        y_cols = self.selected_y_columns()
        self.plot_panel.plot_lines(df, x_col, y_cols)

    # Export
    def export_full_sheet_csv(self):
        if not self.workbook:
            return
        sh = self.sheet_combo.currentText()
        df = self.workbook.cleaned.get(sh, pd.DataFrame())
        if df.empty:
            QMessageBox.information(self, "Nothing to export", "The current sheet has no cleaned data.")
            return

        default_name = re.sub(r"[^\w\-]+", "_", str(sh)).strip("_") + ".clean.csv"
        out_path, _ = QFileDialog.getSaveFileName(self, "Save cleaned sheet CSV", default_name, "CSV (*.csv)")
        if not out_path:
            return
        df.to_csv(out_path, index=False)
        QMessageBox.information(self, "Exported", f"Saved:\n{out_path}")

    def export_selected_columns_csv(self):
        if not self.workbook:
            return
        sh = self.sheet_combo.currentText()
        df = self.workbook.cleaned.get(sh, pd.DataFrame())
        if df.empty:
            QMessageBox.information(self, "Nothing to export", "The current sheet has no cleaned data.")
            return

        x_col = self.x_combo.currentText()
        y_cols = self.selected_y_columns()
        cols = [c for c in [x_col, *y_cols] if c in df.columns]
        if not cols:
            QMessageBox.information(self, "No columns selected", "Select at least one column to export.")
            return

        default_name = re.sub(r"[^\w\-]+", "_", str(sh)).strip("_") + ".selected.csv"
        out_path, _ = QFileDialog.getSaveFileName(self, "Save selected columns CSV", default_name, "CSV (*.csv)")
        if not out_path:
            return

        df.loc[:, cols].to_csv(out_path, index=False)
        QMessageBox.information(self, "Exported", f"Saved:\n{out_path}")

    def export_metadata(self):
        if not self.workbook:
            return
        sh = self.sheet_combo.currentText()
        md = self.workbook.metadata.get(sh, {})
        if not md:
            QMessageBox.information(self, "Nothing to export", "No metadata found for the current sheet.")
            return

        default_base = re.sub(r"[^\w\-]+", "_", str(sh)).strip("_") or "sheet"
        # Offer json or csv via extension
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save metadata",
            default_base + ".metadata.json",
            "JSON (*.json);;CSV (*.csv)",
        )
        if not out_path:
            return

        out_p = Path(out_path)
        suffix = out_p.suffix.lower()
        if suffix == ".csv":
            pd.DataFrame({"key": list(md.keys()), "value": list(md.values())}).to_csv(out_path, index=False)
        else:
            # write JSON without importing json (pandas handles nicely)
            pd.Series(md).to_json(out_path, indent=2)

        QMessageBox.information(self, "Exported", f"Saved:\n{out_path}")

    def export_full_sheet_with_metadata(self):
        """Export cleaned data plus metadata in a single CSV file.

        We prepend metadata as commented lines starting with '# ' so the file stays readable.
        It's also easy to parse later.
        """
        if not self.workbook:
            return
        sh = self.sheet_combo.currentText()
        df = self.workbook.cleaned.get(sh, pd.DataFrame())
        if df.empty:
            QMessageBox.information(self, "Nothing to export", "The current sheet has no cleaned data.")
            return

        md = self.workbook.metadata.get(sh, {})

        default_name = re.sub(r"[^\w\-]+", "_", str(sh)).strip("_") + ".clean_with_meta.csv"
        out_path, _ = QFileDialog.getSaveFileName(self, "Save cleaned sheet + metadata CSV", default_name, "CSV (*.csv)")
        if not out_path:
            return

        # Write with a small header block + then the CSV table.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# source_file: {self.workbook.path}\n")
            f.write(f"# sheet: {sh}\n")
            if md:
                for k, v in sorted(md.items(), key=lambda kv: kv[0].lower()):
                    f.write(f"# {k}: {v}\n")
            else:
                f.write("# (no metadata found)\n")
            f.write("\n")
        df.to_csv(out_path, index=False, mode="a")

        QMessageBox.information(self, "Exported", f"Saved:\n{out_path}")


def main():
    app = QApplication([])
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
