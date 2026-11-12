"""Editable vector export: one path per curve, real text, no raster backgrounds."""
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from vector_figure_export import export_vector_files


class VectorExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def build(self, with_image=False):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        plot = pg.PlotWidget(title="PSTH: mean SEM", background="#111d30")
        layout.addWidget(plot)
        x = np.linspace(-5, 10, 800)
        plot.plot(x, np.sin(x), pen=pg.mkPen("#ff0000", width=2.5))        # thick: segmented on screen
        high = plot.plot(x, np.sin(x) + 0.2, pen=None)
        low = plot.plot(x, np.sin(x) - 0.2, pen=None)
        plot.addItem(pg.FillBetweenItem(high, low, brush=(0, 255, 0, 60)))
        plot.plot([0, 1, 2], [0, 1, 0], pen=None, symbol="o", symbolBrush=["#0000ff", "#00ffff", "#ff00ff"])
        plot.addItem(pg.BarGraphItem(x=[5, 6], height=[1, 2], width=0.5, brush="#ffaa00"))
        plot.addItem(pg.ErrorBarItem(x=np.array([8.0]), y=np.array([0.0]), top=np.array([0.5]),
                                     bottom=np.array([0.5]), beam=0.2, pen=pg.mkPen("#ffffff")))
        plot.addLine(x=0, pen=pg.mkPen("#cccccc", style=QtCore.Qt.PenStyle.DashLine))
        note = pg.TextItem("note text", anchor=(0.5, 0))
        note.setPos(2, 1.2)
        plot.addItem(note)
        plot.setLabel("left", "dFF")
        plot.setLabel("bottom", "Time (s)")
        if with_image:
            image = pg.ImageItem(np.random.default_rng(0).random((10, 6)))
            plot.addItem(image)
        root.resize(640, 400)
        root.show()
        self.app.processEvents()
        return root

    def test_svg_and_pdf_are_editable_vectors(self):
        root = self.build()
        try:
            with tempfile.TemporaryDirectory() as folder:
                written = export_vector_files(root, str(Path(folder) / "figure"))
                svg = Path(written["svg"]).read_text(encoding="utf-8")
                pdf = Path(written["pdf"]).read_bytes()
        finally:
            root.close()
        # One continuous path for the 800-sample red curve (Qt's SVG export
        # splits such thick curves into hundreds of 2-point polylines).
        red_paths = re.findall(r"<path[^>]*stroke: ?#ff0000", svg)
        self.assertEqual(len(red_paths), 1)
        self.assertNotIn("non-scaling-stroke", svg)
        self.assertNotIn("<polyline", svg)
        self.assertNotIn("<image", svg)         # no rasterized plot background
        for text in ("note text", "Time (s)", "dFF", "PSTH: mean SEM"):
            self.assertIn(text, svg)              # text stays text
        self.assertIn("#00ff00", svg.lower())     # fill color is a plain editable color
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(pdf.count(b"/Subtype /Image"), 0)
        self.assertIn(b"FontFile2", pdf)          # embedded TrueType: editable text
        box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", pdf)
        self.assertAlmostEqual(float(box.group(1)), 480, delta=1)
        self.assertAlmostEqual(float(box.group(2)), 300, delta=1)

    def test_heatmap_image_is_the_only_raster(self):
        root = self.build(with_image=True)
        try:
            with tempfile.TemporaryDirectory() as folder:
                written = export_vector_files(root, str(Path(folder) / "figure"), formats=("svg",))
                svg = Path(written["svg"]).read_text(encoding="utf-8")
        finally:
            root.close()
        self.assertEqual(svg.count("<image"), 1)


if __name__ == "__main__":
    unittest.main()
