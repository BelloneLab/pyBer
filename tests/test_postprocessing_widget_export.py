"""Export real Qt plots, including the QGraphicsView render overload trap."""

import os
from pathlib import Path
import re
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from gui_postprocessing import PostProcessingPanel


class WidgetExportTests(unittest.TestCase):
    """Use actual painters and files rather than mocks that hide Qt overloads."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_plot_widget_exports_png_and_pdf(self):
        """Trace-only export must accept PlotWidget and include visible data."""
        plot = pg.PlotWidget(background="white")
        plot.resize(640, 400)
        plot.plot([0, 1, 2, 3], [0, 3, -2, 1], pen=pg.mkPen("red", width=4))
        plot.show()
        self.app.processEvents()
        panel = SimpleNamespace()
        for name in ("_render_widget_image", "_write_widget_pdf"):
            setattr(panel, name, MethodType(getattr(PostProcessingPanel, name), panel))
        try:
            with tempfile.TemporaryDirectory(prefix="pyber-widget-export-") as folder:
                ok, png, pdf = PostProcessingPanel._export_widget_png_pdf(
                    panel, plot, str(Path(folder) / "trace")
                )
                self.assertTrue(ok)
                image = QtGui.QImage(png)
                self.assertFalse(image.isNull())
                # A colored trace verifies real scene rendering, not merely an
                # allocated background image or a mocked save success.
                red_pixels = sum(
                    image.pixelColor(x, y).red() > 180
                    and image.pixelColor(x, y).green() < 100
                    for x in range(0, image.width(), 3)
                    for y in range(0, image.height(), 3)
                )
                self.assertGreater(red_pixels, 50)
                document = Path(pdf).read_bytes()
                self.assertTrue(document.startswith(b"%PDF"))
                self.assertGreater(len(document), 1000)
                # PDF dimensions should retain the widget's 96-DPI physical
                # size, independent of the high-resolution output painter.
                box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)", document)
                self.assertIsNotNone(box)
                self.assertAlmostEqual(float(box.group(1)), 480, delta=1)
                self.assertAlmostEqual(float(box.group(2)), 300, delta=1)
        finally:
            plot.close()
            plot.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)

    def test_regular_widget_retains_children_and_transparent_background(self):
        """Explicit QWidget dispatch must also retain ordinary panel content."""
        widget = QtWidgets.QWidget()
        widget.resize(160, 100)
        child = QtWidgets.QLabel(widget)
        child.setGeometry(100, 60, 50, 30)
        child.setStyleSheet("background: #ff0000;")
        widget.show()
        self.app.processEvents()
        try:
            image = PostProcessingPanel._render_widget_image(None, widget)
            dpr = image.devicePixelRatio()
            self.assertEqual(image.pixelColor(int(120 * dpr), int(70 * dpr)).name(), "#ff0000")
            self.assertEqual(image.pixelColor(int(10 * dpr), int(10 * dpr)).alpha(), 0)
        finally:
            widget.close()
            widget.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    unittest.main()
