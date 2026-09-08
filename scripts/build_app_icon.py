"""Build the Windows icon from its editable vector source.

Run in the pyBer environment from an IDE or with python scripts/build_app_icon.py.
Each shell size is rendered separately with supersampling for clean small edges.
"""
from pathlib import Path
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
SUPERSAMPLE = 4


def main():
    """Write the taskbar PNG and full ICO directory without changing splash art."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    renderer = QtSvg.QSvgRenderer(str(ROOT / "assets/pyBer_taskbar.svg"))
    if not renderer.isValid():
        raise RuntimeError("Invalid application icon SVG")
    frames = []
    for size in SIZES:
        image = QtGui.QImage(size * SUPERSAMPLE, size * SUPERSAMPLE,
                            QtGui.QImage.Format.Format_RGBA8888)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        renderer.render(painter)
        painter.end()
        frame = Image.frombytes("RGBA", (image.width(), image.height()),
                                bytes(image.constBits())).resize((size, size), Image.Resampling.LANCZOS)
        frames.append(frame)
    frames[-1].save(ROOT / "assets/pyBer_taskbar.png")
    frames[-1].save(ROOT / "assets/pyBer.ico", format="ICO",
                    sizes=[(size, size) for size in SIZES], append_images=frames[:-1])


if __name__ == "__main__":
    main()
