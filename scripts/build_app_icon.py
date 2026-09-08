"""Build the Windows icon from its editable vector source.

Run in the pyBer environment from an IDE or with python scripts/build_app_icon.py.
Each shell size is rendered separately with supersampling for clean small edges.
"""
from pathlib import Path
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
SUPERSAMPLE = 4
LOGO_SIZE = 1000


def render_frame(renderer, size):
    """Rasterize vector art with transparent margins and antialiased edges."""
    image = QtGui.QImage(size * SUPERSAMPLE, size * SUPERSAMPLE,
                        QtGui.QImage.Format.Format_RGBA8888)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    return Image.frombytes("RGBA", (image.width(), image.height()),
                           bytes(image.constBits())).resize((size, size), Image.Resampling.LANCZOS)


def main():
    """Build splash and frameless shell assets from the same editable SVG logo.

    Only the frame is removed for the taskbar; the mouse, optics and fiber stay
    identical, so future changes to the original motif reach every app icon.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = ROOT / "assets/pyBer_logo.svg"
    logo_renderer = QtSvg.QSvgRenderer(str(source))
    if not logo_renderer.isValid():
        raise RuntimeError("Invalid application logo SVG")
    render_frame(logo_renderer, LOGO_SIZE).save(ROOT / "assets/pyBer_logo_big.png")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    artwork = ET.parse(source)
    root = artwork.getroot()
    for element in list(root):
        if element.get("id") == "logo-frame":
            root.remove(element)
    # Crop the former border margin to give the original motif room at 16 px.
    root.set("viewBox", "24 22 452 452")
    # Removing frame/comments can leave whitespace-only XML tails.
    serialized = ET.tostring(root, encoding="unicode")
    (ROOT / "assets/pyBer_taskbar.svg").write_text(
        "\n".join(line.rstrip() for line in serialized.splitlines()) + "\n", encoding="utf-8")
    renderer = QtSvg.QSvgRenderer(str(ROOT / "assets/pyBer_taskbar.svg"))
    if not renderer.isValid():
        raise RuntimeError("Invalid application icon SVG")
    frames = []
    for size in SIZES:
        frames.append(render_frame(renderer, size))
    frames[-1].save(ROOT / "assets/pyBer_taskbar.png")
    frames[-1].save(ROOT / "assets/pyBer.ico", format="ICO",
                    sizes=[(size, size) for size in SIZES], append_images=frames[:-1])


if __name__ == "__main__":
    main()
