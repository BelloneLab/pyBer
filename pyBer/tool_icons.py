"""Small, optically balanced vector drawings for the analysis tool rails.

Artwork uses one 24-unit canvas, rounded 1.65-unit strokes and no fonts or
external files. Keep the paths here so every tool can be refined in one place.
"""

from __future__ import annotations

from functools import lru_cache

from PySide6 import QtCore, QtGui, QtSvg

ICON_COLOR = "#ffffff"
ICON_STROKE = 1.65
ICON_SIZES = (16, 20, 22, 24, 28, 32, 40, 44, 48, 64, 80, 96)

# Each drawing has a distinct analytical meaning, not just a decorative shape.
ICON_ART = {
    "data": '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
            '<path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
    "setup": '<path d="M3 5h5m4 0h9M3 12h11m4 0h3M3 19h3m4 0h11"/>'
             '<circle cx="10" cy="5" r="2"/><circle cx="16" cy="12" r="2"/>'
             '<circle cx="8" cy="19" r="2"/>',
    "artifacts": '<path d="M2.5 15h4l2-4 2.5 10 3-18 2.5 12H21M18 3l3 3m0-3-3 3"/>',
    "filtering": '<path d="M3 3.5h18l-7 8.5v7l-4 2v-9Z"/>'
                 '<path d="M7 7h10"/>',
    "baseline": '<path d="M3 7c2-6 3 7 5 3s3-5 5 0 3 4 4 2 2-1 4 0"/>'
                '<path d="M3 19h3m3 0h3m3 0h3m3 0h.1"/>',
    "output": '<path d="M3 3v18h18M6 15c3 0 3-8 6-8s3 5 6 5h3"/>'
              '<path d="m18 9 3 3-3 3"/>',
    "psth": '<path d="M2.5 20.5H5v-5h3v-5h3v-7h3v4h3v8h3v5h1.5"/>',
    "qc": '<path d="M12 2.5 20 6v5c0 5-3.5 8.5-8 10.5C7.5 19.5 4 16 4 11V6Z"/>'
          '<path d="m8 11.5 2.5 2.5 5.5-6"/>',
    "export": '<path d="M12 2.5v12m-4-4 4 4 4-4M3.5 15v5.5h17V15"/>',
    "config": '<path d="M9.6 2h4.8l.6 3 2 .9 2.7-1 2.3 4-2.2 2v2.2l2.2 2-2.3 4-2.7-1-2 .9-.6 3H9.6l-.6-3-2-.9-2.7 1-2.3-4 2.2-2v-2.2L2 9l2.3-4L7 6l2-.9Z"/>'
              '<circle cx="12" cy="12" r="3"/>',
    "spatial": '<rect x="2.5" y="2.5" width="19" height="19" rx="3"/>'
               '<path d="M6 16c1-6 5 3 6-4s5-5 6-5"/>'
               '<circle cx="6" cy="16" r="1"/><circle cx="18" cy="7" r="1"/>',
    "modeling": '<path d="M3 3v18h18M6 17c5 0 4-11 14-11"/>'
                '<circle cx="7" cy="13" r=".75"/><circle cx="12" cy="9" r=".75"/>'
                '<circle cx="17" cy="10" r=".75"/>',
    "events": '<path d="M2.5 17h4V6h6v11h3V9h4v8h2"/>'
              '<path d="M6.5 2v1m0 18v1M15.5 3v2m0 16v1"/>',
    "behavior": '<path d="M7.5 13c1.5-1 1.6-3 4.5-3s3 2 4.5 3c4 3 1 8-2 6.5-2-1-3-1-5 0C6.5 21 3.5 16 7.5 13Z"/>'
                '<ellipse cx="4.5" cy="9" rx="1.5" ry="2.2" transform="rotate(-25 4.5 9)"/>'
                '<ellipse cx="9" cy="5" rx="1.5" ry="2.2" transform="rotate(-10 9 5)"/>'
                '<ellipse cx="15" cy="5" rx="1.5" ry="2.2" transform="rotate(10 15 5)"/>'
                '<ellipse cx="19.5" cy="9" rx="1.5" ry="2.2" transform="rotate(25 19.5 9)"/>',
    "sync": '<path d="M2.5 9h5V4h5v5h9M2.5 19h5v-5h5v5h9"/>'
            '<path d="M7.5 1.5V2m0 9v1m0 9v1.5M17 3.5h4m-2-2 2 2-2 2"/>',
    "list": '<path d="M8 5h13M8 12h13M8 19h13M3 5h.1M3 12h.1M3 19h.1"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/>'
              '<path d="M12 1v4m0 14v4M1 12h4m14 0h4"/>',
}


@lru_cache(maxsize=96)
def _renderer(name: str, color: str) -> QtSvg.QSvgRenderer:
    """Cache parsed vector paths; all consumers draw on the Qt GUI thread."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
           f'fill="none" stroke="{color}" stroke-width="{ICON_STROKE}" '
           f'stroke-linecap="round" stroke-linejoin="round">{ICON_ART[name]}</svg>')
    return QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("ascii")))


def _icon_painter(name: str):
    """Adapt a named drawing to the legacy (painter, bounds, color) interface."""
    def paint(painter, bounds, color):
        """Render without leaving altered pen, transform, or opacity state."""
        painter.save()
        _renderer(name, QtGui.QColor(color).name()).render(painter, QtCore.QRectF(bounds))
        painter.restore()
    paint.__name__ = f"paint_{name}"
    return paint


def _make_icon(painter_fn, size: int = 40, color: str = ICON_COLOR) -> QtGui.QIcon:
    """Rasterize each target size directly, avoiding blurry scaled tiny icons.

    Explicit active/on images prevent Qt from tinting white artwork on selected
    buttons. Disabled icons retain their hue with lower alpha for legibility.
    """
    icon = QtGui.QIcon()
    for edge in sorted(set(ICON_SIZES + (max(1, int(size)),))):
        pixmap = QtGui.QPixmap(edge, edge)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter_fn(painter, QtCore.QRectF(0, 0, edge, edge), QtGui.QColor(color))
        painter.end()
        checked = pixmap
        if QtGui.QColor(color) != QtGui.QColor(ICON_COLOR):
            # Light-theme rails use ink at rest, but checked buttons have a
            # violet background and need the same white art as the dark theme.
            checked = QtGui.QPixmap(edge, edge)
            checked.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(checked)
            painter_fn(painter, QtCore.QRectF(0, 0, edge, edge), QtGui.QColor(ICON_COLOR))
            painter.end()
        for mode in (QtGui.QIcon.Mode.Normal, QtGui.QIcon.Mode.Active, QtGui.QIcon.Mode.Selected):
            for state in (QtGui.QIcon.State.Off, QtGui.QIcon.State.On):
                selected = state == QtGui.QIcon.State.On or mode == QtGui.QIcon.Mode.Selected
                icon.addPixmap(checked if selected else pixmap, mode, state)
        disabled = QtGui.QPixmap(edge, edge)
        disabled.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(disabled)
        painter.setOpacity(0.38)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        for state in (QtGui.QIcon.State.Off, QtGui.QIcon.State.On):
            icon.addPixmap(disabled, QtGui.QIcon.Mode.Disabled, state)
    return icon


# Retain imports used by existing rails while sharing the new vector artwork.
_paint_database = _icon_painter("data")
_paint_sliders = _icon_painter("setup")
_paint_artifacts = _icon_painter("artifacts")
_paint_filter = _icon_painter("filtering")
_paint_wave = _icon_painter("baseline")
_paint_output = _icon_painter("output")
_paint_chart = _icon_painter("psth")
_paint_badge = _icon_painter("qc")
_paint_export = _icon_painter("export")
_paint_gear = _icon_painter("config")
_paint_grid = _icon_painter("spatial")
_paint_temporal = _icon_painter("modeling")
_paint_pulse = _icon_painter("events")
_paint_paw = _icon_painter("behavior")
_paint_sync = _icon_painter("sync")
_paint_list = _icon_painter("list")
_paint_target = _icon_painter("target")
