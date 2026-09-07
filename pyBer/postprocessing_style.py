"""Cohesive presentation presets for postprocessing plots and plot cards.

This module changes presentation only. Presets retain the scientific colormap,
and none of the helpers modify curves, samples, plot ranges, or analysis state.
"""

from copy import deepcopy
from typing import Any, Dict, Union

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# All visual tuning lives here so palettes can be adjusted without touching
# the processing or rendering methods. Plot-style entries match panel._style.
POSTPROCESSING_PRESETS: Dict[str, Dict[str, Any]] = {
    "Midnight": {
        "label": "Midnight",
        "surface": "#111d30", "border": "#26364c",
        "text": "#e5eef8", "muted": "#95a8c0", "axis": "#44556e",
        "accent": "#56d5d0", "accent_soft": "#153d47",
        "style": {
            "plot_bg": (17, 29, 48), "trace": (101, 196, 235),
            "avg": (86, 213, 208), "behavior": (236, 187, 106),
            "sem_edge": (86, 213, 208, 85), "sem_fill": (86, 213, 208, 42),
            "grid_enabled": True, "grid_alpha": 0.09,
            "heatmap_cmap": "viridis",
        },
    },
    "Paper": {
        "label": "Paper",
        "surface": "#ffffff", "border": "#dce5e9",
        "text": "#23374b", "muted": "#64798a", "axis": "#bccbd3",
        "accent": "#087f8c", "accent_soft": "#e6f4f3",
        "style": {
            "plot_bg": (255, 255, 255), "trace": (55, 117, 165),
            "avg": (8, 127, 140), "behavior": (184, 127, 47),
            "sem_edge": (8, 127, 140, 65), "sem_fill": (8, 127, 140, 35),
            "grid_enabled": True, "grid_alpha": 0.12,
            "heatmap_cmap": "viridis",
        },
    },
    "Sand": {
        "label": "Sand",
        "surface": "#fcfaf5", "border": "#e3dccf",
        "text": "#3f463d", "muted": "#7c8072", "axis": "#c6c5b6",
        "accent": "#347d72", "accent_soft": "#eaf0e5",
        "style": {
            "plot_bg": (252, 250, 245), "trace": (67, 118, 144),
            "avg": (52, 125, 114), "behavior": (178, 123, 63),
            "sem_edge": (52, 125, 114, 65), "sem_fill": (52, 125, 114, 36),
            "grid_enabled": True, "grid_alpha": 0.12,
            "heatmap_cmap": "viridis",
        },
    },
}

Preset = Union[str, Dict[str, Any]]


def _palette(preset: Preset) -> Dict[str, Any]:
    """Resolve a named palette, rejecting typos rather than changing silently."""
    if isinstance(preset, str):
        if preset not in POSTPROCESSING_PRESETS:
            raise ValueError(f"Unknown postprocessing preset: {preset}")
        return POSTPROCESSING_PRESETS[preset]
    return preset


def style_plot(plot: pg.PlotWidget, preset: Preset = "Midnight") -> None:
    """Apply restrained axes and hierarchy without changing plotted content.

    The existing title text is retained. Axis tick values, ranges, interaction,
    and custom scientific axis labels continue to be controlled by the panel.
    """
    palette = _palette(preset)
    item = plot.getPlotItem()
    plot.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    plot.setStyleSheet("border: none; background: transparent;")
    plot.setBackground(palette["surface"])
    item.layout.setContentsMargins(8, 8, 12, 6)
    item.layout.setSpacing(4)
    item.showGrid(x=False, y=True, alpha=palette["style"]["grid_alpha"])

    tick_font = QtGui.QFont("Segoe UI", 9)
    tick_font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
    for side in ("left", "right", "bottom", "top"):
        axis = item.getAxis(side)
        axis.setPen(pg.mkPen(palette["axis"], width=0.7))
        axis.setTextPen(pg.mkPen(palette["muted"]))
        axis.setTickFont(tick_font)
        axis.setStyle(tickLength=-4, tickTextOffset=7,
                      autoExpandTextSpace=True, hideOverlappingLabels=True)
        axis.label.setDefaultTextColor(QtGui.QColor(palette["muted"]))
        axis.label.setFont(tick_font)

    # LabelItem retains these options when the panel updates dynamic titles.
    title = item.titleLabel
    title.setText(title.text, color=palette["text"], size="10pt",
                  bold=True, justify="left")


class PlotCard(QtWidgets.QFrame):
    """A compact, themed frame around an existing interactive plot widget."""

    def __init__(self, widget: QtWidgets.QWidget, title: str, subtitle: str = "",
                 preset: Preset = "Midnight", parent=None):
        super().__init__(parent)
        self.setObjectName("postprocessingPlotCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("postprocessingCardTitle")
        self.title_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.title_label)
        self.subtitle_label = QtWidgets.QLabel(subtitle)
        self.subtitle_label.setObjectName("postprocessingCardSubtitle")
        self.subtitle_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)
        self.plot_widget = widget
        layout.addWidget(widget, stretch=1)
        self.set_preset(preset)

    def set_preset(self, preset: Preset) -> None:
        """Refresh only this card's colors, avoiding application-wide styles."""
        palette = _palette(preset)
        self.setStyleSheet(f"""
            QFrame#postprocessingPlotCard {{
                background: {palette['surface']};
                border: 1px solid {palette['border']}; border-radius: 12px;
            }}
            QLabel#postprocessingCardTitle {{
                color: {palette['text']}; background: transparent; border: none;
                font-family: 'Segoe UI'; font-size: 13px; font-weight: 600;
                padding: 0px 2px 2px 2px;
            }}
            QLabel#postprocessingCardSubtitle {{
                color: {palette['muted']}; background: transparent; border: none;
                font-family: 'Segoe UI'; font-size: 11px; padding: 0px 2px 5px 2px;
            }}
        """)
        if isinstance(self.plot_widget, pg.PlotWidget):
            style_plot(self.plot_widget, palette)


def create_plot_card(widget: QtWidgets.QWidget, title: str, subtitle: str = "",
                     preset: Preset = "Midnight", parent=None) -> PlotCard:
    """Wrap an existing widget; callers retain their original plot reference."""
    return PlotCard(widget, title, subtitle, preset, parent)


def apply_plot_preset(panel: Any, preset: Preset = "Midnight") -> None:
    """Apply a palette to a postprocessing panel and its registered plot cards.

    Custom analysis parameters and heatmap contrast limits are retained. The
    caller decides when to save settings, making this suitable for previews.
    """
    palette = _palette(preset)
    panel._style.update(deepcopy(palette["style"]))
    panel._style["postprocessing_preset"] = palette["label"]
    panel._apply_plot_style()
    # Inspect named plot attributes only, so embedded layouts and other tabs
    # retain ownership of their styling and interactive behavior.
    for name, widget in vars(panel).items():
        if name.startswith("plot_") and isinstance(widget, pg.PlotWidget):
            style_plot(widget, palette)
    for card in getattr(panel, "_postprocessing_plot_cards", []):
        card.set_preset(palette)
