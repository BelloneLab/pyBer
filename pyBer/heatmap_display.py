"""Display-only heatmap colors shared by the live plot and figure exports."""
import numpy as np


def color_name(style):
    """Old projects keep their palette; new PSTH choices do not affect spatial maps."""
    return str(style.get("psth_heatmap_cmap") or style.get("heatmap_cmap", "viridis"))


def color_map(style):
    import pyqtgraph as pg
    name = color_name(style)
    try:
        return pg.colormap.get(name)
    except (FileNotFoundError, ValueError):
        # Some palettes in the legacy style dialog (e.g. gray) are supplied by
        # matplotlib rather than pyqtgraph's bundled maps.
        try:
            return pg.colormap.get(name, source="matplotlib")
        except (FileNotFoundError, ValueError, KeyError):
            return pg.colormap.get("viridis")


def color_limits(matrix, mode=0, style=None):
    """Finite-only limits; never alter, normalize or clip the source array."""
    values = np.asarray(matrix, float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0., 1.
    low, high = float(finite.min()), float(finite.max())
    if mode == 1:
        low, high = np.percentile(finite, [2, 98]).tolist()
    elif mode == 2:
        limit = max(abs(low), abs(high))
        low, high = -limit, limit
    style = style or {}
    if style.get("heatmap_levels_manual", False):
        try:
            bounds = float(style["heatmap_min"]), float(style["heatmap_max"])
            if np.all(np.isfinite(bounds)) and bounds[0] < bounds[1]:
                return bounds
        except (KeyError, TypeError, ValueError):
            pass
    if high <= low:
        low, high = low - .5, high + .5
    return low, high


def matplotlib_color_map(style):
    """Use precisely the same lookup table in PNG/PDF/SVG as in the live view."""
    from matplotlib.colors import ListedColormap
    lookup = color_map(style).getLookupTable(nPts=256, alpha=True)
    return ListedColormap(np.asarray(lookup, float) / 255., name=color_name(style))
