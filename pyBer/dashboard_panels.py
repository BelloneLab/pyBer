"""Seamless results figure: invisible dividers, removable and replaceable panels.

The postprocessing dashboard is presented as one figure (no card outlines),
yet every boundary stays draggable. Right-clicking a plot adds three panel
actions on top of pyqtgraph's own menu:

* **Remove this panel**: the panel hides and its neighbor takes the space.
* **Change to**: any available analysis (behavior summary, pre/post metric,
  global signal metric). A slot's own family switches its existing selector;
  another family is drawn by a substitute plot placed in the same slot.
* **Restore removed panels / Reset dashboard layout**.

Native plot widgets are reparented, never recreated, so their data, exports
and signal connections are untouched. Only presentation state is stored.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from global_signal_metrics import GLOBAL_SIGNAL_METRICS
from psth_metrics import METRICS, metric_id

# Slot order is also the order used in menus and persisted state.
SLOT_KEYS = ("trace", "heatmap", "psth", "bout1", "bout2", "comparison", "global")
SLOT_LABELS = {
    "trace": "Trace preview", "heatmap": "Heatmap", "psth": "Average PSTH",
    "bout1": "Behavior plot 1", "bout2": "Behavior plot 2",
    "comparison": "Pre/post comparison", "global": "Global signal metric",
}
# Which analysis family a slot natively displays (switch its own selector).
NATIVE_FAMILY = {"bout1": "behavior", "bout2": "behavior", "comparison": "metric", "global": "global"}
FAMILY_LABELS = {"behavior": "Behavior summary", "metric": "Pre/post comparison", "global": "Global signal"}
HANDLE_PX = 8            # thickness of invisible draggable dividers
TIME_RATIO_DEFAULT = [1000, 1000]


# ---------------------------------------------------------------------------
# Invisible divider for the shared heatmap / PSTH grid
# ---------------------------------------------------------------------------
class GridRowHandle(QtWidgets.QWidget):
    """A transparent, draggable divider between two QGridLayout rows.

    Unlike a QSplitter, both plots stay in one grid column, so their plotting
    areas keep identical widths and the shared time axis stays pixel aligned.
    It mimics the small QSplitter API used by PlotSplitterPreferences.
    """

    splitterMoved = QtCore.Signal(int, int)

    def __init__(self, grid: QtWidgets.QGridLayout, top_row: int, bottom_row: int,
                 top_widget: QtWidgets.QWidget, bottom_widget: QtWidgets.QWidget, parent=None) -> None:
        super().__init__(parent)
        self.grid, self.top_row, self.bottom_row = grid, top_row, bottom_row
        self.top_widget, self.bottom_widget = top_widget, bottom_widget
        self._ratio = list(TIME_RATIO_DEFAULT)
        self._drag_origin: Optional[Tuple[float, int, int]] = None
        self._hover = False
        self.accent = QtGui.QColor("#56d5d0")
        self.setFixedHeight(HANDLE_PX)
        self.setCursor(QtCore.Qt.CursorShape.SplitVCursor)
        self.setToolTip("Drag to change the heatmap / PSTH height ratio")
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)

    # QSplitter-like API -------------------------------------------------
    def count(self) -> int:
        return 2

    def sizes(self) -> List[int]:
        return list(self._ratio)

    def setSizes(self, sizes: Sequence[int]) -> None:
        if len(sizes) == 2 and all(int(v) > 0 for v in sizes):
            self._ratio = [int(v) for v in sizes]
            self.refresh()

    def refresh(self) -> None:
        """Apply the ratio; a hidden plot's row gets no stretch (no empty gap)."""
        top_visible = not self.top_widget.isHidden()
        bottom_visible = not self.bottom_widget.isHidden()
        self.grid.setRowStretch(self.top_row, self._ratio[0] if top_visible else 0)
        self.grid.setRowStretch(self.bottom_row, self._ratio[1] if bottom_visible else 0)
        self.setVisible(top_visible and bottom_visible)

    # Interaction ----------------------------------------------------------
    def enterEvent(self, event):
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        if not (self._hover or self._drag_origin):
            return
        painter = QtGui.QPainter(self)
        color = QtGui.QColor(self.accent)
        color.setAlpha(110)
        painter.fillRect(QtCore.QRectF(0, self.height() / 2 - 1, self.width(), 2), color)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_origin = (event.globalPosition().y(), self.top_widget.height(), self.bottom_widget.height())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is None:
            return super().mouseMoveEvent(event)
        y0, top0, bottom0 = self._drag_origin
        delta = event.globalPosition().y() - y0
        total = max(2, top0 + bottom0)
        top = int(np.clip(top0 + delta, self.top_widget.minimumHeight() or 40,
                          total - (self.bottom_widget.minimumHeight() or 40)))
        bottom = max(1, total - top)
        scale = 2000.0 / total
        self._ratio = [max(1, int(round(top * scale))), max(1, int(round(bottom * scale)))]
        self.refresh()
        self.splitterMoved.emit(top, 1)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag_origin is not None:
            self._drag_origin = None
            self.update()
            self.splitterMoved.emit(self.top_widget.height(), 1)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click restores equal heights."""
        self.setSizes(TIME_RATIO_DEFAULT)
        self.splitterMoved.emit(self.top_widget.height(), 1)


# ---------------------------------------------------------------------------
# Seamless styling
# ---------------------------------------------------------------------------
def seamless_figure_qss(palette: Dict[str, object]) -> str:
    """One rounded surface for the whole results figure, invisible dividers."""
    accent = QtGui.QColor(str(palette["accent"]))
    hover = f"rgba({accent.red()},{accent.green()},{accent.blue()},90)"
    return f"""
        QScrollArea#resultsFigure {{
            background: {palette['surface']};
            border: 1px solid {palette['border']}; border-radius: 12px;
        }}
        QWidget#resultsFigureViewport, QWidget#figureSurface,
        QSplitter#figureSurface, QFrame#postprocessingPlotCard {{
            background: transparent; border: none;
        }}
        QSplitter#figureSurface::handle {{ background: transparent; }}
        QSplitter#figureSurface::handle:hover {{ background: {hover}; }}
    """


def draw_global_metric(plot: pg.PlotWidget, key: str, values: np.ndarray, grouped: bool,
                       palette: Dict[str, object], colors: Optional[Sequence[str]] = None) -> bool:
    """Recording points, median line and mean +/- SEM (same encoding as the native panel)."""
    info = GLOBAL_SIGNAL_METRICS.get(key)
    plot.clear()
    if info is None:
        return False
    plot.setTitle(info["label"], color=palette["text"], size="11pt")
    plot.setLabel("left", info["unit"])
    plot.setToolTip(info["description"] + "\nPoints are recordings, the line is the median, "
                    "the diamond is mean +/- SEM (group scope).")
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if not finite.any():
        return False
    vals = values[finite]
    n = int(vals.size)
    x = np.linspace(-0.12, 0.12, n) if n > 1 else np.zeros(1)
    if colors is not None and len(colors) == values.size:
        brushes = [pg.mkBrush(c) for c, keep in zip(colors, finite) if keep]
    else:
        brushes = [pg.mkBrush(str(palette["accent"]))] * n
    plot.plot(x, vals, pen=None, symbol="o", symbolSize=5, symbolBrush=brushes, symbolPen=None)
    median = float(np.median(vals))
    neutral = str(palette["text"]) if colors is not None else str(palette["accent"])
    plot.plot([-0.2, 0.2], [median, median], pen=pg.mkPen(neutral, width=2))
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if grouped and n > 1:
        plot.plot([0.28], [mean], pen=None, symbol="d", symbolSize=7,
                  symbolBrush=pg.mkBrush(str(palette["text"])), symbolPen=None)
        plot.addItem(pg.ErrorBarItem(x=np.array([0.28]), y=np.array([mean]), top=np.array([sem]),
                                     bottom=np.array([sem]), beam=0.1, pen=pg.mkPen(str(palette["text"]), width=1)))
    plot.getAxis("bottom").setTicks([[(0.0, f"{n} recordings" if grouped else "Recording")]])
    plot.setXRange(-0.5, 0.5, padding=0)
    low, high = min(float(vals.min()), mean - sem), max(float(vals.max()), mean + sem)
    margin = max((high - low) * 0.2, abs(mean) * 0.12, 1e-6)
    plot.setYRange(low - margin, high + margin, padding=0)
    return True


# ---------------------------------------------------------------------------
# Panel manager
# ---------------------------------------------------------------------------
class DashboardPanels(QtCore.QObject):
    """Remove, restore and change dashboard panels; keep the figure seamless."""

    changed = QtCore.Signal()

    def __init__(self, panel) -> None:
        super().__init__(panel)
        self.panel = panel
        self.removed: set = set()
        self.substitutes: Dict[str, str] = {}
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        self._layout_allows: Dict[str, bool] = {key: True for key in SLOT_KEYS}
        self._row_avg_layout = True
        self._applying = False
        self._hooked: set = set()
        self._holder = QtWidgets.QWidget(panel)
        self._holder.hide()
        for slot in SLOT_KEYS:
            self.hook_plot(self.native(slot), slot)

    # -- slot geometry ------------------------------------------------------
    def native(self, slot: str) -> QtWidgets.QWidget:
        p = self.panel
        return {"trace": p.plot_trace, "heatmap": p.plot_heat, "psth": p.plot_avg,
                "bout1": p.plot_dur, "bout2": p.plot_bout_second,
                "comparison": p.plot_metrics, "global": p.plot_global}[slot]

    def displayed(self, slot: str) -> QtWidgets.QWidget:
        return self._widgets.get(slot) if slot in self.substitutes and slot in self._widgets else self.native(slot)

    def _swap(self, slot: str, old: QtWidgets.QWidget, new: QtWidgets.QWidget) -> None:
        """Put ``new`` exactly where ``old`` is; park ``old`` in a hidden holder."""
        p = self.panel
        if slot in ("bout1", "bout2"):
            splitter = p._view_splitters["bouts"]
            index = splitter.indexOf(old)
            if index < 0:
                return
            policy = old.sizePolicy()
            new.setSizePolicy(policy)
            new.setMinimumWidth(0)
            splitter.replaceWidget(index, new)
        else:
            if slot == "trace":
                layout = p.trace_card.layout()
            elif slot in ("heatmap", "psth"):
                layout = p.heat_figure.layout()
            else:
                layout = p._plot_card_by_widget[self.native(slot)].layout()
            if layout.replaceWidget(old, new) is None:
                return
        old.setParent(self._holder)
        new.show()

    # -- catalog -------------------------------------------------------------
    def catalog(self) -> List[Tuple[str, str, str]]:
        """(family, code, label) for every analysis a panel can show."""
        p = self.panel
        items = []
        combo = p.combo_psth_behavior_metric
        for index in range(combo.count()):
            items.append(("behavior", str(combo.itemData(index)), combo.itemText(index)))
        for key, label in METRICS.items():
            items.append(("metric", key, label))
        for key, info in GLOBAL_SIGNAL_METRICS.items():
            items.append(("global", key, info["label"]))
        return items

    def current_spec(self, slot: str) -> Optional[str]:
        """'family:code' currently drawn in ``slot`` (None for fixed views)."""
        if slot in self.substitutes:
            return self.substitutes[slot]
        p = self.panel
        if slot == "bout1":
            return f"behavior:{p.combo_psth_behavior_metric.currentData()}"
        if slot == "bout2":
            return f"behavior:{p.combo_psth_behavior_metric_second.currentData()}"
        if slot == "comparison":
            return f"metric:{metric_id(p.combo_metric.currentText())}"
        if slot == "global":
            return f"global:{p.combo_global_metric.currentData()}"
        return None

    def label_for(self, slot: str) -> str:
        spec = self.current_spec(slot)
        if spec:
            family, code = spec.split(":", 1)
            for fam, key, label in self.catalog():
                if fam == family and key == code:
                    return label
        return SLOT_LABELS[slot]

    # -- edits ---------------------------------------------------------------
    def set_spec(self, slot: str, spec: str, emit: bool = True) -> None:
        family, code = spec.split(":", 1)
        if NATIVE_FAMILY.get(slot) == family:
            self._restore_native(slot)
            self._set_native_selector(slot, code)
        else:
            widget = self._substitute_widget(slot, family)
            current = self.displayed(slot)
            if current is not widget:
                self._swap(slot, current, widget)
            self.substitutes[slot] = spec
        self.removed.discard(slot)
        self.apply()
        self._sync_heat_controls()
        self.render_substitutes()
        if emit:
            self.changed.emit()

    def _set_native_selector(self, slot: str, code: str) -> None:
        p = self.panel
        if slot in ("bout1", "bout2"):
            combo = p.combo_psth_behavior_metric if slot == "bout1" else p.combo_psth_behavior_metric_second
            index = combo.findData(code)
        elif slot == "comparison":
            combo = p.combo_metric
            index = combo.findText(METRICS.get(code, ""))
        else:
            combo = p.combo_global_metric
            index = combo.findData(code)
        if index >= 0 and combo.currentIndex() != index:
            combo.setCurrentIndex(index)

    def _restore_native(self, slot: str) -> None:
        if slot not in self.substitutes:
            return
        widget = self._widgets.get(slot)
        native = self.native(slot)
        if widget is not None and native.parent() is self._holder:
            self._swap(slot, widget, native)
        del self.substitutes[slot]

    def restore_original(self, slot: str) -> None:
        self._restore_native(slot)
        self.apply()
        self._sync_heat_controls()
        self.changed.emit()

    def remove(self, slot: str) -> None:
        visible = [key for key in SLOT_KEYS if key not in self.removed]
        if slot in self.removed or len(visible) <= 1:
            return   # never remove the last panel
        self.removed.add(slot)
        self.apply()
        self._sync_heat_controls()
        self.changed.emit()

    def restore(self, slot: str) -> None:
        if slot in self.removed:
            self.removed.discard(slot)
            self.apply()
            self._sync_heat_controls()
            self.changed.emit()

    def restore_all(self) -> None:
        if self.removed:
            self.removed.clear()
            self.apply()
            self._sync_heat_controls()
            self.changed.emit()

    def reset(self) -> None:
        for slot in list(self.substitutes):
            self._restore_native(slot)
        self.removed.clear()
        handle = getattr(self.panel, "psth_time_handle", None)
        if handle is not None:
            handle.setSizes(TIME_RATIO_DEFAULT)
        self.panel.plot_splitter_preferences.reset()
        self.apply()
        self._sync_heat_controls()
        self.changed.emit()

    # -- persistence ---------------------------------------------------------
    def state(self) -> Dict[str, object]:
        handle = getattr(self.panel, "psth_time_handle", None)
        return {"removed": [key for key in SLOT_KEYS if key in self.removed],
                "substitutes": dict(self.substitutes),
                "time_ratio": handle.sizes() if handle is not None else list(TIME_RATIO_DEFAULT)}

    def set_state(self, state: object) -> None:
        if not isinstance(state, dict):
            return
        valid_specs = {f"{fam}:{code}" for fam, code, _ in self.catalog()}
        for slot in list(self.substitutes):
            self._restore_native(slot)
        substitutes = state.get("substitutes", {}) or {}
        if isinstance(substitutes, dict):
            for slot, spec in substitutes.items():
                if slot in SLOT_KEYS and spec in valid_specs and NATIVE_FAMILY.get(slot) != str(spec).split(":")[0]:
                    self.set_spec(slot, spec, emit=False)
        removed = state.get("removed", []) or []
        self.removed = {slot for slot in removed if slot in SLOT_KEYS} if isinstance(removed, list) else set()
        if len(self.removed) >= len(SLOT_KEYS):
            self.removed.clear()
        ratio = state.get("time_ratio")
        handle = getattr(self.panel, "psth_time_handle", None)
        if handle is not None and isinstance(ratio, list) and len(ratio) == 2:
            try:
                if all(0 < int(v) <= 1000000 for v in ratio):
                    handle.setSizes([int(v) for v in ratio])
            except (TypeError, ValueError):
                pass
        self.apply()
        self._sync_heat_controls()

    # -- visibility post-pass -------------------------------------------------
    def note_layout(self, slot: str, allowed: bool) -> None:
        self._layout_allows[slot] = bool(allowed)

    def capture_layout(self) -> None:
        """Record what the View layout preset shows, before removals apply.

        ``_apply_view_layout`` sets every native's visibility (even a native
        parked in the holder), so explicit hidden flags reflect the preset.
        """
        for slot in SLOT_KEYS:
            if slot == "comparison":
                continue   # owned by the metric layout pass
            self._layout_allows[slot] = not self.native(slot).isHidden()
        self._row_avg_layout = not self.panel.row_avg.isHidden()

    def slot_visible(self, slot: str) -> bool:
        return self._layout_allows.get(slot, True) and slot not in self.removed

    def apply(self) -> None:
        """Hide removed panels and collapse containers so neighbors take the space."""
        if self._applying:
            return
        self._applying = True
        p = self.panel
        try:
            vis = {slot: self.slot_visible(slot) for slot in SLOT_KEYS}
            for slot in SLOT_KEYS:
                widget = self.displayed(slot)
                if widget is self.native(slot) and slot in ("comparison", "global", "trace"):
                    # Natives inside cards: the card carries the visibility.
                    widget.setVisible(self._layout_allows.get(slot, True))
                else:
                    widget.setVisible(vis[slot])
            heat_native = vis["heatmap"] and "heatmap" not in self.substitutes
            p.psth_scale_container.setVisible(heat_native)
            psth_card = vis["heatmap"] or vis["psth"]
            p.psth_shared_card.setVisible(psth_card)
            bout_card = vis["bout1"] or vis["bout2"]
            p.bout_shared_card.setVisible(bout_card)
            cards = p._plot_card_by_widget
            cmp_card, glob_card = cards.get(p.plot_metrics), cards.get(p.plot_global)
            if cmp_card is not None:
                cmp_card.setVisible(vis["comparison"])
            if glob_card is not None:
                glob_card.setVisible(vis["global"])
            extras = not p.metric_panels_widget.isHidden()
            row_avg = vis["comparison"] or vis["global"] or extras
            p.row_avg.setVisible(self._row_avg_layout and row_avg)
            side = bout_card or not p.row_avg.isHidden()
            p.dashboard_side.setVisible(side)
            p.row_heat.setVisible(psth_card or side)
            p.trace_card.setVisible(vis["trace"])
            handle = getattr(p, "psth_time_handle", None)
            if handle is not None:
                handle.refresh()
            controls = getattr(p, "heatmap_color_controls", None)
            if controls is not None and not (heat_native and not p.plot_heat.isHidden()):
                # Hide only: showing is owned by the controls' own sync(), which
                # also respects temporary export views.
                controls.setVisible(False)
        finally:
            self._applying = False

    def _sync_heat_controls(self) -> None:
        controls = getattr(self.panel, "heatmap_color_controls", None)
        if controls is not None:
            controls.sync()

    # -- substitutes ---------------------------------------------------------
    def _substitute_widget(self, slot: str, family: str) -> QtWidgets.QWidget:
        """One reusable substitute per slot and family."""
        key = f"{slot}:{family}"
        widget = self._widgets.get(key)
        if widget is None:
            if family == "metric":
                from psth_metric_panels import MetricPanel
                widget = MetricPanel()
            else:
                widget = pg.PlotWidget()
            self.panel._prepare_substitute_plot(widget)
            self.hook_plot(widget, slot)
            self._widgets[key] = widget
        # The currently displayed substitute of this slot is tracked by slot too.
        self._widgets[slot] = widget
        return widget

    def render_substitutes(self, families: Optional[Sequence[str]] = None) -> None:
        """Redraw substitute panels (optionally only those of some families)."""
        for slot, spec in list(self.substitutes.items()):
            widget = self._widgets.get(slot)
            if widget is None:
                continue
            family, code = spec.split(":", 1)
            if families is not None and family not in families:
                continue
            try:
                self.panel._render_substitute(widget, family, code)
            except Exception as exc:  # a display problem must not break analysis
                widget.setToolTip(f"Could not draw this panel: {exc}")

    # -- context menu --------------------------------------------------------
    def hook_plot(self, plot: QtWidgets.QWidget, slot: str) -> None:
        if not isinstance(plot, pg.PlotWidget) or id(plot) in self._hooked:
            return
        menu = plot.getPlotItem().getViewBox().menu
        if menu is None:
            return
        self._hooked.add(id(plot))
        menu._pyber_panel_actions = []
        menu.aboutToShow.connect(lambda m=menu, s=slot: self._inject_menu(m, s))

    def _inject_menu(self, menu: QtWidgets.QMenu, slot: str) -> None:
        for action in getattr(menu, "_pyber_panel_actions", []):
            menu.removeAction(action)
        first = menu.actions()[0] if menu.actions() else None
        actions = []

        def add(action: QtGui.QAction) -> None:
            menu.insertAction(first, action)
            actions.append(action)

        title = QtGui.QAction(f"Panel: {self.label_for(slot)}", menu)
        title.setEnabled(False)
        add(title)
        remove = QtGui.QAction("Remove this panel", menu)
        remove.setToolTip("Hide this panel; the neighboring panel takes its space.")
        remove.triggered.connect(lambda: self.remove(slot))
        remove.setEnabled(len([k for k in SLOT_KEYS if k not in self.removed]) > 1)
        add(remove)
        change = QtWidgets.QMenu("Change to", menu)
        current = self.current_spec(slot)
        for family in ("behavior", "metric", "global"):
            sub = change.addMenu(FAMILY_LABELS[family])
            for fam, code, label in self.catalog():
                if fam != family:
                    continue
                action = sub.addAction(label)
                action.setCheckable(True)
                action.setChecked(current == f"{fam}:{code}")
                action.triggered.connect(lambda _=False, s=slot, spec=f"{fam}:{code}": self.set_spec(s, spec))
        if slot in self.substitutes:
            change.addSeparator()
            change.addAction(f"Original: {SLOT_LABELS[slot]}", lambda: self.restore_original(slot))
        change_action = change.menuAction()
        change_action.setParent(menu)
        menu._pyber_change_menu = change
        add(change_action)
        restore = QtWidgets.QMenu("Restore removed panels", menu)
        for key in SLOT_KEYS:
            if key in self.removed:
                restore.addAction(SLOT_LABELS[key], lambda k=key: self.restore(k))
        if self.removed:
            restore.addSeparator()
            restore.addAction("Restore all", self.restore_all)
        restore.setEnabled(bool(self.removed))
        menu._pyber_restore_menu = restore
        add(restore.menuAction())
        reset = QtGui.QAction("Reset dashboard layout", menu)
        reset.triggered.connect(self.reset)
        add(reset)
        separator = QtGui.QAction(menu)
        separator.setSeparator(True)
        add(separator)
        menu._pyber_panel_actions = actions
