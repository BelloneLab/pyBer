"""Exercise actual publication rendering, including finite-p significance labels."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

import numpy as np

from gui_postprocessing import PostProcessingPanel


def _publication_panel(*, negative=False, pooled_trials=False):
    """Build analytical trial data and only controls consumed by the exporter."""
    time = np.linspace(-2, 2, 41)
    # Six consistent positive paired differences have a finite exact p-value,
    # forcing the significance-bracket drawing path that previously crashed.
    baseline = np.arange(6, dtype=float)[:, None] * 0.01
    matrix = baseline + np.where(time[None, :] < 0, 0.0, 1.0)
    if negative:
        matrix -= 5.0
    messages = []
    return SimpleNamespace(
        _get_all_behavior_names=lambda: ["Synthetic event"],
        _compute_psth_for_behavior=lambda _name: (matrix, time, [f"Animal {i + 1}" for i in range(6)]),
        _psth_group_trial_view_enabled=lambda: pooled_trials,
        _finite_mean_sem=PostProcessingPanel._finite_mean_sem,
        _psth_units=lambda: "Processed units",
        _style={"heatmap_cmap": "viridis"},
        spin_metric_pre0=SimpleNamespace(value=lambda: -1.0),
        spin_metric_pre1=SimpleNamespace(value=lambda: -0.1),
        spin_metric_post0=SimpleNamespace(value=lambda: 0.1),
        spin_metric_post1=SimpleNamespace(value=lambda: 1.0),
        combo_metric=SimpleNamespace(currentText=lambda: "Mean signal"),
        tab_sources=SimpleNamespace(currentIndex=lambda: 1),
        tab_visual_mode=SimpleNamespace(currentIndex=lambda: 1),
        statusUpdate=SimpleNamespace(emit=lambda *args: messages.append(args)),
        messages=messages,
    )


class PublicationRenderingTests(unittest.TestCase):
    """Verify all three real figure files are produced for meaningful cases."""

    def _assert_exports(self, panel):
        """Export to an isolated temporary folder and check nonempty artifacts."""
        with tempfile.TemporaryDirectory(prefix="pyber_publication_test_") as directory:
            PostProcessingPanel._export_publication_figure(
                panel, directory, "regression", "Heatmap + Avg PSTH + Metrics",
            )
            for extension in ("png", "pdf", "svg"):
                path = Path(directory) / f"regression_publication_figure.{extension}"
                self.assertTrue(path.is_file(), f"Missing {extension} publication output")
                self.assertGreater(path.stat().st_size, 100)
            self.assertTrue(any("Publication figure saved" in str(message) for message in panel.messages))

    def test_finite_sign_test_pvalue_renders_significance_bracket(self):
        self._assert_exports(_publication_panel())

    def test_negative_metric_values_render_without_bracket_failure(self):
        self._assert_exports(_publication_panel(negative=True))

    def test_pooled_trials_export_descriptive_metrics_without_inferential_bracket(self):
        self._assert_exports(_publication_panel(pooled_trials=True))

    def test_export_retains_selected_heatmap_palette_and_fixed_limits(self):
        from matplotlib.axes import Axes
        from heatmap_display import color_map
        panel = _publication_panel()
        panel._style.update(psth_heatmap_cmap='CET-D1', heatmap_levels_manual=True,
                            heatmap_min=-.5, heatmap_max=.5)
        original = Axes.imshow
        captured = []
        def capture(axis, *args, **kwargs):
            captured.append(kwargs)
            return original(axis, *args, **kwargs)
        with patch.object(Axes, 'imshow', capture):
            self._assert_exports(panel)
        self.assertEqual(len(captured), 1)
        self.assertEqual((captured[0]['vmin'], captured[0]['vmax']), (-.5, .5))
        np.testing.assert_array_equal(captured[0]['cmap'](np.linspace(0, 1, 256)),
                                      color_map(panel._style).getLookupTable(nPts=256, alpha=True) / 255.)


if __name__ == "__main__":
    unittest.main()
