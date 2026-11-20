"""PCA state-space trajectories: numerics against known answers, then the dashboard panel."""
import unittest

import numpy as np

import test_group_view_dashboard as dashboard
from pca_trajectories import (
    METHOD_DELAY,
    METHOD_RECORDINGS,
    POOLED,
    SPEC,
    delay_embed,
    delay_sample_count,
    delay_trajectories,
    fit_pca,
    population_trajectories,
)

N = dashboard.N_PER_GROUP


class PcaNumericsTests(unittest.TestCase):
    def test_pca_recovers_a_known_axis_and_explained_variance(self):
        rng = np.random.default_rng(0)
        latent = rng.normal(0, 3.0, 500)
        noise = rng.normal(0, 0.1, (500, 3))
        direction = np.array([2.0, 1.0, -2.0]) / 3.0
        samples = latent[:, None] * direction + noise + np.array([10.0, -4.0, 1.0])
        mean, components, ratio = fit_pca(samples)
        np.testing.assert_allclose(mean, samples.mean(axis=0))
        self.assertGreater(abs(float(components[0] @ direction)), 0.999)
        self.assertGreater(ratio[0], 0.99)
        self.assertGreater(components[0][np.argmax(np.abs(components[0]))], 0)   # fixed sign
        np.testing.assert_allclose(components @ components.T, np.eye(3), atol=1e-10)
        # Explained ratios equal the variances of the scores.
        scores = (samples - mean) @ components.T
        np.testing.assert_allclose(scores.var(axis=0) / scores.var(axis=0).sum(), ratio, rtol=1e-9)
        with self.assertRaises(ValueError):
            fit_pca(np.array([[1.0, np.nan], [2.0, 3.0]]))

    def test_delay_embedding_rows_centers_and_missing_samples(self):
        trace = np.arange(10, dtype=float)
        vectors, centers = delay_embed(trace, 3)
        self.assertEqual(vectors.shape, (8, 3))
        np.testing.assert_array_equal(vectors[0], [0, 1, 2])
        np.testing.assert_array_equal(centers, np.arange(1, 9))
        trace[5] = np.nan
        vectors, centers = delay_embed(trace, 3)
        self.assertEqual(vectors.shape[0], 5)               # the three windows touching sample 5 are dropped
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_array_equal(centers, [1, 2, 3, 7, 8])
        self.assertEqual(delay_embed(np.arange(2.0), 3)[0].shape[0], 0)
        time = np.arange(0, 10, 0.1)
        self.assertEqual(delay_sample_count(1.0, time), 11)
        self.assertEqual(delay_sample_count(0.01, time), 3)   # never below the minimum
        self.assertEqual(delay_sample_count(100.0, time), 50) # never above half of the trace

    def test_delay_trajectory_of_a_sine_is_a_circle_shared_by_all_conditions(self):
        # 21 s with a 1 s window: the embedded points cover ten whole 2 s periods.
        time = np.arange(0, 21, 0.05)
        traces = {"sine": (time, np.sin(2 * np.pi * 0.5 * time)),
                  "double": (time, 2 * np.sin(2 * np.pi * 0.5 * time)),
                  "flat": (time, np.zeros_like(time))}
        trajectories, ratio = delay_trajectories(traces, span_s=1.0)
        self.assertEqual(set(trajectories), {"sine", "double", "flat"})
        self.assertGreater(ratio[:2].sum(), 0.999)                      # a sine lives in a plane
        t, scores = trajectories["sine"]
        # A closed ellipse in PC1-PC2: constant radius once each axis is scaled by its spread.
        spread = scores[:, :2].std(axis=0)
        radius = np.hypot(scores[:, 0] / spread[0], scores[:, 1] / spread[1])
        self.assertLess(radius.std() / radius.mean(), 0.02)
        # One shared space: twice the amplitude gives the same shape, twice as large.
        np.testing.assert_allclose(trajectories["double"][1], 2 * scores, atol=1e-9)
        np.testing.assert_allclose(trajectories["flat"][1], 0.0, atol=1e-9)
        self.assertAlmostEqual(float(t[0]), 0.5, places=6)              # point time = center of its window

    def test_population_trajectories_use_recordings_as_dimensions(self):
        time = np.linspace(-1, 1, 41)
        ramp = np.clip(time, 0, None)
        loadings = np.array([1.0, 2.0, -1.0, 0.5])
        responses = {"go": (time, ramp[:, None] * loadings), "stay": (time, 0.0 * ramp[:, None] * loadings)}
        trajectories, ratio = population_trajectories(responses)
        self.assertGreater(ratio[0], 0.999)                             # one shared response pattern
        _t, go = trajectories["go"]
        _t, stay = trajectories["stay"]
        self.assertGreater(np.ptp(go[:, 0]), 10 * max(np.ptp(stay[:, 0]), 1e-12))
        np.testing.assert_allclose(np.ptp(go[:, 0]), np.linalg.norm(loadings), rtol=1e-9)
        responses["go"][1][3, 1] = np.nan                               # a time point missing in one recording
        trajectories, _ratio = population_trajectories(responses)
        self.assertEqual(trajectories["go"][0].size, time.size - 1)


class PcaPanelTests(dashboard.GroupViewDashboardTests):
    """Six synthetic recordings: WT responds up, Het responds down, after each event."""

    def _panel(self):
        self.load()
        panel = self.panel
        panel._settings.remove("pca_trajectories_options")
        panel.pca_trajectories.options.update({"behaviors": [], "groups": None, "method": METHOD_DELAY,
                                               "span_s": 1.0, "axes": [0, 1]})
        panel.cb_group_view.setChecked(True)
        panel.dashboard_panels.set_spec("global", SPEC)
        self.app.processEvents()
        return panel, panel.dashboard_panels.displayed("global")

    def test_panel_replaces_a_metric_panel_and_separates_opposite_groups(self):
        panel, widget = self._panel()
        self.assertEqual(panel.dashboard_panels.current_spec("global"), SPEC)
        self.assertEqual(panel.dashboard_panels.label_for("global"), "PCA trajectories")
        self.assertIn(("trajectory", "pca", "PCA trajectories"), panel.dashboard_panels.catalog())
        report = panel.pca_trajectories.last_report
        self.assertEqual(report["method"], METHOD_DELAY)
        curves = {curve["group"]: curve for curve in report["curves"]}
        self.assertEqual(sorted(curves), ["Het", "WT"])
        self.assertEqual([curve["n"] for curve in curves.values()], [N, N])
        # Opposite responses move in opposite directions along PC1 after the event.
        def displacement(curve):
            after = curve["time"] > 0.5
            before = curve["time"] < -1.5
            return float(curve["scores"][after, 0].mean() - curve["scores"][before, 0].mean())
        self.assertLess(displacement(curves["WT"]) * displacement(curves["Het"]), 0)
        self.assertIn("PC1", widget.getPlotItem().getAxis("bottom").labelText)
        self.assertIn("% of variance", widget.getPlotItem().getAxis("left").labelText)
        self.assertTrue(widget.property("hasPlotData"))

    def test_right_click_options_choose_groups_method_and_axes(self):
        panel, widget = self._panel()
        view = panel.pca_trajectories
        menu = widget.getPlotItem().getViewBox().menu
        panel.dashboard_panels._inject_menu(menu, "global")
        titles = [action.text() for action in menu.actions()]
        self.assertIn("PCA trajectory options", titles)
        options = {action.text(): action.menu() for action in menu._pyber_trajectory_menu.actions()}
        self.assertEqual(list(options), ["Behaviors", "Groups", "Method", "Delay window", "Axes"])
        group_actions = {action.text(): action for action in options["Groups"].actions() if action.text()}
        self.assertTrue(group_actions["Automatic (groups shown by Hide/show groups)"].isChecked())
        self.assertIn(f"WT ({N})", group_actions)

        group_actions[f"WT ({N})"].trigger()                     # only WT
        self.app.processEvents()
        self.assertEqual([curve["group"] for curve in view.last_report["curves"]], ["WT"])
        view.set_option("groups", [POOLED])                      # everything pooled
        self.assertEqual([curve["n"] for curve in view.last_report["curves"]], [2 * N])
        view.set_option("groups", None)
        panel.set_group_hidden(self.het, True)                   # automatic follows Hide/show groups
        self.app.processEvents()
        self.assertEqual([curve["group"] for curve in view.last_report["curves"]], ["WT"])
        panel.show_all_groups()

        view.set_option("method", METHOD_RECORDINGS)
        report = view.last_report
        self.assertEqual(len(report["recordings"]), 2 * N)
        self.assertEqual(len(report["curves"]), 1)
        self.assertGreater(report["ratio"][0], 0.8)              # WT up / Het down is the main axis
        view.set_option("axes", [1, 2])
        self.assertIn("PC2", widget.getPlotItem().getAxis("bottom").labelText)
        self.assertIn("PC3", widget.getPlotItem().getAxis("left").labelText)

        stored = panel._settings.value("pca_trajectories_options")
        self.assertIn(METHOD_RECORDINGS, str(stored))

    def test_empty_state_keeps_the_panel_reachable(self):
        panel, widget = self._panel()
        panel._processed = []
        panel.pca_trajectories.render(widget)
        self.assertEqual(panel.pca_trajectories.last_report["curves"], [])
        self.assertTrue(widget.getPlotItem().isVisible())
        self.assertIn("Load recordings", widget.toolTip())


for _name in [name for name in dir(dashboard.GroupViewDashboardTests) if name.startswith("test_")]:
    setattr(PcaPanelTests, _name, None)


if __name__ == "__main__":
    unittest.main()
