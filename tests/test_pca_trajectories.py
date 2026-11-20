"""PCA state-space panels: numerics against known answers, then the dashboard panels."""
import unittest
from unittest.mock import patch

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui

import test_group_view_dashboard as dashboard
from pca_trajectories import (
    BOUT_PHASE_POINTS,
    DEFAULT_METHOD,
    METHOD_BOUT,
    METHOD_LAG,
    METHOD_RECORDINGS,
    METHOD_TRIALS,
    LINE_STYLES,
    PC_ALPHA,
    POOLED,
    SPEC,
    SPEC_LOADINGS,
    SPEC_SEPARATION,
    SPEC_CLUSTER_SEPARATION,
    bouts_to_phase,
    decimate_rows,
    delay_embed,
    delay_sample_count,
    delay_trajectories,
    embed_units,
    fit_pca,
    permutation_separation,
    population_trajectories,
    separation_statistic,
)

N = dashboard.N_PER_GROUP
EVENTS = dashboard.EVENTS


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
        np.testing.assert_array_equal(centers, [1, 2, 3, 7, 8])
        time = np.arange(0, 10, 0.1)
        self.assertEqual(delay_sample_count(1.0, time), 11)
        self.assertEqual(delay_sample_count(0.01, time), 3)   # never below the minimum
        self.assertEqual(delay_sample_count(100.0, time), 50) # never above half of the trace

    def test_delay_trajectory_of_a_sine_is_an_ellipse_shared_by_all_conditions(self):
        # 21 s with a 1 s window: the embedded points cover ten whole 2 s periods.
        time = np.arange(0, 21, 0.05)
        traces = {"sine": (time, np.sin(2 * np.pi * 0.5 * time)),
                  "double": (time, 2 * np.sin(2 * np.pi * 0.5 * time)),
                  "flat": (time, np.zeros_like(time))}
        trajectories, ratio = delay_trajectories(traces, span_s=1.0)
        self.assertGreater(ratio[:2].sum(), 0.999)                      # a sine lives in a plane
        _t, scores = trajectories["sine"]
        spread = scores[:, :2].std(axis=0)
        radius = np.hypot(scores[:, 0] / spread[0], scores[:, 1] / spread[1])
        self.assertLess(radius.std() / radius.mean(), 0.02)
        np.testing.assert_allclose(trajectories["double"][1], 2 * scores, atol=1e-9)   # one shared space
        np.testing.assert_allclose(trajectories["flat"][1], 0.0, atol=1e-9)

    def test_embed_units_matches_the_single_trace_embedding_and_keeps_gaps(self):
        time = np.arange(0, 21, 0.05)
        sine = np.sin(2 * np.pi * 0.5 * time)
        units = np.vstack([sine, 2 * sine, 0 * sine])
        m = delay_sample_count(1.0, time)
        scores, centers, components, ratio = embed_units(units, m)
        self.assertEqual(scores.shape, (3, time.size - m + 1, 3))
        reference, _ratio = delay_trajectories({"a": (time, sine), "b": (time, 2 * sine), "c": (time, 0 * sine)}, 1.0)
        np.testing.assert_allclose(scores[0], reference["a"][1], atol=1e-9)
        np.testing.assert_allclose(time[centers], reference["a"][0])
        self.assertEqual(components.shape, (3, m))
        self.assertGreater(ratio[:2].sum(), 0.999)
        units[1, 100] = np.nan
        scores, *_ = embed_units(units, m)
        self.assertEqual(int(np.isnan(scores[1, :, 0]).sum()), m)       # only windows touching the gap
        self.assertFalse(np.isnan(scores[0]).any())

    def test_loadings_of_a_slow_signal_are_level_then_slope(self):
        # A slow random walk: PC1 weights the whole window with one sign (local level),
        # PC2 changes sign once across the window (local slope).
        rng = np.random.default_rng(2)
        walk = np.cumsum(rng.normal(0, 1, (40, 400)), axis=1)
        _scores, _centers, components, ratio = embed_units(walk, 21)
        self.assertEqual(abs(int(np.sign(components[0]).sum())), 21)
        self.assertEqual(int(np.count_nonzero(np.diff(np.sign(components[1])))), 1)
        self.assertGreater(ratio[0], 0.9)

    def test_bouts_are_resampled_from_onset_to_offset(self):
        time = np.arange(-1.0, 6.0, 0.1)
        ramp = np.where(time >= 0, time, 0.0)                     # value = seconds since onset
        matrix = np.vstack([ramp, ramp, ramp, ramp])
        durations = np.array([2.0, 4.0, np.nan, 50.0])            # short, long, no duration, longer than the window
        rows, kept, truncated = bouts_to_phase(matrix, time, durations)
        np.testing.assert_array_equal(kept, [0, 1, 3])
        self.assertEqual((rows.shape, truncated), ((3, BOUT_PHASE_POINTS), 1))
        phase = np.linspace(0, 1, BOUT_PHASE_POINTS)
        np.testing.assert_allclose(rows[0], 2.0 * phase, atol=0.06)     # onset 0 -> offset 2 s
        np.testing.assert_allclose(rows[1], 4.0 * phase, atol=0.06)     # same phase axis, longer bout
        self.assertAlmostEqual(float(rows[2][-1]), float(time[-1]), places=6)   # cut at the window end

    def test_decimation_block_averages_and_keeps_missing_blocks(self):
        time = np.arange(1000, dtype=float)
        matrix = np.vstack([time, np.full(1000, np.nan)])
        reduced, axis = decimate_rows(matrix, time, max_points=200)
        self.assertEqual(reduced.shape, (2, 200))
        np.testing.assert_allclose(reduced[0], axis)                     # mean of a ramp = block center
        self.assertTrue(np.isnan(reduced[1]).all())
        same, _axis = decimate_rows(matrix[:, :100], time[:100], max_points=200)
        self.assertEqual(same.shape[1], 100)

    def test_population_trajectories_use_recordings_as_dimensions(self):
        time = np.linspace(-1, 1, 41)
        ramp = np.clip(time, 0, None)
        loadings = np.array([1.0, 2.0, -1.0, 0.5])
        responses = {"go": (time, ramp[:, None] * loadings), "stay": (time, 0.0 * ramp[:, None] * loadings)}
        trajectories, ratio, components = population_trajectories(responses, return_model=True)
        self.assertGreater(ratio[0], 0.999)
        np.testing.assert_allclose(np.abs(components[0]), np.abs(loadings) / np.linalg.norm(loadings), atol=1e-9)
        np.testing.assert_allclose(np.ptp(trajectories["go"][1][:, 0]), np.linalg.norm(loadings), rtol=1e-9)

    def test_permutation_test_detects_a_real_separation_and_not_a_null_one(self):
        rng = np.random.default_rng(4)

        def noise(count):
            return rng.normal(0, 1.0, (count, 30, 2))
        apart_a, apart_b = noise(12) + 1.5, noise(12)
        result = permutation_separation(apart_a, apart_b)
        self.assertAlmostEqual(result["observed"], separation_statistic(apart_a, apart_b))
        self.assertFalse(result["exact"])
        self.assertLess(result["p"], 0.01)
        self.assertEqual(result["null"].size, result["n_permutations"])
        self.assertGreater(permutation_separation(noise(12), noise(12))["p"], 0.05)
        self.assertEqual(permutation_separation(apart_a, apart_b)["p"], result["p"])   # fixed seed

    def test_small_samples_are_enumerated_exactly(self):
        a = np.array([[[3.0]], [[4.0]], [[5.0]]])
        b = np.array([[[0.0]], [[1.0]], [[2.0]]])
        result = permutation_separation(a, b)
        self.assertTrue(result["exact"])
        self.assertEqual(result["n_permutations"], 20)                  # C(6, 3) labelings
        self.assertAlmostEqual(result["p"], 2 / 20)                     # this split and its mirror image
        one_each = permutation_separation(a[:1], b[:1])
        self.assertEqual((one_each["n_permutations"], one_each["p"]), (2, 1.0))   # nothing can be concluded


class PcaPanelTests(dashboard.GroupViewDashboardTests):
    """Six synthetic recordings: WT responds up, Het down, for 2 s after each event (bouts of 1 s)."""

    def _panel(self, method=METHOD_LAG, slot="global", spec=SPEC):
        self.load()
        panel = self.panel
        panel._settings.remove("pca_trajectories_options")
        panel.pca_trajectories.options.update({"behaviors": [], "groups": None, "method": method,
                                               "span_s": 1.0, "axes": [0, 1], "units": True})
        panel.pca_trajectories._cache = None
        panel.cb_group_view.setChecked(True)
        panel.dashboard_panels.set_spec(slot, spec)
        self.app.processEvents()
        return panel, panel.dashboard_panels.displayed(slot)

    def test_pca_panels_are_offered_and_lag_trajectories_separate_opposite_groups(self):
        panel, widget = self._panel(METHOD_LAG)
        catalog = [(code, label) for family, code, label in panel.dashboard_panels.catalog() if family == "trajectory"]
        self.assertEqual(catalog, [("pca", "PCA trajectories"), ("loadings", "PCA loadings"),
                                   ("separation", "PCA separation test"),
                                   ("cluster_separation", "Cluster separation")])
        self.assertEqual(panel.dashboard_panels.label_for("global"), "PCA trajectories")
        report = panel.pca_trajectories.last_report
        self.assertEqual(report["method"], METHOD_LAG)
        curves = {curve["group"]: curve for curve in report["curves"]}
        self.assertEqual(sorted(curves), ["Het", "WT"])
        for curve in curves.values():
            self.assertEqual((curve["n"], curve["n_recordings"]), (N * EVENTS, N))   # every trial is a unit
            self.assertEqual(curve["units"].shape[0], N * EVENTS)
            self.assertEqual(curve["units"].shape[1:], curve["scores"].shape)

        def displacement(curve):
            after, before = curve["time"] > 0.5, curve["time"] < -1.5
            return float(curve["scores"][after, 0].mean() - curve["scores"][before, 0].mean())
        self.assertLess(displacement(curves["WT"]) * displacement(curves["Het"]), 0)
        self.assertIn("PC1", widget.getPlotItem().getAxis("bottom").labelText)
        self.assertTrue(widget.property("hasPlotData"))

    def test_onset_to_offset_is_the_default_and_uses_each_bout(self):
        self.assertEqual(DEFAULT_METHOD, METHOD_BOUT)
        panel, widget = self._panel(METHOD_BOUT)
        report = panel.pca_trajectories.last_report
        self.assertEqual((report["method"], report["fallback"]), (METHOD_BOUT, ""))
        curve = report["curves"][0]
        self.assertEqual(curve["n"], N * EVENTS)
        # Phase axis from onset (0) to offset (1), shortened by the delay window at both ends.
        self.assertGreaterEqual(curve["time"][0], 0.0)
        self.assertLessEqual(curve["time"][-1], 1.0)
        self.assertIn("% of bout", report["loading_label"])
        self.assertIn("onset-to-offset", widget.getPlotItem().titleLabel.text)

        # Events without a duration: the lag-embedded view is shown and says why.
        events = {p.path: np.linspace(15, 100, EVENTS) for p in panel._processed}
        panel._get_events_for_proc = lambda proc: (events[proc.path], np.full(EVENTS, np.nan))
        panel.pca_trajectories.refresh()
        report = panel.pca_trajectories.last_report
        self.assertEqual((report["method"], report["requested_method"]), (METHOD_LAG, METHOD_BOUT))
        self.assertIn("lag-embedded", report["fallback"])
        self.assertIn("lag-embedded", widget.toolTip())
        panel.dashboard_panels.set_spec("bout2", SPEC_LOADINGS)
        companion = panel.dashboard_panels.displayed("bout2")
        self.assertEqual(companion.getAxis("left").labelText, "Mean PC score")
        self.assertIn("No bout", companion.toolTip())

    def test_trial_response_pca_has_one_point_per_event(self):
        panel, _widget = self._panel(METHOD_TRIALS)
        report = panel.pca_trajectories.last_report
        self.assertEqual(report["method"], METHOD_TRIALS)
        curves = {curve["group"]: curve for curve in report["curves"]}
        for curve in curves.values():
            self.assertEqual(curve["units"].shape, (N * EVENTS, 3))
        # Up versus down responses are separated along PC1.
        gap = abs(float(np.nanmean(curves["WT"]["units"][:, 0]) - np.nanmean(curves["Het"]["units"][:, 0])))
        self.assertGreater(gap, 4 * float(np.nanstd(curves["WT"]["units"][:, 0])))
        self.assertEqual(report["loading_label"], "Time from event (s)")
        # The first loading is the response time course: flat before the event, a step after it.
        time, loading = report["loading_axis"], report["components"][0]
        self.assertGreater(abs(loading[(time > 0.2) & (time < 1.8)].mean()), 5 * abs(loading[time < -0.5].mean()))

    def test_cluster_outlines_and_separation_share_the_trial_assignments(self):
        """Reproject one fitted mixture and compare the companion bars with its trial scores."""
        from trial_response_clusters import cluster_ellipse

        panel, scatter = self._panel(METHOD_TRIALS)
        view = panel.pca_trajectories
        first = view.last_report["trial_clusters"]
        self.assertGreaterEqual(first["n_clusters"], 2)
        self.assertIsNone(first["silhouettes"])
        panel.dashboard_panels.set_spec("bout2", SPEC_CLUSTER_SEPARATION)
        separation = panel.dashboard_panels.displayed("bout2")
        view.set_option("axes", [1, 2])
        model = view.last_report["trial_clusters"]
        np.testing.assert_array_equal(first["labels"], model["labels"])
        outlines = [line for line in scatter.listDataItems() if (line.name() or "").startswith("Cluster ")]
        self.assertEqual(len(outlines), model["n_clusters"])
        for k, outline in enumerate(outlines):
            expected = cluster_ellipse(model["means"][k], model["covariances"][k], 1, 2)
            np.testing.assert_allclose(np.column_stack(outline.getData()), expected)

        bars = [item for item in separation.getPlotItem().items if isinstance(item, pg.BarGraphItem)]
        self.assertEqual(len(bars), model["n_clusters"])
        for k, bar in enumerate(bars):
            scores = model["silhouettes"][model["labels"] == k]
            np.testing.assert_allclose(np.sort(bar.opts["height"]), np.sort(scores))
        self.assertIn("Silhouette", separation.getAxis("left").labelText)

        menu = view.build_menu(scatter.getPlotItem().getViewBox().menu)
        toggle = next(action for action in menu.actions() if action.text() == "Show GMM clusters")
        self.assertTrue(toggle.isChecked())
        toggle.trigger()
        self.assertFalse(any((item.name() or "").startswith("Cluster ") for item in scatter.listDataItems()))
        self.assertIn("Silhouette", separation.getAxis("left").labelText)
        view.set_option("method", METHOD_BOUT)
        self.assertIn("Trial-response PCA", separation.toolTip())

    def test_loadings_and_separation_panels_share_the_analysis(self):
        panel, widget = self._panel(METHOD_LAG, slot="bout2", spec=SPEC_LOADINGS)
        view = panel.pca_trajectories
        self.assertEqual(panel.dashboard_panels.label_for("bout2"), "PCA loadings")
        self.assertIn("Event-aligned PC scores", widget.getPlotItem().titleLabel.text)
        self.assertEqual(widget.getAxis("left").labelText, "Mean PC score")
        self.assertIn("Time from", widget.getAxis("bottom").labelText)
        report = view.last_report
        lines = iter(widget.listDataItems())
        for curve in report["curves"]:
            for pc in range(report["ratio"].size):
                line = next(lines)
                times, values = line.getData()
                pen = line.opts["pen"]
                self.assertEqual(pen.color().name(), curve["group_color"])
                self.assertEqual((pen.style(), pen.color().alpha()), (LINE_STYLES[pc], PC_ALPHA[pc]))
                np.testing.assert_array_equal(times, curve["time"])
                np.testing.assert_allclose(values, np.nanmean(curve["units"][:, :, pc], axis=0))
                self.assertLess(times[0], -2)
                self.assertGreater(times[-1], 3)
        self.assertIsNone(next(lines, None))
        markers = [item for item in widget.getPlotItem().items if isinstance(item, pg.InfiniteLine)]
        self.assertEqual([(item.angle, item.value()) for item in markers], [(90, 0)])

        # Each group's loadings must come from its own delay windows.
        menu = view.build_menu(widget.getPlotItem().getViewBox().menu)
        next(action for action in menu.actions() if action.text() == "Show delay-window weights").trigger()
        self.assertEqual(widget.getAxis("left").labelText, "Loading (weight)")
        self.assertEqual(widget.getAxis("bottom").labelText, "Lag inside the delay window (s)")
        self.app.processEvents()
        view.render(widget, "loadings")  # a second draw starts with sub-unit axis ranges
        for side in ("bottom", "left"):
            self.assertEqual(widget.getAxis(side).autoSIPrefixScale, 1.0)
        grouped = view.last_report
        self.assertEqual(len(widget.listDataItems()), 6)
        lines = iter(widget.listDataItems())
        for (group, color, traces), model in zip(grouped["loading_groups"], grouped["group_models"]):
            self.assertEqual(model[0], group)
            windows = np.lib.stride_tricks.sliding_window_view(traces, grouped["loading_delays"], axis=1)
            _, expected, ratio = fit_pca(windows[np.isfinite(windows).all(axis=-1)])
            for pc, weights in enumerate(expected):
                line = next(lines)
                pen = line.opts["pen"]
                self.assertEqual((pen.color().name(), pen.style(), pen.color().alpha()),
                                 (color, LINE_STYLES[pc], PC_ALPHA[pc]))
                x, observed = line.getData()
                np.testing.assert_array_equal(x, report["loading_axis"])
                np.testing.assert_allclose(abs(observed), abs(weights))
                self.assertGreaterEqual(observed @ report["components"][pc], -1e-12)
            np.testing.assert_allclose(model[3], ratio)

        # A shared fit remains available for interpreting the trajectory axes.
        menu = view.build_menu(widget.getPlotItem().getViewBox().menu)
        next(action for action in menu.actions() if action.text() == "Fit weights separately by group").trigger()
        self.assertEqual(len(widget.listDataItems()), 3)
        for line, weights in zip(widget.listDataItems(), report["components"]):
            np.testing.assert_allclose(line.getData()[1], weights)

        panel.dashboard_panels.set_spec("bout2", SPEC_SEPARATION)
        self.app.processEvents()
        widget = panel.dashboard_panels.displayed("bout2")
        result = view.separation(view.last_report)
        self.assertTrue(result["available"])
        self.assertEqual(result["level"], "recordings")             # three recordings per group: animal level
        self.assertEqual((result["n_a"], result["n_b"]), (N, N))
        self.assertTrue(result["exact"])                             # C(6, 3) = 20 labelings
        self.assertAlmostEqual(result["p"], 2 / 20)                  # the most extreme split of 20
        self.assertIn("animal-level", widget.getPlotItem().titleLabel.text)
        self.assertIn("Observed separation", widget.toolTip())

        # One condition only: the panel says what the test needs.
        view.set_option("groups", [self.wt])
        self.assertFalse(view.separation(view.last_report)["available"])
        self.assertIn("two conditions", widget.toolTip())

    def test_trial_level_permutation_title_and_tooltip(self):
        panel, widget = self._panel(METHOD_LAG, slot="bout2", spec=SPEC_SEPARATION)
        view = panel.pca_trajectories
        panel.recording_groups.assign(["wt1", "wt2", "het1", "het2"], None)     # one recording left per group
        view.set_option("groups", [self.wt, self.het])
        result = view.separation(view.last_report)
        self.assertTrue(result["available"])
        self.assertEqual(result["level"], "trials")
        self.assertEqual((result["n_a"], result["n_b"]), (EVENTS, EVENTS))
        self.assertLess(result["p"], 0.01)
        self.assertEqual(
            widget.getPlotItem().titleLabel.text,
            f"Separation · trial-level permutation (p = {result['p']:.3g})",
        )
        self.assertIn("not independent", widget.toolTip())

    def test_multiple_behaviors_keep_custom_group_colors_and_have_distinct_symbols(self):
        panel, widget = self._panel(METHOD_LAG, slot="bout2", spec=SPEC_LOADINGS)
        view = panel.pca_trajectories
        panel.recording_groups.update_group(self.wt, color="#783eb6")
        panel.recording_groups.update_group(self.het, color="#178377")
        original, time, info = view._trial_responses(view.selected_behaviors(), False)
        trials = next(iter(original.values()))
        responses = {"contact": trials, "rearing": {fid: rows * 0.5 for fid, rows in trials.items()}}
        with patch.object(view, "_trial_responses", return_value=(responses, time, info)), \
                patch.object(view, "selected_behaviors", return_value=list(responses)):
            view.refresh()
        curves = view.last_report["curves"]
        self.assertEqual(len(curves), 4)
        lines = [item for item in widget.listDataItems() if isinstance(item, pg.PlotDataItem)]
        symbols = [item.opts["symbol"] for item in widget.getPlotItem().items if isinstance(item, pg.ScatterPlotItem)]
        self.assertEqual(len(lines), 12)
        self.assertEqual(len(symbols), 12)
        for i, curve in enumerate(curves):
            expected = "#783eb6" if curve["group"] == "WT" else "#178377"
            group_lines = lines[3 * i:3 * (i + 1)]
            self.assertEqual([line.opts["pen"].color().name() for line in group_lines], [expected] * 3)
        self.assertEqual(len(set(symbols)), 2)

    def test_group_loadings_skip_flat_or_incomplete_groups_and_unsupported_pcs(self):
        panel, _widget = self._panel(METHOD_TRIALS, slot="bout2", spec=SPEC_LOADINGS)
        view = panel.pca_trajectories
        groups = [("varying", "#1188aa", np.array([[-1., 0.], [1., 0.], [np.nan, 5.]])),
                  ("flat", "#008800", np.ones((3, 2))),
                  ("incomplete", "#993300", np.array([[np.nan, 0.], [2., 1.]]))]
        models = view._group_weights(dict(loading_groups=groups, loading_delays=None, components=np.eye(2)))
        self.assertEqual(len(models), 1)
        name, _, weights, fractions = models[0]
        self.assertEqual(name, "varying")
        np.testing.assert_allclose(weights, [[1., 0.]])
        np.testing.assert_allclose(fractions, [1.])

    def test_right_click_options_and_missing_component_pair(self):
        panel, widget = self._panel(METHOD_LAG)
        view = panel.pca_trajectories
        menu = widget.getPlotItem().getViewBox().menu
        panel.dashboard_panels._inject_menu(menu, "global")
        self.assertIn("PCA options", [action.text() for action in menu.actions()])
        options = {action.text(): action for action in menu._pyber_trajectory_menu.actions()}
        self.assertEqual(list(options), ["Method", "Behaviors", "Groups", "Show single trials / bouts",
                                         "Delay window", "Show delay-window weights",
                                         "Fit weights separately by group", "Axes"])
        methods = [action.text() for action in options["Method"].menu().actions()]
        self.assertEqual(len(methods), 4)
        self.assertTrue(methods[0].startswith("Onset-to-offset"))

        view.set_option("groups", [POOLED])
        self.assertEqual([curve["n"] for curve in view.last_report["curves"]], [2 * N * EVENTS])
        view.set_option("groups", None)
        panel.set_group_hidden(self.het, True)                       # automatic follows Hide/show groups
        self.app.processEvents()
        self.assertEqual([curve["group"] for curve in view.last_report["curves"]], ["WT"])
        panel.show_all_groups()

        # Two recordings as dimensions give two components: PC1 vs PC3 falls back to PC1 vs PC2.
        panel.recording_groups.assign(["wt1", "wt2", "het1", "het2"], None)
        view.set_option("groups", [self.wt, self.het])
        view.set_option("axes", [0, 2])
        view.set_option("method", METHOD_RECORDINGS)
        report = view.last_report
        self.assertEqual((len(report["recordings"]), report["ratio"].size), (2, 2))
        self.assertEqual(view._resolve_axes(report["ratio"].size), (0, 1))
        self.assertIn("PC2", widget.getPlotItem().getAxis("left").labelText)
        self.assertEqual(report["loading_ticks"], report["recordings"])
        self.assertIn(METHOD_RECORDINGS, str(panel._settings.value("pca_trajectories_options")))

    def test_panels_follow_the_plot_background_not_the_preset_surface(self):
        panel, widget = self._panel(METHOD_LAG)
        for preset, background in (("Paper", (12, 15, 22)), ("Midnight", (250, 251, 254))):
            panel._style["postprocessing_preset"] = preset
            panel.plot_trace.setBackground(QtGui.QColor(*background))     # what is on screen decides
            palette = panel._current_palette()
            self.assertEqual(QtGui.QColor(str(palette["surface"])).getRgb()[:3], background)
            dark = sum(background) / 3 < 128
            self.assertEqual(QtGui.QColor(str(palette["text"])).lightness() > 128, dark)   # readable on it
            panel.pca_trajectories.render(widget)
            self.assertEqual(widget.backgroundBrush().color().getRgb()[:3], background)
        panel._style["postprocessing_preset"] = "Midnight"
        panel.plot_trace.setBackground(QtGui.QColor(12, 15, 22))
        self.assertEqual(panel._current_palette()["text"], "#e5eef8")   # a matching preset keeps its own colors

    def test_empty_state_keeps_the_panel_reachable(self):
        panel, widget = self._panel(METHOD_LAG)
        panel._processed = []
        panel.pca_trajectories.refresh()
        self.assertEqual(panel.pca_trajectories.last_report["curves"], [])
        self.assertTrue(widget.getPlotItem().isVisible())
        self.assertIn("Load recordings", widget.toolTip())


for _name in [name for name in dir(dashboard.GroupViewDashboardTests) if name.startswith("test_")]:
    setattr(PcaPanelTests, _name, None)


if __name__ == "__main__":
    unittest.main()
