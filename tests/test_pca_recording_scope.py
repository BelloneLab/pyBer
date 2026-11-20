"""Check PCA fits and companion panels when the toolbar scope changes."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtWidgets

import test_group_view_dashboard as dashboard
from pca_trajectories import (
    CODE_LOADINGS, DEFAULT_METHOD, FAMILY, METHOD_BOUT, METHOD_LAG, METHOD_RECORDINGS,
    METHOD_TRIALS, METHODS, POOLED, SETTINGS_KEY, SPEC, SPEC_LOADINGS, decimate_rows, fit_pca, fit_trial_clusters,
)


class PcaRecordingScopeTests(unittest.TestCase):
    """Use independent preference files and six known recordings from the dashboard fixture."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = dashboard.GroupViewDashboardTests.setUp
    tearDown = dashboard.GroupViewDashboardTests.tearDown
    load = dashboard.GroupViewDashboardTests.load

    def test_toolbar_switch_refits_trials_and_loadings_and_restores_group_filter(self):
        self.load()
        panel = self.panel
        view = panel.pca_trajectories
        view.set_option("method", METHOD_TRIALS)
        view.set_option("groups", [self.het])
        panel.dashboard_panels.set_spec("global", SPEC)
        panel.dashboard_panels.set_spec("bout2", SPEC_LOADINGS)
        loadings = panel.dashboard_panels.displayed("bout2")
        panel.tab_visual_mode.setCurrentIndex(0)
        for fid in ("wt0", "het1"):
            with self.subTest(recording=fid):
                panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText(fid))
                report = view.last_report
                self.assertEqual(report["individual_recording"], fid)
                self.assertEqual(len(report["curves"]), 1)
                curve = report["curves"][0]
                self.assertIn(fid, curve["label"])
                self.assertEqual(curve["unit_files"].tolist(), [fid] * dashboard.EVENTS)
                # Derive the independent reference from the displayed file's PSTH matrix.
                times, rows = panel._per_file_mats[fid]
                rows, times = decimate_rows(rows, times)
                mean, components, ratio = fit_pca(rows)
                np.testing.assert_allclose(report["ratio"], ratio)
                np.testing.assert_allclose(curve["units"], (rows - mean) @ components.T)
                for line, weights in zip(loadings.listDataItems(), components):
                    x, y = line.getData()
                    np.testing.assert_allclose(x, times)
                    np.testing.assert_allclose(y, weights)
                self.assertEqual(len(loadings.listDataItems()), len(components))
        panel.tab_visual_mode.setCurrentIndex(1)
        grouped = view.last_report
        self.assertIsNone(grouped["individual_recording"])
        self.assertEqual([(curve["group"], curve["n_recordings"]) for curve in grouped["curves"]],
                         [("Het", dashboard.N_PER_GROUP)])

    def test_individual_scope_precedes_event_extraction_for_all_trajectory_methods(self):
        self.load()
        panel = self.panel
        view = panel.pca_trajectories
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText("wt1"))
        with patch.object(panel, "_get_events_for_proc", wraps=panel._get_events_for_proc) as events:
            for method in (METHOD_LAG, METHOD_BOUT, METHOD_TRIALS):
                with self.subTest(method=method):
                    view.options["method"] = method
                    report = view._compute()
                    self.assertEqual(report["method"], method)
                    self.assertEqual(report["curves"][0]["unit_files"].tolist(), ["wt1"] * dashboard.EVENTS)
                    self.assertEqual([panel._file_id_for_proc(call.args[0]) for call in events.call_args_list],
                                     ["wt1"])
                    events.reset_mock()
        # If this file loses its events, the bout-to-lag fallback must stay empty.
        view.options["method"] = METHOD_BOUT
        with patch.object(panel, "_get_events_for_proc", return_value=([], [])) as missing:
            empty = view._compute()
        self.assertEqual(empty["curves"], [])
        self.assertEqual(empty["individual_recording"], "wt1")
        self.assertEqual([panel._file_id_for_proc(call.args[0]) for call in missing.call_args_list], ["wt1", "wt1"])
        menu = view.build_menu(panel)
        top = {action.text(): action for action in menu.actions()}
        self.assertNotIn("Groups", top)
        self.assertIn("Recording · wt1", top)
        method_actions = {action.text(): action for action in top["Method"].menu().actions()}
        self.assertFalse(method_actions[dict(METHODS)[METHOD_RECORDINGS]].isEnabled())

    def test_recordings_as_dimensions_adapts_to_trials_and_returns_to_group_method(self):
        self.load()
        panel = self.panel
        view = panel.pca_trajectories
        view.set_option("method", METHOD_RECORDINGS)
        panel.dashboard_panels.set_spec("global", SPEC_LOADINGS)
        widget = panel.dashboard_panels.displayed("global")
        self.assertEqual(view.last_report["method"], METHOD_RECORDINGS)
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText("wt2"))
        individual = view.last_report
        self.assertEqual((individual["method"], individual["requested_method"]), (METHOD_TRIALS, METHOD_RECORDINGS))
        self.assertEqual(individual["curves"][0]["unit_files"].tolist(), ["wt2"] * dashboard.EVENTS)
        self.assertIn("Individual view uses trial-response PCA", widget.toolTip())
        menu = view.build_menu(panel)
        method_menu = next(action.menu() for action in menu.actions() if action.text() == "Method")
        checked = [action.text() for action in method_menu.actions() if action.isChecked()]
        self.assertEqual(checked, [dict(METHODS)[METHOD_TRIALS] + " (Individual view)"])
        with patch.object(panel, "_get_events_for_proc", return_value=([15., 20.], [1., 1.])):
            view.refresh()
            self.assertIn("three complete trials", widget.toolTip())
        panel.tab_visual_mode.setCurrentIndex(1)
        self.assertEqual(view.last_report["method"], METHOD_RECORDINGS)
        self.assertEqual(len(view.last_report["recordings"]), 2 * dashboard.N_PER_GROUP)
        self.assertEqual(json.loads(str(panel._settings.value(SETTINGS_KEY)))["method"], METHOD_RECORDINGS)

    def test_display_options_reuse_fits_and_real_data_changes_refit(self):
        self.load()
        panel = self.panel
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText("wt0"))
        view = panel.pca_trajectories
        view.set_option("method", METHOD_TRIALS)
        panel.dashboard_panels.set_spec("global", SPEC)
        with patch.object(view, "_compute", wraps=view._compute) as pca, \
                patch("pca_trajectories.fit_trial_clusters", wraps=fit_trial_clusters) as mixture:
            view.refresh()
            self.assertEqual((pca.call_count, mixture.call_count), (1, 1))
            first = view.last_report
            for key, value in (("axes", [1, 2]), ("units", False), ("trial_clusters", False),
                               ("trial_clusters", True), ("show_lag_weights", True), ("group_loadings", False)):
                view.set_option(key, value)
                self.assertIs(view.last_report, first)
            self.assertEqual((pca.call_count, mixture.call_count), (1, 1))
            panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText("het1"))
            self.assertGreaterEqual(pca.call_count, 2)
            self.assertGreaterEqual(mixture.call_count, 2)
            self.assertEqual(view.last_report["individual_recording"], "het1")
            self.assertIsNot(view.last_report, first)

    def test_event_column_cache_uses_recording_picker_even_if_entry_is_reused(self):
        self.load()
        panel = self.panel
        view = panel.pca_trajectories
        view.options["method"] = METHOD_TRIALS
        panel.tab_visual_mode.setCurrentIndex(0)
        panel.dashboard_panels.set_spec("global", SPEC_LOADINGS)
        widget = panel.dashboard_panels.displayed("global")
        entry = {"name": view.selected_behaviors()[0]}
        for fid in ("wt2", "het2"):
            panel.combo_individual_file.setCurrentIndex(panel.combo_individual_file.findText(fid))
            panel.dashboard_panels._draw_column_substitute(widget, entry, FAMILY, CODE_LOADINGS)
            report = entry["_column_pca"][1]
            self.assertEqual(report["individual_recording"], fid)
            np.testing.assert_array_equal(report["curves"][0]["unit_files"], [fid] * dashboard.EVENTS)
            view.set_option("axes", [0, 2])
            panel.dashboard_panels._draw_column_substitute(widget, entry, FAMILY, CODE_LOADINGS)
            self.assertIs(entry["_column_pca"][1], report)
        # A real data refresh invalidates even a retained column with the same file ID.
        with patch.object(panel, "_get_events_for_proc", return_value=([], [])):
            view.refresh()
            panel.dashboard_panels._draw_column_substitute(widget, entry, FAMILY, CODE_LOADINGS)
            self.assertFalse(entry["_column_pca"][1]["curves"])

    def test_pooled_and_named_group_selections_never_duplicate_trials(self):
        self.load()
        panel = self.panel
        view = panel.pca_trajectories
        panel._settings.setValue(SETTINGS_KEY, json.dumps({"method": DEFAULT_METHOD, "groups": [self.wt, POOLED]}))
        view._load_options()
        self.assertEqual(view.options["groups"], [POOLED])
        view._toggle_group(self.het, True)
        self.assertEqual(view.options["groups"], [self.het])
        view._toggle_group(POOLED, True)
        self.assertEqual(view.options["groups"], [POOLED])
        report = view.compute()
        self.assertEqual(len(report["curves"]), 1)
        self.assertEqual(report["curves"][0]["n"], 2 * dashboard.N_PER_GROUP * dashboard.EVENTS)


if __name__ == "__main__":
    unittest.main()
