"""OR events preserve originals, missing observations and per-recording scope."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore, QtWidgets
import test_postprocessing_empty_state as fixture
from analysis_core import ProcessedTrial
from combined_events import union_states, union_intervals, refresh_combined_labels
from combined_events_dialog import CombineEventsDialog
from postprocessing_core import extract_complete_events, compute_psth_matrix


class UnionNumericalTests(unittest.TestCase):
    def test_three_valued_or_preserves_unknown_and_true_dominates(self):
        a = np.array([0., 1, np.nan, np.nan, 0])
        b = np.array([0., np.nan, 1, 0, np.nan])
        before = a.copy(), b.copy()
        np.testing.assert_equal(union_states([a, b]), [0, 1, 1, np.nan, np.nan])
        np.testing.assert_equal(a, before[0]); np.testing.assert_equal(b, before[1])

    def test_zone_crossing_becomes_one_bout_and_gap_is_not_bridged(self):
        t = np.arange(8, dtype=float)
        values = union_states([[0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]])
        on, off, dur = extract_complete_events(t, values)
        np.testing.assert_equal(on, [1]); np.testing.assert_equal(off, [4]); np.testing.assert_equal(dur, [3])
        missing = union_states([[0, 1, np.nan, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]])
        self.assertEqual(len(extract_complete_events(t, missing)[0]), 0)

    def test_interval_union_merges_overlap_touching_and_points(self):
        result = union_intervals([(np.array([1, 8]), np.array([4, 9]), None),
                                  (np.array([3, 9, 15, 15]), np.array([6, 12, 15, 15]), None)])
        np.testing.assert_equal(result['on'], [1, 8, 15])
        np.testing.assert_equal(result['off'], [6, 12, 15])
        np.testing.assert_equal(result['dur'], [5, 4, np.nan])
        self.assertEqual(union_intervals([])['on'].size, 0)

    def test_bad_data_is_rejected_without_guessing(self):
        for arrays in ([[0, 1], [0]], [[0, 2], [1, 0]]):
            with self.assertRaises(ValueError): union_states(arrays)
        with self.assertRaises(ValueError):
            union_intervals([(np.array([1.]), np.array([np.nan]), None)])


class CombinedEventGuiTests(unittest.TestCase):
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def setup_sources(self, zone=False, missing=False):
        p = self.panel
        t = np.linspace(0, 30, 301)
        names = ['Zone: One', 'Zone: Two'] if zone else ['walking', 'running']
        procs = [ProcessedTrial(path=f'animal-{i}.csv', channel_id='AIN01', time=t,
                               raw_signal=np.sin(t), raw_reference=np.cos(t),
                               output=np.sin(t) + i, output_label='dFF') for i in range(2)]
        p.receive_current_processed(procs)
        sources = {}
        for i, proc in enumerate(procs):
            states = {names[0]: ((t >= 8) & (t < 12)).astype(float),
                      names[1]: (((t >= 10) & (t < 14)) | ((t >= 20) & (t < 22))).astype(float)}
            if missing and i == 1: states.pop(names[1])
            sources[f'animal-{i}'] = {'kind': 'binary_columns', 'time': t.copy(), 'behaviors': states,
                                    'import_report': {'paired_path': proc.path, 'paired_index': i}}
        p._behavior_sources = sources
        p._refresh_behavior_list()
        return names, procs

    def test_psth_comparison_individual_group_and_trial_rows(self):
        p = self.panel
        names, procs = self.setup_sources()
        original = copy.deepcopy(p._behavior_sources)
        name, count, skipped = p._create_combined_event('moving', names)
        self.assertEqual((name, count, skipped), ('moving', 2, []))
        self.assertEqual(p.combo_behavior_name.currentText(), 'moving')
        self.assertEqual(p._last_mat.shape[0], 2)
        _, expected = compute_psth_matrix(procs[0].time, procs[0].output, np.array([8., 20.]),
                         (-p.spin_pre.value(), p.spin_post.value()),
                         (p.spin_b0.value(), p.spin_b1.value()), p.spin_resample.value(),
                         normalization=p._psth_normalization())
        np.testing.assert_allclose(p._last_mat, expected, equal_nan=True)
        p.tab_sources.setCurrentIndex(1)
        p.tab_visual_mode.setCurrentIndex(1); p._compute_psth()
        self.assertEqual(p._last_mat.shape[0], 2)
        p.cb_group_keep_trials.setChecked(True); p._compute_psth()
        self.assertEqual(p._last_mat.shape[0], 4)
        view = p.behavior_zone_panel
        listing = view.selectors['behavior']
        with QtCore.QSignalBlocker(listing):
            for i in range(listing.count()):
                item = listing.item(i)
                item.setCheckState(QtCore.Qt.CheckState.Checked if item.text() == 'moving' else QtCore.Qt.CheckState.Unchecked)
        view._compute()
        self.assertEqual(len(view._last_rows), 2)
        self.assertEqual(view._last_group_rows[0]['recordings'], 2)
        self.assertIn('moving', view._last_export_context['combined_labels'])
        self.assertIn('walking', str(p._collect_psth_parameter_sections()))
        for key, old in original.items():
            np.testing.assert_equal(p._behavior_sources[key]['time'], old['time'])
            for label in names:
                np.testing.assert_equal(p._behavior_sources[key]['behaviors'][label], old['behaviors'][label])

    def test_zone_union_keeps_zone_mode_and_survives_project(self):
        p = self.panel
        names, _ = self.setup_sources(zone=True)
        p._set_event_category('zone')
        name, _, _ = p._create_combined_event('Either zone', names)
        self.assertEqual(name, 'Zone: Either zone')
        self.assertEqual(p._event_category(), 'zone')
        self.assertFalse(p.behavior_zone_panel.isVisible())
        for info in p._behavior_sources.values():
            on, off, dur = p._extract_behavior_events(info, name)
            np.testing.assert_allclose(on, [8, 20]); np.testing.assert_allclose(off, [14, 22])
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'combined.h5')
            p._save_project_h5(path)
            restored = p._load_project_h5(path)['behavior_sources']
        p._behavior_sources = restored; p._refresh_behavior_list()
        info = restored['animal-0']
        self.assertEqual(info['import_report']['combined_labels'][name]['members'], names)
        np.testing.assert_allclose(p._extract_behavior_events(info, name)[0], [8, 20])

    def test_missing_components_exclude_recording_and_collisions_are_atomic(self):
        p = self.panel
        names, procs = self.setup_sources(missing=True)
        name, count, skipped = p._create_combined_event('moving', names)
        self.assertEqual((count, skipped), (1, ['animal-1']))
        self.assertNotIn(name, p._behavior_sources['animal-1']['behaviors'])
        with self.assertRaises(ValueError): p._create_combined_event(names[0], names)
        with self.assertRaises(ValueError): p._create_combined_event('bad', [names[0], 'missing'])
        self.assertNotIn('bad', p._behavior_sources['animal-0']['behaviors'])
        p._behavior_sources['animal-1']['behaviors'][names[1]] = p._behavior_sources['animal-0']['behaviors'][names[1]].copy()
        self.assertEqual(p._create_combined_event('moving', names)[1], 2)

    def test_derived_union_updates_after_source_change_and_disappears_if_missing(self):
        p = self.panel
        names, _ = self.setup_sources()
        p._create_combined_event('moving', names)
        source = p._behavior_sources['animal-0']
        source['behaviors'][names[0]][:] = 0
        source['behaviors'][names[1]][:] = 0
        p._refresh_behavior_list()
        self.assertFalse(source['behaviors']['moving'].any())
        source['behaviors'].pop(names[1]); p._refresh_behavior_list()
        self.assertNotIn('moving', source['behaviors'])

    def test_event_lists_and_legacy_timestamp_columns(self):
        p = self.panel
        self.setup_sources()
        source = p._behavior_sources['animal-0']
        source['behaviors'] = {}
        source['event_behaviors'] = {'a': {'on': np.array([8.]), 'off': np.array([12.])},
                                     'b': {'on': np.array([10.]), 'off': np.array([14.])}}
        p._create_combined_event('both', ['a', 'b'])
        np.testing.assert_equal(p._extract_behavior_events(source, 'both')[2], [6.])
        source['kind'] = 'timestamp_columns'; source['behaviors'] = {'x': np.array([8., 10.]), 'y': np.array([10., 15.])}
        p._create_combined_event('points', ['x', 'y'])
        np.testing.assert_equal(p._extract_behavior_events(source, 'points')[0], [8, 10, 15])

    def test_dialog_selection_validation_filter_and_cancel(self):
        self.setup_sources()
        dialog = CombineEventsDialog(['walking', 'running'], ['walking'], self.panel)
        self.assertFalse(dialog.create_button.isEnabled())
        dialog.name.setText('moving')
        dialog.choices.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
        self.assertTrue(dialog.create_button.isEnabled())
        dialog._filter('walk')
        self.assertEqual(dialog.members(), ['walking', 'running'])
        dialog.close()
        with patch('combined_events_dialog.CombineEventsDialog.exec', return_value=QtWidgets.QDialog.DialogCode.Rejected):
            self.panel._open_combine_events()
        self.assertNotIn('moving', self.panel._get_all_behavior_names())

    def test_combine_button_accepts_selection_and_activates_only_the_union(self):
        names, _ = self.setup_sources()
        def accept(dialog):
            dialog.name.setText('moving')
            for i in range(dialog.choices.count()):
                dialog.choices.item(i).setCheckState(QtCore.Qt.CheckState.Checked)
            return QtWidgets.QDialog.DialogCode.Accepted
        with patch.object(CombineEventsDialog, 'exec', accept):
            self.panel.btn_combine_events.click()
        self.assertEqual(self.panel.combo_behavior_name.currentText(), 'moving')
        self.assertEqual(self.panel.behavior_zone_panel._selected_labels(), ['moving'])
        self.assertTrue(set(names).issubset(self.panel._get_all_behavior_names()))

    def test_mixed_formats_abort_without_partial_combined_labels(self):
        names, _ = self.setup_sources()
        source = self.panel._behavior_sources['animal-1']
        source['behaviors'].pop(names[1])
        source['event_behaviors'] = {names[1]: {'on': np.array([10.]), 'off': np.array([14.])}}
        with self.assertRaisesRegex(ValueError, 'separately'):
            self.panel._create_combined_event('moving', names)
        for info in self.panel._behavior_sources.values():
            self.assertNotIn('moving', info.get('behaviors', {}))
            self.assertNotIn('moving', info.get('event_behaviors', {}))

    def test_arena_switch_rebuilds_union_even_with_a_different_frame_axis(self):
        import pandas as pd
        p = self.panel
        self.setup_sources()
        p._behavior_sources = {}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'arenas.xlsx'
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                pd.DataFrame({'Trial time': [1, 2, 3, 4], 'In zone 1': [0, 1, 0, 0],
                              'In zone 2': [0, 0, 1, 0]}).to_excel(writer, sheet_name='First', index=False)
                pd.DataFrame({'Trial time': [10, 11, 12, 13, 14], 'In zone 1': [0, 0, 0, 0, 0],
                              'In zone 2': [0, 1, 1, 1, 0]}).to_excel(writer, sheet_name='Second', index=False)
            original = path.read_bytes()
            options = p._inspect_arena_workbook(str(path))
            p._add_generic_behavior_zone_file(str(path), 'zone', sheet_name='First', arena_options=options, target_index=0)
            name, _, _ = p._create_combined_event('Either', ['Zone: In zone 1', 'Zone: In zone 2'])
            p._add_generic_behavior_zone_file(str(path), 'zone', sheet_name='Second', arena_options=options,
                                              target_index=0, replace_arena=True)
            source = p._behavior_sources['animal-0']
            np.testing.assert_equal(source['behaviors'][name], [0, 1, 1, 1, 0])
            np.testing.assert_equal(p._extract_behavior_events(source, name)[0], [11])
            self.assertEqual(path.read_bytes(), original)
