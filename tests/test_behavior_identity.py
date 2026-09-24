"""Flat multi-animal recordings must be selected, never deduplicated globally."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PySide6 import QtWidgets

import test_postprocessing_empty_state as fixture
from analysis_core import ProcessedTrial
from behavior_import import select_identity_rows, BehaviorImportCancelled
from gui_postprocessing import _load_behavior_csv, _behavior_table_info


def table(n=4):
    rows = []
    for frame in range(100):
        for animal in range(1, n + 1):
            rows.append({'Frame': frame, 'Trial time (s)': 10 + frame / 10,
                         'Animal ID': animal, 'Arena ID': animal,
                         'Mouse': f'mouse-{animal}', 'Stream': 'video-1',
                         'Condition': 'Defeated' if animal == 1 else 'Cagemate',
                         'IZ': np.nan if frame == 70 else int(30 <= frame < 40 + animal)})
    return pd.DataFrame(rows)


class IdentitySelectionTests(unittest.TestCase):
    def test_four_ids_preserve_every_selected_value_and_original_order(self):
        data = table()
        original = data.copy(deep=True)
        for i in range(4):
            with self.subTest(id=i + 1):
                def choose(labels):
                    self.assertEqual(len(labels), 4)
                    self.assertIn(f'Animal ID: {i + 1}', labels[i])
                    return i
                selected, report = select_identity_rows(data, choose)
                pd.testing.assert_frame_equal(selected, data[data['Animal ID'] == i + 1])
                self.assertEqual(report['identity_selection']['Animal ID'], str(i + 1))
                self.assertEqual(report['selected_row_count'], 100)
                self.assertEqual(report['source_row_count'], 400)
        pd.testing.assert_frame_equal(data, original)

    def test_no_silent_default_and_cancel_does_not_change_data(self):
        data = table(2)
        with self.assertRaisesRegex(ValueError, 'Choose one ID'):
            select_identity_rows(data)
        with self.assertRaises(BehaviorImportCancelled):
            select_identity_rows(data, lambda _: None)
        self.assertEqual(len(data), 200)

    def test_ids_reused_across_videos_are_separate_choices(self):
        data = pd.concat([table(2), table(2).assign(Stream='video-2')], ignore_index=True)
        def choose(labels):
            self.assertEqual(len(labels), 4)
            return next(i for i, label in enumerate(labels) if 'video-2' in label and 'Animal ID: 2' in label)
        selected, _ = select_identity_rows(data, choose)
        self.assertEqual(selected['Stream'].unique().tolist(), ['video-2'])
        self.assertEqual(selected['Animal ID'].unique().tolist(), [2])

    def test_selected_timeline_still_rejects_real_duplicate_or_missing_times(self):
        for bad in (10., np.nan):
            data = table(2)
            data.loc[2, 'Trial time (s)'] = bad
            with self.assertRaisesRegex(ValueError, 'timestamps still'):
                select_identity_rows(data, lambda _: 0)

    def test_single_id_and_wide_legacy_tables_do_not_prompt(self):
        for data in (table(1), pd.DataFrame({'time': [0, 1, 2], 'groom': [0, 1, 0],
                                           'mouse_1_center_x': [1, 2, 3], 'mouse_2_center_x': [3, 2, 1]})):
            with patch('builtins.input', side_effect=AssertionError('Unexpected prompt')):
                selected, report = select_identity_rows(data, lambda _: self.fail('Unexpected picker'))
                pd.testing.assert_frame_equal(selected, data)
                self.assertEqual(report, {})


class IdentityImportGuiTests(unittest.TestCase):
    setUpClass = classmethod(fixture.PostprocessingEmptyStateTests.setUpClass.__func__)
    setUp = fixture.PostprocessingEmptyStateTests.setUp
    tearDown = fixture.PostprocessingEmptyStateTests.tearDown

    def fibers(self):
        t = np.linspace(10, 20, 1001)
        procs = [ProcessedTrial(path=f'fiber-{i + 1}.csv', channel_id='AIN01', time=t,
                                raw_signal=np.sin(t), raw_reference=np.cos(t),
                                output=np.sin(t), output_label='dFF') for i in range(4)]
        self.panel.receive_current_processed(procs)
        return procs

    def test_new_importer_selects_fourth_id_and_pairs_only_its_fiber(self):
        panel = self.panel
        procs = self.fibers()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'animals.csv'
            table().to_csv(path, index=False)
            original = path.read_bytes()
            with patch.object(QtWidgets.QInputDialog, 'getItem', side_effect=lambda *args: (args[3][3], True)) as pick:
                panel._load_behavior_zone_files('behavior', [str(path)])
            self.assertEqual(pick.call_count, 2)
            source = panel._match_behavior_source(procs[3])
            self.assertEqual(source['import_report']['identity_selection']['Animal ID'], '4')
            for proc in procs[:3]:
                self.assertIsNone(panel._match_behavior_source(proc))
            np.testing.assert_equal(source['time'], table(1)['Trial time (s)'])
            np.testing.assert_equal(source['behaviors']['IZ'], table().query('`Animal ID` == 4')['IZ'])
            self.assertEqual(len(panel._extract_behavior_events(source, 'IZ')[0]), 1)
            project = str(Path(folder) / 'test.h5')
            panel._save_project_h5(project)
            restored = panel._load_project_h5(project)
            saved_source = restored['behavior_sources']['fiber-4']
            self.assertEqual(saved_source['import_report']['identity_selection'],
                             source['import_report']['identity_selection'])
            self.assertEqual(saved_source['import_report']['paired_path'], procs[3].path)
            np.testing.assert_equal(saved_source['time'], source['time'])
            np.testing.assert_equal(saved_source['behaviors']['IZ'], source['behaviors']['IZ'])
            panel._behavior_sources = restored['behavior_sources']
            self.assertIsNotNone(panel._match_behavior_source(procs[3]))
            self.assertIsNone(panel._match_behavior_source(procs[0]))
            self.assertEqual(path.read_bytes(), original)

    def test_setup_importer_and_group_use_selected_recordings_only(self):
        panel = self.panel
        procs = self.fibers()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'animals.csv'
            table().to_csv(path, index=False)
            for index in (0, 2):
                with patch.object(QtWidgets.QInputDialog, 'getItem', side_effect=lambda *args: (args[3][index], True)):
                    panel._load_behavior_paths([str(path)], replace=False)
            self.assertIsNotNone(panel._match_behavior_source(procs[0]))
            self.assertIsNotNone(panel._match_behavior_source(procs[2]))
            self.assertIsNone(panel._match_behavior_source(procs[1]))
            self.assertIsNone(panel._match_behavior_source(procs[3]))
            panel._set_event_category('behavior')
            panel.tab_visual_mode.setCurrentIndex(1)
            view = panel.behavior_zone_panel
            view.reload()
            view.follow_psth_selection('IZ')
            view._compute()
            self.assertEqual(len(view._last_rows), 2)
            self.assertEqual(view._last_group_rows[0]['recordings'], 2)

    def test_cancel_in_either_importer_keeps_existing_sources(self):
        panel = self.panel
        self.fibers()
        existing = _behavior_table_info(pd.DataFrame({'time': [0, 1, 2], 'old': [0, 1, 0]}), 'binary_columns', 0)
        panel._behavior_sources['existing'] = existing
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'animals.csv'
            table().to_csv(path, index=False)
            for legacy in (True, False):
                with patch.object(QtWidgets.QInputDialog, 'getItem', return_value=('', False)), \
                     patch.object(QtWidgets.QMessageBox, 'warning') as warning:
                    if legacy:
                        panel._load_behavior_paths([str(path)], replace=True)
                    else:
                        panel._load_behavior_zone_files('behavior', [str(path)])
                    warning.assert_not_called()
                self.assertEqual(list(panel._behavior_sources), ['existing'])
                self.assertIs(panel._behavior_sources['existing'], existing)

    def test_invalid_embedded_timeline_is_reported_without_unexpected_error(self):
        panel = self.panel
        procs = self.fibers()
        source = _behavior_table_info(pd.DataFrame({'time': [10., 11., 12.], 'IZ': [0, 1, 0]}), 'binary_columns', 0)
        source['time'] = np.array([10., 10., 12.])
        source['import_report'].update(paired_path=procs[0].path, paired_index=0)
        panel._behavior_sources = {'fiber-1': source}
        panel._set_event_category('behavior')
        view = panel.behavior_zone_panel
        view.reload()
        view.follow_psth_selection('IZ')
        view._compute()
        self.assertFalse(view._last_rows)
        self.assertIn('invalid behavior timeline', view.status.text())
