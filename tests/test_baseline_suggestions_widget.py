"""Exercise inline auto-refresh, caching and stale-result suppression in real Qt."""
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pyBer"))

import numpy as np
from PySide6 import QtCore, QtTest, QtWidgets
from baseline_advisor import BaselineRecording
from baseline_suggestions import BaselineSuggestionConfig, suggest_baselines
from baseline_suggestions_widget import BaselineSuggestionsWidget, _ACTIVE_JOBS


def _result(pre=5.):
    """Clearly synthetic GUI fixture; numerical quality is tested separately."""
    return dict(status="ready", summary="Synthetic fixture", config=dict(pre_window_s=pre),
                choices=[dict(start=-pre, end=-.2, window=[-pre, -.2], score=82,
                              quality="Supported", summary="Synthetic fixture", diagnostics={})])


class BaselineSuggestionsWidgetTests(unittest.TestCase):
    """Widget input snapshots may change at any point during background work."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.config = BaselineSuggestionConfig()
        t = np.arange(0., 160., .05)
        self.recording = BaselineRecording("synthetic", t, np.random.default_rng(5).normal(size=len(t)), np.arange(10, 151, 10.))
        self.fail_snapshot = False
        self.widget = BaselineSuggestionsWidget(self.snapshot, lambda: self.config)
        self.widget.timer.setInterval(10)

    def snapshot(self):
        """Return a detached request or simulate a newly invalid alignment input."""
        if self.fail_snapshot:
            raise ValueError("new input is invalid")
        return [self.recording], "Synthetic individual scope"

    def wait_for(self, condition, timeout=3):
        """Keep processing queued signals while waiting for a bounded test result."""
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            QtTest.QTest.qWait(5)
        self.assertTrue(condition(), "Timed out waiting for queued Qt work")

    def tearDown(self):
        self.widget.stop()
        self.widget.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        self.wait_for(lambda: not _ACTIVE_JOBS)

    def test_automatic_result_never_applies_without_click(self):
        choices = []
        self.widget.selected.connect(choices.append)
        self.widget.queue()
        self.wait_for(lambda: self.widget.report is not None)
        self.assertEqual(choices, [])
        self.assertTrue(self.widget.buttons[0].isEnabled())
        self.widget.buttons[0].click()
        self.assertEqual(len(choices), 1)
        self.assertLess(choices[0][1], 0)
        self.assertIn("not confidence", self.widget.buttons[0].toolTip())

    def test_unchanged_baseline_edits_hit_cache_but_prewindow_changes_recompute(self):
        with patch("baseline_suggestions_widget.suggest_baselines", wraps=suggest_baselines) as estimate:
            self.widget.queue()
            self.wait_for(lambda: self.widget.report is not None)
            self.assertEqual(estimate.call_count, 1)
            for _ in range(3):
                self.widget.queue()
            self.wait_for(lambda: not self.widget.timer.isActive() and self.widget.buttons[0].isEnabled())
            self.assertEqual(estimate.call_count, 1)
            self.config = BaselineSuggestionConfig(pre_window_s=4.)
            self.widget.queue()
            self.wait_for(lambda: (self.widget.report or {}).get("config", {}).get("pre_window_s") == 4.)
            self.assertEqual(estimate.call_count, 2)

    def test_superseded_result_never_replaces_newer_scope(self):
        entered, release = threading.Event(), threading.Event()

        def estimate(_recordings, config):
            if config.pre_window_s == 5:
                entered.set()
                release.wait(3)
            return _result(config.pre_window_s)

        shown = []
        original = self.widget._show_report

        def display(report):
            shown.append(report["config"]["pre_window_s"])
            original(report)

        with patch("baseline_suggestions_widget.suggest_baselines", side_effect=estimate), \
                patch.object(self.widget, "_show_report", side_effect=display):
            try:
                self.widget.queue()
                self.wait_for(entered.is_set)
                self.config = BaselineSuggestionConfig(pre_window_s=4.)
                self.widget.queue()
                self.wait_for(lambda: self.widget._pending is not None)
                release.set()
                self.wait_for(lambda: self.widget.report is not None)
                self.assertEqual(shown, [4.])
            finally:
                release.set()

    def test_snapshot_error_invalidates_an_older_running_result(self):
        entered, release = threading.Event(), threading.Event()

        def estimate(*_args):
            entered.set()
            release.wait(3)
            return _result()

        with patch("baseline_suggestions_widget.suggest_baselines", side_effect=estimate):
            try:
                self.widget.queue()
                self.wait_for(entered.is_set)
                self.fail_snapshot = True
                self.widget.queue()
                self.wait_for(lambda: "new input is invalid" in self.widget.status.text())
                release.set()
                self.wait_for(lambda: not _ACTIVE_JOBS)
                self.assertIsNone(self.widget.report)
                self.assertTrue(all(not button.isEnabled() for button in self.widget.buttons))
                self.assertIn("new input is invalid", self.widget.status.text())
            finally:
                release.set()

    def test_closure_drops_callbacks_without_destroying_running_work(self):
        entered, release = threading.Event(), threading.Event()

        def estimate(*_args):
            entered.set()
            release.wait(3)
            return _result()

        with patch("baseline_suggestions_widget.suggest_baselines", side_effect=estimate):
            try:
                self.widget.queue()
                self.wait_for(entered.is_set)
                self.widget.stop()
                release.set()
                self.wait_for(lambda: not _ACTIVE_JOBS)
                self.assertIsNone(self.widget.report)
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
