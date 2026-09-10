"""Deferred initialization must never access a panel that has been destroyed."""
import sys
import unittest
from unittest.mock import patch

import test_postprocessing_empty_state as fixture
from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid


class PostprocessingLifetimeTests(unittest.TestCase):
    """Closing a newly created panel cancels its queued initialization work."""

    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    setUp = fixture.PostprocessingEmptyStateTests.setUp

    def tearDown(self):
        if isValid(self.panel):
            fixture.PostprocessingEmptyStateTests.tearDown(self)
        else:
            self.resources.close()

    def test_deleted_panel_cancels_deferred_history_and_layout_callbacks(self):
        # Destroy before the queued zero-delay initialization gets a turn.
        # PySide routes callback failures through excepthook, outside unittest.
        with patch.object(sys, "excepthook") as errors:
            self.panel.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            self.assertFalse(isValid(self.panel))
            for _ in range(5):
                self.app.processEvents()
            errors.assert_not_called()
