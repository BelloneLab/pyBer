"""Regression tests for unambiguous postprocessing export filenames."""

import os
import sys
import unittest
from types import MethodType, SimpleNamespace


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYBER_SMOKE_TEST", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "pyBer"))

from gui_postprocessing import PostProcessingPanel  # noqa: E402


class _ComboStub:
    """Small QComboBox stand-in used to test naming without building the GUI."""

    def __init__(self, text: str) -> None:
        self.text = text

    def currentText(self) -> str:
        return self.text


class _TabStub:
    """Small QTabWidget stand-in for the single/group export context."""

    def currentIndex(self) -> int:
        return 0


def _panel_stub(
    *,
    source: str,
    behavior: str = "Social contact",
    behavior_align: str = "Align to onset",
    dio: str = "DIO 1",
    dio_align: str = "Align to onset",
) -> SimpleNamespace:
    """Create only the state needed by the production prefix methods."""
    panel = SimpleNamespace(
        _processed=[SimpleNamespace(path=os.path.join("recordings", "mouse_01.csv"))],
        combo_align=_ComboStub(source),
        combo_behavior_name=_ComboStub(behavior),
        combo_behavior_align=_ComboStub(behavior_align),
        combo_behavior_from=_ComboStub("Approach"),
        combo_behavior_to=_ComboStub("Contact"),
        combo_dio=_ComboStub(dio),
        combo_dio_align=_ComboStub(dio_align),
        tab_sources=_TabStub(),
    )
    # Bind the exact production methods while avoiding construction of the
    # large postprocessing widget and its unrelated rendering dependencies.
    for name in (
        "_behavior_suffix",
        "_alignment_export_suffix",
        "_is_group_export_context",
        "_group_export_prefix",
    ):
        setattr(panel, name, MethodType(getattr(PostProcessingPanel, name), panel))
    return panel


class PostprocessingExportNameTests(unittest.TestCase):
    def test_behavior_onset_and_offset_have_distinct_prefixes(self) -> None:
        panel = _panel_stub(source="Behavior (CSV/XLSX)")
        onset = PostProcessingPanel._default_export_prefix(panel)

        panel.combo_behavior_align.text = "Align to offset"
        offset = PostProcessingPanel._default_export_prefix(panel)

        self.assertEqual(onset, "mouse_01_Social_contact_onset")
        self.assertEqual(offset, "mouse_01_Social_contact_offset")
        self.assertNotEqual(onset, offset)

    def test_dio_onset_and_offset_have_distinct_prefixes(self) -> None:
        panel = _panel_stub(source="Analog/Digital channel (from Doric)")
        onset = PostProcessingPanel._default_export_prefix(panel)

        panel.combo_dio_align.text = "Align to offset"
        offset = PostProcessingPanel._default_export_prefix(panel)

        self.assertEqual(onset, "mouse_01_DIO_1_onset")
        self.assertEqual(offset, "mouse_01_DIO_1_offset")
        self.assertNotEqual(onset, offset)

    def test_generated_continuous_name_does_not_repeat_edge(self) -> None:
        panel = _panel_stub(
            source="Behavior (CSV/XLSX)",
            behavior="Speed [> 5] offset",
            behavior_align="Align to offset",
        )

        prefix = PostProcessingPanel._default_export_prefix(panel)

        self.assertEqual(prefix, "mouse_01_Speed__5_offset")
        self.assertNotIn("offset_offset", prefix)


if __name__ == "__main__":
    unittest.main()
