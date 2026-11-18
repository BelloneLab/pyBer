"""The automatic release step maps pyBer/version.py to the historical tag names."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from release_tag import version_to_tag, current_version


class ReleaseTagTest(unittest.TestCase):
    def test_tag_convention(self):
        self.assertEqual(version_to_tag("0.57.0"), "v0.57")
        self.assertEqual(version_to_tag("0.57.1"), "v0.57.1")
        self.assertEqual(version_to_tag("0.60.10"), "v0.60.10")
        with self.assertRaises(ValueError):
            version_to_tag("0.57")

    def test_current_version_has_release_notes(self):
        """A pushed version without notes would block the automatic release."""
        tag = version_to_tag(current_version())
        notes = Path(__file__).resolve().parents[1] / "docs" / "releases" / f"{tag}.md"
        self.assertTrue(notes.is_file(), f"Missing {notes.name}")


if __name__ == "__main__":
    unittest.main()
