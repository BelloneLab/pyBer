"""Print the GitHub release tag for the version in pyBer/version.py.

Convention used by every pyBer release: a ``.0`` patch is dropped
(0.57.0 -> v0.57) and any other patch is kept (0.57.1 -> v0.57.1).
Used by .github/workflows/build-and-publish.yml to publish automatically.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parents[1] / "pyBer" / "version.py"


def version_to_tag(version: str) -> str:
    """Map ``X.Y.Z`` to the release tag naming convention."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"Unexpected version format: {version!r}")
    return "v" + (version[:-2] if version.endswith(".0") else version)


def current_version() -> str:
    """Read ``__version__`` without importing the application."""
    match = re.search(r'__version__\s*=\s*"([^"]+)"', VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"No __version__ found in {VERSION_FILE}")
    return match.group(1)


if __name__ == "__main__":
    print(version_to_tag(sys.argv[1] if len(sys.argv) > 1 else current_version()))
