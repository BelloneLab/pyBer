# splash_helper.py
"""Helper process that plays pyBer's animated startup splash.

pyBer's main process is busy while it loads and cannot repaint, so it starts
this tiny process to draw the animation smoothly (see splash_screen.py).

Run from source:   python splash_helper.py --pyber-splash --pid N --stop FILE --pos X,Y
Frozen build:      pyBer.exe --pyber-splash ...   (main.py forwards here)
Preview (IDE):     run this file with no arguments; close with Alt+F4 or wait 20 s.

Keep this file small: it is executed as a script, so Python compiles it on
every launch.
"""
from __future__ import annotations

import os
import sys


def _add_conda_dll_directories() -> None:
    """Let Qt's DLLs load when the conda environment is not activated."""
    if os.name != "nt" or getattr(sys, "frozen", False) or not hasattr(os, "add_dll_directory"):
        return
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    for parts in ((), ("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"), ("Library", "bin"), ("Scripts",)):
        path = os.path.join(prefix, *parts)
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass


def main(argv=None) -> int:
    """Play the animation until the stop file appears or the launcher exits."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    _add_conda_dll_directories()
    folder = os.path.dirname(os.path.abspath(__file__))
    if not getattr(sys, "frozen", False) and folder not in sys.path:
        sys.path.insert(0, folder)
    import splash_screen

    if splash_screen.HELPER_FLAG not in arguments:
        # Preview mode: no launcher to watch, so stop after a while.
        splash_screen.HELPER_MAX_LIFETIME_S = 20.0
    return splash_screen.run_helper(arguments)


if __name__ == "__main__":
    sys.exit(main())
