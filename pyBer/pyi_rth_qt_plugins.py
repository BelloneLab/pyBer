import os
import sys


def _set_qt_plugin_paths():
    if not getattr(sys, "frozen", False):
        return

    base = getattr(sys, "_MEIPASS", "")
    if not base:
        return

    plugin_roots = [
        os.path.join(base, "PySide6", "plugins"),
        os.path.join(base, "PySide6", "Qt", "plugins"),
        os.path.join(base, "PySide6", "Qt6", "plugins"),
        os.path.join(base, "plugins"),
    ]

    for root in plugin_roots:
        if os.path.isdir(root):
            os.environ.setdefault("QT_PLUGIN_PATH", root)
            platforms = os.path.join(root, "platforms")
            if os.path.isdir(platforms):
                os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", platforms)
            break

    # Ensure bundled Qt DLLs are discoverable for plugin dependencies.
    search_paths = [base, os.path.join(base, "PySide6")]
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(search_paths + [existing])


_set_qt_plugin_paths()
