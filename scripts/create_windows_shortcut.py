"""Create an Explorer shortcut using the currently selected pyBer environment.

Run with the pyBer environment's Python. Double-click the resulting shortcut
in ordinary File Explorer to avoid inheriting an elevated IDE's privileges.
This never restarts an existing session or changes Windows security settings.
"""
from pathlib import Path
import os
import subprocess
import sys


def create_shortcut():
    """Write a normal, non-elevating shortcut beside the source checkout."""
    if sys.platform != "win32":
        raise RuntimeError("This shortcut is for Windows.")
    root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).with_name("pythonw.exe")
    if not python.is_file():
        raise RuntimeError("Run this script with the pyBer conda environment's Python.")
    destination = root / "Launch pyBer.lnk"
    # Paths travel as environment data, never executable PowerShell fragments.
    environment = os.environ.copy()
    environment.update(PYBER_LINK=str(destination), PYBER_PYTHON=str(python),
                       PYBER_ROOT=str(root), PYBER_ENTRY=str(root / "pyBer" / "main.py"),
                       PYBER_ICON=str(root / "assets" / "pyBer.ico"))
    script = '''
$ErrorActionPreference = 'Stop'
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($env:PYBER_LINK)
$link.TargetPath = $env:PYBER_PYTHON
$link.Arguments = '"' + $env:PYBER_ENTRY + '"'
$link.WorkingDirectory = $env:PYBER_ROOT
$link.IconLocation = $env:PYBER_ICON + ',0'
$link.Description = 'Launch pyBer from File Explorer without administrator mode'
$link.Save()
'''
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                   env=environment, check=True)
    # A previous shortcut may have had RunAsUser set. Clear that flag only in
    # this generated shortcut; no application compatibility settings are touched.
    contents = bytearray(destination.read_bytes())
    flags = int.from_bytes(contents[20:24], "little")
    contents[20:24] = (flags & ~0x2000).to_bytes(4, "little")
    destination.write_bytes(contents)
    return destination


if __name__ == "__main__":
    print(create_shortcut())
