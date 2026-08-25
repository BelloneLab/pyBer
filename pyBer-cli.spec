# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path


def _env_bin_dir() -> Path:
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    return Path(prefix) / "Library" / "bin"


def _existing_binaries(names):
    bin_dir = _env_bin_dir()
    return [(str(bin_dir / name), ".") for name in names if (bin_dir / name).is_file()]


a = Analysis(
    ["pyBer\\cli.py"],
    pathex=[],
    binaries=_existing_binaries([
        "hdf5.dll", "hdf5_hl.dll", "zlib.dll", "blosc.dll", "libblosc2.dll",
        "libmmd.dll", "libifcoremd.dll", "libifportmd.dll", "libiomp5md.dll",
        "libimalloc.dll", "svml_dispmd.dll", "libpng16.dll", "freetype.dll",
    ]),
    datas=[],
    hiddenimports=["matplotlib.backends.backend_agg"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["rpy2"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="pyBer-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["assets/pyBer.ico"],
)
