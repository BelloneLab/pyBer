# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path


def _env_bin_dir() -> Path:
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    return Path(prefix) / "Library" / "bin"


def _existing_binaries(names):
    bin_dir = _env_bin_dir()
    binaries = []
    for name in names:
        path = bin_dir / name
        if path.is_file():
            binaries.append((str(path), "."))
    return binaries


a = Analysis(
    ['pyBer\\main.py'],
    pathex=[],
    binaries=_existing_binaries([
        'hdf5.dll',
        'hdf5_hl.dll',
        'zlib.dll',
        'blosc.dll',
        'libblosc2.dll',
        'libmmd.dll',
        'libifcoremd.dll',
        'libifportmd.dll',
        'libiomp5md.dll',
        'libimalloc.dll',
        'svml_dispmd.dll',
        'libpng16.dll',
        'freetype.dll',
    ]),
    datas=[('assets/pyBer_logo_big.png', 'assets'), ('assets/pyBer.ico', 'assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='pyBer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/pyBer.ico'],
)
