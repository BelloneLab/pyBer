# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path


def _env_bin_dir() -> Path:
    # Use the runtime belonging to the interpreter that executes PyInstaller,
    # even when the parent shell has a stale CONDA_PREFIX.
    return Path(sys.prefix) / "Library" / "bin"


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
        "liblzma.dll", "libbz2.dll", "ffi-8.dll", "libexpat.dll", "sqlite3.dll",
        "libcrypto-3-x64.dll", "libssl-3-x64.dll",
        # Native runtimes used by scipy and analysis_core.QtCore. Conda stores
        # these outside site-packages, so dependency discovery cannot find them
        # unless the spec seeds them explicitly.
        "libblas.dll", "libcblas.dll", "liblapack.dll",
        "libgcc_s_seh-1.dll", "libgomp-1.dll", "libquadmath-0.dll", "vcomp140.dll",
        "pyside6.cp311-win_amd64.dll", "shiboken6.cp311-win_amd64.dll",
        "Qt6Core.dll", "Qt6Network.dll",
        "jpeg8.dll", "lcms2.dll", "openjp2.dll", "qhull_r.dll", "tiff.dll",
        "libwebp.dll", "libwebpdemux.dll", "libwebpmux.dll",
        "mkl_rt.2.dll",
        "mkl_core.2.dll", "mkl_intel_thread.2.dll", "mkl_sequential.2.dll",
        "mkl_def.2.dll", "mkl_avx2.2.dll", "mkl_avx512.2.dll", "mkl_mc3.2.dll",
        "mkl_vml_avx2.2.dll", "mkl_vml_avx512.2.dll", "mkl_vml_cmpt.2.dll",
        "mkl_vml_def.2.dll", "mkl_vml_mc3.2.dll",
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
