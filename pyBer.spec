# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path


def _env_bin_dir() -> Path:
    # Bind native libraries to the interpreter running PyInstaller. A stale
    # CONDA_PREFIX can point at base even when PyInstaller is invoked through a
    # different environment's python.exe, producing an executable that fails
    # before startup when _ctypes loads the wrong ffi DLL.
    return Path(sys.prefix) / "Library" / "bin"


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
        'liblzma.dll',
        'libbz2.dll',
        'ffi-8.dll',
        'libexpat.dll',
        'sqlite3.dll',
        'libcrypto-3-x64.dll',
        'libssl-3-x64.dll',
        # Conda keeps BLAS/LAPACK and Qt beside the interpreter rather than
        # inside their Python packages. PyInstaller sees the extension modules
        # but cannot resolve these native dependencies unless they are seeded
        # explicitly. Without them, the one-file app only works on machines
        # that already have the build environment on PATH.
        'libblas.dll',
        'libcblas.dll',
        'liblapack.dll',
        'libgcc_s_seh-1.dll',
        'libgomp-1.dll',
        'libquadmath-0.dll',
        'vcomp140.dll',
        'pyside6.cp311-win_amd64.dll',
        'shiboken6.cp311-win_amd64.dll',
        'Qt6Core.dll',
        'Qt6Network.dll',
        'Qt6Gui.dll',
        'Qt6Widgets.dll',
        'Qt6Svg.dll',
        'Qt6OpenGL.dll',
        'Qt6OpenGLWidgets.dll',
        'Qt6Test.dll',
        'jpeg8.dll',
        'lcms2.dll',
        'openjp2.dll',
        'qhull_r.dll',
        'tiff.dll',
        'libwebp.dll',
        'libwebpdemux.dll',
        'libwebpmux.dll',
        # conda-forge's libblas/liblapack shims resolve MKL symbols by the
        # versioned name at runtime, so dependency scanners do not see it.
        'mkl_rt.2.dll',
        'mkl_core.2.dll',
        'mkl_intel_thread.2.dll',
        'mkl_sequential.2.dll',
        'mkl_def.2.dll',
        'mkl_avx2.2.dll',
        'mkl_avx512.2.dll',
        'mkl_mc3.2.dll',
        'mkl_vml_avx2.2.dll',
        'mkl_vml_avx512.2.dll',
        'mkl_vml_cmpt.2.dll',
        'mkl_vml_def.2.dll',
        'mkl_vml_mc3.2.dll',
    ]),
    datas=[('assets/pyBer_logo_big.png', 'assets'), ('assets/pyBer.ico', 'assets')],
    hiddenimports=[
        'PySide6.QtOpenGL',
        'rpy2.rinterface',
        'rpy2.rinterface_lib',
        'rpy2.robjects',
        'rpy2.robjects.packages',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# One-DIR build, no UPX. The previous one-FILE build re-extracted the whole
# ~800 MB bundle (Qt, MKL, scipy) to a temp dir on EVERY launch and UPX added
# a decompress pass on top: 36-60 s from double-click to window. The folder
# build loads DLLs in place: a few seconds. Distribute by zipping dist/pyBer.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pyBer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets/pyBer.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='pyBer',
)
