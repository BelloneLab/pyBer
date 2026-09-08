"""Explicit Windows taskbar identity and persistent relaunch artwork.

Window icons alone do not specify the shell's taskbar-group icon. These window
properties implement the separate shell contract without requiring pywin32.
Reference: https://learn.microsoft.com/en-us/windows/win32/properties/props-system-appusermodel-relaunchiconresource
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import uuid


class GUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]

    @classmethod
    def parse(cls, value):
        return cls((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]


class _Value(ctypes.Union):
    # PROPVARIANT's largest member contains a count and an aligned pointer.
    _fields_ = [("text", ctypes.c_wchar_p), ("storage", ctypes.c_void_p * 2)]


class PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3), ("value", _Value)]


@lru_cache(maxsize=4)
def persistent_icon(path: str) -> str:
    """Give the shell a durable, content-versioned path, including frozen apps."""
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    destination = root / "pyBer" / "icons" / f"pyBer-{digest}.ico"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or destination.read_bytes() != payload:
        destination.write_bytes(payload)
    return str(destination)


def relaunch_command() -> str:
    """Launch the application itself, never an IDE debugger or temporary script."""
    executable = Path(sys.executable)
    if getattr(sys, "frozen", False):
        return subprocess.list2cmdline([str(executable)])
    windowed = executable.with_name("pythonw.exe")
    if windowed.is_file():
        executable = windowed
    script = Path(__file__).resolve().with_name("main.py")
    return subprocess.list2cmdline([str(executable), str(script)])


def window_properties(hwnd: int, values: dict[int, str] | None = None) -> dict[int, str]:
    """Write/read IPropertyStore strings and release every native allocation.

    Property IDs: 2 relaunch command, 3 icon resource, 4 display name, 5 app ID.
    Set relaunch metadata before the explicit window ID as required by Windows.
    """
    shell = ctypes.windll.shell32
    ole = ctypes.windll.ole32
    getter = shell.SHGetPropertyStoreForWindow
    getter.argtypes = [wintypes.HWND, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    getter.restype = ctypes.HRESULT
    store = ctypes.c_void_p()
    iid = GUID.parse("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
    result = getter(hwnd, ctypes.byref(iid), ctypes.byref(store))
    if result < 0:
        raise OSError(f"Window property store failed: {result:#x}")
    table = ctypes.cast(store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    release = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(table[2])
    read = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT))(table[5])
    write = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT))(table[6])
    allocate = ole.CoTaskMemAlloc
    allocate.argtypes = [ctypes.c_size_t]
    allocate.restype = ctypes.c_void_p
    clear = ole.PropVariantClear
    clear.argtypes = [ctypes.POINTER(PROPVARIANT)]
    clear.restype = ctypes.HRESULT
    fmtid = GUID.parse("9f4c2855-9f79-4b39-a8d0-e1d42de1d5f3")
    observed = {}
    try:
        for pid in (2, 3, 4, 5):
            key = PROPERTYKEY(fmtid, pid)
            value = PROPVARIANT()
            try:
                if values is not None:
                    text = ctypes.create_unicode_buffer(values[pid])
                    memory = allocate(ctypes.sizeof(text))
                    if not memory:
                        raise OSError("Could not allocate taskbar property string")
                    ctypes.memmove(memory, text, ctypes.sizeof(text))
                    value.vt = 31  # VT_LPWSTR, owned by PropVariantClear.
                    value.value.text = ctypes.cast(memory, ctypes.c_wchar_p)
                    result = write(store, ctypes.byref(key), ctypes.byref(value))
                else:
                    result = read(store, ctypes.byref(key), ctypes.byref(value))
                if result < 0:
                    raise OSError(f"Taskbar property {pid} failed: {result:#x}")
                if value.vt == 31:
                    observed[pid] = value.value.text
            finally:
                clear(ctypes.byref(value))
    finally:
        release(store)
    return observed


def install_window_identity(hwnd: int, app_id: str, icon: str) -> None:
    """Bind a taskbar group to the frameless icon instead of the Python host."""
    resource = persistent_icon(icon)
    window_properties(hwnd, {2: relaunch_command(), 3: f"{resource},0", 4: "pyBer", 5: app_id})
