# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the single no-console ``GlassTranslate.exe`` (docs/GLASS_DESIGN.md section 6).

Driven by ``build.py`` (the one command); can also be run directly::

    set PYTHONUTF8=1 && .venv\\Scripts\\python -m PyInstaller --noconfirm --clean packaging\\GlassTranslate.spec

Environment knobs (PyInstaller specs take no CLI arguments):

* ``GT_BUILD_ENTRY``  - entry script (default ``run.py``; ``build.py --entry`` uses it for probe builds)
* ``GT_BUILD_NAME``   - exe base name (default ``GlassTranslate``)
* ``GT_BUILD_ONEDIR`` - ``1`` -> onedir (``COLLECT``) instead of onefile; starts in well under a
  second and avoids antivirus re-scans of the extracted DLLs on every launch
* ``GT_BUILD_STRICT`` - ``0`` -> a pefile closure warning no longer aborts the build (debugging only)

Pruning strategy (verified in demo/output/probes/packaging/): PyInstaller's PySide6.QtQml hook
collects *every* qml module directory and the binary dependency walk then drags in WebEngine,
Quick3D, opengl32sw and 246 translations (165 MB for a hello-world).  We therefore (1) exclude
every ``PySide6.Qt*`` module outside the keep set at Analysis level and (2) filter ``a.binaries`` /
``a.datas`` with allowlists for Qt6 DLLs, qml module dirs and plugins, then (3) verify with pefile
that no kept DLL/pyd imports a pruned Qt6 DLL.
"""
import glob
import os
import re
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

PACKAGING = os.path.abspath(SPECPATH)  # noqa: F821 - injected by PyInstaller
ROOT = os.path.dirname(PACKAGING)
ENTRY = os.path.abspath(os.environ.get("GT_BUILD_ENTRY") or os.path.join(ROOT, "run.py"))
EXE_NAME = os.environ.get("GT_BUILD_NAME") or "GlassTranslate"
ONEDIR = os.environ.get("GT_BUILD_ONEDIR") == "1"
STRICT = os.environ.get("GT_BUILD_STRICT", "1") != "0"

if not os.path.isfile(ENTRY):
    raise SystemExit(f"[spec] entry script not found: {ENTRY}")

# --------------------------------------------------------------------------- PySide6: python-level excludes
# Every control is a QtQuick.Templates type and there is no QQuickStyle call, so QtQuickControls2 and
# QtQuickWidgets are NOT needed.  QtQuick hard-requires QtOpenGL + QtNetwork at the shiboken level.
PYSIDE_KEEP = {"QtCore", "QtGui", "QtWidgets", "QtQml", "QtQuick", "QtNetwork", "QtOpenGL"}
import PySide6  # noqa: E402

PYSIDE_DIR = os.path.dirname(PySide6.__file__)
_all_pyside = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(PYSIDE_DIR, "Qt*.pyd"))}
EXCLUDES = ["PySide6." + m for m in sorted(_all_pyside - PYSIDE_KEEP)]
EXCLUDES += [
    "PySide6.QtAsyncio", "PySide6.support", "PySide6.scripts",
    # optional ML engines rapidocr probes for: keep excluded so a future install never bloats the exe
    "torch", "torchvision", "openvino", "paddle", "tensorrt", "MNN", "cupy", "onnx", "onnxsim",
    # big, clearly unused stdlib / third-party packages
    "tkinter", "unittest", "pydoc", "doctest", "xmlrpc", "sqlite3", "test", "curses",
    "matplotlib", "scipy", "pandas", "IPython", "setuptools", "pkg_resources",
]
# NEVER exclude logging/inspect/argparse/zipfile/tempfile/shutil/lzma/bz2/dis/ast/ssl/hashlib/socket/http/email/
# urllib: shiboken6's embedded signature bootstrap imports them at PySide6 import time (the exe dies with
# "Problem importing shibokensupport: No module named 'logging'") and the LibreTranslate backend needs https.
SHIBOKEN_RUNTIME_STDLIB = [
    "logging", "inspect", "argparse", "zipfile", "tempfile", "shutil", "typing", "dis", "ast", "tokenize", "glob",
    "fnmatch", "textwrap", "warnings", "traceback", "random", "base64", "gettext", "enum", "keyword", "struct",
    "pathlib", "lzma", "bz2", "zlib", "binascii",
]

# --------------------------------------------------------------------------- native deps (verified frozen)
rapidocr_datas = collect_data_files("rapidocr", includes=["*.yaml", "models/*", "inference_engine/**/*.yaml"])
rapidocr_hidden = (
    ["rapidocr.main", "rapidocr.utils.load_image", "rapidocr.utils.vis_res", "rapidocr.utils.download_models",
     "rapidocr.inference_engine.onnxruntime"]
    + collect_submodules("rapidocr.ch_ppocr_det")
    + collect_submodules("rapidocr.ch_ppocr_cls")
    + collect_submodules("rapidocr.ch_ppocr_rec")
)
ctranslate2_bins = collect_dynamic_libs("ctranslate2")  # ctranslate2.dll, libiomp5md.dll (cudnn stub pruned below)
sentencepiece_datas = collect_data_files("sentencepiece", includes=["package_data/*"])
py3langid_datas = collect_data_files("py3langid", includes=["data/*"])  # data/model.npz.xz
# resources_rc is imported dynamically (importlib) in control.py, invisible to Analysis.
extra_hidden = ["mss.windows", "comtypes.gen", "dxcam.processor.numpy_processor",
                "glasstranslate.ui.resources_rc"]
# Bundle-root marker: run.py sweeps stale %TEMP%\_MEI* dirs that contain it (onefile lifecycle, section 6).
marker_datas = [(os.path.join(PACKAGING, "gt_bundle.marker"), ".")]

a = Analysis(  # noqa: F821
    [ENTRY],
    pathex=[ROOT],
    binaries=ctranslate2_bins,
    datas=rapidocr_datas + sentencepiece_datas + py3langid_datas + marker_datas,
    hiddenimports=SHIBOKEN_RUNTIME_STDLIB + rapidocr_hidden + extra_hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=1,
)

# --------------------------------------------------------------------------- spec-level pruning (allowlists)
# Qt6 DLL closure verified with pefile on 6.11.2: Qt6Quick -> Core/Gui/Network/OpenGL/Qml/QmlMeta/QmlModels,
# Qt6QmlMeta -> QmlWorkerScript, Qt6QuickTemplates2 -> Core/Gui/Qml/QmlModels/Quick, qmodernwindowsstyle -> Widgets.
# No QuickControls2* (no QQuickStyle, Templates only) and no Svg.
QT_DLL_KEEP = {
    "Qt6Core", "Qt6Gui", "Qt6Widgets", "Qt6Qml", "Qt6QmlMeta", "Qt6QmlModels", "Qt6QmlWorkerScript", "Qt6Quick",
    "Qt6QuickTemplates2", "Qt6QuickLayouts", "Qt6Network", "Qt6OpenGL",
}
# qml module dirs (relative to PySide6/qml); a file is kept iff its dirname is EXACTLY one of these.
# QtQml/Base does not exist in 6.11; QtQuick/Controls* must never be listed (tools/build_resources.py --check
# fails the build if our QML ever imports QtQuick.Controls).
QML_KEEP = ("QtQuick", "QtQuick/Window", "QtQuick/Templates", "QtQuick/Layouts", "QtQml", "QtQml/Models",
            "QtQml/WorkerScript")
PLUGIN_KEEP = {
    "plugins/platforms/qwindows.dll", "plugins/styles/qmodernwindowsstyle.dll",
    "plugins/imageformats/qjpeg.dll", "plugins/imageformats/qico.dll", "plugins/imageformats/qgif.dll",
    "plugins/imageformats/qwebp.dll",
}
# PySide6's own libcrypto-3.dll / libssl-3.dll (QtNetwork TLS plugin) are dropped; CPython's libcrypto-3-x64.dll /
# libssl-3-x64.dll (no match: the regex is anchored on "-3.dll") stay for urllib https (LibreTranslate backend).
NON_QT_DROP = re.compile(r"^(opengl32sw|libcrypto-3|libssl-3|av(codec|format|util)-\d+|sw(scale|resample)-\d+)\.dll$", re.I)
DEPS_DROP = re.compile(
    r"(^|[\\/])(opencv_videoio_ffmpeg\d+_64\.dll|cv2[\\/]data[\\/].*|PIL[\\/]_avif[^\\/]*\.pyd|PIL[\\/]_imagingtk[^\\/]*\.pyd"
    r"|onnxruntime[\\/](datasets|tools)[\\/].*|nvidia[\\/].*|ctranslate2[\\/]cudnn64_9\.dll)$",
    re.I,
)


def keep(entry):
    """Allowlist filter for one TOC entry ``(dest, src, typecode)``."""
    dest = entry[0].replace("\\", "/")
    if DEPS_DROP.search(dest):
        return False
    if dest.startswith("PySide6/"):
        rel = dest[len("PySide6/"):]
        if rel.startswith("translations/"):
            return False
        if rel.startswith("qml/"):
            return os.path.dirname(rel[len("qml/"):]) in QML_KEEP
        if rel.startswith("plugins/"):
            return rel in PLUGIN_KEEP
        m = re.match(r"^(Qt6[A-Za-z0-9_]+)\.dll$", rel)
        if m:
            return m.group(1) in QT_DLL_KEEP
        return not NON_QT_DROP.match(rel)
    return not NON_QT_DROP.match(os.path.basename(dest))


nb, nd = len(a.binaries), len(a.datas)
a.binaries = [e for e in a.binaries if keep(e)]
a.datas = [e for e in a.datas if keep(e)]
print(f"[prune] binaries {nb} -> {len(a.binaries)}, datas {nd} -> {len(a.datas)}")

# Closure check: every kept DLL/pyd must have its Qt6 imports kept as well (catches allowlist mistakes at build
# time instead of as a DLL-load failure in the frozen exe).  Must print no warnings (section 6).
_closure_warnings = []
try:
    import pefile
except ImportError:  # pragma: no cover - requirements-build.txt installs it
    pefile = None
    print("[prune] WARNING: pefile not available, closure check skipped")
if pefile is not None:
    kept_names = {os.path.basename(e[0]).lower() for e in a.binaries}
    for dest, src, typ in a.binaries:
        if not src.lower().endswith((".dll", ".pyd")):
            continue
        pe = pefile.PE(src, fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        for imp in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            d = imp.dll.decode(errors="replace").lower()
            if d.startswith("qt6") and d not in kept_names:
                _closure_warnings.append(f"{dest} imports {d} which was pruned")
        pe.close()
    for w in _closure_warnings:
        print(f"[prune] WARNING: {w}")
    if _closure_warnings and STRICT:
        raise SystemExit(f"[prune] closure check failed with {len(_closure_warnings)} warning(s)")
    if not _closure_warnings:
        print(f"[prune] closure check OK ({len(kept_names)} binaries)")

_total_mb = sum(os.path.getsize(s) for _, s, _ in a.binaries + a.datas if os.path.isfile(s)) / 1e6
print(f"[prune] uncompressed payload (binaries + datas): {_total_mb:.1f} MB")

pyz = PYZ(a.pure)  # noqa: F821

exe_kwargs = dict(
    name=EXE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                      # GUI subsystem: sys.stdout/sys.stderr are None -> app logs to a file
    disable_windowed_traceback=False,   # unhandled exception -> message box
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(PACKAGING, "glasstranslate.ico"),        # 7 sizes, derived by packaging/make_icon.py
    version=os.path.join(PACKAGING, "version_info.txt"),       # generated by packaging/make_version_info.py
    manifest=os.path.join(PACKAGING, "glasstranslate.manifest"),  # PerMonitorV2 + UTF-8 ACP + asInvoker
    uac_admin=False,
    uac_uiaccess=False,
)

if ONEDIR:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_kwargs)  # noqa: F821
    coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=EXE_NAME)  # noqa: F821
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], runtime_tmpdir=None, **exe_kwargs)  # noqa: F821
