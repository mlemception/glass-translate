"""Assert that a PyInstaller-built exe carries none of the GPU / OCR heavyweights.

The app exe must stay torch-free (``docs/plans/2026-09-11-portable-bundle.md``
section 1): the GPU stack ships only inside the frozen sidecar
``renderer/glassrenderer.exe``.  This module inspects what is *actually* in the
built archive rather than trusting the spec's ``excludes`` list, so a hook that
quietly pulls ``torch`` back in is caught by the build instead of by a user
downloading two extra gigabytes.

Both PyInstaller layouts are read: a onefile exe (everything in the appended
CArchive plus its embedded PYZ) and a onedir launcher (the CArchive bootstrap
plus the sibling ``_internal`` contents directory).

    .venv\\Scripts\\python packaging/verify_torch_free.py dist/GlassTranslate.exe

prints the entry count and exits 0 when clean, or lists every offending entry
and exits 1.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "FORBIDDEN",
    "FORBIDDEN_BINARIES",
    "ArchiveScanError",
    "forbidden_entries",
    "frozen_toc",
    "frozen_top_level_modules",
    "verify_torch_free",
]

# Top-level import names that must never appear in the app archive.  Matched
# case-insensitively: Windows paths and PyInstaller TOC entries do not agree on
# case, and ``Torch\lib\torch_cpu.dll`` must not slip through.
FORBIDDEN: Tuple[str, ...] = (
    "torch",
    "torchvision",
    "paddle",
    "paddleocr",
    "manga_ocr",
    "glassrenderer",
    "diffusers",
    "transformers",
    "safetensors",
    "accelerate",
    "nvidia",  # the nvidia-*-cu13 wheels all install under this package
)

# Shared libraries that come with that stack even when the Python package is
# pruned (PyInstaller collects them as plain binaries, under any parent path).
# The CUDA families are the ones the frozen sidecar ships from torch/lib; none of
# them has any business in the app exe.
FORBIDDEN_BINARIES: Tuple[str, ...] = (
    "torch*.dll",
    "libtorch*",
    "c10*.dll",
    "c10*.so",
    "cudnn*",
    "cublas*",
    "cudart*",
    "cufft*",
    "cusparse*",
    "cusolver*",
    "curand*",
    "nvrtc*",
    "nvjitlink*",
    "cupti*",
    "nvtoolsext*",
)


class ArchiveScanError(RuntimeError):
    """The exe could not be scanned, so "no forbidden entries" cannot be claimed."""

# PyInstaller 6's onedir contents directory (the ``pyi-contents-directory`` option).
CONTENTS_DIR = "_internal"

# Archive entries that are data, not importable modules.
_DATA_SUFFIXES = (".pyz", ".dll", ".pyd", ".so", ".dylib", ".exe", ".zip", ".manifest")
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pyz_modules(reader: object, name: str) -> List[str]:
    """Module names of the embedded PYZ entry ``name`` of an open CArchive."""
    archive = reader.open_embedded_archive(name)  # type: ignore[attr-defined]
    return list(archive.toc)


def _onedir_entries(exe: Path, has_embedded_pyz: bool) -> List[str]:
    """Contents of a onedir launcher's ``_internal`` folder (empty for onefile).

    Fails closed: an exe with neither an embedded PYZ nor a contents directory has
    not been scanned at all, and reporting "no forbidden entries" for it would be a
    false clean bill of health.
    """
    internal = exe.parent / CONTENTS_DIR
    if not internal.is_dir():
        if not has_embedded_pyz:
            raise ArchiveScanError(
                f"{exe} has no embedded PYZ and no {CONTENTS_DIR} directory next to it: "
                "nothing could be scanned (is this a PyInstaller build?)"
            )
        return []
    from PyInstaller.loader.pyimod01_archive import ZlibArchiveReader

    names = [str(p.relative_to(internal)) for p in internal.rglob("*") if p.is_file()]
    for pyz in sorted(internal.glob("*.pyz")):
        names.extend(ZlibArchiveReader(str(pyz)).toc)
    return names


def frozen_toc(exe_path: Path | str) -> List[str]:
    """Every entry name of the PyInstaller archive(s) behind ``exe_path``.

    The list mixes three name shapes: dotted module names from the PYZ,
    backslash-separated binary/data paths from the CArchive, and the archive's
    runtime options.  Callers filter it with :func:`forbidden_entries` or
    :func:`frozen_top_level_modules`.
    """
    from PyInstaller.archive.readers import PKG_ITEM_PYZ, CArchiveReader

    exe = Path(exe_path)
    if not exe.is_file():
        raise FileNotFoundError(f"{exe} does not exist")
    reader = CArchiveReader(str(exe))
    names: List[str] = list(reader.toc) + list(reader.options)
    has_pyz = False
    for name, entry in reader.toc.items():
        # PyInstaller 6.22 names it ``PYZ.pyz``; older builds use ``PYZ-00.pyz``.
        if entry[-1] == PKG_ITEM_PYZ:
            has_pyz = True
            names.extend(_pyz_modules(reader, name))
    names.extend(_onedir_entries(exe, has_pyz))
    return names


def _parts(name: str) -> List[str]:
    return name.replace("\\", "/").split("/")


def frozen_top_level_modules(names: Iterable[str]) -> List[str]:
    """The distinct top-level import names among ``names``, sorted.

    A dotted PYZ entry contributes its first component; a binary path such as
    ``PySide6\\Qt6Core.dll`` contributes its first directory, which is how a
    C-extension-only package shows up.  Bare data entries and runtime options
    contribute nothing.
    """
    tops = set()
    for name in names:
        parts = _parts(name)
        if len(parts) > 1:
            if _MODULE_NAME.match(parts[0]):
                tops.add(parts[0])
            continue
        head = parts[0].split(".")[0]
        if parts[0].lower().endswith(_DATA_SUFFIXES) or not _MODULE_NAME.match(head):
            continue
        tops.add(head)
    return sorted(tops)


def _is_forbidden(name: str, forbidden: Sequence[str]) -> bool:
    parts = _parts(name)
    head = parts[0].lower()
    for item in forbidden:
        lowered = item.lower()
        if head == lowered or head.startswith(lowered + "."):
            return True
    basename = parts[-1].lower()
    return any(fnmatch.fnmatch(basename, pattern) for pattern in FORBIDDEN_BINARIES)


def forbidden_entries(
    names: Iterable[str], forbidden: Sequence[str] = FORBIDDEN
) -> List[str]:
    """The entries of ``names`` that must not be in the app archive, sorted.

    Matching is case-insensitive on both sides.  A module name matches as ``name``
    or ``name.``-prefixed, so the app's own ``glasstranslate.ocr.mangaocr`` is not
    confused with the ``manga_ocr`` package; binaries match by basename glob
    (``torch*.dll``, ``cudnn*``, ``cudart*``, ...) under any parent path.
    """
    return sorted({name for name in names if _is_forbidden(name, forbidden)})


def verify_torch_free(exe: Path | str, forbidden: Sequence[str] = FORBIDDEN) -> List[str]:
    """Offending entries of the built ``exe`` - an empty list means it is clean."""
    return forbidden_entries(frozen_toc(exe), forbidden)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("exe", type=Path, help="the PyInstaller-built exe to scan")
    args = parser.parse_args(argv)
    names = frozen_toc(args.exe)
    offenders = forbidden_entries(names)
    if offenders:
        print(f"{args.exe}: {len(offenders)} forbidden entries in {len(names)} archive entries")
        for name in offenders:
            print(f"  {name}")
        return 1
    print(f"{args.exe}: torch-free ({len(names)} archive entries, "
          f"{len(frozen_top_level_modules(names))} top-level modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
