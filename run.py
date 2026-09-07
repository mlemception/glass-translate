"""Launch GlassTranslate: ``python run.py`` from a source checkout, or the PyInstaller entry point.

When frozen (onefile ``GlassTranslate.exe``) the bootloader extracts the bundle into
``%TEMP%\\_MEIxxxxxx`` on every launch and removes it only on a normal exit; a Task-Manager kill
or a crash leaves ~360 MB behind.  Before anything from Qt is imported, this script sweeps stale
extraction directories that carry our ``gt_bundle.marker`` (docs/GLASS_DESIGN.md section 6).  A
concurrently running copy keeps its DLLs locked and simply survives the ``rmtree``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BUNDLE_MARKER = "gt_bundle.marker"


def sweep_stale_bundles(temp_dir: Path, own_bundle: str) -> list[Path]:
    """Remove ``_MEI*`` dirs under *temp_dir* that hold our marker and are not *own_bundle*.

    Returns the directories that were attempted (each ``rmtree`` ignores errors, so a directory
    still in use by another running copy stays behind).
    """
    import shutil

    own = os.path.normcase(os.path.abspath(own_bundle)) if own_bundle else ""
    swept: list[Path] = []
    try:
        candidates = list(temp_dir.glob("_MEI*"))
    except OSError:
        return swept
    for d in candidates:
        try:
            if not d.is_dir() or not (d / BUNDLE_MARKER).exists():
                continue
        except OSError:
            continue
        if os.path.normcase(os.path.abspath(str(d))) == own:
            continue
        shutil.rmtree(d, ignore_errors=True)
        swept.append(d)
    return swept


if getattr(sys, "frozen", False):
    import tempfile

    sweep_stale_bundles(Path(tempfile.gettempdir()), getattr(sys, "_MEIPASS", ""))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from glasstranslate.ui.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
