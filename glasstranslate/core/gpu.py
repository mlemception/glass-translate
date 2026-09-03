"""Runtime GPU library discovery.  Must be imported before ctranslate2 so the
pip-installed CUDA runtime DLLs (nvidia-cublas-cu12 etc.) are on PATH."""
from __future__ import annotations

import glob
import os
import site
import sys

_done = False


def prepare_cuda_path() -> list[str]:
    global _done
    found: list[str] = []
    if _done:
        return found
    _done = True
    if sys.platform != "win32":
        return found
    candidates = list(site.getsitepackages())
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in candidates:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            found.append(d)
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    return found
