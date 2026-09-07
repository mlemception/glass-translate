"""Compile the Qt resources of the control window (docs/GLASS_DESIGN.md section 4).

Steps of :func:`build`:

1. compile every ``glasstranslate/ui/qml/shaders/*.frag`` to ``<name>.frag.qsb`` with ``qsb``
   (``--glsl "100 es,120,150" --hlsl 50 --msl 12``; the direct ``PySide6/qsb.exe`` is preferred -
   the ``pyside6-qsb`` wrapper compiles fine but hangs on ``--help``, so it is never asked for help);
2. regenerate ``glasstranslate/ui/resources.qrc`` from a directory scan (prefix ``/qml``: every
   ``.qml``, ``qmldir`` and ``shaders/*.qsb``; ``/fonts``: the two Anime Ace files; ``/icons``:
   ``app.png``) so a QML file added by another owner can never be missing from the bundle;
3. run ``rcc -g python`` into ``glasstranslate/ui/resources_rc.py`` and prepend the header line
   ``# inputs-sha256: <digest>`` where the digest is the sha256 over the sorted
   ``(relpath, sha256)`` pairs of every qrc input and every ``.frag`` source.

``--check`` recomputes the digest and compares it with the header (no rebuild, no byte
comparison of outputs: mtimes are meaningless after a checkout and rcc/qsb bytes differ across
PySide6 versions) and runs the ``qmlimportscanner`` guard: our QML may only import modules whose
qml directory the PyInstaller spec ships (``QtQuick``, ``QtQuick/Window``, ``QtQuick/Templates``,
``QtQuick/Layouts``, ``QtQml``, ``QtQml/Models``, ``QtQml/WorkerScript``).  Exit codes: 0 ok,
1 digest / missing output, 2 import guard.

Usage: ``python tools/build_resources.py [--check] [--verbose]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

__all__ = [
    "ALLOWED_QML_MODULES",
    "DIGEST_PREFIX",
    "BuildError",
    "build",
    "check",
    "check_digest",
    "check_imports",
    "compile_shaders",
    "compute_digest",
    "main",
    "qrc_entries",
    "read_digest",
    "scan_qml_imports",
    "write_qrc",
]

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIGEST_PREFIX = "# inputs-sha256: "
QSB_FLAGS = ("--glsl", "100 es,120,150", "--hlsl", "50", "--msl", "12")
FONT_FILES = ("animeace2_reg.ttf", "animeace2_ital.ttf")
ALLOWED_QML_MODULES = frozenset({
    "QtQuick", "QtQuick/Window", "QtQuick/Templates", "QtQuick/Layouts",
    "QtQml", "QtQml/Models", "QtQml/WorkerScript",
})
# Built-in modules the scanner lists without a qml directory (they live inside Qt6Qml.dll).
BUILTIN_QML_MODULES = frozenset({"QML", "QtQml.Base"})
TOOL_TIMEOUT_S = 120


class BuildError(RuntimeError):
    """A tool was missing or failed."""


# --------------------------------------------------------------------------------------- paths
def ui_dir(root: Path = PROJECT_ROOT) -> Path:
    return root / "glasstranslate" / "ui"


def qml_dir(root: Path = PROJECT_ROOT) -> Path:
    return ui_dir(root) / "qml"


def shader_dir(root: Path = PROJECT_ROOT) -> Path:
    return qml_dir(root) / "shaders"


def qrc_path(root: Path = PROJECT_ROOT) -> Path:
    return ui_dir(root) / "resources.qrc"


def rc_path(root: Path = PROJECT_ROOT) -> Path:
    return ui_dir(root) / "resources_rc.py"


def font_dir(root: Path = PROJECT_ROOT) -> Path:
    return root / "glasstranslate" / "render" / "fonts" / "animeace"


def icon_path(root: Path = PROJECT_ROOT) -> Path:
    return ui_dir(root) / "icons" / "app.png"


def pyside_dir() -> Optional[Path]:
    try:
        import PySide6  # noqa: WPS433

        return Path(PySide6.__file__).resolve().parent
    except ImportError:
        return None


def tool_path(name: str) -> Optional[str]:
    """Locate a Qt tool: ``PySide6/<name>[.exe]`` first, then ``pyside6-<name>`` next to the
    interpreter or on PATH."""
    pdir = pyside_dir()
    if pdir is not None:
        for cand in (pdir / f"{name}.exe", pdir / name):
            if cand.is_file():
                return str(cand)
    scripts = Path(sys.executable).resolve().parent
    for cand in (scripts / f"pyside6-{name}.exe", scripts / f"pyside6-{name}"):
        if cand.is_file():
            return str(cand)
    return shutil.which(f"pyside6-{name}") or shutil.which(name)


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    try:
        return subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                              timeout=TOOL_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{cmd[0]} timed out after {TOOL_TIMEOUT_S} s") from exc
    except OSError as exc:
        raise BuildError(f"cannot run {cmd[0]}: {exc}") from exc


# ------------------------------------------------------------------------------------- shaders
def compile_shaders(root: Path = PROJECT_ROOT) -> List[Path]:
    """Compile every ``qml/shaders/*.frag`` to ``.frag.qsb``; returns the outputs."""
    sdir = shader_dir(root)
    frags = sorted(sdir.glob("*.frag")) if sdir.is_dir() else []
    if not frags:
        return []
    qsb = tool_path("qsb")
    if qsb is None:
        raise BuildError("qsb not found (PySide6/qsb.exe or pyside6-qsb)")
    outputs: List[Path] = []
    for frag in frags:
        out = frag.with_name(frag.name + ".qsb")
        proc = _run([qsb, *QSB_FLAGS, "-o", str(out), str(frag)])
        if proc.returncode != 0 or not out.is_file():
            raise BuildError(f"qsb failed for {frag.name}:\n{proc.stdout}\n{proc.stderr}")
        outputs.append(out)
        log.info("compiled %s", out.relative_to(root))
    return outputs


# ----------------------------------------------------------------------------------------- qrc
def qrc_entries(root: Path = PROJECT_ROOT) -> Dict[str, List[Tuple[str, Path]]]:
    """``{prefix: [(alias, absolute_path), ...]}`` for everything the bundle needs, sorted."""
    entries: Dict[str, List[Tuple[str, Path]]] = {"/qml": [], "/fonts": [], "/icons": []}
    qdir = qml_dir(root)
    if qdir.is_dir():
        files = list(qdir.rglob("*.qml")) + list(qdir.rglob("qmldir")) + list(qdir.rglob("*.qsb"))
        for f in files:
            if not f.is_file():
                continue
            entries["/qml"].append((f.relative_to(qdir).as_posix(), f))
    fdir = font_dir(root)
    for name in FONT_FILES:
        f = fdir / name
        if f.is_file():
            entries["/fonts"].append((name, f))
    icon = icon_path(root)
    if icon.is_file():
        entries["/icons"].append(("app.png", icon))
    for prefix in entries:
        entries[prefix] = sorted(set(entries[prefix]), key=lambda e: e[0])
    return entries


def render_qrc(root: Path = PROJECT_ROOT) -> str:
    """The ``resources.qrc`` text (paths relative to ``glasstranslate/ui``)."""
    base = ui_dir(root)
    lines = ["<!DOCTYPE RCC>", '<RCC version="1.0">']
    for prefix, items in qrc_entries(root).items():
        if not items:
            continue
        lines.append(f'  <qresource prefix="{prefix}">')
        for alias, path in items:
            rel = Path(os.path.relpath(path, base)).as_posix()
            lines.append(f'    <file alias="{escape(alias)}">{escape(rel)}</file>')
        lines.append("  </qresource>")
    lines.append("</RCC>")
    return "\n".join(lines) + "\n"


def write_qrc(root: Path = PROJECT_ROOT) -> bool:
    """Regenerate ``resources.qrc``; True when the file changed."""
    text = render_qrc(root)
    path = qrc_path(root)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8", newline="\n")
    log.info("wrote %s", path.relative_to(root))
    return True


# -------------------------------------------------------------------------------------- digest
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_inputs(root: Path = PROJECT_ROOT) -> List[Tuple[str, str]]:
    """Sorted ``(relpath, sha256)`` of every qrc input and every ``.frag`` source."""
    files = {p for items in qrc_entries(root).values() for _alias, p in items}
    sdir = shader_dir(root)
    if sdir.is_dir():
        files.update(p for p in sdir.glob("*.frag") if p.is_file())
    pairs = [(p.resolve().relative_to(root.resolve()).as_posix(), _sha256(p)) for p in files]
    return sorted(pairs)


def compute_digest(root: Path = PROJECT_ROOT) -> str:
    h = hashlib.sha256()
    for rel, sha in digest_inputs(root):
        h.update(f"{rel}\t{sha}\n".encode("utf-8"))
    return h.hexdigest()


def read_digest(path: Path) -> Optional[str]:
    """The digest recorded in the header of a generated ``resources_rc.py`` (None if absent)."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for _ in range(3):
            line = fh.readline()
            if line.startswith(DIGEST_PREFIX):
                return line[len(DIGEST_PREFIX):].strip()
    return None


# ----------------------------------------------------------------------------------------- rcc
def compile_rcc(root: Path = PROJECT_ROOT) -> Path:
    rcc = tool_path("rcc")
    if rcc is None:
        raise BuildError("rcc not found (PySide6/rcc.exe or pyside6-rcc)")
    out = rc_path(root)
    proc = _run([rcc, "-g", "python", "-o", str(out), str(qrc_path(root))], cwd=ui_dir(root))
    if proc.returncode != 0 or not out.is_file():
        raise BuildError(f"rcc failed:\n{proc.stdout}\n{proc.stderr}")
    body = out.read_text(encoding="utf-8")
    out.write_text(f"{DIGEST_PREFIX}{compute_digest(root)}\n{body}", encoding="utf-8", newline="\n")
    return out


def build(root: Path = PROJECT_ROOT) -> Path:
    """Compile shaders, regenerate the qrc, run rcc; returns the ``resources_rc.py`` path."""
    root = Path(root)
    compile_shaders(root)
    write_qrc(root)
    out = compile_rcc(root)
    log.info("wrote %s (%d bytes)", out.relative_to(root), out.stat().st_size)
    return out


# --------------------------------------------------------------------------------------- check
def check_digest(root: Path = PROJECT_ROOT) -> Tuple[bool, str]:
    """Compare the recorded digest with a fresh one."""
    rc = rc_path(root)
    recorded = read_digest(rc)
    if recorded is None:
        return False, f"{rc.relative_to(root)} missing or without a '{DIGEST_PREFIX.strip()}' header"
    current = compute_digest(root)
    if recorded != current:
        return False, f"resources_rc.py is stale: header {recorded[:12]}..., inputs {current[:12]}..."
    return True, "digest ok"


def scan_qml_imports(root: Path = PROJECT_ROOT, qml_root: Optional[Path] = None) -> List[dict]:
    """Run ``qmlimportscanner`` over the QML directory and return its JSON list."""
    qdir = qml_root or qml_dir(root)
    if not qdir.is_dir():
        return []
    scanner = tool_path("qmlimportscanner")
    pdir = pyside_dir()
    if scanner is None or pdir is None:
        raise BuildError("qmlimportscanner not found (PySide6/qmlimportscanner.exe)")
    proc = _run([scanner, "-rootPath", str(qdir), "-importPath", str(pdir / "qml")])
    if proc.returncode != 0:
        raise BuildError(f"qmlimportscanner failed:\n{proc.stdout}\n{proc.stderr}")
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise BuildError(f"qmlimportscanner output is not JSON: {exc}") from exc
    return data if isinstance(data, list) else []


def check_imports(root: Path = PROJECT_ROOT, qml_root: Optional[Path] = None) -> List[str]:
    """Modules imported by our QML that the frozen exe does not ship (empty = ok).

    Every ``type == "module"`` entry with a qml directory outside :data:`ALLOWED_QML_MODULES`
    is reported (optional plugins included: the exe needs their qmldir either way); built-ins
    without a directory (``QML``) are fine.
    """
    bad: List[str] = []
    for entry in scan_qml_imports(root, qml_root):
        if entry.get("type") != "module":
            continue
        name = str(entry.get("name", "?"))
        rel = entry.get("relativePath")
        if rel is None:
            if name in BUILTIN_QML_MODULES or "path" not in entry:
                continue
            rel = name.replace(".", "/")
        rel = str(rel).replace("\\", "/")
        if rel not in ALLOWED_QML_MODULES:
            optional = " (optional plugin)" if entry.get("pluginIsOptional") else ""
            bad.append(f"{name} -> {rel}{optional}")
    return sorted(set(bad))


def check(root: Path = PROJECT_ROOT) -> int:
    """``--check``: 0 ok, 1 digest / missing output, 2 forbidden QML import."""
    root = Path(root)
    ok, msg = check_digest(root)
    print(msg)
    if not ok:
        return 1
    bad = check_imports(root)
    if bad:
        print("forbidden QML imports (not shipped by packaging/GlassTranslate.spec):")
        for line in bad:
            print("  " + line)
        return 2
    print("qml imports ok")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify the digest and the QML import guard")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.check:
            return check(args.root)
        out = build(args.root)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
