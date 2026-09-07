"""Static uniform contract between the compiled shaders and their QML wrappers (docs/GLASS_DESIGN.md 7).

For every ``qml/shaders/*.qsb`` the reflection JSON of ``qsb -d`` is parsed and every uniform-block
member (except ``qt_Matrix`` / ``qt_Opacity``) and every sampler must be declared as a QML property of
the matching type on each wrapper that references the shader: ``float <- real``, ``vec2 <- point/size``,
``vec4 <- rect/color``, ``sampler2D <- var``.  Every ``fragmentShader:`` string in ``qml/`` must exist in
the compiled resources.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set

import pytest

from tools import build_resources as B

ROOT = Path(__file__).resolve().parents[1]
QML_DIR = ROOT / "glasstranslate" / "ui" / "qml"
SHADER_DIR = QML_DIR / "shaders"
TYPE_MAP = {"float": {"real", "double"}, "vec2": {"point", "size"}, "vec4": {"rect", "color", "vector4d"},
            "sampler2D": {"var", "ShaderEffectSource", "Item", "QtObject"}}
BUILTIN = {"qt_Matrix", "qt_Opacity"}
_PROP_RE = re.compile(r"^\s*(?:readonly\s+|required\s+|default\s+)*property\s+([A-Za-z_][\w.]*)\s+([A-Za-z_]\w*)", re.M)
_FRAG_RE = re.compile(r"fragmentShader\s*:\s*\"([^\"]+)\"")


def _reflection(qsb: Path) -> dict:
    tool = B.tool_path("qsb")
    assert tool, "qsb not found"
    proc = subprocess.run([tool, "-d", str(qsb)], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    start = text.index("Reflection info:") + len("Reflection info:")
    data, _end = json.JSONDecoder().raw_decode(text[start:].lstrip())  # shader sources follow the JSON
    return data


def _qml_files() -> List[Path]:
    return sorted(p for p in QML_DIR.rglob("*.qml"))


def _declared(qml: Path) -> Dict[str, Set[str]]:
    props: Dict[str, Set[str]] = {}
    for qtype, name in _PROP_RE.findall(qml.read_text(encoding="utf-8")):
        props.setdefault(name, set()).add(qtype)
    return props


def _wrappers() -> Dict[str, List[Path]]:
    """``{shader file name: [qml files referencing it]}``."""
    out: Dict[str, List[Path]] = {}
    for qml in _qml_files():
        for ref in _FRAG_RE.findall(qml.read_text(encoding="utf-8")):
            out.setdefault(Path(ref).name, []).append(qml)
    return out


@pytest.fixture(scope="module")
def qsb_files(resources_built: Path) -> List[Path]:
    files = sorted(SHADER_DIR.glob("*.qsb"))
    assert files, "no compiled shaders"
    return files


def test_every_shader_has_a_wrapper(qsb_files: List[Path]) -> None:
    wrappers = _wrappers()
    missing = [q.name for q in qsb_files if q.name not in wrappers]
    if missing and not (QML_DIR / "Main.qml").is_file():
        pytest.skip(f"Main.qml not delivered yet; shaders without a wrapper so far: {missing}")
    assert not missing, f"compiled shaders without a QML wrapper: {missing}"


def test_uniform_contract(qsb_files: List[Path]) -> None:
    wrappers = _wrappers()
    problems: List[str] = []
    for qsb in qsb_files:
        refl = _reflection(qsb)
        blocks = refl.get("uniformBlocks") or []
        assert blocks and blocks[0]["binding"] == 0, qsb.name
        members = [(m["name"], m["type"]) for m in blocks[0]["members"] if m["name"] not in BUILTIN]
        first = [m["name"] for m in blocks[0]["members"]][:2]
        assert first == ["qt_Matrix", "qt_Opacity"], f"{qsb.name}: block must start with qt_Matrix, qt_Opacity"
        samplers = [(s["name"], "sampler2D") for s in refl.get("combinedImageSamplers") or []]
        for s in refl.get("combinedImageSamplers") or []:
            assert s["binding"] >= 1, f"{qsb.name}: sampler {s['name']} needs an explicit binding >= 1"
        for qml in wrappers.get(qsb.name, []):
            declared = _declared(qml)
            for name, gtype in members + samplers:
                kinds = declared.get(name)
                allowed = TYPE_MAP.get(gtype)
                if allowed is None:
                    problems.append(f"{qsb.name}: unsupported uniform type {gtype} for {name}")
                elif kinds is None:
                    problems.append(f"{qml.relative_to(ROOT)}: no property for uniform {name} ({gtype})")
                elif not kinds & allowed:
                    problems.append(f"{qml.relative_to(ROOT)}: {name} declared as {sorted(kinds)}, shader wants {gtype}")
    assert not problems, "\n".join(problems)


def test_float_uniforms_are_never_bool_or_int(qsb_files: List[Path]) -> None:
    """Qt writes bool/int bits into float slots (a `property bool solidMode` reads as ~0.0)."""
    wrappers = _wrappers()
    offenders: List[str] = []
    for qsb in qsb_files:
        refl = _reflection(qsb)
        floats = {m["name"] for m in refl["uniformBlocks"][0]["members"] if m["type"] == "float"} - BUILTIN
        for qml in wrappers.get(qsb.name, []):
            for name, kinds in _declared(qml).items():
                if name in floats and kinds & {"bool", "int"}:
                    offenders.append(f"{qml.relative_to(ROOT)}: {name} is {sorted(kinds)}")
    assert not offenders, "\n".join(offenders)


def test_fragment_shader_urls_exist_in_resources(qapp) -> None:
    from PySide6.QtCore import QFile

    missing: List[str] = []
    for qml in _qml_files():
        for ref in _FRAG_RE.findall(qml.read_text(encoding="utf-8")):
            if ref.startswith("qrc:/"):
                path = ":" + ref[len("qrc:"):]
            elif ref.startswith(":/"):
                path = ref
            else:
                path = ":/qml/" + (qml.parent.relative_to(QML_DIR) / ref).as_posix().replace("./", "")
            if not QFile.exists(path):
                missing.append(f"{qml.relative_to(ROOT)} -> {ref} ({path})")
    assert not missing, "\n".join(missing)
