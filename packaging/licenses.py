"""Licence index for the portable bundle: what ships, under what, and the texts.

Step 4 of ``build_portable.py``.  The closure that actually ships is read from
the built archives (``verify_torch_free.frozen_top_level_modules``), mapped onto
the installed distributions of the venv that produced them, and written out as
``licenses/LICENSES.md`` plus a verbatim copy of every distribution's own
licence files.

The build gates on copyleft: a bundled distribution classified GPL/AGPL with no
other option fails the build.  LGPL is allowed and recorded (Qt/PySide6,
``pynput``, the GEOS DLL inside ``shapely``); ``pyphen`` is tri-licensed and is
recorded as used under its MPL/LGPL option.  PyInstaller and the other build
tools never ship, so they are excluded from the bundled set - PyInstaller's own
GPL-2.0-with-bootloader-exception is still explained in the index, because the
exe it produces relies on that exception.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

__all__ = [
    "ALLOWLIST",
    "BUILD_ONLY",
    "LicenceError",
    "MAX_LICENCE_FIELD_CHARS",
    "bundled_distributions",
    "classify",
    "collect_distributions",
    "gpl_violations",
    "write_license_index",
]

Dist = Dict[str, Any]
PathLike = Union[str, Path]

# A ``License:`` field longer than this is the licence *text*, not a name
# (scipy's is 46 kB of aggregated licences); classify by classifiers instead.
MAX_LICENCE_FIELD_CHARS = 200

# Present in a venv but never inside a frozen build: PyInstaller and its helpers,
# the packaging tooling and the test/eval-only extras.
BUILD_ONLY = frozenset(
    {
        "pyinstaller",
        "pyinstaller-hooks-contrib",
        "pefile",
        "altgraph",
        "pywin32-ctypes",
        "pip",
        "setuptools",
        "wheel",
        "pytest",
        "pluggy",
        "iniconfig",
        "lpips",
    }
)

# Bundled distributions whose GPL-family classification is resolved by another
# option in the same grant; the value is the note the index records.
ALLOWLIST: Dict[str, str] = {
    "pyphen": (
        "GPL-2.0-or-later OR LGPL-2.1-or-later OR MPL-1.1 - used under the MPL/LGPL option"
    ),
}

LGPL_NOTE = "dynamically linked; licence text shipped; sources at upstream"

# Qt ships only LicenseRef-Qt-Commercial.txt in its dist-info - no LGPL text at all - so
# the bundle carries one canonical copy, taken from the LGPL text another bundled
# distribution already ships (pynput's licenses/COPYING.LGPL).  Every LGPL-classified
# distribution without its own copy points at it.
LGPL_TEXT_NAME = "LGPL-3.0.txt"
QT_SOURCE_URL = "https://code.qt.io"
QT_LGPL_OBLIGATIONS_URL = "https://www.qt.io/licensing/open-source-lgpl-obligations"

_CLASSIFIER_SPDX = {
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "BSD License": "BSD",
    "Apache Software License": "Apache-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "ISC License (ISCL)": "ISC",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "Mozilla Public License 1.1 (MPL 1.1)": "MPL-1.1",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.1-or-later",
    "GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU Affero General Public License v3": "AGPL-3.0-only",
    "GNU Affero General Public License v3 or later (AGPLv3+)": "AGPL-3.0-or-later",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "The Unlicense (Unlicense)": "Unlicense",
    "zlib/libpng License": "Zlib",
    "Eclipse Public License 2.0 (EPL-2.0)": "EPL-2.0",
}

# "GPL" / "AGPL" but never the "GPL" inside "LGPL".
_STRONG_COPYLEFT = re.compile(r"(?<![A-Za-z])A?GPL", re.IGNORECASE)
# Options of a multi-licence grant: "A OR B", "A / B".  The separator must be
# surrounded by whitespace (or be a slash), or "GPL-3.0-or-later" would split.
_OPTION_SPLIT = re.compile(r"\s+OR\s+|\s*/\s*", re.IGNORECASE)

# The four model groups of renderer/MODELS.md, with the licence texts the
# models zip carries next to the weights.
MODEL_GROUPS = (
    (
        "LaMa inpainting",
        "models/quality/lama/",
        "MIT (checkpoint and packaging), Apache-2.0 notice for the vendored FFC architecture",
        "models/quality/lama/LICENSE, models/quality/lama/NOTICE",
        "TareHimself/AnimeMangaInpainting-torchscript",
    ),
    (
        "SDXL checkpoint and base configuration tree",
        "models/quality/sdxl/, models/quality/sdxl-base-config/",
        "CreativeML Open RAIL++-M (the SDXL 1.0 licence)",
        "models/quality/sdxl/LICENSE.md",
        "OnomaAIResearch/Illustrious-XL-v1.0, stabilityai/stable-diffusion-xl-base-1.0",
    ),
    (
        "ControlNet union, line-art mode",
        "models/quality/controlnet/",
        "Apache-2.0",
        "see renderer/MODELS.md section 3 for the grant and its URL",
        "xinsir/controlnet-union-sdxl-1.0",
    ),
    (
        "VAE (fp16 fix)",
        "models/quality/vae/",
        "MIT",
        "see renderer/MODELS.md section 2b for the grant and its URL",
        "madebyollin/sdxl-vae-fp16-fix",
    ),
)

_PYINSTALLER_NOTE = (
    "Both executables are produced with **PyInstaller**, which is GPL-2.0-or-later *with the "
    "bootloader exception*: the exception explicitly allows building and distributing non-free "
    "programs with it, and no PyInstaller source is part of either program beyond the bootloader "
    "the exception covers.  PyInstaller itself is not bundled."
)

_LGPL_NOTE_PARAGRAPH = (
    "**LGPL components.**  Qt (PySide6, PySide6-Essentials, PySide6-Addons, shiboken6), `pynput` "
    f"and the GEOS DLL inside `shapely` are LGPL: {LGPL_NOTE}.  Each one is dynamically linked or "
    "imported as a separate module, and the unmodified upstream sources are available from the "
    "projects named in the tables above.  Qt's own dist-info carries only its commercial licence "
    f"reference, so every LGPL-classified package that ships no LGPL text of its own points at "
    f"`{LGPL_TEXT_NAME}` in this folder, which is the verbatim GNU Lesser General Public License "
    f"text.  Qt's sources are at <{QT_SOURCE_URL}> and Qt's own summary of the LGPL obligations "
    f"is at <{QT_LGPL_OBLIGATIONS_URL}>."
)

# Dumped inside the target interpreter: importlib.metadata is the only source of
# truth for what that venv holds, so it must run there, not here.
_PROBE = r"""
import json, sys
from importlib import metadata

MAX_TEXT = 200000
LICENCE_NAMES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")


def licence_files(dist):
    out = []
    for entry in (dist.files or []):
        parts = str(entry).replace("\\", "/").split("/")
        if len(parts) < 2 or not parts[0].endswith((".dist-info", ".egg-info")):
            continue
        rel = "/".join(parts[1:])
        if not parts[-1].upper().startswith(LICENCE_NAMES) and parts[1] not in ("licenses", "license_files"):
            continue
        try:
            text = dist.read_text(rel)
        except (OSError, UnicodeDecodeError):
            text = None
        if not text:
            continue
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "\n... [truncated by packaging/licenses.py]\n"
        out.append({"name": rel, "text": text})
    return out


modules = {}
for module, owners in metadata.packages_distributions().items():
    for owner in owners:
        modules.setdefault(owner.lower(), set()).add(module)

rows, seen = [], set()
for dist in metadata.distributions():
    md = dist.metadata
    name = md["Name"] or ""
    if not name or name.lower() in seen:
        continue
    seen.add(name.lower())
    rows.append({
        "name": name,
        "version": dist.version or "",
        "license_expression": md.get("License-Expression", "") or "",
        "license_field": md.get("License", "") or "",
        "classifiers": md.get_all("Classifier") or [],
        "license_files": licence_files(dist),
        "modules": sorted(modules.get(name.lower(), ())),
    })
json.dump(rows, sys.stdout)
"""


class LicenceError(RuntimeError):
    """A bundled distribution fails the copyleft gate, or its metadata is unreadable."""


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def collect_distributions(python_exe: PathLike) -> List[Dist]:
    """Every installed distribution of ``python_exe``'s environment.

    Each row is ``{name, version, license_expression, license_field,
    classifiers, license_files: [{name, text}], modules}``; ``modules`` are the
    top-level import names ``importlib.metadata.packages_distributions()``
    attributes to it.
    """
    proc = subprocess.run(
        [str(python_exe), "-c", _PROBE],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise LicenceError(f"{python_exe} could not list its distributions: {proc.stderr.strip()}")
    try:
        rows = json.loads(proc.stdout)
    except ValueError as exc:
        raise LicenceError(f"{python_exe} returned unreadable metadata ({exc})") from exc
    return sorted(rows, key=lambda row: row["name"].lower())


def bundled_distributions(
    top_level_modules: Iterable[str], dists: Sequence[Dist]
) -> List[Dist]:
    """The distributions of ``dists`` that own one of ``top_level_modules``.

    Build-only tooling (:data:`BUILD_ONLY`) is never bundled even when a hook
    dragged one of its modules into the archive.
    """
    wanted = set(top_level_modules)
    keep = [
        dist
        for dist in dists
        if _normalise(dist["name"]) not in BUILD_ONLY and wanted.intersection(dist.get("modules") or ())
    ]
    return sorted(keep, key=lambda dist: dist["name"].lower())


def _classifier_licences(classifiers: Sequence[str]) -> List[str]:
    out: List[str] = []
    for classifier in classifiers:
        if not classifier.startswith("License ::"):
            continue
        label = classifier.split("::")[-1].strip()
        if not label or label == "OSI Approved":
            continue
        spdx = _CLASSIFIER_SPDX.get(label, label)
        if spdx not in out:
            out.append(spdx)
    return out


def classify(dist: Mapping[str, Any]) -> str:
    """An SPDX-ish classification for ``dist``, or ``"UNKNOWN"``.

    ``License-Expression`` is authoritative; otherwise the trove classifiers are
    joined with ``OR``; a short ``License:`` field is the last resort.  A long
    ``License:`` field is licence text, not a name, and is ignored.
    """
    expression = str(dist.get("license_expression") or "").strip()
    if expression:
        return expression
    from_classifiers = _classifier_licences(dist.get("classifiers") or ())
    if from_classifiers:
        return " OR ".join(from_classifiers)
    field = str(dist.get("license_field") or "").strip()
    if field and len(field) <= MAX_LICENCE_FIELD_CHARS:
        return field.splitlines()[0].strip()
    return "UNKNOWN"


def _is_strong_copyleft(option: str) -> bool:
    return bool(_STRONG_COPYLEFT.search(option))


def gpl_violations(
    dists: Sequence[Dist], allow: Mapping[str, str] = ALLOWLIST
) -> List[str]:
    """``"<name> (<classification>)"`` for every bundled dist that is GPL-only.

    A grant offering any non-GPL option (``pyphen``'s MPL/LGPL branches) passes,
    LGPL passes, and a distribution listed in ``allow`` passes with the note that
    mapping records.
    """
    allowed = {_normalise(name) for name in allow}
    out = []
    for dist in dists:
        if _normalise(dist["name"]) in allowed:
            continue
        classification = classify(dist)
        options = [part.strip() for part in _OPTION_SPLIT.split(classification) if part.strip()]
        if options and all(_is_strong_copyleft(option) for option in options):
            out.append(f"{dist['name']} ({classification})")
    return sorted(out)


def _note_for(dist: Mapping[str, Any], classification: str) -> str:
    note = ALLOWLIST.get(_normalise(dist["name"]), "")
    if note:
        return note
    return LGPL_NOTE if "LGPL" in classification.upper() else ""


def _copy_texts(out_dir: Path, dist: Mapping[str, Any]) -> List[str]:
    """Write ``dist``'s licence files under ``<name>-<version>/`` and list them."""
    folder = out_dir / f"{dist['name']}-{dist['version']}"
    written = []
    for item in dist.get("license_files") or ():
        parts = [p for p in str(item["name"]).replace("\\", "/").split("/") if p not in ("", ".", "..")]
        if not parts:
            continue
        target = folder.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item["text"], encoding="utf-8")
        written.append(f"{folder.name}/{'/'.join(parts)}")
    return written


def _looks_lgpl(item: Mapping[str, Any]) -> bool:
    """True when a dist-info licence file is (or names) an LGPL text."""
    if "LGPL" in str(item.get("name", "")).upper():
        return True
    return "LESSER GENERAL PUBLIC LICENSE" in str(item.get("text", ""))[:400].upper()


def needs_shared_lgpl(dist: Mapping[str, Any]) -> bool:
    """True when ``dist`` is LGPL-classified but ships no LGPL text of its own."""
    if "LGPL" not in classify(dist).upper():
        return False
    return not any(_looks_lgpl(item) for item in dist.get("license_files") or ())


def canonical_lgpl_text(dists: Sequence[Dist]) -> str:
    """The verbatim LGPL text another bundled distribution ships (pynput's COPYING.LGPL)."""
    for dist in dists:
        for item in dist.get("license_files") or ():
            if _looks_lgpl(item) and item.get("text"):
                return str(item["text"])
    raise LicenceError(
        "no bundled distribution ships an LGPL text, so the canonical "
        f"licenses/{LGPL_TEXT_NAME} cannot be produced: install pynput in the app venv "
        "(its dist-info licenses/COPYING.LGPL is the source) or add the text by hand"
    )


def _dist_table(out_dir: Path, dists: Sequence[Dist]) -> List[str]:
    lines = ["| Package | Version | Licence | Note | Texts |", "|---|---|---|---|---|"]
    for dist in dists:
        classification = classify(dist)
        texts = _copy_texts(out_dir, dist)
        if needs_shared_lgpl(dist):
            texts.append(LGPL_TEXT_NAME)
        lines.append(
            f"| `{dist['name']}` | {dist['version']} | {classification} | "
            f"{_note_for(dist, classification) or '-'} | "
            f"{', '.join(f'`{t}`' for t in texts) if texts else '-'} |"
        )
    return lines


def _models_table() -> List[str]:
    lines = ["| Group | Files | Licence | Licence text in this bundle | Source |", "|---|---|---|---|---|"]
    for group, files, licence, texts, source in MODEL_GROUPS:
        lines.append(f"| {group} | `{files}` | {licence} | {texts} | {source} |")
    return lines


def _packs_table(packs: Sequence[Mapping[str, Any]]) -> List[str]:
    """One row per translation pack, from the manifest's ``argos_packs`` entries."""
    if not packs:
        return ["No translation packs are shipped in this bundle."]
    lines = ["| Pair | Directory | Licence | Source | Pack README |", "|---|---|---|---|---|"]
    for pack in packs:
        location = pack.get("directory") or pack.get("archive") or "-"
        readme = pack.get("readme")
        lines.append(
            f"| {pack.get('pair', '-')} | `models/{location}` | {pack.get('licence') or 'unknown'} "
            f"| <{pack.get('source', '-')}> | {f'`models/{readme}`' if readme else '-'} |"
        )
    return lines


PERSONAL_USE_NOTE = (
    "This bundle is built for personal use on the machine that built it; do not "
    "redistribute it while the Sugoi pack is inside (use `--no-research-models` for a "
    "shareable build)."
)


def _personal_use_lines(packs: Sequence[Mapping[str, Any]]) -> List[str]:
    """The warning that one shipped pack may not be passed on, when one is shipped."""
    if all(pack.get("redistributable", True) for pack in packs):
        return []
    return ["", f"**{PERSONAL_USE_NOTE}**"]


def _excluded_packs_lines(excluded: Sequence[Mapping[str, Any]]) -> List[str]:
    """The packs the store holds but the bundle deliberately does not ship."""
    if not excluded:
        return []
    lines = ["", "**Not shipped in this bundle.**  The following translation packs exist "
                 "upstream but their licence does not permit redistribution here:", ""]
    for pack in excluded:
        lines.append(
            f"* `models/{pack.get('directory') or pack.get('archive') or '?'}` "
            f"({pack.get('pair', '-')}) - {pack.get('reason', 'not redistributable')}.  "
            f"Source: <{pack.get('source', '-')}>."
        )
    return lines


def _models_md_label(models_md_path: PathLike) -> str:
    """``renderer/MODELS.md`` - never the absolute developer path of the build machine."""
    parts = Path(models_md_path).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else Path(models_md_path).name


def write_license_index(
    dest_dir: PathLike,
    app_dists: Sequence[Dist],
    sidecar_dists: Sequence[Dist],
    models_md_path: PathLike,
    packs: Sequence[Mapping[str, Any]] = (),
    excluded_packs: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Write ``<dest_dir>/licenses/`` (index + verbatim texts) and return the index.

    ``packs`` and ``excluded_packs`` are the manifest's entries of the same names,
    rendered as section 3a.  Raises :class:`LicenceError` before writing anything when a bundled
    distribution fails the copyleft gate, or when the canonical LGPL text is needed but
    no bundled distribution ships one.
    """
    all_dists = list(app_dists) + list(sidecar_dists)
    violations = gpl_violations(all_dists)
    if violations:
        raise LicenceError("GPL-only distributions would be bundled: " + ", ".join(violations))
    shared_lgpl = canonical_lgpl_text(all_dists) if any(map(needs_shared_lgpl, all_dists)) else None
    out_dir = Path(dest_dir) / "licenses"
    out_dir.mkdir(parents=True, exist_ok=True)
    if shared_lgpl is not None:
        (out_dir / LGPL_TEXT_NAME).write_text(shared_lgpl, encoding="utf-8")
    lines = [
        "# GlassTranslate - third-party licences",
        "",
        "Everything this bundle ships, with the licence each component is used under.",
        "The full text of every distribution's own licence files is in this folder,",
        "one directory per package.  Model provenance, digests and the licence",
        f"reasoning are in `{_models_md_label(models_md_path)}` in the GlassTranslate",
        "source tree.",
        "",
        "## 1. GlassTranslate.exe",
        "",
        *_dist_table(out_dir, app_dists),
        "",
        "## 2. renderer\\glassrenderer.exe (the quality sidecar)",
        "",
        *_dist_table(out_dir, sidecar_dists),
        "",
        "## 3. Models",
        "",
        *_models_table(),
        "",
        "The manga-ocr ONNX bundle (`models/manga-ocr/`) is Apache-2.0 (both source",
        "repositories).",
        "",
        "### 3a. Translation packs",
        "",
        *_packs_table(packs),
        "",
        "A licence recorded as `unknown` means the upstream page states no licence for the",
        "pack itself: the argospm index licenses its own repository, not the packages it",
        "lists.  Check the source before redistributing such a pack.",
        *_personal_use_lines(packs),
        *_excluded_packs_lines(excluded_packs),
        "",
        "## 4. Build tooling and copyleft notes",
        "",
        _PYINSTALLER_NOTE,
        "",
        _LGPL_NOTE_PARAGRAPH,
        "",
    ]
    index = out_dir / "LICENSES.md"
    index.write_text("\n".join(lines), encoding="utf-8")
    return index
