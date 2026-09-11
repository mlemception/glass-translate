"""Assemble the two portable zips from the built exes and the local model store.

    .venv\\Scripts\\python build_portable.py               # build both zips
    .venv\\Scripts\\python build_portable.py --single      # one -full-win64.zip with program + models
    .venv\\Scripts\\python build_portable.py --dry-run     # list what would be zipped
    .venv\\Scripts\\python build_portable.py --stage-only  # refresh the staged folder only

Steps:
  (1) preconditions          - dist/GlassTranslate.exe and dist/renderer/glassrenderer.exe
  (2) packaging/verify_torch_free.py - the app exe carries no torch / OCR heavyweight
  (3) packaging/portable_store.py    - re-hash the whole local store, discover the Argos packs,
                                       report the uncovered pairs, write models/MANIFEST.json
  (4) stage build/portable/GlassTranslate-<version>/ - exe, renderer/, portable.txt,
                                       README-portable.txt, licenses/ (packaging/licenses.py)
  (5) dist/GlassTranslate-<version>-portable-win64.zip + .sha256, from the staged folder
  (6) dist/GlassTranslate-<version>-models-win64.zip + .sha256, straight from models/ (never
                                       staged: copying 11 GB to zip it costs an hour and a disk)
  (7) a size / timing table, also written to build/portable-build.json

``--single`` replaces (5) and (6) with one ``dist/GlassTranslate-<version>-full-win64.zip`` that
holds the program and the models in the same layout (the staged folder plus ``models/`` and its
``MANIFEST.json`` under the one top-level folder); the pair from an earlier build is left alone.

Each zip is written to ``<name>.part`` and renamed into place only when it is complete, and its
checksum is written immediately after it, so a failed run never leaves a stale zip/checksum pair.

Nothing is ever downloaded: a store file that is missing or does not match its pinned digest
fails the build.  ``--dry-run`` performs (1)-(3) and lists the entries of both zips without
writing anything; a missing sidecar is a warning there and a hard failure in a real build.
``--stage-only`` stops after (4), so the staged folder (licence index included) can be
refreshed without spending an hour rewriting 14 GB of zips.  The bundle is built for
personal use on the machine that builds it, so the research-licensed Sugoi pack is
included by default; ``--no-research-models`` leaves it out for a bundle that will be
shared.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import textwrap
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
PACKAGING = ROOT / "packaging"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
MODELS_DIR = ROOT / "models"

sys.path.insert(0, str(PACKAGING))
import licenses  # noqa: E402
import make_version_info  # noqa: E402
import portable_store  # noqa: E402
import verify_torch_free  # noqa: E402

APP_EXE = DIST / "GlassTranslate.exe"
RENDERER_DIR = DIST / "renderer"
RENDERER_EXE = RENDERER_DIR / "glassrenderer.exe"
APP_VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RENDERER_VENV_PYTHON = ROOT / "renderer" / ".venv" / "Scripts" / "python.exe"
MODELS_MD = ROOT / "renderer" / "MODELS.md"
README_TEMPLATE = PACKAGING / "README-portable.txt"

STAGE_ROOT = BUILD / "portable"
REPORT_PATH = BUILD / "portable-build.json"
MANIFEST_NAME = "MANIFEST.json"
MARKER_NAME = "portable.txt"
MARKER_TEXT = (
    "GlassTranslate portable marker — keep this file next to GlassTranslate.exe; "
    "models/, config/ and logs/ live in this folder. Version {version}, built {date}.\n"
)

# Already-compressed payloads: deflating them costs minutes and saves nothing.
STORED_SUFFIXES = (".safetensors", ".onnx", ".bin", ".pt")
STORED_MIN_BYTES = 64 * 1024 * 1024

Entry = Tuple[str, Optional[Path]]
# One archive of the build plan: (step banner, label, zip name, sorted entries).
ZipPlan = Tuple[str, str, str, List[Entry]]


class PortableBuildError(RuntimeError):
    """A build step failed; the message is already user-readable."""


def _banner(step: str, title: str) -> None:
    print(f"\n=== [{step}] {title} " + "=" * max(0, 70 - len(step) - len(title)), flush=True)


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def _mb(value: int) -> str:
    return f"{value / 1e6:,.1f} MB"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------ steps
def exe_file_version(exe: Path) -> Optional[str]:
    """The ``FileVersion`` string of ``exe``'s version resource, or None if absent."""
    import pefile

    pe = pefile.PE(str(exe), fast_load=True)
    try:
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )
        for file_info in getattr(pe, "FileInfo", []):
            for entry in file_info if isinstance(file_info, list) else [file_info]:
                for table in getattr(entry, "StringTable", []):
                    value = table.entries.get(b"FileVersion")
                    if value:
                        return value.decode(errors="replace").strip()
    finally:
        pe.close()
    return None


def _check_exe_version(version: str, fail: bool) -> None:
    """The built exe must be the version this build stamps on the zips."""
    stamped = exe_file_version(APP_EXE)
    if stamped is None:
        _problem("the app exe carries no version resource - rebuild it with build.bat", fail)
        return
    expected = make_version_info.version_tuple(version)
    if make_version_info.version_tuple(stamped) != expected:
        _problem(
            f"{APP_EXE.name} is version {stamped}, but this build is {version}: "
            "rebuild it with build.bat (or pass --version)", fail,
        )
    else:
        print(f"version:  {version} (exe FileVersion {stamped})")


def _check_sidecar_fresh(fail: bool) -> None:
    """The frozen sidecar must be newer than the sources it was frozen from."""
    report = BUILD / "renderer-build.json"
    package = ROOT / "renderer" / "glassrenderer"
    if not report.is_file():
        _problem(f"{report.relative_to(ROOT)} not found - run build_renderer.bat", fail)
        return
    sources = [p for p in package.rglob("*") if p.is_file()]
    newest = max((p.stat().st_mtime for p in sources), default=0.0)
    if report.stat().st_mtime < newest:
        newer = max(sources, key=lambda p: p.stat().st_mtime)
        _problem(
            f"{newer.relative_to(ROOT)} changed after the last sidecar build - "
            "run build_renderer.bat", fail,
        )
    else:
        print(f"sidecar build: {report.relative_to(ROOT)} is newer than renderer/glassrenderer")


def _problem(message: str, fail: bool) -> None:
    """Fail a real build, warn in a dry run (where the sidecar may not exist yet)."""
    if fail:
        raise PortableBuildError(message)
    print(f"WARNING: {message}")


def step_preconditions(dry_run: bool, version: str) -> None:
    """Both executables must exist, match ``version`` and be newer than their sources.

    In a dry run every one of these is a warning instead: the point of a dry run is to
    see what *would* ship before the sidecar has been built at all.
    """
    _banner("1/7", "Preconditions")
    if not APP_EXE.is_file():
        raise PortableBuildError(f"{APP_EXE.relative_to(ROOT)} not found - run build.bat first")
    if not RENDERER_EXE.is_file():
        _problem(
            f"{RENDERER_EXE.relative_to(ROOT)} not found - run build_renderer.bat first "
            "(the portable zip ships the frozen sidecar, there is no app-only variant)",
            not dry_run,
        )
    else:
        _check_sidecar_fresh(not dry_run)
    _check_exe_version(version, not dry_run)
    for python in (APP_VENV_PYTHON, RENDERER_VENV_PYTHON):
        if not python.is_file():
            print(f"WARNING: {python} not found; its licence closure will be missing")
    print(f"app exe:  {APP_EXE.relative_to(ROOT)} ({_mb(APP_EXE.stat().st_size)})")
    print(f"sidecar:  {RENDERER_DIR.relative_to(ROOT)} ({_mb(_dir_bytes(RENDERER_DIR))})")


def step_torch_free() -> List[str]:
    """Scan the built app exe; return its archive entry names for the licence closure."""
    _banner("2/7", "Torch-free scan (packaging/verify_torch_free.py)")
    names = verify_torch_free.frozen_toc(APP_EXE)
    offenders = verify_torch_free.forbidden_entries(names)
    if offenders:
        raise PortableBuildError(
            f"{APP_EXE.name} carries {len(offenders)} forbidden entries: " + ", ".join(offenders[:10])
        )
    modules = verify_torch_free.frozen_top_level_modules(names)
    print(f"{len(names)} archive entries, {len(modules)} top-level modules, no torch / OCR heavyweight")
    return names


def step_store(generated: str, include_research_models: bool) -> Dict[str, object]:
    """Re-hash the whole local store and build the manifest (nothing is downloaded)."""
    _banner("3/7", "Model store (packaging/portable_store.py)")
    if not MODELS_DIR.is_dir():
        raise PortableBuildError(f"{MODELS_DIR} does not exist; nothing to ship")
    if include_research_models:
        print("NOTE: packs licensed for research use only (Sugoi) are included.  This bundle is "
              "for personal use on this machine - do not redistribute it.  Use "
              "--no-research-models for a shareable bundle.")
    else:
        print(f"NOTE: --no-research-models: {portable_store.SUGOI_EXCLUSION_REASON}")
    try:
        manifest = portable_store.store_manifest(
            MODELS_DIR, generated, include_research_models=include_research_models
        )
    except portable_store.StoreError as exc:
        raise PortableBuildError(f"the local model store is not shippable: {exc}") from exc
    kinds: Dict[str, int] = {}
    for row in manifest["files"]:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print(f"{len(manifest['files'])} files, {_mb(int(manifest['total_bytes']))}, by kind: {kinds}")
    for pack in manifest["argos_packs"]:
        print(f"  pack {pack['pair']:>8}  priority {pack['priority']:>2}  {pack['directory'] or pack['archive']}")
    for pack in manifest["excluded_packs"]:
        print(f"  NOT shipped {pack['pair']:>8}  {pack['directory'] or pack['archive']}: {pack['reason']}")
    uncovered = manifest["uncovered_pairs"]
    print(f"  uncovered pairs ({len(uncovered)}): {', '.join(uncovered) or 'none'}")
    return manifest


def _license_index(stage: Path, toc_names: Sequence[str], packs: Sequence[Dict[str, object]],
                   excluded: Sequence[Dict[str, object]]) -> Path:
    """Write ``<stage>/licenses/`` from both venvs' metadata and the shipped closures."""
    app_dists = licenses.bundled_distributions(
        verify_torch_free.frozen_top_level_modules(toc_names),
        licenses.collect_distributions(APP_VENV_PYTHON),
    )
    sidecar_dists: List[Dict[str, object]] = []
    if RENDERER_EXE.is_file() and RENDERER_VENV_PYTHON.is_file():
        sidecar_dists = licenses.bundled_distributions(
            verify_torch_free.frozen_top_level_modules(verify_torch_free.frozen_toc(RENDERER_EXE)),
            licenses.collect_distributions(RENDERER_VENV_PYTHON),
        )
    index = licenses.write_license_index(stage, app_dists, sidecar_dists, MODELS_MD, packs, excluded)
    print(f"licences: {len(app_dists)} app + {len(sidecar_dists)} sidecar distributions, "
          f"{len(packs)} translation packs ({len(excluded)} excluded) -> {index.relative_to(stage)}")
    return index


SUGOI_README_SENTENCE = (
    "The Japanese-to-English default is the Sugoi model, which is noticeably better on "
    "manga dialogue; it carries a personal-use licence, so see licenses\\LICENSES.md "
    "before passing this folder on."
)


def _translation_packs_text(manifest: Dict[str, object]) -> str:
    """The README's pack paragraph, built from what the manifest actually ships."""
    packs: Sequence[Dict[str, object]] = manifest["argos_packs"]  # type: ignore[assignment]
    lines = ["Translation packs in this bundle:", ""]
    lines += [
        f"  {pack['pair']} ({pack['directory'] or pack['archive']})" for pack in packs
    ] or ["  (none)"]
    research = manifest["research_models_included"] and any(
        not pack.get("redistributable", True) for pack in packs
    )
    if research:
        lines.append("")
        lines.append(textwrap.fill(SUGOI_README_SENTENCE, width=74))
    return "\n".join(lines)


def render_readme(manifest: Dict[str, object], version: str, date: str) -> str:
    """``packaging/README-portable.txt`` with its build-dependent fields filled in."""
    gigabytes = math.ceil(int(manifest["total_bytes"]) / 1e9)  # type: ignore[arg-type]
    return (
        README_TEMPLATE.read_text(encoding="utf-8")
        .replace("{version}", version)
        .replace("{date}", date)
        .replace("{models_zip_gb}", str(gigabytes))
        .replace("{translation_packs}", _translation_packs_text(manifest))
    )


def step_stage(version: str, date: str, toc_names: Sequence[str],
               manifest: Dict[str, object]) -> Path:
    """Build ``build/portable/GlassTranslate-<version>/`` from scratch."""
    _banner("4/7", "Stage the portable folder")
    stage = STAGE_ROOT / f"GlassTranslate-{version}"
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    stage.mkdir(parents=True)
    shutil.copy2(APP_EXE, stage / APP_EXE.name)
    shutil.copytree(RENDERER_DIR, stage / "renderer")
    (stage / MARKER_NAME).write_text(MARKER_TEXT.format(version=version, date=date), encoding="utf-8")
    (stage / README_TEMPLATE.name).write_text(render_readme(manifest, version, date), encoding="utf-8")
    try:
        _license_index(stage, toc_names, manifest["argos_packs"],  # type: ignore[arg-type]
                       manifest["excluded_packs"])  # type: ignore[arg-type]
    except licenses.LicenceError as exc:
        raise PortableBuildError(str(exc)) from exc
    print(f"staged {stage.relative_to(ROOT)} ({_mb(_dir_bytes(stage))})")
    return stage


def _compression(arcname: str, size: int) -> int:
    already_compressed = arcname.lower().endswith(STORED_SUFFIXES) and size >= STORED_MIN_BYTES
    return zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED


def _write_zip(target: Path, entries: Sequence[Entry]) -> Path:
    """Write ``entries`` (sorted ``(arcname, source)`` pairs) into ``target``, atomically.

    The bytes go to ``<target>.part`` and are renamed into place only once the zip is
    closed, and the previous zip *and* its checksum are removed first: an interrupted
    run must never leave a truncated zip next to a checksum that still looks plausible.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    for stale in (target, _checksum_path(target), part):
        stale.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for arcname, source in entries:
                if source is None:
                    raise PortableBuildError(f"{arcname} has no source file")
                zf.write(source, arcname, compress_type=_compression(arcname, source.stat().st_size))
        os.replace(part, target)
    except BaseException:  # an interrupt must clean up too
        part.unlink(missing_ok=True)
        raise
    return target


def _app_entries(version: str, stage: Optional[Path]) -> List[Entry]:
    """What the portable zip carries: the staged tree, or the plan for a dry run."""
    prefix = f"GlassTranslate-{version}"
    if stage is not None:
        return sorted(
            (f"{prefix}/{p.relative_to(stage).as_posix()}", p) for p in stage.rglob("*") if p.is_file()
        )
    planned: List[Entry] = [
        (f"{prefix}/{APP_EXE.name}", APP_EXE),
        (f"{prefix}/{MARKER_NAME}", None),
        (f"{prefix}/{README_TEMPLATE.name}", None),
        (f"{prefix}/licenses/LICENSES.md", None),
    ]
    planned += [
        (f"{prefix}/renderer/{p.relative_to(RENDERER_DIR).as_posix()}", p)
        for p in RENDERER_DIR.rglob("*")
        if p.is_file()
    ]
    return sorted(planned)


def _models_entries(version: str, manifest: Dict[str, object], manifest_path: Optional[Path]) -> List[Entry]:
    """What the models zip carries, read straight from the source store."""
    prefix = f"GlassTranslate-{version}/models"
    entries: List[Entry] = [
        (f"{prefix}/{row['path']}", MODELS_DIR / str(row["path"]))
        for row in manifest["files"]  # type: ignore[union-attr]
    ]
    entries.append((f"{prefix}/{MANIFEST_NAME}", manifest_path))
    return sorted(entries)


def _listing(title: str, entries: Sequence[Entry]) -> int:
    total = sum(source.stat().st_size for _, source in entries if source is not None)
    generated = sum(1 for _, source in entries if source is None)
    print(f"{title}: {len(entries)} entries, {_mb(total)}" + (f" (+{generated} generated at stage time)" if generated else ""))
    for arcname, source in entries[:5]:
        print(f"  {arcname}" + ("" if source is not None else "   (generated)"))
    if len(entries) > 5:
        print(f"  ... {len(entries) - 5} more")
    return total


def _checksum_path(zip_path: Path) -> Path:
    return zip_path.with_suffix(zip_path.suffix + ".sha256")


def _write_sha256(zip_path: Path) -> Path:
    """``<zip>.sha256`` in ``sha256sum`` format (two spaces before the name)."""
    target = _checksum_path(zip_path)
    target.write_text(f"{_sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")
    return target


def _zip_plan(version: str, stage: Optional[Path], manifest: Dict[str, object],
              manifest_path: Optional[Path], *, single: bool = False) -> List[ZipPlan]:
    """The archives a build writes, as ``(step, label, zip name, entries)`` per archive.

    The pair (default): the portable zip from the staged folder and the models zip straight
    from the store.  ``single``: one ``GlassTranslate-<version>-full-win64.zip`` holding the
    program and the models in exactly the layout the pair unpacks to (one top-level folder,
    ``models/`` with its ``MANIFEST.json`` under it), for a one-download bundle.  ``stage``
    and ``manifest_path`` may be None for a dry-run listing.
    """
    app = _app_entries(version, stage)
    models = _models_entries(version, manifest, manifest_path)
    if single:
        return [("5/7", "full", f"GlassTranslate-{version}-full-win64.zip", sorted(app + models))]
    return [
        ("5/7", "portable", f"GlassTranslate-{version}-portable-win64.zip", app),
        ("6/7", "models", f"GlassTranslate-{version}-models-win64.zip", models),
    ]


def _build_zips(version: str, stage: Path, manifest: Dict[str, object], manifest_path: Path,
                timings: Dict[str, float], *, single: bool = False) -> Dict[str, Path]:
    """Steps (5)-(6): write the planned zips and their ``sha256sum`` sidecars."""
    plan = _zip_plan(version, stage, manifest, manifest_path, single=single)
    zips: Dict[str, Path] = {}
    for step, label, name, entries in plan:
        _banner(step, f"Write dist/{name} (+ .sha256)")
        mark = time.perf_counter()
        _listing(name, entries)
        zips[label] = _write_zip(DIST / name, entries)
        # Right after its own zip lands, so the pair is never mismatched.
        checksum = _write_sha256(zips[label])
        timings[f"zip {label}"] = time.perf_counter() - mark
        print(f"wrote {zips[label].relative_to(ROOT)} ({_mb(zips[label].stat().st_size)}) "
              f"and {checksum.name}")
    return zips


def step_report(version: str, generated: str, sizes: Dict[str, int], timings: Dict[str, float],
                manifest: Dict[str, object], zips: Dict[str, Path], *, single: bool = False) -> None:
    """Print the size / timing table and write build/portable-build.json (the perf doc reads it).

    ``sizes`` and ``zips`` describe the archives that were actually written: the pair, or the
    one full archive of ``--single``, which the report records under ``"single"``.
    """
    _banner("7/7", "Summary")
    for label, value in sizes.items():
        print(f"  {label:<22} {_mb(value):>14}")
    for label, seconds in timings.items():
        print(f"  {label:<22} {seconds:>11.1f} s")
    BUILD.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "version": version,
        "generated": generated,
        "single": single,
        "sizes_bytes": sizes,
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "model_files": len(manifest["files"]),  # type: ignore[arg-type]
        "research_models_included": manifest["research_models_included"],
        "excluded_packs": manifest["excluded_packs"],
        "uncovered_pairs": manifest["uncovered_pairs"],
        "zips": {name: {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path)} for name, path in zips.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------- main
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="verify and list what would be zipped; write no zips")
    parser.add_argument("--stage-only", action="store_true",
                        help="run steps (1)-(4) and stop after staging, leaving the zips alone")
    parser.add_argument("--no-research-models", action="store_true",
                        help="leave out packs licensed for research use only (the Sugoi "
                             "conversion), producing a bundle that may be shared")
    parser.add_argument("--single", action="store_true",
                        help="write one GlassTranslate-<v>-full-win64.zip holding the program and "
                             "the models instead of the pair; same layout")
    parser.add_argument("--version", help="override glasstranslate.__version__")
    args = parser.parse_args(argv)
    if args.single and args.stage_only:
        parser.error("--single cannot be combined with --stage-only: nothing is zipped after staging")
    return args


def _version(override: Optional[str]) -> str:
    version = override or make_version_info.read_package_version()
    if not version:
        raise PortableBuildError("glasstranslate/__init__.py has no __version__")
    return version


def _run(args: argparse.Namespace) -> int:
    version = _version(args.version)
    generated = datetime.date.today().isoformat()
    timings: Dict[str, float] = {}
    started = time.perf_counter()
    step_preconditions(args.dry_run, version)
    mark = time.perf_counter()
    toc_names = step_torch_free()
    timings["verify exe"] = time.perf_counter() - mark
    mark = time.perf_counter()
    manifest = step_store(generated, not args.no_research_models)
    timings["hash store"] = time.perf_counter() - mark
    if args.dry_run:
        _banner("4/7", "Dry run - nothing is written")
        for _step, _label, name, entries in _zip_plan(version, None, manifest, None, single=args.single):
            _listing(name, entries)
        print(f"\ndry run complete in {time.perf_counter() - started:.1f} s")
        return 0
    mark = time.perf_counter()
    stage = step_stage(version, generated, toc_names, manifest)
    manifest_path = STAGE_ROOT / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    timings["stage"] = time.perf_counter() - mark
    if args.stage_only:
        print(f"\n--stage-only: stopped after staging; the zips in {DIST.relative_to(ROOT)} "
              "were left untouched")
        return 0
    zips = _build_zips(version, stage, manifest, manifest_path, timings, single=args.single)
    timings["total"] = time.perf_counter() - started
    sizes: Dict[str, int] = {
        "app exe": APP_EXE.stat().st_size,
        "renderer folder": _dir_bytes(RENDERER_DIR),
        "models folder": int(manifest["total_bytes"]),  # type: ignore[arg-type]
    }
    # One row per archive actually written: "portable zip" + "models zip", or "full zip".
    sizes.update({f"{label} zip": path.stat().st_size for label, path in zips.items()})
    step_report(version, generated, sizes, timings, manifest, zips, single=args.single)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]
    try:
        return _run(parse_args(argv))
    except PortableBuildError as exc:
        print(f"\nPORTABLE BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nPORTABLE BUILD INTERRUPTED", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
