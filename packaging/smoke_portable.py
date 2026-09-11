"""The ``portable`` acceptance run of the offline bundle (docs/plans/2026-09-11-portable-bundle.md § 4).

Driven by ``packaging/smoke_test.py --runs portable --portable-zip <zip> --models-zip <zip>``
(or ``--portable-dir <already unpacked root>``).  Both zips are unpacked into a fresh folder
under ``%TEMP%`` whose name carries a space **and** a non-ASCII character; everything launched
from the unpacked root sees a scrubbed environment: ``PATH`` reduced to the System32 family, no
``PYTHON*`` / ``QT_*`` / ``VIRTUAL_ENV`` / ``QML*`` / ``QSG_*``, ``TEMP``/``TMP`` inside the
sandbox, empty decoy ``LOCALAPPDATA`` / ``APPDATA`` / ``USERPROFILE`` folders that must stay
empty, dead proxies on ``http://127.0.0.1:9`` and - deliberately - **no**
``GLASSTRANSLATE_CONFIG``: the app's own portable config path is what is under test.

Checks (``GT_GPU_TESTS=1`` enables the two that need the GPU and the 9.3 GB model store):

1. ``renderer\\glassrenderer.exe serve --fake`` prints ``READY <port>``, answers ``/health`` and
   one ``/inpaint`` byte-identical outside the mask, and exits on stdin EOF; cold start recorded
   for a first and a second launch, stderr free of any download / URL / proxy line.
2. (GPU) the same sidecar with the real stages and ``--models-dir <root>\\models``: every model
   reaches ``ready`` **and** the server reports ``warm`` with no ``load_error`` (``models.*``
   alone is only a file-presence check) without a download, and one 512x768 job is
   byte-identical outside the mask.
3. ``GlassTranslate.exe`` from the root with ``quality_renderer=auto`` and the ``probeSidecar``
   smoke action: the frozen sidecar is found under the root, the models are ready, the paths of
   the report are under the root, the log names the root's Argos store and carries no download /
   URL / proxy line and the decoy profile folders hold nothing of the app's - only what Windows
   and the graphics driver put there themselves: an ``NVIDIA\\ComputeCache`` beside the exe's
   own profile, the ``AppData\\Local`` + ``AppData\\LocalLow`` skeleton the shell materialises
   under a fresh ``USERPROFILE``, and the OS/driver caches inside it (``D3DSCache``,
   ``NVIDIA\\DXCache``, ``AMD\\DxCache``).  Any *other* file under the skeleton fails the row and
   is named in its detail - the app's own Qt caches belong in ``<root>\\cache``
   (``ui/app.py apply_portable_qt_environment``).  Then ``static``, ``A``, ``B`` and
   ``C`` from ``smoke_test.py`` in the same root, and one launch with ``renderer\\`` renamed away
   (the quick fill must still start cleanly).
4. (GPU) the ``feedPage`` action on ``Examples/before.jpg``: a block's ``clean_patch_serial``
   advanced and the overlay re-converted it.
5. The whole root is moved to another space + non-ASCII path and checks (1) and (3) run again.

No exe is built here and nothing is downloaded; the run only reads the two artefacts.  The
sidecar half of checks (1) and (2) - the loopback client, the inpaint assertions and the
zero-network log scanner - lives in ``packaging/smoke_sidecar.py``.
"""
from __future__ import annotations

import contextlib
import ntpath
import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from smoke_sidecar import (
    RENDERER_DIRNAME,
    check_sidecar_fake,
    check_sidecar_gpu,
    noise_lines,
    read_text,
)
from smoke_test import (
    Results,
    Sandbox,
    _rc_text,
    _secret_leak_checks,
    run_a,
    run_b,
    run_c,
    static_checks,
)

ROOT = Path(__file__).resolve().parent.parent
SANDBOX_PREFIX = "gt portable ü_"  # a space AND a non-ASCII character, on purpose
MOVED_DIRNAME = "moved ünïcode"
DEAD_PROXY = "http://127.0.0.1:9"
DECOYS = {"LOCALAPPDATA": "localappdata-decoy", "APPDATA": "appdata-decoy", "USERPROFILE": "profile-decoy"}
# Top-level folders a *graphics driver* creates under the profile as soon as the app touches
# D3D / DirectML (NVIDIA leaves an empty ``NVIDIA\ComputeCache``).  The driver writes them, not
# the bundle, so they do not break the "the decoys stay empty" rule; matched case-insensitively.
DRIVER_CACHE_DIRS = {"NVIDIA", "NVIDIA Corporation", "AMD", "Intel"}
_DRIVER_CACHE_LOWER = {name.lower() for name in DRIVER_CACHE_DIRS}
APPDATA_DIRNAME = "appdata"  # Windows' own skeleton under a fresh USERPROFILE
# Cache directories Windows and the graphics driver fill by themselves, at any depth under that
# skeleton (D3DSCache, NVIDIA\DXCache, AMD\DxCache, ...).  Tolerated; anything else is the app.
OS_CACHE_DIRS = DRIVER_CACHE_DIRS | {"D3DSCache"}
_OS_CACHE_LOWER = {name.lower() for name in OS_CACHE_DIRS}
# A just-finished launch keeps its directory locked while the exe (and its sidecar) unwind, so
# renaming or moving the root waits for the tree to be free first.
PROCESS_WAIT_S = 120.0
PROCESS_POLL_S = 2.0
MOVE_RETRIES = 3
MOVE_RETRY_S = 5.0
ROOT_EXE_NAMES = ("GlassTranslate.exe", "glassrenderer.exe")
AUTOEXIT_APP_MS = 45_000  # the app quits itself once probeSidecar is done
AUTOEXIT_FEED_MS = 900_000  # cap for the GPU feedPage run

CONFIG_DEFAULTS = {"quality_renderer": "auto"}
APP_CONFIG = {
    "running_on_start": True, "translation_backend": "argos",
    "source_lang": "ja", "target_lang": "en", "quality_renderer": "auto",
}
REFERENCE_PAGE = ROOT / "Examples" / "before.jpg"


# --------------------------------------------------------------------------------- the bundle
def read_version() -> str:
    """``__version__`` of ``glasstranslate/__init__.py``, read without importing the package."""
    text = (ROOT / "glasstranslate" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
    if match is None:
        raise ValueError("no __version__ in glasstranslate/__init__.py")
    return match.group(1)


def expected_root_name() -> str:
    """The single top-level folder both zips must carry."""
    return f"GlassTranslate-{read_version()}"


def zip_top_dir(zip_path: Path) -> str:
    """The one top-level folder of ``zip_path`` (a loose file or a second folder is an error)."""
    tops = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            clean = name.replace("\\", "/").strip("/")
            if not clean:
                continue
            head, _, rest = clean.partition("/")
            if not rest and not name.endswith(("/", "\\")):
                raise ValueError(f"{zip_path.name} has the loose entry {name!r} outside a top-level folder")
            tops.add(head)
    if len(tops) != 1:
        raise ValueError(f"{zip_path.name} has {len(tops)} top-level entries: {sorted(tops)}")
    return tops.pop()


def _safe_target(root: Path, name: str) -> Path:
    """Resolve one zip entry under ``root``; absolute names, drive letters and ``..`` are refused."""
    clean = name.replace("\\", "/")
    drive, _ = ntpath.splitdrive(clean)
    if drive or clean.startswith("/"):
        raise ValueError(f"zip entry is absolute: {name!r}")
    if any(part == ".." for part in clean.split("/")):
        raise ValueError(f"zip entry walks up: {name!r}")
    target = (root / clean).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"zip entry escapes the sandbox: {name!r}")
    return target


def safe_extract(zip_path: Path, dest: Path) -> float:
    """Extract ``zip_path`` into ``dest`` guarding every entry against zip slip; returns seconds."""
    started = time.perf_counter()
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = _safe_target(root, info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    return time.perf_counter() - started


def dir_size_mb(path: Path) -> float:
    if not Path(path).is_dir():
        return 0.0
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return round(total / 1e6, 1)


# ------------------------------------------------------------------------------ the sandbox
def portable_env(sandbox: Path) -> Dict[str, str]:
    """Decoy profile folders (they must stay empty) and dead proxies, applied over the base env."""
    (sandbox / "temp").mkdir(parents=True, exist_ok=True)
    env: Dict[str, str] = {}
    for key, name in DECOYS.items():
        decoy = sandbox / name
        decoy.mkdir(parents=True, exist_ok=True)
        env[key] = str(decoy)
    env["HTTP_PROXY"] = DEAD_PROXY
    env["HTTPS_PROXY"] = DEAD_PROXY
    return env


def portable_sandbox(root: Path, sandbox: Path, *, defaults: Optional[Dict[str, Any]] = None) -> Sandbox:
    """A :class:`~smoke_test.Sandbox` over the unpacked root: no ``GLASSTRANSLATE_CONFIG``, the
    config under ``<root>\\config``, the log under ``<root>\\logs`` and ``quality_renderer=auto``
    merged under every config the shared runs write."""
    return Sandbox(
        Path(root) / "GlassTranslate.exe",
        root=Path(root),
        temp=Path(sandbox) / "temp",
        config=Path(root) / "config" / "config.json",
        config_env=False,
        config_defaults=CONFIG_DEFAULTS if defaults is None else defaults,
        env_overlay=portable_env(Path(sandbox)),
        log_file=Path(root) / "logs" / "glasstranslate.log",
    )


@contextlib.contextmanager
def _prefixed(res: Results, prefix: str) -> Iterator[None]:
    """Label every check added inside with ``prefix`` (the relocated second pass)."""
    old, res.prefix = res.prefix, prefix
    try:
        yield
    finally:
        res.prefix = old


@contextlib.contextmanager
def _phase(res: Results, tag: str, name: str) -> Iterator[None]:
    """Run one phase of the acceptance run; an exception becomes a FAIL row, never a traceback.

    A phase that blows up must not cost the rows the run already collected: the table and the
    ``--json`` file are written by ``smoke_test.main`` *after* :func:`run_portable` returns.
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - the failure is the row
        res.add(tag, name, False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------- processes under the root
def _parse_tasklist(text: str, name: str) -> List[int]:
    """PIDs out of ``tasklist /NH /FO CSV`` output for one image name."""
    pids: List[int] = []
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.strip().split('","')]
        if len(parts) >= 2 and parts[0].lower() == name.lower() and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def _run_tool(cmd: List[str]) -> Optional[str]:
    """stdout of a short console tool, or None when it is missing or failed."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,  # noqa: S603
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _wmic_pids(root: Path) -> Optional[List[int]]:
    """PIDs whose ``ExecutablePath`` lies under ``root``; None when ``wmic`` is unavailable."""
    pattern = str(root).replace("\\", "\\\\").replace("'", "''") + "%"
    text = _run_tool(["wmic", "process", "where", f"ExecutablePath like '{pattern}'", "get", "ProcessId"])
    if text is None:
        return None
    return [int(token) for token in text.split() if token.isdigit()]


def pids_under(root: Path, names: Sequence[str] = ROOT_EXE_NAMES) -> List[int]:
    """Processes running out of ``root``.

    ``wmic`` answers exactly; it is gone on recent Windows builds, where the fallback matches the
    bundle's own image names instead - coarser (a copy elsewhere counts too) but never blind.
    """
    pids = _wmic_pids(root)
    if pids is None:
        pids = []
        for name in names:
            text = _run_tool(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH", "/FO", "CSV"])
            if text:
                pids.extend(_parse_tasklist(text, name))
    return sorted(set(pids))


def wait_for_no_process(root: Path, timeout: float = PROCESS_WAIT_S) -> Tuple[bool, List[int]]:
    """Poll until nothing runs out of ``root``; returns ``(clean, the PIDs last seen)``."""
    deadline = time.perf_counter() + timeout
    pids = pids_under(root)
    while pids and time.perf_counter() < deadline:
        time.sleep(PROCESS_POLL_S)
        pids = pids_under(root)
    return not pids, pids


def _free_root(root: Path, res: Results, tag: str, what: str) -> bool:
    """Wait for the tree, and say so in a row; False means the caller must skip ``what``."""
    clean, pids = wait_for_no_process(root)
    return res.add(tag, f"no process still running from the root before the {what}", clean,
                   "clean" if clean else f"still running: {pids}")


def _move_root(root: Path, target: Path) -> None:
    """``shutil.move`` with retries: an exe that has just exited can still hold its directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[BaseException] = None
    for attempt in range(1, MOVE_RETRIES + 1):
        try:
            shutil.move(str(root), str(target))
            return
        except PermissionError as exc:
            last = exc
            if attempt < MOVE_RETRIES:
                time.sleep(MOVE_RETRY_S)
    raise PermissionError(f"could not move the root after {MOVE_RETRIES} attempts: {last}")


def _under(root: Path, value: Any) -> bool:
    if not value:
        return False
    try:
        Path(str(value)).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


# ------------------------------------------------------------------------------------- the app
def _log_checks(sb: Sandbox, res: Results, tag: str, *, expect_argos: bool) -> None:
    root = sb.dir
    text = read_text(sb.log_file)
    res.add(tag, "app log written under the portable root", bool(text), str(sb.log_file))
    if expect_argos:
        wanted = str(root / "models").lower()
        hits = [line for line in text.splitlines() if "Argos packages in" in line and wanted in line.lower()]
        res.add(tag, 'log has "Argos packages in <root>\\models"', bool(hits),
                (hits[0] if hits else f"no such line for {wanted}")[:200])
    noise = noise_lines(text)
    res.add(tag, "app log has no download/URL/proxy line", not noise, "; ".join(noise[:3])[:200])
    renderer_noise = noise_lines(read_text(root / "logs" / "renderer.log"))
    res.add(tag, "renderer log has no download/URL/proxy line", not renderer_noise,
            "; ".join(renderer_noise[:3])[:200])


def _first_foreign_file(entry: Path) -> Optional[str]:
    """The first file under ``entry`` that is **not** inside an OS/driver cache directory.

    Windows and the graphics driver fill a fresh profile on their own - ``AppData\\Local\\
    D3DSCache``, ``AppData\\Local\\NVIDIA\\DXCache``, ``AppData\\LocalLow\\NVIDIA\\DXCache``,
    ``AppData\\Local\\AMD\\DxCache`` - at any depth under the skeleton they create.  Those
    directory names are tolerated wherever they appear; every other file means the bundle
    itself wrote into the profile.  Returned relative to ``entry``'s parent (the decoy), so the
    check row can name it; None when there is nothing to complain about.
    """
    for path in sorted(entry.rglob("*")):
        if not path.is_file():
            continue
        holders = path.relative_to(entry).parts[:-1]
        if not any(part.lower() in _OS_CACHE_LOWER for part in holders):
            return path.relative_to(entry.parent).as_posix()
    return None


def _classify(entry: Path) -> Tuple[Optional[str], str]:
    """``(why this top-level entry may be ignored, the path that forbids it)``."""
    if not entry.is_dir():
        return None, entry.name
    if entry.name.lower() in _DRIVER_CACHE_LOWER:
        return "driver caches", ""
    if entry.name.lower() == APPDATA_DIRNAME:
        offender = _first_foreign_file(entry)
        return (None, offender) if offender is not None else ("OS/driver caches", "")
    return None, entry.name


def decoy_entries(decoy: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """``(unexpected, ignored by reason)`` for a decoy profile folder.

    ``unexpected`` names what the bundle wrote outside its own folder - a top-level entry, or
    the first offending file under the ``AppData`` skeleton.  A decoy that is not there at all
    reports ``["<missing>"]``.
    """
    if not decoy.is_dir():
        return ["<missing>"], {}
    unexpected: List[str] = []
    ignored: Dict[str, List[str]] = {}
    for entry in sorted(decoy.iterdir(), key=lambda p: p.name.lower()):
        reason, offender = _classify(entry)
        if reason is None:
            unexpected.append(offender)
        else:
            ignored.setdefault(reason, []).append(entry.name)
    return unexpected, ignored


def _decoy_checks(sb: Sandbox, res: Results, tag: str) -> None:
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE"):
        unexpected, ignored = decoy_entries(Path(sb.base_env[key]))
        parts = [", ".join(unexpected)] if unexpected else []
        parts += [f"ignored {reason}: {', '.join(names)}" for reason, names in sorted(ignored.items())]
        res.add(tag, f"decoy {key} stayed empty", not unexpected, "; ".join(parts)[:200])


def check_app(sb: Sandbox, res: Results, tag: str, *, expect_sidecar: bool) -> None:
    """Check (3): one ``probeSidecar`` launch from the portable root and everything it proves."""
    root = sb.dir
    sb.write_config(APP_CONFIG)
    lau = sb.launch(tag, AUTOEXIT_APP_MS, {"GLASSTRANSLATE_SMOKE_ACTIONS": "probeSidecar"})
    res.add(tag, "exe exits 0 from the portable root", lau.returncode == 0 and not lau.killed,
            f"{_rc_text(lau.returncode)} wall={lau.wall_s:.1f}s {lau.error}".strip())
    rep = lau.report
    if not res.add(tag, "smoke report written", rep is not None,
                   lau.report_path.name + (f" missing {lau.error}" if rep is None else "")):
        return
    assert rep is not None
    quality = rep.get("quality") or {}
    action = (rep.get("actions") or {}).get("probeSidecar") or {}
    if expect_sidecar:
        res.add(tag, "actions.probeSidecar.ok", bool(action.get("ok")),
                f"{action.get('seconds')} s, kind={action.get('kind')} {action.get('error')}")
        res.add(tag, 'quality.kind == "frozen"', quality.get("kind") == "frozen", str(quality.get("kind")))
        res.add(tag, "quality.launcher under the portable root", _under(root, quality.get("launcher")),
                str(quality.get("launcher")))
        res.add(tag, "quality.models_ready", bool(quality.get("models_ready")), str(quality.get("hint")))
    else:
        res.add(tag, "actions.probeSidecar.ok is False without renderer\\", action.get("ok") is False,
                str(action.get("error")))
        res.add(tag, "quality.launcher is null without renderer\\", quality.get("launcher") is None,
                str(quality.get("launcher")))
    paths = rep.get("paths") or {}
    for key in ("config", "models_dir", "logs"):
        res.add(tag, f"paths.{key} under the portable root", _under(root, paths.get(key)), str(paths.get(key)))
    hist = [str(s) for s in rep.get("status_history") or []]
    res.add(tag, 'status_history has "OCR: mangaocr on ..."',
            any(s.startswith("OCR: mangaocr on") for s in hist),
            next((s for s in hist if s.startswith("OCR: ")), f"{hist[-3:]}")[:160])
    bad = [s for s in hist if s.startswith(("Engine error", "Error:", "Download failed"))]
    res.add(tag, 'no "Engine error"/"Error:"/"Download failed" status', not bad, "; ".join(bad)[:200])
    _secret_leak_checks(tag, rep, res)
    _log_checks(sb, res, tag, expect_argos=expect_sidecar)
    _decoy_checks(sb, res, tag)


def _check_without_renderer(sb: Sandbox, res: Results, tag: str = "P2") -> None:
    """Check (3), second half: the same launch with ``renderer\\`` renamed away.

    A file of the sidecar tree still held open (an anti-virus scan, Explorer) makes the rename
    fail; that is a failed row, never an exception that costs the rest of the run - and the
    restore is reported the same way so a half-renamed root cannot go unnoticed.
    """
    root = sb.dir
    if not _free_root(root, res, tag, "rename"):
        return
    renderer, hidden = root / RENDERER_DIRNAME, root / (RENDERER_DIRNAME + ".off")
    try:
        renderer.rename(hidden)
    except OSError as exc:
        res.add(tag, "renderer\\ could be renamed away", False, f"{type(exc).__name__}: {exc}")
        return
    res.add(tag, "renderer\\ could be renamed away", True, hidden.name)
    try:
        check_app(sb, res, tag, expect_sidecar=False)
    finally:
        ok, detail = False, f"{hidden.name} is gone"
        if hidden.is_dir():
            try:
                hidden.rename(renderer)
                ok, detail = True, renderer.name
            except OSError as exc:
                detail = f"{type(exc).__name__}: {exc}"
        res.add(tag, "renderer\\ restored", ok, detail)


def check_feed_page(sb: Sandbox, res: Results, tag: str = "P3") -> None:
    """Check (4), ``GT_GPU_TESTS=1``: a real page through OCR, typesetting and the sidecar."""
    if not res.add(tag, "reference page present", REFERENCE_PAGE.is_file(), str(REFERENCE_PAGE)):
        return
    sb.write_config(APP_CONFIG)
    lau = sb.launch(tag, AUTOEXIT_FEED_MS, {
        "GLASSTRANSLATE_SMOKE_ACTIONS": "feedPage",
        "GLASSTRANSLATE_SMOKE_PAGE": str(REFERENCE_PAGE.resolve()),
    })
    info = res.info.setdefault("portable", {})
    info[f"{res.prefix}{tag}_exit_s"] = round(lau.wall_s, 1)
    killed = " - KILLED at the deadline, the app never quit by itself" if lau.killed else ""
    res.add(tag, "exe exits 0 from the portable root", lau.returncode == 0 and not lau.killed,
            f"{_rc_text(lau.returncode)} after {lau.wall_s:.1f} s{killed} {lau.error}".strip())
    rep = lau.report
    if not res.add(tag, "smoke report written", rep is not None,
                   lau.report_path.name + (f" missing {lau.error}" if rep is None else "")):
        return
    assert rep is not None
    act = (rep.get("actions") or {}).get("feedPage") or {}
    res.add(tag, "actions.feedPage.finished", bool(act.get("finished")),
            f"status={act.get('status')} after {act.get('seconds')} s, blocks={act.get('blocks')} "
            f"{act.get('error')}".strip())
    res.add(tag, "actions.feedPage.upgraded >= 1", int(act.get("upgraded") or 0) >= 1, str(act.get("upgraded")))
    overlay = rep.get("overlay") or {}
    res.add(tag, "overlay.patch_upgrades >= 1", int(overlay.get("patch_upgrades") or 0) >= 1, str(overlay))
    hist = [str(s) for s in rep.get("status_history") or []]
    res.add(tag, 'status_history has "Quality renderer: ready on cuda"',
            any("Quality renderer: ready on cuda" in s for s in hist),
            next((s for s in hist if "Quality renderer" in s), "missing")[:160])
    res.info.setdefault("portable", {})["feed_page_s"] = act.get("seconds")
    res.info.setdefault("portable", {})["feed_page_quality"] = act.get("quality_stats")


# ------------------------------------------------------------------------------------ the run
def _prepare_root(args: Any, sandbox: Path, res: Results, info: Dict[str, Any]) -> Optional[Path]:
    """Unpack both zips into ``sandbox`` (or adopt ``--portable-dir``) and return the root."""
    expected = expected_root_name()
    if args.portable_dir is not None:
        root = Path(args.portable_dir).resolve()
        res.add("P0", "portable root exists", root.is_dir(), str(root))
        res.add("P0", f"root is named {expected}", root.name == expected, root.name)
        return root if root.is_dir() else None
    if args.portable_zip is None or args.models_zip is None:
        res.add("P0", "--portable-zip and --models-zip given", False,
                "the portable run needs both zips, or --portable-dir")
        return None
    seconds: Dict[str, float] = {}
    tops = set()
    for label, zip_path in (("portable", args.portable_zip), ("models", args.models_zip)):
        path = Path(zip_path)
        if not res.add("P0", f"{label} zip exists", path.is_file(), str(path)):
            return None
        tops.add(zip_top_dir(path))
        seconds[label] = round(safe_extract(path, sandbox), 1)
    info["unzip_s"] = seconds
    res.add("P0", f"both zips carry one top-level folder named {expected}", tops == {expected},
            ", ".join(sorted(tops)))
    root = sandbox / sorted(tops)[0]
    res.add("P0", "portable root unpacked", root.is_dir(), str(root))
    return root if root.is_dir() else None


def _record_sizes(root: Path, info: Dict[str, Any], key: str = "sizes_mb") -> None:
    info[key] = {
        "root": dir_size_mb(root),
        "renderer": dir_size_mb(root / RENDERER_DIRNAME),
        "models": dir_size_mb(root / "models"),
    }


def _shared_runs(sb: Sandbox, res: Results) -> None:
    """``static``, ``A``, ``B`` and ``C`` of ``smoke_test.py``, from the unpacked root."""
    static_checks(sb.exe, res)
    run_a(sb, res, False)
    run_b(sb, res, False)
    run_c(sb, res, False)


def _pass(root: Path, sandbox: Path, res: Results, info: Dict[str, Any], *, full: bool) -> None:
    """One full pass over an unpacked root (``full`` adds the renamed-renderer and GPU checks)."""
    gpu = os.environ.get("GT_GPU_TESTS") == "1"
    sb = portable_sandbox(root, sandbox)
    info["exe"] = str(sb.exe)
    with _phase(res, "S1", "the fake sidecar checks completed"):
        check_sidecar_fake(sb, res)
    if full and gpu:
        with _phase(res, "S2", "the real sidecar checks completed"):
            check_sidecar_gpu(sb, res)
    with _phase(res, "P1", "the app checks completed"):
        check_app(sb, res, "P1", expect_sidecar=True)
    with _phase(res, "shared", "static/A/B/C completed"):
        _shared_runs(sb, res)
    if not full:
        return
    with _phase(res, "P2", "the renderer-less checks completed"):
        _check_without_renderer(sb, res)
    if gpu:
        with _phase(res, "P3", "the feedPage checks completed"):
            check_feed_page(sb, res)


def run_portable(args: Any, res: Results) -> None:
    """Drive the whole acceptance run; every outcome is a row in ``res``."""
    info: Dict[str, Any] = {}
    res.info["portable"] = info
    sandbox = Path(tempfile.mkdtemp(prefix=SANDBOX_PREFIX))
    info["sandbox"] = str(sandbox)
    info["gpu_checks"] = os.environ.get("GT_GPU_TESTS") == "1"
    print(f"portable sandbox: {sandbox}")
    try:
        try:
            root = _prepare_root(args, sandbox, res, info)
        except Exception as exc:  # noqa: BLE001 - a broken zip is a check row, not a traceback
            res.add("P0", "portable bundle unpacked", False, f"{type(exc).__name__}: {exc}")
            return
        if root is None:
            return
        with _phase(res, "P0", "bundle sizes recorded"):
            _record_sizes(root, info)
        with _phase(res, "P1", "the first pass completed"):
            _pass(root, sandbox, res, info, full=True)
        # The launches of that pass keep the tree locked while they unwind; moving it out from
        # under a still-exiting exe is the one thing that can lose a whole run.
        if not _free_root(root, res, "P5", "move"):
            return
        moved = Path(sandbox) / MOVED_DIRNAME / root.name
        with _phase(res, "P5", "portable root moved to another space + non-ASCII path"):
            _move_root(root, moved)
            res.add("P5", "portable root moved to another space + non-ASCII path", moved.is_dir(), str(moved))
        if not moved.is_dir():
            return
        with _prefixed(res, "M-"), _phase(res, "P1", "the relocated pass completed"):
            _pass(moved, sandbox, res, info, full=False)
    finally:
        if getattr(args, "keep", False):
            print(f"kept: {sandbox}")
        else:
            shutil.rmtree(sandbox, ignore_errors=True)
