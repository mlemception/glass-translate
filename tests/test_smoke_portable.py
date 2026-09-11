"""Unit checks for the ``portable`` smoke run (``packaging/smoke_portable.py``).

Only the pure pieces are exercised - the sandbox environment, zip extraction and its zip-slip
refusal, the top-level folder rule, the log scanner, the config merge, the phase guards and the
sidecar readiness rule of ``packaging/smoke_sidecar.py``.  **No exe is launched here**: the run
itself needs the two built zips and is driven by ``packaging/smoke_test.py --runs portable``.

``packaging/`` is not a Python package (an ``__init__.py`` there would shadow the PyPI
``packaging`` distribution), so the scripts are imported off sys.path like ``demo/``'s.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

import smoke_portable as SP  # noqa: E402
import smoke_sidecar as SS  # noqa: E402
import smoke_test as ST  # noqa: E402


def _zip(path: Path, entries: Dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def _sandbox(tmp_path: Path, version: str = "9.9.9"):
    sandbox = tmp_path / "gt portable ü_test"
    root = sandbox / f"GlassTranslate-{version}"
    root.mkdir(parents=True)
    return sandbox, root, SP.portable_sandbox(root, sandbox)


# ------------------------------------------------------------------------------ environment
def test_portable_env_has_decoys_proxies_and_no_config_variable(tmp_path: Path) -> None:
    sandbox, root, sb = _sandbox(tmp_path)
    env = sb.base_env
    assert "GLASSTRANSLATE_CONFIG" not in env  # the portable config path is what is under test
    assert "NO_PROXY" not in env
    assert env["HTTP_PROXY"] == env["HTTPS_PROXY"] == SP.DEAD_PROXY
    assert not [k for k in env if k.upper().startswith(ST.FORBIDDEN_ENV_PREFIXES)]
    for key in ("LOCALAPPDATA", "APPDATA", "USERPROFILE"):
        decoy = Path(env[key])
        assert decoy.is_dir() and decoy.parent == sandbox and not list(decoy.iterdir())
    assert Path(env["TEMP"]) == Path(env["TMP"]) == sandbox / "temp"
    assert sb.exe == root / "GlassTranslate.exe"
    assert sb.config == root / "config" / "config.json"
    assert sb.log_file == root / "logs" / "glasstranslate.log"


def test_the_default_sandbox_is_unchanged(tmp_path: Path) -> None:
    """Runs static/A/B/C/D must behave exactly as before the portable refactor."""
    exe = tmp_path / "GlassTranslate.exe"
    exe.write_bytes(b"MZ")
    sb = ST.Sandbox(exe)
    try:
        assert sb.exe == sb.dir / "GlassTranslate.exe" and sb.exe.exists()
        assert sb.config == sb.dir / "config.json"
        assert sb.base_env["GLASSTRANSLATE_CONFIG"] == str(sb.config)
        assert sb.base_env["TEMP"] == sb.base_env["TMP"] == str(sb.dir)
        assert sb.log_file == ST.LOG_FILE
        sb.write_config({"running_on_start": True})
        assert json.loads(sb.config.read_text(encoding="utf-8")) == {"running_on_start": True}
    finally:
        sb.cleanup()


def test_config_defaults_are_merged_under_every_write(tmp_path: Path) -> None:
    _, _, sb = _sandbox(tmp_path)
    sb.write_config({"running_on_start": True})
    assert json.loads(sb.config.read_text(encoding="utf-8")) == {
        "quality_renderer": "auto", "running_on_start": True
    }
    sb.write_config({"quality_renderer": "off"})  # an explicit override beats the default
    assert json.loads(sb.config.read_text(encoding="utf-8")) == {"quality_renderer": "off"}


# ---------------------------------------------------------------------------------- unzipping
def test_extraction_into_a_space_and_non_ascii_path(tmp_path: Path) -> None:
    zip_path = _zip(tmp_path / "bundle.zip", {
        "GlassTranslate-9.9.9/GlassTranslate.exe": b"MZ",
        "GlassTranslate-9.9.9/renderer/glassrenderer.exe": b"MZ2",
    })
    dest = tmp_path / "gt portable ü_1"
    seconds = SP.safe_extract(zip_path, dest)
    assert seconds >= 0.0
    assert (dest / "GlassTranslate-9.9.9" / "renderer" / "glassrenderer.exe").read_bytes() == b"MZ2"


@pytest.mark.parametrize("name", ["../escape.txt", "sub/../../escape.txt", "/abs.txt", "C:/abs.txt"])
def test_zip_slip_entries_are_refused(tmp_path: Path, name: str) -> None:
    zip_path = _zip(tmp_path / "evil.zip", {name: b"pwned"})
    dest = tmp_path / "out ü"
    with pytest.raises(ValueError):
        SP.safe_extract(zip_path, dest)
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "abs.txt").exists()


def test_zip_top_dir_requires_exactly_one_top_level_folder(tmp_path: Path) -> None:
    one = _zip(tmp_path / "one.zip", {"GlassTranslate-9.9.9/x": b"1", "GlassTranslate-9.9.9/y/z": b"2"})
    assert SP.zip_top_dir(one) == "GlassTranslate-9.9.9"
    two = _zip(tmp_path / "two.zip", {"A/x": b"1", "B/y": b"2"})
    with pytest.raises(ValueError):
        SP.zip_top_dir(two)
    loose = _zip(tmp_path / "loose.zip", {"x.txt": b"1"})
    with pytest.raises(ValueError):
        SP.zip_top_dir(loose)


def test_expected_root_name_follows_the_package_version() -> None:
    from glasstranslate import __version__

    assert SP.expected_root_name() == f"GlassTranslate-{__version__}"


# ------------------------------------------------------------------------------- log scanning
def test_noise_lines_flag_downloads_urls_and_proxies() -> None:
    clean = "INFO Argos packages in C:\\root\\models: ['ja_en']\nINFO OCR: mangaocr on dml"
    assert SS.noise_lines(clean) == []
    noisy = "\n".join([
        clean,
        "INFO Downloading manga-ocr models",
        "WARNING urlopen error [Errno 11001]",
        "INFO fetching https://example.invalid/x",
        "ERROR HTTPSConnectionPool(host='hf', port=443)",
        "INFO using proxy 127.0.0.1:9",
        "INFO argospm install ja_en",
        "INFO GET huggingface.co/api",
    ])
    assert len(SS.noise_lines(noisy)) == 7


# --------------------------------------------------------------------- never lose a run's rows
def test_phase_turns_an_exception_into_a_fail_row() -> None:
    res = ST.Results()
    with SP._phase(res, "P1", "the app checks completed"):
        raise RuntimeError("boom")
    assert len(res.checks) == 1 and not res.ok
    assert res.checks[0].run == "P1" and "RuntimeError: boom" in res.checks[0].detail
    with SP._phase(res, "P1", "the app checks completed"):
        pass  # a phase that worked adds no row of its own
    assert len(res.checks) == 1


def test_run_portable_never_raises_and_keeps_the_rows_it_already_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase that explodes must not cost the table: smoke_test.main prints it afterwards."""
    root = tmp_path / "GlassTranslate-9.9.9"
    root.mkdir()

    def fake_prepare(args, sandbox, res, info):
        res.add("P0", "portable root unpacked", True, str(root))
        return root

    def boom(*args, **kwargs):
        raise PermissionError("the exe still holds the tree")

    monkeypatch.setattr(SP, "_prepare_root", fake_prepare)
    monkeypatch.setattr(SP, "_record_sizes", lambda *a, **k: None)
    monkeypatch.setattr(SP, "wait_for_no_process", lambda r, timeout=0.0: (True, []))
    monkeypatch.setattr(SP, "_pass", boom)

    res = ST.Results()
    SP.run_portable(_args(), res)  # must not raise

    assert not res.ok
    assert any(c.ok and c.name == "portable root unpacked" for c in res.checks)
    assert any("the exe still holds the tree" in c.detail for c in res.checks)


# ------------------------------------------------------------------- processes under the root
def test_parse_tasklist_reads_the_pids_of_one_image() -> None:
    text = ('"GlassTranslate.exe","4242","Console","1","120,000 K"\n'
            '"GlassTranslate.exe","4243","Console","1","98,000 K"\n'
            "INFO: No tasks are running which match the specified criteria.\n")
    assert SP._parse_tasklist(text, "GlassTranslate.exe") == [4242, 4243]
    assert SP._parse_tasklist(text, "glassrenderer.exe") == []


def test_wait_for_no_process_polls_until_the_root_is_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = iter([[1234, 5678], [5678], []])
    monkeypatch.setattr(SP, "pids_under", lambda root, names=None: next(seen))
    monkeypatch.setattr(SP, "PROCESS_POLL_S", 0.0)
    assert SP.wait_for_no_process(tmp_path, timeout=5.0) == (True, [])


def test_wait_for_no_process_gives_up_and_names_the_pids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SP, "pids_under", lambda root, names=None: [4242])
    monkeypatch.setattr(SP, "PROCESS_POLL_S", 0.0)
    assert SP.wait_for_no_process(tmp_path, timeout=0.05) == (False, [4242])


def test_move_root_retries_a_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []

    def flaky(src: str, dst: str) -> None:
        calls.append(1)
        if len(calls) < SP.MOVE_RETRIES:
            raise PermissionError("the exe still holds it")
        Path(dst).mkdir(parents=True)

    monkeypatch.setattr(SP.shutil, "move", flaky)
    monkeypatch.setattr(SP, "MOVE_RETRY_S", 0.0)
    SP._move_root(tmp_path / "root", tmp_path / "moved ünïcode" / "root")
    assert len(calls) == SP.MOVE_RETRIES


def test_move_root_gives_up_after_the_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def always(src: str, dst: str) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(SP.shutil, "move", always)
    monkeypatch.setattr(SP, "MOVE_RETRY_S", 0.0)
    with pytest.raises(PermissionError):
        SP._move_root(tmp_path / "root", tmp_path / "moved" / "root")


# ------------------------------------------------------------------------------ decoy folders
def test_decoy_entries_ignore_driver_caches_but_not_app_folders(tmp_path: Path) -> None:
    """The NVIDIA driver drops an empty ``NVIDIA\\ComputeCache`` in the profile as soon as the
    app touches DirectML; that is the driver writing, not the bundle."""
    decoy = tmp_path / "appdata-decoy"
    (decoy / "NVIDIA" / "ComputeCache").mkdir(parents=True)
    (decoy / "amd").mkdir()  # the name match is case-insensitive
    assert SP.decoy_entries(decoy) == ([], {"driver caches": ["amd", "NVIDIA"]})
    (decoy / "GlassTranslate").mkdir()  # anything the app itself wrote still fails the row
    unexpected, ignored = SP.decoy_entries(decoy)
    assert unexpected == ["GlassTranslate"] and ignored == {"driver caches": ["amd", "NVIDIA"]}


def _file(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_decoy_entries_ignore_the_empty_appdata_skeleton(tmp_path: Path) -> None:
    """Windows materialises ``AppData\\Local`` + ``AppData\\LocalLow`` under any fresh
    USERPROFILE; empty, they are the shell's doing."""
    decoy = tmp_path / "profile-decoy"
    (decoy / "AppData" / "Local").mkdir(parents=True)
    (decoy / "AppData" / "LocalLow").mkdir()
    assert SP.decoy_entries(decoy) == ([], {"OS/driver caches": ["AppData"]})


def test_decoy_entries_ignore_os_and_driver_caches_at_any_depth(tmp_path: Path) -> None:
    """D3DSCache and the drivers' DXCache fill the skeleton on their own."""
    decoy = tmp_path / "profile-decoy"
    _file(decoy / "AppData" / "Local" / "D3DSCache" / "a1b2" / "0dd.idx")
    _file(decoy / "AppData" / "Local" / "NVIDIA" / "DXCache" / "shader.bin")
    _file(decoy / "AppData" / "Local" / "AMD" / "DxCache" / "shader.bin")
    _file(decoy / "AppData" / "locallow" / "NVIDIA" / "DXCache" / "shader.bin")
    assert SP.decoy_entries(decoy) == ([], {"OS/driver caches": ["AppData"]})


def test_decoy_entries_fail_on_the_apps_own_qt_cache_and_name_it(tmp_path: Path) -> None:
    """The bug this rule was written for: Qt's QML cache under the profile is the app writing."""
    decoy = tmp_path / "profile-decoy"
    _file(decoy / "AppData" / "Local" / "D3DSCache" / "0dd.idx")  # tolerated
    _file(decoy / "AppData" / "Local" / "GlassTranslate" / "cache" / "qmlcache" / "x.qmlc")
    unexpected, ignored = SP.decoy_entries(decoy)
    assert unexpected == ["AppData/Local/GlassTranslate/cache/qmlcache/x.qmlc"] and ignored == {}


def test_decoy_entries_reports_a_missing_decoy(tmp_path: Path) -> None:
    assert SP.decoy_entries(tmp_path / "never-created") == (["<missing>"], {})


# ------------------------------------------------------------------------- sidecar readiness
@pytest.mark.parametrize("health, ready", [
    # Loaded, warm, no error: the only state a real job may be sent in.
    ({"models": {"lama": "ready", "sdxl": "ready"}, "warm": True, "load_error": None}, True),
    # A model still loading.
    ({"models": {"lama": "ready", "sdxl": "loading"}, "warm": True, "load_error": None}, False),
    # The broken-build case: models.* is a file-presence check, so it reads all-"ready" at once;
    # only ``warm`` proves the stages really loaded.
    ({"models": {"lama": "ready", "sdxl": "ready"}, "warm": False, "load_error": None}, False),
    # A load error outweighs everything else.
    ({"models": {"lama": "ready", "sdxl": "ready"}, "warm": True, "load_error": "OOM"}, False),
    # No models map at all (an older or stub server) is never "ready" for a real job.
    ({"warm": True, "load_error": None}, False),
])
def test_health_ready_needs_every_model_warm_and_no_load_error(health: Dict, ready: bool) -> None:
    assert SS.health_ready(health) is ready


def test_the_gpu_exit_budget_is_longer_than_the_fake_one() -> None:
    """A real job's CUDA teardown takes 14-19 s; fake mode has nothing to tear down."""
    assert SS.STDIN_EXIT_S == 10.0 and SS.GPU_EXIT_S >= 45.0


# ------------------------------------------------------------------------ inpaint assertions
def test_png_round_trip_preserves_the_panel() -> None:
    image, _ = SS.synthetic_panel(32, 24)
    assert np.array_equal(SS.decode_png(SS.png_b64(image)), image)


def test_same_outside_mask_detects_a_changed_keep_pixel() -> None:
    image, mask = SS.synthetic_panel(64, 48)
    assert image.shape == (48, 64, 3) and mask.shape == (48, 64) and mask.max() == 255
    out = image.copy()
    assert SS.same_outside_mask(image, out, mask)
    out[mask > 0] = 0  # the fill region is free to change
    assert SS.same_outside_mask(image, out, mask)
    ys, xs = np.nonzero(mask == 0)
    out[ys[0], xs[0]] = (1, 2, 3)  # a kept pixel is not
    assert not SS.same_outside_mask(image, out, mask)


# ------------------------------------------------------------------------------ root discovery
def _args(**kwargs) -> SimpleNamespace:
    base = {"portable_zip": None, "models_zip": None, "portable_dir": None, "keep": False}
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_prepare_root_unpacks_both_zips_into_the_sandbox(tmp_path: Path) -> None:
    from glasstranslate import __version__

    name = f"GlassTranslate-{__version__}"
    portable = _zip(tmp_path / "p.zip", {f"{name}/GlassTranslate.exe": b"MZ",
                                         f"{name}/renderer/glassrenderer.exe": b"MZ"})
    models = _zip(tmp_path / "m.zip", {f"{name}/models/MANIFEST.json": b"{}"})
    sandbox = tmp_path / "gt portable ü_prep"
    res, info = ST.Results(), {}
    root = SP._prepare_root(_args(portable_zip=portable, models_zip=models), sandbox, res, info)
    assert root == sandbox / name and root.is_dir()
    assert (root / "models" / "MANIFEST.json").is_file()
    assert set(info["unzip_s"]) == {"portable", "models"} and res.ok


def test_prepare_root_reports_a_wrong_top_level_name(tmp_path: Path) -> None:
    zipped = _zip(tmp_path / "p.zip", {"GlassTranslate-0.0.0/GlassTranslate.exe": b"MZ"})
    res, info = ST.Results(), {}
    root = SP._prepare_root(_args(portable_zip=zipped, models_zip=zipped),
                            tmp_path / "gt portable ü_bad", res, info)
    assert root is not None and not res.ok
    assert any(not c.ok and "top-level folder" in c.name for c in res.checks)


def test_prepare_root_needs_both_zips(tmp_path: Path) -> None:
    res, info = ST.Results(), {}
    assert SP._prepare_root(_args(), tmp_path / "gt portable ü_none", res, info) is None
    assert not res.ok


# --------------------------------------------------------------------------------------- CLI
def test_cli_accepts_the_portable_options() -> None:
    args = ST.parse_args(["--runs", "portable", "--portable-zip", "a.zip", "--models-zip", "b.zip"])
    assert args.runs == "portable"
    assert args.portable_zip == Path("a.zip") and args.models_zip == Path("b.zip")
    assert args.portable_dir is None
    moved = ST.parse_args(["--runs", "portable", "--portable-dir", "r"])
    assert moved.portable_dir == Path("r")
