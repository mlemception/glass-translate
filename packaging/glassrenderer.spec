# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the frozen quality-renderer sidecar (onedir ``renderer/``).

Output: ``<distpath>/renderer/glassrenderer.exe`` + ``<distpath>/renderer/_internal/``.
The app spawns it as ``renderer\\glassrenderer.exe serve --models-dir DIR [--fake]``
(see ``renderer/PROTOCOL.md``); ``console=True`` because the protocol needs a real
stdout (the single ``READY <port>`` line) and a real stdin (EOF = the parent died).
``console=False`` makes both ``None`` and kills the protocol.

Run it through ``build_renderer.py`` (which measures, tees the log and self-checks the
result), or by hand::

    set PYTHONUTF8=1 && renderer\\.venv\\Scripts\\python -m PyInstaller --noconfirm --clean ^
        --distpath dist --workpath build/renderer packaging\\glassrenderer.spec

Every choice below is measured on the probe builds recorded in the R1 freezing
research; the comment next to each one is the evidence it rests on.

Environment knobs (specs take no CLI arguments):

* ``GT_RENDERER_ENTRY``  - entry script (default ``packaging/glassrenderer_entry.py``)
* ``GT_RENDERER_SRC``    - directory holding the ``glassrenderer`` package (default ``<root>/renderer``)
* ``GT_RENDERER_PRUNE``  - ``0`` -> collect everything (the baseline measurement); default ``1``
* ``GT_BUILD_STRICT``    - ``0`` -> a pefile closure warning warns instead of aborting

``GT_RENDERER_DIST`` / ``GT_RENDERER_WORK`` are read by ``build_renderer.py``: a spec
cannot set ``--distpath`` / ``--workpath`` itself.
"""
import os
import re
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

SPEC_DIR = os.path.abspath(SPECPATH)  # noqa: F821 - injected by PyInstaller
ROOT = os.path.dirname(SPEC_DIR)
SRC = os.path.abspath(os.environ.get("GT_RENDERER_SRC") or os.path.join(ROOT, "renderer"))
ENTRY = os.path.abspath(
    os.environ.get("GT_RENDERER_ENTRY") or os.path.join(SPEC_DIR, "glassrenderer_entry.py")
)
PRUNE = os.environ.get("GT_RENDERER_PRUNE", "1") != "0"
STRICT = os.environ.get("GT_BUILD_STRICT", "1") != "0"

if not os.path.isfile(ENTRY):
    raise SystemExit(f"[spec] entry script not found: {ENTRY}")
if not os.path.isdir(os.path.join(SRC, "glassrenderer")):
    raise SystemExit(f"[spec] glassrenderer package not found under: {SRC}")
# ``pathex`` is not enough: the ``collect_*`` helpers below ``__import__`` the package
# they are asked about, and without this the build only logs
# "skipping data collection for module 'glassrenderer' as it is not a package".
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# --------------------------------------------------------------------------- metadata
# Distributions whose *version* is read at import time through importlib.metadata
# (diffusers / transformers / accelerate / peft / huggingface_hub all do, and torch's
# own torch.torch_version).  Without the .dist-info in the bundle the import dies with
# "PackageNotFoundError: No package metadata was found for <name>", usually re-wrapped
# by diffusers as "Failed to import diffusers.X because of ...".  hook-transformers
# already copies everything in transformers' dependency table; this is the remainder.
# Hyphenated distribution names, not module names (huggingface-hub, not huggingface_hub).
METADATA = [
    "torch", "torchvision", "diffusers", "transformers", "tokenizers", "safetensors",
    "accelerate", "peft", "huggingface-hub", "numpy", "pillow", "filelock", "fsspec",
    "packaging", "pyyaml", "regex", "requests", "tqdm", "typing-extensions", "psutil",
    "sympy", "importlib-metadata", "opencv-python-headless",
]
metadata_datas = []
for _dist in METADATA:
    try:
        metadata_datas += copy_metadata(_dist)
    except Exception as exc:  # a distribution that is not installed is not a build error
        print(f"[spec] no metadata for {_dist}: {exc}")

# --------------------------------------------------------------------------- hidden imports
# diffusers and transformers are both `_LazyModule` packages: the import structure is a
# dict consulted at attribute access, so PyInstaller's static analysis sees almost
# nothing, and there is no contrib hook for diffusers / huggingface_hub / accelerate /
# peft.  diffusers is only ~55 MB - collect it whole.
#
# transformers 5.x MUST be collected whole (measured: a CLIP-only collection makes the
# real run die with "Could not import module 'BloomPreTrainedModel'. Are this object's
# requirements defined correctly?" out of diffusers.loaders.peft).  Two reasons:
#   * transformers/__init__.py builds its import map with define_import_structure(),
#     which os.scandir()s transformers/models on the real filesystem - a partial tree
#     yields a partial map and one missing name poisons an unrelated import;
#   * importing the pipeline class alone already pulls 248 transformers.models.*
#     modules (the eager fast-image-processor registry), so "only CLIP" is not what runs.
# Cost: 46.1 MiB on disk, 2637 PYZ modules.  hook-transformers sets
# module_collection_mode="pyz+py" so the .py sources land on disk for that scan - never
# override it to "pyz".
HIDDEN = []
HIDDEN += collect_submodules("diffusers")
HIDDEN += collect_submodules("glassrenderer")
HIDDEN += collect_submodules("transformers")
HIDDEN += collect_submodules("huggingface_hub")
HIDDEN += collect_submodules("torchvision")
HIDDEN += ["safetensors.torch", "tokenizers", "accelerate", "peft"]
# `import torch` on Windows resolves torch/lib itself (os.add_dll_directory, then
# LoadLibraryExW over glob(torch/lib/*.dll)), so no runtime hook is needed; torch.jit
# re-parses the .py sources of what it loads, which hook-torch's "pyz+py" mode provides.
HIDDEN += ["torch", "torch.jit", "torch.fft", "torch.cuda"]

# torchvision 0.29 renamed its operator libraries to `_C_stable.pyd` / `image_stable.pyd`,
# so hook-torchvision's hidden imports ("torchvision._C", "torchvision.image") miss and
# NOTHING collects them - the build only logs `Hidden import "torchvision._C" not found!`.
# Measured failure without this line: the first real job dies with
# `RuntimeError: operator torchvision::nms does not exist`, which surfaces five frames
# later as the BloomPreTrainedModel message above, because peft.utils.constants is the
# first thing to ask transformers for a lazily mapped symbol.  Collects eight files
# (+15.6 MiB): _C_stable, image_stable, jpeg8, libpng16, libwebp, libsharpyuv,
# nvjpeg64_13, zlib.
BINARIES = collect_dynamic_libs("torchvision")

DATAS = list(metadata_datas)
# glassrenderer/models.py resolves the pinned table with Path(__file__).with_name(
# "models.json"), so it must be a real file at _internal/glassrenderer/models.json -
# every real run re-hashes the LaMa checkpoint and the SDXL safetensors against it.
DATAS += [(os.path.join(SRC, "glassrenderer", "models.json"), "glassrenderer")]
DATAS += collect_data_files("diffusers")      # scheduler / pipeline json configs
DATAS += collect_data_files("transformers")   # tokenizer json, dependency tables

# --------------------------------------------------------------------------- excludes
# Checked against the real-mode run's sys.modules: networkx never reaches it; sympy,
# scipy, torchvision and psutil do.
EXCLUDES = [
    "networkx", "lpips", "matplotlib", "pandas", "IPython", "notebook",
    "pytest", "_pytest", "PyInstaller", "pip", "setuptools", "pkg_resources",
    "tkinter", "torchaudio", "torchtext", "tensorboard", "wandb", "onnx",
    "triton",   # not installed on Windows; torch guards every use behind a try/except
    "hf_xet",   # 10 MB Rust uploader; the sidecar never touches the Hub (HF_HUB_OFFLINE=1)
]
# NEVER exclude: sympy / mpmath (torch/__init__ imports torch.fx.experimental.sym_node,
# which imports sympy at module scope), scipy (diffusers' schedulers import it and its
# .pyds really are loaded in the real run), torchvision (transformers' fast image
# processors), filelock, jinja2, fsspec (reached by huggingface_hub), torch._dynamo /
# torch.fx, unittest (reached from torch.testing internals).  omegaconf and antlr4 are
# NOT needed: diffusers 0.40's from_single_file reads the original config with plain
# PyYAML, and neither package ever reached sys.modules in the real run.

a = Analysis(  # noqa: F821
    [ENTRY],
    pathex=[SRC],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    # optimize=1 strips asserts and keeps docstrings.  optimize=2 MUST NOT be used:
    # transformers 5.x validates docstrings at import, so `python -OO -c "import peft"`
    # already fails with "No 'Args' or 'Parameters' section is found in the docstring of
    # 'BaseModelOutputWithPastAndCrossAttentions'".
    optimize=1,
)

# --------------------------------------------------------------------------- pruning
# Evidence: pefile over torch/lib.  The load-time closure of torch_python.dll is
#   torch_python -> shm, torch_cpu, torch_cuda
#   torch_cpu    -> c10, cupti64_2025.3.0, libiomp5md, uv
#   torch_cuda   -> c10, c10_cuda, cublas64_13, cublasLt64_13, cudnn64_9, cufft64_12,
#                   cusolver64_12, cusparse64_12, nvrtc64_130_0, torch_cpu
#   cublas64_13  -> cublasLt64_13 ; cusolver64_12 -> cublas, cublasLt, cusparse
#   cusparse64_12-> nvJitLink_130_0 ; caffe2_nvrtc -> nvrtc64_130_0
# No DLL in torch/lib has a delay-load import; everything else there is either dlopen'd
# by name (the cuDNN 9 engine set, loaded by cudnn64_9.dll) or never referenced at all.
# torch/__init__.py *globs* torch/lib/*.dll and LoadLibraryEx's each hit, so a pruned DLL
# simply drops out of the glob: it never produces an import-time failure, only a use-time
# one - which is why "which DLLs does the process map?" is useless as evidence (a real
# job maps all 39) and why the pefile closure check below exists.
# Each entry was removed from a working bundle and the full real job (LaMa TorchScript
# with torch.fft.rfftn, then SDXL + union ControlNet) re-run: green, byte-identical
# outside the mask, same warm time (1429.6 ms vs 1434.6 ms).
TORCH_LIB_DROP = {
    "cudnn_adv64_9.dll",          # 106.6 MB - RNN / legacy-attention engines; SDXL uses
    #                               SDPA and LaMa conv+FFT.  Not in the load-time closure.
    "cusolvermg64_12.dll",        #  95.4 MB - multi-GPU LAPACK; nothing in torch/lib imports it
    "nvrtc64_130_0.alt.dll",      #  91.0 MB - alternate nvrtc for pre-CUDA-13 drivers;
    #                               torch_cuda links the non-.alt copy
    "curand64_10.dll",            #  58.9 MB - torch's CUDA RNG is Philox inside torch_cuda
    #                               (the SDXL stage's torch.Generator(device="cuda") is green)
    "nvperf_host.dll",            #  27.8 MB - CUPTI's profiling host, only under a profiler
    "cufftw64_12.dll",            #   0.2 MB - FFTW compat shim; torch calls cuFFT directly
    "libiompstubs5md.dll",        #  0.04 MB - OpenMP stubs, superseded by libiomp5md.dll
    "zlibwapi.dll",               #  0.09 MB - cuDNN-7-era zlib shim, unused by cuDNN 9
}
# MUST STAY - each with the measurement that proves it:
#   cudnn_engines_precompiled64_9.dll (221.8 MB) - removing it breaks EVERY convolution:
#     "RuntimeError: GET was unable to find an engine to execute this computation" inside
#     the LaMa TorchScript interpreter, on its first conv (probe run real_noprecomp).
#     The single biggest prunable-looking item in the bundle, and it is not prunable.
#   cupti64_2025.3.0.dll (2.1 MB) - a LOAD-TIME import of torch_cpu.dll (pefile).  The
#     advice to prune it that circulates on the web is wrong: torch_cpu.dll would then
#     fail to load with WinError 126 naming torch_cpu, not the missing file.
#   cudnn_graph / ops / cnn / heuristic / engines_runtime_compiled / engines_tensor_ir /
#     ext / cudnn64_9 - the cuDNN 9 library split; cudnn64_9.dll dlopens them by name.
#   cublasLt64_13, cufft64_12 (LaMa's FFC blocks call torch.fft.rfftn/irfftn),
#     cusparse64_12, cusolver64_12, nvJitLink_130_0 (load-time import of cusparse),
#     nvrtc64_130_0, nvrtc-builtins64_130 (nvrtc needs it to compile anything),
#     cublas64_13, cudart64_13, c10, c10_cuda, uv, shm, torch, torch_global_deps,
#     caffe2_nvrtc, nvToolsExt64_1, libiomp5md.
# hook-torch already drops **/*.h, *.hpp, *.cuh, *.lib, *.cpp, *.pyi and *.cmake from
# torch's datas, so torch/include (65 MB) and torch/lib/*.lib (44 MB) never reach the TOC;
# the patterns below are the remainder - tools and viewers nothing on the inference path
# opens.  NOTE: torch/utils/data/datapipes must NOT be listed - torch.utils.data.dataloader
# imports torch.utils.data.datapipes.iter.sharding at module scope, and dropping its
# source broke `import torch` outright in an early version of this spec.
TORCH_DATA_DROP = re.compile(
    r"^torch/(include/|test/|_C/|bin/|share/|utils/model_dump/)"  # bin/protoc.exe 2.8 MB,
    #                                             utils/model_dump 0.1 MB (an HTML viewer)
    r"|^torch/.*\.(h|hpp|cuh|lib|cpp|cmake|pyi)$",
    re.I,
)
DROP = re.compile(
    r"(^|/)(opencv_videoio_ffmpeg\d+_64\.dll"  # 30.9 MB - the sidecar only calls resize /
    #                                            cvtColor / Canny, never VideoCapture
    r"|cv2/data/.*"                            # haarcascades, never opened
    r"|torchgen/.*"                            # codegen templates, build-time only
    r"|.*\.(lib|pdb|exp)"                      # import libraries and debug info
    r")$",
    re.I,
)


def keep(entry):
    """Allowlist/denylist filter for one TOC entry ``(dest, src, typecode)``."""
    dest = entry[0].replace("\\", "/")
    base = os.path.basename(dest).lower()
    if dest.lower().startswith("torch/lib/") and base in TORCH_LIB_DROP:
        return False
    if TORCH_DATA_DROP.match(dest):
        return False
    return not DROP.search(dest)


def _mb(entries):
    return sum(os.path.getsize(s) for _, s, _ in entries if os.path.isfile(s)) / 1e6


def _report_dropped(dropped):
    """Print the 25 biggest pruned groups, largest first."""
    by_group = {}
    for dest, src, _typecode in dropped:
        key = "/".join(dest.replace("\\", "/").split("/")[:2]) or dest
        by_group[key] = by_group.get(key, 0) + (os.path.getsize(src) if os.path.isfile(src) else 0)
    for key, size in sorted(by_group.items(), key=lambda kv: -kv[1])[:25]:
        print(f"[prune]   -{size / 1e6:9.1f} MB  {key}")


mb_before = _mb(a.binaries) + _mb(a.datas)
print(f"[prune] before: binaries {len(a.binaries)} ({_mb(a.binaries):.1f} MB), "
      f"datas {len(a.datas)} ({_mb(a.datas):.1f} MB), total {mb_before:.1f} MB")
if PRUNE:
    _report_dropped([e for e in a.binaries + a.datas if not keep(e)])
    a.binaries = [e for e in a.binaries if keep(e)]
    a.datas = [e for e in a.datas if keep(e)]
else:
    print("[prune] GT_RENDERER_PRUNE=0 - nothing pruned (baseline build)")
mb_after = _mb(a.binaries) + _mb(a.datas)
print(f"[prune] after : binaries {len(a.binaries)} ({_mb(a.binaries):.1f} MB), "
      f"datas {len(a.datas)} ({_mb(a.datas):.1f} MB), total {mb_after:.1f} MB "
      f"(saved {mb_before - mb_after:.1f} MB)")


# --------------------------------------------------------------------------- closure check
# Every kept DLL/pyd whose load-time imports name a file that exists in the source
# torch/lib must still have that file in the bundle.  Catches an over-eager TORCH_LIB_DROP
# entry at build time instead of as a 0xC0000135 inside the frozen sidecar.
def _torch_lib_source(binaries):
    """The source ``torch/lib`` directory behind the collected torch binaries."""
    for dest, src, _typecode in binaries:
        if dest.replace("\\", "/").lower().startswith("torch/lib/"):
            return os.path.dirname(src)
    return None


def _load_time_imports(pefile_module, src):
    """Lower-cased DLL names ``src`` imports at load time, or ``None`` when unreadable."""
    try:
        pe = pefile_module.PE(src, fast_load=True)
        pe.parse_data_directories(directories=[
            pefile_module.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile_module.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ])
        # Both directories are parsed above, so both are read here: a delay-load
        # import is still a dependency, and torch/lib gained none only by luck.
        names = {e.dll.decode(errors="replace").lower()
                 for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])}
        names |= {e.dll.decode(errors="replace").lower()
                  for e in getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])}
        pe.close()
        return names
    except Exception as exc:
        print(f"[prune] WARNING: cannot read {src}: {exc}")
        return None


def _closure_warnings(binaries):
    """``"<dest> imports <dll> which was pruned"`` for every broken load-time edge."""
    import pefile

    torch_lib = _torch_lib_source(binaries)
    if torch_lib is None:
        return []
    available = {f.lower() for f in os.listdir(torch_lib) if f.lower().endswith(".dll")}
    kept = {os.path.basename(dest).lower() for dest, _src, _typecode in binaries}
    found = set()
    for dest, src, _typecode in binaries:
        if not src.lower().endswith((".dll", ".pyd")):
            continue
        imports = _load_time_imports(pefile, src) or ()
        found.update(f"{dest} imports {name} which was pruned"
                     for name in imports if name in available and name not in kept)
    return sorted(found)


try:
    _problems = _closure_warnings(a.binaries)
except ImportError:
    _problems = []
    print("[prune] WARNING: pefile not available, closure check skipped")
else:
    for _problem in _problems:
        print(f"[prune] WARNING: {_problem}")
    if _problems and STRICT:
        raise SystemExit(f"[prune] closure check failed with {len(_problems)} warning(s)")
    if not _problems:
        print(f"[prune] closure check OK ({len(a.binaries)} binaries)")

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="glassrenderer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # READY on stdout, stdin EOF = the parent died
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="renderer",
)
