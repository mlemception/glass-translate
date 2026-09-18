"""``install-models.ps1`` must stay pinned to the same bytes the app fetches.

The installer is the route for people who take the exe without the offline
models zip, so its catalogue is a second copy of the model pins.  A second copy
drifts.  These tests compare every entry against the canonical tables in both
directions, so adding, repinning or removing a model without touching the
installer fails the suite rather than shipping a release that downloads the
wrong bytes.

Three tables are canonical, and the installer is the union of them:

* :data:`glasstranslate.ocr.models.MANGA_OCR_FILES` - the OCR bundle,
* ``glasstranslate/render/quality_models.json`` - the quality-renderer models,
* :data:`glasstranslate.translate.argos.SUGOI_FILES` - the Sugoi ja->en pack.

The first two pin a commit revision and a sha256 per file.  Sugoi does not, and
that asymmetry is asserted explicitly below rather than left implicit - see
:data:`UNPINNED_BY_DESIGN`.

The catalogue is a single-quoted PowerShell here-string piped to
``ConvertFrom-Json``, so it is plain JSON and is parsed as such.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest

from glasstranslate.ocr.models import HF_RESOLVE_URL, MANGA_OCR_FILES
from glasstranslate.translate.argos import SUGOI_FILES, SUGOI_REPO_URL

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install-models.ps1"
QUALITY_TABLE = ROOT / "glasstranslate" / "render" / "quality_models.json"

# ``$CATALOG = @'`` ... ``'@ | ConvertFrom-Json`` - a single-quoted here-string,
# so PowerShell interpolates nothing and the body is verbatim JSON.
_HERESTRING = re.compile(r"@'\r?\n(.*?)\r?\n'@", re.S)

# One file as both tables describe it: the resolved URL is the identity, and the
# size and digest are what the installer must agree with.
Pin = Tuple[int, str]

# Sugoi is published as a plain CTranslate2 directory with no pinned revision and
# no per-file digest, so ``argos.py`` fetches it from ``resolve/main`` and the
# installer mirrors that.  Every other model is pinned to a commit and verified.
# This set records the exception so that a *new* unverified download fails the
# suite, and so that pinning Sugoi is a deliberate edit here rather than a
# silent change.  Fixing it needs a Hugging Face API lookup for the revision and
# the LFS digests; it is a real integrity gap, not an accident of this test.
UNPINNED_BY_DESIGN: Set[str] = {f"{SUGOI_REPO_URL}/{rel}" for rel in SUGOI_FILES}


def _catalog() -> Dict[str, Pin]:
    """Every file the installer would download, keyed by resolved URL."""
    match = _HERESTRING.search(INSTALLER.read_text(encoding="utf-8-sig"))
    assert match is not None, "the $CATALOG here-string is no longer parseable"
    catalog = json.loads(match.group(1))
    pins: Dict[str, Pin] = {}
    for group in catalog["groups"]:
        for entry in group["files"]:
            pins[entry["url"]] = (int(entry["size"]), entry["sha256"].lower())
    return pins


def _canonical() -> Dict[str, Pin]:
    """Every pinned file the app fetches, keyed by the same resolved URL."""
    pins: Dict[str, Pin] = {}
    for file in MANGA_OCR_FILES:
        if file.sha256 is None:  # "verify by size only" is never used by the shipped table
            continue
        url = HF_RESOLVE_URL.format(repo=file.repo, revision=file.revision, path=file.remote_path)
        pins[url] = (int(file.size), file.sha256.lower())
    table = json.loads(QUALITY_TABLE.read_text(encoding="utf-8"))
    for entry in table["files"]:
        url = HF_RESOLVE_URL.format(
            repo=entry["repo"], revision=entry["revision"], path=entry["remote_path"]
        )
        pins[url] = (int(entry["size"]), entry["sha256"].lower())
    return pins


def test_the_installer_ships_next_to_the_tables_it_mirrors() -> None:
    assert INSTALLER.is_file(), "install-models.ps1 is a required release asset"
    assert QUALITY_TABLE.is_file()


def test_every_model_the_app_fetches_is_in_the_installer() -> None:
    """A model added or repinned without updating the installer fails here."""
    expected = set(_canonical()) | UNPINNED_BY_DESIGN
    missing = sorted(expected - set(_catalog()))
    assert not missing, "install-models.ps1 is missing: " + "\n  ".join(missing)


def test_the_installer_offers_nothing_the_app_does_not_fetch() -> None:
    """A model dropped from the app but left in the installer fails here."""
    expected = set(_canonical()) | UNPINNED_BY_DESIGN
    stale = sorted(set(_catalog()) - expected)
    assert not stale, "install-models.ps1 still offers: " + "\n  ".join(stale)


@pytest.mark.parametrize("url", sorted(_canonical()))
def test_size_and_digest_agree_for_every_pinned_file(url: str) -> None:
    """Same URL, different bytes - the failure a URL-only check would miss."""
    catalog = _catalog()
    if url not in catalog:
        pytest.skip("covered by test_every_model_the_app_fetches_is_in_the_installer")
    assert catalog[url] == _canonical()[url]


def test_only_the_sugoi_pack_is_downloaded_unverified() -> None:
    """Pins the known integrity gap so a new one cannot slip in unnoticed."""
    unverified = {url for url, (size, digest) in _catalog().items() if len(digest) != 64 or size <= 0}
    assert unverified == UNPINNED_BY_DESIGN, (
        "the set of unverified downloads changed; pin them, or update "
        "UNPINNED_BY_DESIGN deliberately"
    )


def test_pinned_digests_are_lowercase_hex_of_the_right_length() -> None:
    for url, (size, digest) in _catalog().items():
        if url in UNPINNED_BY_DESIGN:
            continue
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{url}: {digest!r} is not a sha256"
        assert size > 0, f"{url}: size must be positive"


# The files the Sugoi repository actually publishes, read from the Hugging Face tree API at
# revision 71d67eb8e73ec2f5aaefc0689e03a4eb843d3a2b (the tip of main, last modified 2024-11-21,
# checked 2026-09-18).  It keeps SEPARATE vocabularies - there has never been a
# shared_vocabulary.json - so a list that asks for one downloads the 1.1 GB model.bin and then
# fails on a 404, and never fetches the vocabularies CTranslate2 needs to load the model.
SUGOI_PUBLISHED = frozenset({
    ".gitattributes", "LICENSE", "README.md", "config.json", "model.bin",
    "source_vocabulary.json", "target_vocabulary.json",
    "spm/spm.en.nopretok.model", "spm/spm.en.nopretok.vocab",
    "spm/spm.ja.nopretok.model", "spm/spm.ja.nopretok.vocab",
})


def test_every_sugoi_file_the_app_downloads_is_one_the_repository_publishes():
    missing = sorted(set(SUGOI_FILES) - SUGOI_PUBLISHED)
    assert not missing, f"these would 404 during the Sugoi download: {missing}"


def test_the_sugoi_download_includes_the_vocabularies_ctranslate2_loads():
    assert {"config.json", "model.bin", "source_vocabulary.json", "target_vocabulary.json"} <= set(SUGOI_FILES)
