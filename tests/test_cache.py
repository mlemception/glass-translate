"""Tests for glasstranslate.translate (no GPU, models, network or display)."""
from __future__ import annotations

import json
import threading
import urllib.error
import zipfile
from pathlib import Path

import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.translate import (
    ArgosCT2Translator,
    IdentityTranslator,
    LibreTranslateTranslator,
    TranslationCache,
    available_backends,
    create_translator,
)
from glasstranslate.translate import argos as argos_mod
from glasstranslate.translate.argos import split_sentences


# ------------------------------------------------------------------ cache
def test_cache_hit_miss_counters():
    cache = TranslationCache()
    assert cache.get("hello", "en", "de") is None
    cache.put("hello", "en", "de", "hallo")
    assert cache.get("hello", "en", "de") == "hallo"
    assert cache.get("hello", "en", "fr") is None  # key includes target
    assert len(cache) == 1
    assert (cache.hits, cache.misses) == (1, 2)
    stats = cache.stats()
    assert stats["size"] == 1 and stats["hits"] == 1 and stats["misses"] == 2
    assert stats["hit_rate"] == pytest.approx(1 / 3)
    assert stats["max_size"] == 5000


def test_cache_stats_empty():
    assert TranslationCache(max_size=3).stats() == {
        "size": 0, "max_size": 3, "hits": 0, "misses": 0, "hit_rate": 0.0,
    }


def test_cache_lru_eviction_prefers_recently_used():
    cache = TranslationCache(max_size=2)
    cache.put("a", "en", "de", "A")
    cache.put("b", "en", "de", "B")
    assert cache.get("a", "en", "de") == "A"  # touch a -> b is now LRU
    cache.put("c", "en", "de", "C")
    assert len(cache) == 2
    assert cache.get("b", "en", "de") is None
    assert cache.get("a", "en", "de") == "A"
    assert cache.get("c", "en", "de") == "C"


def test_cache_put_refreshes_existing_entry():
    cache = TranslationCache(max_size=2)
    cache.put("a", "en", "de", "A")
    cache.put("b", "en", "de", "B")
    cache.put("a", "en", "de", "A2")  # refresh, not a new entry
    assert len(cache) == 2
    cache.put("c", "en", "de", "C")
    assert cache.get("b", "en", "de") is None
    assert cache.get("a", "en", "de") == "A2"


def test_cache_rejects_bad_max_size():
    with pytest.raises(ValueError):
        TranslationCache(max_size=0)


def test_cache_thread_safety_smoke():
    cache = TranslationCache(max_size=200)
    errors: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            for i in range(500):
                cache.put(f"t{tid}-{i}", "en", "de", str(i))
                cache.get(f"t{(tid + 1) % 8}-{i}", "en", "de")
                len(cache)
                cache.stats()
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(cache) <= 200
    assert cache.hits + cache.misses == 8 * 500


def test_cache_persistence_roundtrip(tmp_path: Path):
    cache = TranslationCache(max_size=10)
    cache.put("hello", "en", "de", "hallo")
    cache.put("こんにちは", "ja", "en", "hello")
    path = cache.save(tmp_path / "sub" / "cache.json")
    assert path.exists() and not path.with_name("cache.json.tmp").exists()

    other = TranslationCache(max_size=10)
    assert other.load(path) == 2
    assert other.get("hello", "en", "de") == "hallo"
    assert other.get("こんにちは", "ja", "en") == "hello"
    assert len(other) == 2


def test_cache_load_respects_max_size_and_bad_files(tmp_path: Path):
    big = TranslationCache(max_size=10)
    for i in range(5):
        big.put(str(i), "en", "de", str(i))
    path = big.save(tmp_path / "c.json")

    small = TranslationCache(max_size=2)
    assert small.load(path) == 5
    assert len(small) == 2
    assert small.get("4", "en", "de") == "4"  # newest survive
    assert small.get("0", "en", "de") is None

    assert TranslationCache().load(tmp_path / "missing.json") == 0
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert TranslationCache().load(corrupt) == 0
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"entries": [{"text": "x"}, 5]}), encoding="utf-8")
    assert TranslationCache().load(wrong) == 0


# --------------------------------------------------------------- identity
def test_identity_translator():
    tr = IdentityTranslator()
    assert tr.name == "identity"
    assert tr.supports("xx", "yy")
    assert tr.translate_batch(["a", "", "b"], "en", "de") == ["a", "", "b"]


# ---------------------------------------------------------------- factory
def test_factory_backends(tmp_path: Path):
    assert available_backends() == ["argos", "libretranslate", "identity"]
    cfg = AppConfig(translation_backend="identity")
    assert isinstance(create_translator(cfg), IdentityTranslator)
    cfg = AppConfig(translation_backend="libretranslate", translation_api_url="http://x/", translation_api_key="k")
    libre = create_translator(cfg)
    assert isinstance(libre, LibreTranslateTranslator) and libre.url == "http://x" and libre.api_key == "k"
    cfg = AppConfig(translation_backend="argos", translate_device="cpu", models_dir=str(tmp_path))
    argos = create_translator(cfg)
    assert isinstance(argos, ArgosCT2Translator) and argos.device == "cpu"
    assert isinstance(create_translator(AppConfig(translation_backend="nope")), IdentityTranslator)


# ------------------------------------------------------------------ libre
def test_libre_network_failure_returns_inputs(monkeypatch: pytest.MonkeyPatch):
    tr = LibreTranslateTranslator("http://127.0.0.1:9", api_key="")

    def boom(*_a, **_k):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(argos_mod.urllib.request, "urlopen", boom)
    texts = ["hello", "", "world"]
    assert tr.translate_batch(texts, "en", "de") == texts
    assert set(tr.supported_pairs()) == set()
    assert tr.supports("en", "de")  # unknown -> optimistic


def test_libre_parses_languages_and_translates(monkeypatch: pytest.MonkeyPatch):
    tr = LibreTranslateTranslator("http://fake")
    seen: list[dict] = []

    def fake_request(endpoint: str, payload=None):
        seen.append({"endpoint": endpoint, "payload": payload})
        if endpoint == "/languages":
            return [{"code": "en", "targets": ["de", "fr"]}, {"code": "de", "targets": ["en"]}]
        return {"translatedText": ["hallo", "welt"]}

    monkeypatch.setattr(tr, "_request", fake_request)
    assert set(tr.supported_pairs()) == {("en", "de"), ("en", "fr"), ("de", "en")}
    assert tr.supports("en", "de") and not tr.supports("fr", "de")
    assert tr.translate_batch(["hello", "  ", "world"], "en", "de") == ["hallo", "  ", "welt"]
    assert seen[-1]["payload"]["q"] == ["hello", "world"]
    assert seen[-1]["payload"]["source"] == "en" and seen[-1]["payload"]["target"] == "de"


# ------------------------------------------------------------------ argos
def _make_package_dir(root: Path, src: str, tgt: str) -> Path:
    d = root / f"translate-{src}_{tgt}-1_0"
    (d / "model").mkdir(parents=True)
    (d / "sentencepiece.model").write_bytes(b"")
    (d / "metadata.json").write_text(json.dumps({"from_code": src, "to_code": tgt}), encoding="utf-8")
    return d


def _make_package_zip(root: Path, src: str, tgt: str) -> Path:
    archive = root / f"translate-{src}_{tgt}-1_0.argosmodel"
    top = f"translate-{src}_{tgt}-1_0"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{top}/metadata.json", json.dumps({"from_code": src, "to_code": tgt}))
        zf.writestr(f"{top}/model/model.bin", b"x")
        zf.writestr(f"{top}/sentencepiece.model", b"")
    return archive


def test_argos_scans_dirs_and_zips_and_pivots(tmp_path: Path):
    _make_package_dir(tmp_path, "en", "de")
    _make_package_zip(tmp_path, "ja", "en")
    (tmp_path / "stanza").mkdir()  # unrelated dir is ignored
    (tmp_path / "broken.argosmodel").write_bytes(b"not a zip")
    tr = ArgosCT2Translator(tmp_path, device="cpu")
    assert tr.device == "cpu"
    pairs = set(tr.supported_pairs())
    assert pairs == {("en", "de"), ("ja", "en"), ("ja", "de")}
    assert tr.supports("ja", "de") and not tr.supports("de", "ja")
    assert tr._route("ja", "de") == [("ja", "en"), ("en", "de")]
    assert tr._route("de", "ja") is None
    # unsupported pair / same language / empty batch: inputs unchanged, no raise
    assert tr.translate_batch(["x"], "de", "ja") == ["x"]
    assert tr.translate_batch(["x"], "en", "en") == ["x"]
    assert tr.translate_batch(["", "  "], "en", "de") == ["", "  "]


def test_argos_extracts_zip_on_first_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_package_zip(tmp_path, "ja", "en")
    tr = ArgosCT2Translator(tmp_path, device="cpu")
    loaded_from: list[Path] = []
    monkeypatch.setattr(tr, "_load_models", lambda pkg: loaded_from.append(pkg.directory) or object())
    tr._get_loaded(("ja", "en"))
    assert loaded_from == [tmp_path / "translate-ja_en-1_0"]
    assert (tmp_path / "translate-ja_en-1_0" / "metadata.json").is_file()


def test_argos_missing_models_dir(tmp_path: Path):
    tr = ArgosCT2Translator(tmp_path / "nope", device="cpu")
    assert set(tr.supported_pairs()) == set()
    assert tr.translate_batch(["hi"], "en", "de") == ["hi"]


class _FakeTokenizer:
    """Whitespace tokenizer standing in for sentencepiece."""

    def encode(self, s: str, out_type=str):
        return s.split()

    def decode(self, tokens):
        return " ".join(tokens)


class _FakeResult:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _FakeCT2:
    def __init__(self):
        self.calls: list[list[list[str]]] = []

    def translate_batch(self, batch, **_kw):
        self.calls.append(batch)
        return [_FakeResult([t.upper() for t in tokens]) for tokens in batch]


def test_argos_long_inputs_split_on_sentences_and_rejoined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _make_package_dir(tmp_path, "en", "de")
    tr = ArgosCT2Translator(tmp_path, device="cpu")
    fake = _FakeCT2()
    tok = _FakeTokenizer()
    tr._loaded[("en", "de")] = argos_mod._LoadedPair(fake, tok, tok)
    monkeypatch.setattr(argos_mod, "MAX_CHUNK_TOKENS", 5)

    long_text = "one two three. four five six! seven eight nine? ten"
    huge_sentence = " ".join(f"w{i}" for i in range(12))  # single sentence > limit
    out = tr.translate_batch(["short one", "", long_text, huge_sentence], "en", "de")

    assert out[0] == "SHORT ONE"
    assert out[1] == ""
    assert out[2] == long_text.upper()
    assert out[3] == huge_sentence.upper()
    # exactly one ctranslate2 call for the whole batch
    assert len(fake.calls) == 1
    assert all(len(chunk) <= 5 for chunk in fake.calls[0])
    # 1 short + 3 sentence-packed chunks + 3 hard-split pieces of the huge sentence
    assert len(fake.calls[0]) == 1 + 3 + 3


def test_argos_translation_error_returns_inputs(tmp_path: Path):
    _make_package_dir(tmp_path, "en", "de")
    tr = ArgosCT2Translator(tmp_path, device="cpu")

    class Broken:
        def translate_batch(self, *_a, **_k):
            raise RuntimeError("boom")

    tok = _FakeTokenizer()
    tr._loaded[("en", "de")] = argos_mod._LoadedPair(Broken(), tok, tok)
    assert tr.translate_batch(["hello"], "en", "de") == ["hello"]


def test_argos_plain_ct2_layout_with_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A Hugging Face style CTranslate2 directory (model.bin next to a
    hand-written metadata.json with separate source/target SentencePiece
    models) is recognised and, with a higher priority, beats an Argos package
    for the same pair."""
    _make_package_dir(tmp_path, "ja", "en")  # priority 0
    plain = tmp_path / "sugoi"
    (plain / "spm").mkdir(parents=True)
    (plain / "model.bin").write_bytes(b"0")
    (plain / "config.json").write_text("{}")
    (plain / "spm" / "ja.model").write_bytes(b"0")
    (plain / "spm" / "en.model").write_bytes(b"0")
    (plain / "metadata.json").write_text(
        json.dumps(
            {
                "from_code": "ja",
                "to_code": "en",
                "source_spm": "spm/ja.model",
                "target_spm": "spm/en.model",
                "priority": 10,
            }
        )
    )
    tr = ArgosCT2Translator(tmp_path, device="cpu")
    pkg = tr._packages[("ja", "en")]
    assert pkg.directory == plain
    assert pkg.model_subdir == "."
    assert (pkg.source_spm, pkg.target_spm) == ("spm/ja.model", "spm/en.model")

    # Missing SentencePiece file -> directory ignored, Argos package wins again.
    (plain / "spm" / "en.model").unlink()
    tr.rescan()
    assert tr._packages[("ja", "en")].directory == tmp_path / "translate-ja_en-1_0"


def test_split_sentences():
    assert split_sentences("Hello there. How are you?  Fine!") == ["Hello there.", "How are you?", "Fine!"]
    assert split_sentences("今日は。元気？はい") == ["今日は。", "元気？", "はい"]
    assert split_sentences("line one\nline two") == ["line one", "line two"]
    assert split_sentences("   ") == []


def test_available_packages_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    tr = ArgosCT2Translator(tmp_path, device="cpu")

    def boom(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(argos_mod.urllib.request, "urlopen", boom)
    assert tr.available_packages() == []
    with pytest.raises(LookupError):
        tr.download_package("en", "de")
