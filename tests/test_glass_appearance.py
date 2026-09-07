"""Mode derivation and the GLASSTRANSLATE_APPEARANCE override layer (docs/GLASS_DESIGN.md 1.3)."""
from __future__ import annotations

import logging

import pytest

from glasstranslate.ui.glass import appearance as A

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.mark.parametrize("reduce_motion", [False, True])
def test_mode_derivation(reduce_motion: bool) -> None:
    glass = A.AppearanceSignals(reduce_motion=reduce_motion)
    assert glass.mode == "glass" and glass.glass_allowed
    solid_t = A.AppearanceSignals(transparency=False, reduce_motion=reduce_motion)
    solid_c = A.AppearanceSignals(composition=False, reduce_motion=reduce_motion)
    assert solid_t.mode == "solid" and solid_c.mode == "solid" and not solid_t.glass_allowed
    hc = A.AppearanceSignals(high_contrast=True, reduce_motion=reduce_motion)
    assert hc.mode == "hc" and not hc.glass_allowed
    hc_solid = A.AppearanceSignals(high_contrast=True, transparency=False, reduce_motion=reduce_motion)
    assert hc_solid.mode == "hc"  # hc beats solid
    # reduceMotion is orthogonal: it never changes the mode.
    assert A.mode_for(glass) == "glass" and glass.reduce_motion is reduce_motion


def test_parse_overrides() -> None:
    assert A.parse_overrides(None) == [] and A.parse_overrides("") == []
    assert A.parse_overrides(" Glass , hc,,textscale=150 ") == ["glass", "hc", "textscale=150"]


@pytest.mark.parametrize(
    "token, base, expect",
    [
        ("glass", dict(transparency=False, composition=False, high_contrast=True),
         dict(transparency=True, composition=True, high_contrast=False)),
        ("solid", dict(transparency=True), dict(transparency=False)),
        ("hc", dict(high_contrast=False), dict(high_contrast=True)),
        ("reducemotion", dict(reduce_motion=False), dict(reduce_motion=True)),
        ("motion", dict(reduce_motion=True), dict(reduce_motion=False)),
        ("dark", dict(dark_mode=False), dict(dark_mode=True)),
        ("light", dict(dark_mode=True), dict(dark_mode=False)),
        ("textscale=150", dict(text_scale=1.0), dict(text_scale=1.5)),
        ("textscale=100", dict(text_scale=2.0), dict(text_scale=1.0)),
        ("textscale=225", dict(text_scale=1.0), dict(text_scale=2.25)),
    ],
)
def test_every_token(token: str, base: dict, expect: dict) -> None:
    sig = A.apply_overrides(A.AppearanceSignals(**base), [token])
    for key, value in expect.items():
        assert getattr(sig, key) == value, key


def test_precedence_later_token_wins() -> None:
    sig = A.apply_overrides(A.AppearanceSignals(), ["solid", "glass"])
    assert sig.transparency is True and sig.mode == "glass"
    sig = A.apply_overrides(A.AppearanceSignals(), ["glass", "solid"])
    assert sig.transparency is False and sig.mode == "solid"
    sig = A.apply_overrides(A.AppearanceSignals(), ["hc", "glass"])
    assert sig.high_contrast is False
    sig = A.apply_overrides(A.AppearanceSignals(), ["glass", "hc"])
    assert sig.high_contrast is True and sig.mode == "hc"
    sig = A.apply_overrides(A.AppearanceSignals(), ["dark", "light", "dark"])
    assert sig.dark_mode is True
    sig = A.apply_overrides(A.AppearanceSignals(), ["textscale=150", "textscale=125"])
    assert sig.text_scale == 1.25


def test_unknown_and_invalid_tokens_are_logged_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    base = A.AppearanceSignals()
    with caplog.at_level(logging.WARNING, logger=A.log.name):
        sig = A.apply_overrides(base, ["bogus", "textscale=abc", "textscale=50", "textscale=300"])
    assert sig == base
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 4
    assert any("bogus" in m for m in messages)
    assert any("not an integer" in m for m in messages)
    assert sum("allowed 100..225" in m for m in messages) == 2


def test_overrides_survive_every_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env layer is re-applied on the initial snapshot and on each watcher re-emit."""
    monkeypatch.setenv(A.OVERRIDE_ENV, "glass,textscale=150")
    app = A.Appearance(watch=False)
    assert app.tokens == ["glass", "textscale=150"]
    assert app.transparency is True and app.textScale == 1.5 and app.mode == "glass"
    changed: list = []
    app.changed.connect(lambda: changed.append(app.mode))
    # A system snapshot with transparency off cannot undo the override ...
    assert app.apply(A.AppearanceSignals(transparency=False, dark_mode=True)) is True
    assert app.transparency is True and app.mode == "glass" and app.darkMode is True and changed == ["glass"]
    # ... and an identical effective snapshot does not notify again.
    assert app.apply(A.AppearanceSignals(transparency=True, dark_mode=True)) is False
    assert changed == ["glass"]


def test_appearance_qml_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(A.OVERRIDE_ENV, raising=False)
    app = A.Appearance(watch=False, overrides="hc,reducemotion,light")
    assert (app.mode, app.highContrast, app.reduceMotion, app.darkMode, app.glassAllowed) == ("hc", True, True, False, False)
    app.apply(A.AppearanceSignals(accent=(10, 20, 30), composition=True))
    assert app.accent.name() == "#0a141e" and app.composition is True
    solid = A.Appearance(watch=False, overrides="solid")
    assert solid.mode == "solid" and solid.glassAllowed is False


def test_read_signals_matches_watcher(qapp) -> None:
    watcher = A.AppearanceWatcher(poll_ms=0)
    try:
        assert watcher.current == A.read_signals()
        assert watcher.refresh("manual") is False  # nothing changed in between
        assert 1.0 <= watcher.current.text_scale <= 2.25
    finally:
        watcher.stop()
