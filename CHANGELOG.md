# Changelog

All notable changes to GlassTranslate are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.5] - 2026-09-15

Typesetting and erasing, measured page by page against a professionally lettered English
edition. The evaluation harness itself was repaired first, because three bugs in it were
ranking the work wrongly.

### Added

- **Reference-free self-checks** (`glasstranslate/render/selfcheck.py`). Three invariants
  that need only the render and the text it was handed, so unlike every other measurement
  in this project they also hold in normal use, on a page with no reference to compare
  against:
  - `text_complete` — every character handed to the typesetter reached the page;
  - `breaks_clean` — the drawn lines read back as the text given, breaking only at a space
    or behind a hyphen (this is what catches `BEINGAN ADULTISSO CONFUSING!`);
  - `ink_inside_bubble` — no source ink survives inside a balloon, outside the outline band
    the eraser deliberately keeps.
  The corpus regression gate enforces all three per page.
- **`erase.detector_glyph`** — the glyph size the art-line detector scales its thresholds
  by, capped at `HOUGH_GLYPH_MAX` so they stop growing with the lettering.
- **`typeset.sense_costs`** — per-break-position costs feeding `balanced_breaks`, which
  takes a new optional `sense` argument.
- **`demo/corpus_eval.py --quality`** — evaluate with the generative sidecar attached
  instead of the plain eraser, plus `sidecar_python()` and a cached render namespace so the
  sidecar cold-starts once per run rather than once per page.
- **`corpus_metrics.trustworthy_english`** and `ENGLISH_MIN_CONFIDENCE`.

### Changed

- **The eraser no longer switches off its own artwork protection on large lettering.**
  `_art_lines` judged what counts as a drawn line with two thresholds that were multiples
  of the block's glyph — but both describe the *artwork*, which does not get bigger because
  the lettering beside it does. On a page lettered at 241 px, protection demanded a 193 px
  straight run and 0.2 % of that page's art lines could supply one, against 46 % on a page
  lettered at 33 px. Measured over a 50-page sample: **artwork wrongly erased down 3.8 %,
  at a cost of one pixel more source ink left behind.** 25 of 49 pages render identically.
- **Lines break where a letterer breaks them.** Sentence ends were already forced; line
  breaking now also prefers a break after a comma, semicolon, colon or em dash, and avoids
  stranding an article from its noun. The rectangle-balancing term still decides the shape —
  a sense break wins only when the shapes are comparable.
- `corpus_metrics.line_exact` is replaced by `line_closeness`, which grades how close a
  block's line count is to the reference instead of demanding an exact match.
- `corpus_score.c_centre` is floored alongside `c_contain` and `c_leftover`.
- `demo/typeset_dev.py` writes `text_complete`, `breaks_clean` and `bubble_ink_px` into each
  block record; `demo/corpus_eval.py` rolls them up per page and the gate reads them.

### Fixed

- **`c_lines` zeroed pages that were nearly perfect.** It was an exact-match fraction, and
  because the score is a geometric mean a 0.06-weight component annihilated a page as
  thoroughly as the 0.20-weight one. One page scored five components at 1.000 and 0.975 on a
  sixth, and still scored 0.
- **OCR noise counted as dialogue we had failed to letter.** Screentone read as digits, a
  CJK glyph misread off the English page, and readings the OCR itself scored below its own
  0.6 confidence bar were all charged against the typesetter. 13 of 53 clusters over the
  sample.
- **`c_centre` reached a true zero inside the reference's own distribution.** Its span is
  the reference's p90, so a tenth of professionally lettered balloons sat at the point where
  it took a page to zero.
- **Free text over artwork was charged for its own artwork** by the new bubble-ink check,
  which ran on any block carrying a layout mask. Page totals fell from 409,515 px to
  52,492 px once it was limited to blocks actually inside a balloon.
- **Unlettered page numbers read as dropped words.** All 16 blocks the completeness check
  flagged over the sample were page numbers and stray punctuation the renderer never set;
  a block that drew nothing is a different state and is no longer charged here.

### Known issues

- An erased patch over a screentone is about 37 grey levels lighter than the panel around
  it and roughly half as textured, so it can read as a pale rectangle. Three separate fixes
  were built and all three were refuted by measurement; the diagnosis is recorded.
- A drawn balloon can still be classified as free text, in which case its interior is not
  redrawn and the source lettering survives underneath. Tracing the balloon's ink outline
  rather than flood-filling paper is the next piece of work.
- The generative quality renderer repairs tone damage but does not rebuild destroyed line
  art. It is off by default.

## [0.2.0] - earlier

Initial versioned baseline: Qt Quick control window with the Liquid Glass theme, frameless
per-pixel-alpha overlay, RapidOCR on DirectML, manga-ocr ONNX, CTranslate2 and Gemini
translation, the bubble-aware typesetter and eraser, and the optional GPU quality renderer
as a separately frozen sidecar.
