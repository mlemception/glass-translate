# Changelog

All notable changes to GlassTranslate are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-17

Two changes to how balloons are lettered, and the artwork repair is now on by default.

### Changed

- **The quality renderer defaults to automatic.** It regenerates the artwork behind erased
  Japanese text, so line work and speed lines carry on through instead of stopping dead at the
  edge of a cleared rectangle. Measured over 50 pages against the plain eraser, a blind
  comparison preferred it on every region it touched, 6 to 0, with two untouched controls
  called identical; it changed 31 of those 50 pages. **It costs nothing if you cannot use it**
  — with no NVIDIA card or no models downloaded it quietly keeps the ordinary fill, and it
  never downloads anything by itself. Where it does run it only improves a panel *after* that
  panel has already been drawn, so nothing waits on it. Peak VRAM is about 11 GB. Set it to
  **off** on the Engines tab to keep the graphics card free.

  Be clear about what it does: it repairs tone damage. It does not rebuild line art that was
  destroyed, and regenerated areas come back with smoother screentone than the printed page has.

- **Balloon lettering is condensed in the app, not just in the offline renderer.** The bundled
  face is about 14 % wider than a printed English edition's, so a line that did not fit used to
  step *down* a size instead. Glyphs are now squeezed horizontally to 85 % as they are drawn,
  which lets the text keep its size and fill the balloon. A three-line balloon that previously
  set its widest line 15 % wider than the balloon interior — clipped at both walls — now sits
  inside it at the same size. Free text over artwork is deliberately left alone: it has no
  container to grow safely inside, and condensing it walked a block into a panel border.

- **A line may sit slightly off a balloon's centre line to use the width the balloon has.**
  Every line used to be centred exactly, which meant its width was capped by whichever side of
  the balloon was nearer — on a balloon drawn around a vertical Japanese column, often far less
  than the balloon really offers. A line may now sit up to a fifth of an em off centre, which
  removed a whole size step of shrinking on many blocks and cut hyphenated line breaks by about
  an eighth across a 50-page sample.

### Fixed

- **The quality renderer no longer repaints a neighbouring speech balloon.** Where a caption
  sat against a balloon, the feathered edge of its repair could bleed onto the balloon's clean
  paper — on the worst page, 43 % of the pixels it changed. Balloon interiors are now excluded
  from the repair, cutting that by 94 %.

- **A hand-edited settings file can no longer switch the quality renderer on by being
  unreadable.** A malformed value now falls back to *off* explicitly rather than to whatever the
  built-in default happens to be.

## [0.3.0] - 2026-09-16

Lettering, reviewed blind. Every change below was accepted only on a blind comparison:
the two renders unlabelled, their order randomised per comparison, and the answer key
read only afterwards. Two of every eight comparisons were blocks the
change was not expected to touch, as a check that the comparison was not inventing differences.

### Changed

- **Balloons are found by the ink that closes them, not the paper inside them.** The layout
  pass flood-filled paper, and paper leaks: a plainly drawn oval could be missed entirely, so
  its text was lettered as free text over artwork — the original lettering left underneath,
  grey haloed English spilling past the balloon edge. When the paper pass places nothing, the
  renderer now traces what the page's own ink encloses and offers that instead.

- **The ink colour is sampled from the ink, not the average of the ink and the paper.** Colour
  extraction returned the *median* of the foreground cluster, which on fine type is a blend of
  the two, so lettering came out mid-grey against a reference that is solidly black. Blocks
  lettering mid-grey fell from 47 in 316 to 5; blocks reaching pure black rose from 242 to
  283.

- **Line budgets are cut symmetric about the block's own axis.** The budget handed to the
  centring step was the raw per-row measurement, whose middle is not the block's middle, so
  a line with no slack adopted the row's centre and the block wandered down the balloon.
  Placement wander fell from 5.49 to **0.00** mean standard deviation. It costs about 2.4 % in
  mean type size, which was predicted in advance and accepted.

- **Dialogue is lettered upright.** Every speech, thought and narration block was set oblique
  on the stated convention that translated comics italicise dialogue. Measured against a
  professionally lettered edition, that edition's own English de-shears to between −2.0° and
  +1.5°, while the bundled italic face measures **+14.0°**. The convention is real but it is
  for *thought* and flashback, and nothing available distinguishes a thought balloon from a
  speech balloon. The italic face stays bundled for when a signal exists.

- **One stray glyph no longer re-faces a whole balloon.** The lettering face was chosen on
  "does this text contain any CJK at all", so a single misread character threw the entire
  block into the CJK system font — losing its typeface, its weight and its apostrophe shape,
  and displaying the offending character into the bargain. The test is now a ratio: a block
  is lettered in the system face only when CJK is at least half of it. A stray character in
  otherwise-Latin text is dropped instead.

### Fixed

- **The evaluation renderer no longer draws empty boxes for characters its font lacks.**
  Text is folded to the face that will letter it — a table for punctuation and stroked Latin
  letters, then a compatibility fold for accents, then deletion, because a black box is worse
  than any substitute. **This affects the PIL renderer used by the evaluation harness, not the
  application's own Qt renderer**, which does per-character font fallback and did not have the
  defect. It is recorded here for completeness, not as a user-facing improvement.

### Notes

- Upgrading: replace the executable, or unpack both zips into the same folder. The portable
  bundle ships as a **pair** — `GlassTranslate-0.3.0-portable-win64.zip` and
  `GlassTranslate-0.3.0-models-win64.zip` — which unpack into the same layout; each has a
  `.sha256` sidecar, and nothing is downloaded at build time.
- The optional quality renderer repairs tone damage around erased text. It does **not**
  rebuild destroyed line art, measured on six pages, and it peaks around **11 GB of VRAM**.
- No change to the overlay window's appearance or to the control panel.

## [0.2.5] - 2026-09-15

Typesetting and erasing, measured page by page against a professionally lettered English
edition. The evaluation harness itself was repaired first, because three bugs in it were
ranking the work wrongly.

### Added

- **Reference-free self-checks** (`glasstranslate/render/selfcheck.py`). Three invariants
  that need only the render and the text it was handed, rather than a professionally
  lettered page to compare against — so unlike every other measurement in this project they
  *can* run outside the evaluation harness. **Today only the harness calls them**; the app
  does not, so they are absent from the frozen executable. Wiring them into the live render
  path is future work:
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
