# manga-ocr ONNX compatibility spike (F3-1) — 2026-09-09

Machine: this dev box (Windows 11, Python 3.13 venv, onnxruntime-directml 1.24.4, numpy 2.5,
opencv 5, rapidocr 3.9.2). Pages: `Examples/before.jpg` (Japanese, 1600x1096, 50 rapidocr lines)
and `Examples/after.webp` (English scanlation of the same page, 2100x1400, 50 lines).
Probe script: `tools/ocr_probe.py` (dev-only).

## Verdict: GO

manga-ocr runs in-app on onnxruntime-directml with a CPU fallback, reads the Japanese page at
least as well as rapidocr's PP-OCRv5 recogniser, and costs ~20 ms per line on the GPU
(~85 ms per line CPU-only). Two repo files are broken exports (see pitfalls), so the shipped
bundle is **fp16 encoder + int8 decoder**, 201.6 MB total instead of the planned 231 MB.

## 1. Repo contents (`onnx-community/manga-ocr-base-ONNX`, HF tree API with LFS oids)

| path | bytes | sha256 (LFS oid) | loads in ORT 1.24.4 |
|---|---:|---|---|
| `onnx/encoder_model_fp16.onnx` | 171,851,891 | `1a6a57bc3608195c4577b13ac3aadab810dce42fa22c5a3acf0570bffc013b60` | yes (DML + CPU) |
| `onnx/decoder_model_fp16.onnx` | 58,815,651 | `6e651ed621d29ac18e2c40c503d4e7d4a1782735bea39eb53e16b3e4bac70f49` | **no** — `Type Error: Type (tensor(float16)) of output arg (/decoder/bert/encoder/layer.0/attention/self/Cast_output_0) ... does not match expected type (tensor(float))`, on every provider |
| `onnx/decoder_model_q4f16.onnx` | 24,052,350 | `e4074ed66fae74104a95dae8e74360efe6d695ea2df27ebb2f90e3dacc4d8dc6` | **no** — same Cast type error |
| `onnx/decoder_model.onnx` (fp32) | 117,445,718 | `31ca14d6dee6b3966144e128d0481d5a91f5083cbced81fe7a9571713fa50cd4` | yes |
| `onnx/decoder_model_int8.onnx` | 29,627,936 | `2e7177d2b0a59f1c612b694ed70c13971bee765cc2b2bc7bc9376e4753652f27` | yes (chosen) |
| `config.json` | 75,028 | non-LFS (`0b45d5253cef67122d2d35b32408ccffa406a87d560e2c3b7ddec3a987863f2e`) | — |
| `generation_config.json` | 261 | non-LFS (`faa07a187c72b147a39f660993bcbdbb85c5fc1a86f8a557d5487945285648c0`) | — |
| `preprocessor_config.json` | 351 | non-LFS (`445c77049d082004aa07593344d5e1f521f1198228bf586196f70ce7ae021414`) | — |

Other variants present: `decoder_model_{bnb4,q4,quantized,uint8}.onnx`, `encoder_model_{bnb4,int8,q4,q4f16,quantized,uint8}.onnx`
(`decoder_model_quantized` == `int8`, `decoder_model_q4` == `bnb4`, same oids). There is **no**
`decoder_model_merged*.onnx` (no KV-cache decoder) and **no tokenizer files** in the ONNX repo.

Tokenizer files come from the original checkpoint `kha-white/manga-ocr-base` (same author, Apache-2.0):

| path | bytes | sha256 |
|---|---:|---|
| `vocab.txt` | 24,072 | `344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d` |
| `tokenizer_config.json` | 486 | `d775ad1deac162dc56b84e9b8638f95ed8a1f263d0f56f4f40834e26e205e266` |

Both repos: Apache-2.0. Files downloaded to `models/manga-ocr/` (gitignored by `models/*/`;
verified with `git check-ignore`). Both large blobs matched their LFS sha256 after download.

Pinned revisions (HF `api/models/<repo>` → `sha`): ONNX repo
`f9023406bb2f6b17df67bc4a327c56ecd20611f0` (last modified 2026-04-18), tokenizer repo
`aa6573bd10b0d446cbf622e29c3e084914df9741` (2022-06-22). The model store resolves
`resolve/<sha>/<path>` (verified 200 for both) and pins **all seven** files by sha256, so
neither branch drift nor a same-length tampered `vocab.txt` can pass verification.

Shipped bundle (`glasstranslate.ocr.models.MANGA_OCR_FILES`): encoder fp16 171.9 MB +
decoder int8 29.6 MB + 5 small files 0.1 MB = **201.6 MB**.

## 2. ONNX I/O (onnxruntime `get_inputs()/get_outputs()`)

Encoder (`encoder_model_fp16.onnx`) — fp16 **weights**, **float32 I/O**:

```
in  pixel_values          tensor(float)  [batch_size, num_channels, height, width]   (1,3,224,224)
out last_hidden_state     tensor(float)  [batch_size, 197, 768]
```

Decoder (`decoder_model_int8.onnx`, identical signature for `decoder_model.onnx`) — no past-KV
inputs, no `use_cache_branch`:

```
in  input_ids             tensor(int64)  [batch_size, decoder_sequence_length]
in  encoder_hidden_states tensor(float)  [batch_size, encoder_sequence_length, 768]
out logits                tensor(float)  [batch_size, decoder_sequence_length, 6144]
```

So no float16 conversion is needed anywhere; the greedy loop re-runs the decoder on the whole
prefix each step (fine: sequences are 1–15 tokens, ~1–2 ms per step on CPU).

## 3. Configs and tokenizer

* `preprocessor_config.json`: ViTImageProcessor, resize 224x224 (resample 2 = bilinear),
  rescale 1/255, normalize mean/std 0.5 → `(x/255 - 0.5)/0.5`; `do_convert_rgb: null`.
  manga-ocr itself does `img.convert("L").convert("RGB")` first, so we go BGR → grey → 3 identical channels.
* `generation_config.json`: `decoder_start_token_id 2` ([CLS]), `eos_token_id 3` ([SEP]),
  `pad_token_id 0`, `max_length 300`, `num_beams 4`, `no_repeat_ngram_size 3`, `length_penalty 2.0`.
  We decode greedily (see accuracy below); beams are not needed for per-line crops.
* `config.json`: VisionEncoderDecoder; encoder ViT-base (patch 16, 224, hidden 768);
  decoder BERT with **2 layers**, 12 heads, vocab 6144.
* `vocab.txt`: 6144 tokens, `BertJapaneseTokenizer` character-level
  (`cl-tohoku/bert-base-japanese-char-v2`); ids 0–4 = `[PAD] [UNK] [CLS] [SEP] [MASK]`, then
  `<unused*>`, then single characters; **no** `##` continuation tokens (decode still strips them).
* Post-processing, verbatim from `manga_ocr/ocr.py`:
  `''.join(text.split())` → `replace('…','...')` → `re.sub('[・.]{2,}', n*'.')` →
  `jaconv.h2z(text, ascii=True, digit=True)`. `jaconv` 0.5.0 (pure Python) installed into the venv.

## 4. Latency (this machine, warm unless noted)

Session creation:

| session | DML | CPU |
|---|---:|---:|
| encoder fp16 | 256–391 ms | 386–483 ms |
| decoder int8 | 122 ms | 60 ms |
| decoder fp32 | 335 ms | 132 ms |

Single runs:

| op | DML | CPU |
|---|---:|---:|
| encoder, first call (shader compile) | **602 ms** | 89 ms |
| encoder, warm | 4–17 ms | 75–88 ms |
| decoder int8, one step (prefix len 1 / 10 / 30) | 1.8 / 19 / 21 ms | **0.8 / 1.1 / 2.0 ms** |
| decoder fp32, one step (len 1 / 10 / 30) | 1.7 / 10 / 12 ms | 3.0 / 3.5 / 5.6 ms |

The 2-layer decoder is ~10x **slower** through DirectML than on the CPU (dispatch overhead
dominates tiny graphs), so the engine pins the decoder to `CPUExecutionProvider` and only the
encoder goes to DML.

Per detected line, greedy end-to-end (encoder + full decode), 50 lines of `before.jpg`:

| configuration | median | mean | p90 | max | total 50 lines |
|---|---:|---:|---:|---:|---:|
| encoder DML + decoder int8 CPU (shipped) | **20 ms** | 21 ms | 28 ms | 39 ms | 1.05 s |
| encoder CPU + decoder int8 CPU (fallback) | 84 ms | 85 ms | 94 ms | 98 ms | 4.2 s |
| encoder DML + decoder int8 DML (rejected) | 118 ms | — | — | 311 ms | — |

Reference: rapidocr full det+cls+rec on the same page 208–224 ms warm (det only 62 ms);
`MangaOcrEngine.recognize` end-to-end (rapidocr detector + 49 lines) 1081 ms on `before.jpg`,
1294 ms on `after.webp`. Engine construction 278–288 ms (+ ~600 ms first DML encoder run,
hidden by `warmup()`).

## 5. Accuracy vs rapidocr (qualitative)

`before.jpg` (Japanese): 38/50 lines byte-identical to rapidocr after post-processing
(e.g. `領域展延の再現`, `りょういきてんえん`, `惹いたのは`, `宿儺の目を`, `それ以上に`, `再開だった`,
`術式効果の`, `俺も五条悟との`). Of the 12 differences:

| crop | manga-ocr | rapidocr | note |
|---|---|---|---|
| 45x178 column | `戦いで展延を` | `いで展延を` | manga-ocr reads the first glyph rapidocr dropped |
| 98x153 column | `才能の輝きとは` | `輝きとは` | same |
| 28x106 furigana | `ごじょうさとる！！` | `こじょうさとる` | manga-ocr correct (五条悟) |
| 350x42 horizontal title | `第２７話一人外魔境新宿決戦の` | `第247話 人外魔境新宿決戦19` | rapidocr better on digits/long horizontal |
| 47x301 | `細心の注意を払った．．．．．．` | `細心の注意を払った……` | same text, ellipsis normalised |
| 33x51, 26x39, 22x34 small kana | `ぐるぎ` / `たたか` / `かぎ` | `ぐるま` / `なか` / `かき` | tiny furigana crops, both unreliable |
| 148x40 watermark `DL-Raw.Se` | `．．．．．．．．．．．．` | `DL-Raw.Se` | dropped by the degenerate-repeat guard |
| 112x189 two columns in one box | `才」日車その才能の輝きとは` | `そのお能の` | detector box spans two columns; both wrong |

`after.webp` (English): manga-ocr is not usable on Latin text — `ＷＨＡＴＥＡＬＥＮＴ` for
`WHAT CAUGHT`, `ＭＴＥＲＲＵＰＴＯＫ` for `INTERRUPTION`, Japanese hallucinations
(`これでもテレビでは...` for `CAREFUL THAT AMPLIFICATION`) and, before the guard, three runaway
decodes (`おおおお...` to max_length 300, **1.7 s each**). Expected: the model is Japanese-only.

Blank / non-text crops (why the ink guard exists): pure white 120x40 → `それは、`; pure black →
`それは、`; uniform noise → 13 dots.

## 6. Pitfalls and mitigations (implemented in `glasstranslate/ocr/mangaocr.py`)

1. `decoder_model_fp16.onnx` / `q4f16` are broken exports (Cast type mismatch) — unloadable on
   any provider. Shipped decoder is `decoder_model_int8.onnx`; on the 10 lines compared token by
   token its greedy output was identical to the fp32 decoder's, and 2x faster.
2. The fp16 encoder has float32 I/O, so DML needs no fp16 input handling; `preprocess_crop`
   still honours the session's declared input dtype.
3. Decoder on DML is slower than CPU → decoder always on `CPUExecutionProvider`; `device`
   reports where the **encoder** runs (`gpu`/`cpu`), mirroring `rapid.py` (`auto|gpu|cpu`,
   `gpu` without DML → `RuntimeError`).
4. First DML encoder run ~600 ms → `warmup()` runs one crop through both sessions.
5. Hallucination guards: skip crops with ink fraction < 1 % (or > 99 %) before running the model;
   stop decoding when the last 8 tokens are one token / one 2-gram repeated (a runaway now costs
   ~10 steps instead of 1.7 s); drop outputs that are a single char/2-gram repeated ≥ 8 times.
6. Detector output is used for geometry and confidence only; `Segment.lang_hint = "ja"`.
7. No KV-cache decoder in the repo; plain prefix re-run is fine at these sequence lengths.
8. New runtime dependency: `jaconv` (pure Python). `requests` is used directly by the model store.
9. Per-crop exceptions are swallowed (one bad crop must not drop the frame) but the first three
   per engine are logged at WARNING with the exception type, the rest at DEBUG.
10. `ensure_models` serialises concurrent callers with a process-wide lock (startup check and a
    UI "download now" cannot stream into the same `.part`), and every failure — including an
    unwritable `models_dir` — surfaces as `ModelDownloadError`.

## 7. Recommendations for the integrator

* Use manga-ocr only when the source language is Japanese (`source_lang == "ja"`, or `auto`
  resolved to `ja`); route Latin-script pages to the rapidocr engine. Cheap variant: since the
  detector is rapidocr's full pipeline, its per-line text is already available — lines whose
  rapidocr text is Latin could keep rapidocr's reading.
* `MangaOcrEngine` calls `detector.recognize()` which runs det+cls+rec (~210 ms). A det-only
  mode on `RapidOCREngine` (`use_rec=False`, 62 ms) would save ~150 ms per frame; det-only
  scores are box scores (0.55–0.85 range) rather than rec scores, so `min_confidence` would
  need retuning.
* Download size 201.6 MB on first run; the encoder alone took 250 s here on a slow link —
  the status line needs the per-file progress callback `ensure_models` already provides.
