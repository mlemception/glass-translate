# GlassTranslate control panel — Liquid Glass redesign & single-exe build

Contract for the redesign of the settings/control window (`glasstranslate/ui/control.py`) and
the PyInstaller build. The **glass overlay (`overlay.py`) is out of scope** except for loading its
fonts from Qt resources and the two deliberate exceptions of the 2026-09-09 feature batch (F1
capture mode): `GlassOverlay.set_capture_excluded(bool)` and the stored capture affinity that
`showEvent` re-applies (see §2.3 "Misc" and §3 `overlayCaptureMode`; since 2026-09-10 capture mode
covers the control window too — `ControlWindow.set_capture_excluded(bool)`, §1). Everything below was de-risked by the probes under
`demo/output/probes/` (reports in `demo/output/probes/reports/*.md`); numbers quoted there are
measured on this machine (Win 11 26200, RTX 4080, one 5120x1440 monitor at DPR 1.0).

## 0. Goals (from the brief)

1. Replace the stacked QWidget form with a **tabbed** panel: glass segmented tab bar across the top,
   four tabs (Translate / Overlay / Engines / Hotkeys), and a **status strip pinned at the bottom**
   with the live latency / FPS readout visible on every tab.
2. **Real Liquid Glass**, no gimmicks:
   - glass refracts *what is actually behind the window* (live desktop pixels), distortion strongest
     at the rim and falling to ~0 across the middle; never an evenly warped panel, never a flat blur
     rectangle with white on top;
   - crisp specular highlight along the edge that follows the pointer;
   - continuous-curvature (squircle) corners; concentric radii for nested shapes;
   - soft shadows for separation, no hairline borders (except high-contrast mode);
   - tint mostly sampled from the backdrop with a *faint* blue-cyan lean;
   - springy motion: press-compress, tab indicator stretches toward its target then settles, slider
     handle swells on grab and settles with a bounce;
   - one cool accent colour, used sparingly. No neon glows, rainbow gradients, animated borders,
     particles.
3. **Accessibility**: transparency off → solid frosted fill; reduce motion → no elastic motion;
   high contrast → real borders + system colours.
4. **One-command Windows build**: single no-console `.exe`, PyInstaller, icon + version info,
   per-monitor-V2 DPI manifest, QML/shaders/fonts/icon as **Qt resources**, unneeded PySide6 modules
   excluded, then tested from another folder with **no Python on PATH**.

## 1. Architecture

```
app.py ─ GlassTranslateApp ─┬─ ControlWindow (QQuickView subclass, frameless, per-pixel alpha)
                            │     ├─ ControlBridge (QObject, context property "bridge")
                            │     ├─ Appearance   (AppearanceWatcher wrapper, context property "appearance")
                            │     ├─ BackdropProvider (QQuickImageProvider "backdrop") ← BackdropGrabber thread (mss)
                            │     └─ QML: qrc:/qml/Main.qml (+ components, shaders under qrc:/qml/shaders/)
                            ├─ GlassOverlay (unchanged; fonts from qrc:/fonts/)
                            ├─ Pipeline, HotkeyManager (unchanged)
```

### 1.1 Window
- `class ControlWindow(QQuickView)` in `glasstranslate/ui/control.py` (same module/class name so
  `glasstranslate/ui/__init__.py` and `app.py` keep working). Public contract (unchanged, all used
  by `app.py`):
  - signals `config_changed(object)`, `start_stop_requested(bool)`, `grab_mode_requested()`,
    `toggle_glass_requested()`, `capture_mode_requested(bool)` (F1, runtime only; the app answers it by
    calling `set_capture_excluded` on the overlay *and* this window), `models_changed()`,
    `closed()`;
  - methods `show()`, `load_config(cfg)`, `set_running(bool)`, `update_stats(PipelineStats)`,
    `show_status(str)`, `save_soon()`, `save_now()`, property `config`;
  - constructor `ControlWindow(cfg: AppConfig, config_path: Optional[Path] = None, parent=None)`.
  - `closeEvent` flushes a pending save then emits `closed`; `ControlWindow.close()` does the same
    when the window has no platform window yet (`QWindow::close()` returns true without delivering
    `closeEvent` for a never-shown window — the offscreen test case). `ModelDownloadWorker` and
    `LANGUAGES` stay in `control.py` with the same behaviour.
- Flags: `Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint | Qt.WindowSystemMenuHint`
  (taskbar entry, not always-on-top; the two hints add `WS_MINIMIZEBOX|WS_SYSMENU` with zero native
  frame and intact per-pixel alpha, so the taskbar click and Win+Down minimise — verified).
  `setColor(transparent)`; `QSurfaceFormat` with `setAlphaBufferSize(8)` set **before** `QApplication`
  (in `app.main`). **No `QQuickStyle` call**: every Glass* control derives from `QtQuick.Templates`
  (§2.3), never from `QtQuick.Controls`.
- Geometry (§2.1): the visible **slab** is inset in a transparent margin that holds its shadow
  (left/right 36, top 28, bottom 52 logical px). Fixed window size 832x640 (slab 760x560); the window
  is not resizable (`setMinimumSize == setMaximumSize == 832x640`, no resize edges, no Alt+Space
  "Size"). Title bar is our own (`TitleBar.qml`): drag → `bridge.startMove()`, calling
  `startSystemMove()` inside the QML `onPressed` handler (it acts on the current button press). All
  hit-testing (title-bar drag, `Theme.concentric(radiusWindow, pad)`) is relative to the slab rect,
  not the window; the transparent margin has no MouseArea (Qt has no per-pixel hit-testing, so clicks
  there still land on the window — the margin is as small as the shadow allows). Window icon from
  `:/icons/app.png`.
- Keyboard window management (a frameless window loses the native Move/system menu): Alt+Space opens
  our own glass menu (Minimise, Move, Close — no "Size", the window is fixed-size). Move sends
  `WM_SYSCOMMAND` with `SC_MOVE (0xF010)` to our hwnd (DefWindowProc's keyboard move loop: arrow keys,
  Enter/Esc). Win+Arrow snapping / Snap Layouts are documented as unsupported in the README.
- After native creation (`showEvent`, re-applied cheaply on every show):
  `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE=0x11)` so the backdrop grab (and the
  pipeline's capture) see *through* the panel. The flag is stored
  (`ControlWindow.set_capture_excluded(bool)` / `capture_excluded`, default excluded) and `showEvent`
  re-applies the stored value (while no platform window exists — before the first show or after
  `close()` destroyed it — the flag is only stored, never applied through `winId()`), so capture
  mode (§2.3 "Misc", 2026-09-10) can clear it: while the
  window is capturable its backdrop grabber is paused (an mss grab would sample the panel itself),
  the last backdrop stays on the glass, frames still in flight are dropped, and re-excluding resumes
  the grabber with a forced fresh grab. Also `DWMWA_WINDOW_CORNER_PREFERENCE` is not needed
  (we draw our own shape; window is fully transparent outside the slab).
- Anything that used `QWidget` dialogs: models-dir Browse uses `QFileDialog.getExistingDirectory(None, ...)`
  (native dialog, no parent); download progress and errors are shown **inline in QML** (glass
  progress row in the Engines page + status strip message) instead of `QProgressDialog`/`QMessageBox`.

### 1.2 Backdrop (what the glass refracts)
`glasstranslate/ui/glass/backdrop.py`:
- `BackdropGrabber(threading.Thread)`: uses `mss.MSS()` directly (or `MSSCapture` from
  `capture/backends.py` with a `raw=True` flag — **no `cvtColor`**), created in the thread; loops at
  **15 Hz**, grabs `physical_rect(window)` expanded by `MARGIN = 32` **physical** px, clamped to the
  virtual screen. Emits only when the frame changed (compare a 1/8-scale downsample of the raw BGRA
  array). It builds `QImage(shot.bgra, w, h, w*4, QImage.Format_RGB32)` in the grabber thread (QImage
  construction is reentrant; D3D11 uploads BGRA8 without conversion — measured ~60 % cheaper than the
  `cvtColor` + `Format_BGR888` path), takes one `.copy()`, sets `devicePixelRatio =
  window.devicePixelRatio()` at frame time and emits the QImage. `poke()` from
  `moveEvent`/`resizeEvent`/`exposeEvent`/`QWindow.screenChanged` for an immediate grab; `pause()` when
  hidden/minimised/solid-mode; `stop()` on close.
- `physical_rect` uses `GetWindowRect` (physical px), fallback `frameGeometry()*devicePixelRatio()`.
- On the GUI thread `BackdropProvider(QQuickImageProvider)` serves the image as
  `image://backdrop/<serial>`; the bridge exposes `backdropSerial` (int, notify) so QML rebinds
  `Image.source` ("image://backdrop/" + serial), `backdropOrigin` (QPointF, **logical**, relative
  to the window: `(grab.origin - windowPhysicalRect.origin) / window.devicePixelRatio()`, normally
  `(-32/dpr, -32/dpr)` unless clamped at a screen edge) and `backdropSize` (QSizeF, **logical** =
  image physical size / dpr). `Main.qml` sizes `backdropLayer` from `backdropSize`, never from the
  `Image`'s implicit size: Qt Quick ignores `QImage.devicePixelRatio` for image-provider images, so
  `Image.implicitWidth` is the *physical* width and would magnify the backdrop by the DPR at
  125 % / 150 % (verified at `QT_SCALE_FACTOR=1.5`: 1312 physical px became 1312 logical px, a
  6-px stripe shift and no correlation with the desktop; with `backdropSize` the best shift is 0).
- `backdropShift` (QPointF, **logical**; F2-1, 2026-09-10): `(frame.window_rect.origin -
  currentWindowPhysicalOrigin) / dpr`. `GlassControlWindow.moveEvent` feeds the current physical
  origin through `bridge.set_window_origin(x, y)` (two ints, nothing else on the GUI thread);
  `push_backdrop` records the origin the frame was grabbed for; the property is zero before the
  first frame and before the first move, and notifies only when the value changes. `Main.qml`
  draws `backdropLayer` at `backdropOrigin + backdropShift`, so a frame grabbed before the last
  move stays aligned with the desktop content it shows instead of trailing the window until the
  next grab lands (measured 2–6 px per frame at 500 px/s before the fix:
  `docs/perf/2026-09-09-glass-baseline.md`, hypothesis a).
- `backdropLuma` (0..1): mean luma over the **slab rect only** (the margin is cropped before the mean;
  the ink never sits on it), EMA with tau 250 ms. Polarity state machine in the bridge (property
  `inkPolarity`, int 0/1, notify; `0` = dark material + light ink, `1` = light material + dark ink):
  initial value `0 if appearance.darkMode else 1`; flips to 1 when luma > 0.62 and back to 0 when
  luma < 0.38 (a wide band, so a half-dark/half-light desktop keeps the system polarity). Evaluated
  only while `Theme.mode == "glass"` (§1.3 defines the other modes). QML animates the flip over 180 ms
  (`Theme.inkValue`); the same value drives the material lift (§2.2 step 4), so text always sits on a
  surface of its own polarity.
- **Never use dxcam here** (singleton camera, one duplication per process; verified to break the
  pipeline). Also never grab while `appearance.glassAllowed` is false.

### 1.3 Appearance / accessibility
`glasstranslate/ui/glass/appearance.py` = the verified `demo/output/probes/win-a11y/win_a11y.py`
lifted as-is (dataclass `AppearanceSignals`, `AppearanceWatcher(QObject)` with `changed`), plus a
thin QObject `Appearance` exposed to QML with notifying properties:
`transparency, composition, reduceMotion, highContrast, darkMode, accent (color), textScale, glassAllowed`
(`glassAllowed = transparency and composition and not highContrast`; `textScale` = DWORD
`HKCU\Software\Microsoft\Accessibility\TextScaleFactor` / 100, absent → 1.0, refreshed by the
watcher poll — Win32 apps must apply Windows text scaling themselves). Non-Windows: static defaults.

Env override layer: `GLASSTRANSLATE_APPEARANCE` (comma-separated, later tokens win) is applied on
top of **every** snapshot (initial and each `changed`, so the watcher's re-emits cannot undo it):
`glass` (transparency = composition = True, highContrast = False — forces glass on a machine with
transparency off, needed by the packaging smoke test), `solid` (transparency = False), `hc`
(highContrast = True; the Qt palette remains the system palette, so HC screenshots verify borders /
no shadow / no specular only), `reducemotion` / `motion`, `dark` / `light`, `textscale=<100..225>`.
Unknown tokens are logged at WARNING and ignored; the override never writes system settings.
`tests/test_glass_appearance.py` covers every token and the precedence rule.

Mode derivation (Theme.qml):
`mode = highContrast ? "hc" : (!transparency || !composition) ? "solid" : "glass"`; `reduceMotion`
is orthogonal to all three (HC does **not** imply reduced motion). `Theme.ink` reads the backdrop
polarity only when `mode == "glass"` (the grabber is paused otherwise, so `backdropLuma` is stale).
| mode | backdrop | shaders | motion | borders |
|---|---|---|---|---|
| glass | live capture | refraction + specular + frost; polarity from `bridge.inkPolarity` (§1.2) | springs | none (shadows) |
| solid | none (grabber paused) | same shape shader on the fill path (`solidMode = 1`): opaque fill `Theme.fill = darkMode ? Qt.rgba(.14,.15,.18,1) : Qt.rgba(.94,.95,.97,1)`, `ink = darkMode ? light : dark` (`backdropLuma` ignored), ambient specular at .5, shadows kept, no refraction | springs unless reduceMotion | none |
| hc | none | fill path: slab `palette.window`, cards `palette.base`, controls `palette.button`, ink `palette.windowText`, accent `palette.highlight`, no shadow, no specular | springs unless reduceMotion | **2 px** borders in `palette.windowText` |
| reduceMotion (orthogonal) | — | — | all `SpringAnimation`/overshoot replaced by 120 ms OutCubic, no press scale, indicator jumps with a 120 ms fade | — |

## 2. Rendering design (QML + GLSL 440 → `.qsb`)

### 2.1 Layers (Main.qml, top to bottom)
```
Item root (window size W x H)                        // root = window; W = slab.width + 72, H = slab.height + 80
  Item   backdropLayer    x: bridge.backdropOrigin.x + bridge.backdropShift.x, y: likewise, size = bridge.backdropSize (logical; window + 2*MARGIN/dpr nominal)
    Image backdrop        source: "image://backdrop/"+bridge.backdropSerial, cache:false, smooth:true
  ShaderEffectSource sharpSrc { sourceItem: backdropLayer; hideSource: true; live: true; smooth: true;
                                textureSize: Qt.size(backdropLayer.width*dpr, backdropLayer.height*dpr) }
  (blur chain, 1/4 res, hideSource:true on every hop)  blurH1 → blurV1 → blurH2 → blurV2 → ShaderEffectSource frostSrc
  GlassShadow  slabShadow    (analytic SDF shadow of the slab: sigma 12, offset (0,8), alpha .32;
                              extent 3σ + offset = 44 px below, 36 px at the sides — fits the inset)
  GlassSurface slab          x 36, y 28, width W-72, height H-80: variant "regular" (radius 22, rim 34,
                              strength .38, falloff 2, frost 1, frostLift .30)
  Item contentLayer (inside slab, margins 14): TitleBar / GlassSegmentedBar / StackLayout pages / StatusStrip
```
Geometry contract: **root = window; `slab` is inset left/right 36, top 28, bottom 52 logical px**
(fixed window size 832x640 ⇒ slab 760x560). Everything
interactive is laid out and hit-tested relative to the slab rect (§1.1). `backdropLayer` keeps
covering window + MARGIN. **Nothing of the backdrop or the blur intermediates may be composited on
screen**: every `ShaderEffectSource` in the chain has `hideSource: true` (their `ShaderEffect` sources
stay `visible: true` so they render into their layers; with the default `hideSource: false` the
15 Hz desktop copy is drawn under everything, the margins stop being transparent and dragging shows a
stale ghost). The only on-screen pixels are `slabShadow`, `slab` and `contentLayer` (alpha 0
elsewhere). Optional saving: give `backdrop` `opacity: 0` and bind `sharpSource: backdrop` directly
(an Image is a texture provider), dropping the `sharpSrc` pass.

Nesting rules — glass-on-glass is what makes recreations look muddy, so refraction is used exactly
where Apple uses it (container + segmented indicator + slider/toggle thumbs):
- **Refractive nesting is limited to the `GlassSegmentedBar` indicator and the `GlassSlider` /
  `GlassToggle` thumbs** (`variant:"pill"`: rim 8, strength .30, falloff 2, frost 0, frostLift .18,
  squircle 2). Each samples a **local** `ShaderEffectSource` of its own control (`hideSource: false`,
  `live: true`; source item = the layer with track + fill + labels that lies beneath the thumb) so the
  thumb visibly lenses the track. They never sample the slab — refracting a σ ≈ 10 px frost only moves
  blur around and shows nothing — so there is **no `slabSrc`**.
- Everything else (`GlassButton`, `StatusStrip`, `GlassStepper` buttons, `GlassComboBox` closed state
  and popup, `GlassTextField`, progress row, TitleBar buttons) is `variant:"fill"`: the fill path of
  the same SDF shader (`solidMode = 1`, no source sampling, translucent premultiplied output),
  `fillColor = Theme.fillControl` = `Qt.rgba(1,1,1, .10 + .45*inkValue)` (white 10 % over dark
  material, 55 % over light; `primary` buttons blend `accent` at 22 % over it), ambient-only
  `specular .5`, `GlassShadow sigma 5, offsetY 2, alpha .16` — the "soft shadows instead of hairline
  borders" the brief asks for.
- Group cards (`GlassCard`) are insets, not glass: `variant:"card"` = fill path with
  `fillColor = Theme.fillCard` = `Qt.rgba(1,1,1, .07 + .35*inkValue)`, ambient-only `specular .35`,
  no shadow. The segmented-bar track is the same path with `fillColor = Theme.fillTrack` =
  `Qt.rgba(1-v, 1-v, 1-v, .10 - .03*v)` (`v = inkValue`: white 10 % over dark material, black 7 %
  over light — an inset of the opposite polarity to the cards).
- Concentric rule for nested shapes: `radius = parentRadius - inset` (`Theme.concentric`, min 4);
  capsules and circles use `squircle 2` (§2.3).

### 2.2 `glass.frag` uniform block (std140, binding 0; sampler bindings 1,2)
```
mat4  qt_Matrix; float qt_Opacity;
vec2  itemSize;      // logical px
float dpr;           // Screen.devicePixelRatio (device px per logical px)
float radius;        // corner radius, logical px
float squircle;      // superellipse exponent, 2 = circular corners, ~4.5 = continuous curvature
float rimWidth;      // width of the refraction band, logical px
float strength;      // max rim displacement as a fraction of rimWidth (0..1), 0 disables refraction
float falloff;       // exponent of the rim profile (2.0 => ~0 by mid-panel)
float frost;         // 0 = sharp interior, 1 = fully frosted interior
float frostLift;     // 0..1 pull of the frosted interior toward the material base: regular .30, pill .18, card/fill 0
float coolLean;      // 0..1 strength of the cool multiply; default 0.5
vec4  tintColor;     // QML color -> vec4 arrives PREMULTIPLIED by alpha: keep alpha = 1 for any colour used as an
                     // RGB factor; default Qt.rgba(0.90, 0.96, 1.00, 1.0) (at coolLean .5 => factor (0.95, 0.98, 1.00))
float ink;           // material polarity, animated 0..1: 0 = dark material + light ink, 1 = light material + dark ink
vec2  pointer;       // pointer position in item px from the centre (light source); off-item allowed
float specular;      // 0..1 intensity of the edge highlight
float shadowInner;   // 0..1 faint inner shading at the bottom edge (depth), keep <= .12
float solidMode;     // 1 = fill path: ignore sources, output translucent premultiplied fillColor
                     // (card/fill variants always; every variant when Theme.mode != "glass")
vec4  fillColor;     // fill-path colour; arrives PREMULTIPLIED (never multiply .rgb by .a again)
float border;        // border width in logical px (0 normally, 2 in high contrast)
vec4  borderColor;   // always bound with alpha 1
vec4  rect;          // this item's rect in *source* UV space (x, y, w, h)
layout(binding=1) uniform sampler2D sharpSource;
layout(binding=2) uniform sampler2D frostSource;
```
Uniform typing rules (verified on PySide6 6.11.2): QML property names must match uniform names
exactly (`real→float`, `point/size→vec2`, `rect→vec4 (x,y,w,h)`, `color→vec4`,
`var(ShaderEffectSource)→sampler2D`). **Every `float` uniform is backed by a QML `property real` —
never `bool`/`int`**: Qt writes those as int32 bits into the float slot and the shader reads ~0.0
(a `property bool solidMode` silently disables the solid and HC fallbacks, with no warning).
`GlassSurface` declares `property real solidMode: Theme.mode === "glass" ? 0 : 1` (forced to 1 by the
card/fill variants), `property real border: Theme.mode === "hc" ? 2 : 0`, `property real ink:
Theme.inkValue`, etc.; the `variant` preset is a **string** property (`"regular"|"fill"|"pill"|"card"`),
not bool flags. **A QML `color` reaches the shader premultiplied by its alpha**
(`Qt.rgba(0.9,0.96,1,0)` arrives as `(0,0,0,0)`): `tintColor` and `borderColor` are always bound with
alpha 1 (`Theme.tintColor = Qt.rgba(0.90,0.96,1.0,1)`); `fillColor` is consumed as premultiplied
(step 7).

Shader behaviour (per fragment, `p` = item-space px from the centre):
1. `d = sdSquircle(p, halfSize, radius, squircle)` (negative inside). Corner region uses the
   superellipse `pow(pow(qx,n)+pow(qy,n),1/n) - r`; edges use the exact box distance; normal `n`
   from a 2-sample finite difference. Coverage `inside = 1 - smoothstep(-aa, aa, d)`, `aa = 0.75/dpr`.
2. Rim profile `t = clamp(1 + d/rimWidth, 0, 1)`; `mag = strength * pow(t, falloff)` (1 at the edge,
   ~0 from `rimWidth` inward — *never* a uniform warp). Displace toward the centre along `-n`:
   `uvS = uv + (-n * mag * rimWidth) / itemSize`, sampled in `sharpSource` via
   `rect.xy + uvS * rect.zw`. **Invariant: `strength * falloff <= 0.9`.** The inward mapping
   `s(x) = x + strength*rimWidth*(1 - x/rimWidth)^falloff` has edge slope `1 - strength*falloff`; if
   that goes negative the outermost pixels show a *mirrored* copy of the content (a fold) exactly
   where the eye reads the glass — real Liquid Glass compresses at the rim, it never folds.
   `GlassSurface` clamps `strength` to `0.9/falloff` and `console.warn`s. Slab: strength .38,
   falloff 2 ⇒ peak displacement 12.9 px, ~4x compression at the very edge, ~0 by 30 px.
   Lensing brightness is light-dependent so the rim stays visible on a white backdrop:
   `lightFacing = 0.5 + 0.5*dot(n, normalize(vec2(-0.6,-0.8)))`;
   `col *= 1.0 + mag * mix(-0.10, 0.12, lightFacing)` (darkens up to 10 % on the side away from the
   key light, brightens 12 % toward it).
3. Interior: `frosted = texture(frostSource, rect.xy + uv*rect.zw)`; `col = mix(frosted, refracted, t)`
   blended with `frost` (frost 0 → always sharp). So the rim shows *sharp bent* content and the middle
   is calm.
4. Material (lift toward the material base + cool lean; there is no separate `tint`):
   ```
   vec3 base = mix(vec3(0.11, 0.12, 0.14), vec3(0.97, 0.98, 1.00), ink);   // smoke / frost
   col = mix(col, base, frostLift * (1.0 - 0.5 * t));                      // weaker at the rim so the lensing stays visible
   col *= mix(vec3(1.0), tintColor.rgb, coolLean);                          // <= 5 % red cut: a twinge, not a wash
   ```
   Effect: mid-grey desktop → 0.64 in light polarity / 0.38 in dark; white desktop in light polarity
   stays ~0.99 (no white-on-white wash); ~70 % of the frosted backdrop colour survives ("mostly
   sample what is behind"). Regular glass goes light over light content and dark over dark content
   (then the ink flips); it never pulls both toward mid-grey.
5. Specular — proximity two-lobe model (the naive `max(dot(n, normalize(pointer - p)), 0)` is
   identically zero on the whole rim whenever the pointer is *inside* a convex shape, i.e. exactly
   while the user hovers or interacts with the slab):
   ```
   float R     = max(0.55 * max(itemSize.x, itemSize.y), 60.0);         // slab ~420 px, 20-px knob 60 px
   float near  = 1.0 - smoothstep(0.0, R, length(pointer - p));         // rim segment nearest the pointer
   float far   = 1.0 - smoothstep(0.0, R, length(-pointer - p));        // dim counter-lobe opposite (Apple's second lobe)
   float lobe  = near + 0.35 * far;
   float amb   = max(dot(n, normalize(vec2(-0.6, -0.8))), 0.0);         // fixed top-left key light
   float sheen = clamp(0.45 * amb + 0.55 * lobe, 0.0, 1.0);
   float lw    = 1.0 / dpr;
   float line  = 1.0 - smoothstep(0.0, lw, abs(d + 0.5 * lw));          // 2 device px support, 1 device px FWHM,
                                                                        // centred half a device px inside the coverage edge
   float band  = t * t;
   float spec  = specular * (0.60 * line * sheen + 0.14 * band * lobe);
   col += spec;                                                          // additive white, clamped; nothing coloured, no glow outside the shape
   ```
   With the pointer at rest (`Theme.pointerRest`, far away top-left) both lobes are 0 and only the
   key-light ambient remains — the "ambient only" state of cards and fills, which never bind `pointer`.
6. `shadowInner`: `col *= 1 - shadowInner * smoothstep(0.35, 1.0, (uv.y)) * t` (tiny depth cue).
7. Fill path (`solidMode = 1`): skip 2–4 and 6; `col = fillColor.rgb`, `a = fillColor.a` (premultiplied
   as delivered); step 5 applies with the item's own `specular` (Theme sets 0 in HC).
8. Border (HC): `col = mix(col, borderColor.rgb, 1 - smoothstep(border - aa, border + aa, -d))`.
9. Output premultiplied: glass path `fragColor = vec4(col, 1.0) * inside * qt_Opacity`; fill path
   `fragColor = vec4(col, a) * inside * qt_Opacity` (no further multiplication by `a`).
Reference implementation basis: `demo/output/probes/quick-shaders/glass.frag` (compiles, verified
alignment of the `rect` sampling; note its glassA preset folded at the rim — the invariant above is
new). Keep `#version 440`, no `half` identifier, `layout(std140, binding=0)`, samplers with explicit
bindings. Device-pixel snapping: `GlassSurface` positions its inner `ShaderEffect` at
`x: Math.round(sp.x*dpr)/dpr - sp.x` (same for y; `sp = Theme.scenePos(parent)`) so the edge line
lands on one device-pixel row at 125 % / 150 % scaling instead of straddling two.

`blur.frag`: 9-tap separable Gaussian, weights `0.0162 0.0540 0.1216 0.1945 0.2270` (mirrored),
uniform `vec2 step`. The chain runs at **1/4 resolution** (`ShaderEffectSource.textureSize =
size*dpr/4`, `smooth: true`, `mipmap: false`), H then V, executed **twice** (four hops), giving
σ ≈ 10 logical px at 1x (≈ 2.5 px at 1/4 res). A single 9-tap pass at half resolution is only
σ ≈ 3.5 px and leaves ~42 % of 13-px text contrast legible through the slab; beyond σ ≈ 10 the gain
is marginal. Cost for the slab: 4 passes over ~190x140 texels. `MARGIN 32 ≥ 3σ`, so the clamped
texture border never streaks into the slab. Slab `frost 1`. `shadow.frag`: analytic
Gaussian-blurred rounded-box shadow (Evan Wallace's closed form), uniforms `itemSize, radius,
sigma, offset, color`, drawn on an item inflated by `3*sigma` (+ offset).

### 2.3 QML component API (files in `glasstranslate/ui/qml/`, shaders in `qml/shaders/`)
All components use relative URLs (`"shaders/glass.frag.qsb"`) so they load from `qrc:/qml/` and
from disk (dev lab). First lines of every file: pragmas, then imports — `pragma ComponentBehavior: Bound`
everywhere; `Theme.qml` begins with `pragma Singleton` **followed by** `pragma ComponentBehavior: Bound`
(a qmldir `singleton` without `pragma Singleton` fails to load on 6.11 with "Type Theme unavailable").
`qmldir` declares `singleton Theme 1.0 Theme.qml` (implicit import of the directory); every file under
`pages/` adds `import ".."` (relative directory import; works from disk and from qrc).

**Every interactive Glass\* component is a `QtQuick.Templates` type with replaced visuals**
(`import QtQuick.Templates as T`; never `import QtQuick.Controls`, which would also pull the Basic
style plugins into the exe): GlassButton = `T.Button`, GlassToggle = `T.Switch`, GlassSlider =
`T.Slider`, GlassComboBox = `T.ComboBox` (popup = `T.Popup`), GlassTextField = `T.TextField`,
GlassStepper = `T.SpinBox`, GlassSegmentedBar = `T.TabBar` with `T.TabButton` delegates (indicator
drawn by the bar). Templates give UI Automation roles, `visualFocus`, `focusPolicy` and key handling
for free (a ShaderEffect + MouseArea + Text is invisible to Narrator/NVDA). `Accessible.name` = the
`FormRow` label (passed as `label`), `Accessible.description` = `hint`; TitleBar buttons
`Accessible.name: "Minimise"` / `"Close"`; StatusStrip `Accessible.role: Accessible.StatusBar`.
- Focus ring: 2 px accent at 60 % on every control when `visualFocus` (keyboard focus only); text
  fields and steppers show it on `activeFocus` (keyboard or click). Accent therefore appears on at most
  one control at a time and no control carries a resting hairline. HC: 2 px `palette.highlight`, solid.
- Disabled: `enabled: false` → opacity .45 and no hover/press in glass/solid; HC → text and border
  `palette.disabled.windowText`, no opacity.
- Tab order: title-bar buttons → tab bar → page controls in visual order → status-strip chevron.
  Ctrl+Tab / Ctrl+Shift+Tab cycle tabs (plus Ctrl+1..4), Left/Right inside the bar, Enter commits
  text fields, Esc closes an open popup and otherwise does nothing.

- `Theme.qml` (singleton): tokens & mode logic. Colours: `accent` (`#3A86FF` cool blue; HC → palette
  highlight); `inkValue` (real 0..1 = `mode === "glass" ? bridge.inkPolarity : (appearance.darkMode ? 0 : 1)`,
  `Behavior on inkValue { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }`);
  `ink`/`inkSecondary` (colours following `inkValue`, `ColorAnimation 180`); `tintColor`
  `Qt.rgba(0.90,0.96,1.0,1)` (alpha 1, §2.2); `fill` (solid/HC slab fill, §1.3); `fillControl`,
  `fillCard`, `fillTrack` (§2.1). **Accent inventory** (listed in the file; nothing else may use
  `accent`): Start/Stop primary fill 22 %, toggle on-track, slider fill, 6-px running dot, focus ring,
  download progress bar. There is no `warn` colour (one hue only).
  Metrics: `radiusWindow 22`, `radiusCard 14`, `radiusControl 10`, `radiusPill h/2`, `pad 14`, `gap 10`,
  `controlH 34`, `titleH 40`, `stripH 44` — every height token is a **minimum**:
  `implicitHeight: Math.max(Theme.controlH, fontMetrics.height + 2*Theme.padY)`.
  Fonts: `family` = "Segoe UI Variable Text" → "Segoe UI"; `mono` = "Cascadia Mono" → "Consolas";
  sizes in **points**, scaled by Windows text size: `textScale = appearance.textScale`,
  `fontBody = Qt.application.font.pointSize * textScale`, caption = body-1, title = body+2.
  Motion: `reduceMotion` (from appearance), `springPress {spring: 5, damping: .32}` (press release
  only: 0.3 % overshoot, 226 ms), `springGrab {spring: 5, damping: .38}` (1.28x swell on grab: reaches
  in ~190 ms, 2.8 % overshoot, settled ~260 ms), `springSwell {spring: 3.6, damping: .30}` (release:
  one visible bounce to 0.985, settled ~390 ms), `springLead {spring: 8.0, damping: .50}`,
  `springTrail {spring: 6.0, damping: .45}` (2026-09-10: measured 0.9–1.2 % overshoot, the render loop
  idle ~0.55 s after a switch; the original 4.6/.36 + 3.0/.30 gave 3 % overshoot and ~0.95 s), `fade 160`. `mode`: "glass" | "solid" | "hc".
  `concentric(outerRadius, inset)` = `Math.max(4, outerRadius - inset)`.
  Geometry helpers — the only sanctioned way to compute source rects and pointer positions.
  **Never use `mapToItem`/`mapFromItem` inside a `rect` or `pointer` binding**: such a binding only
  re-evaluates on the item's own x/y/size, so it goes stale on Flickable scroll and page slides
  (verified: the mapped rect stayed put after `contentY` and after moving the Flickable):
  ```
  function scenePos(it){ var x = 0, y = 0; while (it) { x += it.x; y += it.y; it = it.parent } return Qt.point(x, y) }
  function rectIn(item, src){ var a = scenePos(item), b = scenePos(src);
      return Qt.rect((a.x-b.x)/src.width, (a.y-b.y)/src.height, item.width/src.width, item.height/src.height) }
  function pointerIn(item){ var a = scenePos(item); return Qt.point(pointer.x - a.x - item.width/2, pointer.y - a.y - item.height/2) }
  ```
  Reading every ancestor's `x`/`y` (including `Flickable.contentItem.y = -contentY`, the page-slide
  `x` and popups in `Overlay.overlay` — both items are walked up to the window's contentItem) makes
  QML capture them as dependencies. `scale` is deliberately ignored (the .965 press-compress scales
  the lens as a whole). `pointer` (root coordinates, set by `Main.qml`) and
  `pointerRest = Qt.point(-1.5*root.width, -1.5*root.height)` (top-left key-light rest position: both
  specular lobes are 0 there).
- `GlassSurface.qml` (ShaderEffect wrapper): properties `radius`, `squircle (4.5)`, `rimWidth`,
  `strength`, `falloff (2.0)`, `frost`, `frostLift`, `coolLean (.5)`, `specular (1)`, `sharpSource`,
  `frostSource`, `sourceItem` (the Item whose coordinate space the sources cover;
  `property rect rect: Theme.rectIn(this, sourceItem)`), `pointer` (`Theme.pointerIn(this)`, bound
  **only** on refractive variants — fills and cards leave it at rest), `solidMode`, `fillColor`,
  `border`, `borderColor`, `ink` — every numeric one a `property real` (§2.2). Clamps `strength` to
  `0.9/falloff` (§2.2). Presets (`variant` string):
  - `"regular"` (slab): rim 34, strength .38, falloff 2, frost 1, frostLift .30, squircle 4.5, specular 1.
  - `"pill"` (indicator, slider/toggle thumbs): rim 8, strength .30, falloff 2, frost 0, frostLift .18,
    **squircle 2** (capsules and circles are circular by definition; a superellipse with n = 4.5 and
    r = h/2 has flattened ends — the 4.5 exponent applies only to regular/card/fill), specular 1,
    local sources (§2.1), `GlassShadow sigma 5, offsetY 2, alpha .18`.
  - `"fill"` (buttons, fields, steppers, combo, status strip, progress row, title-bar buttons): fill
    path, strength 0, frost 0, frostLift 0, `fillColor: Theme.fillControl`, specular .5 ambient-only,
    `GlassShadow sigma 5, offsetY 2, alpha .16`.
  - `"card"` (cards, segmented track): fill path, `fillColor: Theme.fillCard` (track: `Theme.fillTrack`),
    specular .35 ambient-only, no shadow.
- `GlassShadow.qml`: `radius, sigma, offsetY, color` (default `Qt.rgba(0,0,0,.32)`); hidden in HC.
- `GlassSegmentedBar.qml` (`T.TabBar`): `model: ["Translate","Overlay","Engines","Hotkeys"]`,
  `currentIndex`, signal `activated(int)`. Track = `GlassSurface variant:"card"` with
  `fillColor: Theme.fillTrack`, `height 36, radius: height/2, squircle 2`; indicator =
  `GlassSurface variant:"pill"` with `inset 3`, `radius: track.radius - inset` (= 15, concentric by
  construction), `squircle 2`, sampling a local `ShaderEffectSource` of the track + labels layer; its
  **left and right edges animate separately** (`leadEdge` with `springLead`, `trailEdge` with
  `springTrail`, lead = edge in the travel direction) so it stretches toward the destination then
  settles; both springs stop at `epsilon: 0.25` (a quarter logical px — the Qt default 0.01 px kept
  the whole scene rendering ≈ 1.3 s after each switch; with the 2026-09-10 spring values the loop is
  idle ≈ 0.55 s after a switch, see §2.4). Labels: selected → `Theme.ink`, others `Theme.inkSecondary`; keyboard Left/Right; focus
  ring per the rule above.
- `GlassButton.qml` (`T.Button`): `text`, `primary` (accent blended at 22 % over the fill),
  `checkable/checked`, `iconGlyph` optional; press → `scale .965` with
  `NumberAnimation { duration: 90; easing.type: Easing.OutCubic }` in and `springPress` out
  (reduceMotion: none); `GlassSurface variant:"fill"`.
- `GlassToggle.qml` (`T.Switch`): 44x26 pill track (accent fill when on), 22 px `variant:"pill"`
  knob lensing a local source of the track; stretches 4 px while pressed and springs to the other
  side; label to the right.
- `GlassSlider.qml` (`T.Slider`): `from,to,stepSize,value,live`, `valueText` (e.g. "10 %"); track
  4 px, fill accent; knob 20 px `variant:"pill"` lensing a local source of track + fill; pressed →
  knob scale 1.28 with `springGrab`, released → back with `springSwell` (one bounce, ≤ 400 ms); knob
  shadow grows with scale.
- `GlassComboBox.qml` (`T.ComboBox`): `model` (list of `{value, text}`), `currentValue`, signal
  `selected(value)`; closed state = `variant:"fill"`; popup = `T.Popup` (default parent
  `Overlay.overlay`, `popupType: Popup.Item`, `modal: false`, `dim: false` — a `Popup.Window` has its
  own scene graph and cannot share the glass, and an Item child of the combo would be clipped by the
  page Flickable) whose contentItem is a `GlassCard` + `ListView`. On open (one-shot computation, not
  a binding), bounded by the **slab rect** (`Theme.slabRect = Qt.rect(36, 28, W-72, H-80)`, derived
  from `Theme.rootSize`; the transparent shadow margin never carries a popup sheet — integration
  change, the earlier `window.height` bound let a 19-row list hang 38 px below the slab):
  `spaceBelow = slabBottom - mapToItem(null,0,height).y - Theme.pad`,
  `spaceAbove = mapToItem(null,0,0).y - slabTop - Theme.pad`; open downward when
  `contentHeight <= spaceBelow`, else upward; `height = Math.min(contentHeight, Math.max(spaceBelow,
  spaceAbove))` so the list scrolls instead of being clipped. Keyboard navigable, type-ahead for the font list
  (`GlassFontComboBox` variant with 300+ items: `ListView` with `reuseItems`).
- `GlassTextField.qml` (`T.TextField`): `text`, `placeholder`, `password`, `monospace`, `error`
  (bool); rest state = `variant:"fill"` with **no underline** (a resting accent hairline on every
  field would put 5–6 blue lines on the Engines tab alone); focus = the 2-px accent ring; `error` =
  ink-coloured shake (6 px, 3 cycles, 300 ms; none under reduceMotion) plus the status-strip message;
  emits `committed(text)` on Enter/focus loss.
- `GlassStepper.qml` (`T.SpinBox`): `value, from, to, step, decimals, suffix`; text field + two
  `variant:"fill"` stepper buttons; commits on Enter/blur/step.
- `GlassCard.qml`: `title` optional; column with `FormRow`s; `variant:"card"`, radius `Theme.radiusCard`,
  padding 12.
- `FormRow.qml`: `label`, `hint` (caption under the control), `content` default property; forwards
  `label`/`hint` to the control's `Accessible.name`/`Accessible.description`.
- `TitleBar.qml`: app name (title size, weight 600) left, optional running dot (accent, 6 px, only
  while running), minimise/close buttons (`variant:"fill"`, 28 px). `MouseArea` → `bridge.startMove()`
  in `onPressed` (double-click does nothing). The window is fixed-size, so there are no resize edges.
- `StatusStrip.qml` (bottom, pinned, all tabs): `GlassSurface variant:"fill"` (radius
  `Theme.concentric(Theme.radiusWindow, Theme.pad)` = 8), height ≥ 44. Left: status message
  (`bridge.statusMessage`, elided). Right: `Total 12.3 ms · 8.1 fps` in `Theme.mono` (tabular),
  a running/paused indicator, and a chevron that toggles `detailsOpen`. When open, a `GlassCard`
  drawer expands above the strip (height animates, spring in glass mode) showing the four detail
  lines exactly as the old window formatted them: Stages, Segments, Cache, Devices
  (`bridge.stats.stagesText` etc.) as `TextEdit { readOnly: true; selectByMouse: true }`. Values
  placeholder "-" until the first pass.
- Pages (`pages/*.qml`), each a `Flickable` column of `GlassCard`s, bound to `bridge` properties:
  - `TranslatePage`: card "Languages": Source (`GlassComboBox`, "Auto-detect" first), Target;
    card "Series" (F5): Series name (`GlassTextField`, `objectName seriesNameField`, bound to
    `bridge.seriesName`; trimmed, whitespace-collapsed, 80 chars max; hint names the Gemini-only use);
    card "Session": `GlassButton primary checkable text: bridge.running ? "Stop" : "Start"` (wide),
    `Grab mode`, `Show / hide glass`; hint lines with the current hotkeys.
  - `OverlayPage`: card "Glass": Background opacity (`GlassSlider 0..100 %`, `live: true` so the
    overlay previews while dragging), Font (`GlassFontComboBox`), Hide original (`GlassToggle`);
    card "Typesetting": Manga mode toggle, Uppercase toggle (disabled when manga off), hints = the
    old tooltips; card "Misc": Capture mode (`GlassToggle`, `objectName captureModeToggle`, text
    "Show the glass and this window to screen capture") bound to the runtime-only
    `bridge.overlayCaptureMode` — on: the overlay *and this window* clear `WDA_EXCLUDEFROMCAPTURE`
    (`set_capture_excluded(False)` on both, driven by `ui/capture_mode.py`) so screenshot /
    screen-recording tools see the glass and the panel; the pipeline is paused (its own grabs would
    OCR the lettering) and the panel's backdrop grabber is frozen on its last frame (§1); off: both
    re-excluded, the grabber resumes with a fresh grab, the pipeline resumes iff `running_on_start`.
    Never persisted, starts off every session, cleared again at shutdown (`CaptureMode.restore`).
    (Until 2026-09-10 the control window stayed excluded; the change is deliberate: a capture of
    the app should show the panel too.)
  - `EnginesPage`: card "OCR": engine, device; card "Translation": backend, translate device
    (enabled iff argos), API URL + API key (password; enabled iff libretranslate), Models dir
    (`GlassTextField` + Browse, always enabled), buttons `Download model…` and `Get Sugoi (ja→en)…`
    (both enabled iff argos — a deliberate change, §3), inline progress row
    (`bridge.downloadActive/Label/Progress`, Hide button); card "Gemini" (F4, `objectName geminiCard`,
    visible iff `bridge.backendGemini`): Model (`GlassTextField`, `objectName geminiModelField`, free text so
    any id a gateway serves can be entered; `bridge.geminiModelPresets` / `geminiModelDefault` feed the hint
    and placeholder), API format (`native` / `openai`), Base URL, API key (`GlassTextField password`,
    **write-only**: commits through `bridge.setGeminiApiKey(text)` into the user secret store
    `%LOCALAPPDATA%/GlassTranslate/secrets.json`, is cleared after commit and only
    `bridge.geminiApiKeySet` / a redacted `geminiApiKeyHint` are readable), Timeout (1–300 s), Retries
    (0–10); card "Series context" (F5): System prompt (`GlassTextArea`, `objectName templateArea`,
    bound to `bridge.seriesPromptTemplate`, placeholder = the built-in template; `[Series Name]` is
    substituted case-insensitively at request time, an empty stored template means "built-in", a
    malformed one falls back to the built-in with one status warning) and `Reset to default`
    (`bridge.resetPromptTemplate()`, enabled iff a custom template is stored);
    card "Quality renderer" (`objectName qualityCard`, always visible): Mode
    (`GlassComboBox`, `objectName qualityRendererCombo`, model `bridge.qualityRendererOptions` =
    `[{value:"off",text:"Off"},{value:"auto",text:"On when available"}]`, value
    `bridge.qualityRenderer`) with `bridge.qualityStatus` as the row hint - one of
    `"Off"`, `"Sidecar not installed - run renderer\install.bat"`,
    `"Models not downloaded (<size> GB)"`, `"Ready"` - and `Download quality models…`
    (`GlassButton`, `objectName qualityDownloadButton`, enabled iff `!bridge.downloadActive`,
    `bridge.downloadQualityModels()`, same inline progress row as the manga-ocr download);
    card "Pipeline": Refresh rate
    (`GlassStepper 0.5–60 step .5 " Hz"`), Debounce (`0–2000 step 10 " ms"`), Min OCR confidence
    (`0–1 step .05, 2 decimals`).
  - `HotkeysPage`: card "Global hotkeys" (hint "pynput syntax, e.g. <ctrl>+<alt>+g"): Toggle grab
    mode, Start / stop, Show / hide glass (`GlassTextField monospace`, commit →
    `bridge.setHotkey(key, text)`; on rejection the field reverts and plays `error`; the message goes
    to the status strip).
- `Main.qml`: root layout; pointer tracking: `HoverHandler { id: hover }` on root, `Theme.pointer`
  bound **directly** to `hover.point.position` while `hover.hovered` — no `Behavior` (a smoothed
  light source lags and keeps the render loop alive after every stop; direct binding measured
  ~9 % CPU for 13 glass items at 240 Hz). When `hovered` turns false a one-shot 240 ms OutCubic
  animation moves `Theme.pointer` to `Theme.pointerRest`, so the highlight never freezes at the exit
  point and ends in the ambient-only state; popups are `modal: false` so hover keeps arriving while a
  combo is open. Page switch (`StackLayout` + 160 ms opacity crossfade and 10 px slide in glass mode;
  instant in reduceMotion), `Keys` (Ctrl+1..4 and Ctrl+Tab switch tabs, Esc closes popups, Alt+Space
  opens the window menu of §1.1).

### 2.4 Motion rules
- Every spring goes through `Theme` so reduceMotion turns it into a 120 ms OutCubic (no overshoot).
- Press compress: 0.965 scale on all buttons/steppers (90 ms OutCubic in, `springPress` out — a
  spring on the way in only reaches .965 at ~140 ms and feels laggy); toggles and combos compress 0.97.
- Tab indicator: two-edge spring (lead faster than trail) — stretched during travel, settles with
  a small overshoot (~4 %). Labels crossfade colour 160 ms.
- Slider handle swells to 1.28× on grab with `springGrab`, shrinks back with `springSwell` (one
  visible bounce, ≤ 400 ms).
- Page transition ≤ 180 ms. Nothing loops, nothing animates while idle (the only continuous update
  is the 15 Hz backdrop and that only when the desktop behind changed; the pointer highlight
  re-renders only while the pointer moves over the window).
- Tab-indicator settle (2026-09-10, `docs/perf/2026-09-09-glass-baseline.md`): the springs keep the
  scene rendering at full refresh until they stop — ≈ 0.55 s per switch with the 8.0/.50 + 6.0/.45
  values above (≈ 0.95 s with the original 4.6/.36 + 3.0/.30; more damping alone does not help, a
  critically damped spring reaches the 0.25 px epsilon later). The 180 ms line is the page crossfade
  budget, not the indicator's. Separately, every ink-polarity flip (§1.2) costs ≈ 0.33 s of full-rate
  rendering (the 180 ms `Theme` ink transitions plus every control's colour Behavior); it fires only
  when the desktop behind the window changes, e.g. a drag across moving content.

## 3. Python bridge (`glasstranslate/ui/control.py`)

`ControlBridge(QObject)` exposed as context property `bridge`; all properties notify.
- Config properties (get/set; a set mutates `cfg`, emits `config_changed(cfg)`, restarts the 300 ms
  save timer; setting while `loading` is ignored): `sourceLang, targetLang, ocrEngine, ocrDevice,
  translationBackend, translateDevice, apiUrl, apiKey, modelsDir, overlayOpacity (0..1),
  fontFamily, hideOriginal, mangaMode, uppercase, refreshHz, debounceMs, minConfidence,
  hotkeyGrab, hotkeyRunning, hotkeyHidden`. `apiUrl/apiKey/modelsDir` setters `.strip()`; an empty
  `modelsDir` keeps the previous value (`or cfg.models_dir`, as before).
- Lists (constant, `QVariantList` of `{value, text}`): `languagesSource` ("auto"/"Auto-detect" first),
  `languagesTarget`, `ocrEngines`, `ocrDevices`, `backends`, `translateDevices`,
  `fontFamilies` (QFontDatabase families, strings). Unknown current values are appended like the old
  `_select_data` did. `translateDevices` = `auto, cuda, cpu` in dev; when `getattr(sys, "frozen", False)`
  it is `auto, cpu` and a stored `cuda` is appended by the unknown-value rule with the hint "CUDA is
  not available in the packaged build; translation runs on CPU" (the exe prunes `nvidia/` and cuDNN,
  so an explicit `cuda` fails at engine construction).
- Derived flags: `backendOnline` (libretranslate or gemini), `backendArgos`, `backendLibre`, `backendGemini`.
- Gemini (F4): config properties `geminiModel`, `geminiApiFormat` (`native`|`openai`), `geminiBaseUrl`,
  `geminiTimeoutS` (clamped 1–300), `geminiMaxRetries` (0–10); lists `geminiModels`, `geminiApiFormats`;
  read-only `geminiApiKeySet` / `geminiApiKeyHint` (redacted); slot `setGeminiApiKey(str) -> bool`
  writes `config/secrets.py` (never `AppConfig`) and emits `config_changed` so the translator rebuilds.
- Series context (F5): config properties `seriesName` (trim, collapse, ≤ 80) and
  `seriesPromptTemplate` (≤ 4000; `""` = built-in); slots `defaultPromptTemplate() -> str`,
  `resetPromptTemplate()`.
- Component inventory additions: `GlassTextArea` (`T.TextArea`, `lines`, `committed(text)` on
  focus loss / Ctrl+Enter, same fill material + focus ring as `GlassTextField`).
- Module split (F4-5): `bridge_fields.py` (LANGUAGES, `_CONFIG_FIELDS`, coercers, `_config_property`,
  `format_stats`, `download_label/progress`, `translate_device_items`, `is_frozen`),
  `stats_model.py` (`StatsModel`) and `download_workers.py` (`ModelDownloadWorker`,
  `MangaOcrDownloadWorker`, `QualityModelsDownloadWorker`) are re-exported by `control.py`;
  import sites are unchanged.
- OCR (F3): `ocrEngines` lists `mangaocr` / `paddleocr` with `ocr.factory.engine_label` texts;
  `mangaOcrModelsReady` (re-read on `downloadChanged`), slot `downloadMangaOcr()` reuses the
  progress row and emits `models_changed` on success (the pipeline rebuilds the OCR chain).
- Quality renderer: config property `qualityRenderer` (`off` | `auto`; anything else is coerced to
  `off` by `config.settings.normalize_quality_renderer`, so a hand-edited config can never launch
  the sidecar), list `qualityRendererOptions`, read-only `qualityStatus` and `qualityModelsReady`
  (both notify on `qualityChanged`, which the bridge emits from `downloadChanged` and from a
  `qualityRenderer` / `modelsDir` edit), slot `downloadQualityModels()` (a
  `QualityModelsDownloadWorker` on the shared progress row; `models_changed` on success, which makes
  the pipeline rebuild its quality scheduler).  The worker's `progress` signal is
  `qlonglong`-typed - the bundle is several GB, which overflows a Qt `int` - and
  `_on_download_progress` carries both overloads.
- State: `running`, `overlayCaptureMode` (read/write, runtime only — never touches `AppConfig`;
  a set emits `capture_mode_requested(bool)`; the name predates 2026-09-10, the mode now covers the
  control window too), `statusMessage` (initial value `"Ready"`), `stats` (QObject: `totalText "12.3 ms"`,
  `fpsText "8.1"`, `stagesText`, `segmentsText`, `cacheText`, `devicesText` — exact same formatting as
  the old `update_stats`), `downloadActive`, `downloadLabel`, `downloadProgress` (-1 = indeterminate,
  else 0..100), `backdropSerial`, `backdropOrigin`, `backdropShift`, `backdropLuma`, `inkPolarity` (§1.2), `windowTitle`.
- Slots: `startStop(bool)`, `grabMode()`, `toggleGlass()`, `browseModelsDir()`,
  `downloadModel(bool sugoi)`, `hideDownload()`, `setHotkey(str which, str text) -> bool`,
  `startMove()` → `startSystemMove()` (only valid from a QML `onPressed`), `startResize(int edges)`
  (no-op — the window is fixed-size; kept so old/cached QML cannot crash calling it), `minimize()` →
  `showMinimized()`, `close()` → `ControlWindow.close()` (the `QWindow` close, so `closeEvent` flushes
  the pending save and emits `closed`), `openLogs()` (not required).
- `setHotkey(which, text)`: `text.strip() == ""` → revert the field to the current value silently (no
  status; `normalize_hotkey("")` raises); otherwise store `normalize_hotkey(text)` and the field
  displays the normalized form; `ValueError` → status `f"Invalid hotkey {text!r}: {exc}"`, return False.
- `save_now` failure → `statusMessage = f"Could not save config: {exc}"`.
- Download flow keeps `ModelDownloadWorker` semantics: refuse when running → status "A download is
  already running."; same-language check → status "Source and target language are the same." and no
  download; label exactly `"Sugoi v4 ja → en"` or `f"{src} → {tgt}"`; `downloadLabel` =
  `f"Downloading {label} model…"` until the first progress, then
  `f"Downloading {label}: {done/1e6:.1f} / {total/1e6:.1f} MB"` with `downloadProgress = done*100//total`
  when `total > 0`, else `f"Downloading {label}: {done/1e6:.1f} MB"` with -1; `models_changed` on
  success; "Installed model at …" / "Download failed: …" status. `hideDownload()` hides the row only:
  the worker keeps running, `downloadActive` stays true (a new download is still refused) and the
  result still goes to the status strip (the old `dialog.canceled.connect(dialog.hide)`).
  `ModelDownloadWorker.parent` annotation becomes `Optional[QObject]` (the owner is now a `QWindow`).
- Opacity: `GlassSlider live: true`, so `config_changed` fires on every step (overlay previews while
  dragging, as the old `valueChanged` wiring did). Stats detail lines are selectable
  (`TextEdit { readOnly: true; selectByMouse: true }`, the old `TextSelectableByMouse`).
- `ControlWindow.update_stats` → `bridge.stats`; `show_status` → `bridge.statusMessage`;
  `set_running` → `bridge.running`; `load_config` → sets all properties under `loading`.
- **Deliberate behaviour changes vs the QWidget window** (everything else is byte-for-byte; do not
  "fix" these back — `tests/test_glass_bridge.py::test_backend_flags` encodes 1 and 2):
  1. `Get Sugoi (ja→en)…` is enabled iff `backendArgos` (same rule as Download; it installs into the
     Argos models dir and had no effect until the backend was switched).
  2. Models dir field and Browse stay enabled for every backend (unchanged from the old window).
  3. `QMessageBox`/`QProgressDialog` are replaced by the inline download row + status strip messages.
  4. `app._apply_overlay_settings` now calls `overlay.set_uppercase(cfg.uppercase)` (bug in the
     current wiring, confirmed by the probe).

## 4. Resources & fonts
- `glasstranslate/ui/resources.qrc` → compiled to `glasstranslate/ui/resources_rc.py` by
  `tools/build_resources.py` (`build.py` always regenerates): prefix `/qml` (all `.qml`, `qmldir`,
  `shaders/*.qsb`), `/fonts` (`animeace2_reg.ttf`, `animeace2_ital.ttf`), `/icons` (`app.png` 256 px,
  `app.ico` not needed in qrc). `app.png` is a **committed source asset** at
  `glasstranslate/ui/icons/app.png` (python owns; generated once with PIL: a cool-tinted glass
  squircle lens with a crisp top-left specular and a soft shadow); `packaging/make_icon.py` only
  derives `packaging/glasstranslate.ico` from it (build owns) — no build step writes into the qrc inputs.
- Freshness: `build_resources.py` writes a header line
  `# inputs-sha256: <sha256 over sorted (relpath, sha256) of every qrc input and every .frag source>`;
  `--check` recomputes that digest and compares it to the header — no rebuild, no byte comparison of
  outputs (mtimes are meaningless after a checkout and rcc/qsb bytes differ across PySide6 versions).
- Decision: `resources_rc.py` is **gitignored during parallel implementation** (every QML/shader edit
  would otherwise churn a 100 KB+ generated file owned by another workstream). `control.py` imports it and,
  when missing and not frozen, calls `tools.build_resources.build()` once (dev fallback);
  `tests/conftest.py`'s session fixture calls `build()`; `build.py` always calls it; the integration
  owner (§8) commits it once at the end, after which `--check` guards it in CI.
- `overlay.load_manga_font()` reads `:/fonts/...` via `QFile` → `QFontDatabase.addApplicationFontFromData`,
  falling back to the on-disk files only when the resource is missing (dev without rc). No other
  overlay change.
- `tools/build_resources.py`: compiles every `qml/shaders/*.frag` with
  `.venv/Lib/site-packages/PySide6/qsb.exe --glsl "100 es,120,150" --hlsl 50 --msl 12` (fallback
  `pyside6-qsb`; **never call the wrapper with --help**, it hangs), then `pyside6-rcc -g python`.
  `--check` verifies the digest above and runs the qmlimportscanner guard (§6); exits non-zero on
  either failure (used by a test).

## 5. Frozen-mode fixes (from the hazard audit)
- `settings.default_models_dir()`: when `getattr(sys, "frozen", False)` →
  `%LOCALAPPDATA%/GlassTranslate/models`; dev branch unchanged (tests pin it).
- `app._configure_logging()`: `StreamHandler` only if `sys.stderr` is not None; when frozen or no
  stderr, `RotatingFileHandler` at `%LOCALAPPDATA%/GlassTranslate/logs/glasstranslate.log`
  (2 MB x 3) and `qInstallMessageHandler` routing Qt messages to logger `qt`.
- `GLASSTRANSLATE_SMOKE_LOG=<path>`: the app writes a JSON report
  `{frozen, meipass, exe, qml_status, qml_errors[], qml_warnings[] (engine.warnings collected from
  setSource until the grab), shader_effects: {total, compiled, error, logs[]} (walk
  rootObject().findChildren(QQuickItem) for items exposing 'fragmentShader' and 'status'; `error` is
  a gate, `compiled` is diagnostic only — per-item status is not reliable even on the real RHI),
  rhi_backend, appearance: {mode, transparency, reduceMotion, highContrast, darkMode, textScale},
  backdrop_serial, backdrop_luma, ink_polarity, fonts_ok ('Anime Ace 2.0 BB' in
  QFontDatabase.families()), ssl_ok (`import ssl` succeeds - the exe must carry CPython's OpenSSL DLLs or
  every https path is dead), resources_ok: {qml, shaders, fonts, icon}, control_exposed,
  overlay_shown, pages_ok (every page opened and every Glass* component instantiated),
  dpi_awareness, status_history[] (every show_status message), stats: {totalText, fpsText, stagesText,
  segmentsText, cacheText, devicesText} (the status-strip texts of the last pipeline pass), screenshot: <png path>,
  actions: {<name>: {started, finished, seconds, label, progress, models_dir, models_ready_before,
  models_ready_after}} (only when `GLASSTRANSLATE_SMOKE_ACTIONS` named a whitelisted bridge slot -
  `downloadMangaOcr`; the app reports and quits by itself once the action finishes),
  quality: {setting, launcher, kind ("frozen" | "venv" | null), models_ready, hint, scheduler_active,
  scheduler_available, scheduler_stats} (what `render/quality.find_sidecar_python` found and the
  pipeline's scheduler state), overlay: {patch_conversions, patch_upgrades} (counters of the
  overlay's existing patch cache: conversions of `clean_patch`, and re-conversions forced by a bumped
  `clean_patch_serial`), paths: {config, models_dir, user_data_dir, logs} (the resolved locations,
  under the portable folder when `portable.txt` is present)}`.  Two more whitelisted actions
  (2026-09-11, portable bundle): `probeSidecar` launches the sidecar the app found in **fake** mode
  through the app's own lookup + spawn code and records `actions.probeSidecar: {launcher, kind,
  seconds, health, ok, error}`; `feedPage` (`GLASSTRANSLATE_SMOKE_PAGE=<image>`) replaces screen
  capture with a static page so the pipeline OCRs, typesets and - with the real sidecar - upgrades
  blocks, and records `actions.feedPage: {page, started, finished, seconds, blocks, upgraded,
  overlay_upgrades, quality_stats, status}`; the app reports and quits once a block's serial advanced
  and the overlay re-converted it, or at the autoexit.  Every `GLASSTRANSLATE_SMOKE_*` variable is a
  test-harness hook: none of them does anything unless `GLASSTRANSLATE_SMOKE_LOG` is set, the action
  names are a fixed whitelist, and the page path is validated (existing file, image suffix, ≤ 64 MiB)
  and reported by basename only.  The control window itself is unchanged.
  The grab (`grabWindow()`, saved next to the log) runs from a single-shot timer at
  `AUTOEXIT_MS - 1500` after at least one `frameSwapped`, never from `aboutToQuit` (the window may
  already be unexposed). The assertions live in §6 (`build.py --test`) and §7 (`run.py`, offscreen).
  Fields plus "screenshot non-blank" alone cannot prove the glass rendered: a failed `.qsb` leaves
  `QQuickView.status` Ready with text still visible, and a solid-mode fallback is non-blank too.

## 6. Build (`build.py`, one command)
`.venv\Scripts\python build.py` → (1) `tools/build_resources.py`, (2) `packaging/make_icon.py`
(derives `packaging/glasstranslate.ico`, 7 sizes, from the committed `glasstranslate/ui/icons/app.png`,
§4), (3) `packaging/version_info.txt` from `glasstranslate.__version__` (add `__version__ = "0.2.0"`),
(4) PyInstaller with `packaging/GlassTranslate.spec` (onefile, `console=False`,
`manifest=packaging/glasstranslate.manifest` with `dpiAware true/pm` + `dpiAwareness PerMonitorV2`,
icon, version), (5) `build.py --test` (also run by default at the end): copy `dist/GlassTranslate.exe`
alone into a fresh `%TEMP%/gt_extest_*` folder, run it from there with `PATH` reduced to
System32/Windows/Wbem/PowerShell, no `PYTHON*`/`QT_*`/`VIRTUAL_ENV`, no inherited std handles,
`GLASSTRANSLATE_CONFIG` → temp file, `GLASSTRANSLATE_SMOKE_LOG`, and assert the §5 report:
- run A (defaults + `GLASSTRANSLATE_APPEARANCE=glass`, `GLASSTRANSLATE_AUTOEXIT_MS=8000`):
  `qml_status == "Ready"`, `qml_errors == []`, `qml_warnings == []`, `shader_effects.error == 0`
  (do not assert `compiled == total`: an item sharing a `.qsb` can report Uncompiled while its pixels
  are correct), `rhi_backend == "D3D11"`, `appearance.mode == "glass"`, `backdrop_serial >= 1`,
  `dpi_awareness == 2`, `control_exposed`, `overlay_shown`, `pages_ok`, every `resources_ok` true,
  `fonts_ok`, `ssl_ok`; screenshot: alpha == 0 at the four window corner pixels, alpha mean ≥ 250 inside the slab
  inset 40 px, RGB std ≥ 8 inside (a solid fallback or a dead shader path is flat), tab-bar row mean
  |diff| vs the slab ≥ 6. The last two depend on the desktop behind the window (the glass refracts
  it): measured 6.9–16 over a static desktop and 4.1 with the chat client's page behind the default
  window position (2026-09-10, identical values with the old and new spring constants) — a FAIL on
  those two rows alone is a desktop-content artefact, rerun with a plain window behind the exe.
- run B (temp config with `running_on_start=true`, `translation_backend="identity"`, AUTOEXIT 20000):
  `status_history` contains a message starting `"OCR: <engine> on"` (`mangaocr`, or `paddleocr` when the
  manga-ocr models are not downloaded yet — the bare smoke exe never has them) and none starting `"Engine error"`
  or `"Error:"` (exercises the rapidocr / py3langid data in the frozen tree, which only fail at engine
  construction).
- run D (opt-in `--runs D`, ~200 MB download; temp config as B, `LOCALAPPDATA` pointed at an empty
  folder inside the sandbox, `GLASSTRANSLATE_SMOKE_ACTIONS=downloadMangaOcr`, AUTOEXIT cap 600000):
  `status_history` starts with the `"OCR: mangaocr failed (...); using paddleocr"` fallback,
  `actions.downloadMangaOcr` finished with progress 100 and `models_ready_before/after` False/True
  under the clean profile, a `"manga-ocr models installed at ..."` status, then `"OCR: mangaocr restored"`
  (the fallback chain re-probes the primary every 5 s; a rebuild would say `"OCR: mangaocr on ..."`), no `"Download failed"`, no secret markers, log file under the clean profile.
- run C (onefile lifecycle): start with AUTOEXIT 60000, wait for the report, `taskkill //F //PID`,
  launch run A again and assert the stale `_MEI*` dir is gone (sweep below).
- run `portable` (2026-09-11, `--runs portable --portable-zip … --models-zip …`; the acceptance of
  the portable bundle, implemented in `packaging/smoke_portable.py`): both zips are unpacked into a
  fresh folder whose path holds a space and a non-ASCII character; `PATH` = the System32 family,
  no `PYTHON*`/`QT_*`/`VIRTUAL_ENV`, `HTTP_PROXY` = `HTTPS_PROXY` = `http://127.0.0.1:9` so any
  network attempt fails, `LOCALAPPDATA`/`APPDATA` pointed at empty decoy folders that must stay
  empty, and **no** `GLASSTRANSLATE_CONFIG` (the portable config path is under test; overrides go
  to `<root>\config\config.json`).  Checks: (1) `renderer\glassrenderer.exe serve --fake` prints
  `READY`, answers `/health` and one `/inpaint` byte-identical outside the mask, exits on stdin
  EOF; (2) with `GT_GPU_TESTS=1` the real stages load from `<root>\models` with no download and one
  job completes; (3) the exe from the root with `quality_renderer=auto`, `running_on_start`, Argos
  ja→en and `GLASSTRANSLATE_SMOKE_ACTIONS=probeSidecar`: `actions.probeSidecar.ok`,
  `quality.kind == "frozen"`, `quality.launcher` and every `paths.*` under the root, `OCR: mangaocr
  on …` in `status_history`, `Argos packages in <root>\models` in the log and no download / URL /
  proxy line, then runs static, A, B, C from the root, then the same launch with `renderer\`
  renamed away: no `Engine error`/`Error:`, `quality.launcher` null (the quick fill); (4) with
  `GT_GPU_TESTS=1`, `feedPage` on `Examples/before.jpg`: `actions.feedPage.upgraded >= 1`,
  `overlay.patch_upgrades >= 1`, `Quality renderer: ready on cuda` in `status_history`; (5) the
  root is moved to another space + non-ASCII path and (1) and (3) are rerun.
Spec rules (verified in `demo/output/probes/packaging/` and the review probes): keep PySide6
`QtCore QtGui QtWidgets QtQml QtQuick QtNetwork QtOpenGL` (QtQuickControls2 and QtQuickWidgets are
**not** needed: no `QQuickStyle`, every control is a `QtQuick.Templates` type, §2.3); exclude every
other `PySide6.Qt*`; allowlist `Qt6*.dll` (Core, Gui, Widgets, Qml, QmlMeta, QmlModels,
QmlWorkerScript, Quick, QuickTemplates2, QuickLayouts, Network, OpenGL — **no**
QuickControls2/QuickControls2Impl/QuickControls2Basic/QuickControls2BasicStyleImpl and no Svg;
`Qt6QuickTemplates2.dll` imports only Core/Gui/Qml/QmlModels/Quick and the Templates qmldir depends
only on `QtQuick`, verified with pefile on 6.11.2); qml dirs (the filter keeps files whose dirname is
**exactly** in this set): `QtQuick`, `QtQuick/Window`, `QtQuick/Templates`, `QtQuick/Layouts`, `QtQml`,
`QtQml/Models`, `QtQml/WorkerScript`. `QtQml/Base` does not exist in 6.11 — do not list it. Do not
list `QtQuick/Controls*`: a Basic-style build would also need `QtQuick/Controls/Basic/impl` (nine Basic
components import it; without that directory every ComboBox/SpinBox/TextField fails only in the
frozen exe) — the guard below fails the *build* instead if our QML ever imports `QtQuick.Controls`.
Plugins (`platforms/qwindows`, `styles/qmodernwindowsstyle`, `imageformats/qjpeg,qico,qgif,qwebp`);
drop translations/opengl32sw/PySide6's libcrypto+libssl/ffmpeg DLLs, `nvidia/`, `cudnn64_9.dll`,
`opencv_videoio_ffmpeg*.dll`, `cv2/data`; excludes also `torch openvino paddle tensorrt MNN cupy onnx`.
Data: `collect_data_files('rapidocr', includes=['*.yaml','models/*','inference_engine/**/*.yaml'])`,
py3langid (data), sentencepiece (package_data); `collect_dynamic_libs("ctranslate2")`. Hidden imports
exactly as verified in `packaging.md`: `SHIBOKEN_RUNTIME_STDLIB`, `mss.windows`, `comtypes.gen`,
`dxcam.processor.numpy_processor`, `rapidocr.main`, `rapidocr.utils.load_image`,
`rapidocr.utils.vis_res`, `rapidocr.utils.download_models`, `rapidocr.inference_engine.onnxruntime`
+ `collect_submodules` of `rapidocr.ch_ppocr_det/cls/rec`. pefile closure check must print no
warnings. Guard: `tools/build_resources.py --check` runs
`pyside6-qmlimportscanner -rootPath glasstranslate/ui/qml -importPath <PySide6>/qml` and fails if any
non-optional module outside the qml-dir set is imported by our QML.
Onefile lifecycle: onefile extracts ~360 MB into `%TEMP%\_MEIxxxxxx` per launch and the bootloader
removes it only on normal exit (a Task-Manager kill or crash leaves it behind). Bundle a marker
`datas += [("packaging/gt_bundle.marker", ".")]`; in `run.py`, when frozen and before Qt is imported,
sweep `Path(tempfile.gettempdir()).glob("_MEI*")` for dirs where
`(d/"gt_bundle.marker").exists() and str(d) != sys._MEIPASS` and `shutil.rmtree(d, ignore_errors=True)`
(a concurrently running copy keeps its DLLs locked and survives). The exe is **CPU (translation) +
DirectML (OCR) only** (§3 hides `cuda` when frozen). Target: exe ≈ 150–175 MB, cold start ≈ 5 s
(onefile extraction). README: document the onedir alternative (for users who see antivirus rescans),
the CPU-only translation, and that Win+Arrow snapping is unsupported (§1.1).

## 7. Verification protocol
- `pytest -q` stays green (baseline **160 passed**). `tests/conftest.py` provides a session `qapp`
  fixture (`QT_QPA_PLATFORM=offscreen`, `QApplication.instance() or QApplication([])`) and runs
  `tools.build_resources.build()` once. New tests use the prefix `tests/test_glass_*.py`:
  - `test_glass_bridge.py`: bridge round-trip + save, hotkey validation (empty → silent revert,
    invalid → status + False, normalized form displayed), `test_backend_flags` (§3 deliberate changes
    1–2), stats formatting, download strings, polarity state machine (initial from `darkMode`,
    .38/.62 hysteresis), `translateDevices` frozen rule.
  - `test_glass_qml.py` — QML under offscreen. The offscreen platform always uses the Software scene
    graph on Windows (also with `QSG_RHI_BACKEND=d3d11|opengl` forced), so **shaders never compile
    there**: `ShaderEffect.status` stays Uncompiled with an empty log and nothing render-dependent is
    asserted. `w = ControlWindow(AppConfig(), tmp_path/"c.json")` (never shown); assert
    `w.status() == QQuickView.Status.Ready`, `w.errors() == []`, collected `w.engine().warnings`
    empty, `w.rootObject() is not None`; every `Glass*.qml` and `pages/*.qml` instantiable via
    `QQmlComponent`; each component reports an `Accessible.role` and a non-empty `Accessible.name`;
    all four Theme modes derivable; set each bridge property and assert `config_changed` fired once
    and the config file contains the value after `save_now()`; finally `w.close()` emits `closed`.
  - `test_glass_shaders.py` — static uniform contract: parse `qsb.exe -d <file>.qsb` reflection JSON
    (`uniformBlocks[0].members` + `combinedImageSamplers`) for every `.qsb` and assert every member
    name has a property of the matching QML type on the wrapper that uses it (`float←real`,
    `vec2←point/size`, `vec4←rect/color`, `sampler2D←var`), and that every `fragmentShader:` string in
    `qml/` exists in `resources_rc`.
  - `test_glass_appearance.py`: mode derivation for glass/solid/hc × reduceMotion, every env-override
    token and the precedence rule; `test_glass_resources.py`: `--check` digest; `test_frozen_paths.py`:
    models-dir/logging frozen branches (monkeypatch `sys.frozen`).
  - `python run.py` with `GLASSTRANSLATE_SMOKE_LOG` writes the same §5 report, so CI (offscreen)
    asserts everything except shader/rhi/screenshot. Compile/pixel verification happens **only** in the
    on-screen smoke run (§6 run A), gated on grabbed pixels — never on per-item `ShaderEffect.status`.
- Visual QA: run `python run.py` with `GLASSTRANSLATE_SMOKE_LOG`, take the window grab and an mss
  grab of the same rect; **view only downscaled copies ≤ 1000 px on the long side, JPEG q80, max 8
  views per workstream**; also crop 2× zooms of one corner (rim distortion + specular) and the tab bar
  during a tab switch (frame from a timer) to confirm: rim distortion visible with no fold (mirrored
  content) at the edge, centre calm, edge line = 1 device px, rim still visible over a white desktop,
  no white wash, concentric radii (capsule track / capsule indicator), shadows not borders, no
  resting accent hairlines, ink *and* material lift adapt to a light vs dark desktop (test over a
  white window and over a dark one), highlight follows the pointer while it is *over* the slab and
  fades to ambient when it leaves. One extra pass under `QT_SCALE_FACTOR=1.5` and one with
  `GLASSTRANSLATE_APPEARANCE=textscale=150`: edge line 1 device px, `rect` alignment ≤ 1 px
  (quick-shaders method), no clipped text, no layout overflow at slab 760x560.
- Performance (budgets; measured values from the review probes in parentheses): idle ≤ 3 % of one
  core with a static desktop (2.0 %), ≤ 12 % with the desktop changing under the panel (4.7 %),
  pointer moving over the panel ≤ 15 % at vsync (8.6 % for 13 glass items at 240 Hz), grabber median
  ≤ 8 ms per grab (5–6.5 ms; isolated 12 ms peaks are not failures); expect ~2 rendered frames per
  backdrop update (Image source change + live layers), i.e. ~30 fps while the desktop changes; 0 %
  when nothing changes. Integration measurement of the finished panel
  (`demo/output/dev/visual_qa.py --scenario perf`, 8 s windows, 5120x1440 screen): idle 2.0–2.9 %
  with 0 frames rendered, desktop scrolling behind the panel 5.1–5.5 % at 25.7 fps (103 grabs, 103
  emits per 8 s = one rendered frame per backdrop update), grabber median 6.7–7.1 ms / p90 8.5–8.8 ms,
  pointer sweep over the panel 10.0 % at 109 fps, back to 1.9–2.7 % and 0 frames within 0.8 s.
- Profiling hooks (F2-0): `GLASSTRANSLATE_PROFILE=1` switches `glasstranslate/ui/glass/profile.py`
  on (`move` / `poke` / `grab` / `frame_gui` / `push_backdrop_ms` / `swap` / `tab` marks; a no-op
  object when unset, one attribute lookup per hook; the `frameSwapped` hook is a *queued*
  connection — a direct Python slot on the render thread deadlocks on the GIL).
  `tools/profile_glass.py` drives the real window on screen (scripted drag, 4-tab burst, idle)
  and writes `demo/output/perf/glass-profile-<ts>.jsonl` plus a summary. Baseline and post-F2
  numbers: `docs/perf/2026-09-09-glass-baseline.md` (240 Hz, 240 x 2 px drag: 48 % of one core
  at ~137 grabs/s, tab transitions every refresh, idle 2.3 % / 0 swaps once settled).
- Accessibility: force each mode via `GLASSTRANSLATE_APPEARANCE` (§1.3 tokens) and screenshot each;
  Narrator announces every control by its FormRow label; keyboard-only pass through the tab order.

## 8. File ownership for parallel implementation
| workstream | owns |
|---|---|
| shaders | `glasstranslate/ui/qml/shaders/*.frag`, `qml/GlassSurface.qml`, `qml/GlassShadow.qml`, `qml/Theme.qml`, `demo/glass_lab.py` (dev harness loading the QML dir from disk over a static PNG backdrop, with a mode switch) |
| controls | every other file in `glasstranslate/ui/qml/` incl. `Main.qml`, `qmldir`, `pages/` |
| python | `glasstranslate/ui/control.py`, `glasstranslate/ui/glass/*.py`, `glasstranslate/ui/resources.qrc`, `glasstranslate/ui/icons/app.png` (committed source asset), `glasstranslate/ui/__init__.py` (no change expected; must keep re-exporting `LANGUAGES, ControlWindow`), `tools/build_resources.py`, `overlay.py` (font loading only), `app.py`, `config/settings.py`, `glasstranslate/__init__.py` (`__version__`), `.gitignore` (`dist/`, `build/`, `packaging/glasstranslate.ico`, `glasstranslate/ui/resources_rc.py`), `tests/conftest.py`, `tests/test_glass_*.py` (all new tests use this prefix), `tests/test_frozen_paths.py` |
| build | `build.py`, `build.bat`, `run.py` (frozen `_MEI*` sweep), `requirements-build.txt`, `packaging/*` (incl. `gt_bundle.marker`), README build section, `docs/ARCHITECTURE.md` (UI paragraph: line 63 still describes `ControlWindow(QMainWindow)`) |
| integration | last workstream to land: regenerates and commits `glasstranslate/ui/resources_rc.py` (then removes it from `.gitignore`), runs `pytest -q` and `build.py` |
Shared files are edited by exactly one owner; cross-workstream needs go through this document.
