#version 440
// Liquid Glass surface shader (docs/GLASS_DESIGN.md §2.2).
//
// One shader, two paths selected by the `solidMode` uniform (a float; Qt writes
// QML bool/int properties as raw int bits into a float slot, so every scalar
// here is bound from a QML `property real`):
//   glass path (solidMode 0): squircle-shaped refraction of `sharpSource` in a
//     rim band, frosted `frostSource` in the interior, material lift toward the
//     polarity base, cool lean, proximity specular, inner shadow.
//   fill path  (solidMode 1): translucent premultiplied `fillColor` with the same
//     shape, ambient/proximity specular and the optional HC border.
// Both samplers are read unconditionally (no dynamic branching around texture
// fetches); the fill path zeroes their contribution arithmetically.
//
// Colours from QML arrive PREMULTIPLIED by alpha: tintColor/borderColor are bound
// with alpha 1, fillColor is consumed as premultiplied (never multiplied by .a again).

layout(location = 0) in vec2 qt_TexCoord0;   // 0..1 across this item, (0,0) = top-left
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    vec2  itemSize;      // logical px
    float dpr;           // device px per logical px
    float radius;        // corner radius, logical px
    float squircle;      // superellipse exponent: 2 = circular, ~4.5 = continuous curvature
    float rimWidth;      // width of the refraction band, logical px
    float strength;      // max rim displacement as a fraction of rimWidth (0..1)
    float falloff;       // exponent of the rim profile
    float frost;         // 0 = sharp interior, 1 = fully frosted interior
    float frostLift;     // 0..1 pull of the interior toward the material base
    float coolLean;      // 0..1 strength of the cool multiply
    vec4  tintColor;     // premultiplied; bound with alpha 1
    float ink;           // material polarity 0..1: 0 = dark material, 1 = light material
    vec2  pointer;       // pointer position in item px from the centre
    float specular;      // 0..1 intensity of the edge highlight
    float shadowInner;   // 0..1 faint inner shading at the bottom edge
    float solidMode;     // 1 = fill path
    vec4  fillColor;     // premultiplied fill colour
    float border;        // border width in logical px (0 normally, 2 in high contrast)
    vec4  borderColor;   // bound with alpha 1
    vec4  rect;          // this item's rect in source UV space (x, y, w, h)
};
layout(binding = 1) uniform sampler2D sharpSource;
layout(binding = 2) uniform sampler2D frostSource;

// Signed distance to a superellipse-cornered box (negative inside).  `b` is the
// half size, `r` the corner radius, `n` the exponent.  Edges use the exact box
// distance; the corner square blends both axes with the Lp norm, which is the
// standard rounded box for n = 2 and a continuous-curvature squircle for n ~ 4.5.
float sdSquircle(vec2 p, vec2 b, float r, float n) {
    vec2 q = abs(p) - (b - vec2(r));
    vec2 qc = max(q, vec2(1e-4));                     // pow() needs a positive base
    float corner = pow(pow(qc.x, n) + pow(qc.y, n), 1.0 / n) - r;
    float edge = min(max(q.x, q.y), 0.0);
    return corner + edge;
}

void main() {
    vec2 uv = qt_TexCoord0;
    vec2 p = (uv - 0.5) * itemSize;                   // item px from the centre
    vec2 halfSize = 0.5 * itemSize;
    float n_exp = max(squircle, 1.0);
    float r = min(radius, min(halfSize.x, halfSize.y));

    // 1. shape, coverage and outward normal (2-sample forward difference)
    float d = sdSquircle(p, halfSize, r, n_exp);
    float aa = 0.75 / dpr;
    float inside = 1.0 - smoothstep(-aa, aa, d);
    float eps = 1.0;
    vec2 grad = vec2(sdSquircle(p + vec2(eps, 0.0), halfSize, r, n_exp) - d,
                     sdSquircle(p + vec2(0.0, eps), halfSize, r, n_exp) - d);
    vec2 nrm = grad / max(length(grad), 1e-4);

    // 2. rim profile and inward refraction (compresses at the rim, never folds
    //    while strength * falloff <= 0.9 - enforced by GlassSurface)
    float rw = max(rimWidth, 1e-3);
    float t = clamp(1.0 + d / rw, 0.0, 1.0);          // 1 at the edge, 0 at rimWidth inward
    float mag = strength * pow(max(t, 1e-6), max(falloff, 1e-3));
    vec2 uvS = uv + (-nrm * mag * rimWidth) / itemSize;
    // Sources are premultiplied RGBA.  The slab's backdrop is opaque (a = 1); a
    // pill's local track layer is translucent, and its alpha is carried through so
    // the thumb stays see-through where nothing lies beneath it.
    vec4 refracted = texture(sharpSource, rect.xy + uvS * rect.zw);
    vec2 keyLight = normalize(vec2(-0.6, -0.8));
    float lightFacing = 0.5 + 0.5 * dot(nrm, keyLight);
    refracted.rgb *= 1.0 + mag * mix(-0.10, 0.12, lightFacing);

    // 3. frosted interior, sharp bent content at the rim
    vec4 frosted = texture(frostSource, rect.xy + uv * rect.zw);
    vec4 glass = mix(refracted, frosted, frost * (1.0 - t));

    // 4. material: lift toward the polarity base, faint cool lean
    vec3 base = mix(vec3(0.11, 0.12, 0.14), vec3(0.97, 0.98, 1.00), ink);
    float lift = frostLift * (1.0 - 0.5 * t);
    vec3 col = mix(glass.rgb, base, lift);
    float a = mix(glass.a, 1.0, lift);
    col *= mix(vec3(1.0), tintColor.rgb, coolLean);

    // 6. inner shadow (depth cue at the bottom edge)
    col *= 1.0 - shadowInner * smoothstep(0.35, 1.0, uv.y) * t;

    // 7. fill path: premultiplied fillColor, no sampling
    col = mix(col, fillColor.rgb, solidMode);
    a = mix(a, fillColor.a, solidMode);

    // 5. specular: proximity two-lobe model + fixed key light on a 1-device-px line
    float R = max(0.55 * max(itemSize.x, itemSize.y), 60.0);
    float near = 1.0 - smoothstep(0.0, R, length(pointer - p));
    float far = 1.0 - smoothstep(0.0, R, length(-pointer - p));
    float lobe = near + 0.35 * far;
    float amb = max(dot(nrm, keyLight), 0.0);
    float sheen = clamp(0.45 * amb + 0.55 * lobe, 0.0, 1.0);
    float lw = 1.0 / dpr;
    float line = 1.0 - smoothstep(0.0, lw, abs(d + 0.5 * lw));
    float band = t * t;
    float spec = specular * (0.60 * line * sheen + 0.14 * band * lobe);
    col = min(col + vec3(spec), vec3(1.0));

    // 8. high-contrast border (border = 0 contributes nothing)
    float bw = (1.0 - smoothstep(border - aa, border + aa, -d)) * clamp(border, 0.0, 1.0);
    col = mix(col, borderColor.rgb, bw);
    a = mix(a, 1.0, bw);

    // 9. premultiplied output (glass path: a = 1 over the opaque backdrop)
    fragColor = vec4(col, a) * inside * qt_Opacity;
}
