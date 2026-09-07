#version 440
// Analytic Gaussian-blurred rounded-box shadow (Evan Wallace's closed form,
// docs/GLASS_DESIGN.md §2.2).  Drawn on an item inflated around the shadowed
// box by 3*sigma + |offset| on each axis (GlassShadow.qml), so:
//   itemSize = this effect's size (logical px)
//   box half size = itemSize/2 - (3*sigma + abs(offset))
//   box centre    = effect centre + offset
// The box itself is knocked out (CSS box-shadow semantics) so a translucent
// fill drawn on top is never darkened from underneath.  `color` arrives
// premultiplied; the mask scales it as a whole.

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    vec2  itemSize;   // effect size, logical px
    float radius;     // corner radius of the shadowed box
    float sigma;      // blur sigma, logical px
    vec2  offset;     // shadow offset (x, y), logical px
    vec4  color;      // premultiplied shadow colour
};

float gaussian(float x, float s) {
    const float pi = 3.141592653589793;
    return exp(-(x * x) / (2.0 * s * s)) / (sqrt(2.0 * pi) * s);
}

// Abramowitz-Stegun error function approximation, vectorised over two values.
vec2 erf2(vec2 x) {
    vec2 s = sign(x), a = abs(x);
    x = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
    x *= x;
    return s - s / (x * x);
}

// Blurred coverage along x of the rounded box slice at height y.
float boxShadowX(float x, float y, float s, float corner, vec2 halfSize) {
    float delta = min(halfSize.y - corner - abs(y), 0.0);
    float curved = halfSize.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
    vec2 integral = 0.5 + 0.5 * erf2((x + vec2(-curved, curved)) * (sqrt(0.5) / s));
    return integral.y - integral.x;
}

// Blurred rounded-box mask at `p` (box centred at the origin), 4 samples along y.
float roundedBoxShadow(vec2 halfSize, vec2 p, float s, float corner) {
    float low = p.y - halfSize.y;
    float high = p.y + halfSize.y;
    float start = clamp(-3.0 * s, low, high);
    float end = clamp(3.0 * s, low, high);
    float stepY = (end - start) / 4.0;
    float y0 = start + stepY * 0.5;
    float y1 = y0 + stepY;
    float y2 = y1 + stepY;
    float y3 = y2 + stepY;
    float v = boxShadowX(p.x, p.y - y0, s, corner, halfSize) * gaussian(y0, s);
    v += boxShadowX(p.x, p.y - y1, s, corner, halfSize) * gaussian(y1, s);
    v += boxShadowX(p.x, p.y - y2, s, corner, halfSize) * gaussian(y2, s);
    v += boxShadowX(p.x, p.y - y3, s, corner, halfSize) * gaussian(y3, s);
    return v * stepY;
}

// Exact rounded-box distance for the knockout of the box itself.
float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + vec2(r);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0))) - r;
}

void main() {
    float s = max(sigma, 0.5);
    vec2 pad = vec2(3.0 * s) + abs(offset);
    vec2 halfBox = max(0.5 * itemSize - pad, vec2(0.5));
    float corner = min(radius, min(halfBox.x, halfBox.y));
    vec2 p = (qt_TexCoord0 - 0.5) * itemSize;        // effect px from the effect centre
    float mask = roundedBoxShadow(halfBox, p - offset, s, corner);
    float knockout = smoothstep(-0.75, 0.75, sdRoundBox(p, halfBox, corner));
    fragColor = color * (mask * knockout) * qt_Opacity;
}
