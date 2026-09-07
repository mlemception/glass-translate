#version 440
// 9-tap separable Gaussian (docs/GLASS_DESIGN.md §2.2).  `step` is one tap
// spacing in source UV: (1/texW, 0) for the horizontal pass, (0, 1/texH) for
// the vertical one.  Weights sum to 0.9996.  Run H, V, H, V at 1/4 resolution
// for sigma ~ 10 logical px.

layout(location = 0) in vec2 qt_TexCoord0;
layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4  qt_Matrix;
    float qt_Opacity;
    vec2  step;
};
layout(binding = 1) uniform sampler2D source;

void main() {
    vec2 uv = qt_TexCoord0;
    vec4 c = texture(source, uv) * 0.2270;
    c += (texture(source, uv + step) + texture(source, uv - step)) * 0.1945;
    c += (texture(source, uv + 2.0 * step) + texture(source, uv - 2.0 * step)) * 0.1216;
    c += (texture(source, uv + 3.0 * step) + texture(source, uv - 3.0 * step)) * 0.0540;
    c += (texture(source, uv + 4.0 * step) + texture(source, uv - 4.0 * step)) * 0.0162;
    fragColor = c * qt_Opacity;
}
