#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 pc = gl_PointCoord * 2. - 1.;
    if (length(pc) > 1.) discard;
    outColor = vec4(fragColor, 1.0);
}
