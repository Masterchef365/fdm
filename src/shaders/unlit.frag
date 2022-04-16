#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 pc = gl_PointCoord * 2. - 1.;
    const float w = 0.2;
    // TODO: Make it SUS
    float r = atan(fragColor.y, fragColor.x);
    float k = atan(pc.y, pc.x);
    bool disc = length(pc) < 1. && (r < k);//abs(pc.y) > w && abs(pc.x) > w;
    vec3 color = vec3(fragColor) + vec3(0.5);

    //if (!disc) discard;
    if (!disc) color = vec3(0.);
    outColor = vec4(color, 1.0);
}
