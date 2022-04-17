#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {
    /*
    vec2 pc = gl_PointCoord * 2. - 1.;
    const float w = 0.2;
    // TODO: Make it SUS
    float r = atan(fragColor.y, fragColor.x);
    float k = atan(pc.y, pc.x);
    bool disc = length(pc) < 0.5;// && (r < k);//abs(pc.y) > w && abs(pc.x) > w;
    vec3 color = vec3(fragColor) + vec3(0.5);

    if (length(pc) > 1.) discard;
    if (!disc) color = vec3(0.);
    */
    vec2 i_point_c = ivec2(gl_PointCoord * 5.);

    bool amogus = 
        i_point_c == ivec2(0, 0)
        || i_point_c == ivec2(0, 4)
        || i_point_c == ivec2(0, 3)
        || i_point_c == ivec2(2, 4)
        || i_point_c.x == 4;

    bool visor = i_point_c == ivec2(3, 1)
        || i_point_c == ivec2(2, 1);

    amogus = amogus || visor;

    //if (amogus) discard;

    vec3 color = vec3(fragColor) + vec3(0.5);

    color *= vec3(!amogus);

    outColor = vec4(color, 1.0);
}
