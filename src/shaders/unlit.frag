#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

const float pi = 3.141592;
const float tau = 2. * pi;

vec3 pixel(vec2 coord) {
    vec2 st = coord;//(coord/u_resolution.xy) * 2. - 1.;
    //st.x *= u_resolution.x/u_resolution.y;

    //st *= 0.002;
    //st -= vec2(0.330,-0.100) * 3.;
    //st -= vec2(0.900,-0.490) / 50.;

    const float scale = pow(0.080, 2.);
    st *= scale;
 	st -= vec2((-0.413), -0.216);


    float a = st.x;
    float b = st.y;
    int finish = 0;
    const int steps = 200;
    for (int i = 0; i < steps; i++) {
        float tmp = a * a - b * b + st.x;
        //float tmp = b * b - a * a + st.x;
        b = 2. * a * b + st.y;
        a = tmp;
        if (a > 9e9 || b > 9e9) {
			finish = i;
            break;
        }
    }
    if (finish == 0) return vec3(0);

    float g = float(finish) / float(steps);
    g = pow(g, .5);
    g = cos(g * tau * 3.);

    vec3 color = mix(
        mix(vec3(0.840,0.056,0.461), vec3(0.785,0.157,0.082) * 2., g),
        mix(vec3(1.000,0.886,0.491) * -0.880, vec3(0.179,0.022,0.530), g),
        g
    );


    return color;
}


void main() {
    vec2 c = fragColor.xy;

    const int AA_DIVS = 1;
    const int AA_WIDTH = AA_DIVS*2+1;
    vec3 color = vec3(0.);
 	for (int x = -AA_DIVS; x <= AA_DIVS; x++) {
        for (int y = -AA_DIVS; y <= AA_DIVS; y++) {
        	vec2 off = vec2(x, y) / float(AA_WIDTH);
            color += pixel(c + off / 1000.);
        }
    }
    color /= float(AA_WIDTH*AA_WIDTH);
    outColor = vec4(color, 1.);
}