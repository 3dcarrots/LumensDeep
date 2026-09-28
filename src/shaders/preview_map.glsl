#version 330 core

#if defined VERTEX_SHADER
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

uniform mat4 mvp;

void main() {
    v_texcoord = in_texcoord;
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
}
#endif

#if defined FRAGMENT_SHADER
in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D tex;
uniform int map_type; // 0=depth/grayscale, 1=normals
uniform int has_tex; // 1 if texture valid, 0 if empty

void main() {
    if (has_tex == 0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec4 col = texture(tex, v_texcoord);
    if (map_type == 1) {
        // Normals: already in [0, 1] format
        fragColor = vec4(col.rgb * 0.02, 1.0);
    } else {
        // Depth/Mask: Grayscale
        fragColor = vec4(vec3(col.r * 0.02), 1.0);
    }
}
#endif
