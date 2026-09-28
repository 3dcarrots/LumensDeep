#version 330 core

in vec2 in_position;
in vec2 in_texcoord;

out vec2 uv;
out vec2 crop_ndc;

// Model-View-Projection matrix for geometric transformations (crop, rotate, pan, scale)
uniform mat4 mvp;
uniform mat4 mvp_base;

void main() {
    uv = in_texcoord;
    crop_ndc = (mvp_base * vec4(in_position, 0.0, 1.0)).xy;
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
}
