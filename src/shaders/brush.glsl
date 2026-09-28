#version 330

#if defined VERTEX_SHADER
in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

void main() {
    v_texcoord = in_texcoord;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
#endif

#if defined FRAGMENT_SHADER
in vec2 v_texcoord;
out vec4 fragColor;

uniform vec2 brush_pos; // 0 to 1
uniform float brush_radius; // in 0 to 1 UV space
uniform float brush_hardness; // 0.0 to 1.0
uniform float brush_opacity; 
uniform float aspect_ratio; // width / height to keep brush circular

void main() {
    vec2 delta = v_texcoord - brush_pos;
    delta.x *= aspect_ratio;
    
    float dist = length(delta);
    
    // Smoothstep based on hardness
    float soft_edge = mix(brush_radius, 0.0, brush_hardness);
    float alpha = 1.0 - smoothstep(brush_radius - soft_edge, brush_radius + 0.001, dist);
    
    float val = clamp(brush_opacity * alpha, 0.0, 1.0);
    fragColor = vec4(val);
}
#endif
