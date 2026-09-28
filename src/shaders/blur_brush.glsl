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

uniform sampler2D mask_tex;
uniform vec2 brush_pos; // 0 to 1
uniform float brush_radius; // in 0 to 1 UV space
uniform float brush_hardness; // 0.0 to 1.0
uniform float aspect_ratio; // width / height

void main() {
    vec2 delta = v_texcoord - brush_pos;
    delta.x *= aspect_ratio;
    float dist = length(delta);
    
    float soft_edge = mix(brush_radius, 0.0, brush_hardness);
    float alpha = 1.0 - smoothstep(brush_radius - soft_edge, brush_radius + 0.001, dist);
    
    // Only blur if within brush radius
    if (alpha <= 0.0) {
        fragColor = texture(mask_tex, v_texcoord);
        return;
    }
    
    // Simple 9-tap box blur or Gaussian
    vec2 texel = 1.0 / vec2(textureSize(mask_tex, 0));
    float sum = 0.0;
    
    // Since this runs per frame and mask is 1/4 resolution, a medium sized kernel is okay.
    // Let's do a fast 5x5 blur
    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            sum += texture(mask_tex, v_texcoord + vec2(float(i)*texel.x, float(j)*texel.y)).r;
        }
    }
    float blurred = sum / 25.0;
    
    float orig = texture(mask_tex, v_texcoord).r;
    
    // Blend between original and blurred based on brush alpha (intensity of brush)
    // Actually, to make it blur faster, we can just mix it directly using alpha
    // But we might want the blur to be stronger. 5x5 is somewhat soft.
    float final_val = clamp(mix(orig, blurred, alpha), 0.0, 1.0);
    
    fragColor = vec4(final_val, 0.0, 0.0, 1.0);
}
#endif
