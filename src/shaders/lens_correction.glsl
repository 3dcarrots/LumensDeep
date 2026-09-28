#version 330 core
out vec4 FragColor;
in vec2 uv;

uniform sampler2D image_texture;
uniform sampler2DArray uv_map;

// 0: disable, 1: geometry + TCA (3-channel photo), 2: geometry only (Depth, Normals, Subject Mask)
uniform int apply_correction; 

void main() {
    if (apply_correction == 0) {
        FragColor = texture(image_texture, uv);
        return;
    }
    
    // Inset by half a texel to strictly prevent border wrap/reflection artifacts
    vec2 tex_size = vec2(textureSize(image_texture, 0));
    vec2 half_texel = vec2(1.5) / max(vec2(1.0), tex_size);
    vec2 min_bound = half_texel;
    vec2 max_bound = vec2(1.0) - half_texel;
    
    vec2 clamped_uv = clamp(uv, min_bound, max_bound);
    
    if (apply_correction == 2) {
        // Pure geometry distortion for Depth Map, Normal Map, Subject Mask (anchored to Green geometry channel)
        vec2 uvGeom = clamp(texture(uv_map, vec3(clamped_uv, 1.0)).rg, min_bound, max_bound);
        FragColor = texture(image_texture, uvGeom);
        return;
    }
    
    // Full geometry + TCA for 3-channel RGB image
    vec2 uvR = clamp(texture(uv_map, vec3(clamped_uv, 0.0)).rg, min_bound, max_bound);
    vec2 uvG = clamp(texture(uv_map, vec3(clamped_uv, 1.0)).rg, min_bound, max_bound);
    vec2 uvB = clamp(texture(uv_map, vec3(clamped_uv, 2.0)).rg, min_bound, max_bound);
    
    float r = texture(image_texture, uvR).r;
    float g = texture(image_texture, uvG).g;
    float b = texture(image_texture, uvB).b;
    float a = texture(image_texture, uvG).a;
    
    FragColor = vec4(r, g, b, a);
}
