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

uniform sampler2D accum_tex; // Base or accumulated image
uniform sampler2D layer_tex; // This layer's processed image
uniform sampler2D orig_tex;  // Original linear image (for luma/color parametric mask)
uniform sampler2D mask_tex;  // Painted mask (R channel)
uniform sampler2D global_depth_tex; // Global depth map (0-1)
uniform sampler2D global_normal_tex; // Global normal map (RGB 0-1)

uniform mat4 inv_mvp;

// Parametric settings
uniform vec3 mask_color_target; // luma, sat, hue target
uniform float mask_color_range; // 0.0 to 1.0 (Hue range)
uniform float mask_sat_range;   // 0.0 to 1.0
uniform float mask_luma_target; // 0.0 to 1.0
uniform float mask_luma_range;  // 0.0 to 1.0
uniform float mask_softness;    // 0.0 to 1.0
uniform float mask_opacity;     // 0.0 to 1.0 layer overall opacity

// 3D Geometry Mask Settings
uniform float use_depth_mask;
uniform float depth_min;
uniform float depth_max;
uniform float depth_softness;
uniform float depth_gamma;

uniform float use_normal_mask;
uniform vec3 normal_dir;
uniform float normal_angle;
uniform float normal_softness;
uniform float normal_gamma;

uniform float show_mask; // Output grayscale mask
uniform float fill_mask; // Fill painted mask

// Gradient Masking
uniform float gradient_type; // 0=None, 1=L2R, 2=R2L, 3=T2B, 4=B2T, 5=Radial
uniform float grad_black;
uniform float grad_white;
uniform float grad_gamma;

vec3 linear_srgb_to_oklab(vec3 c) {
    float l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    float m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    float s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    float l_ = sign(l) * pow(abs(l), 1.0/3.0);
    float m_ = sign(m) * pow(abs(m), 1.0/3.0);
    float s_ = sign(s) * pow(abs(s), 1.0/3.0);
    return vec3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    );
}

vec3 oklab_to_oklch(vec3 lab) {
    float C = sqrt(lab.y * lab.y + lab.z * lab.z);
    float H = atan(lab.z, lab.y);
    if (H < 0.0) H += 6.28318530718;
    return vec3(lab.x, C, H);
}

void main() {
    vec2 accum_tc = gl_FragCoord.xy / vec2(textureSize(accum_tex, 0));
    vec4 accum = texture(accum_tex, accum_tc);
    vec4 layer = texture(layer_tex, accum_tc);
    
    vec2 ndc = v_texcoord * 2.0 - 1.0;
    vec4 orig_clip = inv_mvp * vec4(ndc, 0.0, 1.0);
    vec2 orig_uv = (orig_clip.xy / orig_clip.w) * 0.5 + 0.5;
    
    float brush = clamp(texture(mask_tex, orig_uv).r, 0.0, 1.0);
    brush = clamp(max(brush, fill_mask), 0.0, 1.0); // Fill mask allows overriding the painted mask
    
    // Evaluate Gradient
    float grad = 1.0;
    if (gradient_type > 0.5) {
        float val = 0.0;
        if (gradient_type < 1.5) { // L2R
            val = orig_uv.x;
        } else if (gradient_type < 2.5) { // R2L
            val = 1.0 - orig_uv.x;
        } else if (gradient_type < 3.5) { // T2B
            val = 1.0 - orig_uv.y; // orig_uv is bottom-left origin in OpenGL
        } else if (gradient_type < 4.5) { // B2T
            val = orig_uv.y;
        } else if (gradient_type < 5.5) { // Radial
            vec2 center = vec2(0.5, 0.5);
            val = 1.0 - clamp(length(orig_uv - center) * 2.0, 0.0, 1.0);
        }
        
        // Levels on gradient
        val = clamp((val - grad_black) / max(0.0001, grad_white - grad_black), 0.0, 1.0);
        // Gamma (Midtone) adjustment
        grad = pow(val, 1.0 / max(0.001, grad_gamma * 2.0));
    }
    
    float alpha = clamp(brush * mask_opacity * grad, 0.0, 1.0);
    
    // Parametric masking if ranges are valid (> 0)
    if (mask_luma_range > 0.0 || mask_color_range > 0.0 || mask_sat_range > 0.0) {
        vec3 orig = texture(orig_tex, orig_uv).rgb;
        vec3 oklch = oklab_to_oklch(linear_srgb_to_oklab(orig));
        
        // Define softness transition width (at least 0.001 to avoid division by zero in smoothstep)
        // If softness is 0.0, it should be a hard edge. If 1.0, it should be a very soft edge.
        
        if (mask_luma_range > 0.0) {
            float dist = abs(oklch.x - mask_luma_target); // L is 0-1
            float edge = mask_luma_range;
            float inner = max(0.0, edge * (1.0 - mask_softness));
            float luma_alpha = 1.0 - smoothstep(inner, edge + 0.0001, dist);
            alpha *= luma_alpha;
        }
        
        if (mask_color_range > 0.0) {
            // Hue distance
            float dh = abs(oklch.z - mask_color_target.z);
            if (dh > 3.14159265) dh = 6.28318530718 - dh;
            float hue_dist = dh / 3.14159265;
            
            float edge = mask_color_range;
            float inner = max(0.0, edge * (1.0 - mask_softness));
            float color_alpha = 1.0 - smoothstep(inner, edge + 0.0001, hue_dist);
            alpha *= color_alpha;
        }
        
        if (mask_sat_range > 0.0) {
            // Saturation (Chroma) distance
            // Chroma is typically up to 0.4. We'll map the UI 0-1 range to 0.0-0.4
            float target_c = mask_color_target.y * 0.4;
            float dist = abs(oklch.y - target_c);
            
            float edge = mask_sat_range * 0.4;
            float inner = max(0.0, edge * (1.0 - mask_softness));
            float sat_alpha = 1.0 - smoothstep(inner, edge + 0.0001, dist);
            alpha *= sat_alpha;
        }
    }
    
    // 3D Depth Masking
    // Invert depth so 0.0 is front (near camera) and 1.0 is back (far background / horizon).
    float d = 1.0 - texture(global_depth_tex, orig_uv).r;
    float d_alpha = 0.0;
    if (depth_max > depth_min) {
        float eff_min = clamp(depth_min, 0.0, 1.0);
        float eff_max = clamp(depth_max, 0.0, 1.0);
        float soft = clamp(depth_softness, 0.001, 1.0);
        
        float near_disable = (depth_min < 0.0) ? clamp(-depth_min / 0.10, 0.0, 1.0) : 0.0;
        float far_disable = (depth_max > 1.0) ? clamp((depth_max - 1.0) / 0.10, 0.0, 1.0) : 0.0;
        
        float span = max(0.0001, eff_max - eff_min);
        float c = (eff_min + eff_max) * 0.5;
        if (far_disable > 0.0 && near_disable == 0.0) {
            c = eff_min + span * (0.5 + 0.5 * far_disable);
        } else if (near_disable > 0.0 && far_disable == 0.0) {
            c = eff_min + span * (0.5 - 0.5 * near_disable);
        }
        
        float L_near = max(0.0001, c - eff_min);
        float W_near = max(0.0001, L_near * soft);
        
        float L_far = max(0.0001, eff_max - c);
        float W_far = max(0.0001, L_far * soft);
        
        float val = 0.0;
        if (d <= c) {
            float near_val = 0.0;
            if (d >= eff_min + W_near) {
                near_val = 1.0;
            } else if (d >= eff_min) {
                float u = (d - eff_min) / W_near;
                near_val = smoothstep(0.0, 1.0, u);
            }
            val = mix(near_val, 1.0, near_disable);
        } else {
            float far_val = 0.0;
            if (d <= eff_max - W_far) {
                far_val = 1.0;
            } else if (d <= eff_max) {
                float u = (eff_max - d) / W_far;
                far_val = smoothstep(0.0, 1.0, u);
            }
            val = mix(far_val, 1.0, far_disable);
        }
        d_alpha = pow(clamp(val, 0.0, 1.0), max(0.01, depth_gamma));
    }
    alpha *= mix(1.0, d_alpha, float(use_depth_mask > 0.5));
    
    // 3D Normal Masking
    vec3 n_rgb = texture(global_normal_tex, orig_uv).rgb;
    // Convert from [0, 1] to [-1, 1] and normalize
    vec3 n = normalize(n_rgb * 2.0 - 1.0);
    vec3 light_dir = normalize(normal_dir);
    
    // Dot product gives cosine of angle
    float dp = dot(n, light_dir);
    // Map dot product [-1, 1] to [0, 1] where 1 is directly facing light
    float facing = (dp + 1.0) * 0.5;
    
    // target facing depends on normal_angle. If normal_angle is 1.0, we select everything.
    float threshold = 1.0 - normal_angle;
    
    float n_inner = max(0.0, threshold - normal_softness);
    float n_alpha = smoothstep(n_inner, threshold + 0.0001, facing);
    
    n_alpha = pow(n_alpha, normal_gamma);
    
    alpha *= mix(1.0, n_alpha, float(use_normal_mask > 0.5));
    
    if (show_mask > 0.5) {
        if (orig_uv.x < 0.0 || orig_uv.x > 1.0 || orig_uv.y < 0.0 || orig_uv.y > 1.0) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0); // Or whatever background color is appropriate, 0.0 is black
        } else {
            fragColor = vec4(vec3(alpha) * 0.02, 1.0);
        }
    } else {
        fragColor = vec4(mix(accum.rgb, layer.rgb, alpha), 1.0);
    }
}
#endif
