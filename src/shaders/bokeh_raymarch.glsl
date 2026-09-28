#version 330 core
out vec4 fragColor;
in vec2 uv;

uniform sampler2D origTex;
uniform sampler2D depthTex;
uniform sampler2D maskTex;

uniform mat4 inv_mvp;
uniform mat4 mvp;
uniform vec2 u_srcScale; // Used if we need to scale uv

uniform float u_focalDepth;
uniform float u_focalRange;
uniform float u_fgGamma;
uniform float u_bgGamma;
uniform float u_hasSubjectMask;
uniform float u_maskInfluence;
uniform float u_maxCoC; // Max shift in pixels
uniform vec4 u_fboBounds; // (min_x, min_y, max_x, max_y)
uniform float u_apertureSides;
uniform float u_apertureRotation;
uniform float u_chromaticAberration;
uniform float u_highlightBoost;
uniform float u_vignette;
uniform float u_radialCenter;
uniform float u_radialEdge;
uniform float u_radialGamma;
uniform vec2 u_resolution;

uniform int u_frameCount; // For accumulation jitter
uniform int u_maxSamples; // Number of samples for complete Vogel spiral

// Pseudo-random number generator
float rand(vec2 co, int frame) {
    return fract(sin(dot(co + float(frame) * 0.61803398875, vec2(12.9898, 78.233))) * 43758.5453);
}

// Vogel's spiral or uniform disk sampling
vec2 getApertureOffset(vec2 co, int frame) {
    float golden_angle = 2.39996323; // 2 * PI * (1 - 1/phi)
    
    // Create a random starting index for this pixel [0, u_maxSamples-1]
    int start_index = int(rand(co, 0) * float(u_maxSamples));
    
    // Calculate the actual index in the Vogel sequence
    int index = (frame + start_index) % u_maxSamples;
    int sample_index = index + 1; // 1 to 64
    
    float theta = float(sample_index) * golden_angle;
    float r = sqrt(float(sample_index) / float(u_maxSamples));
    
    // To prevent quantization artifacts while keeping the global polygon shape,
    // we can add a very small per-pixel jitter to r and theta, but NOT a full rotation.
    float jitter_r = (rand(co, 1) - 0.5) / float(u_maxSamples);
    float jitter_theta = (rand(co, 2) - 0.5) * 0.1;
    
    theta += jitter_theta;
    r = clamp(r + jitter_r, 0.0, 1.0);
    
    vec2 offset = vec2(cos(theta), sin(theta)) * r;
    
    // Apply aperture polygon shape (petals)
    if (u_apertureSides > 2.5) {
        float angle = atan(offset.y, offset.x);
        angle += 3.14159265 / 2.0; // Rotate so one petal points up
        float slice = 3.14159265 * 2.0 / u_apertureSides;
        float p = mod(angle, slice) - slice / 2.0;
        float r_poly = cos(slice / 2.0) / cos(p);
        offset *= r_poly;
    }
    
    // Rotate the entire resulting aperture shape
    if (abs(u_apertureRotation) > 0.001) {
        float rot_cos = cos(u_apertureRotation);
        float rot_sin = sin(u_apertureRotation);
        offset = vec2(offset.x * rot_cos - offset.y * rot_sin, offset.x * rot_sin + offset.y * rot_cos);
    }
    
    return offset;
}

float get_depth(vec2 sample_uv) {
    if (textureSize(depthTex, 0).x <= 1) return 1.0;
    
    // Dilation (Foreground fanning) to prevent edge pixels from bleeding into the background
    // Edge pixels have mixed colors (anti-aliasing). If they get background depth, their bright colors 
    // will scatter and create a bloom/halo effect. We push the foreground depth slightly outwards.
    vec2 texel = 4.0 / vec2(textureSize(depthTex, 0));
    float d1 = texture(depthTex, sample_uv).r;
    float d2 = texture(depthTex, sample_uv + vec2(texel.x, 0.0)).r;
    float d3 = texture(depthTex, sample_uv + vec2(-texel.x, 0.0)).r;
    float d4 = texture(depthTex, sample_uv + vec2(0.0, texel.y)).r;
    float d5 = texture(depthTex, sample_uv + vec2(0.0, -texel.y)).r;
    float d6 = texture(depthTex, sample_uv + vec2(texel.x, texel.y)).r;
    float d7 = texture(depthTex, sample_uv + vec2(-texel.x, texel.y)).r;
    float d8 = texture(depthTex, sample_uv + vec2(texel.x, -texel.y)).r;
    float d9 = texture(depthTex, sample_uv + vec2(-texel.x, -texel.y)).r;
    float raw_depth = max(d1, max(max(d2, d3), max(d4, d5)));
    raw_depth = max(raw_depth, max(max(d6, d7), max(d8, d9)));
    
    raw_depth = pow(raw_depth, 2.0); // Linearize
    float z = raw_depth; // 1.0 is near (Foreground), 0.0 is far (Background)
    
    if (u_hasSubjectMask > 0.5 && textureSize(maskTex, 0).x > 1) {
        float mask = texture(maskTex, sample_uv).r;
        if (mask > 0.0) {
            // Force the subject to be at the focal plane
            z = mix(z, u_focalDepth, mask * u_maskInfluence);
        }
    }
    return clamp(z, 0.0, 1.0);
}

// Returns CoC in range [-1.0, 1.0]
float get_coc(float Z) {
    float dist = Z - u_focalDepth;
    float coc = 0.0;
    
    // Z = 1.0 is Near (Foreground). So dist > 0 is Foreground.
    if (dist > u_focalRange) { // Foreground
        float raw_coc = (dist - u_focalRange) / max(0.01, 1.0 - (u_focalDepth + u_focalRange));
        coc = pow(clamp(raw_coc, 0.0, 1.0), u_fgGamma);
    } else if (dist < -u_focalRange) { // Background
        float raw_coc = -(dist + u_focalRange) / max(0.01, u_focalDepth - u_focalRange);
        coc = -pow(clamp(raw_coc, 0.0, 1.0), u_bgGamma);
    }
    
    return clamp(coc, -1.0, 1.0);
}

void main() {
    // Just use uv so that in_texcoord doesn't get optimized out by the compiler
    vec2 dummy = uv;
    vec2 lensOffset = getApertureOffset(gl_FragCoord.xy, u_frameCount);
    
    // Calculate the screen position in NDC [-1, 1]
    vec2 ndc = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    
    // Apply Radial / Swirly Stretch
    if (abs(u_radialCenter) > 0.01 || abs(u_radialEdge) > 0.01) {
        vec2 dir = normalize(ndc + 1e-5);
        float rad_comp = dot(lensOffset, dir);
        vec2 rad_vec = dir * rad_comp;
        vec2 tan_vec = lensOffset - rad_vec;
        
        float dist = clamp(length(ndc), 0.0, 1.0);
        float factor = pow(dist, max(0.1, u_radialGamma));
        float stretch = mix(u_radialCenter, u_radialEdge, factor);
        
        rad_vec *= exp(-stretch);
        tan_vec *= exp(stretch);
        
        lensOffset = rad_vec + tan_vec;
    }
    
    // Calculate the base UV in original image space
    vec4 orig_pos = inv_mvp * vec4(ndc, 0.0, 1.0);
    vec2 base_orig_uv = (orig_pos.xy / orig_pos.w) * 0.5 + 0.5;
    
    // UV in origTex (the FBO). The FBO texture is 4096x4096, not the viewport size!
    vec2 fbo_tex_size = vec2(textureSize(origTex, 0));
    vec2 fbo_uv = gl_FragCoord.xy / fbo_tex_size;
    vec2 viewport_uv = gl_FragCoord.xy / u_resolution;
    
    // Max shift in FBO UV coordinates and Viewport UV coordinates
    vec2 shift_px = lensOffset * u_maxCoC;
    vec2 maxShiftFboUV = shift_px / fbo_tex_size;
    vec2 maxShiftViewportUV = shift_px / u_resolution;
    
    // Calculate the shifted NDC
    vec2 shifted_ndc = ((gl_FragCoord.xy + shift_px) / u_resolution) * 2.0 - 1.0;
    vec4 shifted_orig_pos = inv_mvp * vec4(shifted_ndc, 0.0, 1.0);
    vec2 shifted_orig_uv = (shifted_orig_pos.xy / shifted_orig_pos.w) * 0.5 + 0.5;
    
    // The max shift in original UV coordinates
    vec2 maxShiftOrigUV = shifted_orig_uv - base_orig_uv;
    
    // Read the valid image bounds in viewport UV space from the uniform
    vec2 valid_min = u_fboBounds.xy;
    vec2 valid_max = u_fboBounds.zw;
    
    // If the current pixel is outside the valid image bounds (in the black bars),
    // it should not gather bokeh from the image, nor should it blur. 
    // We just return its original color (which is the black bar).
    bool in_image = (viewport_uv.x >= valid_min.x && viewport_uv.x <= valid_max.x && 
                     viewport_uv.y >= valid_min.y && viewport_uv.y <= valid_max.y);
                     
    if (!in_image) {
        fragColor = vec4(texture(origTex, fbo_uv).rgb, 1.0 / float(max(1, u_frameCount)));
        return;
    }

    // 2. Gather loop using Guaranteed Zero-Crossing method
    const int STEPS = 48;
    float best_z = -1.0;
    vec2 best_uv = fbo_uv;
    
    vec2 sample_uv_prev = base_orig_uv + maxShiftOrigUV * (-1.0);
    float depth_prev = get_depth(sample_uv_prev);
    float f_prev = -get_coc(depth_prev) - (-1.0);
    
    float center_depth = get_depth(base_orig_uv);
    float expected_t_center = -get_coc(center_depth);
    
    float closest_dist = 999.0;
    vec2 fallback_uv = fbo_uv; // Default to center if completely occluded
    
    for (int i = 1; i <= STEPS; ++i) {
        float t = mix(-1.0, 1.0, float(i) / float(STEPS));
        vec2 sample_uv = base_orig_uv + maxShiftOrigUV * t;
        float map_z = get_depth(sample_uv);
        float expected_t = -get_coc(map_z);
        float f_curr = expected_t - t;
        
        // Background Extension: track the pixel on the same surface that is CLOSEST to expected_t_center
        if (abs(map_z - center_depth) < 0.1) {
            float dist = abs(t - expected_t_center);
            if (dist < closest_dist) {
                closest_dist = dist;
                fallback_uv = fbo_uv + maxShiftFboUV * t;
            }
        }
        
        if (f_curr * f_prev <= 0.0) {
            // Validate: ignore false roots caused by steep depth discontinuities (edges)
            if (abs(map_z - depth_prev) < 0.1) {
                float w = f_prev / (f_prev - f_curr + 1e-6);
                float exact_t = mix(t - 2.0/float(STEPS), t, clamp(w, 0.0, 1.0));
                vec2 exact_sample_uv = base_orig_uv + maxShiftOrigUV * exact_t;
                vec2 exact_sample_fbo = fbo_uv + maxShiftFboUV * exact_t;
                vec2 exact_sample_viewport = viewport_uv + maxShiftViewportUV * exact_t;
                
                bool valid = true;
                if (exact_sample_uv.x < 0.0 || exact_sample_uv.x > 1.0 || exact_sample_uv.y < 0.0 || exact_sample_uv.y > 1.0) valid = false;
                if (exact_sample_viewport.x < valid_min.x || exact_sample_viewport.x > valid_max.x || exact_sample_viewport.y < valid_min.y || exact_sample_viewport.y > valid_max.y) valid = false;
                
                if (valid && map_z > best_z) {
                    best_z = map_z;
                    best_uv = exact_sample_fbo;
                }
            }
        }
        f_prev = f_curr;
        depth_prev = map_z;
    }
    
    if (best_z < 0.0) {
        // If the ray was occluded by a foreground object and no valid root was found,
        // use the fallback pixel (the furthest valid background pixel along the ray).
        // This acts as background extension/inpainting, correctly blurring the background
        // behind the foreground object without sampling the sharp center pixel.
        best_uv = fallback_uv;
    }
    
    // Valid bounds in FBO UV space
    vec2 fbo_tex_size_local = vec2(textureSize(origTex, 0));
    vec2 fbo_valid_min = valid_min * (u_resolution / fbo_tex_size_local);
    vec2 fbo_valid_max = valid_max * (u_resolution / fbo_tex_size_local);
    
    vec2 hit_fbo_uv = clamp(best_uv, fbo_valid_min, fbo_valid_max);
    
    // Optical Vignette (Cat's eye effect)
    float cat_eye = 1.0;
    if (u_vignette > 0.01) {
        float occlusion = max(0.0, dot(lensOffset, ndc));
        float threshold = 1.5 - u_vignette * 1.5; // u_vignette=1 -> 0.0, u_vignette=0 -> 1.5
        cat_eye = smoothstep(threshold + 0.3, threshold, occlusion);
    }
    
    // Chromatic aberration
    vec3 final_rgb;
    if (u_chromaticAberration > 0.01) {
        float ca = u_chromaticAberration * 0.2;
        vec2 hit_uv_r = clamp(fbo_uv + (best_uv - fbo_uv) * (1.0 + ca), fbo_valid_min, fbo_valid_max);
        vec2 hit_uv_b = clamp(fbo_uv + (best_uv - fbo_uv) * (1.0 - ca), fbo_valid_min, fbo_valid_max);
        
        // Prevent sampling across depth discontinuities!
        vec2 ratio = vec2(1.0);
        if (abs(maxShiftFboUV.x) > 1e-8) ratio.x = maxShiftOrigUV.x / maxShiftFboUV.x;
        if (abs(maxShiftFboUV.y) > 1e-8) ratio.y = maxShiftOrigUV.y / maxShiftFboUV.y;
        
        vec2 orig_uv_r = base_orig_uv + (hit_uv_r - fbo_uv) * ratio;
        vec2 orig_uv_b = base_orig_uv + (hit_uv_b - fbo_uv) * ratio;
        
        // If the shifted pixel is on a different surface (e.g. hits the foreground), fall back to the safe hit_fbo_uv
        if (abs(get_depth(orig_uv_r) - center_depth) > 0.1) hit_uv_r = hit_fbo_uv;
        if (abs(get_depth(orig_uv_b) - center_depth) > 0.1) hit_uv_b = hit_fbo_uv;
        
        float r = texture(origTex, hit_uv_r).r;
        float g = texture(origTex, hit_fbo_uv).g;
        float b = texture(origTex, hit_uv_b).b;
        final_rgb = vec3(r, g, b);
    } else {
        final_rgb = texture(origTex, hit_fbo_uv).rgb;
    }
    
    // Highlight Boost
    if (u_highlightBoost > 0.0) {
        float hit_z = best_z >= 0.0 ? best_z : get_depth(base_orig_uv);
        float defocus = abs(get_coc(hit_z));
        
        float luma = dot(final_rgb * 50.0, vec3(0.299, 0.587, 0.114));
        float boost_factor = smoothstep(0.7, 1.0, luma) * u_highlightBoost * defocus;
        final_rgb += final_rgb * boost_factor;
    }
    
    fragColor = vec4(final_rgb * cat_eye, 1.0 / float(max(1, u_frameCount))) + vec4(uv * 0.00000001, 0.0, 0.0);
}
