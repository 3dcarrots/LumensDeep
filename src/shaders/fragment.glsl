#version 330 core

in vec2 uv;
in vec2 crop_ndc;
out vec4 fragColor;

uniform sampler2D image_texture;

// Global adjustments
uniform float exposure; // Default 1.0
uniform float contrast; // Default 1.0
uniform float gamma; // Default 1.0
uniform float global_chroma; // Default 1.0
uniform float global_vibrance; // Default 0.0
uniform float local_contrast; // Default 0.0 (0.0 to 1.0+)
uniform float temperature; // Default 0.0
uniform float tint; // Default 0.0
uniform float highlight_compression; // Default 0.0 (0.0 to 1.0+)
uniform float black_level; // Default 0.0
uniform float shadows_boost; // Default 0.0
uniform float highlights_boost; // Default 0.0
uniform float white_point; // Default 0.0
uniform float vignette_amount; // Default 0.0
uniform float sharpen_amount; // Default 0.0
uniform float sharpen_threshold; // Default 0.0
uniform float denoise_luma; // Default 0.0
uniform float denoise_chroma; // Default 0.0

uniform sampler2D color_equalizer_lut; // Floating point texture (Hue shift, Saturation scale, Lightness offset)
uniform sampler2D tone_equalizer_lut; // 1D floating point texture (EV adjustment)
uniform sampler2D shadow_color_equalizer_lut; // Same format as color_equalizer

uniform float base_curve_strength; // 0.0 to 1.0+
uniform float ai_blend; // 0.0 to 1.0, opacity of all adjustments

const float PI = 3.14159265359;

// Linear RGB to LMS matrix (Column-major)
const mat3 m1 = mat3(
    0.4122214708, 0.2119034982, 0.0883024619,
    0.5363325363, 0.6806995451, 0.2817188376,
    0.0514459929, 0.1073969566, 0.6299787005
);

// LMS to OKLab matrix (Column-major)
const mat3 m2 = mat3(
    0.2104542553, 1.9779984951, 0.0259040371,
    0.7936177850, -2.4285922050, 0.7827717662,
    -0.0040720468, 0.4505937099, -0.8086757660
);

// OKLab to LMS matrix (Column-major)
const mat3 m2_inv = mat3(
    1.0, 1.0, 1.0,
    0.3963377774, -0.1055613458, -0.0894841775,
    0.2158037573, -0.0638541728, -1.2914855480
);

// LMS to Linear RGB matrix (Column-major)
const mat3 m1_inv = mat3(
    4.0767416621, -1.2684380046, -0.0041960863,
    -3.3077115913, 2.6097574011, -0.7034186147,
    0.2309699292, -0.3413193965, 1.7076147010
);

vec3 linear_srgb_to_oklab(vec3 c) {
    vec3 lms = m1 * c;
    lms = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return m2 * lms;
}

vec3 oklab_to_linear_srgb(vec3 c) {
    vec3 lms = m2_inv * c;
    lms = lms * lms * lms;
    return m1_inv * lms;
}

vec3 oklab_to_oklch(vec3 lab) {
    float L = lab.x;
    float C = sqrt(lab.y * lab.y + lab.z * lab.z);
    float h = atan(lab.z, lab.y); // returns radians
    if (h < 0.0) h += 2.0 * PI;
    h = h * 180.0 / PI; // convert to degrees
    return vec3(L, C, h);
}

vec3 oklch_to_oklab(vec3 lch) {
    float L = lch.x;
    float C = lch.y;
    float h = lch.z * PI / 180.0; // convert to radians
    float a = C * cos(h);
    float b = C * sin(h);
    return vec3(L, a, b);
}

vec3 RGBtoYCbCr(vec3 rgb) {
    return vec3(
        0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b,
        -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b + 0.5,
        0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 0.5
    );
}

vec3 YCbCrtoRGB(vec3 ycbcr) {
    float y = ycbcr.x;
    float cb = ycbcr.y - 0.5;
    float cr = ycbcr.z - 0.5;
    return vec3(
        y + 1.402 * cr,
        y - 0.344136 * cb - 0.714136 * cr,
        y + 1.772 * cb
    );
}


vec3 preprocess_pixel(vec3 rgb) {
    rgb *= exposure;
    
    if (black_level > 0.0) {
        rgb = max(vec3(0.0), (rgb - black_level) / max(0.001, 1.0 - black_level));
    }
    
    rgb = max(rgb, vec3(0.0));
    rgb = pow(rgb, vec3(1.0 / gamma));
    
    rgb.r *= max(0.0, 1.0 + temperature * 0.5);
    rgb.b *= max(0.0, 1.0 - temperature * 0.5);
    
    // Tint: positive = magenta (reduce green), negative = green (boost green)
    rgb.g *= max(0.0, 1.0 - tint * 0.5);
    
    return rgb;
}

void main() {
    // Clip fragments outside the crop region
    if (abs(crop_ndc.x) > 1.0 || abs(crop_ndc.y) > 1.0) {
        discard;
    }

    // 1. Read linear RGB from texture with slight negative LOD bias to prevent blur without causing moire
    vec4 tex_color = texture(image_texture, uv, -0.2);
    vec3 linear_rgb = tex_color.rgb;
    
    vec2 tex_size = vec2(textureSize(image_texture, 0));
    vec2 texel_size = 1.0 / tex_size;
    
    // --- Denoise (Bilateral Filter in YCbCr) ---
    if (denoise_luma > 0.0 || denoise_chroma > 0.0) {
        vec3 center_ycbcr = RGBtoYCbCr(linear_rgb);
        vec3 num = vec3(0.0);
        vec3 denom = vec3(0.0);
        
        // Map slider (0..1) to sigmas. 
        // Luma sigma_r needs to be small to preserve edges.
        // Chroma sigma_r can be larger.
        float sigmaS = 2.0;
        float twoSigmaS2 = 2.0 * sigmaS * sigmaS;
        
        // Fixed reasonable sigmaR for noise.
        // The slider will control how much we blend this in, and how much texture we protect.
        float sigmaR_luma = 0.02; 
        float twoSigmaR2_luma = 2.0 * sigmaR_luma * sigmaR_luma;
        
        float sigmaR_chroma = 0.01 + denoise_chroma * 0.2;
        float twoSigmaR2_chroma = 2.0 * sigmaR_chroma * sigmaR_chroma;
        
        int r = 3; // 7x7 kernel for better noise averaging without needing huge sigma
        for (int i = -r; i <= r; i++) {
            for (int j = -r; j <= r; j++) {
                vec2 offset = vec2(float(i), float(j)) * texel_size;
                vec3 sample_rgb = texture(image_texture, uv + offset).rgb;
                vec3 sample_ycbcr = RGBtoYCbCr(sample_rgb);
                
                float spatialDist2 = float(i*i + j*j);
                float spatialWeight = exp(-spatialDist2 / twoSigmaS2);
                
                vec3 delta = sample_ycbcr - center_ycbcr;
                
                // Calculate weight for Luma (Y) and Chroma (CbCr) separately
                float rw_y = exp(-(delta.x * delta.x) / twoSigmaR2_luma);
                float rw_c = exp(-dot(delta.yz, delta.yz) / twoSigmaR2_chroma);
                
                // If denoise_luma == 0, force luma weight to 1.0 (average only spatially)
                // Wait, if denoise_luma == 0, we don't want to blur luma at all! 
                // So if denoise is 0, we just set the weight so high that only center pixel matters?
                // No, we can just blend the blurred result with original based on slider!
                
                vec3 rangeWeight = vec3(rw_y, rw_c, rw_c);
                vec3 totalWeight = spatialWeight * rangeWeight;
                
                num += sample_ycbcr * totalWeight;
                denom += totalWeight;
            }
        }
        vec3 filtered_ycbcr = num / max(denom, vec3(0.0001));
        
        // --- Detail Protection for Luma ---
        // Extracted detail (noise + texture)
        float detail = center_ycbcr.x - filtered_ycbcr.x;
        
        // Scale down the slider influence dramatically (10x weaker)
        float scaled_luma = denoise_luma * 0.1;
        
        // The threshold is now much smaller, so it protects much more texture
        float noise_threshold = mix(0.0005, 0.005, scaled_luma);
        
        // If detail is larger than threshold, it's texture/edge, so we keep it.
        float keep_detail = smoothstep(noise_threshold, noise_threshold * 2.0, abs(detail));
        
        // Add texture back to the filtered result
        float final_luma = filtered_ycbcr.x + detail * keep_detail;
        
        // Blend based on scaled slider to prevent a 100% blur
        filtered_ycbcr.x = mix(center_ycbcr.x, final_luma, denoise_luma > 0.0 ? 1.0 : 0.0);
        filtered_ycbcr.yz = mix(center_ycbcr.yz, filtered_ycbcr.yz, smoothstep(0.0, 0.1, denoise_chroma));
        
        linear_rgb = YCbCrtoRGB(filtered_ycbcr);
    }
    
    linear_rgb = preprocess_pixel(linear_rgb);

    // --- Local Contrast & Tone Equalizer (Bilateral Filter Approximation) ---
    vec3 center_rgb = linear_rgb;
    float center_luma = dot(center_rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 sum_rgb = vec3(0.0);
    float sum_weight = 0.0;
    
    // Scale the sample radius based on image size to make the effect visible on high-res images
    float radius_scale = max(1.0, tex_size.x / 1000.0 * 4.0);
    float lod = max(0.0, log2(radius_scale));
    
    // 5x5 blur
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 offset = vec2(float(i), float(j)) * texel_size * radius_scale;
            vec4 samp = textureLod(image_texture, uv + offset, lod);
            vec3 sample_rgb = preprocess_pixel(samp.rgb);
            
            float sample_luma = dot(sample_rgb, vec3(0.2126, 0.7152, 0.0722));
            
            float spatial = exp(-(float(i*i + j*j)) / (2.0 * 2.0)); // sigma_s = 2.0
            float range = exp(-pow(sample_luma - center_luma, 2.0) / (2.0 * 0.01)); // sigma_r = 0.1
            float w = spatial * range;
            
            sum_rgb += sample_rgb * w;
            sum_weight += w;
        }
    }
    vec3 base_layer = sum_rgb / max(sum_weight, 0.0001);
    vec3 detail_layer = center_rgb - base_layer;
    
    linear_rgb = base_layer + detail_layer * (1.0 + local_contrast);
    
    // --- Tone Equalizer ---
    // Calculate EV of the base layer to avoid halos
    float base_luma = dot(base_layer, vec3(0.2126, 0.7152, 0.0722));
    float ev = log2(max(base_luma, 0.000001));
    
    // Map EV from [-8.0, +2.0] to UV [0.0, 1.0] (10 EV dynamic range)
    float tone_uv = clamp((ev + 8.0) / 10.0, 0.5/1024.0, 1.0 - 0.5/1024.0);
    float ev_adjust = texture(tone_equalizer_lut, vec2(tone_uv, 0.5)).r;
    
    // Progressive highlight recovery extrapolation for extreme blowouts (EV > +2.0)
    if (ev > 2.0 && ev_adjust < 0.0) {
        ev_adjust += (ev - 2.0) * (ev_adjust / 2.0);
    }
    
    // Apply exposure compensation from tone equalizer
    linear_rgb *= exp2(ev_adjust);
    
    // --- Highlight Recovery / Highlight Compression ---
    if (highlight_compression > 0.0) {
        float luma_hl = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
        float max_c = max(linear_rgb.r, max(linear_rgb.g, linear_rgb.b));
        float peak = max(luma_hl, max_c);
        float hl_threshold = 0.35; // Start progressive shoulder compression above midtones
        if (peak > hl_threshold) {
            float over = peak - hl_threshold;
            float compression_strength = highlight_compression * 3.0;
            float comp_peak = hl_threshold + over / (1.0 + compression_strength * (over / 0.35));
            float scale = comp_peak / max(peak, 0.0001);
            linear_rgb *= scale;
            
            // Smooth desaturation on extreme overexposed channels to prevent hue shifting in blowouts
            float extreme_over = max(0.0, peak - 1.0);
            float desat = clamp(extreme_over * highlight_compression * 0.4, 0.0, 0.75);
            float compressed_luma = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
            linear_rgb = mix(linear_rgb, vec3(compressed_luma), desat);
        }
    }
    
    // Whites point
    if (white_point != 0.0) {
        linear_rgb *= exp2(white_point);
    }
    
    // Shadows / Highlights boost
    if (shadows_boost != 0.0 || highlights_boost != 0.0) {
        float luma_sh = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
        float perc_luma = pow(max(luma_sh, 0.0), 1.0 / 2.2);
        float sm = 1.0 - smoothstep(0.0, 0.35, perc_luma);
        float hm = smoothstep(0.5, 1.0, perc_luma);
        
        if (shadows_boost != 0.0) linear_rgb *= exp2(shadows_boost * sm);
        if (highlights_boost != 0.0) linear_rgb *= exp2(highlights_boost * hm);
    }
    
    // PREVENT NEGATIVE RGB BEFORE OKLAB!
    // Negative values from local contrast or black level cause impossible OKLab colors (L=0 but C>0)
    // which turn into bright blue/magenta artifacts when converted back to RGB.
    linear_rgb = max(linear_rgb, vec3(0.0));

    // 2. Convert to OKLab and then OKLCH
    vec3 oklab = linear_srgb_to_oklab(linear_rgb);
    vec3 oklch = oklab_to_oklch(oklab);

    // 3. Global adjustments in OKLCH
    // Global Chroma
    oklch.y *= max(global_chroma, 0.0); 
    
    // Global Vibrance
    if (global_vibrance != 0.0) {
        float c_in = max(oklch.y, 0.000001);
        // Non-linear boost: less saturated colors get boosted more
        // global_vibrance range [-1.0, 1.0] or similar
        oklch.y += global_vibrance * c_in * (1.0 - clamp(c_in, 0.0, 1.0));
    }
    
    // Contrast (apply to L, assuming mid-point is around 0.5)
    oklch.x = (oklch.x - 0.5) * contrast + 0.5;

    // 4. Color Equalizer
    // Sample LUT based on Hue (oklch.z is in degrees [0, 360))
    float lut_u = clamp(oklch.z / 360.0, 0.0, 1.0);
    vec4 eq_mods = texture(color_equalizer_lut, vec2(lut_u, 0.5));
    // eq_mods: r = hue_shift (degrees), g = saturation_scale, b = lightness_offset
    
    // Protect achromatic colors from getting sudden hue/saturation shifts
    float protection_mask = smoothstep(0.0, 0.05, oklch.y);
    
    oklch.z += eq_mods.r * protection_mask;
    oklch.y *= mix(1.0, max(eq_mods.g, 0.0), protection_mask);
    oklch.x += eq_mods.b * protection_mask;

    // Add Shadow Color Equalizer (masked by luminance)
    float luma_shadow = oklab.x;
    // Mask: 1.0 at luma 0.1, 0.0 at luma 0.7 (shadows and dark midtones)
    float shadow_mask = 1.0 - smoothstep(0.1, 0.7, luma_shadow);
    if (shadow_mask > 0.0) {
        vec4 sceq = texture(shadow_color_equalizer_lut, vec2(lut_u, 0.5));
        oklch.z += sceq.r * protection_mask * shadow_mask;
        oklch.y *= mix(1.0, max(sceq.g, 0.0), protection_mask * shadow_mask);
        oklch.x += sceq.b * protection_mask * shadow_mask;
    }

    // Keep hue in [0, 360)
    oklch.z = mod(oklch.z, 360.0);
    if (oklch.z < 0.0) oklch.z += 360.0;

    // Clamp L and C to avoid extreme blowouts
    oklch.x = clamp(oklch.x, 0.0, 1.0);
    
    // Taper Chroma to 0 at extreme black and white to prevent impossible colors
    float c_damp = smoothstep(0.0, 0.02, oklch.x) * (1.0 - smoothstep(0.98, 1.0, oklch.x));
    oklch.y = max(oklch.y * c_damp, 0.0);

    // 5. Convert back to OKLab then Linear RGB
    vec3 modified_oklab = oklch_to_oklab(oklch);
    vec3 modified_rgb = oklab_to_linear_srgb(modified_oklab);
    
    // Final safeguard against negative values
    modified_rgb = max(modified_rgb, vec3(0.0));
    
    // 5.5 Blend with original image based on ai_blend slider
    // tex_color.rgb is the original linear image before any exposure/contrast adjustments
    modified_rgb = mix(tex_color.rgb, modified_rgb, ai_blend);
    
    // 6. Apply Base Curve (Camera Emulation) if enabled
    if (base_curve_strength > 0.0) {
        // Raw linear data is inherently very dark. Exposure has already been applied via the slider.
        vec3 rgb_linear = modified_rgb;
        
        // Convert to perceptual gamma space
        vec3 rgb_g = pow(rgb_linear, vec3(1.0 / 2.4));
        
        // Lift shadows slightly (like Auto Lighting Optimizer or Darktable's default)
        rgb_g = pow(rgb_g, vec3(0.90));
        
        // Boost saturation slightly to mimic camera processing (which usually adds a bit of vibrance)
        float luma_g = dot(rgb_g, vec3(0.2126, 0.7152, 0.0722));
        rgb_g = mix(vec3(luma_g), rgb_g, 1.15);
        
        rgb_g = clamp(rgb_g, 0.0, 1.0);
        
        // Classic Smoothstep S-curve for punchy contrast
        rgb_g = rgb_g * rgb_g * (3.0 - 2.0 * rgb_g);
        
        // Convert back to linear space
        vec3 curved_rgb = pow(rgb_g, vec3(2.4));
        
        // Blend with uncurved rgb based on strength
        modified_rgb = mix(modified_rgb, curved_rgb, base_curve_strength);
    }
    
    // 7. Sharpening (High-Detail Unsharp Mask)
    if (sharpen_amount > 0.0) {
        float s_radius = max(1.0, tex_size.x / 1800.0);
        vec2 s_off = texel_size * s_radius;

        vec3 c_raw = preprocess_pixel(tex_color.rgb);
        vec3 l_raw = preprocess_pixel(texture(image_texture, uv + vec2(-s_off.x, 0.0)).rgb);
        vec3 r_raw = preprocess_pixel(texture(image_texture, uv + vec2(s_off.x, 0.0)).rgb);
        vec3 t_raw = preprocess_pixel(texture(image_texture, uv + vec2(0.0, -s_off.y)).rgb);
        vec3 b_raw = preprocess_pixel(texture(image_texture, uv + vec2(0.0, s_off.y)).rgb);

        vec3 tl_raw = preprocess_pixel(texture(image_texture, uv + vec2(-s_off.x, -s_off.y)).rgb);
        vec3 tr_raw = preprocess_pixel(texture(image_texture, uv + vec2(s_off.x, -s_off.y)).rgb);
        vec3 bl_raw = preprocess_pixel(texture(image_texture, uv + vec2(-s_off.x, s_off.y)).rgb);
        vec3 br_raw = preprocess_pixel(texture(image_texture, uv + vec2(s_off.x, s_off.y)).rgb);

        vec3 blur_raw = (l_raw + r_raw + t_raw + b_raw) * 0.125 + (tl_raw + tr_raw + bl_raw + br_raw) * 0.0625 + c_raw * 0.25;
        vec3 hp_raw = c_raw - blur_raw;

        float hp_luma = dot(hp_raw, vec3(0.2126, 0.7152, 0.0722));
        float factor = 1.0;
        if (sharpen_threshold > 0.0) {
            factor = smoothstep(sharpen_threshold * 0.01, sharpen_threshold * 0.01 + 0.01, abs(hp_luma));
        }

        vec3 detail = hp_raw * (sharpen_amount * 3.0 * factor);
        modified_rgb += detail;
        modified_rgb = max(modified_rgb, vec3(0.0));
    }
    
    // 8. Vignette
    if (vignette_amount != 0.0) {
        float dist = length(crop_ndc);
        float v = smoothstep(0.5, 1.5, dist);
        // vignette_amount: positive = dark vignette, negative = white vignette
        modified_rgb *= (1.0 - v * vignette_amount);
    }

    // Scale down by 50.0 to safely accumulate blown out pixels in 16-bit FBO
    fragColor = vec4(modified_rgb * 0.02, tex_color.a);
}
