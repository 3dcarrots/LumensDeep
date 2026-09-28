from typing import Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QGridLayout
)
from PySide6.QtCore import Qt
from ..base import BaseEffectPlugin

class BasicAdjustmentsPlugin(BaseEffectPlugin):
    @property
    def plugin_id(self) -> str:
        return "basic_adjustments"

    @property
    def display_name(self) -> str:
        return "Basic Adjustments"

    @property
    def category(self) -> str:
        return "Tone & Exposure"

    @property
    def order(self) -> int:
        return 100

    @property
    def icon(self) -> str:
        return "☀️"

    def get_default_settings(self) -> Dict[str, Any]:
        return {
            'exposure': 0, # Slider EV in tenths (-30 to +30 -> -3.0 to +3.0)
            'contrast': 100, # 1.0
            'gamma': 100, # 1.0
            'base_curve': 30, # 30% Natural filmic S-Curve baseline
            'temperature': 0,
            'tint': 0,
            'highlights': 0,
            'shadows': 0,
            'whites': 0,
            'black_level': 0,
            'highlight_recovery': 0,
            'local_contrast': 0,
            'global_vibrance': 0,
            'global_chroma': 100,
            'denoise_luma': 0,
            'denoise_chroma': 0,
            'sharpen': 0,
            'sharpen_threshold': 0,
            'vignette': 0,
        }

    def get_glsl_uniforms(self) -> str:
        return """
uniform float exposure;
uniform float contrast;
uniform float gamma;
uniform float global_chroma;
uniform float global_vibrance;
uniform float local_contrast;
uniform float temperature;
uniform float tint;
uniform float highlight_compression;
uniform float black_level;
uniform float shadows_boost;
uniform float highlights_boost;
uniform float white_point;
uniform float vignette_amount;
uniform float sharpen_amount;
uniform float sharpen_threshold;
uniform float denoise_luma;
uniform float denoise_chroma;
"""

    def get_glsl_functions(self) -> str:
        return """
vec3 preprocess_pixel(vec3 rgb) {
    rgb *= exposure;
    if (black_level > 0.0) {
        rgb = max(vec3(0.0), (rgb - black_level) / max(0.001, 1.0 - black_level));
    }
    rgb = max(rgb, vec3(0.0));
    rgb = pow(rgb, vec3(1.0 / max(0.01, gamma)));
    rgb.r *= max(0.0, 1.0 + temperature * 0.5);
    rgb.b *= max(0.0, 1.0 - temperature * 0.5);
    rgb.g *= max(0.0, 1.0 - tint * 0.5);
    return rgb;
}
"""

    def get_glsl_execution(self) -> str:
        return """
// --- 1. Denoise (Bilateral Filter in YCbCr) ---
if (denoise_luma > 0.0 || denoise_chroma > 0.0) {
    vec3 center_ycbcr = RGBtoYCbCr(linear_rgb);
    vec3 num = vec3(0.0);
    vec3 denom = vec3(0.0);
    float sigmaS = 2.0;
    float twoSigmaS2 = 2.0 * sigmaS * sigmaS;
    float sigmaR_luma = 0.02; 
    float twoSigmaR2_luma = 2.0 * sigmaR_luma * sigmaR_luma;
    float sigmaR_chroma = 0.01 + denoise_chroma * 0.2;
    float twoSigmaR2_chroma = 2.0 * sigmaR_chroma * sigmaR_chroma;
    
    int r = 3;
    for (int i = -r; i <= r; i++) {
        for (int j = -r; j <= r; j++) {
            vec2 offset = vec2(float(i), float(j)) * texel_size;
            vec3 sample_rgb = texture(image_texture, uv + offset).rgb;
            vec3 sample_ycbcr = RGBtoYCbCr(sample_rgb);
            float spatialDist2 = float(i*i + j*j);
            float spatialWeight = exp(-spatialDist2 / twoSigmaS2);
            vec3 delta = sample_ycbcr - center_ycbcr;
            float rw_y = exp(-(delta.x * delta.x) / twoSigmaR2_luma);
            float rw_c = exp(-dot(delta.yz, delta.yz) / twoSigmaR2_chroma);
            vec3 rangeWeight = vec3(rw_y, rw_c, rw_c);
            vec3 totalWeight = spatialWeight * rangeWeight;
            num += sample_ycbcr * totalWeight;
            denom += totalWeight;
        }
    }
    vec3 filtered_ycbcr = num / max(denom, vec3(0.0001));
    float detail = center_ycbcr.x - filtered_ycbcr.x;
    float scaled_luma = denoise_luma * 0.1;
    float noise_threshold = mix(0.0005, 0.005, scaled_luma);
    float keep_detail = smoothstep(noise_threshold, noise_threshold * 2.0, abs(detail));
    float final_luma = filtered_ycbcr.x + detail * keep_detail;
    filtered_ycbcr.x = mix(center_ycbcr.x, final_luma, denoise_luma > 0.0 ? 1.0 : 0.0);
    filtered_ycbcr.yz = mix(center_ycbcr.yz, filtered_ycbcr.yz, smoothstep(0.0, 0.1, denoise_chroma));
    linear_rgb = YCbCrtoRGB(filtered_ycbcr);
}

linear_rgb = preprocess_pixel(linear_rgb);

// --- 2. Local Contrast ---
if (local_contrast != 0.0) {
    vec3 center_rgb = linear_rgb;
    float center_luma = dot(center_rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 sum_rgb = vec3(0.0);
    float sum_weight = 0.0;
    float radius_scale = max(1.0, tex_size.x / 1000.0 * 4.0);
    float lod = max(0.0, log2(radius_scale));
    
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            vec2 offset = vec2(float(i), float(j)) * texel_size * radius_scale;
            vec4 samp = textureLod(image_texture, uv + offset, lod);
            vec3 sample_rgb = preprocess_pixel(samp.rgb);
            float sample_luma = dot(sample_rgb, vec3(0.2126, 0.7152, 0.0722));
            float spatial = exp(-(float(i*i + j*j)) / (2.0 * 2.0));
            float range = exp(-pow(sample_luma - center_luma, 2.0) / (2.0 * 0.01));
            float w = spatial * range;
            sum_rgb += sample_rgb * w;
            sum_weight += w;
        }
    }
    vec3 base_layer = sum_rgb / max(sum_weight, 0.0001);
    vec3 detail_layer = center_rgb - base_layer;
    linear_rgb = base_layer + detail_layer * (1.0 + local_contrast);
}

// --- 3. Highlight Recovery, Whites, Shadows/Highlights Boost ---
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

if (white_point != 0.0) {
    linear_rgb *= exp2(white_point);
}
if (shadows_boost != 0.0 || highlights_boost != 0.0) {
    float luma_sh = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
    float perc_luma = pow(max(luma_sh, 0.0), 1.0 / 2.2);
    float sm = 1.0 - smoothstep(0.0, 0.35, perc_luma);
    float hm = smoothstep(0.5, 1.0, perc_luma);
    if (shadows_boost != 0.0) linear_rgb *= exp2(shadows_boost * sm);
    if (highlights_boost != 0.0) linear_rgb *= exp2(highlights_boost * hm);
}

linear_rgb = max(linear_rgb, vec3(0.0));

// --- 4. Chroma, Vibrance & Contrast in OKLCH ---
vec3 oklab = linear_srgb_to_oklab(linear_rgb);
vec3 oklch = oklab_to_oklch(oklab);

oklch.y *= max(global_chroma, 0.0);
if (global_vibrance != 0.0) {
    float c_in = max(oklch.y, 0.000001);
    oklch.y += global_vibrance * c_in * (1.0 - clamp(c_in, 0.0, 1.0));
}
oklch.x = (oklch.x - 0.5) * contrast + 0.5;
oklch.x = clamp(oklch.x, 0.0, 1.0);
float c_damp = smoothstep(0.0, 0.02, oklch.x) * (1.0 - smoothstep(0.98, 1.0, oklch.x));
oklch.y = max(oklch.y * c_damp, 0.0);

linear_rgb = max(oklab_to_linear_srgb(oklch_to_oklab(oklch)), vec3(0.0));

// --- 5. Sharpening (High-Detail Unsharp Mask) ---
if (sharpen_amount > 0.0) {
    float s_radius = max(1.0, tex_size.x / 1800.0);
    vec2 s_off = texel_size * s_radius;

    vec3 c_raw = preprocess_pixel(texture(image_texture, uv).rgb);
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
    linear_rgb += detail;
    linear_rgb = max(linear_rgb, vec3(0.0));
}

// --- 6. Vignette ---
if (vignette_amount != 0.0) {
    float dist = length(crop_ndc);
    float v = smoothstep(0.5, 1.5, dist);
    linear_rgb *= (1.0 - v * vignette_amount);
}
"""

    def apply_uniforms(self, program: Any, layer_settings: Dict[str, Any], engine: Any) -> None:
        defaults = self.get_default_settings()
        s = dict(defaults)
        if isinstance(layer_settings, dict):
            if 'basic_adjustments' in layer_settings and isinstance(layer_settings['basic_adjustments'], dict):
                s.update(layer_settings['basic_adjustments'])
            for k, v in layer_settings.items():
                if k != 'basic_adjustments' and k in defaults:
                    s[k] = v

        if 'exposure' in program:
            program['exposure'].value = float(2.22 * (2.0 ** (s['exposure'] / 10.0)))
        if 'contrast' in program:
            program['contrast'].value = float(s['contrast'] / 100.0)
        if 'gamma' in program:
            program['gamma'].value = float(s['gamma'] / 100.0)
        if 'base_curve_strength' in program:
            program['base_curve_strength'].value = float(s.get('base_curve', 30) / 100.0)
        if 'temperature' in program:
            program['temperature'].value = float(s['temperature'] / 100.0)
        if 'tint' in program:
            program['tint'].value = float(s['tint'] / 100.0)
        if 'highlights_boost' in program:
            program['highlights_boost'].value = float(s['highlights'] / 100.0)
        if 'shadows_boost' in program:
            program['shadows_boost'].value = float(s['shadows'] / 100.0)
        if 'white_point' in program:
            program['white_point'].value = float(s['whites'] / 100.0)
        if 'black_level' in program:
            program['black_level'].value = float(s['black_level'] / 1000.0)
        if 'highlight_compression' in program:
            program['highlight_compression'].value = float(s['highlight_recovery'] / 100.0)
        if 'local_contrast' in program:
            program['local_contrast'].value = float(s['local_contrast'] / 100.0)
        if 'global_vibrance' in program:
            program['global_vibrance'].value = float(s['global_vibrance'] / 100.0)
        if 'global_chroma' in program:
            program['global_chroma'].value = float(s['global_chroma'] / 100.0)
        if 'denoise_luma' in program:
            program['denoise_luma'].value = float(s['denoise_luma'] / 100.0)
        if 'denoise_chroma' in program:
            program['denoise_chroma'].value = float(s['denoise_chroma'] / 100.0)
        if 'sharpen_amount' in program:
            program['sharpen_amount'].value = float(s['sharpen'] / 100.0)
        if 'sharpen_threshold' in program:
            program['sharpen_threshold'].value = float(s['sharpen_threshold'] / 100.0)
        if 'vignette_amount' in program:
            program['vignette_amount'].value = float(s['vignette'] / 100.0)

    def create_widget(self, parent: QWidget, on_changed_callback: Any) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        self._sliders: Dict[str, QSlider] = {}
        self._val_labels: Dict[str, QLabel] = {}

        sliders_def = [
            ("Highlight Recovery", "highlight_recovery", 0, 100, 0, "%"),
            ("Whites", "whites", -200, 200, 0, ""),
            ("Black Level", "black_level", 0, 100, 0, ""),
            ("Shadows", "shadows", -200, 200, 0, ""),
            ("Highlights", "highlights", -200, 200, 0, ""),
            ("Exposure", "exposure", -30, 30, 0, " EV"),
            ("Contrast", "contrast", 0, 200, 100, "%"),
            ("Gamma", "gamma", 20, 300, 100, ""),
            ("Base Curve (S-Curve)", "base_curve", 0, 100, 30, "%"),
            ("Temperature", "temperature", -200, 200, 0, ""),
            ("Tint", "tint", -100, 100, 0, ""),
            ("Global Chroma", "global_chroma", 0, 200, 100, "%"),
            ("Global Vibrance", "global_vibrance", -100, 100, 0, "%"),
            ("Local Contrast", "local_contrast", 0, 200, 0, "%"),
            ("Vignette", "vignette", -100, 100, 0, "%"),
            ("Sharpen", "sharpen", 0, 100, 0, "%"),
            ("Sharpen Threshold", "sharpen_threshold", 0, 100, 0, ""),
            ("Denoise Luma", "denoise_luma", 0, 100, 0, "%"),
            ("Denoise Chroma", "denoise_chroma", 0, 100, 0, "%"),
        ]

        grid = QGridLayout()
        grid.setSpacing(4)

        for row, (label_text, key, min_v, max_v, def_v, unit) in enumerate(sliders_def):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("color: #ccc; font-size: 11px;")
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(def_v)
            
            val_lbl = QLabel(self._format_val(key, def_v, unit))
            val_lbl.setFixedWidth(50)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("color: #00d2ff; font-size: 11px; font-weight: bold;")
            
            def make_handler(k, u, sl, vl):
                def handler(v):
                    vl.setText(self._format_val(k, v, u))
                    on_changed_callback()
                return handler

            slider.valueChanged.connect(make_handler(key, unit, slider, val_lbl))
            
            grid.addWidget(lbl, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(val_lbl, row, 2)
            
            self._sliders[key] = slider
            self._val_labels[key] = val_lbl

        layout.addLayout(grid)
        return container

    def _format_val(self, key: str, val: int, unit: str) -> str:
        if key == 'exposure':
            return f"{val / 10.0:+.1f}{unit}"
        elif key in ['contrast', 'gamma', 'global_chroma']:
            return f"{val / 100.0:.2f}"
        else:
            return f"{val}{unit}"

    def get_state(self, widget: QWidget) -> Dict[str, Any]:
        return {key: slider.value() for key, slider in self._sliders.items()}

    def set_state(self, widget: QWidget, state: Dict[str, Any]) -> None:
        defaults = self.get_default_settings()
        s = dict(defaults)
        if isinstance(state, dict):
            if 'basic_adjustments' in state and isinstance(state['basic_adjustments'], dict):
                s.update(state['basic_adjustments'])
            for k, v in state.items():
                if k != 'basic_adjustments' and k in defaults:
                    s[k] = v

        for key, slider in self._sliders.items():
            val = s.get(key, defaults.get(key, 0))
            slider.blockSignals(True)
            slider.setValue(int(val))
            slider.blockSignals(False)
            if key in self._val_labels:
                unit = " EV" if key == 'exposure' else ("%" if key in ['contrast', 'global_chroma', 'global_vibrance', 'local_contrast', 'vignette', 'sharpen', 'denoise_luma', 'denoise_chroma', 'highlight_recovery'] else "")
                self._val_labels[key].setText(self._format_val(key, int(val), unit))

    def reset_state(self, widget: QWidget) -> None:
        self.set_state(widget, self.get_default_settings())
