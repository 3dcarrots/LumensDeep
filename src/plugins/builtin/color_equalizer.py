from typing import Dict, Any
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from ..base import BaseEffectPlugin
try:
    from gui.spline_widget import ColorEqualizerWidget, sanitize_equalizer_ints, sanitize_equalizer_dict, calculate_color_equalizer_lut
except ImportError:
    from ...gui.spline_widget import ColorEqualizerWidget, sanitize_equalizer_ints, sanitize_equalizer_dict, calculate_color_equalizer_lut

class ColorEqualizerPlugin(BaseEffectPlugin):
    @property
    def plugin_id(self) -> str:
        return "color_equalizer"

    @property
    def display_name(self) -> str:
        return "Color Equalizer"

    @property
    def category(self) -> str:
        return "Color"

    @property
    def order(self) -> int:
        return 300

    @property
    def icon(self) -> str:
        return "🎨"

    def get_default_settings(self) -> Dict[str, Any]:
        return {
            'hue': [50] * 7,
            'saturation': [50] * 7,
            'brightness': [50] * 7
        }

    def get_glsl_uniforms(self) -> str:
        return """
uniform sampler2D color_equalizer_lut; // Floating point texture (Hue shift, Saturation scale, Lightness offset)
"""

    def get_glsl_execution(self) -> str:
        return """
// --- Color Equalizer in OKLCH ---
vec3 oklab = linear_srgb_to_oklab(linear_rgb);
vec3 oklch = oklab_to_oklch(oklab);

// Sample LUT based on Hue (oklch.z is in degrees [0, 360))
float lut_u = clamp(oklch.z / 360.0, 0.0, 1.0);
vec4 eq_mods = texture(color_equalizer_lut, vec2(lut_u, 0.5));

// Protect achromatic colors from sudden hue/saturation shifts
float protection_mask = smoothstep(0.0, 0.05, oklch.y);

oklch.z += eq_mods.r * protection_mask;
oklch.y *= mix(1.0, max(eq_mods.g, 0.0), protection_mask);
oklch.x += eq_mods.b * protection_mask;

// Normalize hue [0, 360) and clamp
oklch.z = mod(oklch.z, 360.0);
if (oklch.z < 0.0) oklch.z += 360.0;
oklch.x = clamp(oklch.x, 0.0, 1.0);

float c_damp = smoothstep(0.0, 0.02, oklch.x) * (1.0 - smoothstep(0.98, 1.0, oklch.x));
oklch.y = max(oklch.y * c_damp, 0.0);

linear_rgb = max(oklab_to_linear_srgb(oklch_to_oklab(oklch)), vec3(0.0));
"""

    def apply_uniforms(self, program: Any, layer_settings: Dict[str, Any], engine: Any) -> None:
        if 'color_equalizer_lut' in program:
            program['color_equalizer_lut'].value = 1
        if hasattr(engine, 'update_color_equalizer_lut'):
            hue, sat, bri = None, None, None
            if isinstance(layer_settings, dict):
                s = layer_settings
                if 'color_equalizer' in s and isinstance(s['color_equalizer'], dict):
                    s = s['color_equalizer']
                hue = s.get('hue', s.get('hue_nodes'))
                sat = s.get('saturation', s.get('saturation_nodes', s.get('sat')))
                bri = s.get('brightness', s.get('brightness_nodes', s.get('bri', s.get('lightness'))))

            if hue is not None and sat is not None and bri is not None:
                h_clean = sanitize_equalizer_ints(hue)
                s_clean = sanitize_equalizer_ints(sat)
                b_clean = sanitize_equalizer_ints(bri)
                lut = calculate_color_equalizer_lut(h_clean, s_clean, b_clean, size=1024)
                engine.update_color_equalizer_lut(lut)
            elif hasattr(self, '_eq_widget') and self._eq_widget:
                lut = self.get_lut_array()
                engine.update_color_equalizer_lut(lut)
            elif isinstance(layer_settings.get('color_equalizer'), np.ndarray):
                engine.update_color_equalizer_lut(layer_settings['color_equalizer'])

    def create_widget(self, parent: QWidget, on_changed_callback) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._eq_widget = ColorEqualizerWidget(container)
        self._eq_widget.lut_changed.connect(lambda lut: on_changed_callback())
        layout.addWidget(self._eq_widget)
        return container

    def get_state(self, widget: QWidget) -> Dict[str, Any]:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            return {
                'hue': self._eq_widget.get_hue_nodes_as_ints(),
                'saturation': self._eq_widget.get_saturation_nodes_as_ints(),
                'brightness': self._eq_widget.get_brightness_nodes_as_ints()
            }
        return self.get_default_settings()

    def set_state(self, widget: QWidget, state: Dict[str, Any]) -> None:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            s = state if isinstance(state, dict) else {}
            if 'color_equalizer' in s and isinstance(s['color_equalizer'], dict):
                s = s['color_equalizer']
            h = s.get('hue', s.get('hue_nodes', [50] * 7))
            sat = s.get('saturation', s.get('saturation_nodes', s.get('sat', [50] * 7)))
            b = s.get('brightness', s.get('brightness_nodes', s.get('bri', s.get('lightness', [50] * 7))))
            self._eq_widget.set_nodes_from_ints(h, sat, b)

    def reset_state(self, widget: QWidget) -> None:
        self.set_state(widget, self.get_default_settings())

    def get_lut_array(self) -> np.ndarray:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            return self._eq_widget.get_lut_array()
        return np.zeros((1024, 4), dtype='f4')
