from typing import Dict, Any
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from ..base import BaseEffectPlugin
try:
    from gui.spline_widget import ToneEqualizerWidget, calculate_tone_equalizer_lut
except ImportError:
    from ...gui.spline_widget import ToneEqualizerWidget, calculate_tone_equalizer_lut

class ToneEqualizerPlugin(BaseEffectPlugin):
    @property
    def plugin_id(self) -> str:
        return "tone_equalizer"

    @property
    def display_name(self) -> str:
        return "Tone Equalizer"

    @property
    def category(self) -> str:
        return "Tone & Exposure"

    @property
    def order(self) -> int:
        return 200

    @property
    def icon(self) -> str:
        return "📊"

    def get_default_settings(self) -> Dict[str, Any]:
        return {
            'nodes': [50] * 9  # 9 nodes across -8 EV to +2 EV at 50%
        }

    def get_glsl_uniforms(self) -> str:
        return """
uniform sampler2D tone_equalizer_lut; // 1D floating point texture (EV adjustment)
"""

    def get_glsl_execution(self) -> str:
        return """
// --- Tone Equalizer (Multiband EV adjustments) ---
float base_luma = dot(linear_rgb, vec3(0.2126, 0.7152, 0.0722));
float ev = log2(max(base_luma, 0.000001));

// Map EV [-8.0, 0.0] to UV [0.0, 1.0] (8 EV dynamic range)
float tone_uv = clamp((ev + 8.0) / 8.0, 0.5 / 1024.0, 1.0 - 0.5 / 1024.0);
float ev_adjust = texture(tone_equalizer_lut, vec2(tone_uv, 0.5)).r;

// Progressive highlight recovery extrapolation for extreme blowouts (EV > 0.0)
if (ev > 0.0 && ev_adjust < 0.0) {
    ev_adjust += ev * (ev_adjust * 0.5);
}

linear_rgb *= exp2(ev_adjust);
linear_rgb = max(linear_rgb, vec3(0.0));
"""

    def apply_uniforms(self, program: Any, layer_settings: Dict[str, Any], engine: Any) -> None:
        if 'tone_equalizer_lut' in program:
            program['tone_equalizer_lut'].value = 2
        if hasattr(engine, 'update_tone_equalizer_lut'):
            nodes = None
            if isinstance(layer_settings, dict):
                if 'nodes' in layer_settings:
                    nodes = layer_settings['nodes']
                elif 'tone_equalizer' in layer_settings:
                    te = layer_settings['tone_equalizer']
                    if isinstance(te, dict):
                        nodes = te.get('nodes')
                    elif isinstance(te, (list, tuple)):
                        nodes = te
                elif 'tone_equalizer_nodes' in layer_settings:
                    nodes = layer_settings['tone_equalizer_nodes']
            elif isinstance(layer_settings, (list, tuple)):
                nodes = layer_settings

            if nodes is not None and len(nodes) == 9:
                lut = calculate_tone_equalizer_lut(nodes, 1024)
                engine.update_tone_equalizer_lut(lut)
            elif hasattr(self, '_eq_widget') and self._eq_widget:
                lut = self.get_lut_array()
                engine.update_tone_equalizer_lut(lut)
            elif isinstance(layer_settings.get('tone_equalizer'), np.ndarray):
                engine.update_tone_equalizer_lut(layer_settings['tone_equalizer'])

    def create_widget(self, parent: QWidget, on_changed_callback) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._eq_widget = ToneEqualizerWidget(container)
        self._eq_widget.lut_changed.connect(lambda lut: on_changed_callback())
        layout.addWidget(self._eq_widget)
        return container

    def get_state(self, widget: QWidget) -> Dict[str, Any]:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            return {'nodes': self._eq_widget.get_nodes_as_ints()}
        return self.get_default_settings()

    def set_state(self, widget: QWidget, state: Dict[str, Any]) -> None:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            nodes = None
            if isinstance(state, dict):
                if 'nodes' in state:
                    nodes = state['nodes']
                elif 'tone_equalizer' in state:
                    te = state['tone_equalizer']
                    if isinstance(te, dict):
                        nodes = te.get('nodes')
                    elif isinstance(te, (list, tuple)):
                        nodes = te
                elif 'tone_equalizer_nodes' in state:
                    nodes = state['tone_equalizer_nodes']
            elif isinstance(state, (list, tuple)):
                nodes = state
            if nodes is None or len(nodes) != 9:
                nodes = [50] * 9
            self._eq_widget.set_nodes_from_ints(nodes)

    def reset_state(self, widget: QWidget) -> None:
        self.set_state(widget, self.get_default_settings())

    def get_lut_array(self) -> np.ndarray:
        if hasattr(self, '_eq_widget') and self._eq_widget:
            return self._eq_widget.get_lut_array(1024)
        return np.zeros(1024, dtype='f4')
