import os
import glob
import importlib.util
from typing import Dict, List, Type, Optional
from PySide6.QtCore import QObject, Signal
from .base import BaseEffectPlugin

COMMON_GLSL_HEADER = """#version 330 core

in vec2 uv;
in vec2 crop_ndc;
out vec4 fragColor;

uniform sampler2D image_texture;
uniform float base_curve_strength;
uniform float ai_blend;

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
    float h = atan(lab.z, lab.y);
    if (h < 0.0) h += 2.0 * PI;
    h = h * 180.0 / PI;
    return vec3(L, C, h);
}

vec3 oklch_to_oklab(vec3 lch) {
    float L = lch.x;
    float C = lch.y;
    float h = lch.z * PI / 180.0;
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
"""

class PluginManager(QObject):
    """
    Central Manager for Layer Effect Plugins.
    Handles discovery, registration, active states, and dynamic GLSL fragment shader assembly.
    """
    plugins_changed = Signal()
    shader_recompiled = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._plugins: Dict[str, BaseEffectPlugin] = {}
        self._active_plugin_ids: set = set()
        self._register_builtins()

    def _register_builtins(self):
        """Registers the 4 core built-in effect plugins."""
        try:
            from .builtin.basic_adjustments import BasicAdjustmentsPlugin
            from .builtin.tone_equalizer import ToneEqualizerPlugin
            from .builtin.color_equalizer import ColorEqualizerPlugin
            from .builtin.shadow_color_eq import ShadowColorEqualizerPlugin
            
            self.register_plugin(BasicAdjustmentsPlugin())
            self.register_plugin(ToneEqualizerPlugin())
            self.register_plugin(ColorEqualizerPlugin())
            self.register_plugin(ShadowColorEqualizerPlugin())
        except Exception as e:
            print(f"Warning: Failed to load built-in plugins: {e}")

    def register_plugin(self, plugin: BaseEffectPlugin):
        """Registers a new effect plugin instance."""
        self._plugins[plugin.plugin_id] = plugin
        self._active_plugin_ids.add(plugin.plugin_id)
        self.plugins_changed.emit()

    def unregister_plugin(self, plugin_id: str):
        """Unregisters an effect plugin."""
        if plugin_id in self._plugins:
            del self._plugins[plugin_id]
            self._active_plugin_ids.discard(plugin_id)
            self.plugins_changed.emit()

    def get_plugin(self, plugin_id: str) -> Optional[BaseEffectPlugin]:
        return self._plugins.get(plugin_id)

    def get_all_plugins(self) -> List[BaseEffectPlugin]:
        """Returns all registered plugins sorted by execution order."""
        return sorted(self._plugins.values(), key=lambda p: p.order)

    def get_active_plugins(self) -> List[BaseEffectPlugin]:
        """Returns currently enabled plugins sorted by execution order."""
        return [p for p in self.get_all_plugins() if p.plugin_id in self._active_plugin_ids]

    def is_plugin_active(self, plugin_id: str) -> bool:
        return plugin_id in self._active_plugin_ids

    def set_plugin_active(self, plugin_id: str, active: bool):
        """Enables or disables an effect plugin."""
        if active:
            self._active_plugin_ids.add(plugin_id)
        else:
            self._active_plugin_ids.discard(plugin_id)
        self.plugins_changed.emit()

    def load_external_plugins(self, plugin_dir: str):
        """
        Discovers and dynamically imports community plugins (.py files) from a directory.
        """
        if not os.path.exists(plugin_dir):
            return

        for filepath in glob.glob(os.path.join(plugin_dir, "*.py")):
            basename = os.path.splitext(os.path.basename(filepath))[0]
            if basename.startswith("__"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"external_plugin_{basename}", filepath)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseEffectPlugin) and attr is not BaseEffectPlugin:
                            instance = attr()
                            self.register_plugin(instance)
                            print(f"Loaded external plugin: {instance.display_name} ({instance.plugin_id})")
            except Exception as e:
                print(f"Failed to load external plugin {filepath}: {e}")

    def build_fragment_shader(self) -> str:
        """
        Dynamically constructs the complete GLSL Fragment Shader source code
        by composing the active plugins in their defined pipeline order.
        """
        active_plugins = self.get_active_plugins()

        # 1. Header and common transforms
        code = [COMMON_GLSL_HEADER]

        # 2. Collect Uniform Declarations
        code.append("// --- Dynamic Plugin Uniforms ---")
        for p in active_plugins:
            uniforms = p.get_glsl_uniforms().strip()
            if uniforms:
                code.append(f"// [{p.display_name}] Uniforms\n{uniforms}\n")

        # 3. Collect Helper Functions
        code.append("// --- Dynamic Plugin Functions ---")
        for p in active_plugins:
            funcs = p.get_glsl_functions().strip()
            if funcs:
                code.append(f"// [{p.display_name}] Functions\n{funcs}\n")

        # 4. Construct main() pipeline
        code.append("""
void main() {
    // 1. Clip fragments outside crop region
    if (abs(crop_ndc.x) > 1.0 || abs(crop_ndc.y) > 1.0) {
        discard;
    }

    // 2. Read linear RGB from texture with slight negative LOD bias
    vec4 tex_color = texture(image_texture, uv, -0.2);
    vec3 linear_rgb = tex_color.rgb;
    
    vec2 tex_size = vec2(textureSize(image_texture, 0));
    vec2 texel_size = 1.0 / tex_size;
""")

        # 5. Insert execution blocks of all active plugins
        code.append("    // --- Dynamic Execution Pipeline ---")
        for p in active_plugins:
            execution = p.get_glsl_execution().strip()
            if execution:
                code.append(f"    // >>> Stage: {p.display_name} (Order {p.order})\n    {{\n{self._indent(execution, 8)}\n    }}\n")

        # 6. Final composite & Output
        code.append("""
    // Blend with original linear RGB based on ai_blend
    vec3 modified_rgb = mix(tex_color.rgb, linear_rgb, ai_blend);

    // Apply Camera Base Curve if enabled
    if (base_curve_strength > 0.0) {
        vec3 rgb_linear = modified_rgb;
        vec3 rgb_g = pow(rgb_linear, vec3(1.0 / 2.4));
        rgb_g = pow(rgb_g, vec3(0.90));
        float luma_g = dot(rgb_g, vec3(0.2126, 0.7152, 0.0722));
        rgb_g = mix(vec3(luma_g), rgb_g, 1.15);
        rgb_g = clamp(rgb_g, 0.0, 1.0);
        rgb_g = rgb_g * rgb_g * (3.0 - 2.0 * rgb_g);
        vec3 curved_rgb = pow(rgb_g, vec3(2.4));
        modified_rgb = mix(modified_rgb, curved_rgb, base_curve_strength);
    }

    // Scale down by 50.0 to safely accumulate in 16-bit FBO
    fragColor = vec4(modified_rgb * 0.02, tex_color.a);
}
""")
        return "\n".join(code)

    @staticmethod
    def _indent(text: str, spaces: int) -> str:
        pad = " " * spaces
        return "\n".join(pad + line if line.strip() else line for line in text.splitlines())
