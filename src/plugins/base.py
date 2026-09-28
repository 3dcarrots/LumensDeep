from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PySide6.QtWidgets import QWidget

class BaseEffectPlugin(ABC):
    """
    Abstract base class for all Layer Effect Plugins in Lumens Deep.
    
    Each plugin encapsulates:
    1. Metadata (id, display name, category, execution order).
    2. UI Controls (QWidget displayed in the Layer Effects dock).
    3. GLSL Shader Code (Uniform declarations, helper functions, and pixel processing step).
    4. State Management (Serialization, deserialization, default values).
    """

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique machine identifier for the plugin (e.g., 'basic_adjustments')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name displayed in the UI (e.g., 'Basic Adjustments')."""
        pass

    @property
    def category(self) -> str:
        """Category for grouping in menus (e.g., 'Tone', 'Color', 'Detail', 'Effects')."""
        return "Tone"

    @property
    def order(self) -> int:
        """Execution order in the GLSL pipeline (lower numbers execute earlier)."""
        return 100

    @property
    def icon(self) -> str:
        """Emoji icon or asset path for the effect title card."""
        return "✨"

    @abstractmethod
    def get_default_settings(self) -> Dict[str, Any]:
        """Returns the default parameter values for this effect."""
        pass

    @abstractmethod
    def create_widget(self, parent: QWidget, on_changed_callback) -> QWidget:
        """
        Creates and returns the UI settings widget for this effect.
        on_changed_callback should be called whenever any slider or control changes.
        """
        pass

    @abstractmethod
    def get_state(self, widget: QWidget) -> Dict[str, Any]:
        """Extracts current parameter values from the UI widget."""
        pass

    @abstractmethod
    def set_state(self, widget: QWidget, state: Dict[str, Any]) -> None:
        """Updates the UI widget with values from a state dictionary."""
        pass

    @abstractmethod
    def reset_state(self, widget: QWidget) -> None:
        """Resets the UI widget controls to their default values."""
        pass

    # --- GLSL Shader Generation ---

    def get_glsl_uniforms(self) -> str:
        """
        Returns GLSL uniform declarations required by this effect.
        Example: 'uniform float exposure; uniform float contrast;'
        """
        return ""

    def get_glsl_functions(self) -> str:
        """
        Returns GLSL helper functions required by this effect.
        """
        return ""

    @abstractmethod
    def get_glsl_execution(self) -> str:
        """
        Returns the GLSL code snippet that processes the current pixel.
        The current color is available in `vec3 linear_rgb` or `vec3 oklch` as appropriate,
        and `vec4 tex_color` represents the input texture sample.
        """
        pass

    def apply_uniforms(self, program: Any, layer_settings: Dict[str, Any], engine: Any) -> None:
        """
        Writes uniform values to the compiled ModernGL shader program.
        Called on every frame before rendering.
        """
        pass
