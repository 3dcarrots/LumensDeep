from typing import Dict, Any, Optional
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QScrollArea, QFrame
)
from PySide6.QtCore import Qt
try:
    from plugins.manager import PluginManager
    from plugins.base import BaseEffectPlugin
except ImportError:
    from ...plugins.manager import PluginManager
    from ...plugins.base import BaseEffectPlugin

class EffectCardWidget(QFrame):
    """
    Accordion card container for an individual Layer Effect Plugin.
    Provides Enable/Disable checkbox, collapse/expand toggle, and reset button.
    """
    def __init__(self, plugin: BaseEffectPlugin, plugin_manager: PluginManager, main_window, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.plugin_manager = plugin_manager
        self.main_window = main_window
        self.is_collapsed = False
        
        self._init_ui()

    def _init_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            EffectCardWidget {
                background-color: #262626;
                border: 1px solid #383838;
                border-radius: 6px;
                margin-bottom: 6px;
            }
        """)

        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(4)

        # Header Row
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(2, 2, 2, 2)
        header_layout.setSpacing(6)

        # Enable/Disable Checkbox (Controls dynamic shader inclusion)
        self.chk_enable = QCheckBox()
        self.chk_enable.setChecked(self.plugin_manager.is_plugin_active(self.plugin.plugin_id))
        self.chk_enable.setToolTip("Enable/Disable effect (Recompiles GPU Shader)")
        self.chk_enable.toggled.connect(self._on_enable_toggled)
        header_layout.addWidget(self.chk_enable)

        # Title / Collapse Button
        self.btn_title = QPushButton(f"{self.plugin.icon} {self.plugin.display_name}")
        self.btn_title.setStyleSheet("""
            QPushButton {
                text-align: left;
                font-weight: bold;
                font-size: 11px;
                background: transparent;
                border: none;
                color: #e0e0e0;
                padding: 2px 4px;
            }
            QPushButton:hover {
                color: #00d2ff;
            }
        """)
        self.btn_title.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self.btn_title, 1)

        # Reset button
        self.btn_reset = QPushButton("↺")
        self.btn_reset.setFixedWidth(24)
        self.btn_reset.setToolTip(f"Reset {self.plugin.display_name} to defaults")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        header_layout.addWidget(self.btn_reset)

        card_layout.addWidget(header_widget)

        # Plugin UI Body Widget
        self.body_container = self.plugin.create_widget(self, on_changed_callback=self._on_parameter_changed)
        card_layout.addWidget(self.body_container)

    def _on_enable_toggled(self, checked: bool):
        self.plugin_manager.set_plugin_active(self.plugin.plugin_id, checked)
        if hasattr(self.main_window, 'engine') and self.main_window.engine:
            self.main_window.engine.recompile_layer_shader(self.plugin_manager)
            self.main_window._update_view()

    def _toggle_collapse(self):
        self.is_collapsed = not self.is_collapsed
        self.body_container.setVisible(not self.is_collapsed)

    def _on_reset_clicked(self):
        self.plugin.reset_state(self.body_container)
        self._on_parameter_changed()

    def _on_parameter_changed(self):
        if hasattr(self.main_window, '_on_layer_parameter_changed'):
            self.main_window._on_layer_parameter_changed()
        elif hasattr(self.main_window, '_update_view'):
            self.main_window._update_view()

class LayerEffectsDock(QDockWidget):
    """
    Dynamic Dock Panel displaying all registered Layer Effect Plugins in an accordion layout.
    """
    def __init__(self, main_window, plugin_manager: PluginManager, parent=None):
        super().__init__("Layer Effects", parent)
        self.main_window = main_window
        self.plugin_manager = plugin_manager
        self._card_widgets: Dict[str, EffectCardWidget] = {}

        self._init_ui()
        self.plugin_manager.plugins_changed.connect(self._rebuild_cards)

    def _init_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(8, 8, 8, 8)
        self.container_layout.setSpacing(6)

        self.scroll_area.setWidget(self.container)
        self.setWidget(self.scroll_area)

        self._rebuild_cards()

    def _rebuild_cards(self):
        # Clear existing cards
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._card_widgets.clear()

        # Add card for each plugin sorted by execution order
        for plugin in self.plugin_manager.get_all_plugins():
            card = EffectCardWidget(plugin, self.plugin_manager, self.main_window, self.container)
            self.container_layout.addWidget(card)
            self._card_widgets[plugin.plugin_id] = card

        self.container_layout.addStretch()

    def get_layer_settings(self) -> Dict[str, Any]:
        """Collects settings dictionary from all effect plugin cards."""
        settings = {}
        for pid, card in self._card_widgets.items():
            st = card.plugin.get_state(card.body_container)
            settings[pid] = st
            # Flatten basic adjustments for backward compatibility
            if pid == "basic_adjustments":
                settings.update(st)
        return settings

    def set_layer_settings(self, settings: Dict[str, Any]):
        """Populates all effect plugin cards from a layer settings dictionary."""
        if not isinstance(settings, dict):
            return
        for pid, card in self._card_widgets.items():
            if pid == "basic_adjustments":
                st = dict(settings.get(pid, {})) if isinstance(settings.get(pid), dict) else {}
                for k, v in settings.items():
                    if k != pid:
                        st[k] = v
                card.plugin.set_state(card.body_container, st)
            else:
                plugin_state = settings.get(pid, settings)
                card.plugin.set_state(card.body_container, plugin_state)

    def reset_all_effects(self):
        """Resets all effect cards to their neutral defaults."""
        for card in self._card_widgets.values():
            card.plugin.reset_state(card.body_container)
