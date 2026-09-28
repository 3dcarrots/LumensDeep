from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QListWidgetItem, QPushButton, QLabel, QSlider
)
from PySide6.QtCore import Qt

class LayersDock(QDockWidget):
    """
    Dock panel for managing the layer stack:
    Add / Delete / Duplicate layers, Layer Opacity slider, and selection.
    """
    def __init__(self, main_window, parent=None):
        super().__init__("Layers", parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Layer List
        self.layers_list = QListWidget()
        self.layers_list.setSelectionMode(QListWidget.SingleSelection)
        self.layers_list.setStyleSheet("""
            QListWidget {
                background-color: #222;
                border: 1px solid #383838;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #2d2d2d;
            }
            QListWidget::item:selected {
                background-color: #005577;
                color: #fff;
            }
        """)
        self.layers_list.currentRowChanged.connect(self.main_window._on_layer_selection_changed)
        self.layers_list.itemChanged.connect(self.main_window._on_layer_item_changed)
        self.layers_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.layers_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layers_list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.layers_list)

        # Action Buttons: Add, Delete, Duplicate
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self.btn_add_layer = QPushButton("+ New Layer")
        self.btn_add_layer.clicked.connect(self.main_window._add_layer)
        btn_row.addWidget(self.btn_add_layer)

        self.btn_del_layer = QPushButton("- Delete")
        self.btn_del_layer.clicked.connect(self.main_window._delete_layer)
        btn_row.addWidget(self.btn_del_layer)

        self.btn_dup_layer = QPushButton("Duplicate")
        self.btn_dup_layer.clicked.connect(self.main_window._duplicate_layer)
        btn_row.addWidget(self.btn_dup_layer)

        layout.addLayout(btn_row)

        self.btn_auto_bg = QPushButton("🌄 Auto Background Layer")
        self.btn_auto_bg.setStyleSheet("""
            QPushButton {
                background-color: #1e3a47;
                border: 1px solid #005577;
                color: #00d2ff;
                font-weight: bold;
                padding: 4px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #244b5c;
                border: 1px solid #00aacc;
                color: #ffffff;
            }
        """)
        self.btn_auto_bg.clicked.connect(self.main_window.auto_create_background_layer)
        layout.addWidget(self.btn_auto_bg)

        # Layer Opacity Slider
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Layer Opacity:"))

        self.slider_layer_opacity = QSlider(Qt.Horizontal)
        self.slider_layer_opacity.setRange(0, 100)
        self.slider_layer_opacity.setValue(100)
        self.slider_layer_opacity.valueChanged.connect(self.main_window._on_layer_opacity_changed)
        opacity_row.addWidget(self.slider_layer_opacity)

        self.lbl_layer_opacity = QLabel("100%")
        self.lbl_layer_opacity.setFixedWidth(40)
        self.lbl_layer_opacity.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_layer_opacity.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        opacity_row.addWidget(self.lbl_layer_opacity)

        layout.addLayout(opacity_row)
        self.setWidget(container)

    def _on_item_double_clicked(self, item):
        self.layers_list.editItem(item)

    def _show_context_menu(self, pos):
        item = self.layers_list.itemAt(pos)
        if not item:
            return
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #222;
                border: 1px solid #444;
                padding: 4px;
            }
            QMenu::item {
                padding: 4px 18px 4px 18px;
                color: #ddd;
            }
            QMenu::item:selected {
                background-color: #005577;
                color: #fff;
            }
        """)
        action_rename = menu.addAction("✏️ Rename Layer")
        action_dup = menu.addAction("📋 Duplicate Layer")
        action_del = menu.addAction("🗑️ Delete Layer")

        action = menu.exec(self.layers_list.mapToGlobal(pos))
        if action == action_rename:
            self.layers_list.editItem(item)
        elif action == action_dup:
            self.main_window._duplicate_layer()
        elif action == action_del:
            self.main_window._delete_layer()

    def refresh_layers_list(self, layers, active_index: int = 0):
        self.layers_list.blockSignals(True)
        self.layers_list.clear()
        for i, layer in enumerate(layers):
            item = QListWidgetItem(layer.name)
            item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(Qt.Checked if layer.active else Qt.Unchecked)
            self.layers_list.addItem(item)
        if 0 <= active_index < self.layers_list.count():
            self.layers_list.setCurrentRow(active_index)
        self.layers_list.blockSignals(False)
