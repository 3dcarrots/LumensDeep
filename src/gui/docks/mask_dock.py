from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QCheckBox, QComboBox, QScrollArea
)
from PySide6.QtCore import Qt

class MaskDock(QDockWidget):
    """
    Dock panel for Parametric, Gradient, and 3D Mask controls.
    """
    def __init__(self, main_window, parent=None):
        super().__init__("Mask & Selection", parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Parametric Color Mask Section
        lbl_p = QLabel("🎯 Color / Parametric Mask")
        lbl_p.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px;")
        layout.addWidget(lbl_p)

        grid_p = QGridLayout()
        grid_p.setSpacing(4)

        # Luma Target & Range
        self.slider_mask_luma_t = QSlider(Qt.Horizontal)
        self.slider_mask_luma_t.setRange(0, 100)
        self.slider_mask_luma_t.setValue(50)
        self.slider_mask_luma_t.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Luma Target:"), 0, 0)
        grid_p.addWidget(self.slider_mask_luma_t, 0, 1)

        self.slider_mask_luma_r = QSlider(Qt.Horizontal)
        self.slider_mask_luma_r.setRange(0, 100)
        self.slider_mask_luma_r.setValue(0)
        self.slider_mask_luma_r.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Luma Range:"), 1, 0)
        grid_p.addWidget(self.slider_mask_luma_r, 1, 1)

        # Hue Target & Range
        self.slider_mask_hue_t = QSlider(Qt.Horizontal)
        self.slider_mask_hue_t.setRange(0, 360)
        self.slider_mask_hue_t.setValue(0)
        self.slider_mask_hue_t.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Hue Target:"), 2, 0)
        grid_p.addWidget(self.slider_mask_hue_t, 2, 1)

        self.slider_mask_hue_r = QSlider(Qt.Horizontal)
        self.slider_mask_hue_r.setRange(0, 180)
        self.slider_mask_hue_r.setValue(0)
        self.slider_mask_hue_r.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Hue Range:"), 3, 0)
        grid_p.addWidget(self.slider_mask_hue_r, 3, 1)

        # Saturation Target & Range
        self.slider_mask_sat_t = QSlider(Qt.Horizontal)
        self.slider_mask_sat_t.setRange(0, 100)
        self.slider_mask_sat_t.setValue(50)
        self.slider_mask_sat_t.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Sat Target:"), 4, 0)
        grid_p.addWidget(self.slider_mask_sat_t, 4, 1)

        self.slider_mask_sat_r = QSlider(Qt.Horizontal)
        self.slider_mask_sat_r.setRange(0, 100)
        self.slider_mask_sat_r.setValue(0)
        self.slider_mask_sat_r.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Sat Range:"), 5, 0)
        grid_p.addWidget(self.slider_mask_sat_r, 5, 1)

        # Softness
        self.slider_mask_softness = QSlider(Qt.Horizontal)
        self.slider_mask_softness.setRange(1, 100)
        self.slider_mask_softness.setValue(50)
        self.slider_mask_softness.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Softness:"), 6, 0)
        grid_p.addWidget(self.slider_mask_softness, 6, 1)

        # Fill Mask
        self.slider_mask_fill = QSlider(Qt.Horizontal)
        self.slider_mask_fill.setRange(0, 100)
        self.slider_mask_fill.setValue(0)
        self.slider_mask_fill.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_p.addWidget(QLabel("Fill Mask:"), 7, 0)
        grid_p.addWidget(self.slider_mask_fill, 7, 1)

        layout.addLayout(grid_p)

        # Eyedropper Color Picker Button
        self.btn_pick_mask = QPushButton("🎯 Pick Color Sample")
        self.btn_pick_mask.setCheckable(True)
        self.btn_pick_mask.toggled.connect(self.main_window._toggle_mask_picker)
        layout.addWidget(self.btn_pick_mask)

        # 2. Gradient Mask Section
        lbl_g = QLabel("📐 Gradient Mask")
        lbl_g.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_g)

        self.combo_grad_type = QComboBox()
        self.combo_grad_type.addItems(["None", "Left to Right", "Right to Left", "Top to Bottom", "Bottom to Top", "Radial"])
        self.combo_grad_type.currentIndexChanged.connect(self.main_window._on_mask_parameter_changed)
        layout.addWidget(self.combo_grad_type)

        grid_g = QGridLayout()
        grid_g.setSpacing(4)

        self.slider_grad_black = QSlider(Qt.Horizontal)
        self.slider_grad_black.setRange(0, 100)
        self.slider_grad_black.setValue(0)
        self.slider_grad_black.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_g.addWidget(QLabel("Black Point:"), 0, 0)
        grid_g.addWidget(self.slider_grad_black, 0, 1)

        self.slider_grad_white = QSlider(Qt.Horizontal)
        self.slider_grad_white.setRange(0, 100)
        self.slider_grad_white.setValue(100)
        self.slider_grad_white.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_g.addWidget(QLabel("White Point:"), 1, 0)
        grid_g.addWidget(self.slider_grad_white, 1, 1)

        self.slider_grad_gamma = QSlider(Qt.Horizontal)
        self.slider_grad_gamma.setRange(10, 300)
        self.slider_grad_gamma.setValue(100)
        self.slider_grad_gamma.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_g.addWidget(QLabel("Gamma:"), 2, 0)
        grid_g.addWidget(self.slider_grad_gamma, 2, 1)

        layout.addLayout(grid_g)

        # 3. 3D Depth & Normal Mask Section
        lbl_3d = QLabel("🌐 3D Depth & Normal Mask")
        lbl_3d.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_3d)

        self.chk_use_depth = QCheckBox("Enable 3D Depth Mask")
        self.chk_use_depth.toggled.connect(self.main_window._on_mask_parameter_changed)
        layout.addWidget(self.chk_use_depth)

        grid_d = QGridLayout()
        grid_d.setSpacing(4)

        self.slider_depth_min = QSlider(Qt.Horizontal)
        self.slider_depth_min.setRange(-10, 110)
        self.slider_depth_min.setValue(0)
        self.slider_depth_min.setToolTip("Near Depth threshold (-10% to 110%). At -10%, near-side falloff is disabled (full 100% white on foreground).")
        self.slider_depth_min.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_d.addWidget(QLabel("Min Depth:"), 0, 0)
        grid_d.addWidget(self.slider_depth_min, 0, 1)

        self.slider_depth_max = QSlider(Qt.Horizontal)
        self.slider_depth_max.setRange(-10, 110)
        self.slider_depth_max.setValue(110)
        self.slider_depth_max.setToolTip("Far Depth threshold (-10% to 110%). At 110%, far-side falloff is disabled (full 100% white on background).")
        self.slider_depth_max.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_d.addWidget(QLabel("Max Depth:"), 1, 0)
        grid_d.addWidget(self.slider_depth_max, 1, 1)

        self.slider_depth_softness = QSlider(Qt.Horizontal)
        self.slider_depth_softness.setRange(1, 100)
        self.slider_depth_softness.setValue(50)
        self.slider_depth_softness.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_d.addWidget(QLabel("Depth Softness:"), 2, 0)
        grid_d.addWidget(self.slider_depth_softness, 2, 1)

        self.slider_depth_gamma = QSlider(Qt.Horizontal)
        self.slider_depth_gamma.setRange(10, 300)
        self.slider_depth_gamma.setValue(100)
        self.slider_depth_gamma.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_d.addWidget(QLabel("Depth Gamma:"), 3, 0)
        grid_d.addWidget(self.slider_depth_gamma, 3, 1)

        layout.addLayout(grid_d)

        # One-click Background Layer Creator Button
        self.btn_auto_background_layer = QPushButton("🌄 Create Background Layer (3D Depth)")
        self.btn_auto_background_layer.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #005577, stop:1 #0088aa);
                color: #ffffff;
                font-weight: bold;
                border-radius: 4px;
                padding: 6px;
                margin-top: 2px;
                margin-bottom: 4px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007799, stop:1 #00aacc);
            }
        """)
        self.btn_auto_background_layer.clicked.connect(self.main_window.auto_create_background_layer)
        layout.addWidget(self.btn_auto_background_layer)

        self.chk_use_normal = QCheckBox("Enable 3D Surface Normal Mask")
        self.chk_use_normal.toggled.connect(self.main_window._on_mask_parameter_changed)
        layout.addWidget(self.chk_use_normal)

        grid_n = QGridLayout()
        grid_n.setSpacing(4)

        self.slider_norm_x = QSlider(Qt.Horizontal)
        self.slider_norm_x.setRange(-100, 100)
        self.slider_norm_x.setValue(0)
        self.slider_norm_x.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_n.addWidget(QLabel("Normal X:"), 0, 0)
        grid_n.addWidget(self.slider_norm_x, 0, 1)

        self.slider_norm_y = QSlider(Qt.Horizontal)
        self.slider_norm_y.setRange(-100, 100)
        self.slider_norm_y.setValue(0)
        self.slider_norm_y.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_n.addWidget(QLabel("Normal Y:"), 1, 0)
        grid_n.addWidget(self.slider_norm_y, 1, 1)

        self.slider_norm_angle = QSlider(Qt.Horizontal)
        self.slider_norm_angle.setRange(1, 180)
        self.slider_norm_angle.setValue(45)
        self.slider_norm_angle.valueChanged.connect(self.main_window._on_mask_parameter_changed)
        grid_n.addWidget(QLabel("Angle Tolerance:"), 2, 0)
        grid_n.addWidget(self.slider_norm_angle, 2, 1)

        layout.addLayout(grid_n)
        layout.addStretch()

        scroll.setWidget(container)
        self.setWidget(scroll)
