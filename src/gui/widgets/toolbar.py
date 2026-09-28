from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QSlider, QFrame, QButtonGroup, QComboBox
)
from PySide6.QtCore import Qt, Signal

class ViewportToolbar(QWidget):
    """
    Top quick-access toolbar with 6 organized functional groups.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._init_ui()

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(50, 32)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(400, 32)

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)
        self.setStyleSheet("""
            QPushButton {
                padding: 3px 6px;
                font-size: 11px;
            }
            QComboBox {
                font-size: 11px;
                padding: 2px 4px;
                background-color: #1a222d;
                color: #e6edf3;
                border: 1px solid #2a3b4c;
                border-radius: 4px;
            }
        """)

        def add_vdiv():
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("color: #444; margin: 0 4px;")
            layout.addWidget(line)

        # --- Group 1: Comparison Modes ---
        self.btn_tb_split_v = QPushButton("🗗 Split V")
        self.btn_tb_split_v.setCheckable(True)
        self.btn_tb_split_v.setToolTip("Vertical Split Before/After (V)")
        self.btn_tb_split_v.toggled.connect(self.main_window._toggle_split_v_from_tb)
        layout.addWidget(self.btn_tb_split_v)

        self.btn_tb_split_h = QPushButton("⬒ Split H")
        self.btn_tb_split_h.setCheckable(True)
        self.btn_tb_split_h.setToolTip("Horizontal Split Before/After (H)")
        self.btn_tb_split_h.toggled.connect(self.main_window._toggle_split_h_from_tb)
        layout.addWidget(self.btn_tb_split_h)

        self.btn_tb_side_by_side = QPushButton("◫ Side-by-Side")
        self.btn_tb_side_by_side.setCheckable(True)
        self.btn_tb_side_by_side.setToolTip("Side-by-Side Comparison (B)")
        self.btn_tb_side_by_side.toggled.connect(self.main_window._toggle_side_by_side_from_tb)
        layout.addWidget(self.btn_tb_side_by_side)

        add_vdiv()

        # --- Group 2: Channel & Map Previews ---
        self.btn_tb_mask_prev = QPushButton("🎭 Mask")
        self.btn_tb_mask_prev.setCheckable(True)
        self.btn_tb_mask_prev.setToolTip("Preview Active Layer Mask (M)")
        self.btn_tb_mask_prev.toggled.connect(self.main_window._toggle_mask_prev_from_tb)
        layout.addWidget(self.btn_tb_mask_prev)

        self.btn_tb_depth = QPushButton("🌊 Depth")
        self.btn_tb_depth.setCheckable(True)
        self.btn_tb_depth.setToolTip("Preview 3D Depth Map (D)")
        self.btn_tb_depth.toggled.connect(self.main_window._toggle_depth_from_tb)
        layout.addWidget(self.btn_tb_depth)

        self.btn_tb_normals = QPushButton("🌐 Normals")
        self.btn_tb_normals.setCheckable(True)
        self.btn_tb_normals.setToolTip("Preview 3D Surface Normals (N)")
        self.btn_tb_normals.toggled.connect(self.main_window._toggle_normals_from_tb)
        layout.addWidget(self.btn_tb_normals)

        self.btn_tb_subject = QPushButton("👤 Subject")
        self.btn_tb_subject.setCheckable(True)
        self.btn_tb_subject.setToolTip("Preview AI Subject/Person Mask (S)")
        self.btn_tb_subject.toggled.connect(self.main_window._toggle_subject_from_tb)
        layout.addWidget(self.btn_tb_subject)

        add_vdiv()

        # --- Group 3: Viewport Zoom & Scale ---
        self.btn_tb_fit = QPushButton("Fit")
        self.btn_tb_fit.setToolTip("Fit Photo to Viewport (F)")
        self.btn_tb_fit.clicked.connect(self.main_window.fit_to_viewport)
        layout.addWidget(self.btn_tb_fit)

        self.btn_tb_100 = QPushButton("100%")
        self.btn_tb_100.setToolTip("1:1 Pixel Scale (Z)")
        self.btn_tb_100.clicked.connect(self.main_window.zoom_to_100)
        layout.addWidget(self.btn_tb_100)

        add_vdiv()

        # --- Group 4: Geometry Transform & Crop ---
        self.btn_tb_crop_tool = QPushButton("📐 Crop & Rotate")
        self.btn_tb_crop_tool.setCheckable(True)
        self.btn_tb_crop_tool.setToolTip("Interactive Crop & Straighten on Canvas (C / Enter to apply)")
        self.btn_tb_crop_tool.toggled.connect(self.main_window._toggle_crop_tool_from_tb)
        layout.addWidget(self.btn_tb_crop_tool)

        # 1. Aspect Ratio Lock dropdown — right next to Crop & Rotate!
        self.combo_crop_aspect = QComboBox()
        self.combo_crop_aspect.setToolTip("Crop Aspect Ratio Constraint (e.g. 1:1, 4:5, 3:4, Free)")
        self.combo_crop_aspect.addItems([
            "Free (Custom)",
            "Original (Sensor)",
            "1:1 (Square)",
            "4:5 / 5:4 (Portrait)",
            "3:4 / 4:3 (Photo / iPad)",
            "2:3 / 3:2 (35mm Classic)",
            "9:16 / 16:9 (Stories / TV)",
            "3.5:4.5 / 4.5:3.5 (Passport / ID)",
            "5:7 / 7:5 (Photo 13×18cm)",
            "11:14 / 14:11 (Large Print)"
        ])
        self.combo_crop_aspect.currentTextChanged.connect(self.main_window._on_crop_aspect_ratio_changed)
        layout.addWidget(self.combo_crop_aspect)

        self.btn_crop_swap = QPushButton("🔄")
        self.btn_crop_swap.setToolTip("Swap Aspect Ratio Orientation (Landscape ↔ Portrait)")
        self.btn_crop_swap.clicked.connect(self.main_window._swap_crop_aspect_orientation)
        layout.addWidget(self.btn_crop_swap)

        self.btn_tb_rot_left = QPushButton("↺ 90°")
        self.btn_tb_rot_left.setToolTip("Rotate Left 90°")
        self.btn_tb_rot_left.clicked.connect(self.main_window._rotate_left_90)
        layout.addWidget(self.btn_tb_rot_left)

        self.btn_tb_rot_right = QPushButton("↻ 90°")
        self.btn_tb_rot_right.setToolTip("Rotate Right 90°")
        self.btn_tb_rot_right.clicked.connect(self.main_window._rotate_right_90)
        layout.addWidget(self.btn_tb_rot_right)

        add_vdiv()

        # 2. Multi-Crop Presets dropdown — placed further right with Add and AI Crops buttons!
        self.combo_crop_presets = QComboBox()
        self.combo_crop_presets.setToolTip("Select or switch between Multi-Crop Presets for this photo (e.g. 4:5, 1:1, Full Frame)")
        self.combo_crop_presets.setStyleSheet("""
            QComboBox {
                font-size: 11px;
                font-weight: bold;
                padding: 2px 6px;
                background-color: #16202c;
                color: #80d8ff;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                min-width: 155px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #2a3b4c;
            }
        """)
        self.combo_crop_presets.currentIndexChanged.connect(self.main_window._on_crop_preset_changed)
        layout.addWidget(self.combo_crop_presets)

        self.btn_tb_add_crop = QPushButton("➕")
        self.btn_tb_add_crop.setToolTip("Save current crop framing as a new Preset")
        self.btn_tb_add_crop.setStyleSheet("""
            QPushButton {
                background-color: #16202c;
                border: 1px solid #2d3e52;
                border-radius: 3px;
                color: #80d8ff;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #1e2e40;
                border-color: #00d2ff;
                color: #ffffff;
            }
        """)
        self.btn_tb_add_crop.clicked.connect(self.main_window._on_add_crop_preset_clicked)
        layout.addWidget(self.btn_tb_add_crop)

        self.btn_tb_ai_crop = QPushButton("✨ AI Crops")
        self.btn_tb_ai_crop.setToolTip("Generate intelligent Multi-Crop Presets (4:5 Hero Portrait, 1:1 Square, 4:3 Scene, Full Frame) using AI")
        self.btn_tb_ai_crop.setStyleSheet("""
            QPushButton {
                background-color: #1a2c3d;
                border: 1px solid #00d2ff;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                padding: 2px 6px;
            }
            QPushButton:hover {
                background-color: #243d54;
            }
        """)
        self.btn_tb_ai_crop.clicked.connect(self.main_window.trigger_ai_crop_presets)
        layout.addWidget(self.btn_tb_ai_crop)
        add_vdiv()

        # --- Group 5: Retouch & Sampling ---
        self.btn_tb_wb_pick = QPushButton("🎯 Pick WB")
        self.btn_tb_wb_pick.setCheckable(True)
        self.btn_tb_wb_pick.setToolTip("Pick Neutral White / Gray Balance Sample (W)")
        self.btn_tb_wb_pick.toggled.connect(self.main_window._toggle_wb_picker_from_tb)
        layout.addWidget(self.btn_tb_wb_pick)

        self.btn_tb_focus_pick = QPushButton("🎯 Pick Focus")
        self.btn_tb_focus_pick.setCheckable(True)
        self.btn_tb_focus_pick.setToolTip("Pick 3D Focal Depth point on photo canvas (F)")
        self.btn_tb_focus_pick.toggled.connect(self.main_window._toggle_focus_picker_from_tb)
        layout.addWidget(self.btn_tb_focus_pick)

        self.btn_tb_brush = QPushButton("🖌 Brush Mask")
        self.btn_tb_brush.setCheckable(True)
        self.btn_tb_brush.setToolTip("Paint Custom Layer Mask (P)")
        self.btn_tb_brush.toggled.connect(self.main_window._toggle_brush_from_tb)
        layout.addWidget(self.btn_tb_brush)

        add_vdiv()

        # --- Group 6: AI Magic & Modes ---
        self.btn_tb_ai_auto = QPushButton("✨ AI Auto")
        self.btn_tb_ai_auto.setStyleSheet("QPushButton { background-color: #2b3d4f; border-color: #00d2ff; font-weight: bold; }")
        self.btn_tb_ai_auto.setToolTip("Run AI Auto-Correction (Ctrl+A)")
        self.btn_tb_ai_auto.clicked.connect(self.main_window.trigger_ai_auto_correct)
        layout.addWidget(self.btn_tb_ai_auto)

        self.btn_tb_stop_ai = QPushButton("⏹ Stop AI (Esc)")
        self.btn_tb_stop_ai.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: #ffffff;
                border: 1px solid #ff3333;
                border-radius: 4px;
                font-weight: bold;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #b30000;
                border-color: #ff6666;
            }
        """)
        self.btn_tb_stop_ai.setToolTip("Instantly Stop AI / Batch Auto-Correction (Esc)")
        self.btn_tb_stop_ai.clicked.connect(self.main_window.stop_ai_auto_correct)
        self.btn_tb_stop_ai.setVisible(False)
        layout.addWidget(self.btn_tb_stop_ai)

        add_vdiv()

        # --- Group 7: Fast Viewer Mode ---
        self.btn_tb_viewer = QPushButton("⚡ Fast Viewer")
        self.btn_tb_viewer.setCheckable(True)
        self.btn_tb_viewer.setStyleSheet("""
            QPushButton:checked {
                background-color: #005c8a;
                border: 1px solid #00d2ff;
                color: #ffffff;
                font-weight: bold;
            }
        """)
        self.btn_tb_viewer.setToolTip("Fast Viewer Mode (V): Browse photos instantly in JPEG mode without reading RAW sensor data. Turn off to edit RAW.")
        self.btn_tb_viewer.toggled.connect(self.main_window._on_viewer_mode_toggled)
        layout.addWidget(self.btn_tb_viewer)

        layout.addStretch()

        self._responsive_buttons = [
            (self.btn_tb_split_v, "🗗 Split V", "🗗"),
            (self.btn_tb_split_h, "⬒ Split H", "⬒"),
            (self.btn_tb_side_by_side, "◫ Side-by-Side", "◫"),
            (self.btn_tb_mask_prev, "🎭 Mask", "🎭"),
            (self.btn_tb_depth, "🌊 Depth", "🌊"),
            (self.btn_tb_normals, "🌐 Normals", "🌐"),
            (self.btn_tb_subject, "👤 Subject", "👤"),
            (self.btn_tb_fit, "Fit", "⛶"),
            (self.btn_tb_100, "100%", "1:1"),
            (self.btn_tb_crop_tool, "📐 Crop & Rotate", "📐"),
            (self.btn_tb_rot_left, "↺ 90°", "↺"),
            (self.btn_tb_rot_right, "↻ 90°", "↻"),
            (self.btn_tb_ai_crop, "✨ AI Crops", "✨✂️"),
            (self.btn_tb_wb_pick, "🎯 Pick WB", "🎯W"),
            (self.btn_tb_focus_pick, "🎯 Pick Focus", "🎯F"),
            (self.btn_tb_brush, "🖌 Brush Mask", "🖌"),
            (self.btn_tb_ai_auto, "✨ AI Auto", "✨"),
            (self.btn_tb_stop_ai, "⏹ Stop AI (Esc)", "⏹"),
            (self.btn_tb_viewer, "⚡ Fast Viewer", "⚡"),
        ]
        self._is_compact_mode = None
        self._update_responsive_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_responsive_layout()

    def _update_responsive_layout(self):
        w = self.width()
        is_compact = (w < 1420)
        if getattr(self, '_is_compact_mode', None) == is_compact:
            return
        self._is_compact_mode = is_compact

        for btn, full_t, comp_t in getattr(self, '_responsive_buttons', []):
            if btn:
                btn.setText(comp_t if is_compact else full_t)

        if hasattr(self, 'combo_crop_presets') and self.combo_crop_presets:
            if is_compact:
                self.combo_crop_presets.setMinimumWidth(110)
                self.combo_crop_presets.setMaximumWidth(140)
            else:
                self.combo_crop_presets.setMinimumWidth(155)
                self.combo_crop_presets.setMaximumWidth(220)

        if hasattr(self, 'combo_crop_aspect') and self.combo_crop_aspect:
            if is_compact:
                self.combo_crop_aspect.setMinimumWidth(85)
                self.combo_crop_aspect.setMaximumWidth(110)
            else:
                self.combo_crop_aspect.setMinimumWidth(125)
                self.combo_crop_aspect.setMaximumWidth(160)

    def update_crop_presets_list(self, presets: list, active_id: str = "default"):
        if not hasattr(self, 'combo_crop_presets'):
            return
        self.combo_crop_presets.blockSignals(True)
        self.combo_crop_presets.clear()
        
        if not presets:
            self.combo_crop_presets.addItem("✂️ Default Crop", "default")
            self.combo_crop_presets.blockSignals(False)
            return

        active_idx = 0
        for i, p in enumerate(presets):
            pid = p.get('id', f'preset_{i}')
            pname = p.get('name', f"Preset {i+1}")
            display_text = f"✂️ {pname}"
            self.combo_crop_presets.addItem(display_text, pid)
            if pid == active_id:
                active_idx = i
                
        self.combo_crop_presets.setCurrentIndex(active_idx)
        self.combo_crop_presets.blockSignals(False)

class BrushSettingsBar(QWidget):
    """
    Floating or docked brush settings bar with Radius, Hardness, and Opacity.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)
        self.setStyleSheet("background-color: #252525; border-bottom: 1px solid #3d3d3d;")

        lbl_b = QLabel("🖌 Brush:")
        lbl_b.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px;")
        layout.addWidget(lbl_b)

        # Radius
        layout.addWidget(QLabel("Radius:"))
        self.slider_radius = QSlider(Qt.Horizontal)
        self.slider_radius.setRange(5, 500)
        self.slider_radius.setValue(50)
        self.slider_radius.setFixedWidth(100)
        self.slider_radius.valueChanged.connect(self._on_radius_changed)
        layout.addWidget(self.slider_radius)

        self.lbl_radius = QLabel("50 px")
        self.lbl_radius.setStyleSheet("color: #00d2ff; font-size: 11px;")
        layout.addWidget(self.lbl_radius)

        # Hardness
        layout.addWidget(QLabel("Hardness:"))
        self.slider_hardness = QSlider(Qt.Horizontal)
        self.slider_hardness.setRange(0, 100)
        self.slider_hardness.setValue(50)
        self.slider_hardness.setFixedWidth(80)
        self.slider_hardness.valueChanged.connect(self._on_hardness_changed)
        layout.addWidget(self.slider_hardness)

        self.lbl_hardness = QLabel("50%")
        self.lbl_hardness.setStyleSheet("color: #00d2ff; font-size: 11px;")
        layout.addWidget(self.lbl_hardness)

        # Opacity
        layout.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(1, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.setFixedWidth(80)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.slider_opacity)

        self.lbl_opacity = QLabel("100%")
        self.lbl_opacity.setStyleSheet("color: #00d2ff; font-size: 11px;")
        layout.addWidget(self.lbl_opacity)

        # Clear Mask
        btn_clear = QPushButton("🗑 Clear Mask")
        btn_clear.clicked.connect(self.main_window._clear_brush_mask)
        layout.addWidget(btn_clear)

        # Close
        btn_close = QPushButton("✕")
        btn_close.setFixedWidth(24)
        btn_close.clicked.connect(lambda: self.main_window._toggle_brush_from_tb(False))
        layout.addWidget(btn_close)

        layout.addStretch()

    def _on_radius_changed(self, v):
        self.lbl_radius.setText(f"{v} px")
        self.main_window.image_label.paint_radius = float(v)
        self.main_window.image_label.update()

    def _on_hardness_changed(self, v):
        self.lbl_hardness.setText(f"{v}%")
        self.main_window.image_label.paint_hardness = v / 100.0

    def _on_opacity_changed(self, v):
        self.lbl_opacity.setText(f"{v}%")
        self.main_window.image_label.paint_opacity = v / 100.0
