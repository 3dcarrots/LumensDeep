from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QSlider, QPushButton, QCheckBox, QComboBox, QScrollArea, QGroupBox, QFrame,
    QPlainTextEdit
)
from PySide6.QtCore import Qt

class GlobalDock(QDockWidget):
    """
    Dock panel for Global Settings:
    - Photo Metadata & EXIF
    - Geometry & Crop
    - Lensfun Optical Correction
    - 3D Bokeh Simulation
    - Base Curve & AI Blend
    """
    def __init__(self, main_window, parent=None):
        super().__init__("Global Settings & Lens", parent)
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

        # 0. Photo Metadata (EXIF)
        meta_box = QGroupBox("ℹ️ Photo Info & Metadata")
        meta_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                border: 1px solid #2a3b4c;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 10px;
                background-color: #131922;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        form_meta = QFormLayout(meta_box)
        form_meta.setSpacing(4)
        form_meta.setContentsMargins(8, 8, 8, 8)

        lbl_style = "color: #e0e6ed; font-size: 11px; font-weight: normal;"

        self.lbl_meta_cam = QLabel("-")
        self.lbl_meta_cam.setStyleSheet(lbl_style)
        self.lbl_meta_cam.setWordWrap(True)
        form_meta.addRow("Camera:", self.lbl_meta_cam)

        self.lbl_meta_lens = QLabel("-")
        self.lbl_meta_lens.setStyleSheet(lbl_style)
        self.lbl_meta_lens.setWordWrap(True)
        form_meta.addRow("Lens:", self.lbl_meta_lens)

        self.lbl_meta_focal = QLabel("-")
        self.lbl_meta_focal.setStyleSheet(lbl_style)
        form_meta.addRow("Focal Length:", self.lbl_meta_focal)

        self.lbl_meta_iso = QLabel("-")
        self.lbl_meta_iso.setStyleSheet(lbl_style)
        form_meta.addRow("ISO:", self.lbl_meta_iso)

        self.lbl_meta_aperture = QLabel("-")
        self.lbl_meta_aperture.setStyleSheet(lbl_style)
        form_meta.addRow("Aperture:", self.lbl_meta_aperture)

        self.lbl_meta_shutter = QLabel("-")
        self.lbl_meta_shutter.setStyleSheet(lbl_style)
        form_meta.addRow("Shutter:", self.lbl_meta_shutter)

        self.lbl_meta_res = QLabel("-")
        self.lbl_meta_res.setStyleSheet(lbl_style)
        form_meta.addRow("Dimensions:", self.lbl_meta_res)

        layout.addWidget(meta_box)

        # 0.5 Photo Notes & AI Assessment Box
        notes_box = QGroupBox("📝 Photo Notes & AI Assessment")
        notes_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                border: 1px solid #2a3b4c;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 10px;
                background-color: #131922;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPlainTextEdit {
                background-color: #0d1218;
                color: #e6edf3;
                border: 1px solid #2a3b4c;
                border-radius: 4px;
                padding: 4px;
                font-size: 11px;
            }
            QPlainTextEdit:focus {
                border-color: #00d2ff;
            }
        """)
        notes_layout = QVBoxLayout(notes_box)
        notes_layout.setContentsMargins(6, 6, 6, 6)
        
        self.txt_photo_notes = QPlainTextEdit()
        self.txt_photo_notes.setPlaceholderText("Notes, tags or AI reasoning will appear here...")
        self.txt_photo_notes.setFixedHeight(65)
        self.txt_photo_notes.textChanged.connect(self._on_notes_text_changed)
        notes_layout.addWidget(self.txt_photo_notes)
        layout.addWidget(notes_box)

        # 1. Geometry & Crop (Multi-Crop Presets)
        lbl_geo = QLabel("📐 Geometry & Multi-Crop Presets")
        lbl_geo.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px;")
        layout.addWidget(lbl_geo)

        # Multi-Crop Presets Selector & Actions
        preset_box = QWidget()
        preset_layout = QHBoxLayout(preset_box)
        preset_layout.setContentsMargins(0, 0, 0, 2)
        preset_layout.setSpacing(3)

        lbl_p_tag = QLabel("Preset:")
        lbl_p_tag.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        preset_layout.addWidget(lbl_p_tag)

        self.combo_crop_presets = QComboBox()
        self.combo_crop_presets.setToolTip("Select or switch active Crop Preset for this photo")
        self.combo_crop_presets.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 2px; font-size: 11px;")
        self.combo_crop_presets.addItem("Default Crop (Active)", "default")
        self.combo_crop_presets.currentIndexChanged.connect(self.main_window._on_crop_preset_changed)
        preset_layout.addWidget(self.combo_crop_presets, 1)

        self.btn_add_crop_preset = QPushButton("➕")
        self.btn_add_crop_preset.setFixedWidth(24)
        self.btn_add_crop_preset.setToolTip("Save current crop as a new named Preset")
        self.btn_add_crop_preset.clicked.connect(self.main_window._on_add_crop_preset_clicked)
        preset_layout.addWidget(self.btn_add_crop_preset)

        self.btn_rename_crop_preset = QPushButton("✏️")
        self.btn_rename_crop_preset.setFixedWidth(24)
        self.btn_rename_crop_preset.setToolTip("Rename selected Crop Preset")
        self.btn_rename_crop_preset.clicked.connect(self.main_window._on_rename_crop_preset_clicked)
        preset_layout.addWidget(self.btn_rename_crop_preset)

        self.btn_delete_crop_preset = QPushButton("🗑️")
        self.btn_delete_crop_preset.setFixedWidth(24)
        self.btn_delete_crop_preset.setToolTip("Delete selected Crop Preset")
        self.btn_delete_crop_preset.clicked.connect(self.main_window._on_delete_crop_preset_clicked)
        preset_layout.addWidget(self.btn_delete_crop_preset)

        layout.addWidget(preset_box)

        grid_geo = QGridLayout()
        grid_geo.setSpacing(4)

        # Aspect Ratio Presets
        self.combo_crop_aspect = QComboBox()
        self.combo_crop_aspect.setToolTip("Select standard aspect ratio constraint")
        self.combo_crop_aspect.addItems([
            "Free (Custom)",
            "Original (Sensor)",
            "1:1 (Square / Instagram)",
            "4:5 / 5:4 (Instagram / 8×10\")",
            "3:4 / 4:3 (Photo 6×8\" / iPad)",
            "2:3 / 3:2 (35mm / 4×6\" / 10×15cm)",
            "9:16 / 16:9 (Stories / Reels / TV)",
            "3.5:4.5 / 4.5:3.5 (Passport / ID)",
            "5:7 / 7:5 (Photo 13×18cm)",
            "11:14 / 14:11 (Large Print)"
        ])
        self.combo_crop_aspect.currentTextChanged.connect(self.main_window._on_crop_aspect_ratio_changed)
        grid_geo.addWidget(QLabel("Aspect:"), 0, 0)
        grid_geo.addWidget(self.combo_crop_aspect, 0, 1)

        self.btn_crop_swap = QPushButton("🔄")
        self.btn_crop_swap.setFixedWidth(28)
        self.btn_crop_swap.setToolTip("Swap Aspect Ratio (Landscape ↔ Portrait)")
        self.btn_crop_swap.clicked.connect(self.main_window._swap_crop_aspect_orientation)
        grid_geo.addWidget(self.btn_crop_swap, 0, 2)

        # Straighten
        self.slider_rotation = QSlider(Qt.Horizontal)
        self.slider_rotation.setRange(-450, 450)
        self.slider_rotation.setValue(0)
        self.slider_rotation.valueChanged.connect(self.main_window._on_rotation_changed)
        grid_geo.addWidget(QLabel("Straighten:"), 1, 0)
        grid_geo.addWidget(self.slider_rotation, 1, 1)

        self.lbl_rotation = QLabel("0.0°")
        self.lbl_rotation.setFixedWidth(45)
        self.lbl_rotation.setStyleSheet("color: #00d2ff; font-size: 11px;")
        grid_geo.addWidget(self.lbl_rotation, 1, 2)

        # Crop Left, Right, Top, Bottom
        self.slider_crop_l = QSlider(Qt.Horizontal)
        self.slider_crop_l.setRange(0, 480)
        self.slider_crop_l.setValue(0)
        self.slider_crop_l.valueChanged.connect(self.main_window._on_crop_changed)
        grid_geo.addWidget(QLabel("Crop Left:"), 2, 0)
        grid_geo.addWidget(self.slider_crop_l, 2, 1)

        self.slider_crop_r = QSlider(Qt.Horizontal)
        self.slider_crop_r.setRange(0, 480)
        self.slider_crop_r.setValue(0)
        self.slider_crop_r.valueChanged.connect(self.main_window._on_crop_changed)
        grid_geo.addWidget(QLabel("Crop Right:"), 3, 0)
        grid_geo.addWidget(self.slider_crop_r, 3, 1)

        self.slider_crop_t = QSlider(Qt.Horizontal)
        self.slider_crop_t.setRange(0, 480)
        self.slider_crop_t.setValue(0)
        self.slider_crop_t.valueChanged.connect(self.main_window._on_crop_changed)
        grid_geo.addWidget(QLabel("Crop Top:"), 4, 0)
        grid_geo.addWidget(self.slider_crop_t, 4, 1)

        self.slider_crop_b = QSlider(Qt.Horizontal)
        self.slider_crop_b.setRange(0, 480)
        self.slider_crop_b.setValue(0)
        self.slider_crop_b.valueChanged.connect(self.main_window._on_crop_changed)
        grid_geo.addWidget(QLabel("Crop Bottom:"), 5, 0)
        grid_geo.addWidget(self.slider_crop_b, 5, 1)

        layout.addLayout(grid_geo)

        # 2. Lens Correction
        lbl_lens = QLabel("📷 Lens Correction (Lensfun)")
        lbl_lens.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_lens)

        self.chk_lens_enable = QCheckBox("Enable Optical Correction")
        self.chk_lens_enable.toggled.connect(self.main_window._on_lens_correction_toggled)
        layout.addWidget(self.chk_lens_enable)

        self.combo_cam_maker = QComboBox()
        self.combo_cam_maker.currentIndexChanged.connect(self.main_window._on_cam_maker_changed)
        layout.addWidget(self.combo_cam_maker)

        self.combo_cam_model = QComboBox()
        self.combo_cam_model.currentIndexChanged.connect(self.main_window._on_cam_model_changed)
        layout.addWidget(self.combo_cam_model)

        self.combo_lens_profile = QComboBox()
        self.combo_lens_profile.currentIndexChanged.connect(self.main_window._on_lens_profile_changed)
        layout.addWidget(self.combo_lens_profile)

        # 3. 3D Bokeh Simulation
        lbl_bk = QLabel("✨ 3D Bokeh Blur")
        lbl_bk.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_bk)

        self.check_bokeh_enable = QCheckBox("Enable Bokeh Simulation")
        self.check_bokeh_enable.toggled.connect(self.main_window._on_bokeh_parameter_changed)
        layout.addWidget(self.check_bokeh_enable)

        grid_bk = QGridLayout()
        grid_bk.setSpacing(4)

        # Focal Depth / Distance
        self.slider_bokeh_focal_depth = QSlider(Qt.Horizontal)
        self.slider_bokeh_focal_depth.setRange(0, 100)
        self.slider_bokeh_focal_depth.setValue(50)
        self.slider_bokeh_focal_depth.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Focus (Near ↔ Far):"), 0, 0)
        
        focal_box = QWidget()
        focal_layout = QHBoxLayout(focal_box)
        focal_layout.setContentsMargins(0, 0, 0, 0)
        focal_layout.setSpacing(4)
        focal_layout.addWidget(self.slider_bokeh_focal_depth, 1)

        self.btn_pick_focus = QPushButton("🎯")
        self.btn_pick_focus.setFixedWidth(28)
        self.btn_pick_focus.setCheckable(True)
        self.btn_pick_focus.setToolTip("Pick 3D Focal Depth point from photo on canvas (F)")
        self.btn_pick_focus.setStyleSheet("""
            QPushButton {
                background-color: #1a222d;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                color: #00d2ff;
                font-weight: bold;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #24303f;
                border-color: #00d2ff;
            }
            QPushButton:checked {
                background-color: #006699;
                border-color: #00d2ff;
                color: #ffffff;
            }
        """)
        self.btn_pick_focus.toggled.connect(self.main_window._toggle_focus_picker)
        focal_layout.addWidget(self.btn_pick_focus)

        grid_bk.addWidget(focal_box, 0, 1)

        # Focal Range
        self.slider_bokeh_focal_range = QSlider(Qt.Horizontal)
        self.slider_bokeh_focal_range.setRange(1, 100)
        self.slider_bokeh_focal_range.setValue(12)
        self.slider_bokeh_focal_range.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Focal Range:"), 1, 0)
        grid_bk.addWidget(self.slider_bokeh_focal_range, 1, 1)

        # Max Blur Radius
        self.slider_bokeh_radius = QSlider(Qt.Horizontal)
        self.slider_bokeh_radius.setRange(0, 100)
        self.slider_bokeh_radius.setValue(40)
        self.slider_bokeh_radius.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Max Radius:"), 2, 0)
        grid_bk.addWidget(self.slider_bokeh_radius, 2, 1)

        # Petals / Blades: 0 is 3 blades (Triangle), 1 is 4 blades (Square), ..., 10 is Circular
        self.slider_bokeh_petals = QSlider(Qt.Horizontal)
        self.slider_bokeh_petals.setRange(0, 10)
        self.slider_bokeh_petals.setValue(4) # 7 blades
        self.lbl_petals_title = QLabel("Aperture Blades:")
        self.lbl_petals_val = QLabel("7 (Heptagon)")
        self.lbl_petals_val.setStyleSheet("color: #00d2ff; font-size: 11px;")
        self.slider_bokeh_petals.valueChanged.connect(self._on_petals_slider_changed)
        grid_bk.addWidget(self.lbl_petals_title, 3, 0)
        grid_bk.addWidget(self.slider_bokeh_petals, 3, 1)
        grid_bk.addWidget(self.lbl_petals_val, 3, 2)

        # Samples Quality
        self.combo_bokeh_samples = QComboBox()
        self.combo_bokeh_samples.addItems(["32", "64", "128", "256", "512"])
        self.combo_bokeh_samples.setCurrentText("256")
        self.combo_bokeh_samples.currentIndexChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Quality Samples:"), 4, 0)
        grid_bk.addWidget(self.combo_bokeh_samples, 4, 1)

        # Highlight Boost
        self.slider_bokeh_boost = QSlider(Qt.Horizontal)
        self.slider_bokeh_boost.setRange(0, 200)
        self.slider_bokeh_boost.setValue(50)
        self.slider_bokeh_boost.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Highlight Boost:"), 5, 0)
        grid_bk.addWidget(self.slider_bokeh_boost, 5, 1)

        # FG Gamma
        self.slider_bokeh_fg_gamma = QSlider(Qt.Horizontal)
        self.slider_bokeh_fg_gamma.setRange(0, 400)
        self.slider_bokeh_fg_gamma.setValue(100)
        self.slider_bokeh_fg_gamma.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("FG Gamma:"), 6, 0)
        grid_bk.addWidget(self.slider_bokeh_fg_gamma, 6, 1)

        # BG Gamma
        self.slider_bokeh_bg_gamma = QSlider(Qt.Horizontal)
        self.slider_bokeh_bg_gamma.setRange(0, 400)
        self.slider_bokeh_bg_gamma.setValue(100)
        self.slider_bokeh_bg_gamma.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("BG Gamma:"), 7, 0)
        grid_bk.addWidget(self.slider_bokeh_bg_gamma, 7, 1)

        # Subject Influence
        self.slider_bokeh_subject_influence = QSlider(Qt.Horizontal)
        self.slider_bokeh_subject_influence.setRange(0, 100)
        self.slider_bokeh_subject_influence.setValue(0)
        self.slider_bokeh_subject_influence.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Subject Influence:"), 8, 0)
        grid_bk.addWidget(self.slider_bokeh_subject_influence, 8, 1)

        # Aperture Rotation
        self.slider_bokeh_rotation = QSlider(Qt.Horizontal)
        self.slider_bokeh_rotation.setRange(0, 360)
        self.slider_bokeh_rotation.setValue(0)
        self.slider_bokeh_rotation.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Aperture Angle:"), 9, 0)
        grid_bk.addWidget(self.slider_bokeh_rotation, 9, 1)

        # Chromatic Aberration
        self.slider_bokeh_chromatic = QSlider(Qt.Horizontal)
        self.slider_bokeh_chromatic.setRange(0, 100)
        self.slider_bokeh_chromatic.setValue(0)
        self.slider_bokeh_chromatic.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Chromatic:"), 10, 0)
        grid_bk.addWidget(self.slider_bokeh_chromatic, 10, 1)

        # Optical Vignette / Cat-eye
        self.slider_bokeh_vignette = QSlider(Qt.Horizontal)
        self.slider_bokeh_vignette.setRange(0, 100)
        self.slider_bokeh_vignette.setValue(0)
        self.slider_bokeh_vignette.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Cat-Eye Vignette:"), 11, 0)
        grid_bk.addWidget(self.slider_bokeh_vignette, 11, 1)

        # Radial Center
        self.slider_bokeh_radial_center = QSlider(Qt.Horizontal)
        self.slider_bokeh_radial_center.setRange(0, 100)
        self.slider_bokeh_radial_center.setValue(0)
        self.slider_bokeh_radial_center.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Radial Center:"), 12, 0)
        grid_bk.addWidget(self.slider_bokeh_radial_center, 12, 1)

        # Radial Edge
        self.slider_bokeh_radial_edge = QSlider(Qt.Horizontal)
        self.slider_bokeh_radial_edge.setRange(0, 100)
        self.slider_bokeh_radial_edge.setValue(0)
        self.slider_bokeh_radial_edge.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Radial Edge:"), 13, 0)
        grid_bk.addWidget(self.slider_bokeh_radial_edge, 13, 1)

        # Radial Gamma
        self.slider_bokeh_radial_gamma = QSlider(Qt.Horizontal)
        self.slider_bokeh_radial_gamma.setRange(10, 300)
        self.slider_bokeh_radial_gamma.setValue(100)
        self.slider_bokeh_radial_gamma.valueChanged.connect(self.main_window._on_bokeh_parameter_changed)
        grid_bk.addWidget(QLabel("Radial Gamma:"), 14, 0)
        grid_bk.addWidget(self.slider_bokeh_radial_gamma, 14, 1)

        layout.addLayout(grid_bk)
        layout.addStretch()

        scroll.setWidget(container)
        self.setWidget(scroll)

    def _on_petals_slider_changed(self, val: int):
        blade_names = {
            0: "3 (Triangle)",
            1: "4 (Square)",
            2: "5 (Pentagon)",
            3: "6 (Hexagon)",
            4: "7 (Heptagon)",
            5: "8 (Octagon)",
            6: "9 (Nonagon)",
            7: "10 (Decagon)",
            8: "11 (11-blade)",
            9: "12 (12-blade)",
            10: "Circular (Smooth)"
        }
        self.lbl_petals_val.setText(blade_names.get(val, f"{val+3} blades"))
        self.main_window._on_bokeh_parameter_changed()

    def _on_notes_text_changed(self):
        if hasattr(self.main_window, '_on_photo_notes_changed'):
            self.main_window._on_photo_notes_changed(self.txt_photo_notes.toPlainText())

    def update_metadata(self, metadata: dict, width: int = 0, height: int = 0, ai_notes: str = ""):
        if not metadata:
            metadata = {}

        cam_make = str(metadata.get('cam_maker', '')).strip()
        cam_mod = str(metadata.get('cam_model', '')).strip()
        cam_str = f"{cam_make} {cam_mod}".strip() or "-"
        self.lbl_meta_cam.setText(cam_str)

        lens_str = str(metadata.get('lens_model', '')).strip() or str(metadata.get('lens_maker', '')).strip() or "-"
        self.lbl_meta_lens.setText(lens_str)

        focal = metadata.get('focal_length')
        self.lbl_meta_focal.setText(f"{focal} mm" if focal else "-")

        iso = metadata.get('iso')
        self.lbl_meta_iso.setText(str(iso) if iso else "-")

        aperture = metadata.get('aperture')
        if aperture:
            self.lbl_meta_aperture.setText(f"f/{aperture:.1f}" if isinstance(aperture, float) else f"f/{aperture}")
        else:
            self.lbl_meta_aperture.setText("-")

        shutter = metadata.get('shutter')
        if shutter:
            s_str = str(shutter)
            self.lbl_meta_shutter.setText(s_str if s_str.endswith('s') else f"{s_str}s")
        else:
            self.lbl_meta_shutter.setText("-")

        if width > 0 and height > 0:
            mp = (width * height) / 1_000_000.0
            self.lbl_meta_res.setText(f"{width} × {height} ({mp:.1f} MP)")
        else:
            self.lbl_meta_res.setText("-")

        # Update Notes without triggering recursive textChanged signal
        self.txt_photo_notes.blockSignals(True)
        self.txt_photo_notes.setPlainText(ai_notes or "")
        self.txt_photo_notes.blockSignals(False)

    def update_crop_presets_list(self, presets: list, active_id: str = ""):
        self.combo_crop_presets.blockSignals(True)
        self.combo_crop_presets.clear()
        if not presets:
            self.combo_crop_presets.addItem("Default Crop", "default")
        else:
            for p in presets:
                p_id = p.get('id', 'preset')
                p_name = p.get('name', p_id)
                self.combo_crop_presets.addItem(p_name, p_id)
        
        # Set active item
        if active_id:
            idx = self.combo_crop_presets.findData(active_id)
            if idx >= 0:
                self.combo_crop_presets.setCurrentIndex(idx)
        self.combo_crop_presets.blockSignals(False)
