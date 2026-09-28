import os
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QScrollArea, QProgressBar,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QCheckBox, QSlider
)
from PySide6.QtCore import Qt, Signal

DEFAULT_AI_INSTRUCTIONS = (
    "🎯 CRITICAL MANDATORY DIRECTIVE: PROFESSIONALLY ADJUST THE WHITE BALANCE SO THE IMAGE LOOKS LIKE A PRESTIGIOUS STUDIO PHOTOGRAPH WITH FLAWLESS, PRISTINE NATURAL SKIN TONES AND ABSOLUTELY ZERO BLOWN-OUT HIGHLIGHTS OR HARSH HOTSPOTS ON FACES!\n"
    "• Faces & Skin Tones are the absolute #1 priority! Adjust White Balance (Temperature & Tint) dynamically based on scene lighting to eliminate muddy casts, achieving clean natural porcelain skin and crisp neutral whites. Protect facial highlights gently (rely on Highlights slider and exposure balance; ⚠️ DO NOT overuse Highlight Recovery, as pushing it too hard breaks natural gradient fall-off on lamps, candles, and specular highlights, causing ugly inverted dark artifacts!).\n"
    "• 🎨 Color Equalizer & Warm Artistic Atmosphere: Intelligently tune Color Equalizer to clean unpleasant muddy yellow/green casts from lighting. IMPORTANT: If warmth on the face/scene is an intentional artistic element (e.g. warm candlelight, golden hour glow, cozy ambient mood), allow tasteful, healthy warmth on the skin to preserve the warm emotional atmosphere and color richness of the portrait! Keep adjustments moderate so the photo never looks flat, washed-out, or grey.\n"
    "• 🌟 3D Facial Volume: Preserve natural 3D facial contours and depth (do not flatten faces by over-lifting mid-shadows; keep tonal transitions smooth and realistic).\n"
    "• Sculpt crisp micro-contrast and clarity into eyes, eyelashes, and hair with Local Contrast, Sharpen, and rich anchored blacks.\n"
    "• 📐 Geometry & Subject Centering: Level tilted lines with 'straighten'. When the subject looks into the camera, CENTER the subject horizontally by trimming empty periphery; never cut through limbs, elbows, or dress. If looking to the side, leave lead room in gaze direction without cutting the back of the subject. Always preserve comfortable natural headroom above hair/bows."
)

class PromptInputEdit(QPlainTextEdit):
    """
    Multi-line text editor that submits on Enter / Return and inserts newline on Shift+Enter.
    """
    submit_requested = Signal()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
            else:
                event.accept()
                self.submit_requested.emit()
                return
        else:
            super().keyPressEvent(event)

class AIDock(QDockWidget):
    """
    Dock panel for Gemini 3.8 Flash Assistant, Auto-Correction, Natural Language Prompts,
    and 3D Neural Depth/Normals/Subject Maps.
    """
    def __init__(self, main_window, parent=None):
        super().__init__("AI Assistant & 3D Tools", parent)
        self.main_window = main_window
        self._init_ui()

    def _init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        input_style = """
            QLineEdit {
                background-color: #1a222d;
                color: #ffffff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 11px;
            }
            QLineEdit:focus {
                border: 1px solid #00d2ff;
                background-color: #1e2937;
            }
            QLineEdit::placeholder {
                color: #78909c;
            }
        """

        btn_style = """
            QPushButton {
                background-color: #24303f;
                color: #e0e0e0;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 5px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #314256;
                border-color: #00d2ff;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #1a222d;
            }
        """

        # =========================================================================
        # 1. Gemini 3.8 Flash Configuration Box
        # =========================================================================
        gemini_box = QGroupBox("✨ Gemini 3.8 Flash Engine")
        gemini_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                font-size: 11px;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        gemini_layout = QVBoxLayout(gemini_box)
        gemini_layout.setSpacing(8)

        # Label: Google API Key
        lbl_api_key = QLabel("🔑 Google Gemini API Key:")
        lbl_api_key.setStyleSheet("font-weight: bold; color: #cfd8dc; font-size: 11px;")
        gemini_layout.addWidget(lbl_api_key)

        # API Key Input Row (Input + Show/Hide + Save)
        key_layout = QHBoxLayout()
        key_layout.setSpacing(4)

        self.edit_api_key = QLineEdit()
        self.edit_api_key.setStyleSheet(input_style)
        self.edit_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_api_key.setPlaceholderText("Paste API key here (AIzaSy...)")
        
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key and env_key != "your_api_key_here":
            self.edit_api_key.setText(env_key)

        self.btn_toggle_key = QPushButton("👁")
        self.btn_toggle_key.setFixedWidth(30)
        self.btn_toggle_key.setStyleSheet(btn_style)
        self.btn_toggle_key.setToolTip("Show / Hide API Key")
        self.btn_toggle_key.clicked.connect(self._on_toggle_key_visibility)

        self.btn_save_key = QPushButton("💾 Save")
        self.btn_save_key.setFixedWidth(54)
        self.btn_save_key.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #00d2ff;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #005577;
                color: #ffffff;
            }
        """)
        self.btn_save_key.setToolTip("Save API Key permanently to .env and environment")
        self.btn_save_key.clicked.connect(self._on_save_api_key)
        
        key_layout.addWidget(self.edit_api_key)
        key_layout.addWidget(self.btn_toggle_key)
        key_layout.addWidget(self.btn_save_key)
        gemini_layout.addLayout(key_layout)

        # Model Hyperparameters Form
        form_params = QFormLayout()
        form_params.setSpacing(6)
        form_params.setContentsMargins(0, 4, 0, 2)

        self.combo_thinking_level = QComboBox()
        self.combo_thinking_level.addItems([
            "medium (Recommended / Balanced)",
            "low (Fast / Low Latency)",
            "high (Deep Analysis / Complex Scenes)"
        ])
        self.combo_thinking_level.setCurrentIndex(0)
        self.combo_thinking_level.setToolTip("Gemini reasoning / thinking level (medium - optimal for fine-art, low - fast response, high - deep analysis)")
        self.combo_thinking_level.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 3px;")
        form_params.addRow("Thinking Level:", self.combo_thinking_level)

        self.spin_max_passes = QSpinBox()
        self.spin_max_passes.setRange(1, 50)
        self.spin_max_passes.setValue(5)
        self.spin_max_passes.setToolTip("Maximum refinement iterations (Passes move sliders by 50% towards suggested adjustments until convergence)")
        self.spin_max_passes.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 3px;")
        form_params.addRow("Max Passes:", self.spin_max_passes)

        self.combo_ai_res = QComboBox()
        self.combo_ai_res.addItems([
            "1024px (Optimal for Fine-Art)",
            "768px (Fast Mode)",
            "1536px (Maximum Detail)"
        ])
        self.combo_ai_res.setCurrentIndex(0)
        self.combo_ai_res.setToolTip("Preview resolution sent to Gemini AI. Full resolution is rendered locally at 100%.")
        self.combo_ai_res.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 3px; font-size: 11px;")
        form_params.addRow("AI Image Size:", self.combo_ai_res)

        gemini_layout.addLayout(form_params)
        layout.addWidget(gemini_box)

        # =========================================================================
        # 2. AI Auto-Correction & Prompts
        # =========================================================================
        lbl_auto = QLabel("⚡ Gemini Creative Assistant")
        lbl_auto.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 4px;")
        layout.addWidget(lbl_auto)

        self.btn_auto_correct = QPushButton("✨ Run Gemini Auto-Correction")
        self.btn_auto_correct.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                font-weight: bold;
                padding: 7px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #006699;
            }
        """)
        self.btn_auto_correct.clicked.connect(self.main_window.trigger_ai_auto_correct)
        layout.addWidget(self.btn_auto_correct)

        self.chk_generate_multi_crop = QCheckBox("✨ Auto-Generate Multi-Crop Presets (4:5, 1:1, 4:3)")
        self.chk_generate_multi_crop.setChecked(True)
        self.chk_generate_multi_crop.setToolTip("When running AI Auto-Correction, instructs Gemini to generate 2 to 3 alternative artistic crop presets per photo for one-click multi-crop export.")
        self.chk_generate_multi_crop.setStyleSheet("QCheckBox { color: #80d8ff; font-size: 10px; font-weight: bold; margin-top: 2px; } QCheckBox::indicator { width: 13px; height: 13px; }")
        layout.addWidget(self.chk_generate_multi_crop)

        self.chk_enable_ai_qc = QCheckBox("🔍 Run AI Quality Control Verification (Bokeh / BG)")
        self.chk_enable_ai_qc.setChecked(False)
        self.chk_enable_ai_qc.setToolTip("Runs an extra fast AI Quality Control pass to verify background layer naturalness and subject sharpness before completing.")
        self.chk_enable_ai_qc.setStyleSheet("QCheckBox { color: #80d8ff; font-size: 10px; font-weight: bold; margin-top: 2px; } QCheckBox::indicator { width: 13px; height: 13px; }")
        layout.addWidget(self.chk_enable_ai_qc)

        self.btn_gen_crops = QPushButton("📐 Generate AI Multi-Crops")
        self.btn_gen_crops.setStyleSheet("""
            QPushButton {
                background-color: #16202c;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                font-weight: bold;
                padding: 5px;
                color: #80d8ff;
                font-size: 11px;
                margin-top: 2px;
            }
            QPushButton:hover {
                background-color: #1e3a5f;
                color: #ffffff;
            }
        """)
        self.btn_gen_crops.setToolTip("Analyzes current graded photo without crops and generates 3-4 professional composition presets (Hero 4:5, Square 1:1, Story 4:3, Full Frame)")
        self.btn_gen_crops.clicked.connect(self.main_window.trigger_ai_crop_presets)
        layout.addWidget(self.btn_gen_crops)

        self.btn_batch_crop = QPushButton("📐 Batch AI Crop All Photos...")
        self.btn_batch_crop.setStyleSheet("""
            QPushButton {
                background-color: #142a25;
                border: 1px solid #00e676;
                border-radius: 4px;
                font-weight: bold;
                padding: 5px;
                color: #b9f6ca;
                font-size: 11px;
                margin-top: 2px;
            }
            QPushButton:hover {
                background-color: #1b3d36;
                color: #ffffff;
            }
        """)
        self.btn_batch_crop.setToolTip("Runs AI Auto-Cropping on all photos in the filmstrip (deletes old presets, preserves color grading & layers)")
        self.btn_batch_crop.clicked.connect(lambda: self.main_window.trigger_batch_ai_crop_presets() if hasattr(self.main_window, 'trigger_batch_ai_crop_presets') else None)
        layout.addWidget(self.btn_batch_crop)

        # Batch Aspect Ratio Selector
        aspect_box = QWidget()
        aspect_layout = QHBoxLayout(aspect_box)
        aspect_layout.setContentsMargins(0, 3, 0, 1)
        aspect_layout.setSpacing(4)

        lbl_aspect = QLabel("📐 Aspect:")
        lbl_aspect.setStyleSheet("color: #cfd8dc; font-size: 11px; font-weight: bold;")
        aspect_layout.addWidget(lbl_aspect)

        self.combo_crop_aspect = QComboBox()
        self.combo_crop_aspect.setToolTip("Target Crop Aspect Ratio constraint for all batch processed photos")
        self.combo_crop_aspect.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 3px; font-size: 11px;")
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
        aspect_layout.addWidget(self.combo_crop_aspect, 1)

        self.btn_crop_swap = QPushButton("🔄")
        self.btn_crop_swap.setFixedWidth(28)
        self.btn_crop_swap.setToolTip("Swap Aspect Ratio Orientation (Landscape ↔ Portrait)")
        self.btn_crop_swap.setStyleSheet("""
            QPushButton {
                background-color: #1a222d;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                color: #00d2ff;
                font-weight: bold;
                padding: 3px;
            }
            QPushButton:hover {
                background-color: #24303f;
                border-color: #00d2ff;
            }
        """)
        self.btn_crop_swap.clicked.connect(self.main_window._swap_crop_aspect_orientation)
        aspect_layout.addWidget(self.btn_crop_swap)

        layout.addWidget(aspect_box)

        # --- Batch AI Auto-Export Options Box ---
        batch_export_box = QFrame()
        batch_export_box.setStyleSheet("""
            QFrame {
                background-color: #121820;
                border: 1px solid #233040;
                border-radius: 6px;
            }
        """)
        bx_layout = QVBoxLayout(batch_export_box)
        bx_layout.setContentsMargins(8, 8, 8, 8)
        bx_layout.setSpacing(6)

        self.chk_batch_auto_export = QCheckBox("💾 Auto-Export each photo on AI complete")
        self.chk_batch_auto_export.setChecked(True)
        self.chk_batch_auto_export.setToolTip("Automatically renders and exports each photo to JPEG (85%) after AI auto-correction")
        self.chk_batch_auto_export.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        bx_layout.addWidget(self.chk_batch_auto_export)

        folder_row = QHBoxLayout()
        folder_row.setSpacing(4)
        
        self.txt_batch_export_folder = QLineEdit()
        self.txt_batch_export_folder.setPlaceholderText("Subfolder: LDExport (Default)")
        self.txt_batch_export_folder.setToolTip("Custom export destination directory. Leave blank to automatically save in '<Photo_Folder>/LDExport'.")
        self.txt_batch_export_folder.setStyleSheet("""
            QLineEdit {
                background-color: #0b0e14;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 4px 6px;
                color: #e6edf3;
                font-size: 10px;
            }
        """)
        folder_row.addWidget(self.txt_batch_export_folder, stretch=1)

        self.btn_batch_export_browse = QPushButton("📁")
        self.btn_batch_export_browse.setToolTip("Browse custom export directory...")
        self.btn_batch_export_browse.setFixedSize(26, 24)
        self.btn_batch_export_browse.clicked.connect(self._on_browse_batch_export_folder)
        folder_row.addWidget(self.btn_batch_export_browse)

        self.btn_batch_export_reset = QPushButton("↺")
        self.btn_batch_export_reset.setToolTip("Reset to default '<Photo_Folder>/LDExport' subfolder")
        self.btn_batch_export_reset.setFixedSize(26, 24)
        self.btn_batch_export_reset.clicked.connect(lambda: self.txt_batch_export_folder.clear())
        folder_row.addWidget(self.btn_batch_export_reset)

        bx_layout.addLayout(folder_row)
        
        self.chk_batch_reset_settings = QCheckBox("🔄 Reset adjustments to neutral baseline before AI")
        self.chk_batch_reset_settings.setChecked(True)
        self.chk_batch_reset_settings.setToolTip("Resets all layers, color adjustments, tone curves, and masks to neutral baseline before running AI, ensuring pristine and consistent grading across the entire shoot.")
        self.chk_batch_reset_settings.setStyleSheet("color: #80d8ff; font-weight: bold; font-size: 11px; margin-top: 4px;")
        bx_layout.addWidget(self.chk_batch_reset_settings)

        self.chk_batch_lens_correction = QCheckBox("📷 Apply Optical / Lens Correction to all photos")
        self.chk_batch_lens_correction.setChecked(True)
        self.chk_batch_lens_correction.setToolTip("Automatically identifies camera model, lens, and focal length from EXIF metadata and applies Lensfun optical geometry and chromatic aberration correction to every photo.")
        self.chk_batch_lens_correction.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px; margin-top: 2px;")
        bx_layout.addWidget(self.chk_batch_lens_correction)

        self.chk_ai_disable_bokeh = QCheckBox("🚫 Forbid AI from enabling 3D Bokeh")
        self.chk_ai_disable_bokeh.setChecked(True)
        self.chk_ai_disable_bokeh.setToolTip("When checked, prevents AI from automatically turning on 3D Bokeh on photos (keeps Bokeh disabled).")
        self.chk_ai_disable_bokeh.setStyleSheet("color: #ff80ab; font-weight: bold; font-size: 11px; margin-top: 2px;")
        bx_layout.addWidget(self.chk_ai_disable_bokeh)

        self.chk_batch_skip_background = QCheckBox("🚫 Do not tune Background layer (Delete/Skip)")
        self.chk_batch_skip_background.setChecked(False)
        self.chk_batch_skip_background.setToolTip("When checked, removes the Background layer if present and avoids creating it during batch AI processing (all adjustments apply only to Base Layer).")
        self.chk_batch_skip_background.setStyleSheet("color: #ffb74d; font-weight: bold; font-size: 11px; margin-top: 2px;")
        bx_layout.addWidget(self.chk_batch_skip_background)

        self.chk_batch_dual_bokeh_export = QCheckBox("✨ Dual Export for Bokeh (_bokeh + clean)")
        self.chk_batch_dual_bokeh_export.setChecked(False)
        self.chk_batch_dual_bokeh_export.setToolTip("If Bokeh is enabled on a photo, automatically exports two versions:\n1. With Bokeh effect (suffix '_bokeh.jpg')\n2. Without Bokeh (clean sharp edit, without suffix)")
        self.chk_batch_dual_bokeh_export.setStyleSheet("color: #b388ff; font-weight: bold; font-size: 11px; margin-top: 2px;")
        bx_layout.addWidget(self.chk_batch_dual_bokeh_export)

        self.chk_batch_skip_already_processed = QCheckBox("⏩ Skip already processed (Resume & Export missing)")
        self.chk_batch_skip_already_processed.setChecked(True)
        self.chk_batch_skip_already_processed.setToolTip("When enabled in Batch AI / Full Auto: if a photo already has a saved project (.lumendeep sidecar with adjustments), skips AI re-tuning and immediately exports it if missing from LDExport (or skips entirely if already exported).")
        self.chk_batch_skip_already_processed.setStyleSheet("color: #69f0ae; font-weight: bold; font-size: 11px; margin-top: 2px;")
        bx_layout.addWidget(self.chk_batch_skip_already_processed)

        batch_dn_row = QHBoxLayout()
        batch_dn_row.setSpacing(6)
        self.chk_batch_auto_denoise = QCheckBox("🔇 Auto-Denoise before export when ISO >")
        self.chk_batch_auto_denoise.setChecked(True)
        self.chk_batch_auto_denoise.setStyleSheet("QCheckBox { color: #00e676; font-size: 11px; font-weight: bold; margin-top: 2px; } QCheckBox::indicator { width: 13px; height: 13px; }")
        self.chk_batch_auto_denoise.setToolTip("Automatically executes 'AI Denoise Current Photo (In-Memory)' during batch AI processing / auto-correction before exporting when the photo's ISO is above this threshold.")
        batch_dn_row.addWidget(self.chk_batch_auto_denoise)

        self.spin_batch_auto_denoise_iso = QSpinBox()
        self.spin_batch_auto_denoise_iso.setRange(50, 102400)
        self.spin_batch_auto_denoise_iso.setSingleStep(50)
        self.spin_batch_auto_denoise_iso.setValue(200)
        self.spin_batch_auto_denoise_iso.setFixedWidth(64)
        self.spin_batch_auto_denoise_iso.setStyleSheet("background-color: #14241a; color: #00e676; font-weight: bold; font-size: 11px; border: 1px solid #1b5e20; border-radius: 4px; padding: 2px;")
        batch_dn_row.addWidget(self.spin_batch_auto_denoise_iso)
        batch_dn_row.addStretch()
        bx_layout.addLayout(batch_dn_row)

        layout.addWidget(batch_export_box)

        self.btn_batch_auto_correct = QPushButton("⚡ Batch AI Auto-Correct (Filmstrip)")
        self.btn_batch_auto_correct.setToolTip("Sequentially auto-correct all non-rejected photos currently visible in Filmstrip (auto-creating 3D Background if needed)")
        self.btn_batch_auto_correct.setStyleSheet("""
            QPushButton {
                background-color: #2b2b4f;
                border: 1px solid #7c4dff;
                border-radius: 4px;
                font-weight: bold;
                padding: 6px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4a148c;
            }
        """)
        self.btn_batch_auto_correct.clicked.connect(self.main_window.trigger_batch_ai_auto_correct)
        layout.addWidget(self.btn_batch_auto_correct)

        self.btn_full_auto = QPushButton("🚀 Full Auto: Cull + Auto-Correct All")
        self.btn_full_auto.setToolTip("1-Click Unified Workflow: AI Culling (rate & filter rejects) -> Filter Rating ≥ 4★ -> Auto 3D Depth Map/Background -> Multi-Pass AI Auto-Correction & Export on top photos")
        self.btn_full_auto.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a4a6e, stop:1 #4a148c);
                border: 1px solid #00d2ff;
                border-radius: 4px;
                font-weight: bold;
                padding: 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #256699, stop:1 #6a1b9a);
                border-color: #80d8ff;
            }
        """)
        self.btn_full_auto.clicked.connect(self.main_window.trigger_full_auto_cull_and_process)
        layout.addWidget(self.btn_full_auto)

        self.btn_stop_ai = QPushButton("⏹ Stop AI / Batch Auto-Correction (Esc)")
        self.btn_stop_ai.setToolTip("Instantly stop any active AI Culling, Batch Processing, or Iterative Auto-Correction (Esc)")
        self.btn_stop_ai.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b0000, stop:1 #d32f2f);
                border: 1px solid #ff5252;
                border-radius: 4px;
                font-weight: bold;
                padding: 8px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b71c1c, stop:1 #f44336);
                border-color: #ff8a80;
            }
        """)
        self.btn_stop_ai.setVisible(False)
        self.btn_stop_ai.clicked.connect(self.main_window.stop_ai_auto_correct)
        layout.addWidget(self.btn_stop_ai)

        # =========================================================================
        # 2. Persistent Auto-Correction Rules & Directives
        # =========================================================================
        rules_box = QGroupBox("📋 Auto-Correction Rules & Directives")
        rules_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                font-size: 11px;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        rbox_layout = QVBoxLayout(rules_box)
        rbox_layout.setSpacing(6)

        rules_header = QWidget()
        rules_hlayout = QHBoxLayout(rules_header)
        rules_hlayout.setContentsMargins(0, 0, 0, 0)
        rules_hlayout.setSpacing(4)

        lbl_rules_title = QLabel("Rules for 'Run Gemini Auto-Correction' & Batch AI:")
        lbl_rules_title.setStyleSheet("font-weight: bold; color: #cfd8dc; font-size: 10px;")
        rules_hlayout.addWidget(lbl_rules_title, 1)

        self.btn_reset_rules = QPushButton("↺ Reset")
        self.btn_reset_rules.setFixedWidth(56)
        self.btn_reset_rules.setToolTip("Reset to default fine-art retouching guidelines")
        self.btn_reset_rules.setStyleSheet("""
            QPushButton {
                background-color: #24303f;
                border: 1px solid #3b4d61;
                border-radius: 3px;
                color: #00d2ff;
                font-size: 10px;
                padding: 2px 4px;
            }
            QPushButton:hover {
                background-color: #2d3e52;
                border: 1px solid #00d2ff;
            }
        """)
        self.btn_reset_rules.clicked.connect(self.reset_autocorrect_rules)
        rules_hlayout.addWidget(self.btn_reset_rules)
        rbox_layout.addWidget(rules_header)

        self.edit_autocorrect_rules = QPlainTextEdit()
        self.edit_autocorrect_rules.setPlainText(DEFAULT_AI_INSTRUCTIONS)
        self.edit_autocorrect_rules.setMaximumHeight(90)
        self.edit_autocorrect_rules.setToolTip("Persistent fine-art retouching guidelines used by Run Gemini Auto-Correction and Batch AI processing")
        self.edit_autocorrect_rules.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1a222d;
                color: #ffffff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 6px;
                font-size: 10px;
                line-height: 1.3;
            }
            QPlainTextEdit:focus {
                border: 1px solid #00d2ff;
                background-color: #1e2937;
            }
        """)
        rbox_layout.addWidget(self.edit_autocorrect_rules)
        layout.addWidget(rules_box)

        # =========================================================================
        # 3. 3D & AI Neural Map Tools
        # =========================================================================
        lbl_3d = QLabel("🧠 3D Neural Maps")
        lbl_3d.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_3d)

        self.btn_calc_all_ai = QPushButton("🚀 Calculate All AI Maps (Depth, Normals, Subject)")
        self.btn_calc_all_ai.setStyleSheet("background-color: #1f3b4d; border: 1px solid #00d2ff; font-weight: bold; padding: 6px; color: #00d2ff; border-radius: 4px;")
        self.btn_calc_all_ai.clicked.connect(self.main_window.trigger_compute_all_ai_maps)
        layout.addWidget(self.btn_calc_all_ai)

        maps_grid = QHBoxLayout()
        self.btn_calc_depth = QPushButton("🌊 Depth Map")
        self.btn_calc_depth.setStyleSheet(btn_style)
        self.btn_calc_depth.clicked.connect(self.main_window.trigger_compute_depth_map)
        maps_grid.addWidget(self.btn_calc_depth)

        self.btn_calc_normals = QPushButton("🌐 Normals")
        self.btn_calc_normals.setStyleSheet(btn_style)
        self.btn_calc_normals.clicked.connect(self.main_window.trigger_compute_normal_map)
        maps_grid.addWidget(self.btn_calc_normals)

        self.btn_calc_subject = QPushButton("👤 Subject")
        self.btn_calc_subject.setStyleSheet(btn_style)
        self.btn_calc_subject.clicked.connect(self.main_window.trigger_compute_subject_mask)
        maps_grid.addWidget(self.btn_calc_subject)
        layout.addLayout(maps_grid)

        # Normals Method & Quality Settings
        form_normals = QFormLayout()
        form_normals.setSpacing(4)
        form_normals.setContentsMargins(0, 2, 0, 2)

        self.combo_normals_method = QComboBox()
        self.combo_normals_method.addItems(["Hybrid (Marigold + Depth)", "Marigold", "From Depth Map (Instant)"])
        self.combo_normals_method.setCurrentIndex(0)
        self.combo_normals_method.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        form_normals.addRow("Normals Method:", self.combo_normals_method)

        self.combo_normals_ensemble = QComboBox()
        self.combo_normals_ensemble.addItems(["1 (Fast)", "2 (Balanced)", "3 (High Quality)", "5 (Maximum Quality)"])
        self.combo_normals_ensemble.setCurrentIndex(0)
        self.combo_normals_ensemble.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        form_normals.addRow("Ensemble Size:", self.combo_normals_ensemble)

        self.combo_normals_scale = QComboBox()
        self.combo_normals_scale.addItems(["1x (Original)", "1/2x (Half)", "1/4x (Quarter)"])
        self.combo_normals_scale.setCurrentIndex(0)
        self.combo_normals_scale.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        form_normals.addRow("Normals Scale:", self.combo_normals_scale)

        layout.addLayout(form_normals)

        # =========================================================================
        # 3.4. AI Noise Reduction (NAFNet In-Memory)
        # =========================================================================
        denoise_box = QGroupBox("🔇 AI Noise Reduction (NAFNet In-Memory)")
        denoise_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00e676;
                font-size: 11px;
                border: 1px solid #1b5e20;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 12px;
                background-color: #0d1a14;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        denoise_layout = QVBoxLayout(denoise_box)
        denoise_layout.setSpacing(6)

        self.btn_denoise_raw = QPushButton("🚀 AI Denoise Current Photo (In-Memory)")
        self.btn_denoise_raw.setStyleSheet("""
            QPushButton {
                background-color: #1b5e20;
                border: 1px solid #00e676;
                border-radius: 4px;
                font-weight: bold;
                padding: 7px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2e7d32;
            }
        """)
        self.btn_denoise_raw.setToolTip("Removes noise using NAFNet neural network directly in RAM (In-Memory) without creating extra files. Works on any opened RAW or raster photo.")
        self.btn_denoise_raw.clicked.connect(lambda: self.main_window.trigger_ai_raw_denoise() if hasattr(self.main_window, 'trigger_ai_raw_denoise') else None)
        denoise_layout.addWidget(self.btn_denoise_raw)

        slider_green_style = """
            QSlider::groove:horizontal {
                border: 1px solid #2a3b2e;
                height: 4px;
                background: #14241a;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #00e676;
                border: 1px solid #00b252;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover {
                background: #69f0ae;
            }
        """

        # Slider: Denoise Strength (0 - 100%, default 100%)
        strength_row = QHBoxLayout()
        lbl_s_title = QLabel("Strength:")
        lbl_s_title.setStyleSheet("color: #ccc; font-size: 11px;")
        lbl_s_title.setToolTip("AI noise reduction blend intensity. 100% provides full noise suppression.")
        strength_row.addWidget(lbl_s_title)
        
        self.slider_denoise_strength = QSlider(Qt.Horizontal)
        self.slider_denoise_strength.setRange(0, 100)
        self.slider_denoise_strength.setValue(100)
        self.slider_denoise_strength.setStyleSheet(slider_green_style)
        self.slider_denoise_strength.setToolTip("AI noise reduction blend intensity. 100% provides full noise suppression.")
        strength_row.addWidget(self.slider_denoise_strength)

        self.lbl_denoise_strength = QLabel("100%")
        self.lbl_denoise_strength.setFixedWidth(32)
        self.lbl_denoise_strength.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_denoise_strength.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        self.slider_denoise_strength.valueChanged.connect(lambda v: self.lbl_denoise_strength.setText(f"{v}%"))
        strength_row.addWidget(self.lbl_denoise_strength)
        denoise_layout.addLayout(strength_row)

        # Slider: Texture / Detail Retention (0 - 100%, default 25%)
        texture_row = QHBoxLayout()
        lbl_t_title = QLabel("Texture / Detail:")
        lbl_t_title.setStyleSheet("color: #ccc; font-size: 11px;")
        lbl_t_title.setToolTip("Restores fine skin micro-texture, pores, eyelashes, and sharp hair strands from the original RAW data without bringing back noise.")
        texture_row.addWidget(lbl_t_title)

        self.slider_denoise_texture = QSlider(Qt.Horizontal)
        self.slider_denoise_texture.setRange(0, 100)
        self.slider_denoise_texture.setValue(25)
        self.slider_denoise_texture.setStyleSheet(slider_green_style)
        self.slider_denoise_texture.setToolTip("Restores fine skin micro-texture, pores, eyelashes, and sharp hair strands from the original RAW data without bringing back noise.")
        texture_row.addWidget(self.slider_denoise_texture)

        self.lbl_denoise_texture = QLabel("25%")
        self.lbl_denoise_texture.setFixedWidth(32)
        self.lbl_denoise_texture.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_denoise_texture.setStyleSheet("color: #00e676; font-weight: bold; font-size: 11px;")
        self.slider_denoise_texture.valueChanged.connect(lambda v: self.lbl_denoise_texture.setText(f"{v}%"))
        texture_row.addWidget(self.lbl_denoise_texture)
        denoise_layout.addLayout(texture_row)

        self.chk_denoise_save_tiff = QCheckBox("💾 Also export clean 16-bit TIFF to disk")
        self.chk_denoise_save_tiff.setChecked(False)
        self.chk_denoise_save_tiff.setToolTip("If unchecked, denoises purely in RAM inside the editor without saving any files to disk.")
        self.chk_denoise_save_tiff.setStyleSheet("QCheckBox { color: #80d8ff; font-size: 10px; font-weight: bold; } QCheckBox::indicator { width: 13px; height: 13px; }")
        denoise_layout.addWidget(self.chk_denoise_save_tiff)

        form_denoise = QFormLayout()
        form_denoise.setSpacing(4)
        form_denoise.setContentsMargins(0, 2, 0, 2)

        self.combo_denoise_model = QComboBox()
        self.combo_denoise_model.addItems([
            "NAFNet Ultra (width64 - High Fidelity)",
            "NAFNet Standard (width32 - Fast)"
        ])
        self.combo_denoise_model.setCurrentIndex(0)
        self.combo_denoise_model.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        form_denoise.addRow("Model:", self.combo_denoise_model)

        self.combo_denoise_tile = QComboBox()
        self.combo_denoise_tile.addItems(["512x512 (Safe / Balanced)", "1024x1024 (Fast / High VRAM)"])
        self.combo_denoise_tile.setCurrentIndex(0)
        self.combo_denoise_tile.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        form_denoise.addRow("Tile Size:", self.combo_denoise_tile)
        denoise_layout.addLayout(form_denoise)

        auto_dn_row = QHBoxLayout()
        auto_dn_row.setSpacing(6)
        self.chk_auto_denoise = QCheckBox("⚡ Auto-Denoise before export when ISO >")
        self.chk_auto_denoise.setChecked(True)
        self.chk_auto_denoise.setStyleSheet("QCheckBox { color: #00e676; font-size: 10px; font-weight: bold; } QCheckBox::indicator { width: 13px; height: 13px; }")
        self.chk_auto_denoise.setToolTip("Automatically executes 'AI Denoise Current Photo (In-Memory)' during automatic AI batch processing / auto-correction before exporting when the photo's ISO is greater than this value.")
        auto_dn_row.addWidget(self.chk_auto_denoise)

        self.spin_auto_denoise_iso = QSpinBox()
        self.spin_auto_denoise_iso.setRange(50, 102400)
        self.spin_auto_denoise_iso.setSingleStep(50)
        self.spin_auto_denoise_iso.setValue(200)
        self.spin_auto_denoise_iso.setFixedWidth(64)
        self.spin_auto_denoise_iso.setToolTip("ISO threshold for auto-denoise (default: 200). Photos with ISO higher than this value will be automatically denoised in RAM before export.")
        self.spin_auto_denoise_iso.setStyleSheet("background-color: #14241a; color: #00e676; font-weight: bold; font-size: 11px; border: 1px solid #1b5e20; border-radius: 4px; padding: 2px;")
        auto_dn_row.addWidget(self.spin_auto_denoise_iso)
        auto_dn_row.addStretch()
        denoise_layout.addLayout(auto_dn_row)

        if hasattr(self, 'chk_batch_auto_denoise'):
            self.chk_auto_denoise.toggled.connect(self.chk_batch_auto_denoise.setChecked)
            self.chk_batch_auto_denoise.toggled.connect(self.chk_auto_denoise.setChecked)
            self.spin_auto_denoise_iso.valueChanged.connect(self.spin_batch_auto_denoise_iso.setValue)
            self.spin_batch_auto_denoise_iso.valueChanged.connect(self.spin_auto_denoise_iso.setValue)

        self.lbl_denoise_status = QLabel("Ready • NAFNet Ultra (width64) In-Memory")
        self.lbl_denoise_status.setStyleSheet("color: #81c784; font-size: 10px; font-style: italic;")
        denoise_layout.addWidget(self.lbl_denoise_status)

        layout.addWidget(denoise_box)

        # =========================================================================
        # 3.5. AI Hardware & GPU Acceleration Settings
        # =========================================================================
        hw_box = QGroupBox("⚡ AI Hardware & GPU Acceleration")
        hw_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                font-size: 11px;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        hw_layout = QVBoxLayout(hw_box)
        hw_layout.setSpacing(6)

        from core.adaptive_depth_anything import detect_device_capabilities
        caps = detect_device_capabilities()
        
        gpu_display_str = caps.gpu_name
        if caps.vram_total_gb > 0:
            gpu_display_str += f" ({caps.vram_total_gb} GB VRAM)"
            
        self.lbl_detected_gpu = QLabel(f"🎮 Device: {gpu_display_str}")
        self.lbl_detected_gpu.setStyleSheet("color: #81c784; font-weight: bold; font-size: 10px;")
        self.lbl_detected_gpu.setWordWrap(True)
        hw_layout.addWidget(self.lbl_detected_gpu)

        form_hw = QFormLayout()
        form_hw.setSpacing(4)
        form_hw.setContentsMargins(0, 2, 0, 2)

        self.combo_hw_mode = QComboBox()
        self.combo_hw_mode.addItems([
            "Auto (Optimal Performance)",
            "NVIDIA CUDA / TensorRT",
            "DirectML (DirectX 12 / AMD / Intel)",
            "CPU Safe Mode (Force CPU)"
        ])
        saved_mode = os.environ.get("AI_HARDWARE_MODE", "Auto")
        mode_idx = 0
        if "CUDA" in saved_mode or "TensorRT" in saved_mode: mode_idx = 1
        elif "DirectML" in saved_mode: mode_idx = 2
        elif "CPU" in saved_mode: mode_idx = 3
        self.combo_hw_mode.setCurrentIndex(mode_idx)
        self.combo_hw_mode.setStyleSheet("background-color: #1a222d; color: #ddd; padding: 3px; font-size: 11px; border: 1px solid #3b4d61; border-radius: 4px;")
        self.combo_hw_mode.currentIndexChanged.connect(self._on_hardware_setting_changed)
        form_hw.addRow("Engine:", self.combo_hw_mode)
        hw_layout.addLayout(form_hw)

        self.chk_disable_gpu = QCheckBox("🛡️ Disable GPU (Safe Mode / CPU Only)")
        self.chk_disable_gpu.setToolTip("Completely disables all GPU & CUDA execution to prevent crashes on unstable drivers or low VRAM.")
        self.chk_disable_gpu.setStyleSheet("QCheckBox { color: #cfd8dc; font-size: 11px; font-weight: bold; } QCheckBox::indicator { width: 14px; height: 14px; }")
        saved_disable_gpu = os.environ.get("AI_DISABLE_GPU", "0") in ["1", "true", "True"]
        self.chk_disable_gpu.setChecked(saved_disable_gpu)
        self.chk_disable_gpu.toggled.connect(self._on_hardware_setting_changed)
        hw_layout.addWidget(self.chk_disable_gpu)

        self.lbl_hw_status = QLabel("⚡ Status: Ready (GPU Accelerated)")
        self.lbl_hw_status.setStyleSheet("color: #00d2ff; font-size: 10px; font-style: italic;")
        hw_layout.addWidget(self.lbl_hw_status)
        layout.addWidget(hw_box)

        self.progress_ai = QProgressBar()
        self.progress_ai.setTextVisible(True)
        self.progress_ai.setStyleSheet("QProgressBar { background: #1a1a1a; border: 1px solid #333; height: 14px; text-align: center; font-size: 10px; } QProgressBar::chunk { background: #00d2ff; }")
        self.progress_ai.setVisible(False)
        layout.addWidget(self.progress_ai)

        # =========================================================================
        # 4. AI Assistant Output & Reasoning Log
        # =========================================================================
        lbl_history = QLabel("🤖 Gemini Thoughts & Activity Log")
        lbl_history.setStyleSheet("font-weight: bold; color: #00d2ff; font-size: 11px; margin-top: 6px;")
        layout.addWidget(lbl_history)

        self.txt_ai_output = QTextEdit()
        self.txt_ai_output.setReadOnly(True)
        self.txt_ai_output.setMinimumHeight(150)
        self.txt_ai_output.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-size: 11px;
                font-family: Consolas, Monaco, monospace;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.txt_ai_output)

        # =========================================================================
        # 5. One-Time AI Action / Command Input (At Bottom)
        # =========================================================================
        one_time_box = QGroupBox("💬 One-Time AI Action / Custom Prompt")
        one_time_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                font-size: 11px;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        ot_layout = QVBoxLayout(one_time_box)
        ot_layout.setSpacing(6)

        lbl_ot_hint = QLabel("One-time action (e.g. focus targeting, tone shift). Clears automatically after execution:")
        lbl_ot_hint.setStyleSheet("color: #90a4ae; font-size: 10px;")
        lbl_ot_hint.setWordWrap(True)
        ot_layout.addWidget(lbl_ot_hint)

        self.edit_one_time_prompt = PromptInputEdit()
        self.edit_one_time_prompt.setPlaceholderText("e.g. 'Focus on the child's face', 'Warm glowing skin tones', 'Blur background bokeh'...")
        self.edit_one_time_prompt.setMaximumHeight(55)
        self.edit_one_time_prompt.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1a222d;
                color: #ffffff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
                line-height: 1.3;
            }
            QPlainTextEdit:focus {
                border: 1px solid #00d2ff;
                background-color: #1e2937;
            }
        """)
        self.edit_one_time_prompt.submit_requested.connect(self.main_window._on_apply_prompt_style)
        ot_layout.addWidget(self.edit_one_time_prompt)

        # Quick preset suggestion chips
        chips_layout = QHBoxLayout()
        chips_layout.setSpacing(4)
        
        presets = [
            ("🎯 Focus on Face", "Focus on the child's face and eyes"),
            ("🌊 Blur Background", "Enable 3D optical bokeh blur on background"),
            ("🍑 Warm Skin", "Make skin tones luminous, warm, and healthy"),
            ("🌟 3D Volume", "Add dimensional 3D chiaroscuro contrast on face"),
            ("📐 Smart Crop", "Crop bottom furniture, eliminate edge clutter, and balance composition"),
        ]
        for btn_lbl, prompt_text in presets:
            btn_chip = QPushButton(btn_lbl)
            btn_chip.setToolTip(f"Insert: '{prompt_text}'")
            btn_chip.setStyleSheet("""
                QPushButton {
                    background-color: #16202c;
                    border: 1px solid #2d3e52;
                    border-radius: 3px;
                    color: #80d8ff;
                    font-size: 10px;
                    padding: 3px 5px;
                }
                QPushButton:hover {
                    background-color: #1e2e40;
                    border-color: #00d2ff;
                    color: #ffffff;
                }
            """)
            btn_chip.clicked.connect(lambda _, p=prompt_text: self._on_chip_clicked(p))
            chips_layout.addWidget(btn_chip)
        ot_layout.addLayout(chips_layout)

        self.btn_apply_prompt = QPushButton("🚀 Send Prompt (Enter)")
        self.btn_apply_prompt.setToolTip("Send one-time action command to Gemini (Enter or Ctrl+Enter). Field clears automatically.")
        self.btn_apply_prompt.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #006699, stop:1 #0099cc);
                border: 1px solid #00d2ff;
                border-radius: 4px;
                font-weight: bold;
                padding: 7px;
                color: #ffffff;
                font-size: 11px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0088cc, stop:1 #00bbee);
                border-color: #ffffff;
            }
        """)
        self.btn_apply_prompt.clicked.connect(self.main_window._on_apply_prompt_style)
        ot_layout.addWidget(self.btn_apply_prompt)

        layout.addWidget(one_time_box)

        layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

    def _on_toggle_key_visibility(self):
        if self.edit_api_key.echoMode() == QLineEdit.EchoMode.Password:
            self.edit_api_key.setEchoMode(QLineEdit.EchoMode.Normal)
            self.btn_toggle_key.setText("🔒")
            self.btn_toggle_key.setToolTip("Hide API Key")
        else:
            self.edit_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.btn_toggle_key.setText("👁")
            self.btn_toggle_key.setToolTip("Show API Key")

    def _on_save_api_key(self):
        key = self.edit_api_key.text().strip()
        if not key or key == "your_api_key_here":
            self.txt_ai_output.append("⚠️ Please paste a valid Google Gemini API Key.")
            return
            
        os.environ["GEMINI_API_KEY"] = key
        
        # Also update root .env file
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_path = os.path.join(root_dir, ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            found = False
            for i, line in enumerate(lines):
                if line.startswith("GEMINI_API_KEY="):
                    lines[i] = f"GEMINI_API_KEY={key}\n"
                    found = True
                    break
            if not found:
                lines.append(f"GEMINI_API_KEY={key}\n")
                
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
                
            self.txt_ai_output.append(f"✅ Google Gemini API Key ({key[:6]}...{key[-4:]}) saved successfully to .env and environment.")
        except Exception as e:
            self.txt_ai_output.append(f"⚠️ Saved to session memory (could not write .env: {e})")

    def get_gemini_config(self) -> dict:
        level_str = "medium"
        if hasattr(self, 'combo_thinking_level'):
            txt = self.combo_thinking_level.currentText()
            if "low" in txt: level_str = "low"
            elif "high" in txt: level_str = "high"
            else: level_str = "medium"

        res_txt = self.combo_ai_res.currentText() if hasattr(self, 'combo_ai_res') else "1024"
        ai_size = 1024
        if "768" in res_txt: ai_size = 768
        elif "512" in res_txt: ai_size = 512
        elif "1024" in res_txt: ai_size = 1024
        elif "1536" in res_txt: ai_size = 1536

        # Legacy budget mapping for compatibility with older callers
        budget_map = {"low": 512, "medium": 1024, "high": 4096}

        return {
            "api_key": self.edit_api_key.text().strip() or os.environ.get("GEMINI_API_KEY", ""),
            "thinking_level": level_str,
            "thinking_budget": budget_map.get(level_str, 1024),
            "max_passes": self.spin_max_passes.value(),
            "ai_image_size": ai_size,
            "generate_multi_crop": self.chk_generate_multi_crop.isChecked() if hasattr(self, 'chk_generate_multi_crop') else True,
            "auto_denoise_enabled": self.is_auto_denoise_enabled(),
            "auto_denoise_iso_threshold": self.get_auto_denoise_iso_threshold()
        }

    def is_auto_denoise_enabled(self) -> bool:
        if hasattr(self, 'chk_batch_auto_denoise'):
            return self.chk_batch_auto_denoise.isChecked()
        if hasattr(self, 'chk_auto_denoise'):
            return self.chk_auto_denoise.isChecked()
        return True

    def is_batch_auto_denoise_enabled(self) -> bool:
        return self.is_auto_denoise_enabled()

    def get_auto_denoise_iso_threshold(self) -> int:
        if hasattr(self, 'spin_batch_auto_denoise_iso'):
            return self.spin_batch_auto_denoise_iso.value()
        if hasattr(self, 'spin_auto_denoise_iso'):
            return self.spin_auto_denoise_iso.value()
        return 200

    def get_batch_auto_denoise_iso_threshold(self) -> int:
        return self.get_auto_denoise_iso_threshold()

    def set_loading(self, active: bool, status_text: str = ""):
        self.btn_auto_correct.setEnabled(not active)
        if hasattr(self, 'btn_batch_auto_correct'):
            self.btn_batch_auto_correct.setEnabled(not active)
        if hasattr(self, 'btn_full_auto'):
            self.btn_full_auto.setEnabled(not active)
        self.btn_apply_prompt.setEnabled(not active)
        self.btn_stop_ai.setVisible(active)
        self.progress_ai.setVisible(active)
        if active:
            self.progress_ai.setRange(0, 0)
            if status_text:
                self.progress_ai.setFormat(status_text)
        else:
            self.progress_ai.setRange(0, 100)
            self.progress_ai.setValue(100)
            self.progress_ai.setFormat("")

    def _on_chip_clicked(self, text: str):
        if hasattr(self, 'edit_one_time_prompt'):
            cur = self.edit_one_time_prompt.toPlainText().strip()
            if cur:
                self.edit_one_time_prompt.setPlainText(f"{cur}, {text}")
            else:
                self.edit_one_time_prompt.setPlainText(text)
            self.edit_one_time_prompt.setFocus()

    def get_autocorrect_rules(self) -> str:
        """Returns the persistent fine-art rules for Auto-Correction & Batch processing."""
        if hasattr(self, 'edit_autocorrect_rules'):
            return self.edit_autocorrect_rules.toPlainText().strip()
        return DEFAULT_AI_INSTRUCTIONS

    def reset_autocorrect_rules(self):
        """Resets the persistent fine-art rules to defaults."""
        if hasattr(self, 'edit_autocorrect_rules'):
            self.edit_autocorrect_rules.setPlainText(DEFAULT_AI_INSTRUCTIONS)

    def get_and_clear_one_time_prompt(self) -> str:
        """Gets the one-time command for Gemini and immediately clears the input field."""
        if hasattr(self, 'edit_one_time_prompt'):
            cmd = self.edit_one_time_prompt.toPlainText().strip()
            self.edit_one_time_prompt.clear()
            return cmd
        return ""

    def get_user_prompt(self) -> str:
        """Returns persistent autocorrect rules (never polluted by one-time commands)."""
        return self.get_autocorrect_rules()

    def reset_user_prompt(self):
        """Resets both autocorrect rules and one-time prompt field."""
        self.reset_autocorrect_rules()
        if hasattr(self, 'edit_one_time_prompt'):
            self.edit_one_time_prompt.clear()

    def _on_hardware_setting_changed(self):
        disable_gpu = self.chk_disable_gpu.isChecked()
        mode_text = self.combo_hw_mode.currentText()
        if "CUDA" in mode_text or "TensorRT" in mode_text:
            mode = "CUDA"
        elif "DirectML" in mode_text:
            mode = "DirectML"
        elif "CPU" in mode_text:
            mode = "CPU"
        else:
            mode = "Auto"
            
        if disable_gpu:
            self.lbl_hw_status.setText("🛡️ Status: Safe Mode (CPU Only)")
            self.lbl_hw_status.setStyleSheet("color: #ffb74d; font-size: 10px; font-style: italic;")
            self.txt_ai_output.append("🛡️ GPU Processing disabled by user (Safe Mode / CPU Only).")
        else:
            self.lbl_hw_status.setText(f"⚡ Status: Hardware Accelerated ({mode})")
            self.lbl_hw_status.setStyleSheet("color: #00d2ff; font-size: 10px; font-style: italic;")
            self.txt_ai_output.append(f"⚡ Hardware acceleration engine set to: {mode}")

        os.environ["AI_HARDWARE_MODE"] = mode
        os.environ["AI_DISABLE_GPU"] = "1" if disable_gpu else "0"
        self._save_hardware_env(mode, disable_gpu)

    def _save_hardware_env(self, mode: str, disable_gpu: bool):
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_path = os.path.join(root_dir, ".env")
            lines = []
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            def set_key(k, v):
                nonlocal lines
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"{k}="):
                        lines[i] = f"{k}={v}\n"
                        found = True
                        break
                if not found:
                    lines.append(f"{k}={v}\n")

            set_key("AI_HARDWARE_MODE", mode)
            set_key("AI_DISABLE_GPU", "1" if disable_gpu else "0")

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            pass

    def get_hardware_mode(self) -> str:
        mode_text = self.combo_hw_mode.currentText()
        if "CUDA" in mode_text or "TensorRT" in mode_text: return "CUDA"
        if "DirectML" in mode_text: return "DirectML"
        if "CPU" in mode_text: return "CPU"
        return "Auto"

    def get_disable_gpu(self) -> bool:
        return self.chk_disable_gpu.isChecked()

    def is_ai_disable_bokeh_enabled(self) -> bool:
        return hasattr(self, 'chk_ai_disable_bokeh') and self.chk_ai_disable_bokeh.isChecked()

    def is_batch_reset_settings_enabled(self) -> bool:
        return hasattr(self, 'chk_batch_reset_settings') and self.chk_batch_reset_settings.isChecked()

    def is_batch_auto_export_enabled(self) -> bool:
        return hasattr(self, 'chk_batch_auto_export') and self.chk_batch_auto_export.isChecked()

    def is_batch_skip_background_enabled(self) -> bool:
        return hasattr(self, 'chk_batch_skip_background') and self.chk_batch_skip_background.isChecked()

    def is_batch_dual_bokeh_export_enabled(self) -> bool:
        return hasattr(self, 'chk_batch_dual_bokeh_export') and self.chk_batch_dual_bokeh_export.isChecked()

    def is_batch_lens_correction_enabled(self) -> bool:
        return hasattr(self, 'chk_batch_lens_correction') and self.chk_batch_lens_correction.isChecked()

    def is_batch_skip_already_processed_enabled(self) -> bool:
        if hasattr(self, 'chk_batch_skip_already_processed'):
            return self.chk_batch_skip_already_processed.isChecked()
        return True

    def get_batch_export_folder(self) -> str:
        if hasattr(self, 'txt_batch_export_folder'):
            return self.txt_batch_export_folder.text().strip()
        return ""

    def _on_browse_batch_export_folder(self):
        from PySide6.QtWidgets import QFileDialog
        curr = self.get_batch_export_folder()
        folder = QFileDialog.getExistingDirectory(self, "Select Batch Export Destination Folder", curr or "")
        if folder:
            self.txt_batch_export_folder.setText(folder)
