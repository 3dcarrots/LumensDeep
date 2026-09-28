"""
Modern High-Resolution 16-Bit Export Dialog for Lumens Deep.
Provides options for format (TIFF, PNG, JPEG, WebP), bit depth (16-bit/8-bit/32-bit),
color space, scaling, and sidecar options.
"""

import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSlider, 
    QCheckBox, QFileDialog, QFrame, QGroupBox, QMessageBox
)


class ExportDialog(QDialog):
    """
    Modal dialog providing advanced export settings for full-resolution sensor captures.
    """
    def __init__(self, parent=None, source_image_path: str = "", image_width: int = 0, image_height: int = 0):
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("Lumens Deep — Export High-Resolution Image")
        self.setMinimumWidth(680)
        self.setMinimumHeight(560)
        self.resize(720, 620)

        self.source_image_path = source_image_path
        self.orig_w = image_width
        self.orig_h = image_height
        self.main_window = parent
        
        self.export_settings = None

        self._init_ui()
        self._apply_styles()
        self._update_telemetry()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # --- Header Card ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(14)

        icon_badge = QLabel("💾")
        icon_badge.setAlignment(Qt.AlignCenter)
        icon_badge.setFixedSize(52, 52)
        icon_badge.setObjectName("exportIconBadge")
        header_layout.addWidget(icon_badge)

        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(3)

        lbl_title = QLabel("Export High-Resolution Image")
        lbl_title.setObjectName("dialogTitle")
        header_text_layout.addWidget(lbl_title)

        megapixels = (self.orig_w * self.orig_h) / 1_000_000.0 if (self.orig_w and self.orig_h) else 0.0
        file_name = os.path.basename(self.source_image_path) if self.source_image_path else "Untitled"
        lbl_subtitle = QLabel(f"{file_name} • {self.orig_w} × {self.orig_h} px ({megapixels:.1f} MP)")
        lbl_subtitle.setObjectName("dialogSubtitle")
        header_text_layout.addWidget(lbl_subtitle)

        header_layout.addLayout(header_text_layout)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # --- Section: Output File Path ---
        path_box = QFrame()
        path_box.setObjectName("sectionCard")
        path_layout = QVBoxLayout(path_box)
        path_layout.setContentsMargins(16, 14, 16, 14)
        path_layout.setSpacing(8)

        lbl_dest = QLabel("OUTPUT DESTINATION:")
        lbl_dest.setObjectName("sectionHeader")
        path_layout.addWidget(lbl_dest)

        dest_row = QHBoxLayout()
        dest_row.setSpacing(10)

        default_dir = os.path.dirname(self.source_image_path) if self.source_image_path else ""
        default_stem = os.path.splitext(os.path.basename(self.source_image_path))[0] if self.source_image_path else "export"
        export_subfolder = os.path.join(default_dir, "LDExport") if default_dir else "LDExport"
        default_out = os.path.join(export_subfolder, f"{default_stem}.jpg")

        self.txt_output_path = QLineEdit(default_out)
        self.txt_output_path.setObjectName("pathInput")
        dest_row.addWidget(self.txt_output_path, stretch=1)

        btn_browse = QPushButton("📁 Browse...")
        btn_browse.setObjectName("btnSecondary")
        btn_browse.clicked.connect(self._on_browse_clicked)
        dest_row.addWidget(btn_browse)

        path_layout.addLayout(dest_row)
        main_layout.addWidget(path_box)

        # --- Section: Format, Bit Depth & Color Space ---
        settings_box = QFrame()
        settings_box.setObjectName("sectionCard")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(16, 14, 16, 14)
        settings_layout.setSpacing(12)

        lbl_fmt_hdr = QLabel("IMAGE ENCODING & COLOR SPACE:")
        lbl_fmt_hdr.setObjectName("sectionHeader")
        settings_layout.addWidget(lbl_fmt_hdr)

        # Row 1: Format & Bit Depth
        row1 = QHBoxLayout()
        row1.setSpacing(14)

        # Format (Default: JPEG)
        fmt_layout = QVBoxLayout()
        fmt_layout.setSpacing(4)
        fmt_layout.addWidget(QLabel("File Format:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["JPEG (*.jpg)", "TIFF (*.tif, *.tiff)", "PNG (*.png)", "WebP (*.webp)"])
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        fmt_layout.addWidget(self.combo_format)
        row1.addLayout(fmt_layout, stretch=1)

        # Bit Depth
        depth_layout = QVBoxLayout()
        depth_layout.setSpacing(4)
        depth_layout.addWidget(QLabel("Bit Depth:"))
        self.combo_depth = QComboBox()
        self.combo_depth.addItems(["16-bit (Full Precision)", "8-bit (Standard Web)", "32-bit Float (RAW HDR)"])
        self.combo_depth.setCurrentIndex(1) # 8-bit default for JPEG
        self.combo_depth.setEnabled(False) # JPEG is 8-bit
        self.combo_depth.currentIndexChanged.connect(self._update_telemetry)
        depth_layout.addWidget(self.combo_depth)
        row1.addLayout(depth_layout, stretch=1)

        settings_layout.addLayout(row1)

        # Row 2: Color Space & Quality Slider
        row2 = QHBoxLayout()
        row2.setSpacing(14)

        # Color Space
        cs_layout = QVBoxLayout()
        cs_layout.setSpacing(4)
        cs_layout.addWidget(QLabel("Color Profile:"))
        self.combo_colorspace = QComboBox()
        self.combo_colorspace.addItems(["sRGB (Standard)", "Display P3 (Wide Gamut)", "Adobe RGB (1998)", "Rec. 2020 (Cinema HDR)"])
        cs_layout.addWidget(self.combo_colorspace)
        row2.addLayout(cs_layout, stretch=1)

        # Quality (for JPG/WebP)
        self.quality_container = QVBoxLayout()
        self.quality_container.setSpacing(4)
        q_label_row = QHBoxLayout()
        q_label_row.addWidget(QLabel("JPEG Quality:"))
        self.lbl_quality_val = QLabel("85%")
        self.lbl_quality_val.setStyleSheet("color: #00d2ff; font-weight: bold;")
        q_label_row.addWidget(self.lbl_quality_val)
        self.quality_container.addLayout(q_label_row)

        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(50, 100)
        self.slider_quality.setValue(85)
        self.slider_quality.setEnabled(True)
        self.slider_quality.valueChanged.connect(lambda v: self.lbl_quality_val.setText(f"{v}%"))
        self.quality_container.addWidget(self.slider_quality)
        row2.addLayout(self.quality_container, stretch=1)

        settings_layout.addLayout(row2)
        main_layout.addWidget(settings_box)

        # --- Section: Resolution Scale & Options ---
        opt_box = QFrame()
        opt_box.setObjectName("sectionCard")
        opt_layout = QVBoxLayout(opt_box)
        opt_layout.setContentsMargins(16, 14, 16, 14)
        opt_layout.setSpacing(10)

        lbl_opt_hdr = QLabel("SCALE & METADATA OPTIONS:")
        lbl_opt_hdr.setObjectName("sectionHeader")
        opt_layout.addWidget(lbl_opt_hdr)

        opt_row = QHBoxLayout()
        opt_row.setSpacing(14)

        scale_layout = QVBoxLayout()
        scale_layout.setSpacing(4)
        scale_layout.addWidget(QLabel("Output Resolution:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["100% Original Sensor Size", "50% Half Size", "200% High-res (Lanczos)"])
        self.combo_scale.currentIndexChanged.connect(self._update_telemetry)
        scale_layout.addWidget(self.combo_scale)
        opt_row.addLayout(scale_layout, stretch=1)

        cb_layout = QVBoxLayout()
        cb_layout.setSpacing(6)
        self.chk_save_sidecar = QCheckBox("Save .lumendeep Sidecar alongside file")
        self.chk_save_sidecar.setChecked(True)
        cb_layout.addWidget(self.chk_save_sidecar)

        self.chk_embed_exif = QCheckBox("Preserve Camera EXIF Metadata")
        self.chk_embed_exif.setChecked(True)
        cb_layout.addWidget(self.chk_embed_exif)

        self.chk_dual_bokeh = QCheckBox("✨ Dual Export for Bokeh (_bokeh + clean)")
        if hasattr(self.main_window, 'dock_ai') and hasattr(self.main_window.dock_ai, 'is_batch_dual_bokeh_export_enabled'):
            self.chk_dual_bokeh.setChecked(self.main_window.dock_ai.is_batch_dual_bokeh_export_enabled())
        else:
            self.chk_dual_bokeh.setChecked(False)
        self.chk_dual_bokeh.setToolTip("When Bokeh is enabled on this photo, exports two versions:\n1. With Bokeh ('*_bokeh.jpg')\n2. Clean sharp version ('*.jpg')")
        cb_layout.addWidget(self.chk_dual_bokeh)

        # Multi-crop presets export checkbox
        self.chk_export_all_crops = QCheckBox("✂️ Export all Crop Presets for this photo")
        presets_count = len(getattr(self.main_window, 'crop_presets', [])) if hasattr(self.main_window, 'crop_presets') else 0
        self.chk_export_all_crops.setChecked(presets_count > 1)
        self.chk_export_all_crops.setToolTip(f"When multiple Crop Presets exist ({presets_count} configured), exports each cropped variant as a separate high-resolution file (e.g. *_crop_portrait_4x5.jpg, *_crop_square_1x1.jpg).")
        cb_layout.addWidget(self.chk_export_all_crops)

        opt_row.addLayout(cb_layout, stretch=1)

        opt_layout.addLayout(opt_row)
        main_layout.addWidget(opt_box)

        # --- Telemetry Bar ---
        telemetry_layout = QHBoxLayout()
        self.lbl_telemetry = QLabel("Estimated Output: —")
        self.lbl_telemetry.setObjectName("telemetryLabel")
        telemetry_layout.addWidget(self.lbl_telemetry)
        telemetry_layout.addStretch()
        main_layout.addLayout(telemetry_layout)

        # --- Bottom Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_export = QPushButton("🚀 Export Image")
        self.btn_export.setObjectName("btnPrimary")
        self.btn_export.setDefault(True)
        self.btn_export.clicked.connect(self._on_export_clicked)
        btn_layout.addWidget(self.btn_export)

        main_layout.addLayout(btn_layout)

    def _on_browse_clicked(self):
        curr = self.txt_output_path.text()
        initial_dir = os.path.dirname(curr) if curr else ""
        if initial_dir and not os.path.exists(initial_dir):
            os.makedirs(initial_dir, exist_ok=True)
        ext = self._get_selected_extension()
        filter_str = f"Image Files (*{ext});;All Files (*)"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Export File", curr, filter_str)
        if file_path:
            self.txt_output_path.setText(file_path)
            self._update_telemetry()

    def _on_format_changed(self, idx):
        fmt = self.combo_format.currentText()
        curr_path = self.txt_output_path.text()
        base, _ = os.path.splitext(curr_path)
        
        if "TIFF" in fmt:
            new_ext = ".tif"
            self.combo_depth.setEnabled(True)
            self.combo_depth.setCurrentIndex(0) # 16-bit
            self.slider_quality.setEnabled(False)
        elif "PNG" in fmt:
            new_ext = ".png"
            self.combo_depth.setEnabled(True)
            self.combo_depth.setCurrentIndex(0) # 16-bit
            self.slider_quality.setEnabled(False)
        elif "JPEG" in fmt:
            new_ext = ".jpg"
            self.combo_depth.setCurrentIndex(1) # 8-bit
            self.combo_depth.setEnabled(False)
            self.slider_quality.setEnabled(True)
        elif "WebP" in fmt:
            new_ext = ".webp"
            self.combo_depth.setCurrentIndex(1) # 8-bit
            self.combo_depth.setEnabled(False)
            self.slider_quality.setEnabled(True)
        else:
            new_ext = ".jpg"

        if base:
            self.txt_output_path.setText(base + new_ext)
        self._update_telemetry()

    def _get_selected_extension(self) -> str:
        fmt = self.combo_format.currentText()
        if "TIFF" in fmt: return ".tif"
        if "PNG" in fmt: return ".png"
        if "JPEG" in fmt: return ".jpg"
        if "WebP" in fmt: return ".webp"
        return ".jpg"

    def _update_telemetry(self):
        scale_text = self.combo_scale.currentText()
        scale = 1.0
        if "50%" in scale_text: scale = 0.5
        elif "200%" in scale_text: scale = 2.0
        
        w = int(self.orig_w * scale)
        h = int(self.orig_h * scale)
        mp = (w * h) / 1_000_000.0
        
        depth_text = self.combo_depth.currentText()
        fmt_text = self.combo_format.currentText()
        
        if "16-bit" in depth_text:
            depth_label = "16-bit Uncompressed"
            approx_mb = (w * h * 6) / (1024 * 1024)
        elif "32-bit" in depth_text:
            depth_label = "32-bit Float"
            approx_mb = (w * h * 12) / (1024 * 1024)
        else:
            depth_label = "8-bit Standard"
            approx_mb = (w * h * 3) / (1024 * 1024) if "PNG" in fmt_text else (w * h * 3 * 0.15) / (1024 * 1024)

        self.lbl_telemetry.setText(f"Output: {w} × {h} px ({mp:.1f} MP) • {depth_label} • Approx. size: ~{approx_mb:.1f} MB")

    def _on_export_clicked(self):
        file_path = self.txt_output_path.text().strip()
        if not file_path:
            QMessageBox.warning(self, "Invalid Destination", "Please specify a valid file path for export.")
            return

        out_dir = os.path.dirname(os.path.abspath(file_path))
        if out_dir and not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Directory Error", f"Failed to create export directory:\n{e}")
                return

        depth_str = self.combo_depth.currentText()
        bit_depth = 16
        if "8-bit" in depth_str: bit_depth = 8
        elif "32-bit" in depth_str: bit_depth = 32

        scale_text = self.combo_scale.currentText()
        scale = 1.0
        if "50%" in scale_text: scale = 0.5
        elif "200%" in scale_text: scale = 2.0

        self.export_settings = {
            'file_path': file_path,
            'bit_depth': bit_depth,
            'color_space': self.combo_colorspace.currentText(),
            'quality': self.slider_quality.value(),
            'scale': scale,
            'save_sidecar': self.chk_save_sidecar.isChecked(),
            'embed_exif': self.chk_embed_exif.isChecked(),
            'dual_bokeh': self.chk_dual_bokeh.isChecked(),
            'export_all_crops': self.chk_export_all_crops.isChecked() if hasattr(self, 'chk_export_all_crops') else False,
        }
        self.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0e1218;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            #exportIconBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a2230, stop:1 #121822);
                border: 1px solid #00d2ff;
                border-radius: 10px;
                font-size: 26px;
            }
            #dialogTitle {
                font-size: 18px;
                font-weight: 700;
                color: #ffffff;
            }
            #dialogSubtitle {
                font-size: 12px;
                color: #8b949e;
            }
            #sectionCard {
                background-color: #151b23;
                border: 1px solid #28303c;
                border-radius: 8px;
            }
            #sectionHeader {
                color: #00d2ff;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.6px;
            }
            #pathInput {
                background-color: #0b0e14;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 7px 10px;
                color: #e6edf3;
                font-size: 12px;
                min-height: 20px;
                font-family: monospace;
            }
            #pathInput:focus {
                border-color: #00d2ff;
            }
            QComboBox {
                background-color: #0b0e14;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 5px 10px;
                min-height: 24px;
                color: #e6edf3;
                font-size: 12px;
            }
            QComboBox:focus {
                border-color: #00d2ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background-color: #151b23;
                border: 1px solid #30363d;
                selection-background-color: #005577;
                color: #e6edf3;
                padding: 4px;
            }
            QComboBox QAbstractItemView::item {
                min-height: 24px;
                padding: 3px 6px;
            }
            QCheckBox {
                color: #c9d1d9;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #444c56;
                border-radius: 3px;
                background-color: #0b0e14;
            }
            QCheckBox::indicator:checked {
                background-color: #00d2ff;
                border-color: #00d2ff;
            }
            #telemetryLabel {
                color: #8b949e;
                font-size: 11px;
                font-style: italic;
            }
            #btnPrimary {
                background-color: #00d2ff;
                color: #05080e;
                font-weight: 700;
                border: none;
                border-radius: 6px;
                padding: 9px 20px;
                font-size: 13px;
            }
            #btnPrimary:hover {
                background-color: #33dcff;
            }
            #btnPrimary:pressed {
                background-color: #00a8cc;
            }
            #btnSecondary {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                color: #c9d1d9;
                padding: 6px 14px;
                font-size: 12px;
            }
            #btnSecondary:hover {
                background-color: #30363d;
                color: #fff;
            }
            #btnCancel {
                background-color: transparent;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #8b949e;
                padding: 9px 18px;
                font-size: 13px;
            }
            #btnCancel:hover {
                background-color: #21262d;
                color: #c9d1d9;
            }
        """)


class ExportProgressDialog(QDialog):
    """
    Sleek, dark-themed modal progress dialog displaying live GPU bokeh passes
    and asynchronous file encoding status for High-Resolution exports.
    """
    def __init__(self, parent=None, destination_path: str = "", dimensions: str = ""):
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle("Exporting Image — Lumens Deep")
        self.setMinimumWidth(520)
        self.resize(560, 230)
        self._is_cancelled = False
        self._init_ui(destination_path, dimensions)
        self._apply_styles()

    def _init_ui(self, dest_path: str, dimensions: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # Header with icon badge and details
        h_layout = QHBoxLayout()
        h_layout.setSpacing(14)

        icon_badge = QLabel("💾")
        icon_badge.setAlignment(Qt.AlignCenter)
        icon_badge.setFixedSize(48, 48)
        icon_badge.setObjectName("progressIconBadge")
        h_layout.addWidget(icon_badge)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        lbl_title = QLabel("Exporting High-Resolution Image")
        lbl_title.setObjectName("progressTitle")
        text_layout.addWidget(lbl_title)

        filename = os.path.basename(dest_path) if dest_path else "Untitled"
        self.lbl_subtitle = QLabel(f"{filename} • {dimensions}")
        self.lbl_subtitle.setObjectName("progressSubtitle")
        text_layout.addWidget(self.lbl_subtitle)

        h_layout.addLayout(text_layout)
        h_layout.addStretch()
        layout.addLayout(h_layout)

        # Progress bar
        from PySide6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setObjectName("exportProgressBar")
        layout.addWidget(self.progress_bar)

        # Status text
        self.lbl_status = QLabel("Initializing GPU rendering pipeline...")
        self.lbl_status.setObjectName("progressStatus")
        layout.addWidget(self.lbl_status)

        # Action Buttons (Cancel)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_cancel = QPushButton("Cancel Export")
        self.btn_cancel.setObjectName("btnCancelExport")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def _on_cancel_clicked(self):
        self._is_cancelled = True
        self.lbl_status.setText("Aborting export...")
        self.btn_cancel.setEnabled(False)

    def set_progress(self, percent: int, status_text: str = ""):
        self.progress_bar.setValue(int(percent))
        if status_text:
            self.lbl_status.setText(status_text)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0e1218;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            #progressIconBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a2230, stop:1 #121822);
                border: 1px solid #00d2ff;
                border-radius: 10px;
                font-size: 24px;
            }
            #progressTitle {
                font-size: 15px;
                font-weight: 700;
                color: #ffffff;
            }
            #progressSubtitle {
                font-size: 11px;
                color: #8b949e;
            }
            #exportProgressBar {
                background-color: #0b0e14;
                border: 1px solid #30363d;
                border-radius: 6px;
                height: 20px;
                text-align: center;
                color: #ffffff;
                font-size: 11px;
                font-weight: bold;
            }
            #exportProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0088cc, stop:1 #00d2ff);
                border-radius: 5px;
            }
            #progressStatus {
                color: #00d2ff;
                font-size: 12px;
                font-weight: 500;
            }
            #btnCancelExport {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 5px;
                color: #c9d1d9;
                padding: 6px 16px;
                font-size: 12px;
            }
            #btnCancelExport:hover {
                background-color: #30363d;
                color: #ffffff;
            }
        """)


from PySide6.QtCore import QThread, Signal
import numpy as np
from core.image_io import export_image_from_linear

class ExportWorkerThread(QThread):
    """
    Background worker thread that runs image scaling and linear RGB disk encoding
    asynchronously so the application never freezes or hangs.
    """
    progress = Signal(int, str)
    finished = Signal(str, int, int) # file_path, width, height
    error = Signal(str)

    def __init__(self, file_path: str, pixels: np.ndarray, bit_depth: int = 16, quality: int = 85, scale: float = 1.0, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.pixels = pixels
        self.bit_depth = bit_depth
        self.quality = quality
        self.scale = scale
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            pixels = self.pixels
            if self._is_cancelled:
                return

            if self.scale != 1.0:
                self.progress.emit(82, f"Scaling image to {int(self.scale * 100)}% (Lanczos-4)...")
                import cv2
                new_w = max(1, int(pixels.shape[1] * self.scale))
                new_h = max(1, int(pixels.shape[0] * self.scale))
                pixels = cv2.resize(pixels, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            if self._is_cancelled:
                return

            ext = os.path.splitext(self.file_path)[1].upper()
            self.progress.emit(88, f"Encoding {self.bit_depth}-bit {ext.replace('.', '')} to disk...")

            export_image_from_linear(
                self.file_path, pixels,
                apply_gamma=True,
                bit_depth=self.bit_depth,
                quality=self.quality
            )

            if self._is_cancelled:
                return

            self.progress.emit(100, "Export completed successfully!")
            self.finished.emit(self.file_path, pixels.shape[1], pixels.shape[0])
        except Exception as e:
            self.error.emit(str(e))
