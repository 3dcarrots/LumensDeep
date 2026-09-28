import os
from typing import List, Dict, Any, Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QPlainTextEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QRadioButton, QButtonGroup, QGroupBox, QSpinBox,
    QMessageBox, QWidget, QFrame, QCheckBox, QComboBox
)
from PySide6.QtGui import QColor, QFont, QPixmap, QIcon
from PySide6.QtCore import Qt, Signal

from gui.workers.ai_culling_worker import AICullingWorker, DEFAULT_AI_CULLING_CRITERIA
from core.filmstrip_cache import FilmstripCache

class AICullingDialog(QDialog):
    """
    Dialog for Gemini 3.8 Flash AI Batch Photo Culling & Assessment.
    Allows user to customize culling criteria, view live progress, and
    apply ratings, rejection flags, and 3D depth requirements to the Filmstrip.
    """
    culling_completed = Signal(list) # list of evaluated dicts

    def __init__(self, folder_path: str, all_photo_paths: List[str], api_key: str = "", parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.all_photo_paths = list(all_photo_paths)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.worker = None
        self.cache = FilmstripCache.get_instance()
        self.evaluated_records = []

        self.setWindowTitle("✨ Gemini AI Photo Culling & 3D Depth Assessment")
        self.resize(780, 620)
        self.setStyleSheet("""
            QDialog {
                background-color: #141920;
                color: #e6edf3;
            }
            QLabel {
                color: #d0d7de;
            }
            QGroupBox {
                font-weight: bold;
                color: #00d2ff;
                border: 1px solid #2d3846;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: #1a222d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QPlainTextEdit {
                background-color: #10151c;
                color: #ffffff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 6px;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 11px;
            }
            QPlainTextEdit:focus {
                border: 1px solid #00d2ff;
            }
            QTableWidget {
                background-color: #10151c;
                color: #ffffff;
                gridline-color: #242f3d;
                border: 1px solid #2d3846;
                border-radius: 4px;
                font-size: 11px;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #1c2633;
                color: #9ab0c7;
                border: 1px solid #242f3d;
                padding: 4px;
                font-weight: bold;
                font-size: 11px;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Header Banner
        header_layout = QHBoxLayout()
        lbl_title = QLabel("🤖 Gemini 3.8 Flash: Batch Photo Selector & Culler")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d2ff;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch()

        self.lbl_photo_count = QLabel(f"📁 Folder: {os.path.basename(self.folder_path)} ({len(self.all_photo_paths)} photos)")
        self.lbl_photo_count.setStyleSheet("color: #8fa3b7; font-size: 11px;")
        header_layout.addWidget(self.lbl_photo_count)
        layout.addLayout(header_layout)

        # 2. Culling Criteria Group
        criteria_box = QGroupBox("📝 User Preferences & Culling Rules")
        criteria_layout = QVBoxLayout(criteria_box)
        criteria_layout.setContentsMargins(8, 8, 8, 8)
        criteria_layout.setSpacing(6)

        sub_header = QHBoxLayout()
        lbl_desc = QLabel("Define your criteria for star rating, photo rejection (blurry/closed eyes/duplicates), and 3D depth requirement:")
        lbl_desc.setStyleSheet("color: #8fa3b7; font-size: 11px;")
        sub_header.addWidget(lbl_desc)
        sub_header.addStretch()

        btn_reset = QPushButton("↺ Reset Criteria")
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #243142;
                color: #00d2ff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #2e3e54;
            }
        """)
        btn_reset.clicked.connect(self._reset_criteria)
        sub_header.addWidget(btn_reset)
        criteria_layout.addLayout(sub_header)

        self.txt_criteria = QPlainTextEdit()
        self.txt_criteria.setPlainText(DEFAULT_AI_CULLING_CRITERIA)
        self.txt_criteria.setFixedHeight(135)
        criteria_layout.addWidget(self.txt_criteria)

        layout.addWidget(criteria_box)

        # 3. Scope & Options
        options_box = QGroupBox("⚙️ Scope & Two-Phase Inspection")
        options_vlayout = QVBoxLayout(options_box)
        options_vlayout.setContentsMargins(8, 8, 8, 8)
        options_vlayout.setSpacing(6)

        scope_row = QHBoxLayout()
        self.radio_all = QRadioButton(f"All Photos ({len(self.all_photo_paths)})")
        self.radio_all.setChecked(True)
        scope_row.addWidget(self.radio_all)

        # Count unrated and phase 2 unverified
        unrated_count = 0
        unverified_phase2_count = 0
        cached_data = self.cache.get_folder_cache(self.folder_path)
        for p in self.all_photo_paths:
            rec = cached_data.get(os.path.abspath(p))
            if not rec or rec.get('rating', 0) == 0:
                unrated_count += 1
            if rec:
                if (rec.get('rating', 0) >= 3 or not rec.get('is_rejected', 0)) and not rec.get('phase2_verified', 0):
                    unverified_phase2_count += 1
            else:
                unverified_phase2_count += 1

        self.radio_unrated = QRadioButton(f"Unrated Photos ({unrated_count})")
        scope_row.addWidget(self.radio_unrated)

        self.radio_phase2_only = QRadioButton(f"🔬 Phase 2 Only: Unverified ({unverified_phase2_count})")
        self.radio_phase2_only.setToolTip("Run ONLY Phase 2 micro-focus inspection on keeper photos where Phase 2 has not been executed yet.")
        scope_row.addWidget(self.radio_phase2_only)

        self.radio_all.toggled.connect(self._on_scope_changed)
        self.radio_unrated.toggled.connect(self._on_scope_changed)
        self.radio_phase2_only.toggled.connect(self._on_scope_changed)

        scope_row.addStretch()

        scope_row.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro"
        ])
        self.combo_model.setStyleSheet("background-color: #10151c; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 2px;")
        scope_row.addWidget(self.combo_model)

        scope_row.addWidget(QLabel("Thinking Level:"))
        self.combo_thinking = QComboBox()
        self.combo_thinking.addItems([
            "low (Fast Screening)",
            "medium (Balanced)",
            "high (Deep Focus)"
        ])
        self.combo_thinking.setCurrentIndex(0)
        self.combo_thinking.setStyleSheet("background-color: #10151c; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 2px;")
        scope_row.addWidget(self.combo_thinking)
        options_vlayout.addLayout(scope_row)

        self.chk_two_phase = QCheckBox("🔬 Two-Phase Verification: Fast screening at 1024px, then re-inspect keepers with 50% central crop (16x pixel reduction)")
        self.chk_two_phase.setChecked(True)
        self.chk_two_phase.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 11px;")
        self.chk_two_phase.setToolTip("Phase 1 screens all photos at 1024px. Phase 2 re-evaluates top keepers with a 50% scale central crop (outer 25% margins trimmed, 16x fewer pixels, 4x faster) to verify razor-sharp eye focus and micro-details on subjects.")
        options_vlayout.addWidget(self.chk_two_phase)

        layout.addWidget(options_box)

        # 4. Results Table
        self.table_results = QTableWidget(0, 5)
        self.table_results.setHorizontalHeaderLabels(["Photo", "Rating", "Status", "3D Depth", "AI Reasoning / Justification"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table_results)

        # 5. Progress Bar & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #2d3846;
                border-radius: 4px;
                text-align: center;
                background-color: #10151c;
                color: #ffffff;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0078d4, stop:1 #00d2ff);
                border-radius: 3px;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready to assess photos.")
        self.lbl_status.setStyleSheet("color: #8fa3b7; font-size: 11px; font-style: italic;")
        layout.addWidget(self.lbl_status)

        # 6. Action Buttons
        btn_bar = QHBoxLayout()
        btn_bar.setSpacing(8)

        self.btn_start = QPushButton("🚀 Start AI Batch Assessment")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #0096ff;
            }
        """)
        self.btn_start.clicked.connect(self._start_culling)
        btn_bar.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #a82020;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c93030;
            }
        """)
        self.btn_stop.clicked.connect(self._stop_culling)
        self.btn_stop.hide()
        btn_bar.addWidget(self.btn_stop)

        btn_bar.addStretch()

        self.btn_close = QPushButton("✓ Close")
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: #243142;
                color: #ffffff;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2e3e54;
            }
        """)
        self.btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(self.btn_close)

        layout.addLayout(btn_bar)

    def _on_scope_changed(self):
        if self.radio_phase2_only.isChecked():
            self.chk_two_phase.setEnabled(False)
            self.btn_start.setText("🔬 Run Phase 2 Micro-Focus (Center Crop)")
        else:
            self.chk_two_phase.setEnabled(True)
            self.btn_start.setText("🚀 Start AI Batch Assessment")

    def _reset_criteria(self):
        self.txt_criteria.setPlainText(DEFAULT_AI_CULLING_CRITERIA)

    def _start_culling(self):
        if not self.api_key:
            QMessageBox.warning(
                self,
                "API Key Required",
                "Google Gemini API Key is missing. Please enter your key in the AI Assistant dock on the left or in your .env file."
            )
            return

        is_phase2_only = self.radio_phase2_only.isChecked()
        cached_data = self.cache.get_folder_cache(self.folder_path)

        # Determine target paths
        if self.radio_unrated.isChecked():
            target_paths = [
                p for p in self.all_photo_paths
                if not cached_data.get(os.path.abspath(p)) or cached_data.get(os.path.abspath(p)).get('rating', 0) == 0
            ]
        elif is_phase2_only:
            target_paths = [
                p for p in self.all_photo_paths
                if not cached_data.get(os.path.abspath(p)) or (
                    (cached_data.get(os.path.abspath(p)).get('rating', 0) >= 3 or not cached_data.get(os.path.abspath(p)).get('is_rejected', 0)) and
                    not cached_data.get(os.path.abspath(p)).get('phase2_verified', 0)
                )
            ]
        else:
            target_paths = list(self.all_photo_paths)

        if not target_paths:
            QMessageBox.information(self, "No Photos to Assess", "No photos match the selected criteria.")
            return

        self.table_results.setRowCount(0)
        self.evaluated_records.clear()
        self.progress_bar.setRange(0, len(target_paths))
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.btn_start.hide()
        self.btn_stop.show()
        self.txt_criteria.setEnabled(False)
        desc_text = "Phase 2 micro-focus inspection" if is_phase2_only else "AI batch assessment"
        selected_model = self.combo_model.currentText()
        self.lbl_status.setText(f"Starting {desc_text} of {len(target_paths)} photos with {selected_model}...")

        lvl_txt = self.combo_thinking.currentText()
        if "medium" in lvl_txt:
            th_lvl = "medium"
        elif "high" in lvl_txt:
            th_lvl = "high"
        else:
            th_lvl = "low"

        self.worker = AICullingWorker(
            photo_paths=target_paths,
            user_criteria=self.txt_criteria.toPlainText(),
            api_key=self.api_key,
            model_name=selected_model,
            thinking_level=th_lvl,
            enable_two_phase=self.chk_two_phase.isChecked(),
            run_only_phase_2=is_phase2_only,
            parent=self
        )
        self.worker.photo_evaluated.connect(self._on_photo_evaluated)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.finished_success.connect(self._on_finished_success)
        self.worker.error_occurred.connect(self._on_error_occurred)
        self.worker.start()

    def _stop_culling(self):
        if self.worker:
            self.worker.cancel()
            self.worker.wait(100)
        self.lbl_status.setText("⏹ Assessment stopped by user.")
        self.btn_stop.hide()
        self.btn_start.show()
        self.txt_criteria.setEnabled(True)

    def _on_photo_evaluated(self, path: str, rating: int, is_rejected: bool, depth_needed: bool, reason: str):
        norm_path = os.path.abspath(path)
        base_name = os.path.basename(norm_path)
        record = {
            'path': norm_path,
            'rating': rating,
            'is_rejected': is_rejected,
            'depth_needed': depth_needed,
            'ai_notes': reason
        }
        
        # Check if record already exists (e.g. from Phase 1, being updated in Phase 2)
        existing_idx = -1
        for idx, rec in enumerate(self.evaluated_records):
            if os.path.abspath(rec['path']) == norm_path:
                existing_idx = idx
                break

        if existing_idx >= 0:
            self.evaluated_records[existing_idx] = record
            row = existing_idx
        else:
            self.evaluated_records.append(record)
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)

        item_name = QTableWidgetItem(base_name)
        item_rating = QTableWidgetItem(f"{'★' * rating}{'☆' * (5 - rating)}")
        item_rating.setForeground(QColor(255, 205, 0))

        if is_rejected:
            item_status = QTableWidgetItem("❌ REJECT")
            item_status.setForeground(QColor(255, 80, 80))
        elif "[Center 50% Verified]" in reason or "[50% Verified]" in reason:
            item_status = QTableWidgetItem("🔬 Micro-Focus 50%")
            item_status.setForeground(QColor(0, 220, 255))
        else:
            item_status = QTableWidgetItem("✓ Keep")
            item_status.setForeground(QColor(80, 220, 100))

        if depth_needed:
            item_depth = QTableWidgetItem("🧊 3D Depth")
            item_depth.setForeground(QColor(0, 210, 255))
        else:
            item_depth = QTableWidgetItem("—")
            item_depth.setForeground(QColor(120, 120, 120))

        item_reason = QTableWidgetItem(reason)

        self.table_results.setItem(row, 0, item_name)
        self.table_results.setItem(row, 1, item_rating)
        self.table_results.setItem(row, 2, item_status)
        self.table_results.setItem(row, 3, item_depth)
        self.table_results.setItem(row, 4, item_reason)
        self.table_results.scrollToItem(item_name)

        # Update parent filmstrip in real-time
        parent_window = self.parent()
        if hasattr(parent_window, 'dock_filmstrip') and hasattr(parent_window.dock_filmstrip, 'filmstrip_widget'):
            parent_window.dock_filmstrip.filmstrip_widget.set_photo_ai_assessment(
                path, rating, is_rejected, depth_needed, reason
            )

    def _on_progress_updated(self, current: int, total: int, status_text: str):
        if total > 0:
            self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(status_text)

    def _on_finished_success(self, results: list):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.lbl_status.setText(f"✓ Completed assessment of {len(results)} photos.")
        self.btn_stop.hide()
        self.btn_start.show()
        self.txt_criteria.setEnabled(True)
        self.culling_completed.emit(self.evaluated_records)

    def _on_error_occurred(self, err_msg: str):
        self.lbl_status.setText(f"⚠️ Error: {err_msg}")
        self.btn_stop.hide()
        self.btn_start.show()
        self.txt_criteria.setEnabled(True)
        QMessageBox.warning(self, "AI Assessment Error", err_msg)
