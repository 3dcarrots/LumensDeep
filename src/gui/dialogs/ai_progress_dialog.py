"""
Modal AI Calculation Progress Dialog for Lumens Deep.
Provides real-time feedback during AI Depth, Normals, and Subject segmentation tasks,
including stage indicators, tile counts, parallel worker thread stats, and elapsed time.
"""

import time
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QProgressBar, QPushButton, QFrame, QTextEdit, QWidget
)


class AIProgressDialog(QDialog):
    """
    Modal dialog providing live visual feedback during AI map calculations.
    """
    cancelled = Signal()

    def __init__(self, parent=None, task_title: str = "Depth Map"):
        super().__init__(parent)
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(f"Lumens Deep — Computing {task_title}")
        self.setMinimumWidth(520)
        self.setMinimumHeight(380)
        
        # Prevent accidental close via standard X without cancel confirmation
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.start_time = time.time()
        self.task_title = task_title
        self.current_stage = "Initializing"
        self.cur_tiles = 0
        self.total_tiles = 0
        self.worker_threads = 1
        self._is_finished = False

        self._setup_ui()
        self._setup_timer()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 20)
        main_layout.setSpacing(16)

        # --- Header Card ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        self.icon_badge = QLabel("🧊")
        self.icon_badge.setAlignment(Qt.AlignCenter)
        self.icon_badge.setFixedSize(52, 52)
        self.icon_badge.setObjectName("iconBadge")
        header_layout.addWidget(self.icon_badge)

        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(4)

        self.lbl_title = QLabel(f"Computing {self.task_title}...")
        self.lbl_title.setObjectName("dialogTitle")
        header_text_layout.addWidget(self.lbl_title)

        self.lbl_queue_status = QLabel("Task 1 of 1")
        self.lbl_queue_status.setObjectName("dialogSubtitle")
        header_text_layout.addWidget(self.lbl_queue_status)

        header_layout.addLayout(header_text_layout)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # --- Stage & Status Card ---
        stage_frame = QFrame()
        stage_frame.setObjectName("stageCard")
        stage_layout = QVBoxLayout(stage_frame)
        stage_layout.setContentsMargins(16, 14, 16, 14)
        stage_layout.setSpacing(8)

        stage_top_layout = QHBoxLayout()
        self.lbl_stage_badge = QLabel("STAGE: INITIALIZING")
        self.lbl_stage_badge.setObjectName("stageBadge")
        stage_top_layout.addWidget(self.lbl_stage_badge)
        stage_top_layout.addStretch()

        self.lbl_timer = QLabel("⏱️ 00:00")
        self.lbl_timer.setObjectName("timerLabel")
        stage_top_layout.addWidget(self.lbl_timer)
        stage_layout.addLayout(stage_top_layout)

        self.lbl_status_msg = QLabel("Preparing neural network models and verifying dependencies...")
        self.lbl_status_msg.setObjectName("statusMsg")
        self.lbl_status_msg.setWordWrap(True)
        stage_layout.addWidget(self.lbl_status_msg)

        main_layout.addWidget(stage_frame)

        # --- Progress Bar ---
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(6)

        progress_top_layout = QHBoxLayout()
        self.lbl_progress_percent = QLabel("0%")
        self.lbl_progress_percent.setObjectName("percentLabel")
        progress_top_layout.addWidget(self.lbl_progress_percent)
        progress_top_layout.addStretch()

        self.lbl_tile_summary = QLabel("Initializing pipeline...")
        self.lbl_tile_summary.setObjectName("tileSummaryLabel")
        progress_top_layout.addWidget(self.lbl_tile_summary)
        progress_layout.addLayout(progress_top_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate initially
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setObjectName("neonProgressBar")
        progress_layout.addWidget(self.progress_bar)

        main_layout.addLayout(progress_layout)

        # --- Stats Telemetry Grid ---
        stats_frame = QFrame()
        stats_frame.setObjectName("statsFrame")
        stats_layout = QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(16)

        # Tile Stat
        tile_stat_layout = QVBoxLayout()
        tile_stat_layout.setSpacing(2)
        lbl_tile_stat_hdr = QLabel("TILES PROCESSED")
        lbl_tile_stat_hdr.setObjectName("statHeader")
        self.lbl_stat_tiles = QLabel("— / —")
        self.lbl_stat_tiles.setObjectName("statValue")
        tile_stat_layout.addWidget(lbl_tile_stat_hdr)
        tile_stat_layout.addWidget(self.lbl_stat_tiles)
        stats_layout.addLayout(tile_stat_layout)

        # Workers Stat
        worker_stat_layout = QVBoxLayout()
        worker_stat_layout.setSpacing(2)
        lbl_worker_stat_hdr = QLabel("PARALLEL WORKERS")
        lbl_worker_stat_hdr.setObjectName("statHeader")
        self.lbl_stat_workers = QLabel("Auto (Multi-threaded)")
        self.lbl_stat_workers.setObjectName("statValue")
        worker_stat_layout.addWidget(lbl_worker_stat_hdr)
        worker_stat_layout.addWidget(self.lbl_stat_workers)
        stats_layout.addLayout(worker_stat_layout)

        # Status Stat
        status_stat_layout = QVBoxLayout()
        status_stat_layout.setSpacing(2)
        lbl_status_stat_hdr = QLabel("EXECUTION")
        lbl_status_stat_hdr.setObjectName("statHeader")
        self.lbl_stat_exec = QLabel("Active (GPU/CPU)")
        self.lbl_stat_exec.setObjectName("statValue")
        status_stat_layout.addWidget(lbl_status_stat_hdr)
        status_stat_layout.addWidget(self.lbl_stat_exec)
        stats_layout.addLayout(status_stat_layout)

        main_layout.addWidget(stats_frame)

        # --- Live Log Terminal ---
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(80)
        self.txt_log.setObjectName("logTerminal")
        main_layout.addWidget(self.txt_log)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel Computation")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        btn_layout.addWidget(self.btn_cancel)

        main_layout.addLayout(btn_layout)

    def _setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_timer)
        self.timer.start(1000)

    def _update_timer(self):
        if self._is_finished:
            return
        elapsed = int(time.time() - self.start_time)
        mins = elapsed // 60
        secs = elapsed % 60
        self.lbl_timer.setText(f"⏱️ {mins:02d}:{secs:02d}")

    def set_task(self, map_type: str, task_index: int = 1, total_tasks: int = 1):
        """Set the active task type and queue status."""
        self.task_title = map_type
        self.setWindowTitle(f"Lumens Deep — Computing {map_type}")
        self.lbl_title.setText(f"Computing {map_type}...")
        
        if total_tasks > 1:
            self.lbl_queue_status.setText(f"Task {task_index} of {total_tasks}: {map_type}")
        else:
            self.lbl_queue_status.setText("Single AI Task")

        # Update Icon Badge
        if "Depth" in map_type:
            self.icon_badge.setText("🧊")
        elif "Normal" in map_type:
            self.icon_badge.setText("🧭")
        elif "Subject" in map_type or "Background" in map_type:
            self.icon_badge.setText("✂️")
        else:
            self.icon_badge.setText("✨")

        self.progress_bar.setRange(0, 0)
        self.lbl_progress_percent.setText("0%")
        self.lbl_tile_summary.setText("Starting task...")
        self.lbl_stat_tiles.setText("— / —")

    def update_progress_msg(self, msg: str):
        """Update live status text and log."""
        self.lbl_status_msg.setText(msg)
        self.txt_log.append(f"• {msg}")
        
        # Auto-detect stage keywords from message
        lower_msg = msg.lower()
        if "installing" in lower_msg or "dependencies" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: DEPENDENCY CHECK")
        elif "loading" in lower_msg or "model" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: MODEL INITIALIZATION")
        elif "intermediate" in lower_msg or "hierarchical" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: HIERARCHICAL BASE PASS")
        elif "global" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: GLOBAL MACRO PASS")
        elif "tiling" in lower_msg or "patchfusion" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: LAPLACIAN PATCHFUSION")
        elif "guided filter" in lower_msg or "vit" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: GUIDED FILTERING")
        elif "calibrat" in lower_msg or "scale" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: SURFACE CALIBRATION")
        elif "removal" in lower_msg or "matting" in lower_msg:
            self.lbl_stage_badge.setText("STAGE: SUBJECT MATTING")

    def update_tile_progress(self, cur: int, total: int, stage_desc: str = "", workers: int = 1):
        """Update determinate progress and tile count stats."""
        self.cur_tiles = cur
        self.total_tiles = total
        self.worker_threads = workers

        if stage_desc:
            self.lbl_stage_badge.setText(f"STAGE: {stage_desc.upper()}")

        if workers > 1:
            self.lbl_stat_workers.setText(f"{workers} threads (Parallel)")
        else:
            self.lbl_stat_workers.setText("1 thread (Sequential)")

        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(cur)
            pct = int((cur / total) * 100)
            self.lbl_progress_percent.setText(f"{pct}%")
            self.lbl_tile_summary.setText(f"Tile {cur} of {total} completed")
            self.lbl_stat_tiles.setText(f"{cur} / {total} ({pct}%)")
        else:
            self.progress_bar.setRange(0, 0)
            self.lbl_progress_percent.setText("...")
            self.lbl_tile_summary.setText("Processing...")

    def set_finished(self, success: bool, message: str = ""):
        """Mark task as finished."""
        self._is_finished = True
        self.timer.stop()
        if success:
            self.lbl_stage_badge.setText("STAGE: COMPLETED ✅")
            self.lbl_stage_badge.setStyleSheet("background: rgba(0, 230, 118, 0.2); color: #00e676; border-color: #00e676;")
            self.lbl_status_msg.setText(message or "AI calculation completed successfully!")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.lbl_progress_percent.setText("100%")
            self.lbl_stat_exec.setText("Completed")
            self.btn_cancel.setText("Close")
            self.btn_cancel.setStyleSheet("background: #00d2ff; color: #080c14; font-weight: 700;")
            
            # Automatically close after brief delay
            QTimer.singleShot(600, self.accept)
        else:
            self.lbl_stage_badge.setText("STAGE: ERROR ❌")
            self.lbl_stage_badge.setStyleSheet("background: rgba(255, 82, 82, 0.2); color: #ff5252; border-color: #ff5252;")
            self.lbl_status_msg.setText(f"Error: {message}")
            self.lbl_stat_exec.setText("Failed")
            self.btn_cancel.setText("Close")

    def _on_cancel_clicked(self):
        if self._is_finished:
            self.accept()
        else:
            self.cancelled.emit()
            self.reject()

    def closeEvent(self, event):
        if not self._is_finished:
            self.cancelled.emit()
        event.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 12px;
            }
            #iconBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #161b22, stop:1 #21262d);
                border: 1px solid #388bfd;
                border-radius: 12px;
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
            #stageCard {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            #stageBadge {
                background: rgba(0, 210, 255, 0.12);
                color: #00d2ff;
                border: 1px solid rgba(0, 210, 255, 0.4);
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            #timerLabel {
                font-size: 12px;
                color: #c9d1d9;
                font-weight: 600;
                font-family: monospace;
            }
            #statusMsg {
                font-size: 13px;
                color: #e6edf3;
                line-height: 1.4;
            }
            #percentLabel {
                font-size: 14px;
                font-weight: 700;
                color: #00d2ff;
            }
            #tileSummaryLabel {
                font-size: 12px;
                color: #8b949e;
            }
            #neonProgressBar {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            #neonProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2ff, stop:1 #7928ca);
                border-radius: 5px;
            }
            #statsFrame {
                background-color: #11161d;
                border: 1px solid #21262d;
                border-radius: 8px;
            }
            #statHeader {
                font-size: 10px;
                font-weight: 700;
                color: #8b949e;
                letter-spacing: 0.5px;
            }
            #statValue {
                font-size: 12px;
                font-weight: 600;
                color: #ffffff;
            }
            #logTerminal {
                background-color: #090d13;
                border: 1px solid #21262d;
                border-radius: 6px;
                color: #8b949e;
                font-family: Consolas, Monaco, monospace;
                font-size: 11px;
                padding: 6px;
            }
            #btnCancel {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                color: #c9d1d9;
                padding: 8px 18px;
                font-size: 13px;
                font-weight: 600;
            }
            #btnCancel:hover {
                background-color: #b91c1c;
                border-color: #ef4444;
                color: #ffffff;
            }
        """)
