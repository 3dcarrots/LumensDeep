import os
import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QGraphicsDropShadowEffect, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon, QColor, QFont

class LumensSplashScreen(QWidget):
    """
    Sleek, high-performance dark-themed splash screen that appears in < 0.15s
    to provide instant visual feedback while neural models and GPU shaders initialize.
    """
    def __init__(self, icon_path: str = ""):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.SplashScreen
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(560, 310)

        # Center on screen
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(
                (geom.width() - self.width()) // 2,
                (geom.height() - self.height()) // 2
            )

        # Main container with rounded card styling and drop shadow
        container = QWidget(self)
        container.setGeometry(10, 10, 540, 290)
        container.setObjectName("splashContainer")
        container.setStyleSheet("""
            QWidget#splashContainer {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f1319, stop:0.5 #141923, stop:1 #0c0f14);
                border: 1px solid #30363d;
                border-radius: 16px;
            }
        """)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(0, 8)
        container.setGraphicsEffect(shadow)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(12)

        # 1. Top row: Logo + Title + Version pill badge
        top_layout = QHBoxLayout()
        top_layout.setSpacing(16)

        # Logo icon
        self.logo_label = QLabel()
        if icon_path and os.path.exists(icon_path):
            pix = QPixmap(icon_path)
            if not pix.isNull():
                self.logo_label.setPixmap(pix.scaled(56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                self.logo_label.setText("📸")
                self.logo_label.setStyleSheet("font-size: 40px;")
        else:
            self.logo_label.setText("📸")
            self.logo_label.setStyleSheet("font-size: 40px;")
        top_layout.addWidget(self.logo_label)

        # Title & Subtitle block
        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)

        title_label = QLabel("Lumens Deep")
        title_label.setStyleSheet("font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px;")
        title_row.addWidget(title_label)

        version_badge = QLabel("v0.1 Alpha")
        version_badge.setStyleSheet("""
            background-color: rgba(31, 111, 235, 0.25);
            color: #58a6ff;
            border: 1px solid rgba(88, 166, 255, 0.4);
            border-radius: 8px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        """)
        title_row.addWidget(version_badge)
        title_row.addStretch()
        text_layout.addLayout(title_row)

        subtitle_label = QLabel("32-Bit Linear RAW Photo Studio & 3D Neural Relighting Suite")
        subtitle_label.setStyleSheet("font-size: 12px; color: #8b949e; font-weight: 400;")
        text_layout.addWidget(subtitle_label)

        top_layout.addLayout(text_layout)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # 2. Features / Accent banner
        accent_label = QLabel("⚡ Hardware-Accelerated OpenGL 3.3+ & AI Neural Engine")
        accent_label.setStyleSheet("""
            color: #38bdf8;
            font-size: 11px;
            font-weight: 500;
            background-color: rgba(56, 189, 248, 0.08);
            border-radius: 6px;
            padding: 6px 10px;
        """)
        layout.addWidget(accent_label)

        layout.addStretch()

        # 3. Status text and progress bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Starting studio...")
        self.status_label.setStyleSheet("font-size: 11px; color: #c9d1d9; font-weight: 500;")
        status_layout.addWidget(self.status_label)

        self.percent_label = QLabel("0%")
        self.percent_label.setStyleSheet("font-size: 11px; color: #8b949e; font-weight: 600;")
        status_layout.addWidget(self.percent_label, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(status_layout)

        # Custom Slim Neon Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(10)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #21262d;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d2ff, stop:0.5 #3a7bd5, stop:1 #7928ca);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

    def set_status(self, message: str, progress: int = -1):
        """Update loading status message and progress percentage, immediately refreshing the UI."""
        self.status_label.setText(message)
        if progress >= 0:
            self.progress_bar.setValue(progress)
            self.percent_label.setText(f"{progress}%")
        QApplication.processEvents()
