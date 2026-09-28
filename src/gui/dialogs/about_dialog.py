import os
import sys
import webbrowser
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextBrowser, QFrame
)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QIcon, QDesktopServices

class AboutDialog(QDialog):
    """
    Comprehensive About & License dialog for Lumens Deep.
    Displays application version, developer attribution, core architecture summary,
    and a complete list of open-source third-party libraries and their licenses.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Lumens Deep")
        self.setFixedSize(760, 620)
        self.setStyleSheet("""
            QDialog {
                background-color: #0d1117;
                color: #e6edf3;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                background: #161b22;
                border-radius: 6px;
                top: -1px;
            }
            QTabBar::tab {
                background: #0d1117;
                color: #8b949e;
                border: 1px solid #30363d;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: 600;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #161b22;
                color: #58a6ff;
                border-color: #30363d;
                border-bottom: 2px solid #58a6ff;
            }
            QTabBar::tab:hover {
                color: #f0f6fc;
                background: #21262d;
            }
            QTableWidget {
                background-color: #161b22;
                color: #e6edf3;
                gridline-color: #30363d;
                border: none;
                font-size: 11px;
                selection-background-color: #1f6feb;
            }
            QHeaderView::section {
                background-color: #21262d;
                color: #8b949e;
                padding: 6px 10px;
                font-weight: bold;
                border: 1px solid #30363d;
                font-size: 11px;
            }
            QTextBrowser {
                background-color: #161b22;
                color: #e6edf3;
                border: none;
                font-family: 'Consolas', 'JetBrains Mono', monospace;
                font-size: 11px;
                line-height: 1.5;
            }
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
                padding: 6px 14px;
                border-radius: 4px;
                font-weight: 600;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #ffffff;
                border-color: #58a6ff;
            }
            QPushButton#btnPrimary {
                background-color: #238636;
                border-color: #2ea043;
                color: #ffffff;
            }
            QPushButton#btnPrimary:hover {
                background-color: #2ea043;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)

        # --- Header Section ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        logo_label = QLabel("📸")
        logo_label.setStyleSheet("font-size: 42px;")
        header_layout.addWidget(logo_label)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)

        title_lbl = QLabel("Lumens Deep <span style='color:#58a6ff; font-size:13px; font-weight:normal;'>v1.0.0 Production Release</span>")
        title_lbl.setStyleSheet("font-size: 20px; font-weight: 800; color: #ffffff;")
        title_layout.addWidget(title_lbl)

        tagline_lbl = QLabel("Next-Generation 32-Bit Linear RAW Photo Studio & 3D Neural Relighting Suite")
        tagline_lbl.setStyleSheet("font-size: 11px; color: #8b949e; font-weight: 500;")
        title_layout.addWidget(tagline_lbl)

        author_lbl = QLabel("Developed by <b>Volodymyr Makovetskyi</b> &nbsp;•&nbsp; © 2026 All Rights Reserved")
        author_lbl.setStyleSheet("font-size: 11px; color: #00d2ff; margin-top: 2px;")
        title_layout.addWidget(author_lbl)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #30363d; margin: 2px 0;")
        layout.addWidget(div)

        # --- Tab Widget ---
        tabs = QTabWidget()
        tabs.addTab(self._create_overview_tab(), "✨ Overview & Core Tech")
        tabs.addTab(self._create_libraries_tab(), "📚 Third-Party Libraries & Licenses")
        tabs.addTab(self._create_license_tab(), "📜 Application License (MIT)")
        layout.addWidget(tabs)

        # --- Footer Actions ---
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

        btn_web = QPushButton("🌐 Website")
        btn_web.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://lumensdeep.com")))
        footer_layout.addWidget(btn_web)

        btn_docs = QPushButton("📖 Documentation")
        btn_docs.clicked.connect(self._open_docs)
        footer_layout.addWidget(btn_docs)

        btn_shortcuts = QPushButton("⌨️ Shortcuts")
        btn_shortcuts.clicked.connect(self._open_shortcuts)
        footer_layout.addWidget(btn_shortcuts)

        btn_gh = QPushButton("🐙 GitHub")
        btn_gh.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/3dcarrots/LumensDeep")))
        footer_layout.addWidget(btn_gh)

        footer_layout.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setObjectName("btnPrimary")
        btn_close.clicked.connect(self.accept)
        footer_layout.addWidget(btn_close)

        layout.addLayout(footer_layout)

    def _open_docs(self):
        docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "USER_GUIDE.md"))
        if os.path.exists(docs_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(docs_path))
        else:
            QDesktopServices.openUrl(QUrl("https://lumensdeep.com/docs.html"))

    def _open_shortcuts(self):
        sc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs", "SHORTCUTS.md"))
        if os.path.exists(sc_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(sc_path))
        else:
            QDesktopServices.openUrl(QUrl("https://lumensdeep.com/shortcuts.html"))

    def _create_overview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        html = """
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; color: #c9d1d9; line-height: 1.6;">
            <h3 style="color: #58a6ff; margin-top: 0; font-size: 14px;">🌟 About Lumens Deep</h3>
            <p>
                <strong>Lumens Deep</strong> is a high-performance, non-destructive computational photography studio engineered from the ground up for modern high-resolution digital cameras (Canon EOS R5, Sony Alpha, Nikon Z, Fujifilm).
            </p>
            
            <h4 style="color: #00d2ff; margin-bottom: 6px; font-size: 13px;">⚡ Key Architectural Innovations:</h4>
            <ul>
                <li><strong>32-Bit Linear GPU Pipeline (ModernGL):</strong> All demosaicing, curves, lens geometry, and blending operations run on 32-bit floating-point GPU framebuffers with hardware trilinear mipmap anti-aliasing.</li>
                <li><strong>3D Optical Bokeh Raymarching:</strong> Physically-based Monte Carlo Vogel spiral raymarching across metric depth maps, supporting polygonal iris blades (3 to 10 blades), cat-eye vignetting, and chromatic aberration.</li>
                <li><strong>Multimodal AI Co-Pilot (Google Gemini):</strong> Deep reasoning auto-tuning with dampened convergence, separating retoucher instructions from dynamic photo prompts.</li>
                <li><strong>2-Phase Intelligent Culling:</strong> Global triage combined with high-resolution Center 50% Crop optical analysis to distinguish genuine camera shake from artistic shallow depth of field.</li>
                <li><strong>Aspect Ratio Subfolder Organization & 5★ Rule:</strong> Automatically categorizes batch exports into dedicated aspect subfolders (<code>Original</code>, <code>Photo_iPad_4x5</code>, <code>Portrait_2x3</code>, <code>Square_1x1</code>, <code>Story_3x4</code>, <code>Cinematic_16x9</code>) with 5★ master preservation.</li>
                <li><strong>Sub-50ms Fast Viewer Mode:</strong> Hardware libjpeg draft scaling, vectorized 256-LUT mapping, and 3-channel GPU direct uploads for instant browsing.</li>
                <li><strong>Lensfun Optical Undistortion on 3D Maps:</strong> Mode 2 geometric distortion correction for Depth, Normal, and Subject maps eliminates ghosting and displacement artifacts.</li>
            </ul>
        </div>
        """
        browser.setHtml(html)
        layout.addWidget(browser)
        return tab

    def _create_libraries_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        info_lbl = QLabel("Lumens Deep is built with gratitude upon the following world-class open-source libraries:")
        info_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        layout.addWidget(info_lbl)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Library", "License", "Purpose / Role", "Website / Project"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setStyleSheet("QTableWidget { alternate-background-color: #1c2128; }")

        libraries = [
            ("PySide6 / Qt 6", "LGPL-3.0 / Commercial", "GUI framework, multi-dock layout, signals/slots, OpenGL context", "https://qt.io"),
            ("ModernGL", "MIT License", "High-performance Python wrapper for hardware OpenGL 3.3+ rendering & shaders", "https://github.com/moderngl/moderngl"),
            ("rawpy / LibRaw", "LGPL-2.1 / CDDL-1.0", "Direct camera sensor RAW decoding (CR2, CR3, NEF, ARW, DNG, RAF, RW2)", "https://github.com/letmaik/rawpy"),
            ("Pillow (PIL Fork)", "HPND / Historical", "Image I/O, draft scale-on-decode, thumbnail reading, format encoders", "https://python-pillow.org"),
            ("NumPy", "BSD-3-Clause", "Vectorized multi-dimensional array mathematics and LUT transfer mapping", "https://numpy.org"),
            ("PyTorch (torch, torchvision)", "BSD-3-Clause", "Deep learning tensor computation backend for neural depth & surface normals", "https://pytorch.org"),
            ("ONNX Runtime / DirectML", "MIT License", "Cross-vendor hardware-accelerated neural inference (AMD, Intel, NVIDIA, DirectML)", "https://onnxruntime.ai"),
            ("google-genai / google-generativeai", "Apache-2.0", "Google Gemini multimodal vision API & reasoning co-pilot", "https://ai.google.dev"),
            ("Lensfun (lensfunpy)", "LGPL-3.0 / GPL-3.0", "Optical lens calibration database, geometric distortion & TCA calibration", "https://lensfun.github.io"),
            ("pyexiv2 / exifread", "GPL-3.0 / BSD", "Camera EXIF, lens metadata, and camera maker notes extraction", "https://github.com/LeoHsiao1/pyexiv2"),
            ("tifffile", "BSD-3-Clause", "16-bit float and uncompressed multi-channel TIFF file I/O", "https://github.com/cgohlke/tifffile"),
            ("FastMCP / mcp", "MIT License", "Model Context Protocol server for external AI assistant pair-programming", "https://github.com/modelcontextprotocol"),
            ("OpenGL / Khronos", "Khronos Free License", "Industry-standard GPU 2D/3D graphics specification & GLSL shading language", "https://khronos.org/opengl"),
        ]

        table.setRowCount(len(libraries))
        for row, (name, lic, purpose, url) in enumerate(libraries):
            item_name = QTableWidgetItem(name)
            item_name.setFont(QFont("Segoe UI", 9, QFont.Bold))
            item_name.setForeground(Qt.white)
            table.setItem(row, 0, item_name)

            item_lic = QTableWidgetItem(lic)
            item_lic.setForeground(Qt.yellow if "GPL" in lic else Qt.cyan)
            table.setItem(row, 1, item_lic)

            item_purp = QTableWidgetItem(purpose)
            table.setItem(row, 2, item_purp)

            item_url = QTableWidgetItem(url)
            item_url.setForeground(Qt.magenta)
            table.setItem(row, 3, item_url)

        layout.addWidget(table)
        return tab

    def _create_license_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)

        browser = QTextBrowser()
        mit_license_text = """
================================================================================
MIT LICENSE
================================================================================
Copyright (c) 2026 Volodymyr Makovetskyi (3D-Coat / Lumens Deep Contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
================================================================================
"""
        browser.setText(mit_license_text.strip())
        layout.addWidget(browser)
        return tab
