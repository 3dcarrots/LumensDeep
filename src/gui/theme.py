from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt

DARK_PALETTE = {
    'window': QColor(40, 40, 40),
    'window_text': Qt.white,
    'base': QColor(25, 25, 25),
    'alternate_base': QColor(40, 40, 40),
    'tooltip_base': Qt.white,
    'tooltip_text': Qt.white,
    'text': Qt.white,
    'button': QColor(53, 53, 53),
    'button_text': Qt.white,
    'bright_text': Qt.red,
    'link': QColor(42, 130, 218),
    'highlight': QColor(42, 130, 218),
    'highlighted_text': Qt.black,
}

MAIN_STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QDockWidget {
    color: #e0e0e0;
    font-weight: bold;
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(float.png);
}
QDockWidget::title {
    text-align: left;
    background: #2d2d2d;
    padding: 6px;
    border-bottom: 1px solid #3d3d3d;
}
QScrollArea {
    border: none;
    background-color: #1e1e1e;
}
QScrollBar:vertical {
    border: none;
    background: #252525;
    width: 10px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #444;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #555;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QSlider::groove:horizontal {
    border: 1px solid #3d3d3d;
    height: 4px;
    background: #2d2d2d;
    margin: 2px 0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00d2ff;
    border: 1px solid #0099bb;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #33ddff;
}
QPushButton {
    background-color: #333;
    color: #eee;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}
QPushButton:hover {
    background-color: #444;
    border-color: #00d2ff;
}
QPushButton:pressed {
    background-color: #222;
}
QPushButton:checked {
    background-color: #007799;
    border-color: #00d2ff;
    color: #fff;
}
QCheckBox {
    color: #ccc;
    font-size: 11px;
    spacing: 5px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #555;
    border-radius: 3px;
    background: #2d2d2d;
}
QCheckBox::indicator:checked {
    background: #00d2ff;
    border-color: #0099bb;
}
QComboBox {
    background-color: #2d2d2d;
    color: #eee;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 11px;
}
QComboBox:hover {
    border-color: #00d2ff;
}
QComboBox::drop-down {
    border: none;
}
"""

def set_dark_theme(app):
    """Applies the dark Fusion palette and stylesheet to the QApplication."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, DARK_PALETTE['window'])
    palette.setColor(QPalette.WindowText, DARK_PALETTE['window_text'])
    palette.setColor(QPalette.Base, DARK_PALETTE['base'])
    palette.setColor(QPalette.AlternateBase, DARK_PALETTE['alternate_base'])
    palette.setColor(QPalette.ToolTipBase, DARK_PALETTE['tooltip_base'])
    palette.setColor(QPalette.ToolTipText, DARK_PALETTE['tooltip_text'])
    palette.setColor(QPalette.Text, DARK_PALETTE['text'])
    palette.setColor(QPalette.Button, DARK_PALETTE['button'])
    palette.setColor(QPalette.ButtonText, DARK_PALETTE['button_text'])
    palette.setColor(QPalette.BrightText, DARK_PALETTE['bright_text'])
    palette.setColor(QPalette.Link, DARK_PALETTE['link'])
    palette.setColor(QPalette.Highlight, DARK_PALETTE['highlight'])
    palette.setColor(QPalette.HighlightedText, DARK_PALETTE['highlighted_text'])
    app.setPalette(palette)
    app.setStyleSheet(MAIN_STYLESHEET)
