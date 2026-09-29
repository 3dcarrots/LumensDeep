import sys
import os
import multiprocessing

# Critical for frozen Windows apps: intercept multiprocessing child worker processes before Qt initializes
multiprocessing.freeze_support()

# Set working directory & sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Set Windows App ID for taskbar icon grouping
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LumensDeep.PhotoEditor.0.1")
    except Exception:
        pass

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

def get_app_icon_path() -> str:
    if getattr(sys, 'frozen', False):
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        for rel in [os.path.join('assets', 'app_icon.ico'), 'app_icon.ico']:
            cand = os.path.join(base_dir, rel)
            if os.path.exists(cand):
                return cand
    else:
        cand = os.path.join(ROOT_DIR, 'assets', 'app_icon.ico')
        if os.path.exists(cand):
            return cand
    return ""

def main():
    app = QApplication(sys.argv)
    
    icon_path = get_app_icon_path()
    if icon_path and os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 1. Instant Startup Feedback: Show Splash Screen immediately (< 0.15s)
    from gui.widgets.splash_screen import LumensSplashScreen
    splash = LumensSplashScreen(icon_path)
    splash.show()
    splash.set_status("Starting Lumens Deep...", 15)

    # 2. Initialize theme
    splash.set_status("Loading studio theme...", 35)
    from gui.theme import set_dark_theme
    set_dark_theme(app)

    # 3. Load GPU & 3D Neural Engine
    splash.set_status("Initializing GPU & 3D Neural Engine...", 65)
    from gui.main_window import PhotoEditorWindow

    # 4. Initialize studio workspace
    splash.set_status("Configuring studio workspace...", 85)
    window = PhotoEditorWindow()
    if icon_path and os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))

    splash.set_status("Ready!", 100)
    window.show()
    splash.close()

    sys.exit(app.exec())

def check_ai_diagnostics():
    import tempfile
    diag_file = os.path.join(tempfile.gettempdir(), "lumensdeep_diagnostics.txt")
    lines = ["=== Lumens Deep AI Diagnostics ==="]
    success = False
    try:
        import torch
        lines.append(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        from core.adaptive_depth_anything import AdaptiveDepthAnything, detect_device_capabilities
        caps = detect_device_capabilities()
        lines.append(f"Caps: GPU='{caps.gpu_name}', CUDA={caps.cuda_available}, Rec={caps.recommended_backend}")
        engine = AdaptiveDepthAnything(hardware_mode="Auto")
        lines.append(f"DepthAnything Backend: {engine.active_backend} ({engine.backend_description})")
        lines.append("[OK] AdaptiveDepthAnything initialized successfully!")
        success = True
    except Exception as e:
        lines.append(f"[ERROR] AdaptiveDepthAnything failed: {e}")
        import traceback
        lines.append(traceback.format_exc())
    
    result_str = "\n".join(lines)
    print(result_str)
    try:
        with open(diag_file, "w", encoding="utf-8") as f:
            f.write(result_str)
    except Exception:
        pass
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    if "--check-ai" in sys.argv:
        check_ai_diagnostics()
    main()
