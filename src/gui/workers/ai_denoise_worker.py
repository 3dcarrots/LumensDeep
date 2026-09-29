import os
import sys
import numpy as np
from PySide6.QtCore import QThread, Signal

class AIDenoiseWorker(QThread):
    """
    Background worker thread for NAFNet AI Noise Reduction.
    Processes images directly in RAM (In-Memory) without requiring disk writes,
    with an optional background export to TIFF if requested.
    """
    progress_updated = Signal(int, int, str)   # pct, total_pct, message
    status_updated = Signal(str)               # status line message
    finished_in_memory = Signal(object, str)   # (denoised_rgb_numpy, optional_output_path)
    error_occurred = Signal(str)               # error message

    def __init__(
        self,
        input_path: str = None,
        linear_rgb: np.ndarray = None,
        save_to_disk: bool = False,
        output_path: str = None,
        model_path: str = "models/NAFNet-SIDD-width64.pth",
        tile_size: int = 512,
        strength: float = 1.0,
        texture_retention: float = 0.25,
        parent=None
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.linear_rgb = linear_rgb
        self.save_to_disk = save_to_disk
        self.output_path = output_path
        self.model_path = model_path
        self.tile_size = tile_size
        self.strength = float(strength)
        self.texture_retention = float(texture_retention)
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)

            import importlib
            import ai_denoise
            importlib.reload(ai_denoise)
            from ai_denoise import RawDenoiser

            self.status_updated.emit("Initializing NAFNet Neural Network...")
            denoiser = RawDenoiser(
                model_path=self.model_path,
                tile_size=self.tile_size,
                overlap=64,
                status_callback=self.status_updated.emit
            )
            model_label = f"NAFNet Ultra (width64)" if getattr(denoiser, 'width', 32) == 64 else f"NAFNet (width32)"
            self.status_updated.emit(f"Running {model_label} AI Noise Reduction...")

            def progress_cb(current, total, msg):
                if self._is_cancelled:
                    raise InterruptedError("Denoise operation cancelled by user.")
                self.progress_updated.emit(current, total, msg)

            denoised_rgb = None
            out_file = None

            if self.linear_rgb is not None:
                # Direct in-memory denoising from active editor viewport
                self.status_updated.emit("Denoising current photo in RAM (In-Memory)...")
                denoised_rgb = denoiser.denoise_in_memory(
                    self.linear_rgb,
                    progress_callback=progress_cb,
                    strength=self.strength,
                    texture_retention=self.texture_retention
                )

                if self.save_to_disk and self.input_path:
                    dir_name = os.path.dirname(self.input_path)
                    stem = os.path.splitext(os.path.basename(self.input_path))[0]
                    lumen_dir = os.path.join(dir_name, "LumenDeep")
                    os.makedirs(lumen_dir, exist_ok=True)
                    out_file = self.output_path or os.path.join(lumen_dir, f"{stem}_denoised.tiff")
                    
                    import tifffile
                    u16 = np.clip(denoised_rgb * 65535.0, 0, 65535).astype(np.uint16)
                    tifffile.imwrite(out_file, u16, compression='deflate')
                    denoiser.copy_metadata(self.input_path, out_file)

            elif self.input_path:
                # Load from file and denoise
                self.status_updated.emit(f"Loading and denoising {os.path.basename(self.input_path)}...")
                if self.save_to_disk:
                    dir_name = os.path.dirname(self.input_path)
                    stem = os.path.splitext(os.path.basename(self.input_path))[0]
                    lumen_dir = os.path.join(dir_name, "LumenDeep")
                    os.makedirs(lumen_dir, exist_ok=True)
                    out_file = self.output_path or os.path.join(lumen_dir, f"{stem}_denoised.tiff")

                denoised_rgb, _, out_file = denoiser.denoise_file(
                    self.input_path,
                    output_path=out_file,
                    progress_callback=progress_cb,
                    strength=self.strength,
                    texture_retention=self.texture_retention
                )

            self.finished_in_memory.emit(denoised_rgb, out_file or "")

        except Exception as e:
            self.error_occurred.emit(str(e))
