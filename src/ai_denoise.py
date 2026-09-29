import os
import sys
import shutil
import subprocess
import numpy as np
import torch
import tifffile

try:
    from models.nafnet import NAFNet
except ImportError:
    try:
        from src.models.nafnet import NAFNet
    except ImportError:
        from .models.nafnet import NAFNet

try:
    from core.image_io import load_image_to_linear
except ImportError:
    try:
        from src.core.image_io import load_image_to_linear
    except ImportError:
        load_image_to_linear = None


class RawDenoiser:
    """
    High-Performance Neural Denoiser powered by NAFNet (SIDD).
    Operates directly in RAM (In-Memory) on linear RGB images from RAW or standard formats,
    with optional 16-bit linear TIFF export and EXIF metadata preservation.
    """
    def __init__(self, model_path="models/NAFNet-SIDD-width64.pth", device='cuda', tile_size=512, overlap=64, status_callback=None):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Look for weights (prioritizing high-capacity width64)
        found_weights = None
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            model_path,
            os.path.join(base_dir, model_path) if model_path else None,
            os.path.join(base_dir, "models", "NAFNet-SIDD-width64.pth"),
            os.path.join(base_dir, "models", "NAFNet-SIDD-width32.pth"),
            os.path.join("models", "NAFNet-SIDD-width64.pth"),
            os.path.join(os.path.dirname(__file__), "..", "models", "NAFNet-SIDD-width64.pth"),
            os.path.join(os.path.dirname(__file__), "models", "NAFNet-SIDD-width64.pth"),
            os.path.join("models", "NAFNet-SIDD-width32.pth"),
            os.path.join(os.path.dirname(__file__), "..", "models", "NAFNet-SIDD-width32.pth"),
            "NAFNet-SIDD-width64.pth",
            "NAFNet-SIDD-width32.pth",
            "NAFNet-SIDD.pth"
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                found_weights = candidate
                break
                
        # On-demand download from Hugging Face if not bundled locally
        if not found_weights:
            target_file = "NAFNet-SIDD-width64.pth" if "width64" in str(model_path) else "NAFNet-SIDD-width32.pth"
            if status_callback:
                status_callback(f"Downloading {target_file} from Hugging Face (one-time setup)...")
            try:
                from huggingface_hub import hf_hub_download
                print(f"[RawDenoiser] Weights not found locally. Downloading {target_file} from Hugging Face...")
                found_weights = hf_hub_download(repo_id="nyanko7/nafnet-models", filename=target_file)
                print(f"[RawDenoiser] Successfully cached weights at: {found_weights}")
            except Exception as e:
                print(f"[RawDenoiser] Failed to auto-download {target_file}: {e}")

        self.width = 64
        if found_weights:
            try:
                state_dict = torch.load(found_weights, map_location='cpu')
                if "params" in state_dict:
                    state_dict = state_dict["params"]
                
                # Auto-detect width from intro.weight shape [width, 3, 3, 3]
                intro_w = state_dict.get("intro.weight", None)
                if intro_w is not None and hasattr(intro_w, "shape"):
                    self.width = int(intro_w.shape[0])
                elif "width32" in found_weights.lower():
                    self.width = 32
                elif "width64" in found_weights.lower():
                    self.width = 64

                self.model = NAFNet(
                    img_channel=3,
                    width=self.width,
                    middle_blk_num=12,
                    enc_blk_nums=[2, 2, 4, 8],
                    dec_blk_nums=[2, 2, 2, 2]
                )
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[RawDenoiser] Successfully loaded pre-trained NAFNet-width{self.width} from: {found_weights}")
            except Exception as e:
                print(f"[RawDenoiser] Error loading weights from {found_weights}: {e}")
                self.model = NAFNet(img_channel=3, width=self.width, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
        else:
            print(f"[RawDenoiser] Warning: Pre-trained weights not found. Check {model_path}.")
            self.model = NAFNet(img_channel=3, width=self.width, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
            
        self.model.to(self.device)
        self.model.eval()

    def denoise_in_memory(
        self,
        linear_rgb: np.ndarray,
        progress_callback=None,
        strength: float = 1.0,
        texture_retention: float = 0.25
    ) -> np.ndarray:
        """
        Denoises a 3-channel linear RGB image (float32 [0.0, 1.0]) directly in RAM.
        Preserves color science, white balance, and dynamics without altering camera calibration.
        Supports adjustable denoise strength and micro-texture / sharpness retention.
        """
        if linear_rgb is None:
            raise ValueError("linear_rgb image data cannot be None.")

        # Ensure float32 contiguous array in range [0.0, 1.0]
        if linear_rgb.dtype != np.float32:
            linear_rgb = linear_rgb.astype(np.float32)

        orig_shape = linear_rgb.shape[:2] # H, W
        if len(linear_rgb.shape) == 2:
            linear_rgb = np.stack([linear_rgb, linear_rgb, linear_rgb], axis=-1)
        elif linear_rgb.shape[2] == 4:
            linear_rgb = linear_rgb[:, :, :3]

        orig_linear = linear_rgb.copy()

        if progress_callback:
            progress_callback(5, 100, "Preparing image for NAFNet GPU inference...")

        # Dynamic range auto-exposure normalization for NAFNet:
        # NAFNet was trained on standard exposure sRGB images (p99 ~ 0.70-0.85).
        # Underexposed RAW data has very small values (p99 < 0.1), which causes LayerNorm
        # channel variance to vanish and neural activations to diverge in dark corners.
        p99 = float(np.percentile(orig_linear, 99))
        norm_scale = float(np.clip(0.70 / max(p99, 1e-4), 1.0, 16.0))

        scaled_linear = orig_linear * norm_scale

        # Transfer to perceptual sRGB curve so NAFNet neural features match training distribution
        mask = scaled_linear <= 0.0031308
        srgb = np.where(
            mask,
            scaled_linear * 12.92,
            1.055 * (np.maximum(scaled_linear, 0.0) ** (1.0 / 2.4)) - 0.055
        )
        srgb = np.clip(srgb, 0.0, 1.0)

        # Convert to tensor (1, 3, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(srgb)).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run tiled inference with Hann window overlap and blending to prevent seams and OOM
        out_tensor = self.process_tiled(tensor, progress_callback=progress_callback)

        # Convert back to numpy (H, W, 3)
        out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip(out_np, 0.0, 1.0)

        # Invert sRGB OETF back to true linear space, then unscale back to original RAW dynamic range
        mask_inv = out_np <= 0.04045
        denoised_linear = np.where(
            mask_inv,
            out_np / 12.92,
            ((np.maximum(out_np, 0.0) + 0.055) / 1.055) ** 2.4
        ) / norm_scale
        denoised_linear = np.clip(denoised_linear, 0.0, 1.0).astype(np.float32)

        # Clean chromatic noise in shadows (eliminates colored checkerboard artifacts in dark hair)
        try:
            import cv2
            # Linear RGB -> YCbCr
            M = np.array([
                [ 0.299,     0.587,     0.114    ],
                [-0.168736, -0.331264,  0.5      ],
                [ 0.5,      -0.418688, -0.081312 ]
            ], dtype=np.float32)
            M_inv = np.array([
                [1.0,  0.0,       1.402   ],
                [1.0, -0.344136, -0.714136],
                [1.0,  1.772,     0.0     ]
            ], dtype=np.float32)

            ycbcr = denoised_linear @ M.T
            y = ycbcr[..., 0]
            cb = cv2.bilateralFilter(ycbcr[..., 1], d=5, sigmaColor=0.015, sigmaSpace=5)
            cr = cv2.bilateralFilter(ycbcr[..., 2], d=5, sigmaColor=0.015, sigmaSpace=5)
            denoised_linear = np.clip(np.stack([y, cb, cr], axis=-1) @ M_inv.T, 0.0, 1.0).astype(np.float32)
        except Exception:
            pass

        # Micro-texture & Sharpness Retention (preserves skin pores, eyelashes, and fine hair strands)
        if texture_retention > 0.0:
            detail = orig_linear - denoised_linear
            noise_floor = 0.0003 / norm_scale
            clean_detail = np.sign(detail) * np.maximum(np.abs(detail) - noise_floor, 0.0)
            denoised_linear = np.clip(denoised_linear + float(texture_retention) * clean_detail, 0.0, 1.0)

        # Global Strength blend
        if strength < 1.0:
            result = float(strength) * denoised_linear + (1.0 - float(strength)) * orig_linear
        else:
            result = denoised_linear

        result = np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)

        if progress_callback:
            progress_callback(100, 100, "AI Denoising complete in memory!")

        return result

    def process_tiled(self, img_tensor: torch.Tensor, progress_callback=None) -> torch.Tensor:
        """Processes tensor in overlapping tiles using Hann window interior blending and FP32 accumulation."""
        B, C, H, W = img_tensor.shape
        stride = self.tile_size - self.overlap
        
        # Accumulators in float32 for maximum numerical stability and zero subnormal distortion
        out_tensor = torch.zeros((B, C, H, W), dtype=torch.float32, device=self.device)
        weight_tensor = torch.zeros((B, C, H, W), dtype=torch.float32, device=self.device)
        
        # Smooth continuous Hann window ramp
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(self.overlap) + 0.5) / self.overlap)
        ramp_t = torch.from_numpy(ramp.astype(np.float32)).to(self.device)

        # Precompute coordinate list
        coords = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                coords.append((y, x))
        total_tiles = len(coords)

        for idx, (y, x) in enumerate(coords, 1):
            y1 = min(y, H - self.tile_size)
            x1 = min(x, W - self.tile_size)
            if y1 < 0: y1 = 0
            if x1 < 0: x1 = 0
            
            y2 = min(y1 + self.tile_size, H)
            x2 = min(x1 + self.tile_size, W)
            
            tile = img_tensor[:, :, y1:y2, x1:x2]
            
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out_tile = self.model(tile)
                else:
                    out_tile = self.model(tile)
                out_tile = torch.clamp(out_tile.to(torch.float32), 0.0, 1.0)
            
            th, tw = out_tile.shape[2:]

            # Mask with interior-only tapering (prevents corner/boundary ring oscillations)
            m = torch.ones((1, C, th, tw), dtype=torch.float32, device=self.device)
            if y1 > 0 and th >= self.overlap:
                m[:, :, :self.overlap, :] *= ramp_t.view(1, 1, self.overlap, 1)
            if y2 < H and th >= self.overlap:
                m[:, :, -self.overlap:, :] *= ramp_t.flip(0).view(1, 1, self.overlap, 1)
            if x1 > 0 and tw >= self.overlap:
                m[:, :, :, :self.overlap] *= ramp_t.view(1, 1, 1, self.overlap)
            if x2 < W and tw >= self.overlap:
                m[:, :, :, -self.overlap:] *= ramp_t.flip(0).view(1, 1, 1, self.overlap)

            out_tensor[:, :, y1:y2, x1:x2] += out_tile * m
            weight_tensor[:, :, y1:y2, x1:x2] += m

            if progress_callback:
                pct = int(10 + (idx / max(1, total_tiles)) * 85)
                progress_callback(pct, 100, f"Denoising tile {idx}/{total_tiles}")
                
        out_tensor = out_tensor / torch.clamp(weight_tensor, min=1e-5)
        return out_tensor

    def denoise_file(self, input_path: str, output_path: str = None, progress_callback=None, strength: float = 1.0, texture_retention: float = 0.25):
        """
        Loads any photo (RAW or raster), denoises it in RAM, and optionally saves to 16-bit TIFF.
        """
        if progress_callback:
            progress_callback(0, 100, f"Decoding {os.path.basename(input_path)}...")

        if load_image_to_linear:
            linear_rgb, metadata = load_image_to_linear(input_path)
        else:
            from PIL import Image
            img = Image.open(input_path).convert('RGB')
            linear_rgb = np.ascontiguousarray(np.array(img, dtype=np.float32) / 255.0)
            metadata = {}

        denoised_rgb = self.denoise_in_memory(
            linear_rgb,
            progress_callback=progress_callback,
            strength=strength,
            texture_retention=texture_retention
        )

        if output_path:
            if progress_callback:
                progress_callback(95, 100, f"Saving 16-bit linear TIFF to disk...")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            # Save 16-bit linear TIFF
            u16 = np.clip(denoised_rgb * 65535.0, 0, 65535).astype(np.uint16)
            tifffile.imwrite(output_path, u16, compression='deflate')
            self.copy_metadata(input_path, output_path)

        return denoised_rgb, metadata, output_path

    def copy_metadata(self, input_path, output_path):
        """Copies full EXIF & Makernotes from original photo to destination TIFF."""
        if not output_path or not os.path.exists(output_path):
            return
        if shutil.which('exiftool'):
            cmd = ['exiftool', '-tagsFromFile', input_path, '-all:all>all:all', '-overwrite_original', output_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("[RawDenoiser] Metadata copied successfully via exiftool.")
                return
            except Exception as e:
                print(f"[RawDenoiser] exiftool notice: {e}")
        
        try:
            from pyexiv2 import Image
            with Image(input_path) as src, Image(output_path) as dst:
                src.copy(dst)
            print("[RawDenoiser] Metadata copied successfully using pyexiv2.")
        except Exception:
            pass
