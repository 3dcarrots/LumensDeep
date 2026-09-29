import subprocess
import sys
import os
import tempfile
import numpy as np
from PIL import Image

# Static imports for PyInstaller frozen bundle discovery
try:
    import transformers.models.depth_anything.modeling_depth_anything
    import transformers.models.depth_anything.configuration_depth_anything
    import transformers.models.segformer.modeling_segformer
    import transformers.models.segformer.configuration_segformer
    import diffusers.pipelines.marigold.pipeline_marigold_normals
except ImportError:
    pass

def _get_local_model_path(repo_id: str) -> str:
    try:
        from pathlib import Path
        from huggingface_hub.constants import HF_HUB_CACHE
        
        # Convert repo_id like 'prs-eth/marigold-normals-v1-1' to folder format
        folder_name = "models--" + repo_id.replace("/", "--")
        cache_dir = Path(HF_HUB_CACHE) / folder_name / "snapshots"
        
        if cache_dir.exists():
            snaps = list(cache_dir.iterdir())
            if snaps:
                # Return the absolute path of the first snapshot
                return str(snaps[0])
    except Exception:
        pass
    return repo_id

def _notify_tile_progress(cb, cur: int, total: int, stage: str = "Tiling", workers: int = 1):
    """Helper to invoke tile_progress_callback supporting both 2 and 4 argument signatures."""
    if cb:
        try:
            cb(cur, total, stage, workers)
        except TypeError:
            try:
                cb(cur, total)
            except Exception:
                pass
        except Exception:
            pass

class AIMaskGenerator:
    def __init__(self):
        self.models = {}

    def ensure_dependencies(self, progress_callback=None):
        """Check and install required packages."""
        # In a frozen standalone Windows application (.exe), packages cannot and must not be pip-installed,
        # and calling sys.executable would accidentally launch a second instance of LumensDeep.exe!
        if getattr(sys, 'frozen', False):
            return

        missing = []
        try:
            import torch
        except ImportError:
            missing.append('torch')
            missing.append('torchvision')
            
        try:
            import transformers
        except ImportError:
            missing.append('transformers')
            
        try:
            import rembg
        except ImportError:
            missing.append('rembg[cli]')
            
        try:
            import onnxruntime
        except ImportError:
            missing.append('onnxruntime')

        try:
            import diffusers
        except ImportError:
            missing.append('diffusers')

        try:
            import accelerate
        except ImportError:
            missing.append('accelerate')

        if missing:
            if progress_callback:
                progress_callback(f"Installing missing AI packages: {', '.join(missing)}... This may take a few minutes.")
            
            # Using subprocess to install and stream output
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in process.stdout:
                    if progress_callback:
                        progress_callback(line.strip()[:80]) # truncate to avoid huge labels
                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(f"Pip install failed with code {process.returncode}")
            except Exception as e:
                raise RuntimeError(f"Failed to install dependencies: {e}")
                
            if progress_callback:
                progress_callback("Finished installing packages. Loading models...")

        # Hardware check: if NVIDIA GPU is present but torch does not have CUDA enabled
        try:
            from core.adaptive_depth_anything import detect_device_capabilities
            caps = detect_device_capabilities()
            if caps.has_nvidia_gpu and not caps.cuda_available:
                if progress_callback:
                    progress_callback("NVIDIA GPU detected without CUDA PyTorch! Installing PyTorch CUDA acceleration...")
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu126"]
                subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            pass

    def generate_mask(self, image_np, mask_type, high_res_depth=True, high_res_normals=True, normals_method="Hybrid (Marigold + Depth)", normals_ensemble=1, normals_scale=1.0, cached_depth=None, cached_normals=None, disable_gpu=False, hardware_mode="Auto", progress_callback=None, tile_progress_callback=None):
        """
        Generate a mask from a numpy image (H, W, 3) in RGB 0-255 uint8 format.
        Returns a single-channel numpy array (H, W) in uint8 format (0-255).
        """
        self.ensure_dependencies(progress_callback)
        
        if mask_type == "Subject/Background":
            return self._generate_rembg(image_np, progress_callback, cached_depth, tile_progress_callback=tile_progress_callback)
        elif mask_type == "Depth Map":
            return self._generate_depth(image_np, high_res_depth, progress_callback, tile_progress_callback=tile_progress_callback, disable_gpu=disable_gpu, hardware_mode=hardware_mode)
        elif mask_type == "Normal Map":
            return self._generate_normals(
                image_np, high_res_normals, normals_method, normals_ensemble, normals_scale,
                depth_map=cached_depth, progress_callback=progress_callback, tile_progress_callback=tile_progress_callback
            )
        elif mask_type == "Geometry":
            if cached_depth is not None:
                depth = cached_depth
                if progress_callback: progress_callback("Using cached 32-bit depth map...")
            else:
                depth = self._generate_depth(image_np, high_res_depth, progress_callback, tile_progress_callback=tile_progress_callback, disable_gpu=disable_gpu, hardware_mode=hardware_mode)
                
            if cached_normals is not None:
                normals = cached_normals
                if progress_callback: progress_callback("Using cached normal map...")
            else:
                normals = self._generate_normals(image_np, high_res=high_res_normals, normals_method=normals_method, normals_ensemble=normals_ensemble, normals_scale=normals_scale, depth_map=depth, progress_callback=progress_callback, tile_progress_callback=tile_progress_callback)
            return (depth, normals)
        else:
            return self._generate_face_parsing(image_np, mask_type, progress_callback)

    def _guided_filter(self, I, p, r, eps):
        import cv2
        mean_I = cv2.blur(I, (r, r))
        mean_p = cv2.blur(p, (r, r))
        mean_II = cv2.blur(I * I, (r, r))
        mean_Ip = cv2.blur(I * p, (r, r))
        
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = cv2.blur(a, (r, r))
        mean_b = cv2.blur(b, (r, r))
        
        q = mean_a * I + mean_b
        return q

    def _generate_rembg(self, image_np, progress_callback, cached_depth=None, tile_progress_callback=None):
        import rembg
        import cv2
        import numpy as np
        from PIL import Image
        
        if progress_callback: progress_callback("Running Background Removal (Global BiRefNet)...")
        _notify_tile_progress(tile_progress_callback, 1, 3, "BiRefNet Subject Segmentation", 1)
        
        # Using birefnet-portrait generally gives better results for hair and people
        if 'rembg_session' not in self.models:
            self.models['rembg_session'] = rembg.new_session("birefnet-portrait")
            
        pil_img = Image.fromarray(image_np)
        
        # 1. Global Mask
        global_mask = rembg.remove(
            pil_img, 
            session=self.models['rembg_session'], 
            only_mask=True,
            post_process_mask=False
        )
        global_mask_np = np.array(global_mask, dtype=np.float32) / 255.0
        
        # Apply Depth-Aware Guided Filtering to the Global Mask to sharpen hard surfaces
        if cached_depth is None:
            if progress_callback: progress_callback("Generating high-res Depth Map for contour refinement...")
            depth_map = self._generate_depth(image_np, high_res=True, progress_callback=progress_callback, tile_progress_callback=tile_progress_callback)
        else:
            depth_map = cached_depth
            
        if progress_callback: progress_callback("Refining body contours using Depth Map Guided Filter...")
        _notify_tile_progress(tile_progress_callback, 2, 3, "Depth Contour Refinement", 1)
        # Depth map is usually uint8 or float32. Ensure it's float32 [0..1]
        if depth_map.dtype == np.uint8:
            depth_guide = depth_map.astype(np.float32) / 255.0
        else:
            depth_guide = depth_map.astype(np.float32)
        # Instead of strict binarization which destroys edge anti-aliasing (causing jagged binary edges),
        # we will use a smooth curve (like smoothstep) to make the interior solid white while preserving soft edges.
        global_mask_np = (global_mask_np - 0.1) / 0.8
        global_mask_np = np.clip(global_mask_np, 0.0, 1.0)
        # Apply a slight gamma to boost the midtones to white, making the interior solid
        global_mask_np = np.power(global_mask_np, 0.5)

        # High-Resolution Fix: Global Guided Filter using RGB guidance
        # This acts as an Adaptive Guided Upsampling that mathematically transfers 
        # the high-frequency microstructures (hair, meshes) from the 8K photo onto the blurry mask.
        if progress_callback: progress_callback("Refining all edges globally using High-Resolution Guided Filter...")
        import cv2
        gray_guide = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # We use r=20 to cover the blur radius of the base mask, 
        # and a very low eps (1e-3) to force the mask to strictly follow the sharp RGB edges (hair).
        global_mask_np = self._guided_filter(gray_guide, global_mask_np, r=20, eps=1e-3)
        global_mask_np = np.clip(global_mask_np, 0.0, 1.0)


        final_result = (np.clip(global_mask_np, 0, 1) * 255).astype(np.uint8)
        if progress_callback: progress_callback("High-res mask generation complete.")
        return final_result

    def _get_adaptive_depth_engine(self, progress_callback=None, disable_gpu=False, hardware_mode="Auto"):
        key = ("adaptive_depth", disable_gpu, hardware_mode)
        if key not in self.models:
            if progress_callback:
                progress_callback(f"Loading Adaptive DepthAnythingV2 Engine (mode={hardware_mode}, disable_gpu={disable_gpu})...")
            from core.adaptive_depth_anything import AdaptiveDepthAnything
            
            repo_id = "depth-anything/Depth-Anything-V2-Large-hf"
            model_path = _get_local_model_path(repo_id)
            
            engine = AdaptiveDepthAnything(
                model_id=model_path,
                disable_gpu=disable_gpu,
                hardware_mode=hardware_mode,
                input_size=(518, 518),
                progress_callback=progress_callback
            )
            self.models[key] = engine
        return self.models[key]

    def _generate_depth(self, image_np, high_res=False, progress_callback=None, is_recursive=False, tile_progress_callback=None, disable_gpu=False, hardware_mode="Auto"):
        engine = self._get_adaptive_depth_engine(progress_callback, disable_gpu=disable_gpu, hardware_mode=hardware_mode)
        
        if progress_callback and not is_recursive:
            progress_callback(f"Running Depth Estimation ({engine.backend_description})...")
            
        def process_image(img):
            return engine.infer(img)

        # Reduce patch size to 518 to match DepthAnything V2's internal processing size exactly
        if high_res and max(image_np.shape[0], image_np.shape[1]) > 518:
            H, W = image_np.shape[:2]
            
            # HIERARCHICAL TILING: Prevent the global map from being stretched too much (max 4x upscale)
            MAX_UPSCALE = 4.0
            import cv2
            if max(H, W) > 518 * MAX_UPSCALE:
                if progress_callback: progress_callback("Generating intermediate depth map for Hierarchical Tiling...")
                scale = (518 * MAX_UPSCALE) / max(H, W)
                intermediate_w, intermediate_h = int(W * scale), int(H * scale)
                intermediate_img = cv2.resize(image_np, (intermediate_w, intermediate_h), interpolation=cv2.INTER_AREA)
                # Recursively generate a high-quality intermediate reference map
                global_depth = self._generate_depth(
                    intermediate_img, high_res=True, progress_callback=progress_callback,
                    is_recursive=True, tile_progress_callback=tile_progress_callback,
                    disable_gpu=disable_gpu, hardware_mode=hardware_mode
                )
                global_depth = cv2.resize(global_depth, (W, H), interpolation=cv2.INTER_CUBIC)
            else:
                if progress_callback and not is_recursive: progress_callback("Generating base global depth map...")
                scale = 518 / max(H, W)
                global_w, global_h = int(W * scale), int(H * scale)
                global_img = cv2.resize(image_np, (global_w, global_h), interpolation=cv2.INTER_AREA)
                global_depth = process_image(global_img)
                global_depth = cv2.resize(global_depth, (W, H), interpolation=cv2.INTER_CUBIC)
                
            if progress_callback: progress_callback(f"Tiling depth map via Laplacian PatchFusion ({engine.active_backend})...")
            
            # IMPORTANT: DepthAnything V2 processes images internally at 518x518 (a multiple of 14).
            # By passing exactly 518x518, we bypass internal resampling and avoid Moiré artifacts.
            patch_size = 518
            overlap = 126
            stride = patch_size - overlap
            
            final_depth = np.zeros((H, W), dtype=np.float32)
            weight_sum = np.zeros((H, W), dtype=np.float32)
            
            # Create a 1D blend window
            def create_blend_window(size, overlap_size):
                w = np.ones(size, dtype=np.float32)
                ramp = (1.0 - np.cos(np.linspace(0, np.pi, overlap_size, endpoint=True, dtype=np.float32))) / 2.0
                w[:overlap_size] = ramp
                w[-overlap_size:] = ramp[::-1]
                return w
            
            import cv2
            blur_radius = max(3, (patch_size // 2 | 1))
            global_depth_blurred = cv2.GaussianBlur(global_depth, (blur_radius, blur_radius), 0)

            y_steps = list(range(0, max(1, H - patch_size + 1), stride))
            if H > patch_size and (H - patch_size) % stride != 0:
                y_steps.append(H - patch_size)
                
            x_steps = list(range(0, max(1, W - patch_size + 1), stride))
            if W > patch_size and (W - patch_size) % stride != 0:
                x_steps.append(W - patch_size)
                
            tile_coords = []
            for y in y_steps:
                for x in x_steps:
                    actual_h = min(patch_size, H - y)
                    actual_w = min(patch_size, W - x)
                    tile_coords.append((y, x, actual_h, actual_w))
                    
            total_tiles = len(tile_coords)
            
            # Use batched inference for GPU or multi-threading for CPU
            is_gpu = engine.active_backend not in ["PyTorch_CPU", "ONNX_CPU"]
            batch_size = 4 if is_gpu else 2
            
            if progress_callback:
                progress_callback(f"Tiling depth map: 0 / {total_tiles} (Engine: {engine.active_backend}, batch={batch_size})...")
            _notify_tile_progress(tile_progress_callback, 0, total_tiles, f"Laplacian PatchFusion ({engine.active_backend})", batch_size)
            
            completed_tiles = 0
            
            # Process in batches
            for b_idx in range(0, total_tiles, batch_size):
                cur_batch_coords = tile_coords[b_idx : b_idx + batch_size]
                cur_patches = [image_np[y:y+ah, x:x+aw] for y, x, ah, aw in cur_batch_coords]
                
                # Perform batched GPU/CPU forward pass
                batch_depths = engine.infer_batch(cur_patches)
                
                for (y, x, actual_h, actual_w), local_depth in zip(cur_batch_coords, batch_depths):
                    global_patch = global_depth[y:y+actual_h, x:x+actual_w]
                    global_patch_blurred = global_depth_blurred[y:y+actual_h, x:x+actual_w]
                    
                    local_flat = local_depth.flatten()
                    global_flat = global_patch.flatten()
                    A = np.vstack([local_flat, np.ones_like(local_flat)]).T
                    res = np.linalg.lstsq(A, global_flat, rcond=None)[0]
                    s = res[0]
                    
                    # SAFETY: Prevent extreme contrast on flat patches (s=0) or inverted depth (s<0)
                    if np.isnan(s) or np.isinf(s): s = 1.0
                    s = np.clip(s, 0.1, 10.0)
                    
                    local_blurred = cv2.GaussianBlur(local_depth, (blur_radius, blur_radius), 0)
                    tile_details = local_depth - local_blurred
                    aligned_local = (s * tile_details) + global_patch_blurred
                    
                    wy_local = create_blend_window(actual_h, overlap)
                    wx_local = create_blend_window(actual_w, overlap)
                    if y == 0: wy_local[:overlap] = 1.0
                    if y + actual_h == H: wy_local[-overlap:] = 1.0
                    if x == 0: wx_local[:overlap] = 1.0
                    if x + actual_w == W: wx_local[-overlap:] = 1.0
                    
                    w_patch = wy_local[:, None] * wx_local[None, :]
                    
                    final_depth[y:y+actual_h, x:x+actual_w] += aligned_local * w_patch
                    weight_sum[y:y+actual_h, x:x+actual_w] += w_patch
                    
                    completed_tiles += 1
                
                _notify_tile_progress(tile_progress_callback, completed_tiles, total_tiles, f"Laplacian PatchFusion ({engine.active_backend})", batch_size)
                if progress_callback:
                    progress_callback(f"Tiling depth map: Completed tile {completed_tiles} of {total_tiles} ({engine.active_backend})...")
                    
            depth_np = final_depth / np.maximum(weight_sum, 1e-8)
            
        else:
            if progress_callback: progress_callback(f"Computing full-frame depth ({engine.backend_description})...")
            _notify_tile_progress(tile_progress_callback, 1, 1, f"Single-Pass Depth ({engine.active_backend})", 1)
            depth_np = process_image(image_np)
            
        # Fast Self-Guided Filter to eliminate Vision Transformer (ViT) patch grid artifacts 
        # (horizontal/vertical lines every ~28 pixels) without blurring sharp object contours.
        if progress_callback: progress_callback("Applying Guided Filter to eliminate ViT grid artifacts...")
        _notify_tile_progress(tile_progress_callback, 1, 1, "Guided Filter Post-Processing", 1)
        import cv2
        r = max(15, min(depth_np.shape[0], depth_np.shape[1]) // 40)
        eps = 1e-4
        
        # Self-Guided Filter O(1) implementation
        mean_p = cv2.boxFilter(depth_np, cv2.CV_32F, (r, r))
        mean_pp = cv2.boxFilter(depth_np * depth_np, cv2.CV_32F, (r, r))
        var_p = mean_pp - mean_p * mean_p
        
        a = var_p / (var_p + eps)
        b = mean_p - a * mean_p
        
        mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
        
        depth_np = mean_a * depth_np + mean_b
            
        if not is_recursive:
            depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            
        return depth_np.astype(np.float32)

    def _safe_load_face_model(self, progress_callback):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch
        import shutil
        from pathlib import Path
        
        repo_id = "jonathandinu/face-parsing"
        
        def load():
            proc = SegformerImageProcessor.from_pretrained(repo_id)
            mod = SegformerForSemanticSegmentation.from_pretrained(repo_id)
            if torch.cuda.is_available():
                mod = mod.to("cuda")
            return proc, mod
            
        try:
            return load()
        except (OSError, ValueError, RuntimeError) as e:
            if progress_callback: progress_callback(f"Cache error detected: {e}. Clearing cache and retrying...")
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            return load()

    def _generate_normals_iclight(self, image_np, high_res=False, normals_ensemble=3, progress_callback=None, tile_progress_callback=None):
        if progress_callback: progress_callback("Running IC-Light Normal extraction (Photometric Stereo)...")
        
        # This is a stub for the Photometric Stereo IC-Light implementation.
        # A full implementation requires replacing the SD UNet with lllyasviel/ic-light weights,
        # generating 4 relit images (Left, Right, Top, Bottom) and solving the linear system:
        # I = L * N  => N = (L^T L)^-1 L^T I
        
        # For now, we fallback to Marigold as the pipeline requires custom diffusers blocks.
        if progress_callback: progress_callback("IC-Light UNet not fully integrated. Falling back to Marigold...")
        return self._generate_normals(image_np, high_res, "Marigold", normals_ensemble, progress_callback=progress_callback, tile_progress_callback=tile_progress_callback)

    def _depth_to_normals_3d(self, depth_f, normals_scale=1.0, override_bump_scale=None):
        import cv2
        H, W = depth_f.shape
        
        # TRUE 3D POINT CLOUD NORMAL CALCULATION (Perspective Projection)
        

        
        # The user found that squaring the depth map (Pow 2) produces an ideal linear heightfield for 3D displacement.
        # We use this as our heightmap H(x, y).
        height_map = (depth_f ** 2).astype(np.float32)
        
        # Apply UI scale to control bump strength. 
        if override_bump_scale is not None:
            bump_scale = override_bump_scale
        else:
            bump_scale = 1500.0 * normals_scale
        Z_bump = (height_map * bump_scale).astype(np.float32)
        
        # Calculate gradients using Sobel (standard heightmap-to-normal method)
        dZ_dx = cv2.Sobel(Z_bump, cv2.CV_32F, 1, 0, ksize=3)
        dZ_dy = cv2.Sobel(Z_bump, cv2.CV_32F, 0, 1, ksize=3)
        
        # Sobel includes a factor of 8 (since ksize=3 sum of weights is 8). Divide it out.
        dZ_dx /= 8.0
        dZ_dy /= 8.0
        
        # Orthographic Normal vector: [-dZ/dx, -dZ/dy, 1]
        # X points right, Y points down (in image space).
        # To make Y point UP for OpenGL Normal Map format, we invert the Y gradient logic.
        nx = -dZ_dx
        ny = dZ_dy  # Inverted for OpenGL (Y+ points UP)
        nz = np.ones_like(dZ_dx)
        
        normals_v = np.stack([nx, ny, nz], axis=-1)
        
        # FIX FOR STRIPES & NOISE:
        # Apply a spatial 8x5 median filter (8 vertical, 5 horizontal) across all channels.
        # This removes hallucinated stripes and stray pixels while preserving sharp geometric edges.
        from scipy.ndimage import median_filter
        normals_v = median_filter(normals_v, size=(8, 5, 1))
        
        # Normalize
        norm = np.linalg.norm(normals_v, axis=2, keepdims=True)
        normals_v = normals_v / (norm + 1e-8)
        
        # Ensure Z is always positive
        normals_v[..., 2] = np.abs(normals_v[..., 2])
        
        return normals_v

    def _generate_normals(self, image_np, high_res=False, normals_method="Marigold", normals_ensemble=3, normals_scale=1.0, depth_map=None, progress_callback=None, tile_progress_callback=None):
        if "Hybrid" in normals_method:
            if progress_callback: progress_callback("Calibrating depth normal scale using Marigold reference...")
            
            # Step 1: Base normals from Marigold (low res for speed and as a structural reference)
            base_normals = self._generate_normals(image_np, high_res=False, normals_method="Marigold", normals_ensemble=normals_ensemble, progress_callback=progress_callback, tile_progress_callback=tile_progress_callback)
            
            import cv2
            
            orig_h, orig_w = image_np.shape[:2]
            base_h, base_w = base_normals.shape[:2]
            
            # Step 2: Depth map
            if depth_map is None:
                depth_map = self._generate_depth(image_np, high_res, progress_callback, tile_progress_callback=tile_progress_callback)
            depth_f = depth_map.astype(np.float32)
            
            # Match resolutions for calibration
            depth_low = cv2.resize(depth_f, (base_w, base_h), interpolation=cv2.INTER_AREA)
            
            # Blur both to ignore high frequency noise and focus on the main volume
            blur_rad = max(5, int(min(base_w, base_h) * 0.05) | 1)
            base_n_blurred = cv2.GaussianBlur(base_normals, (blur_rad, blur_rad), 0)
            depth_blurred = cv2.GaussianBlur(depth_low, (blur_rad, blur_rad), 0)
            
            # Compute Marigold normal vectors in [-1, 1]
            base_vec = base_n_blurred * 2.0 - 1.0
            
            # Calculate Marigold's average surface tilt (bumpiness) on the foreground
            foreground_mask = depth_blurred > (np.max(depth_blurred) * 0.1)
            if np.sum(foreground_mask) < 100:
                foreground_mask = np.ones_like(depth_blurred, dtype=bool)
                
            marigold_tilt = np.sqrt(base_vec[..., 0]**2 + base_vec[..., 1]**2)
            target_bumpiness = np.mean(marigold_tilt[foreground_mask])
            
            # Compute depth gradients
            height_map_low = depth_blurred ** 2
            dZ_dx = cv2.Sobel(height_map_low, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            dZ_dy = cv2.Sobel(height_map_low, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            
            best_scale = 1500.0
            min_error = float('inf')
            
            # Search for best bump scale (logarithmic search from 50 to 5000)
            scales_to_test = np.logspace(np.log10(50), np.log10(5000), 40)
            
            import time
            if progress_callback: progress_callback("Calibrating ideal bump scale (testing 40 variations)...")
            
            for i, test_scale in enumerate(scales_to_test):
                # Fake a tiny delay so the user sees the progress bar move (UX improvement)
                time.sleep(0.02)
                _notify_tile_progress(tile_progress_callback, i + 1, len(scales_to_test), "Surface Normal Scale Calibration", 1)
                    
                nx = -dZ_dx * test_scale
                ny = dZ_dy * test_scale
                
                # Normalize
                norm = np.sqrt(nx**2 + ny**2 + 1.0)
                nx /= norm
                ny /= norm
                
                # Calculate the average tilt of these depth normals
                depth_tilt = np.sqrt(nx**2 + ny**2)
                current_bumpiness = np.mean(depth_tilt[foreground_mask])
                
                error = abs(current_bumpiness - target_bumpiness)
                
                if error < min_error:
                    min_error = error
                    best_scale = test_scale
            
            # Scale down autoscale by 0.5 to make bump softer than raw calculation
            best_scale = best_scale * 0.5
            print(f"Auto-calibrated bump_scale: {best_scale}")
            if progress_callback: progress_callback(f"Generating optimized depth normal map (Auto-Scale: {best_scale:.1f})...")
            
            # Use the mathematically optimal scale directly on the full-res depth map
            final_bump_scale = best_scale * normals_scale
            normals_v = self._depth_to_normals_3d(depth_f, normals_scale=1.0, override_bump_scale=final_bump_scale)
            
            return ((normals_v + 1.0) * 0.5).clip(0, 1).astype(np.float32)
            
        elif normals_method == "From Depth Map":
            if progress_callback: progress_callback("Calculating Normals from Depth Map...")
            if depth_map is None:
                depth_map = self._generate_depth(image_np, high_res, progress_callback)
            
            depth_f = depth_map.astype(np.float32)
            normals_v = self._depth_to_normals_3d(depth_f, normals_scale)
            
            # Convert float32 [-1, 1] vectors to [0, 1] for saving as image
            return ((normals_v + 1.0) * 0.5).clip(0, 1).astype(np.float32)

        if normals_method == "IC-Light":
            return self._generate_normals_iclight(image_np, high_res, normals_ensemble, progress_callback, tile_progress_callback=tile_progress_callback)
            
        import torch
        from diffusers import MarigoldNormalsPipeline
        
        if 'marigold_normals' not in self.models:
            model_path = _get_local_model_path("prs-eth/marigold-normals-v1-1")
            
            # Check if Marigold weights need to be downloaded from Hugging Face
            is_marigold_cached = False
            try:
                from pathlib import Path
                from huggingface_hub.constants import HF_HUB_CACHE
                folder_name = "models--" + "prs-eth/marigold-normals-v1-1".replace("/", "--")
                cache_dir = Path(HF_HUB_CACHE) / folder_name / "snapshots"
                if cache_dir.exists():
                    for snap in cache_dir.iterdir():
                        if (snap / "unet").exists() or (snap / "model_index.json").exists():
                            is_marigold_cached = True
                            break
            except Exception:
                pass

            if not is_marigold_cached and not os.path.exists(model_path):
                if progress_callback:
                    progress_callback("Downloading Marigold Normals weights (~1.4 GB) from Hugging Face... (one-time download)")
            else:
                if progress_callback:
                    progress_callback("Loading Marigold Normals Pipeline...")

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                variant = "fp16" if device == "cuda" else None
                
                pipe = None
                import time
                
                for attempt in range(3):
                    try:
                        pipe = MarigoldNormalsPipeline.from_pretrained(
                            model_path,
                            variant=variant,
                            torch_dtype=dtype
                        ).to(device)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(3)
            except Exception as e:
                import traceback
                err_path = os.path.join(tempfile.gettempdir(), "ai_error_marigold.txt")
                try:
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(traceback.format_exc())
                except Exception:
                    pass
                raise
                
            self.models['marigold_normals'] = pipe
            
        pipe = self.models['marigold_normals']
        
        if progress_callback: progress_callback(f"Generating Normal Map (High Res: {high_res})...")
        
        if high_res:
            # Enable VAE tiling to prevent OOM on 4K images
            try:
                if hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()
                elif hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
                    pipe.vae.enable_tiling()
            except Exception as e:
                print(f"Warning: Could not enable VAE tiling: {e}")
            
        pil_img = Image.fromarray(image_np)
        
        def process_image(img, esize):
            out = pipe(
                img,
                num_inference_steps=4,
                ensemble_size=esize,
                match_input_resolution=True
            )
            # Extract raw float32 normals directly from the prediction instead of the 8-bit visualizer
            pred = out.prediction
            if isinstance(pred, np.ndarray):
                # Diffusers Marigold output is often (B, 3, H, W) or (B, H, W, 3)
                if pred.shape[1] == 3:
                    norm_f = pred[0].transpose(1, 2, 0).astype(np.float32)
                else:
                    norm_f = pred[0].astype(np.float32)
            else:
                # If torch tensor
                pred = pred.cpu().numpy()
                if pred.shape[1] == 3:
                    norm_f = pred[0].transpose(1, 2, 0).astype(np.float32)
                else:
                    norm_f = pred[0].astype(np.float32)
            
            return norm_f

        if not high_res:
            result_img = process_image(pil_img, normals_ensemble)
        else:
            if progress_callback: progress_callback(f"Tiling normal map (Laplacian PatchFusion, Scale: {normals_scale}x)...")
            H_orig, W_orig = image_np.shape[:2]
            H, W = H_orig, W_orig
            
            if normals_scale != 1.0:
                new_w, new_h = int(W_orig * normals_scale), int(H_orig * normals_scale)
                image_np = np.array(Image.fromarray(image_np).resize((new_w, new_h), Image.LANCZOS))
                H, W = new_h, new_w
                pil_img = Image.fromarray(image_np)
                
            max_global_size = 1024
            global_img_pil = pil_img.copy()
            if max(H, W) > max_global_size:
                scale = max_global_size / max(H, W)
                new_w, new_h = int(W * scale), int(H * scale)
                global_img_pil = global_img_pil.resize((new_w, new_h), Image.LANCZOS)
            
            global_normals = process_image(global_img_pil, max(1, normals_ensemble - 1))
            global_normals = np.array(Image.fromarray(global_normals.astype(np.uint8)).resize((W, H), Image.BILINEAR)).astype(np.float32)
            
            patch_size = 1024
            overlap = 256
            stride = patch_size - overlap
            
            def create_blend_window(size, overlap_size):
                w = np.ones(size, dtype=np.float32)
                ramp = (1.0 - np.cos(np.linspace(0, np.pi, overlap_size, endpoint=True, dtype=np.float32))) / 2.0
                w[:overlap_size] = ramp
                w[-overlap_size:] = ramp[::-1]
                return w
            
            y_1d = create_blend_window(patch_size, overlap)
            x_1d = create_blend_window(patch_size, overlap)
            w_patch = np.outer(y_1d, x_1d).astype(np.float32)
            w_patch = np.expand_dims(w_patch, axis=-1)
            
            # Start with a very low-weight global normal map to fill gaps
            # The global map provides the low-frequency structure
            global_img_pil = Image.fromarray(image_np).resize((1024, int(1024 * H/W)), Image.LANCZOS)
            global_normals = process_image(global_img_pil, normals_ensemble)
            import cv2
            global_normals = cv2.resize(global_normals, (W, H), interpolation=cv2.INTER_CUBIC)
            
            final_normals = global_normals * 0.1
            weight_sum = np.ones((H, W, 1), dtype=np.float32) * 0.1
            
            y_steps = list(range(0, H - patch_size + 1, stride))
            if H % stride != 0 or H < patch_size: y_steps.append(max(0, H - patch_size))
            x_steps = list(range(0, W - patch_size + 1, stride))
            if W % stride != 0 or W < patch_size: x_steps.append(max(0, W - patch_size))
            
            y_steps = sorted(list(set(y_steps)))
            x_steps = sorted(list(set(x_steps)))
            
            total_tiles = len(y_steps) * len(x_steps)
            current_tile = 0
            
            import psutil
            import concurrent.futures
            from threading import Lock
            import os
            import torch
            
            # Dynamic thread calculation to protect RAM
            RAM_PER_THREAD_GB = 1.5
            mem = psutil.virtual_memory()
            safe_available = mem.available - (mem.total * 0.15) # 15% safety margin
            allowed_threads = max(1, int(safe_available / (RAM_PER_THREAD_GB * 1024 * 1024 * 1024)))
            max_workers = min(allowed_threads, os.cpu_count() or 4)
            if torch.cuda.is_available():
                max_workers = min(max_workers, 2)
                
            if progress_callback: progress_callback(f"Tiling normal map (Laplacian PatchFusion): 0 / {total_tiles} ({max_workers} threads)...")
            
            write_lock = Lock()
            completed_tiles = 0
            
            def process_normals_tile(y, x):
                actual_h = min(patch_size, H - y)
                actual_w = min(patch_size, W - x)
                patch = Image.fromarray(image_np[y:y+actual_h, x:x+actual_w])
                local_normals = process_image(patch, normals_ensemble)
                return y, x, actual_h, actual_w, local_normals
                
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for y in y_steps:
                    for x in x_steps:
                        futures.append(executor.submit(process_normals_tile, y, x))
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        y, x, actual_h, actual_w, local_normals = future.result()
                        with write_lock:
                            final_normals[y:y+actual_h, x:x+actual_w] += local_normals * w_patch[:actual_h, :actual_w]
                            weight_sum[y:y+actual_h, x:x+actual_w] += w_patch[:actual_h, :actual_w]
                            completed_tiles += 1
                            if tile_progress_callback:
                                tile_progress_callback(completed_tiles, total_tiles)
                            if completed_tiles % 5 == 0 or completed_tiles == total_tiles:
                                if progress_callback: 
                                    progress_callback(f"Tiling normal map: Processing tile {completed_tiles} / {total_tiles}...")
                    except Exception as e:
                        print(f"Error processing normal tile: {e}")
                        
            result_img = final_normals / weight_sum
            
            # Normalize vectors geometrically
            norm = np.linalg.norm(result_img, axis=2, keepdims=True)
            result_img = result_img / (norm + 1e-8)
            
        # Convert float32 [-1, 1] normals to float32 [0, 255] for consistency if needed, 
        # or leave as is if we expect uint8 output?
        # Wait, the rest of the code expects uint8 [0, 255] output for Marigold branch?
        # If Hybrid uses it: base_normals_8bit.astype(np.float32) / 255.0 * 2.0 - 1.0.
        # So we MUST return a uint8 [0, 255] image UNLESS we refactor Hybrid.
        # It's better to just return the true float32 if the method expects it!
        
        # Normalize to [0, 255] float32 instead of uint8 to preserve sub-pixel accuracy
        result_img = ((result_img + 1.0) / 2.0 * 255.0).astype(np.float32)
        
        if normals_scale != 1.0 or (high_res == False and result_img.shape[:2] != (image_np.shape[0], image_np.shape[1])):
            import cv2
            H_orig, W_orig = image_np.shape[:2]
            result_img = cv2.resize(result_img, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)

        if high_res:
            try:
                if hasattr(pipe, 'disable_vae_tiling'):
                    pipe.disable_vae_tiling()
                elif hasattr(pipe, 'vae') and hasattr(pipe.vae, 'disable_tiling'):
                    pipe.vae.disable_tiling()
            except Exception:
                pass

        return result_img / 255.0
