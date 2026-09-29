# -*- mode: python ; coding: utf-8 -*-
import sys
import os

block_cipher = None

# Base directory (In PyInstaller spec files, SPECPATH is provided by PyInstaller)
BASE_DIR = os.path.abspath(SPECPATH) if 'SPECPATH' in globals() else os.path.abspath(os.getcwd())

added_datas = [
    (os.path.join(BASE_DIR, 'src', 'shaders'), 'shaders'),
    (os.path.join(BASE_DIR, 'src', 'shaders'), os.path.join('src', 'shaders')),
    (os.path.join(BASE_DIR, 'src', 'plugins'), os.path.join('src', 'plugins')),
    (os.path.join(BASE_DIR, 'assets'), 'assets'),
]

# Neural network weights (NAFNet, DepthAnything, BiRefNet) are loaded on-demand from cache/HuggingFace!
# No heavy .pth model weights are bundled into the installer to minimize package size.

# Include package metadata (.dist-info) required by Hugging Face diffusers / transformers / requests
from PyInstaller.utils.hooks import copy_metadata, collect_data_files, collect_submodules

required_metadata_pkgs = [
    'requests',
    'diffusers',
    'transformers',
    'torch',
    'torchvision',
    'huggingface_hub',
    'accelerate',
    'safetensors',
    'tokenizers',
    'regex',
    'packaging',
    'filelock',
    'numpy',
    'rembg',
    'onnxruntime',
    'scipy',
    'Pillow',
]

for pkg in required_metadata_pkgs:
    try:
        added_datas += copy_metadata(pkg)
    except Exception:
        pass

for pkg in ['diffusers', 'transformers']:
    try:
        added_datas += collect_data_files(pkg)
    except Exception:
        pass

# Hidden imports for dynamic/lazy-loaded modules
hidden_imports = [
    'requests',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'moderngl',
    'glcontext',
    'rawpy',
    'tifffile',
    'imageio',
    'lensfunpy',
    'pyexiv2',
    'PIL',
    'PIL.Image',
    'torch',
    'torch.distributed',
    'torch.testing',
    'torchvision',
    'torchvision.ops',
    'torchvision.transforms',
    'torchvision.io',
    'transformers',
    'diffusers',
    'accelerate',
    'rembg',
    'onnxruntime',
    'fastmcp',
    'mcp',
    'starlette',
    'uvicorn',
    'unittest',
    'unittest.mock',
    'google.genai',
    'pydantic',
    'scipy',
    'scipy.ndimage',
    'scipy.signal',
    'numpy',
    'plugins.builtin.basic_adjustments',
    'plugins.builtin.tone_equalizer',
    'plugins.builtin.color_equalizer',
    'plugins.builtin.shadow_color_eq',
]

# Ensure ALL model submodules and dynamic dependencies for AI features are compiled into PYZ
for model_pkg in [
    'transformers.models.depth_anything',
    'transformers.models.segformer',
    'transformers.models.dpt',
    'transformers.models.dinov2',
    'transformers.models.clip',
    'transformers.models.vit',
    'transformers.models.vit_mae',
    'transformers.models.siglip',
    'transformers.models.auto',
    'diffusers.pipelines.marigold',
    'diffusers.models',
    'diffusers.schedulers',
    'torchvision',
    'rembg',
    'accelerate',
    'huggingface_hub',
]:
    try:
        hidden_imports += collect_submodules(model_pkg)
    except Exception as e:
        print(f"[WARN] Failed to collect submodules for {model_pkg}: {e}")

a = Analysis(
    ['main.py'],
    pathex=[BASE_DIR, os.path.join(BASE_DIR, 'src')],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'IPython',
        'notebook',
        'pytest',
        'imageio_ffmpeg',
        'av',
        'torchaudio',
        'pandas',
        'scipy.tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Security and Optimization Filter:
# Exclude credentials (.env, secrets), unused external binaries, tests, and heavy unused components
excluded_binaries_names = {
    'ffmpeg-win-x86_64-v7.1.exe',
    'opencv_videoio_ffmpeg500_64.dll',
    'cusolvermg64_11.dll',
}

def should_keep_binary(bin_tuple):
    dest_name = bin_tuple[0]
    basename = os.path.basename(dest_name).lower()
    if basename in excluded_binaries_names:
        return False
    dest_lower = dest_name.lower().replace('\\', '/')
    if 'imageio_ffmpeg' in dest_lower or 'imageio-ffmpeg' in dest_lower:
        return False
    if 'av.libs' in dest_lower or '/av/' in dest_lower:
        return False
    return True

a.binaries = [b for b in a.binaries if should_keep_binary(b)]

def should_keep_data(data_tuple):
    dest_name = data_tuple[0]
    dest_lower = dest_name.lower().replace('\\', '/')
    base_lower = os.path.basename(dest_lower)
    
    # 1. Security: Strictly purge any .env, secrets, or credential files
    if base_lower.startswith('.env') or base_lower.endswith('.key') or base_lower.endswith('.token') or base_lower.endswith('.secret'):
        return False
        
    # 2. Exclude tests and test suites (keep torch internal modules like torch.testing required by autograd)
    parts = dest_lower.split('/')
    if any(p in ['tests', 'testing', 'test'] for p in parts) and 'torch' not in dest_lower:
        return False
        
    # 3. Exclude video tools, ffmpeg, av
    if 'imageio_ffmpeg' in dest_lower or 'imageio-ffmpeg' in dest_lower or 'av.libs' in dest_lower:
        return False
        
    # 4. Exclude models folder if present (all weights downloaded on demand)
    if parts[0] == 'models' or 'blaze_face.tflite' in base_lower:
        return False
        
    return True

a.datas = [d for d in a.datas if should_keep_data(d)]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LumensDeep',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(BASE_DIR, 'assets', 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='LumensDeep',
)
