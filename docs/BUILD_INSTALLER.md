# Building the Lumens Deep Windows Installer (Setup .exe)

This guide explains how to package the **Lumens Deep** Python application into a standalone Windows installer (`LumensDeep_Setup_v0.1_Alpha_x64.exe`).

Any user can install and run the application without needing to install Python, configure environments, or open a command prompt. The installer automatically creates shortcuts on the **Desktop** and in the **Start Menu**.

---

## 🏗️ How It Works (Architecture)

The build process is based on the industry standard for packaging desktop applications on Windows:

```
[Lumens Deep Source Code]
         │
         ▼  (Stage 1: PyInstaller)
[Standalone Directory: dist\LumensDeep\]
├── LumensDeep.exe (Main application executable)
├── python311.dll, PySide6, PyTorch, ModernGL
├── shaders/ (GLSL runtime shaders)
└── assets/ (Application icon)
         │
         ▼  (Stage 2: Inno Setup)
[LumensDeep_Setup_v0.1_Alpha_x64.exe]
         │
         ▼  (End-user runs Setup)
- Installs to Program Files or LocalAppData
- Creates shortcuts on Desktop and in Start Menu
- Registers clean uninstaller in Windows Settings > Installed Apps
- 1-click launch!
```

---

## 📋 Prerequisites

1. **Python 3.10 or 3.11** with project dependencies installed (`requirements.txt`).
2. **PyInstaller**:
   ```bash
   pip install pyinstaller
   ```
3. **Inno Setup 6** (free, open-source Windows installer compiler):
   - Download: [https://jrsoftware.org/isdl.php](https://jrsoftware.org/isdl.php)
   - Install with standard defaults.

---

## 🚀 One-Click Build

A ready-to-run automation script `build_installer.bat` is available in the repository root.

1. Double-click **`build_installer.bat`** (or execute it via terminal).
2. The script will automatically:
   - Check for `pyinstaller` (and install it if missing).
   - Clean up previous temporary build artifacts.
   - Freeze the application into `dist\LumensDeep\` using `lumensdeep.spec`.
   - Locate the installed Inno Setup compiler (`ISCC.exe`) and compile the installer.
3. The final installer executable will be generated in:
   ```
   output\LumensDeep_Setup_v0.1_Alpha_x64.exe
   ```

---

## 🔒 Security & API Key Protection (.env)

The `.env` file contains your private credentials (such as `GEMINI_API_KEY`). **Four distinct defense-in-depth security layers** are implemented to guarantee that sensitive files are **never** bundled or distributed to end users:

1. **PyInstaller Spec ([lumensdeep.spec](file:///e:/3dcoat/GitHub/LumensDeep/lumensdeep.spec))**: Analyzes and purges bundled data (`a.datas`), strictly discarding any file matching `.env`, `*.env.*`, `.key`, or `.token`.
2. **Inno Setup Script ([installer.iss](file:///e:/3dcoat/GitHub/LumensDeep/installer.iss))**: Includes the directive `Excludes: "*.env,*.env.*,.env,*.key,*.token,*.secret"`, preventing Inno Setup from ever packing these files into the setup wizard.
3. **Build Script ([build_installer.bat](file:///e:/3dcoat/GitHub/LumensDeep/build_installer.bat))**: Before Inno Setup runs, an automated scan checks `dist\LumensDeep\` and immediately purges any stray `.env` or credential files.
4. **Git Protection ([.gitignore](file:///e:/3dcoat/GitHub/LumensDeep/.gitignore))**: `.env` files as well as `dist/`, `build/`, and `output/` directories are permanently excluded from Git commits.

After installation, users can optionally enter their own Gemini API key directly in the **AI Assistant Dock** inside the application GUI.

---

## 🛠️ Configuration Files in the Repository

- **`main.py`** — Primary entry point that configures the Windows AppUserModelID for proper taskbar grouping, initializes application styling, and resolves bundled resources.
- **`lumensdeep.spec`** — PyInstaller specification:
  - Bundles GLSL shaders (`shaders/`), plugins, and application icons.
  - Declares hidden imports for PySide6, ModernGL, PyTorch, rawpy, lensfunpy, etc.
  - Hides the terminal window (`console=False`).
  - Utilizes `--onedir` mode for instant startup without extracting gigabytes to `%TEMP%` on each launch.
- **`installer.iss`** — Inno Setup script:
  - Supports English and Ukrainian setup wizard languages.
  - Automatically creates a Desktop shortcut (`{autodesktop}\Lumens Deep.lnk`).
  - Automatically registers the application in the Start Menu (`{autoprograms}\Lumens Deep`).
  - Creates a clean uninstaller registered in Windows Settings / Control Panel.
  - Supports both system-wide (`Program Files`) and per-user (`LocalAppData`) installations without requiring administrator privileges.
  - Prompts to launch the application immediately upon setup completion.

---

## 💡 Size Optimization Tips

1. **PyTorch with CUDA Footprint**:
   - PyTorch GPU/CUDA wheels are relatively large (~2–4 GB).
   - Inno Setup uses **LZMA2 Ultra** compression, typically reducing the final installer size by 2–3×.
2. **CPU-Only Lightweight Build**:
   - If a lightweight package is preferred for machines without dedicated NVIDIA GPUs, build within a virtual environment containing CPU-only PyTorch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`). This reduces the installer size significantly.
