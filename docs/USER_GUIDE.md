# Lumens Deep User Guide & Manual

Welcome to **Lumens Deep**, a next-generation, non-destructive 32-bit floating-point RAW photo editor and AI-powered 3D relighting suite.

---

## 1. Getting Started

### 1.1. System Requirements & Installation
- **Operating System**: Windows 10/11 (64-bit), macOS, or Linux.
- **GPU**: OpenGL 3.3+ capable dedicated or integrated GPU.
- **Python**: Python 3.10, 3.11, or 3.12+.

Install dependencies via pip:
```bash
pip install -r requirements.txt
```

Launch the application:
```bash
python src/gui/main_window.py
```

### 1.2. Workspace Overview
The workspace is organized into intuitive, high-productivity dock panels:

![Workspace Overview](images/01_workspace_overview.png)

```
┌─────────────────┬───────────────────────────────────────────┬─────────────────┐
│  LEFT PANELS    │              CENTRAL VIEWPORT             │  RIGHT PANELS   │
│  [Folders]      │  [ Top Toolbar: Zoom / Crop / Compare ]   │  [Layers]       │
│  [Global & 3D]  │  ───────────────────────────────────────  │  [Layer Effects]│
│  [AI Assistant] │             Interactive Canvas            │  [Masks]        │
├─────────────────┴───────────────────────────────────────────┴─────────────────┤
│                        BOTTOM PANEL: FILMSTRIP DOCK                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Left Panels**:
  - **Folders**: Browse directories and subfolders.
  - **Global Settings**: Lensfun distortion profiles, 3D Bokeh simulation, and global base curve output.
  - **AI Assistant**: Compute neural depth maps, high-frequency surface normals, and subject alpha masks.
- **Central Viewport**: High-DPI canvas with GPU zoom, pan, interactive crop/straighten HUD, split before/after wipe, and brush painting.
- **Right Panels**:
  - **Layers**: Non-destructive layer stack (add, duplicate, delete, reorder, adjust opacity).
  - **Layer Effects**: Active layer adjustment plugins (Basic Adjustments, Tone Equalizer, Color Equalizer, Shadow Color Equalizer).
  - **Masks**: Precision masking tools (Parametric OKLCH Color/Luma, Linear/Radial Gradients, 3D Depth/Normals, Painted Brush Mask).
- **Bottom Panel (Filmstrip)**: Async multi-threaded thumbnail browser with caching and instant image switching.

---

## 2. Working with Images & RAW Files

### 2.1. Supported File Formats
Lumens Deep supports all major camera RAW formats and standard image formats via `rawpy` and `imageio`:
- **RAW Sensor Formats**: Canon (`.CR3`, `.CR2`), Nikon (`.NEF`), Sony (`.ARW`), Fujifilm (`.RAF`), Panasonic/Leica (`.RW2`), Adobe Digital Negative (`.DNG`).
- **Standard Image Formats**: `.TIFF`, `.PNG`, `.JPG`, `.JPEG`, `.WEBP`, `.EXR`.

### 2.2. Folder Navigation & Filmstrip
1. In the **Folders** dock on the left, navigate and click any folder containing images.
2. The bottom **Filmstrip** dock will asynchronously generate and display high-quality thumbnails.
3. Click any thumbnail on the filmstrip to open and decode the 32-bit linear floating-point RAW data in the viewport.

---

## 3. Viewport & Canvas Navigation

### 3.1. Pan & Zoom
- **Zoom**: Scroll the **Mouse Wheel** up/down, or use `Ctrl + +` / `Ctrl + -`.
- **Pan**: Hold **Left Click and Drag** (when brush tool is inactive) or **Middle Mouse Drag**.
- **Fit View**: Click `Fit View` on the toolbar or press `Ctrl + 0`.
- **100% (1:1 Pixel) View**: Click `1:1 View` on the toolbar or press `Ctrl + 1`.

### 3.2. Interactive Crop & Straighten
1. Click the **`📐 Crop & Straighten`** button on the viewport toolbar.
2. The canvas enters interactive crop mode with a Rule-of-Thirds overlay.
3. **Crop Handles**:
   - Drag any of the **8 corner L-brackets** or **4 midpoint edge tabs** to adjust the crop box.
4. **Rotation & Straighten**:
   - Hover slightly outside the corner brackets (the cursor changes to a curved rotation arrow) and drag to rotate the crop angle interactively.
   - Alternatively, adjust the **`Rotation`** slider in the Global dock.
5. Click **`Apply`** to confirm or uncheck the crop button to exit.

![Interactive Crop HUD](images/03_crop_and_straighten_hud.png)

### 3.3. Before / After & Split Comparison Modes
Compare adjustments against the unprocessed RAW image:
- **`👁 Before Mode`**: Press and hold or toggle to view the original untouched image.
- **`◧ Vertical Split`**: Left side shows original image, right side shows processed output. Drag the white divider line horizontally across the canvas.
- **`⬓ Horizontal Split`**: Top side shows original image, bottom side shows processed output. Drag the divider line vertically.
- **`◫ Side-by-Side`**: Dual view showing Before and After in split halves.

![Split Wipe Comparison](images/04_split_wipe_comparison.png)

---

## 4. Non-Destructive Layer System

Lumens Deep uses a layer-based workflow:

1. **Base Layer**: Contains the global baseline adjustments applied to the whole photo.
2. **Adjustment Layers**: Add local adjustment layers by clicking **`+ New Layer`** in the Layers dock.
3. **Layer Opacity**: Control the blend strength of each layer independently via the opacity slider.
4. **Reordering & Duplication**: Duplicate or delete layers without altering underlying image data.

![Layers Stack](images/05_layer_stack_dock.png)

---

## 5. Built-in Adjustment Plugins

Each layer can run any combination of modular GPU-accelerated effect plugins:

### 5.1. ☀️ Basic Adjustments
- **Exposure**: Precise exposure compensation from $-3.0\text{ EV}$ to $+3.0\text{ EV}$.
- **Contrast & Gamma**: Adjust dynamic range and midtone tonal response.
- **White Balance**: Temperature (Cool $\leftrightarrow$ Warm) and Tint (Green $\leftrightarrow$ Magenta).
- **Highlights & Shadows Recovery**: Tame blown-out skies and lift deep shadow details.
- **Whites & Black Level**: Set pure white point and black clipping threshold.
- **Local Contrast & Clarity**: Enhance micro-contrast around textures and edges.
- **Global Vibrance & Chroma**: Boost color saturation in perceptual OKLCH color space.
- **Denoise (Luma & Chroma)**: Bilateral filter noise reduction in YCbCr color channels.
- **Sharpening**: High-pass detail sharpening with threshold control.
- **Vignette**: Add artistic darkening or lightening toward photo borders.

![Basic Adjustments Plugin](images/06_basic_adjustments_plugin.png)

### 5.2. 📊 Tone Equalizer
- Features an interactive **8-band OKLCH curve editor**.
- Allows targeted brightening or darkening of specific luminance zones (from deep blacks at $-8\text{ EV}$ to specular highlights at $+1\text{ EV}$) without causing clipping or color shifts.

![Tone Equalizer Spline](images/07_tone_equalizer_spline.png)

### 5.3. 🎨 Color Equalizer & 🌓 Shadow Color Equalizer
- Features an 8-band hue-based equalizer for selective color grading.
- Adjust Hue Shift, Saturation Scale, and Lightness Offset independently across 8 color ranges (Reds, Oranges, Yellows, Greens, Cyans, Blues, Purples, Magentas).
- **Shadow Color Equalizer**: Specialized 8-band color equalizer that only applies inside the shadow and mid-shadow regions, allowing cinematic split-toning and shadow tinting.

![Color Equalizer Cards](images/08_color_equalizer_cards.png)

---

## 6. Precision Masking System

Control exactly where each adjustment layer takes effect:

### 6.1. 🖌️ GPU Brush Painting
1. Select a layer and activate the **`🖌 Brush`** tool on the toolbar.
2. Adjust **Brush Size**, **Edge Hardness**, and **Opacity** in the brush settings bar.
3. **Draw on Canvas**: Left-click and drag to paint the mask. Lumens Deep features GPU sub-pixel stroke interpolation for smooth, uninterrupted lines during fast mouse sweeps.
4. **Erase Mode**: Hold **Right Click** or check the `Erase` option to erase parts of the mask.
5. **Blur Mask**: Soften hard mask boundaries under the cursor.

![GPU Brush Painting](images/09_brush_painting_action.png)

### 6.2. 🎯 Parametric Color & Luminance Masking
- **Color Range (OKLCH Hue)**: Pick or select target hues to isolate specific colors (e.g. skin tones, sky, foliage).
- **Saturation Range**: Filter by color saturation.
- **Luminance Range**: Target specific brightness levels.
- **Softness**: Adjust smooth transition falloff.

### 6.3. 📐 Gradient Masks
- Types: Linear (Left-to-Right, Right-to-Left, Top-to-Bottom, Bottom-to-Top) and Radial.
- Black Point, White Point, and Midtone Gamma controls.

### 6.4. 🧊 3D Geometric Masking
- **Depth Mask**: Isolate foreground, middle ground, or background based on estimated 3D depth intervals.
- **Surface Normal Mask**: Isolate surfaces facing specific 3D directions (e.g. surfaces facing upward for snow/dust or facing left for directional relighting).

![Masks Dock and 3D Masks](images/10_mask_dock_and_3d_masks.png)

---

## 7. 3D & AI Neural Tools

### 7.1. Automatic LumenDeep Map Discovery
When opening an image `photo.CR3`, the editor automatically scans for pre-computed 3D maps in `photo_dir/LumenDeep/`:
- `photo_depth.png` $\rightarrow$ Metric/Relative Depth Map
- `photo_normals.png` $\rightarrow$ Surface Normal Map
- `photo_subject.png` $\rightarrow$ Subject Alpha Mask

### 7.2. Generating AI Maps
Under the **`AI & 3D Tools`** menu or the **AI Assistant Dock**:
- **Compute Depth Map (`Ctrl+Shift+D`)**: Generates high-resolution depth map.
- **Compute Normal Map (`Ctrl+Shift+N`)**: Generates high-frequency 3D surface orientations.
- **Compute Subject Mask (`Ctrl+Shift+S`)**: Extracts subject foreground segmentation.
- **Compute All AI Maps (`Ctrl+Shift+A`)**: Runs the full neural inference queue in the background with async progress reporting.

![AI Neural Maps Showcase](images/11_ai_maps_4quadrant_showcase.png)

### 7.3. ✨ 3D Bokeh Blur Simulation
Transform flat photos into shallow depth-of-field captures with cinematic optical simulation:
- **Enable Bokeh Simulation**: Activates progressive Monte Carlo depth-of-field raymarching.
- **Focal Distance**: Move right (towards 100) to shift focus deeper into the background; move left (towards 0) to focus on the foreground.
- **Focal Range**: Adjusts the thickness of the in-focus plane (Depth of Field).
- **Max Radius**: Controls blur disc size.
- **Aperture Blades**:
  - `0`: 3 Blades (Triangle)
  - `1`: 4 Blades (Square)
  - `2`: 5 Blades (Pentagon)
  - `3`: 6 Blades (Hexagon)
  - `4`: 7 Blades (Heptagon)
  - `5`: 8 Blades (Octagon)
  - `10`: Circular (Smooth lens)
- **Optical Vignette (Cat-Eye)**: Simulates realistic lens pupil truncation towards frame corners.
- **Chromatic Aberration**: Adds radial color fringing in out-of-focus highlights.
- **Highlight Boost**: Amplifies bokeh balls in specular regions.
- **Swirly Bokeh**: Simulates vintage Petzval lens radial distortion.

![3D Bokeh Simulation Effect](images/12_bokeh_simulation_effect.png)

---

## 8. Exporting & Sidecar Saving

- **Save Project / Sidecar (`Ctrl+S`)**: Saves all layers, masks, curve points, crop geometry, and plugin settings to a JSON sidecar file (`.json`) alongside the photo.
- **Export Image (`Ctrl+E`)**: Renders full-resolution image to 16-bit TIFF, PNG, or high-quality JPEG with all non-destructive adjustments applied.

![16-Bit Export Dialog](images/13_export_dialog.png)

---

## 9. Keyboard Shortcuts Summary

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + O` | Open Image or RAW file |
| `Ctrl + S` | Save Adjustments / Sidecar |
| `Ctrl + E` | Export Processed Image |
| `Ctrl + 0` | Fit to Viewport |
| `Ctrl + 1` | Zoom to 100% (1:1 View) |
| `Ctrl + +` / `Ctrl + -` | Zoom In / Zoom Out |
| `Ctrl + Z` / `Ctrl + Y` | Undo / Redo |
| `Ctrl + Shift + D` | Compute AI Depth Map |
| `Ctrl + Shift + N` | Compute AI Surface Normal Map |
| `Ctrl + Shift + S` | Compute AI Subject Mask |
| `Ctrl + Shift + A` | Compute All AI Maps |
| `D` | Toggle Depth Map Visualizer |
| `N` | Toggle Normal Map Visualizer |
| `S` | Toggle Subject Mask Visualizer |
| `M` | Toggle Layer Mask Overlay |
| `B` | Toggle Brush Mask Painting Tool |
| `C` | Toggle Interactive Crop & Straighten |
| `\` | Toggle Before / After View |
