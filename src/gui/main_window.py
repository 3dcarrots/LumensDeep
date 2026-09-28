import os
import sys
import threading
import json
import copy
from typing import Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFileDialog, QMessageBox, QMenu, QDockWidget, QDialog
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QSize, QThread
from PySide6.QtGui import QImage, QPixmap, QAction, QIcon

# Add src to path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(src_dir)
sys.path.insert(0, src_dir)

load_dotenv()
for env_candidate in [os.path.join(root_dir, ".env"), os.path.join(src_dir, ".env")]:
    if os.path.exists(env_candidate):
        load_dotenv(env_candidate, override=(os.environ.get("GEMINI_API_KEY", "") in ("", "your_api_key_here")))

from core.engine import PhotoEngine, Layer
from core.image_io import load_image_to_linear, export_image_from_linear, extract_raw_thumbnail_linear
from core.profiler import profiler
import mcp_server

from plugins.manager import PluginManager
from gui.theme import set_dark_theme
from gui.widgets.viewport import ImagePreviewLabel
from gui.widgets.toolbar import ViewportToolbar, BrushSettingsBar
from gui.docks.layer_effects_dock import LayerEffectsDock
from gui.docks.layers_dock import LayersDock
from gui.docks.mask_dock import MaskDock
from gui.docks.global_dock import GlobalDock
from gui.docks.ai_dock import AIDock
from gui.docks.filmstrip_dock import FilmstripDock
from gui.docks.folders_dock import FoldersDock
from gui.gemini_worker import GeminiAssistantWorker, GeminiQualityControlWorker, GeminiBokehVerifierWorker
from core.gemini_assistant import (
    blend_state_dicts, compute_state_delta, compute_state_deltas,
    parse_aspect_ratio_text, enforce_crop_aspect_ratio,
    sanitize_equalizer_ints, sanitize_equalizer_dict, sanitize_state_equalizers
)

class ImageLoaderThread(QThread):
    image_loaded = Signal(str, object, dict, object, object, object, int) # (path, linear_data, metadata, depth_arr, norm_arr, subj_arr, gen_id)
    load_error = Signal(str, str, int)
    status_updated = Signal(str, str, int)

    def __init__(self, file_path: str, generation_id: int, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.generation_id = generation_id
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def _load_ai_maps_background(self, image_path: str):
        depth_arr = None
        norm_arr = None
        subj_arr = None
        try:
            from PIL import Image
            dir_name = os.path.dirname(image_path)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            lumen_dir = os.path.join(dir_name, "LumenDeep")

            # 1. Depth
            for c in [
                os.path.join(lumen_dir, f"{stem}_depth.png"),
                os.path.join(lumen_dir, f"{stem}.depth.png"),
                os.path.join(dir_name, f"{stem}_depth.png"),
                os.path.join(lumen_dir, f"{stem}_depth.npy"),
            ]:
                if os.path.exists(c):
                    try:
                        depth_arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                        break
                    except Exception: pass

            # 2. Normals
            for c in [
                os.path.join(lumen_dir, f"{stem}_normals.png"),
                os.path.join(lumen_dir, f"{stem}.normals.png"),
                os.path.join(dir_name, f"{stem}_normals.png"),
                os.path.join(lumen_dir, f"{stem}_normals.npy"),
            ]:
                if os.path.exists(c):
                    try:
                        norm_arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                        break
                    except Exception: pass

            # 3. Subject
            for c in [
                os.path.join(lumen_dir, f"{stem}_subject.png"),
                os.path.join(lumen_dir, f"{stem}.subject.png"),
                os.path.join(dir_name, f"{stem}_subject.png"),
                os.path.join(lumen_dir, f"{stem}_subject.npy"),
            ]:
                if os.path.exists(c):
                    try:
                        subj_arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                        break
                    except Exception: pass
        except Exception:
            pass
        return depth_arr, norm_arr, subj_arr

    def run(self):
        try:
            self.status_updated.emit(self.file_path, "Parsing EXIF metadata...", self.generation_id)
            if self._is_cancelled:
                return

            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw']:
                self.status_updated.emit(self.file_path, "Demosaicing RAW sensor data...", self.generation_id)
            else:
                self.status_updated.emit(self.file_path, "Decoding image file...", self.generation_id)

            linear_data, metadata = load_image_to_linear(self.file_path)

            if ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw']:
                try:
                    from core.image_io import save_rhalf_jpeg
                    save_rhalf_jpeg(self.file_path, linear_data)
                except Exception:
                    pass

            if self._is_cancelled:
                return

            self.status_updated.emit(self.file_path, "Loading 3D & AI maps in background...", self.generation_id)
            depth_arr, norm_arr, subj_arr = self._load_ai_maps_background(self.file_path)

            if not self._is_cancelled:
                self.status_updated.emit(self.file_path, "Uploading textures to GPU...", self.generation_id)
                self.image_loaded.emit(self.file_path, linear_data, metadata, depth_arr, norm_arr, subj_arr, self.generation_id)
        except Exception as e:
            if not self._is_cancelled:
                self.load_error.emit(self.file_path, str(e), self.generation_id)

class AIMapWorkerThread(QThread):
    progress = Signal(str)
    tile_progress = Signal(int, int, str, int) # cur, total, stage, workers
    finished = Signal(str, object, str) # map_type, result_data, image_path
    error = Signal(str, str) # map_type, error_msg

    def __init__(self, image_np: np.ndarray, map_type: str, image_path: str, high_res: bool = True, cached_depth=None, normals_method: str = "Hybrid (Marigold + Depth)", normals_ensemble: int = 1, normals_scale: float = 1.0, disable_gpu: bool = False, hardware_mode: str = "Auto", parent=None):
        super().__init__(parent)
        self.image_np = image_np
        self.map_type = map_type
        self.image_path = image_path
        self.high_res = high_res
        self.cached_depth = cached_depth
        self.normals_method = normals_method
        self.normals_ensemble = normals_ensemble
        self.normals_scale = normals_scale
        self.disable_gpu = disable_gpu
        self.hardware_mode = hardware_mode
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            from core.ai_masking import AIMaskGenerator
            generator = AIMaskGenerator()
            res = generator.generate_mask(
                self.image_np,
                self.map_type,
                high_res_depth=self.high_res,
                high_res_normals=self.high_res,
                normals_method=self.normals_method,
                normals_ensemble=self.normals_ensemble,
                normals_scale=self.normals_scale,
                cached_depth=self.cached_depth,
                disable_gpu=self.disable_gpu,
                hardware_mode=self.hardware_mode,
                progress_callback=self._on_progress,
                tile_progress_callback=self._on_tile_progress
            )
            if not self._is_cancelled:
                self.finished.emit(self.map_type, res, self.image_path)
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(self.map_type, str(e))

    def _on_progress(self, msg: str):
        if not self._is_cancelled:
            self.progress.emit(msg)

    def _on_tile_progress(self, cur: int, total: int, stage: str = "Tiling", workers: int = 1):
        if not self._is_cancelled:
            self.tile_progress.emit(cur, total, stage, workers)

class PhotoEditorWindow(QMainWindow):
    """
    Main Application Window for Lumens Deep.
    Acts as the high-level orchestrator connecting plugins, GPU engine, docks, and worker threads.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumens Deep")
        self.resize(1280, 850)
        self.setMinimumSize(600, 400)

        # 1. Initialize Plugin Manager & GPU Engine
        self.plugin_manager = PluginManager()
        shader_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shaders')
        vs_path = os.path.join(shader_dir, 'vertex.glsl')
        fs_path = os.path.join(shader_dir, 'fragment.glsl')
        self.engine = PhotoEngine(vs_path, fs_path, plugin_manager=self.plugin_manager)

        # State Variables
        self.loaded_image_path = None
        self.split_view_mode = None
        self.split_position = 0.5
        self._cached_depth_np = None
        self.flip_90 = 0
        self.undo_stack = []
        self.redo_stack = []
        self.fast_viewer_mode = False

        # Multi-Crop Presets per Photo
        self.crop_presets = []
        self.active_crop_preset_id = "default"
        self._is_updating_crop_preset = False
        self._batch_crop_running = False
        self._batch_crop_queue = []
        self._batch_crop_total = 0
        self._batch_crop_current_idx = 0
        self._batch_waiting_for_crop_image = False

        # 2. Setup Central Viewport & Toolbars
        self._init_central_ui()

        # 3. Setup Modular Docks
        self._init_docks()

        # 4. Wire proxy properties and signals
        self._wire_proxies_and_signals()

        # 5. Menus, Status Bar & Timers
        self._create_menus()
        self._init_timers()
        self.statusBar().setStyleSheet("QStatusBar { background: #1a1a1a; color: #aaa; font-size: 11px; border-top: 1px solid #2e2e2e; padding: 2px; }")
        self.statusBar().showMessage("Ready")

        # 6. Initialize Lensfun database & MCP
        self._init_lensfun_database()
        self._setup_mcp_actions()

    def _set_neutral_passthrough_rendering(self):
        """Sets GPU engine to 1:1 clean passthrough without base curves, exposure boosts or adjustments for in-camera JPEG preview."""
        if hasattr(self, 'engine'):
            self.engine.base_curve_strength = 0.0
            self.engine.exposure = 1.0
            self.engine.contrast = 1.0
            self.engine.gamma = 1.0
            self.engine.ai_blend = 1.0
            self.engine.temperature = 0.0
            self.engine.tint = 0.0
            self.engine.highlight_recovery = 0.0
            self.engine.shadows = 0.0
            self.engine.whites = 0.0
            self.engine.black_level = 0.0
            self.engine.vignette = 0.0
            self.engine.sharpen = 0.0

    def _init_central_ui(self):
        self.central_container = QWidget()
        self.central_layout = QVBoxLayout(self.central_container)
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        self.central_layout.setSpacing(0)

        self.viewport_toolbar = ViewportToolbar(self)
        self.brush_settings_bar = BrushSettingsBar(self)
        self.brush_settings_bar.setVisible(False)
        self.image_label = ImagePreviewLabel(self)

        self.central_layout.addWidget(self.viewport_toolbar)
        self.central_layout.addWidget(self.brush_settings_bar)
        self.central_layout.addWidget(self.image_label, stretch=1)

        self.setCentralWidget(self.central_container)
        self.setDockOptions(QMainWindow.AnimatedDocks | QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
        self.setCorner(Qt.BottomLeftCorner, Qt.BottomDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)

    def _init_docks(self):
        # Left Panel: Folders, Global Settings, AI Assistant tabbed together
        self.dock_folders = FoldersDock(self)
        self.dock_global = GlobalDock(self)
        self.dock_ai = AIDock(self)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_folders)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_global)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_ai)

        self.tabifyDockWidget(self.dock_folders, self.dock_global)
        self.tabifyDockWidget(self.dock_global, self.dock_ai)
        self.dock_folders.raise_()

        self.dock_folders.folder_tree.folder_selected.connect(self._on_directory_selected)

        # Bottom Panel: Filmstrip
        self.dock_filmstrip = FilmstripDock(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_filmstrip)
        self.dock_filmstrip.filmstrip_widget.image_selected.connect(self.open_image_file)
        self.dock_filmstrip.filmstrip_widget.ai_culling_requested.connect(self.show_ai_culling_dialog)
        self.dock_filmstrip.filmstrip_widget.full_auto_requested.connect(self.trigger_full_auto_cull_and_process)
        self.dock_filmstrip.filmstrip_widget.stop_batch_requested.connect(self.stop_ai_auto_correct)
        self.dock_filmstrip.filmstrip_widget.aspect_ratio_changed.connect(self._on_crop_aspect_ratio_changed)
        self.dock_filmstrip.filmstrip_widget.swap_aspect_requested.connect(self._swap_crop_aspect_orientation)
        self.dock_filmstrip.filmstrip_widget.photo_notes_updated.connect(self._on_filmstrip_notes_updated)
        self.dock_filmstrip.filmstrip_widget.batch_ai_requested.connect(self.trigger_batch_ai_auto_correct)
        self.dock_filmstrip.filmstrip_widget.batch_crop_requested.connect(self.trigger_batch_ai_crop_presets)
        self.dock_filmstrip.filmstrip_widget.batch_export_requested.connect(self.export_batch_photos)
        self.dock_filmstrip.filmstrip_widget.batch_denoise_requested.connect(self.trigger_batch_ai_denoise)

        # Right Panel: Layers, Effects, Mask
        self.dock_layers = LayersDock(self)
        self.dock_effects = LayerEffectsDock(self, self.plugin_manager)
        self.dock_mask = MaskDock(self)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_layers)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_effects)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_mask)

        self.tabifyDockWidget(self.dock_effects, self.dock_mask)
        self.dock_effects.raise_()

        # Minimum width constraints on Left & Right dock panels to prevent content clipping
        for d in [self.dock_folders, self.dock_global, self.dock_ai]:
            d.setMinimumWidth(220)

        for d in [self.dock_layers, self.dock_effects, self.dock_mask]:
            d.setMinimumWidth(250)

        # Initialize default Base Layer in engine and layers dock
        if not self.engine.layers:
            self.engine.layers = [Layer(id="layer_base", name="Base Layer", is_base=True, active=True)]
            self.engine.active_layer_index = 0
        self.dock_layers.refresh_layers_list(self.engine.layers, 0)

    def _wire_proxies_and_signals(self):
        # Global Dock proxies
        self.slider_crop_l = self.dock_global.slider_crop_l
        self.slider_crop_r = self.dock_global.slider_crop_r
        self.slider_crop_t = self.dock_global.slider_crop_t
        self.slider_crop_b = self.dock_global.slider_crop_b
        self.slider_rotation = self.dock_global.slider_rotation
        self.lbl_rotation = self.dock_global.lbl_rotation
        self.check_bokeh_enable = self.dock_global.check_bokeh_enable
        self.slider_bokeh_focal_depth = self.dock_global.slider_bokeh_focal_depth
        self.slider_bokeh_focal_range = self.dock_global.slider_bokeh_focal_range
        self.slider_bokeh_radius = self.dock_global.slider_bokeh_radius
        self.slider_bokeh_petals = self.dock_global.slider_bokeh_petals
        self.combo_bokeh_samples = self.dock_global.combo_bokeh_samples
        self.slider_bokeh_boost = self.dock_global.slider_bokeh_boost
        self.slider_base_curve = getattr(self.dock_global, 'slider_base_curve', None)
        self.slider_ai_blend = getattr(self.dock_global, 'slider_ai_blend', None)
        self.chk_lens_enable = self.dock_global.chk_lens_enable
        self.check_lens_correction = self.dock_global.chk_lens_enable
        self.combo_cam_maker = self.dock_global.combo_cam_maker
        self.combo_cam_model = self.dock_global.combo_cam_model
        self.combo_lens_profile = self.dock_global.combo_lens_profile

        # Layers Dock proxies
        self.layers_list = self.dock_layers.layers_list
        self.slider_layer_opacity = self.dock_layers.slider_layer_opacity
        self.slider_mask_opacity = self.dock_layers.slider_layer_opacity
        self.lbl_layer_opacity = self.dock_layers.lbl_layer_opacity

        # Toolbar proxies
        self.btn_tb_split_v = self.viewport_toolbar.btn_tb_split_v
        self.btn_tb_split_h = self.viewport_toolbar.btn_tb_split_h
        self.btn_tb_side_by_side = self.viewport_toolbar.btn_tb_side_by_side
        self.btn_tb_crop_tool = self.viewport_toolbar.btn_tb_crop_tool
        self.btn_tb_wb_pick = self.viewport_toolbar.btn_tb_wb_pick
        self.btn_tb_focus_pick = self.viewport_toolbar.btn_tb_focus_pick
        self.btn_tb_brush = self.viewport_toolbar.btn_tb_brush

        # Basic Adjustments Plugin proxies
        ba = self.plugin_manager.get_plugin('basic_adjustments')
        if ba and hasattr(ba, '_sliders'):
            self.slider_exposure = ba._sliders.get('exposure')
            self.slider_contrast = ba._sliders.get('contrast')
            self.slider_gamma = ba._sliders.get('gamma')
            self.slider_temperature = ba._sliders.get('temperature')
            self.slider_tint = ba._sliders.get('tint')
            self.slider_highlights = ba._sliders.get('highlights')
            self.slider_shadows = ba._sliders.get('shadows')
            self.slider_whites = ba._sliders.get('whites')
            self.slider_black_level = ba._sliders.get('black_level')
            self.slider_highlight_recovery = ba._sliders.get('highlight_recovery')
            self.slider_local_contrast = ba._sliders.get('local_contrast')
            self.slider_global_vibrance = ba._sliders.get('global_vibrance')
            self.slider_global_chroma = ba._sliders.get('global_chroma')
            self.slider_denoise_luma = ba._sliders.get('denoise_luma')
            self.slider_denoise_chroma = ba._sliders.get('denoise_chroma')
            self.slider_sharpen = ba._sliders.get('sharpen')
            self.slider_sharpen_threshold = ba._sliders.get('sharpen_threshold')
            self.slider_vignette = ba._sliders.get('vignette')

        # Equalizer Proxies
        te = self.plugin_manager.get_plugin('tone_equalizer')
        self.tone_equalizer = getattr(te, '_eq_widget', None)
        ce = self.plugin_manager.get_plugin('color_equalizer')
        self.color_equalizer = getattr(ce, '_eq_widget', None)
        sce = self.plugin_manager.get_plugin('shadow_color_equalizer')
        self.shadow_color_equalizer = getattr(sce, '_eq_widget', None)

        # Mask Dock proxies
        self.group_depth_mask = self.dock_mask.chk_use_depth
        self.group_normal_mask = self.dock_mask.chk_use_normal
        self.slider_mask_fill = self.dock_mask.slider_mask_fill
        self.slider_mask_softness = self.dock_mask.slider_mask_softness
        self.slider_mask_luma_target = self.dock_mask.slider_mask_luma_t
        self.slider_mask_luma_range = self.dock_mask.slider_mask_luma_r
        self.slider_mask_hue_target = self.dock_mask.slider_mask_hue_t
        self.slider_mask_hue_range = self.dock_mask.slider_mask_hue_r
        self.slider_mask_sat_target = self.dock_mask.slider_mask_sat_t
        self.slider_mask_sat_range = self.dock_mask.slider_mask_sat_r
        self.combo_gradient_type = self.dock_mask.combo_grad_type
        self.slider_grad_black = self.dock_mask.slider_grad_black
        self.slider_grad_white = self.dock_mask.slider_grad_white
        self.slider_grad_gamma = self.dock_mask.slider_grad_gamma
        self.slider_depth_min = self.dock_mask.slider_depth_min
        self.slider_depth_max = self.dock_mask.slider_depth_max
        self.slider_depth_softness = self.dock_mask.slider_depth_softness
        self.slider_depth_gamma = self.dock_mask.slider_depth_gamma
        self.slider_normal_x = self.dock_mask.slider_norm_x
        self.slider_normal_y = self.dock_mask.slider_norm_y
        self.slider_normal_angle = self.dock_mask.slider_norm_angle

        # Extended Bokeh proxies
        self.slider_bokeh_fg_gamma = self.dock_global.slider_bokeh_fg_gamma
        self.slider_bokeh_bg_gamma = self.dock_global.slider_bokeh_bg_gamma
        self.slider_bokeh_subject_influence = self.dock_global.slider_bokeh_subject_influence
        self.slider_bokeh_rotation = self.dock_global.slider_bokeh_rotation
        self.slider_bokeh_chromatic = self.dock_global.slider_bokeh_chromatic
        self.slider_bokeh_vignette = self.dock_global.slider_bokeh_vignette
        self.slider_bokeh_radial_center = self.dock_global.slider_bokeh_radial_center
        self.slider_bokeh_radial_edge = self.dock_global.slider_bokeh_radial_edge
        self.slider_bokeh_radial_gamma = self.dock_global.slider_bokeh_radial_gamma

        # Aliases for compatibility
        self._on_image_loaded = self._on_image_loaded_async
        self.add_layer = self._add_layer
        self.delete_layer = self._delete_layer
        self.duplicate_layer = self._duplicate_layer
        self.get_current_state = self.serialize_state
        self.apply_state = self.deserialize_state

    def _init_timers(self):
        self.bokeh_timer = QTimer(self)
        self.bokeh_timer.setInterval(30)
        self.bokeh_timer.timeout.connect(self._on_bokeh_timer)
        
        self.undo_debounce_timer = QTimer(self)
        self.undo_debounce_timer.setSingleShot(True)
        self.undo_debounce_timer.timeout.connect(self.push_undo_state)

        self.thumb_refresh_timer = QTimer(self)
        self.thumb_refresh_timer.setSingleShot(True)
        self.thumb_refresh_timer.timeout.connect(self._refresh_filmstrip_active_thumbnail)

        self.image_load_generation_id = 0
        self.image_loader_thread = None
        self.loading_anim_timer = QTimer(self)
        self.loading_anim_timer.setInterval(25)
        self.loading_anim_timer.timeout.connect(self._on_loading_anim_tick)

    def _create_menus(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        
        open_act = QAction("&Open Image...", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._on_open_file_dialog)
        file_menu.addAction(open_act)

        save_act = QAction("&Save Project / Sidecar", self)
        save_act.setShortcut("Ctrl+S")
        save_act.triggered.connect(self.save_project_sidecar)
        file_menu.addAction(save_act)

        export_act = QAction("&Export High-Resolution Image...", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self.export_image)
        file_menu.addAction(export_act)

        file_menu.addSeparator()
        exit_act = QAction("E&xit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        edit_menu = mb.addMenu("&Edit")
        undo_act = QAction("&Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction("&Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        redo_act.triggered.connect(self.redo)
        edit_menu.addAction(redo_act)

        reset_act = QAction("&Reset All Adjustments", self)
        reset_act.triggered.connect(self.reset_settings)
        edit_menu.addAction(reset_act)

        edit_menu.addSeparator()
        del_rejected_act = QAction("🗑️ Delete Rejected Photos from Disk...", self)
        del_rejected_act.triggered.connect(lambda: self.dock_filmstrip.filmstrip_widget.delete_rejected_photos_from_disk())
        edit_menu.addAction(del_rejected_act)

        view_menu = mb.addMenu("&View")
        fit_act = QAction("&Fit to Viewport", self)
        fit_act.setShortcut("F")
        fit_act.triggered.connect(self.fit_to_viewport)
        view_menu.addAction(fit_act)

        zoom100_act = QAction("&100% Zoom", self)
        zoom100_act.setShortcut("Z")
        zoom100_act.triggered.connect(self.zoom_to_100)
        view_menu.addAction(zoom100_act)

        view_menu.addSeparator()
        self.act_view_depth = QAction("🌊 Preview Depth Map", self, checkable=True)
        self.act_view_depth.setShortcut("D")
        self.act_view_depth.toggled.connect(self.viewport_toolbar.btn_tb_depth.setChecked)
        view_menu.addAction(self.act_view_depth)

        self.act_view_normals = QAction("🌐 Preview Surface Normals", self, checkable=True)
        self.act_view_normals.setShortcut("N")
        self.act_view_normals.toggled.connect(self.viewport_toolbar.btn_tb_normals.setChecked)
        view_menu.addAction(self.act_view_normals)

        self.act_view_subject = QAction("👤 Preview Subject Mask", self, checkable=True)
        self.act_view_subject.setShortcut("S")
        self.act_view_subject.toggled.connect(self.viewport_toolbar.btn_tb_subject.setChecked)
        view_menu.addAction(self.act_view_subject)

        self.act_view_mask = QAction("🎭 Preview Active Layer Mask", self, checkable=True)
        self.act_view_mask.setShortcut("M")
        self.act_view_mask.toggled.connect(self.viewport_toolbar.btn_tb_mask_prev.setChecked)
        view_menu.addAction(self.act_view_mask)

        view_menu.addSeparator()
        self.act_fast_viewer = QAction("⚡ Fast Viewer Mode (Instant Browse)", self, checkable=True)
        self.act_fast_viewer.setShortcut("V")
        self.act_fast_viewer.toggled.connect(self._on_viewer_mode_toggled)
        view_menu.addAction(self.act_fast_viewer)

        self.act_profiler_hud = QAction("📊 Performance Profiler HUD", self, checkable=True)
        self.act_profiler_hud.setShortcut("F3")
        self.act_profiler_hud.toggled.connect(self._toggle_profiler_hud)
        view_menu.addAction(self.act_profiler_hud)

        # AI & 3D Neural Tools Menu
        ai_menu = mb.addMenu("&AI & 3D Tools")

        ai_culling_act = QAction("✨ Gemini &AI Photo Culling & 3D Selector...", self)
        ai_culling_act.setShortcut("Ctrl+Shift+C")
        ai_culling_act.triggered.connect(self.show_ai_culling_dialog)
        ai_menu.addAction(ai_culling_act)

        ai_menu.addSeparator()

        calc_all_act = QAction("🚀 Compute &All 3D AI Maps", self)
        calc_all_act.setShortcut("Ctrl+Shift+A")
        calc_all_act.triggered.connect(self.trigger_compute_all_ai_maps)
        ai_menu.addAction(calc_all_act)

        ai_menu.addSeparator()

        calc_depth_act = QAction("🌊 Compute &Depth Map", self)
        calc_depth_act.setShortcut("Ctrl+Shift+D")
        calc_depth_act.triggered.connect(self.trigger_compute_depth_map)
        ai_menu.addAction(calc_depth_act)

        calc_norm_act = QAction("🌐 Compute Surface &Normals", self)
        calc_norm_act.setShortcut("Ctrl+Shift+N")
        calc_norm_act.triggered.connect(self.trigger_compute_normal_map)
        ai_menu.addAction(calc_norm_act)

        ai_menu.addSeparator()

        ai_denoise_act = QAction("🔇 AI Denoise Current &Photo (In-Memory)...", self)
        ai_denoise_act.setShortcut("Ctrl+Shift+R")
        ai_denoise_act.triggered.connect(self.trigger_ai_raw_denoise)
        ai_menu.addAction(ai_denoise_act)

        ai_menu.addSeparator()

        auto_bg_act = QAction("🌄 Create &Background Layer (3D Depth)", self)
        auto_bg_act.setShortcut("Ctrl+Shift+B")
        auto_bg_act.triggered.connect(self.auto_create_background_layer)
        ai_menu.addAction(auto_bg_act)

        ai_menu.addSeparator()

        auto_enhance_act = QAction("✨ Run AI Auto-Correction", self)
        auto_enhance_act.triggered.connect(self.trigger_ai_auto_correct)
        ai_menu.addAction(auto_enhance_act)

        batch_ai_act = QAction("⚡ Run &Batch AI Auto-Correction on Filmstrip", self)
        batch_ai_act.setShortcut("Ctrl+Shift+A")
        batch_ai_act.triggered.connect(self.trigger_batch_ai_auto_correct)
        ai_menu.addAction(batch_ai_act)

        full_auto_act = QAction("🚀 Run &Full Auto Workflow (Cull + Auto-Correct All)", self)
        full_auto_act.setShortcut("Ctrl+Shift+F")
        full_auto_act.triggered.connect(self.trigger_full_auto_cull_and_process)
        ai_menu.addAction(full_auto_act)

        ai_menu.addSeparator()

        self.stop_ai_act = QAction("⏹ &Stop AI / Batch Auto-Correction", self)
        self.stop_ai_act.setShortcut("Escape")
        self.stop_ai_act.triggered.connect(self.stop_ai_auto_correct)
        ai_menu.addAction(self.stop_ai_act)

        # =====================================================================
        # Window / Panels Menu (Restore & Toggle Closed Dock Panels)
        # =====================================================================
        window_menu = mb.addMenu("&Window")

        # Left Docks
        window_menu.addAction(self.dock_folders.toggleViewAction())
        window_menu.addAction(self.dock_global.toggleViewAction())
        window_menu.addAction(self.dock_ai.toggleViewAction())

        window_menu.addSeparator()

        # Bottom Dock
        window_menu.addAction(self.dock_filmstrip.toggleViewAction())

        window_menu.addSeparator()

        # Right Docks
        window_menu.addAction(self.dock_layers.toggleViewAction())
        window_menu.addAction(self.dock_effects.toggleViewAction())
        window_menu.addAction(self.dock_mask.toggleViewAction())

        window_menu.addSeparator()

        show_all_act = QAction("👁️ Show All Panels", self)
        show_all_act.triggered.connect(self.show_all_panels)
        window_menu.addAction(show_all_act)

        reset_layout_act = QAction("🔄 Reset Panels to Default Layout", self)
        reset_layout_act.triggered.connect(self.reset_panels_layout)
        window_menu.addAction(reset_layout_act)

        # =====================================================================
        # Help Menu (Documentation, Shortcuts, Website, About Dialog)
        # =====================================================================
        help_menu = mb.addMenu("&Help")

        docs_act = QAction("📖 &User Guide & Documentation...", self)
        docs_act.setShortcut("F1")
        docs_act.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_act)

        shortcuts_act = QAction("⌨️ &Keyboard Shortcuts Reference...", self)
        shortcuts_act.triggered.connect(self.open_shortcuts_reference)
        help_menu.addAction(shortcuts_act)

        help_menu.addSeparator()

        website_act = QAction("🌐 Official &Website (lumensdeep.com)...", self)
        website_act.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://lumensdeep.com")))
        help_menu.addAction(website_act)

        github_act = QAction("🐙 &GitHub Repository && Issues...", self)
        github_act.triggered.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/3dcarrots/LumensDeep")))
        help_menu.addAction(github_act)

        help_menu.addSeparator()

        about_act = QAction("ℹ️ &About Lumens Deep...", self)
        about_act.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_act)

    def _on_viewer_mode_toggled(self, checked: bool):
        self.fast_viewer_mode = checked
        if hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'btn_tb_viewer'):
            self.viewport_toolbar.btn_tb_viewer.blockSignals(True)
            self.viewport_toolbar.btn_tb_viewer.setChecked(checked)
            self.viewport_toolbar.btn_tb_viewer.blockSignals(False)
        if hasattr(self, 'act_fast_viewer'):
            self.act_fast_viewer.blockSignals(True)
            self.act_fast_viewer.setChecked(checked)
            self.act_fast_viewer.blockSignals(False)

        # Visually disable/enable edit docks so it's clear editing is paused in Viewer mode
        for dock in [self.dock_effects, self.dock_layers, self.dock_mask]:
            dock.setEnabled(not checked)

        if checked:
            # Entering Fast Viewer Mode
            if self.image_loader_thread and self.image_loader_thread.isRunning():
                self.image_loader_thread.cancel()
            self.image_label.hide_background_loading()
            self.image_label.hide_loading()
            self.loading_anim_timer.stop()
            self.statusBar().showMessage("⚡ Fast Viewer Mode ACTIVE • Browsing photos in instant JPEG mode (RAW decoding paused). Turn off Viewer Mode to edit RAW.")
            self._update_view()
        else:
            # Exiting Fast Viewer Mode: If currently loaded image is RAW and not full-decoded yet, launch background decode!
            self.statusBar().showMessage("⚡ Fast Viewer Mode DISABLED • Edit Mode active.")
            if self.loaded_image_path:
                ext = os.path.splitext(self.loaded_image_path)[1].lower()
                is_raw = ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw']
                if is_raw and getattr(self.engine, 'is_passthrough', False):
                    filename = os.path.basename(self.loaded_image_path)
                    self.image_label.show_background_loading(filename, "Decoding full RAW & 3D maps in background...")
                    self.loading_anim_timer.start()
                    self.image_load_generation_id += 1
                    gen_id = self.image_load_generation_id
                    self.image_loader_thread = ImageLoaderThread(self.loaded_image_path, gen_id, self)
                    self.image_loader_thread.status_updated.connect(self._on_image_status_updated_async)
                    self.image_loader_thread.image_loaded.connect(self._on_image_loaded_async)
                    self.image_loader_thread.load_error.connect(self._on_image_load_error_async)
                    self.image_loader_thread.start()

    def _on_loading_anim_tick(self):
        is_bg = hasattr(self, 'image_label') and getattr(self.image_label, 'is_background_loading', False)
        is_modal = hasattr(self, 'image_label') and getattr(self.image_label, 'is_loading', False)
        is_ai = hasattr(self, 'image_label') and getattr(self.image_label, 'is_ai_running', False)
        if is_bg or is_modal or is_ai:
            self.image_label.loading_anim_tick = getattr(self.image_label, 'loading_anim_tick', 0) + 1
            self.image_label.update()
        else:
            if hasattr(self, 'loading_anim_timer'):
                self.loading_anim_timer.stop()

    def _on_directory_selected(self, path: str):
        self.dock_filmstrip.filmstrip_widget.load_directory(path)

    def _on_open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.dng *.raw *.orf *.rw2 *.pef *.raf *.srw);;All Files (*)"
        )
        if file_path:
            self.open_image_file(file_path)

    def open_image_file(self, file_path: str):
        if self.loaded_image_path and self.loaded_image_path != file_path:
            if not self._check_and_prompt_save():
                return

        self.image_load_generation_id += 1
        gen_id = self.image_load_generation_id

        if self.image_loader_thread and self.image_loader_thread.isRunning():
            self.image_loader_thread.cancel()

        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        is_raw = ext in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw']

        # Check for fast instant preview candidates in priority:
        preview_jpg_path = None
        rhalf_jpg_path = None
        half_jpg_path = None

        dir_name = os.path.dirname(file_path)
        stem = os.path.splitext(filename)[0]
        lumen_dir = os.path.join(dir_name, "LumenDeep")

        # Priority 1: _preview.jpg (Rendered viewport composite snapshot with all edits, 3D maps and effects)
        for p in [os.path.join(lumen_dir, f"{stem}_preview.jpg"), os.path.join(dir_name, f"{stem}_preview.jpg")]:
            if os.path.exists(p):
                preview_jpg_path = p
                break

        # Priority 2: _rhalf.jpg (Exact RAW sensor linear preview)
        if is_raw:
            for r in [os.path.join(lumen_dir, f"{stem}_rhalf.jpg"), os.path.join(dir_name, f"{stem}_rhalf.jpg")]:
                if os.path.exists(r):
                    rhalf_jpg_path = r
                    break

            # Priority 3: _half.jpg (In-camera JPEG preview)
            for h in [os.path.join(lumen_dir, f"{stem}_half.jpg"), os.path.join(dir_name, f"{stem}_half.jpg")]:
                if os.path.exists(h):
                    half_jpg_path = h
                    break

        if self.fast_viewer_mode:
            # === FAST VIEWER MODE: Ultra-fast 2560px preview (sub-50ms), NEVER decode full RAW! ===
            loaded_fast = False
            if preview_jpg_path:
                try:
                    linear_prev, meta_prev = load_image_to_linear(preview_jpg_path, parse_exif=False, max_size=2560)
                    self.loaded_image_path = file_path
                    self._cached_depth_np = None
                    self.engine.is_passthrough = True
                    self.engine.set_image(linear_prev, meta_prev)
                    self._set_neutral_passthrough_rendering()
                    mcp_server.state.image_loaded = True
                    mcp_server.state.image_width = self.engine.width
                    mcp_server.state.image_height = self.engine.height
                    loaded_fast = True
                except Exception: pass

            if not loaded_fast and half_jpg_path:
                try:
                    linear_half, meta_half = load_image_to_linear(half_jpg_path, parse_exif=False, max_size=2560)
                    self.loaded_image_path = file_path
                    self._cached_depth_np = None
                    self.engine.is_passthrough = True
                    self.engine.set_image(linear_half, meta_half)
                    mcp_server.state.image_loaded = True
                    mcp_server.state.image_width = self.engine.width
                    mcp_server.state.image_height = self.engine.height
                    self._set_neutral_passthrough_rendering()
                    loaded_fast = True
                except Exception: pass

            if not loaded_fast and is_raw:
                # Fast extract embedded JPEG directly from RAW in <30ms
                linear_thumb, meta_thumb = extract_raw_thumbnail_linear(file_path, max_size=2560)
                if linear_thumb is not None:
                    self.loaded_image_path = file_path
                    self._cached_depth_np = None
                    self.engine.is_passthrough = True
                    self.engine.set_image(linear_thumb, meta_thumb)
                    mcp_server.state.image_loaded = True
                    mcp_server.state.image_width = self.engine.width
                    mcp_server.state.image_height = self.engine.height
                    self._set_neutral_passthrough_rendering()
                    loaded_fast = True
            elif not loaded_fast:
                try:
                    linear_std, meta_std = load_image_to_linear(file_path, parse_exif=False, max_size=2560)
                    self.loaded_image_path = file_path
                    self._cached_depth_np = None
                    self.engine.is_passthrough = True
                    self.engine.set_image(linear_std, meta_std)
                    mcp_server.state.image_loaded = True
                    mcp_server.state.image_width = self.engine.width
                    mcp_server.state.image_height = self.engine.height
                    loaded_fast = True
                except Exception: pass

            self.dock_filmstrip.filmstrip_widget.set_active_image(file_path)
            self.image_label.hide_background_loading()
            self.image_label.hide_loading()
            self.loading_anim_timer.stop()

            self.setWindowTitle(f"Lumens Deep - {filename}  [⚡ Fast Viewer Mode]")
            self.fit_to_viewport()
            self.statusBar().showMessage(f"⚡ Fast Viewer Mode • {filename} ({self.engine.width}x{self.engine.height}) • RAW decoding paused. Press 'V' to edit RAW.")
            return

        # === REGULAR EDIT MODE: Instant Neutral Preview (Fast Viewer style) + Background 32-bit RAW Loader ===
        loaded_instant_preview = False

        if preview_jpg_path:
            try:
                linear_prev, meta_prev = load_image_to_linear(preview_jpg_path, parse_exif=False, max_size=2560)
                self.loaded_image_path = file_path
                self._cached_depth_np = None
                self.engine.is_passthrough = True
                self.engine.set_image(linear_prev, meta_prev)
                self._set_neutral_passthrough_rendering()
                mcp_server.state.image_loaded = True
                mcp_server.state.image_width = self.engine.width
                mcp_server.state.image_height = self.engine.height
                loaded_instant_preview = True
            except Exception as e:
                print(f"Failed to load preview {preview_jpg_path}: {e}")

        if not loaded_instant_preview and half_jpg_path:
            try:
                linear_half, meta_half = load_image_to_linear(half_jpg_path, parse_exif=False, max_size=2560)
                self.loaded_image_path = file_path
                self._cached_depth_np = None
                self.engine.is_passthrough = True
                self.engine.set_image(linear_half, meta_half)
                self._set_neutral_passthrough_rendering()
                mcp_server.state.image_loaded = True
                mcp_server.state.image_width = self.engine.width
                mcp_server.state.image_height = self.engine.height
                loaded_instant_preview = True
            except Exception as e:
                print(f"Failed to load half preview {half_jpg_path}: {e}")

        if not loaded_instant_preview and is_raw:
            try:
                linear_thumb, meta_thumb = extract_raw_thumbnail_linear(file_path, max_size=2560)
                if linear_thumb is not None:
                    self.loaded_image_path = file_path
                    self._cached_depth_np = None
                    self.engine.is_passthrough = True
                    self.engine.set_image(linear_thumb, meta_thumb)
                    self._set_neutral_passthrough_rendering()
                    mcp_server.state.image_loaded = True
                    mcp_server.state.image_width = self.engine.width
                    mcp_server.state.image_height = self.engine.height
                    loaded_instant_preview = True
            except Exception as e:
                print(f"Failed to extract instant raw thumbnail from {file_path}: {e}")

        if not loaded_instant_preview and not is_raw:
            try:
                linear_std, meta_std = load_image_to_linear(file_path, parse_exif=False, max_size=2560)
                self.loaded_image_path = file_path
                self._cached_depth_np = None
                self.engine.is_passthrough = True
                self.engine.set_image(linear_std, meta_std)
                self._set_neutral_passthrough_rendering()
                mcp_server.state.image_loaded = True
                mcp_server.state.image_width = self.engine.width
                mcp_server.state.image_height = self.engine.height
                loaded_instant_preview = True
            except Exception as e:
                print(f"Failed to load std preview: {e}")

        self.dock_filmstrip.filmstrip_widget.set_active_image(file_path)
        self.setWindowTitle(f"Lumens Deep - {filename}  [Instant Preview]")
        self.fit_to_viewport()

        if loaded_instant_preview:
            self.image_label.show_background_loading(filename, "Decoding full 32-bit RAW sensor data in background...")
            self.statusBar().showMessage(f"⚡ Instant Preview loaded ({self.engine.width}x{self.engine.height}) • Decoding full RAW in background...")
        else:
            self.image_label.show_loading(filename, "Decoding RAW sensor data..." if is_raw else "Decoding image file...")
            self.statusBar().showMessage(f"⏳ Decoding {filename}...")
        self.loading_anim_timer.start()

        self.image_loader_thread = ImageLoaderThread(file_path, gen_id, self)
        self.image_loader_thread.status_updated.connect(self._on_image_status_updated_async)
        self.image_loader_thread.image_loaded.connect(self._on_image_loaded_async)
        self.image_loader_thread.load_error.connect(self._on_image_load_error_async)
        self.image_loader_thread.start()

    def _on_image_status_updated_async(self, path: str, status_text: str, gen_id: int):
        if gen_id != self.image_load_generation_id: return
        if getattr(self.image_label, 'is_background_loading', False):
            self.image_label.update_background_loading_status(status_text)
            self.statusBar().showMessage(f"⚡ Instant Preview • {status_text}")
        else:
            self.image_label.update_loading_status(status_text)
            self.statusBar().showMessage(f"⏳ {status_text}")

    def _on_image_loaded_async(self, path: str, linear_data, metadata, depth_arr, norm_arr, subj_arr, gen_id: int):
        if gen_id != self.image_load_generation_id: return
        
        # Check if we were already displaying an instant preview of this same file
        is_same_file = (self.loaded_image_path == path)
        was_passthrough = getattr(self.engine, 'is_passthrough', False)

        self.loaded_image_path = path
        self.image_label.hide_loading()
        self.image_label.hide_background_loading()
        self.loading_anim_timer.stop()

        self.engine.is_passthrough = False
        self._pristine_linear_raw = linear_data.copy() if linear_data is not None else None
        self.engine.set_image(linear_data, metadata)
        mcp_server.state.image_loaded = True
        mcp_server.state.image_width = self.engine.width
        mcp_server.state.image_height = self.engine.height

        # Set preloaded AI maps immediately on GPU
        self.engine.set_depth_map(depth_arr)
        self._cached_depth_np = depth_arr
        self.engine.set_normal_map(norm_arr)
        self.engine.set_subject_mask(subj_arr)

        if is_same_file and not was_passthrough:
            # User was actively editing during _rhalf.jpg preview: PRESERVE ALL LIVE EDITS!
            current_state = self.serialize_state()
            self.deserialize_state(current_state)
        else:
            # First time loading or coming from raw camera JPEG passthrough: restore sidecar
            self._restore_sidecar_or_defaults(path)

        self.dock_filmstrip.filmstrip_widget.set_active_image(path)
        
        # Update Window Title with photo EXIF metadata
        meta_str = []
        if metadata:
            if metadata.get('iso'): meta_str.append(f"ISO {metadata['iso']}")
            if metadata.get('shutter'):
                s_val = str(metadata['shutter'])
                meta_str.append(s_val if s_val.endswith('s') else f"{s_val}s")
            if metadata.get('aperture'):
                ap = metadata['aperture']
                meta_str.append(f"f/{ap:.1f}" if isinstance(ap, float) else f"f/{ap}")
        filename = os.path.basename(path)
        title = f"Lumens Deep - {filename}"
        if meta_str:
            title += f"  ({', '.join(meta_str)})"
        self.setWindowTitle(title)

        # Get AI notes / user comments from cache or sidecar
        ai_notes = ""
        try:
            from core.filmstrip_cache import FilmstripCache
            cached_rec = FilmstripCache.get_instance().get_photo(path)
            if cached_rec and cached_rec.get('ai_notes'):
                ai_notes = cached_rec['ai_notes']
        except Exception:
            pass

        if hasattr(self, 'dock_global'):
            self.dock_global.update_metadata(metadata, width=self.engine.width, height=self.engine.height, ai_notes=ai_notes)

        # Auto-detect Camera & Lens Profile from EXIF metadata
        if metadata and hasattr(self, 'lf_db'):
            self._sync_lens_combos_from_metadata(metadata)

        # Apply Optical / Lens Correction if enabled
        should_enable_lens = False
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_lens_correction_enabled') and self.dock_ai.is_batch_lens_correction_enabled():
            should_enable_lens = True
        elif hasattr(self, 'chk_lens_enable') and self.chk_lens_enable.isChecked():
            should_enable_lens = True

        if should_enable_lens and hasattr(self, 'chk_lens_enable'):
            self.chk_lens_enable.blockSignals(True)
            self.chk_lens_enable.setChecked(True)
            self.chk_lens_enable.blockSignals(False)
            self.engine.enable_lens_correction = True
            self.engine.apply_lens_correction()

        self._update_view()
        self.statusBar().showMessage(f"✓ Loaded {os.path.basename(path)} ({self.engine.width}x{self.engine.height})")

        # Check if we were waiting for this image in Batch AI
        if getattr(self, '_batch_ai_running', False) and getattr(self, '_batch_waiting_for_full_image', False):
            self._batch_waiting_for_full_image = False
            QTimer.singleShot(150, self._proceed_batch_photo_ai)

        # Check if we were waiting for this image in Batch AI Crop
        if getattr(self, '_batch_crop_running', False) and getattr(self, '_batch_waiting_for_crop_image', False):
            self._batch_waiting_for_crop_image = False
            QTimer.singleShot(150, self._proceed_batch_photo_crop)

        # Check if user had requested manual Gemini run while RAW was decoding
        if getattr(self, '_pending_gemini_start', None) is not None:
            pending_prompt, pending_auto = self._pending_gemini_start
            self._pending_gemini_start = None
            QTimer.singleShot(100, lambda: self._start_gemini_assistant(pending_prompt, pending_auto))

    def _on_image_load_error_async(self, path: str, err_msg: str, gen_id: int):
        if gen_id != self.image_load_generation_id: return
        self.image_label.hide_loading()
        self.image_label.hide_background_loading()
        self.loading_anim_timer.stop()
        self.statusBar().showMessage(f"❌ Failed to load {os.path.basename(path)}: {err_msg}")
        if getattr(self, '_batch_ai_running', False):
            self.dock_ai.txt_ai_output.append(f"❌ Failed to load {os.path.basename(path)}: {err_msg}. Skipping to next photo...")
            self._batch_ai_current_idx += 1
            QTimer.singleShot(1000, self._run_next_batch_photo)

    def _auto_load_ai_maps(self, image_path: str):
        """Scans LumenDeep subfolder and current directory for pre-computed 3D/AI maps."""
        if not image_path or not os.path.exists(image_path):
            return

        dir_name = os.path.dirname(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        lumen_dir = os.path.join(dir_name, "LumenDeep")

        # 1. Depth Map candidates
        depth_candidates = [
            os.path.join(lumen_dir, f"{stem}_depth.png"),
            os.path.join(lumen_dir, f"{stem}.depth.png"),
            os.path.join(dir_name, f"{stem}_depth.png"),
            os.path.join(dir_name, f"{stem}.depth.png"),
            os.path.join(lumen_dir, f"{stem}_depth.npy"),
        ]
        loaded_depth = False
        for c in depth_candidates:
            if os.path.exists(c):
                try:
                    from PIL import Image
                    arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                    self.engine.set_depth_map(arr)
                    self._cached_depth_np = arr
                    print(f"Loaded depth map: {c}")
                    loaded_depth = True
                    break
                except Exception as e:
                    print(f"Failed to load depth map {c}: {e}")
        if not loaded_depth:
            self.engine.set_depth_map(None)
            self._cached_depth_np = None
            if hasattr(self, 'check_bokeh_enable') and self.check_bokeh_enable.isChecked():
                self.check_bokeh_enable.blockSignals(True)
                self.check_bokeh_enable.setChecked(False)
                self.check_bokeh_enable.blockSignals(False)
                self.engine.bokeh_enabled = False
                self.bokeh_timer.stop()

        # 2. Normal Map candidates
        normal_candidates = [
            os.path.join(lumen_dir, f"{stem}_normals.png"),
            os.path.join(lumen_dir, f"{stem}.normals.png"),
            os.path.join(dir_name, f"{stem}_normals.png"),
            os.path.join(dir_name, f"{stem}.normals.png"),
            os.path.join(lumen_dir, f"{stem}_normals.npy"),
        ]
        loaded_normals = False
        for c in normal_candidates:
            if os.path.exists(c):
                try:
                    from PIL import Image
                    arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                    self.engine.set_normal_map(arr)
                    print(f"Loaded normal map: {c}")
                    loaded_normals = True
                    break
                except Exception as e:
                    print(f"Failed to load normal map {c}: {e}")
        if not loaded_normals:
            self.engine.set_normal_map(None)

        # 3. Subject Mask candidates
        subject_candidates = [
            os.path.join(lumen_dir, f"{stem}_subject.png"),
            os.path.join(lumen_dir, f"{stem}.subject.png"),
            os.path.join(lumen_dir, f"{stem}_mask.png"),
            os.path.join(lumen_dir, f"{stem}.mask.png"),
            os.path.join(dir_name, f"{stem}_subject.png"),
            os.path.join(dir_name, f"{stem}.subject.png"),
            os.path.join(lumen_dir, f"{stem}_subject.npy"),
        ]
        loaded_subject = False
        for c in subject_candidates:
            if os.path.exists(c):
                try:
                    from PIL import Image
                    arr = np.load(c) if c.endswith('.npy') else np.array(Image.open(c))
                    self.engine.set_subject_mask(arr)
                    self._cached_subject_np = arr
                    print(f"Loaded subject mask: {c}")
                    loaded_subject = True
                    break
                except Exception as e:
                    print(f"Failed to load subject mask {c}: {e}")
        if not loaded_subject:
            self.engine.set_subject_mask(None)
            self._cached_subject_np = None

    def _get_cached_depth_map(self) -> Optional[np.ndarray]:
        """Returns in-memory or loaded on-disk 32-bit depth map in float32 [0..1]."""
        if hasattr(self, '_cached_depth_np') and self._cached_depth_np is not None:
            if self._cached_depth_np.dtype == np.uint16:
                return self._cached_depth_np.astype(np.float32) / 65535.0
            elif self._cached_depth_np.dtype == np.uint8:
                return self._cached_depth_np.astype(np.float32) / 255.0
            return self._cached_depth_np.astype(np.float32)

        if not hasattr(self, 'loaded_image_path') or not self.loaded_image_path:
            return None

        dir_name = os.path.dirname(self.loaded_image_path)
        stem = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        lumen_dir = os.path.join(dir_name, "LumenDeep")
        depth_candidates = [
            os.path.join(lumen_dir, f"{stem}_depth.png"),
            os.path.join(lumen_dir, f"{stem}.depth.png"),
            os.path.join(dir_name, f"{stem}_depth.png"),
            os.path.join(dir_name, f"{stem}.depth.png"),
            os.path.join(lumen_dir, f"{stem}_depth.npy"),
        ]
        for p in depth_candidates:
            if os.path.exists(p):
                try:
                    from PIL import Image
                    if p.endswith('.npy'):
                        arr = np.load(p)
                    else:
                        img = Image.open(p)
                        arr = np.array(img)
                    if arr.dtype == np.uint16:
                        depth_f32 = arr.astype(np.float32) / 65535.0
                    elif arr.dtype == np.uint8:
                        depth_f32 = arr.astype(np.float32) / 255.0
                    else:
                        depth_f32 = arr.astype(np.float32)
                    self._cached_depth_np = depth_f32
                    return depth_f32
                except Exception as e:
                    print(f"Failed to load cached depth map from {p}: {e}")
        return None

    def _get_cached_subject_mask(self) -> Optional[np.ndarray]:
        if getattr(self, '_cached_subject_np', None) is not None:
            return self._cached_subject_np
        if not self.loaded_image_path:
            return None
        dir_name = os.path.dirname(self.loaded_image_path)
        stem = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
        lumen_dir = os.path.join(dir_name, "LumenDeep")
        subj_candidates = [
            os.path.join(lumen_dir, f"{stem}_subject.png"),
            os.path.join(lumen_dir, f"{stem}.subject.png"),
            os.path.join(lumen_dir, f"{stem}_mask.png"),
            os.path.join(lumen_dir, f"{stem}.mask.png"),
            os.path.join(dir_name, f"{stem}_subject.png"),
            os.path.join(dir_name, f"{stem}.subject.png"),
            os.path.join(lumen_dir, f"{stem}_subject.npy"),
        ]
        for p in subj_candidates:
            if os.path.exists(p):
                try:
                    from PIL import Image
                    if p.endswith('.npy'):
                        arr = np.load(p)
                    else:
                        img = Image.open(p)
                        arr = np.array(img)
                    if arr.dtype == np.uint8:
                        subj_f32 = arr.astype(np.float32) / 255.0
                    else:
                        subj_f32 = arr.astype(np.float32)
                    self._cached_subject_np = subj_f32
                    return subj_f32
                except Exception as e:
                    print(f"Failed to load cached subject mask from {p}: {e}")
        return None

    # --- AI Map Computation Pipeline ---
    def _get_current_image_uint8(self) -> Optional[np.ndarray]:
        """
        Extracts full sensor resolution RGB uint8 image for AI depth/normals/subject computation.
        Uses engine.get_ai_image_numpy(high_res=True) first to get original uncropped raw frame,
        with robust fallbacks for RAW (.CR3/.NEF/.ARW) and standard raster images.
        """
        if hasattr(self, 'engine') and self.engine and self.engine.texture is not None:
            arr = self.engine.get_ai_image_numpy(high_res=True)
            if arr is not None:
                return arr

        if not self.loaded_image_path:
            return None

        import numpy as np
        from PIL import Image, ImageOps
        try:
            img = Image.open(self.loaded_image_path)
            img = ImageOps.exif_transpose(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception:
            try:
                from core.image_io import load_image_to_linear
                linear_data, _ = load_image_to_linear(self.loaded_image_path)
                arr = np.clip(linear_data, 0.0, 1.0)
                mask = arr <= 0.0031308
                srgb = np.where(mask, arr * 12.92, 1.055 * (arr ** (1.0 / 2.4)) - 0.055)
                return (np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            except Exception as e:
                print(f"Failed to get full-res image for AI: {e}")
                px = self.engine.read_pixels(to_preview=False)
                if px.dtype != np.uint8:
                    px = (np.clip(px, 0.0, 1.0) * 255.0).astype(np.uint8)
                return px

    def trigger_compute_depth_map(self):
        self._start_ai_map_computation("Depth Map")

    def trigger_compute_normal_map(self):
        self._start_ai_map_computation("Normal Map")

    def trigger_compute_subject_mask(self):
        self._start_ai_map_computation("Subject/Background")

    def trigger_compute_all_ai_maps(self):
        self._start_ai_map_computation("All")

    # =========================================================================
    # AI Noise Reduction (NAFNet Pipeline - In-Memory RAM Execution)
    # =========================================================================
    def trigger_ai_raw_denoise(self):
        if not self.loaded_image_path or not mcp_server.state.image_loaded:
            QMessageBox.information(self, "No Image Loaded", "Please open a photo first.")
            return

        if hasattr(self, 'denoise_worker') and self.denoise_worker and self.denoise_worker.isRunning():
            QMessageBox.warning(self, "Task Running", "An AI Denoise task is already in progress. Please wait.")
            return

        tile_size = 512
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_tile'):
            if "1024" in self.dock_ai.combo_denoise_tile.currentText():
                tile_size = 1024

        save_to_disk = False
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'chk_denoise_save_tiff'):
            save_to_disk = self.dock_ai.chk_denoise_save_tiff.isChecked()

        strength = 1.0
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_strength'):
            strength = self.dock_ai.slider_denoise_strength.value() / 100.0

        texture_retention = 0.25
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_texture'):
            texture_retention = self.dock_ai.slider_denoise_texture.value() / 100.0

        linear_rgb = None
        if hasattr(self, '_pristine_linear_raw') and self._pristine_linear_raw is not None:
            linear_rgb = self._pristine_linear_raw.copy()
        elif hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'raw_texture') and self.engine.raw_texture:
            try:
                if hasattr(self.engine, 'get_linear_raw_numpy'):
                    linear_rgb = self.engine.get_linear_raw_numpy()
                else:
                    raw_bytes = self.engine.raw_texture.read()
                    tex = self.engine.raw_texture
                    dtype = np.float16 if getattr(tex, 'dtype', 'f4') == 'f2' else np.float32
                    arr_full = np.frombuffer(raw_bytes, dtype=dtype).reshape((self.engine.height, self.engine.width, tex.components))
                    linear_rgb = np.ascontiguousarray(arr_full[..., :3].astype(np.float32))
            except Exception as e:
                print(f"[AI Denoise] Could not extract from engine texture: {e}")
                linear_rgb = None

        if linear_rgb is None and self.loaded_image_path:
            try:
                from core.image_io import load_image_to_linear
                linear_rgb, _ = load_image_to_linear(self.loaded_image_path)
            except Exception as e:
                print(f"[AI Denoise] Failed to load linear data from path: {e}")

        model_path = "models/NAFNet-SIDD-width64.pth"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_model'):
            if "width32" in self.dock_ai.combo_denoise_model.currentText():
                model_path = "models/NAFNet-SIDD-width32.pth"

        from gui.workers.ai_denoise_worker import AIDenoiseWorker
        self.denoise_worker = AIDenoiseWorker(
            input_path=self.loaded_image_path,
            linear_rgb=linear_rgb,
            save_to_disk=save_to_disk,
            model_path=model_path,
            tile_size=tile_size,
            strength=strength,
            texture_retention=texture_retention,
            parent=self
        )
        
        self.dock_ai.progress_ai.setVisible(True)
        self.dock_ai.progress_ai.setRange(0, 100)
        self.dock_ai.progress_ai.setValue(0)
        model_name = "NAFNet Ultra (width64)" if "width64" in model_path else "NAFNet (width32)"
        self.statusBar().showMessage(f"⏳ Starting {model_name} AI Denoise on {os.path.basename(self.loaded_image_path)} (In-Memory)...")
        self.dock_ai.txt_ai_output.append(f"\n[AI Denoise] Running {model_name} noise reduction in RAM for {os.path.basename(self.loaded_image_path)} (Strength: {int(strength*100)}%, Detail: {int(texture_retention*100)}%, Tile: {tile_size}x{tile_size})...")

        self.denoise_worker.progress_updated.connect(self._on_denoise_progress)
        self.denoise_worker.status_updated.connect(self._on_denoise_status)
        self.denoise_worker.finished_in_memory.connect(self._on_denoise_finished_in_memory)
        self.denoise_worker.error_occurred.connect(self._on_denoise_error)
        self.denoise_worker.start()

    def _on_denoise_progress(self, current, total, text):
        pct = int((current / max(1, total)) * 100)
        self.dock_ai.progress_ai.setValue(pct)
        self.statusBar().showMessage(f"⏳ [AI Denoise {pct}%] {text}")
        if hasattr(self.dock_ai, 'lbl_denoise_status'):
            self.dock_ai.lbl_denoise_status.setText(f"Denoising... ({pct}%)")

    def _on_denoise_status(self, text):
        self.dock_ai.txt_ai_output.append(f"  • {text}")
        self.statusBar().showMessage(f"⏳ {text}")

    def _on_denoise_finished_in_memory(self, denoised_rgb, output_path):
        self.dock_ai.progress_ai.setVisible(False)
        
        if denoised_rgb is not None:
            if hasattr(self.engine, 'update_image_data'):
                self.engine.update_image_data(denoised_rgb, is_denoised=True)
            else:
                meta = getattr(self.engine, 'metadata', None) or {}
                self.engine.load_texture(denoised_rgb, meta)
            self._update_view()

        self.statusBar().showMessage("✅ AI Denoise complete in RAM! Ready for editing.", 8000)
        self.dock_ai.txt_ai_output.append("✅ AI Denoise finished successfully in RAM! Viewport texture updated.")
        
        if output_path:
            self.dock_ai.txt_ai_output.append(f"  • Saved TIFF copy: {output_path}")

        if hasattr(self.dock_ai, 'lbl_denoise_status'):
            self.dock_ai.lbl_denoise_status.setText("Done (Updated in RAM)")

    def _on_denoise_error(self, err_msg):
        self.dock_ai.progress_ai.setVisible(False)
        self.statusBar().showMessage(f"❌ AI Denoise failed: {err_msg}", 8000)
        self.dock_ai.txt_ai_output.append(f"❌ Error during AI Denoise: {err_msg}")
        if hasattr(self.dock_ai, 'lbl_denoise_status'):
            self.dock_ai.lbl_denoise_status.setText("Error occurred")
        QMessageBox.critical(self, "AI Denoise Error", f"Failed to perform AI Denoise:\n{err_msg}")

    def trigger_batch_ai_denoise(self, photo_items: list):
        if not photo_items:
            return

        photo_paths = []
        for item in photo_items:
            if isinstance(item, dict):
                p = item.get('path') or item.get('file_path') or ""
            else:
                p = str(item)
            if p:
                photo_paths.append(p)

        if not photo_paths:
            return

        reply = QMessageBox.question(
            self,
            "Batch AI Denoise",
            f"Found {len(photo_paths)} photo(s).\n\n"
            "Do you want to run NAFNet AI Denoise on all selected photos?\n"
            "Each photo will be processed in RAM and exported as a clean 16-bit linear TIFF in the LumenDeep folder.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        self._batch_denoise_queue = list(photo_paths)
        self._total_batch_denoise = len(photo_paths)
        self._run_next_batch_denoise()

    def _run_next_batch_denoise(self):
        if not hasattr(self, '_batch_denoise_queue') or not self._batch_denoise_queue:
            self.dock_ai.progress_ai.setVisible(False)
            self.statusBar().showMessage(f"✅ Batch AI Denoise complete for {getattr(self, '_total_batch_denoise', 0)} photos.", 8000)
            self.dock_ai.txt_ai_output.append(f"\n🎉 [Batch AI Denoise] All {getattr(self, '_total_batch_denoise', 0)} photos processed successfully!")
            QMessageBox.information(self, "Batch Complete", f"All {getattr(self, '_total_batch_denoise', 0)} photos have been denoised successfully!")
            return

        next_path = self._batch_denoise_queue.pop(0)
        idx = self._total_batch_denoise - len(self._batch_denoise_queue)
        
        tile_size = 512
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_tile'):
            if "1024" in self.dock_ai.combo_denoise_tile.currentText():
                tile_size = 1024

        strength = 1.0
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_strength'):
            strength = self.dock_ai.slider_denoise_strength.value() / 100.0

        texture_retention = 0.25
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_texture'):
            texture_retention = self.dock_ai.slider_denoise_texture.value() / 100.0

        model_path = "models/NAFNet-SIDD-width64.pth"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_model'):
            if "width32" in self.dock_ai.combo_denoise_model.currentText():
                model_path = "models/NAFNet-SIDD-width32.pth"

        from gui.workers.ai_denoise_worker import AIDenoiseWorker
        self.denoise_worker = AIDenoiseWorker(
            input_path=next_path,
            save_to_disk=True,
            model_path=model_path,
            tile_size=tile_size,
            strength=strength,
            texture_retention=texture_retention,
            parent=self
        )
        
        self.dock_ai.progress_ai.setVisible(True)
        self.dock_ai.progress_ai.setRange(0, 100)
        self.dock_ai.progress_ai.setValue(0)
        self.statusBar().showMessage(f"⏳ [{idx}/{self._total_batch_denoise}] AI Denoise: {os.path.basename(next_path)}...")
        self.dock_ai.txt_ai_output.append(f"\n[Batch AI Denoise {idx}/{self._total_batch_denoise}] Processing {os.path.basename(next_path)}...")

        self.denoise_worker.progress_updated.connect(self._on_denoise_progress)
        self.denoise_worker.status_updated.connect(self._on_denoise_status)
        
        def on_batch_item_finished(denoised_rgb, out_p):
            self.dock_ai.txt_ai_output.append(f"  ✓ Finished: {os.path.basename(out_p)}")
            self._run_next_batch_denoise()

        def on_batch_item_error(err):
            self.dock_ai.txt_ai_output.append(f"  ❌ Error on {os.path.basename(next_path)}: {err}")
            self._run_next_batch_denoise()

        self.denoise_worker.finished_in_memory.connect(on_batch_item_finished)
        self.denoise_worker.error_occurred.connect(on_batch_item_error)
        self.denoise_worker.start()

    def _start_ai_map_computation(self, map_type: str):
        if not mcp_server.state.image_loaded or not self.loaded_image_path:
            QMessageBox.information(self, "No Image Loaded", "Please open an image first before calculating AI maps.")
            return

        img_np = self._get_current_image_uint8()
        if img_np is None:
            return

        if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread and self.ai_worker_thread.isRunning():
            QMessageBox.warning(self, "AI Task Running", "An AI calculation is already in progress. Please wait.")
            return

        from gui.dialogs.ai_progress_dialog import AIProgressDialog
        self.ai_progress_dialog = AIProgressDialog(self, task_title=map_type)
        self.ai_progress_dialog.cancelled.connect(self._on_ai_cancelled)

        self.dock_ai.progress_ai.setVisible(True)
        self.dock_ai.progress_ai.setRange(0, 0)
        self.statusBar().showMessage(f"⏳ Starting AI calculation: {map_type}...")
        self.dock_ai.txt_ai_output.append(f"Starting {map_type} computation for {os.path.basename(self.loaded_image_path)} ({img_np.shape[1]}x{img_np.shape[0]}px)...")

        if map_type == "All":
            self._ai_queue = ["Depth Map", "Normal Map", "Subject/Background"]
            self._total_ai_tasks = len(self._ai_queue)
            self._current_ai_task_idx = 0
            self.ai_progress_dialog.show()
            self._run_next_queued_ai(img_np)
        else:
            self._ai_queue = []
            self._total_ai_tasks = 1
            self._current_ai_task_idx = 1
            self.ai_progress_dialog.set_task(map_type, 1, 1)
            self.ai_progress_dialog.show()
            self._launch_ai_worker(img_np, map_type)

    def _launch_ai_worker(self, img_np: np.ndarray, map_type: str):
        cached_depth = None
        if map_type in ["Normal Map", "Subject/Background"]:
            cached_depth = self._get_cached_depth_map()
            if cached_depth is not None:
                self.dock_ai.txt_ai_output.append(f"  • Re-using cached depth map ({cached_depth.shape[1]}x{cached_depth.shape[0]}px) to accelerate computation.")

        normals_method = "Hybrid (Marigold + Depth)"
        if hasattr(self.dock_ai, 'combo_normals_method'):
            normals_method = self.dock_ai.combo_normals_method.currentText()

        normals_ensemble = 1
        if hasattr(self.dock_ai, 'combo_normals_ensemble'):
            try:
                normals_ensemble = int(self.dock_ai.combo_normals_ensemble.currentText().split(" ")[0])
            except Exception:
                normals_ensemble = 1

        normals_scale = 1.0
        if hasattr(self.dock_ai, 'combo_normals_scale'):
            scale_str = self.dock_ai.combo_normals_scale.currentText()
            if "1/2" in scale_str: normals_scale = 0.5
            elif "1/4" in scale_str: normals_scale = 0.25

        disable_gpu = False
        hardware_mode = "Auto"
        if hasattr(self.dock_ai, 'get_disable_gpu'):
            disable_gpu = self.dock_ai.get_disable_gpu()
        if hasattr(self.dock_ai, 'get_hardware_mode'):
            hardware_mode = self.dock_ai.get_hardware_mode()

        self.ai_worker_thread = AIMapWorkerThread(
            img_np, map_type, self.loaded_image_path,
            high_res=True,
            cached_depth=cached_depth,
            normals_method=normals_method,
            normals_ensemble=normals_ensemble,
            normals_scale=normals_scale,
            disable_gpu=disable_gpu,
            hardware_mode=hardware_mode,
            parent=self
        )
        self.ai_worker_thread.progress.connect(self._on_ai_progress)
        self.ai_worker_thread.tile_progress.connect(self._on_ai_tile_progress)
        self.ai_worker_thread.finished.connect(self._on_ai_finished)
        self.ai_worker_thread.error.connect(self._on_ai_error)
        self.ai_worker_thread.start()

    def _run_next_queued_ai(self, img_np: np.ndarray):
        if self._ai_queue:
            next_type = self._ai_queue.pop(0)
            self._current_ai_task_idx += 1
            if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
                self.ai_progress_dialog.set_task(next_type, self._current_ai_task_idx, self._total_ai_tasks)
            self._launch_ai_worker(img_np, next_type)
        else:
            self.dock_ai.progress_ai.setVisible(False)
            self.statusBar().showMessage("✅ All AI maps calculated and loaded!")
            self.dock_ai.txt_ai_output.append("✅ All AI maps calculation complete.")
            if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
                self.ai_progress_dialog.set_finished(True, "All AI maps calculated and loaded successfully!")
            self._update_view()

    def _on_ai_progress(self, msg: str):
        self.statusBar().showMessage(f"🤖 {msg}")
        self.dock_ai.txt_ai_output.append(f"  • {msg}")
        if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
            self.ai_progress_dialog.update_progress_msg(msg)

    def _on_ai_tile_progress(self, cur: int, total: int, stage: str = "Tiling", workers: int = 1):
        self.dock_ai.progress_ai.setRange(0, total)
        self.dock_ai.progress_ai.setValue(cur)
        if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
            self.ai_progress_dialog.update_tile_progress(cur, total, stage, workers)

    def _on_ai_finished(self, map_type: str, result_data: np.ndarray, image_path: str):
        if image_path != self.loaded_image_path:
            return

        dir_name = os.path.dirname(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        lumen_dir = os.path.join(dir_name, "LumenDeep")
        os.makedirs(lumen_dir, exist_ok=True)

        from PIL import Image
        if map_type == "Depth Map":
            self._cached_depth_np = result_data
            save_path = os.path.join(lumen_dir, f"{stem}_depth.png")
            if result_data.dtype == np.float32 or result_data.dtype == np.float64:
                depth_save = (np.clip(result_data, 0.0, 1.0) * 65535.0).astype(np.uint16)
            elif result_data.dtype == np.uint8:
                depth_save = (result_data.astype(np.uint16) * 257)
            else:
                depth_save = result_data
            Image.fromarray(depth_save).save(save_path)
            self.engine.set_depth_map(result_data)
            self.dock_ai.txt_ai_output.append(f"Saved 16-bit high-res depth map to: {save_path}")
            if hasattr(self, 'dock_filmstrip') and self.dock_filmstrip and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
                self.dock_filmstrip.filmstrip_widget.set_photo_has_depth_map(image_path, True)
        elif map_type == "Normal Map":
            save_path = os.path.join(lumen_dir, f"{stem}_normals.png")
            if result_data.dtype == np.uint8:
                normal_save = result_data
            elif result_data.dtype == np.float32 or result_data.dtype == np.float64:
                if np.min(result_data) < 0.0:
                    normal_save = (np.clip((result_data * 0.5 + 0.5), 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    normal_save = (np.clip(result_data, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                normal_save = result_data
            Image.fromarray(normal_save).save(save_path)
            self.engine.set_normal_map(result_data)
            self.dock_ai.txt_ai_output.append(f"Saved normal map to: {save_path}")
        elif map_type == "Subject/Background":
            self._cached_subject_np = result_data
            save_path = os.path.join(lumen_dir, f"{stem}_subject.png")
            if result_data.dtype == np.uint8:
                subj_save = result_data
            else:
                subj_save = (np.clip(result_data, 0.0, 1.0) * 255.0).astype(np.uint8)
            Image.fromarray(subj_save).save(save_path)
            self.engine.set_subject_mask(result_data)
            self.dock_ai.txt_ai_output.append(f"Saved subject mask to: {save_path}")

        if self._ai_queue:
            img_np = self._get_current_image_uint8()
            self._run_next_queued_ai(img_np)
        else:
            self.dock_ai.progress_ai.setVisible(False)
            self.statusBar().showMessage(f"✅ {map_type} calculated successfully!")
            if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
                self.ai_progress_dialog.set_finished(True, f"{map_type} generated and loaded successfully!")
            
            # If Auto Background layer creation was pending completion of depth/subject maps, trigger it now
            if getattr(self, '_pending_auto_background_creation', False):
                self._pending_auto_background_creation = False
                self._execute_create_background_layer()
            else:
                self._update_view()

    def _on_ai_error(self, map_type: str, error_msg: str):
        self.dock_ai.progress_ai.setVisible(False)
        self._ai_queue = []
        self.dock_ai.txt_ai_output.append(f"❌ Error during {map_type}: {error_msg}")
        self.statusBar().showMessage("Ready")
        if hasattr(self, 'ai_progress_dialog') and self.ai_progress_dialog and self.ai_progress_dialog.isVisible():
            self.ai_progress_dialog.set_finished(False, error_msg)
        else:
            QMessageBox.critical(self, "AI Estimation Error", f"Failed to compute {map_type}:\n{error_msg}")

    def _on_ai_cancelled(self):
        if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread and self.ai_worker_thread.isRunning():
            self.ai_worker_thread.cancel()
        self._ai_queue = []
        self.dock_ai.progress_ai.setVisible(False)
        self.statusBar().showMessage("⚠️ AI calculation cancelled by user.")
        self.dock_ai.txt_ai_output.append("⚠️ AI calculation cancelled by user.")

    def _on_image_load_error_async(self, path: str, error_str: str, gen_id: int):
        if gen_id != self.image_load_generation_id: return
        self.image_label.hide_loading()
        self.loading_anim_timer.stop()
        QMessageBox.critical(self, "Image Load Error", f"Failed to load {os.path.basename(path)}:\n{error_str}")
        self.statusBar().showMessage("Ready")

    def _update_view(self):
        if not mcp_server.state.image_loaded or not self.engine.texture:
            return
        profiler.start_frame()
        vw = max(10, self.image_label.width())
        vh = max(10, self.image_label.height())
        self.engine.set_preview_size(vw, vh)
        self.engine.set_view(self.image_label.zoom, self.image_label.pan_x, self.image_label.pan_y)
        self.engine.split_view_mode = getattr(self, 'split_view_mode', None)
        self.engine.split_position = float(getattr(self, 'split_position', 0.5))
        if getattr(self.engine, 'bokeh_enabled', False):
            self.engine.frame_count = 0
            if hasattr(self, 'bokeh_timer') and not self.bokeh_timer.isActive():
                self.bokeh_timer.start(16)
        self.engine.render(to_preview=True)
        self.update_preview()
        if hasattr(self, 'thumb_refresh_timer'):
            self.thumb_refresh_timer.start(500)
        profiler.end_frame()

    def update_preview(self):
        if not self.engine.texture: return
        with profiler.scope("present_ms"):
            raw_bytes = self.engine.read_preview_bytes()
            if not raw_bytes: return
            w = self.engine.prev_w
            h = self.engine.prev_h
            q_img = QImage(raw_bytes, w, h, w * 3, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        self.image_label.update()

    def _toggle_profiler_hud(self, checked: bool):
        """Toggles the Real-time Performance HUD overlay in Viewport."""
        if hasattr(self, 'image_label') and hasattr(self.image_label, 'set_show_profiler_hud'):
            self.image_label.set_show_profiler_hud(checked)

    def _refresh_filmstrip_active_thumbnail(self):
        """Refreshes the Filmstrip thumbnail for the active photo after edits/adjustments."""
        if not getattr(self, 'loaded_image_path', None) or not self.engine.texture:
            return
        try:
            if hasattr(self, 'image_label') and self.image_label.pixmap():
                pm = self.image_label.pixmap()
                if pm and not pm.isNull():
                    thumb_pm = pm.scaled(QSize(360, 270), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
                        self.dock_filmstrip.filmstrip_widget.update_thumbnail_pixmap(self.loaded_image_path, thumb_pm)
        except Exception as e:
            print("Error refreshing filmstrip thumbnail:", e)

    def fit_to_viewport(self):
        self.image_label.zoom = 1.0
        self.image_label.pan_x = 0.0
        self.image_label.pan_y = 0.0
        self._update_view()

    def zoom_to_100(self):
        if hasattr(self.engine, 'width') and self.engine.width > 0:
            vp_w = max(1, self.image_label.width())
            self.image_label.zoom = float(self.engine.width) / float(vp_w)
        else:
            self.image_label.zoom = 1.0
        self.image_label.pan_x = 0.0
        self.image_label.pan_y = 0.0
        self._update_view()

    # --- Toolbar Actions ---
    def _toggle_split_v_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_split_h.setChecked(False)
            self.viewport_toolbar.btn_tb_side_by_side.setChecked(False)
            self.split_view_mode = 'vertical'
        else:
            if self.split_view_mode == 'vertical': self.split_view_mode = None
        self.engine.split_view_mode = self.split_view_mode
        self._update_view()

    def _toggle_split_h_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_split_v.setChecked(False)
            self.viewport_toolbar.btn_tb_side_by_side.setChecked(False)
            self.split_view_mode = 'horizontal'
        else:
            if self.split_view_mode == 'horizontal': self.split_view_mode = None
        self.engine.split_view_mode = self.split_view_mode
        self._update_view()

    def _toggle_side_by_side_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_split_v.setChecked(False)
            self.viewport_toolbar.btn_tb_split_h.setChecked(False)
            self.split_view_mode = 'side_by_side'
        else:
            if self.split_view_mode == 'side_by_side': self.split_view_mode = None
        self.engine.split_view_mode = self.split_view_mode
        self._update_view()

    def _toggle_mask_prev_from_tb(self, checked):
        self.engine.preview_mask = checked
        self._update_view()

    def _toggle_depth_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_normals.blockSignals(True)
            self.viewport_toolbar.btn_tb_normals.setChecked(False)
            self.viewport_toolbar.btn_tb_normals.blockSignals(False)
            self.viewport_toolbar.btn_tb_subject.blockSignals(True)
            self.viewport_toolbar.btn_tb_subject.setChecked(False)
            self.viewport_toolbar.btn_tb_subject.blockSignals(False)
            self.engine.preview_mode = 'depth'
        else:
            if self.engine.preview_mode == 'depth':
                self.engine.preview_mode = None
        if hasattr(self, 'act_view_depth'):
            self.act_view_depth.blockSignals(True)
            self.act_view_depth.setChecked(checked)
            self.act_view_depth.blockSignals(False)
        self._update_view()

    def _toggle_normals_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_depth.blockSignals(True)
            self.viewport_toolbar.btn_tb_depth.setChecked(False)
            self.viewport_toolbar.btn_tb_depth.blockSignals(False)
            self.viewport_toolbar.btn_tb_subject.blockSignals(True)
            self.viewport_toolbar.btn_tb_subject.setChecked(False)
            self.viewport_toolbar.btn_tb_subject.blockSignals(False)
            self.engine.preview_mode = 'normals'
        else:
            if self.engine.preview_mode == 'normals':
                self.engine.preview_mode = None
        if hasattr(self, 'act_view_normals'):
            self.act_view_normals.blockSignals(True)
            self.act_view_normals.setChecked(checked)
            self.act_view_normals.blockSignals(False)
        self._update_view()

    def _toggle_subject_from_tb(self, checked):
        if checked:
            self.viewport_toolbar.btn_tb_depth.blockSignals(True)
            self.viewport_toolbar.btn_tb_depth.setChecked(False)
            self.viewport_toolbar.btn_tb_depth.blockSignals(False)
            self.viewport_toolbar.btn_tb_normals.blockSignals(True)
            self.viewport_toolbar.btn_tb_normals.setChecked(False)
            self.viewport_toolbar.btn_tb_normals.blockSignals(False)
            self.engine.preview_mode = 'subject'
        else:
            if self.engine.preview_mode == 'subject':
                self.engine.preview_mode = None
        if hasattr(self, 'act_view_subject'):
            self.act_view_subject.blockSignals(True)
            self.act_view_subject.setChecked(checked)
            self.act_view_subject.blockSignals(False)
        self._update_view()

    def _toggle_crop_tool_from_tb(self, checked):
        self.image_label.is_cropping = checked
        if not checked:
            self.image_label.crop_drag_mode = None
            self.image_label.crop_drag_start_pos = None
            self.image_label.unsetCursor()
        self.engine.set_crop_interactive_mode(checked)
        self._update_view()

    def _rotate_left_90(self):
        self.engine.rotate_left_90()
        self.flip_90 = self.engine.flip_90
        self._update_view()

    def _rotate_right_90(self):
        self.engine.rotate_right_90()
        self.flip_90 = self.engine.flip_90
        self._update_view()

    def _toggle_wb_picker_from_tb(self, checked):
        self.image_label.is_picking_wb = checked
        if checked: self.image_label.setCursor(Qt.CrossCursor)
        else: self.image_label.unsetCursor()

    def _toggle_focus_picker(self, checked):
        self.image_label.is_picking_focus = checked
        if hasattr(self, 'btn_tb_focus_pick') and self.btn_tb_focus_pick.isChecked() != checked:
            self.btn_tb_focus_pick.blockSignals(True)
            self.btn_tb_focus_pick.setChecked(checked)
            self.btn_tb_focus_pick.blockSignals(False)
        if hasattr(self, 'dock_global') and hasattr(self.dock_global, 'btn_pick_focus'):
            if self.dock_global.btn_pick_focus.isChecked() != checked:
                self.dock_global.btn_pick_focus.blockSignals(True)
                self.dock_global.btn_pick_focus.setChecked(checked)
                self.dock_global.btn_pick_focus.blockSignals(False)
        if checked:
            self.image_label.setCursor(Qt.CrossCursor)
            self.statusBar().showMessage("🎯 Click on any point on the photo (e.g. eye or face) to set 3D Focal Depth...")
        else:
            self.image_label.unsetCursor()

    def _toggle_focus_picker_from_tb(self, checked):
        self._toggle_focus_picker(checked)

    def apply_picked_focus_depth(self, z_linear: float, u: float, v: float):
        # 0 on slider = Near Foreground (Z=1.0), 100 on slider = Far Background (Z=0.0)
        focal_depth = int(round((1.0 - np.clip(z_linear, 0.0, 1.0)) * 100.0))
        self.slider_bokeh_focal_depth.setValue(focal_depth)
        self.check_bokeh_enable.setChecked(True)
        self.statusBar().showMessage(f"🎯 3D Focal Plane locked to ({u*100:.0f}%, {v*100:.0f}%): Focal Depth = {focal_depth}%")
        self._toggle_focus_picker(False)

    def _toggle_brush_from_tb(self, checked):
        self.image_label.is_painting_mask = checked
        self.brush_settings_bar.setVisible(checked)
        if checked: self.image_label.setCursor(Qt.BlankCursor)
        else: self.image_label.unsetCursor()

    def _clear_brush_mask(self):
        self.engine.clear_brush_mask()
        self._update_view()

    def _toggle_mask_picker(self, checked):
        self.image_label.is_picking_mask = checked
        if checked: self.image_label.setCursor(Qt.CrossCursor)
        else: self.image_label.unsetCursor()

    # --- Parameter Handlers ---
    def update_engine_from_ui(self):
        self._on_layer_parameter_changed()
        self._on_mask_parameter_changed()
        self._on_bokeh_parameter_changed()
        self._on_global_parameter_changed()
        self._update_view()

    def _on_layer_parameter_changed(self):
        if getattr(self, '_is_updating_ui', False) or not self.engine.layers: return
        settings = self.dock_effects.get_layer_settings()
        if self.engine.layers:
            self.engine.layers[self.engine.active_layer_index].settings = settings
        self.undo_debounce_timer.start(300)
        self._update_view()

    def _on_mask_parameter_changed(self):
        if getattr(self, '_is_updating_ui', False) or not self.engine.layers: return
        cur = self.engine.layers[self.engine.active_layer_index]
        cur.mask_fill = self.dock_mask.slider_mask_fill.value() / 100.0
        cur.mask_luma_target = self.dock_mask.slider_mask_luma_t.value() / 100.0
        cur.mask_luma_range = self.dock_mask.slider_mask_luma_r.value() / 100.0
        cur.mask_hue_target = self.dock_mask.slider_mask_hue_t.value() / 360.0
        cur.mask_hue_range = self.dock_mask.slider_mask_hue_r.value() / 180.0
        cur.mask_sat_target = self.dock_mask.slider_mask_sat_t.value() / 100.0
        cur.mask_sat_range = self.dock_mask.slider_mask_sat_r.value() / 100.0
        cur.mask_softness = self.dock_mask.slider_mask_softness.value() / 100.0
        cur.mask_gradient_type = float(self.dock_mask.combo_grad_type.currentIndex())
        cur.mask_grad_black = self.dock_mask.slider_grad_black.value() / 100.0
        cur.mask_grad_white = self.dock_mask.slider_grad_white.value() / 100.0
        cur.mask_grad_gamma = self.dock_mask.slider_grad_gamma.value() / 100.0
        cur.use_depth_mask = self.dock_mask.chk_use_depth.isChecked()
        cur.mask_depth_min = self.dock_mask.slider_depth_min.value() / 100.0
        cur.mask_depth_max = self.dock_mask.slider_depth_max.value() / 100.0
        cur.mask_depth_softness = self.dock_mask.slider_depth_softness.value() / 100.0
        cur.mask_depth_gamma = self.dock_mask.slider_depth_gamma.value() / 100.0
        cur.use_normal_mask = self.dock_mask.chk_use_normal.isChecked()
        cur.mask_normal_dir = (self.dock_mask.slider_norm_x.value() / 100.0, self.dock_mask.slider_norm_y.value() / 100.0, 1.0)
        cur.mask_normal_angle = self.dock_mask.slider_norm_angle.value() / 180.0
        self._update_view()

    def _on_rotation_changed(self, val):
        self._sync_current_crop_to_active_preset()
        rot = val / 10.0
        self.lbl_rotation.setText(f"{rot:+.1f}°")
        self.engine.rotation_deg = rot
        self._update_view()

    def _on_crop_changed(self, val):
        self._sync_current_crop_to_active_preset()
        self.engine.set_crop(
            self.slider_crop_l.value() / 1000.0,
            self.slider_crop_r.value() / 1000.0,
            self.slider_crop_t.value() / 1000.0,
            self.slider_crop_b.value() / 1000.0
        )
        self._update_view()

    def _on_crop_aspect_ratio_changed(self, text: str):
        # Synchronize toolbar, global dock, AI dock, and filmstrip comboboxes
        for widget_holder, attr_name in [
            (getattr(self, 'viewport_toolbar', None), 'combo_crop_aspect'),
            (getattr(self, 'dock_global', None), 'combo_crop_aspect'),
            (getattr(self, 'dock_ai', None), 'combo_crop_aspect'),
            (getattr(getattr(self, 'dock_filmstrip', None), 'filmstrip_widget', None), 'combo_crop_aspect')
        ]:
            if widget_holder and hasattr(widget_holder, attr_name):
                combo = getattr(widget_holder, attr_name)
                if combo and combo.currentText() != text:
                    combo.blockSignals(True)
                    combo.setCurrentText(text)
                    combo.blockSignals(False)

        ratio, name = parse_aspect_ratio_text(text, self.engine.width, self.engine.height)
        self.image_label.set_crop_aspect_ratio(ratio, name)

        if ratio is not None and self.engine.width > 0 and self.engine.height > 0:
            self._fit_crop_to_aspect(ratio)
        self._sync_current_crop_to_active_preset()

    def _fit_crop_to_aspect(self, ratio: float):
        """Fits and centers the crop sliders (l, r, t, b) to match the specified aspect ratio."""
        if ratio <= 0 or self.engine.width <= 0 or self.engine.height <= 0:
            return

        cl = self.slider_crop_l.value()
        cr = self.slider_crop_r.value()
        ct = self.slider_crop_t.value()
        cb = self.slider_crop_b.value()

        new_l, new_r, new_t, new_b = enforce_crop_aspect_ratio(
            cl, cr, ct, cb, self.engine.width, self.engine.height, ratio, allow_orientation_flip=False
        )

        self.slider_crop_l.setValue(new_l)
        self.slider_crop_r.setValue(new_r)
        self.slider_crop_t.setValue(new_t)
        self.slider_crop_b.setValue(new_b)

    def _swap_crop_aspect_orientation(self):
        """Swaps active crop aspect ratio between landscape and portrait."""
        cur_ar = getattr(self.image_label, 'crop_aspect_ratio', None)
        if cur_ar is not None and cur_ar > 0:
            new_ar = 1.0 / cur_ar
            cur_name = getattr(self.image_label, 'crop_aspect_name', 'Custom')
            if "Portrait" in cur_name:
                new_name = cur_name.replace("Portrait", "Landscape")
            elif "Landscape" in cur_name:
                new_name = cur_name.replace("Landscape", "Portrait")
            else:
                new_name = f"Inverted ({new_ar:.2f}:1)"

            self.image_label.set_crop_aspect_ratio(new_ar, new_name)
            self._fit_crop_to_aspect(new_ar)
            self.statusBar().showMessage(f"🔄 Swapped Crop Aspect Ratio: {new_name}")

    def _on_bokeh_parameter_changed(self):
        # Always synchronize all bokeh parameters to engine
        self.engine.bokeh_focal_depth = 1.0 - (self.slider_bokeh_focal_depth.value() / 100.0)
        self.engine.bokeh_focal_range = self.slider_bokeh_focal_range.value() / 100.0
        self.engine.bokeh_radius = float(self.slider_bokeh_radius.value())
        petals_val = self.slider_bokeh_petals.value()
        self.engine.bokeh_petals = 0.0 if petals_val >= 10 else float(petals_val + 3)
        self.engine.bokeh_max_samples = int(self.combo_bokeh_samples.currentText()) if hasattr(self, 'combo_bokeh_samples') else 256
        self.engine.bokeh_highlight_boost = self.slider_bokeh_boost.value() / 10.0
        self.engine.bokeh_fg_gamma = self.slider_bokeh_fg_gamma.value() / 100.0
        self.engine.bokeh_bg_gamma = self.slider_bokeh_bg_gamma.value() / 100.0
        self.engine.bokeh_subject_influence = self.slider_bokeh_subject_influence.value() / 100.0
        self.engine.bokeh_rotation = float(self.slider_bokeh_rotation.value())
        self.engine.bokeh_chromatic = self.slider_bokeh_chromatic.value() / 100.0
        self.engine.bokeh_vignette = self.slider_bokeh_vignette.value() / 100.0
        self.engine.bokeh_radial_center = self.slider_bokeh_radial_center.value() / 100.0
        self.engine.bokeh_radial_edge = self.slider_bokeh_radial_edge.value() / 100.0
        self.engine.bokeh_radial_gamma = self.slider_bokeh_radial_gamma.value() / 100.0

        b_en = self.check_bokeh_enable.isChecked()
        if b_en:
            # Check if 3D depth map is available
            depth_np = self._get_cached_depth_map()
            if depth_np is None and getattr(self.engine, 'global_depth_tex', None) is None:
                self.check_bokeh_enable.blockSignals(True)
                self.check_bokeh_enable.setChecked(False)
                self.check_bokeh_enable.blockSignals(False)
                self.engine.bokeh_enabled = False
                self.bokeh_timer.stop()
                self.statusBar().showMessage("⚠️ Bokeh effect requires a 3D Depth Map. Compute Depth Map in AI & 3D Tools first.")
                if not getattr(self, '_is_updating_ui', False) and not getattr(self, '_ai_is_running', False) and not getattr(self, '_batch_ai_running', False):
                    QMessageBox.information(
                        self,
                        "3D Depth Map Required",
                        "The 3D Bokeh Simulation requires a calculated 3D Depth Map.\n\n"
                        "Please compute the Depth Map first:\n"
                        "• Go to 'AI & 3D Tools' → 'Compute Depth Map' (Ctrl+Shift+D), or\n"
                        "• Click 'Create Background Layer (3D Depth)' (Ctrl+Shift+B)."
                    )
                self._update_view()
                return

            if getattr(self.engine, 'global_depth_tex', None) is None and depth_np is not None:
                self.engine.set_depth_map(depth_np)

            self.engine.bokeh_enabled = True
            self.engine.frame_count = 0
            self.bokeh_timer.start(16)
        else:
            self.engine.bokeh_enabled = False
            self.bokeh_timer.stop()
            self.engine.frame_count = 0
        self._update_view()

    def _on_bokeh_timer(self):
        if getattr(self.engine, 'bokeh_enabled', False):
            has_more = self.engine.accumulate_bokeh_step()
            self.update_preview()
            if not has_more:
                self.bokeh_timer.stop()
        else:
            self.bokeh_timer.stop()

    def _on_global_parameter_changed(self):
        if getattr(self, 'slider_base_curve', None):
            self.engine.base_curve_strength = self.slider_base_curve.value() / 100.0
        if getattr(self, 'slider_ai_blend', None):
            self.engine.ai_blend = self.slider_ai_blend.value() / 100.0
        self._update_view()

    def apply_picked_mask_color(self, r, g, b):
        h, s, v = self._rgb_to_hsv(r, g, b)
        self.dock_mask.slider_mask_hue_t.setValue(int(h * 360))
        self.dock_mask.slider_mask_sat_t.setValue(int(s * 100))
        self.dock_mask.slider_mask_luma_t.setValue(int(v * 100))
        self._toggle_mask_picker(False)

    def apply_picked_wb_color(self, r, g, b):
        avg = (r + g + b) / 3.0
        if avg > 0.001:
            dr = (avg - r) / avg
            db = (avg - b) / avg
            temp_shift = int(np.clip((dr - db) * 100, -100, 100))
            tint_shift = int(np.clip(((avg - g) / avg) * 100, -100, 100))
            if hasattr(self, 'slider_temperature') and self.slider_exposure:
                self.slider_temperature.setValue(temp_shift)
                self.slider_tint.setValue(tint_shift)
        self._toggle_wb_picker_from_tb(False)

    @staticmethod
    def _rgb_to_hsv(r, g, b):
        max_c, min_c = max(r, g, b), min(r, g, b)
        d = max_c - min_c
        v = max_c
        s = 0 if max_c == 0 else d / max_c
        if d == 0: h = 0
        elif max_c == r: h = ((g - b) / d) % 6
        elif max_c == g: h = (b - r) / d + 2
        else: h = (r - g) / d + 4
        return (h / 6.0), s, v

    # --- Layer Stack Handlers ---
    def add_layer(self):
        return self._add_layer()

    def delete_layer(self):
        return self._delete_layer()

    def duplicate_layer(self):
        return self._duplicate_layer()

    def get_current_state(self):
        return self.serialize_state()

    def apply_state(self, state):
        return self.deserialize_state(state)

    def _apply_ai_data(self, data):
        return self.deserialize_state(data)

    def _on_image_loaded(self, path, linear_data, metadata, gen_id):
        return self._on_image_loaded_async(path, linear_data, metadata, gen_id)

    def _add_layer(self):
        new_layer = Layer(id=f"layer_{len(self.engine.layers)+1}", name=f"Layer {len(self.engine.layers)+1}")
        self.engine.layers.append(new_layer)
        self.engine.active_layer_index = len(self.engine.layers) - 1
        self.dock_layers.refresh_layers_list(self.engine.layers, self.engine.active_layer_index)
        self._update_view()

    def _delete_layer(self):
        if len(self.engine.layers) > 1:
            idx = self.layers_list.currentRow()
            del self.engine.layers[idx]
            self.dock_layers.refresh_layers_list(self.engine.layers, max(0, idx-1))
            self._update_view()

    def _duplicate_layer(self):
        if self.engine.layers:
            idx = self.layers_list.currentRow()
            cur = self.engine.layers[idx]
            import copy
            dup = copy.deepcopy(cur)
            dup.id = f"layer_{len(self.engine.layers)+1}"
            dup.name = f"{cur.name} Copy"
            self.engine.layers.insert(idx+1, dup)
            self.dock_layers.refresh_layers_list(self.engine.layers, idx+1)
            self._update_view()

    def _on_layer_selection_changed(self, row: int):
        if 0 <= row < len(self.engine.layers):
            self._is_updating_ui = True
            try:
                self.engine.active_layer_index = row
                cur = self.engine.layers[row]
                self.dock_effects.set_layer_settings(cur.settings)
                self.slider_layer_opacity.setValue(int(cur.mask_opacity * 100))
                self.dock_mask.slider_mask_fill.setValue(int(cur.mask_fill * 100))
                self.dock_mask.slider_mask_luma_t.setValue(int(cur.mask_luma_target * 100))
                self.dock_mask.slider_mask_luma_r.setValue(int(cur.mask_luma_range * 100))
                self.dock_mask.slider_mask_hue_t.setValue(int(cur.mask_hue_target * 360))
                self.dock_mask.slider_mask_hue_r.setValue(int(cur.mask_hue_range * 180))
                self.dock_mask.slider_mask_sat_t.setValue(int(cur.mask_sat_target * 100))
                self.dock_mask.slider_mask_sat_r.setValue(int(cur.mask_sat_range * 100))
                self.dock_mask.slider_mask_softness.setValue(int(cur.mask_softness * 100))
                self.dock_mask.combo_grad_type.setCurrentIndex(int(cur.mask_gradient_type))
                self.dock_mask.slider_grad_black.setValue(int(cur.mask_grad_black * 100))
                self.dock_mask.slider_grad_white.setValue(int(cur.mask_grad_white * 100))
                self.dock_mask.slider_grad_gamma.setValue(int(cur.mask_grad_gamma * 100))
                self.dock_mask.chk_use_depth.setChecked(cur.use_depth_mask)
                self.dock_mask.slider_depth_min.setValue(int(cur.mask_depth_min * 100))
                self.dock_mask.slider_depth_max.setValue(int(cur.mask_depth_max * 100))
                self.dock_mask.slider_depth_softness.setValue(int(cur.mask_depth_softness * 100))
                self.dock_mask.slider_depth_gamma.setValue(int(cur.mask_depth_gamma * 100))
                self.dock_mask.chk_use_normal.setChecked(cur.use_normal_mask)
                self.dock_mask.slider_norm_x.setValue(int(cur.mask_normal_dir[0] * 100))
                self.dock_mask.slider_norm_y.setValue(int(cur.mask_normal_dir[1] * 100))
                self.dock_mask.slider_norm_angle.setValue(int(cur.mask_normal_angle * 180))
            finally:
                self._is_updating_ui = False
            self._update_view()

    def _on_layer_item_changed(self, item):
        row = self.layers_list.row(item)
        if 0 <= row < len(self.engine.layers):
            layer = self.engine.layers[row]
            layer.active = (item.checkState() == Qt.Checked)
            new_name = item.text().strip()
            if new_name and new_name != layer.name:
                layer.name = new_name
            self._update_view()

    def _on_layer_opacity_changed(self, val):
        self.lbl_layer_opacity.setText(f"{val}%")
        if self.engine.layers:
            self.engine.layers[self.engine.active_layer_index].mask_opacity = val / 100.0
            self._update_view()

    # --- Auto Background Layer Creator ---
    def auto_create_background_layer(self):
        """
        One-click automated Background Layer isolation using 3D Depth Masking.
        Computes Depth Map and Subject Mask if missing, extracts subject boundary depth,
        and creates a dedicated 'Background' layer with 100% Depth Softness.
        """
        if not mcp_server.state.image_loaded or not self.loaded_image_path:
            QMessageBox.information(self, "No Image Loaded", "Please open an image first before creating a background layer.")
            return

        depth_np = self._get_cached_depth_map()
        subj_np = self._get_cached_subject_mask()

        # If Depth Map is already cached, we can isolate background instantly without running neural network!
        tasks_needed = []
        if depth_np is None:
            tasks_needed.append("Depth Map")
        if subj_np is None and depth_np is None:
            tasks_needed.append("Subject/Background")

        if tasks_needed:
            self._pending_auto_background_creation = True
            img_np = self._get_current_image_uint8()
            if img_np is None:
                return

            if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread and self.ai_worker_thread.isRunning():
                QMessageBox.warning(self, "AI Task Running", "An AI calculation is already in progress. Please wait.")
                return

            from gui.dialogs.ai_progress_dialog import AIProgressDialog
            self.ai_progress_dialog = AIProgressDialog(self, task_title="Auto Background Isolation")
            self.ai_progress_dialog.cancelled.connect(self._on_ai_cancelled)

            self.dock_ai.progress_ai.setVisible(True)
            self.dock_ai.progress_ai.setRange(0, 0)
            self.statusBar().showMessage("⏳ Calculating Depth Map & Subject Mask for Auto Background...")
            self.dock_ai.txt_ai_output.append(f"\n⏳ Starting computation of required AI maps ({', '.join(tasks_needed)}) for Background layer...")

            self._ai_queue = tasks_needed.copy()
            self._total_ai_tasks = len(self._ai_queue)
            self._current_ai_task_idx = 0
            self.ai_progress_dialog.show()
            self._run_next_queued_ai(img_np)
        else:
            self._execute_create_background_layer()

    def _execute_create_background_layer(self):
        depth_np = self._get_cached_depth_map()
        subj_np = self._get_cached_subject_mask()

        if depth_np is None:
            QMessageBox.warning(self, "AI Maps Missing", "Failed to retrieve Depth Map for background isolation.")
            return

        if subj_np is None:
            # Fallback to direct depth thresholding (0.45 depth metric) for instant background isolation
            min_depth_pct = 45
            focus_depth_pct = 20
            
            # Find or create Background layer
            bg_layer = None
            for l in self.engine.layers:
                if l.name.lower() == "background" and not l.is_base:
                    bg_layer = l
                    break
            if bg_layer is None:
                bg_layer = Layer(id=f"layer_bg_{len(self.engine.layers)+1}", name="Background", is_base=False, active=True)
                self.engine.layers.append(bg_layer)

            bg_layer.use_depth_mask = True
            bg_layer.mask_depth_min = float(min_depth_pct) / 100.0
            bg_layer.mask_depth_max = 1.10
            bg_layer.mask_depth_softness = 1.0
            bg_layer.mask_depth_gamma = 1.0

            bg_idx = self.engine.layers.index(bg_layer)
            self.engine.active_layer_index = bg_idx
            self.dock_layers.refresh_layers_list(self.engine.layers, bg_idx)
            self._on_layer_selection_changed(bg_idx)
            self._update_view()
            if getattr(self, '_pending_batch_after_depth', False):
                self._pending_batch_after_depth = False
                QTimer.singleShot(200, self._start_batch_gemini_on_current_photo)
            return

        # 1. Normalize subject mask to [0.0, 1.0]
        if subj_np.dtype == np.uint8:
            s_mask = subj_np.astype(np.float32) / 255.0
        else:
            s_mask = np.clip(subj_np.astype(np.float32), 0.0, 1.0)

        # 2. Normalize depth map to [0.0, 1.0] and invert to match shader depth metric
        # In shader: d = 1.0 - global_depth_tex.r (0.0=foreground, 1.0=background)
        if depth_np.dtype == np.uint16:
            d_norm = depth_np.astype(np.float32) / 65535.0
        elif depth_np.dtype == np.uint8:
            d_norm = depth_np.astype(np.float32) / 255.0
        else:
            d_norm = np.clip(depth_np.astype(np.float32), 0.0, 1.0)
        d_metric = 1.0 - d_norm

        # Resize depth metric to match subject mask if dimensions differ
        if d_metric.shape != s_mask.shape:
            from PIL import Image
            d_img = Image.fromarray(d_metric).resize((s_mask.shape[1], s_mask.shape[0]), Image.Resampling.BILINEAR)
            d_metric = np.array(d_img, dtype=np.float32)

        # 3. Extract confident bright subject pixels with erosion to avoid background boundary bleed
        import scipy.ndimage as ndi
        bright_mask = s_mask >= 0.70
        eroded_mask = ndi.binary_erosion(bright_mask, iterations=3)
        if not np.any(eroded_mask):
            eroded_mask = s_mask >= 0.50
        if not np.any(eroded_mask):
            eroded_mask = s_mask > 0.10

        if np.any(eroded_mask):
            subject_depths = d_metric[eroded_mask]
            # Ignore 3% furthest outliers to prevent edge background contamination (97th percentile)
            furthest_subject_depth = float(np.percentile(subject_depths, 97))
            min_depth = np.clip(furthest_subject_depth, 0.0, 0.95)
            # Find nearest / frontmost subject depth (3rd percentile) for Bokeh Focus
            nearest_subject_depth = float(np.percentile(subject_depths, 3))
            focus_depth = np.clip(nearest_subject_depth, 0.0, 1.0)
        else:
            min_depth = 0.30
            focus_depth = 0.20

        min_depth_pct = int(round(min_depth * 100))
        focus_depth_pct = int(round(focus_depth * 100))

        # Set Bokeh focal depth to the frontmost point of the subject and quality samples to 256
        if hasattr(self, 'slider_bokeh_focal_depth'):
            self.slider_bokeh_focal_depth.setValue(focus_depth_pct)
        self.engine.bokeh_focal_depth = 1.0 - (focus_depth_pct / 100.0)
        if hasattr(self, 'combo_bokeh_samples'):
            self.combo_bokeh_samples.blockSignals(True)
            self.combo_bokeh_samples.setCurrentText("256")
            self.combo_bokeh_samples.blockSignals(False)
        self.engine.bokeh_max_samples = 256

        # 4. Check if Background layer already exists or create new
        bg_layer = None
        for l in self.engine.layers:
            if l.name.lower() == "background":
                bg_layer = l
                break

        if bg_layer is None:
            bg_layer = Layer(
                id=f"layer_{len(self.engine.layers)+1}",
                name="Background",
                active=True,
                is_base=False
            )
            self.engine.layers.append(bg_layer)

        bg_layer.use_depth_mask = True
        bg_layer.mask_depth_min = min_depth_pct / 100.0
        bg_layer.mask_depth_max = 1.10 # 110% (disables far-side falloff so horizon/background stays 100% white)
        bg_layer.mask_depth_softness = 1.0 # 100% Depth Softness
        bg_layer.mask_depth_gamma = 1.0

        bg_idx = self.engine.layers.index(bg_layer)
        self.engine.active_layer_index = bg_idx
        self.dock_layers.refresh_layers_list(self.engine.layers, bg_idx)
        self._on_layer_selection_changed(bg_idx)

        self.statusBar().showMessage(f"🌄 Background Layer created! Min Depth: {min_depth_pct}%, Bokeh Focus: {focus_depth_pct}%, Max Depth: 110%, Softness: 100%.")
        self.dock_ai.txt_ai_output.append(f"\n🌄 [Background Layer Created]\n• Analyzed Subject Mask & Depth Map\n• Subject furthest boundary (Min Depth): {min_depth_pct}%\n• Subject frontmost point (Bokeh Focus): {focus_depth_pct}%\n• Enable 3D Depth Mask: ON (Min: {min_depth_pct}%, Max: 110%, Softness: 100%)\n")
        self._update_view()

        if getattr(self, '_pending_batch_after_depth', False):
            self._pending_batch_after_depth = False
            QTimer.singleShot(200, self._start_batch_gemini_on_current_photo)

    # --- AI & Gemini Assistant ---
    def trigger_ai_auto_correct(self):
        if getattr(self, 'fast_viewer_mode', False):
            self._on_viewer_mode_toggled(False)

        # If 2 or more photos are selected in Filmstrip, delegate to batch AI auto-correct
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            selected = self.dock_filmstrip.filmstrip_widget.get_selected_photos(include_rejected=False)
            if len(selected) >= 2:
                self.trigger_batch_ai_auto_correct(custom_photos_list=selected)
                return

        target_aspect = "Free"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_crop_aspect'):
            target_aspect = self.dock_ai.combo_crop_aspect.currentText()
        elif hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'combo_crop_aspect'):
            target_aspect = self.viewport_toolbar.combo_crop_aspect.currentText()

        if target_aspect and not target_aspect.startswith("Free"):
            self._on_crop_aspect_ratio_changed(target_aspect)

        user_prompt = self.dock_ai.get_user_prompt()
        if target_aspect and not target_aspect.startswith("Free"):
            user_prompt = f"TARGET ASPECT RATIO: {target_aspect}.\n" + (user_prompt or "")

        self._start_gemini_assistant(user_prompt=user_prompt, is_auto_correct=True)

    def trigger_batch_ai_auto_correct(self, custom_photos_list: Optional[list] = None):
        """
        Batch processes photos using Gemini AI Auto-Correction.
        If custom_photos_list is provided or 2+ photos are selected on Filmstrip,
        processes only those selected photos; otherwise processes all visible non-rejected photos.
        """
        if not hasattr(self, 'dock_filmstrip') or not hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            QMessageBox.information(self, "No Filmstrip", "Filmstrip widget is not available.")
            return

        fw = self.dock_filmstrip.filmstrip_widget
        if custom_photos_list and isinstance(custom_photos_list, list) and len(custom_photos_list) > 0:
            photos = custom_photos_list
            is_selection_batch = True
        else:
            selected_photos = fw.get_selected_photos(include_rejected=False)
            if len(selected_photos) >= 2:
                photos = selected_photos
                is_selection_batch = True
            else:
                photos = fw.get_visible_non_rejected_photos()
                is_selection_batch = False

        if not photos:
            QMessageBox.information(
                self, "No Photos Found",
                "No photos currently selected or visible in Filmstrip matching your active filter (excluding rejected photos).\n\n"
                "Please make sure photos are loaded in the Filmstrip and not marked for deletion."
            )
            return

        cfg = self.dock_ai.get_gemini_config()
        api_key = cfg.get("api_key")
        if not api_key:
            QMessageBox.warning(
                self, "Gemini API Key Required",
                "Google Gemini API Key is missing.\n\nPlease enter your API Key in the AI panel on the left dock or add GEMINI_API_KEY to your .env file."
            )
            return

        max_passes = cfg.get("max_passes", 5)
        n_depth_needed = sum(1 for p in photos if p.get('depth_needed', False))
        auto_export = self.dock_ai.is_batch_auto_export_enabled() if hasattr(self, 'dock_ai') else False
        skip_bg = self.dock_ai.is_batch_skip_background_enabled() if hasattr(self, 'dock_ai') else False
        dual_bokeh = self.dock_ai.is_batch_dual_bokeh_export_enabled() if hasattr(self, 'dock_ai') else False
        custom_folder = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
        export_desc = f"Enabled ({custom_folder if custom_folder else 'Subfolder: LDExport'}) [JPEG 85%]" if auto_export else "Disabled"
        if auto_export and dual_bokeh:
            export_desc += " [Dual: _bokeh.jpg + .jpg]"
        bg_desc = "🚫 Removed & Ignored (All adjustments to Base Layer)" if skip_bg else f"Auto-created if needed ({n_depth_needed} photos flagged +3D)"
        scope_label = f"{len(photos)} selected photo(s)" if is_selection_batch else f"{len(photos)} photo(s) currently displayed in the Filmstrip"

        auto_denoise_enabled = self.dock_ai.is_auto_denoise_enabled() if hasattr(self, 'dock_ai') else True
        iso_threshold = self.dock_ai.get_auto_denoise_iso_threshold() if hasattr(self, 'dock_ai') else 200
        denoise_desc = f"Enabled for ISO > {iso_threshold}" if auto_denoise_enabled else "Disabled"

        reply = QMessageBox.question(
            self,
            "Run Batch AI Auto-Correction",
            f"Process {scope_label} with AI Auto-Correction?\n\n"
            f"• Max Passes per photo: {max_passes}\n"
            f"• Background Layer: {bg_desc}\n"
            f"• Auto-Export on Complete: {export_desc}\n"
            f"• Auto-Denoise (In-Memory): {denoise_desc}\n"
            f"• Marked for Deletion (❌): Ignored\n\n"
            "Do you want to start batch processing?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        # Automatically disable Fast Viewer when starting Batch AI
        if getattr(self, 'fast_viewer_mode', False):
            self._on_viewer_mode_toggled(False)

        self._batch_ai_running = True
        self._batch_skip_bg = skip_bg
        self._batch_ai_queue = photos
        self._batch_ai_total = len(photos)
        self._batch_ai_current_idx = 0
        self._set_batch_ui_active(True, status_text=f"Batch AI (0/{self._batch_ai_total})...")

        self.dock_ai.txt_ai_output.append(f"\n========================================================")
        self.dock_ai.txt_ai_output.append(f"⚡ BATCH AI AUTO-CORRECTION STARTED: {self._batch_ai_total} photo(s)")
        self.dock_ai.txt_ai_output.append(f"• Scope: {scope_label}")
        self.dock_ai.txt_ai_output.append(f"• Max passes per photo: {max_passes}")
        self.dock_ai.txt_ai_output.append(f"• Background Layer: {bg_desc}")
        self.dock_ai.txt_ai_output.append(f"• Auto-Export: {export_desc}")
        self.dock_ai.txt_ai_output.append(f"• Auto-Denoise (In-Memory): {denoise_desc}")
        self.dock_ai.txt_ai_output.append(f"• Tip: Press Escape or click '⏹ Stop' at any time to cancel.")
        self.dock_ai.txt_ai_output.append(f"========================================================\n")

        self._run_next_batch_photo()

    def export_batch_photos(self, custom_photos_list: Optional[list] = None):
        """
        Batch exports multiple photos (e.g. selected photos from Filmstrip).
        Renders each photo full-resolution on GPU and saves to disk without blocking.
        """
        if not hasattr(self, 'dock_filmstrip') or not hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            return

        fw = self.dock_filmstrip.filmstrip_widget
        if custom_photos_list and isinstance(custom_photos_list, list) and len(custom_photos_list) > 0:
            photos = custom_photos_list
        else:
            selected = fw.get_selected_photos(include_rejected=False)
            photos = selected if len(selected) >= 2 else fw.get_visible_non_rejected_photos()

        if not photos:
            QMessageBox.information(self, "No Photos Found", "No photos are selected or visible to export.")
            return

        first_p = photos[0]['path'] if isinstance(photos[0], dict) else photos[0]
        default_dir = os.path.dirname(first_p)
        custom_dir = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
        export_dir = custom_dir if custom_dir else os.path.join(default_dir, "LDExport")
        os.makedirs(export_dir, exist_ok=True)

        res = QMessageBox.question(
            self,
            "Batch Export Photos",
            f"Export {len(photos)} photo(s) to:\n{export_dir}\n\n"
            f"• Slices & sorts into aspect subfolders (Original, Photo_iPad_4x5, Portrait_2x3, Square_1x1, Story_3x4, Cinematic_16x9)\n"
            f"• 5★ top picks save an additional copy in root LDExport\n\n"
            f"Format: High-Quality JPEG (85% Quality, 8-bit)\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if res != QMessageBox.Yes:
            return

        from gui.dialogs.export_dialog import ExportProgressDialog
        progress_dlg = ExportProgressDialog(self, destination_path=export_dir, dimensions=f"{len(photos)} photos batch")
        progress_dlg.show()
        QApplication.processEvents()

        total = len(photos)
        exported_count = 0

        for idx, photo in enumerate(photos):
            if progress_dlg.is_cancelled():
                break

            target_p = photo['path'] if isinstance(photo, dict) else str(photo)
            pct = int((idx / total) * 100)
            progress_dlg.set_progress(pct, f"Exporting photo {idx + 1}/{total}: {os.path.basename(target_p)}...")
            QApplication.processEvents()

            try:
                self.open_image_file(target_p)
                if getattr(self, 'image_loader_thread', None) and self.image_loader_thread.isRunning():
                    self.image_loader_thread.wait(5000)

                # Export using the aspect subfolder sorting & 5-star rule!
                self._export_current_batch_photo()
                exported_count += 1
            except Exception as e:
                print(f"Error exporting {target_p}: {e}")

        progress_dlg.close()
        self.statusBar().showMessage(f"✅ Batch export complete: {exported_count} photo(s) saved to {export_dir}")
        QMessageBox.information(
            self,
            "Batch Export Complete",
            f"Successfully exported {exported_count} of {total} photo(s) to:\n{export_dir}"
        )

    def _run_next_batch_photo(self):
        if not getattr(self, '_batch_ai_running', False):
            return

        if self._batch_ai_current_idx >= len(self._batch_ai_queue):
            self._finish_batch_ai()
            return

        self._batch_photo_fail_count = 0
        photo = self._batch_ai_queue[self._batch_ai_current_idx]
        target_path = photo['path']
        photo_name = os.path.basename(target_path)
        stem = os.path.splitext(photo_name)[0]
        photo_dir = os.path.dirname(target_path)

        # Highlight in filmstrip
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            self.dock_filmstrip.filmstrip_widget.set_active_image(target_path)

        # Check if "Skip already processed" is enabled
        skip_processed = hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_skip_already_processed_enabled') and self.dock_ai.is_batch_skip_already_processed_enabled()
        if skip_processed:
            sidecar_candidates = [
                os.path.join(photo_dir, "LumenDeep", f"{stem}.lumendeep"),
                os.path.join(photo_dir, f"{stem}.lumendeep"),
                os.path.join(photo_dir, "LumenDeep", f"{stem}.json"),
                os.path.join(photo_dir, f"{stem}.json"),
            ]
            has_project = any(os.path.exists(c) for c in sidecar_candidates)

            custom_dir = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
            export_dir = custom_dir if custom_dir else os.path.join(photo_dir, "LDExport")
            export_candidates = [
                os.path.join(export_dir, "Original", f"{stem}.jpg"),
                os.path.join(export_dir, f"{stem}.jpg"),
                os.path.join(export_dir, "Photo_iPad_4x5", f"{stem}.jpg")
            ]
            has_export = any(os.path.exists(c) for c in export_candidates)

            needs_denoise = self._should_auto_denoise_current_photo(target_path)

            is_sidecar_denoised = False
            if needs_denoise:
                for sc in sidecar_candidates:
                    if os.path.exists(sc):
                        try:
                            with open(sc, 'r', encoding='utf-8') as f:
                                sc_data = json.load(f)
                                if sc_data.get('is_denoised') or sc_data.get('has_ai_denoised'):
                                    is_sidecar_denoised = True
                                    break
                        except Exception:
                            pass

            if has_project and has_export and (not needs_denoise or is_sidecar_denoised):
                # 100% finished, already exported, and (denoise not needed OR already denoised): skip immediately!
                self.dock_ai.txt_ai_output.append(f"⏩ [Skip {self._batch_ai_current_idx + 1}/{self._batch_ai_total}] {photo_name} is already processed & exported.")
                self._batch_ai_current_idx += 1
                QTimer.singleShot(40, self._run_next_batch_photo)
                return

        # Check if the target photo is ALREADY 100% loaded as true RAW sensor data
        is_same_file = (self.loaded_image_path and os.path.abspath(self.loaded_image_path) == os.path.abspath(target_path))
        is_loader_busy = (getattr(self, 'image_loader_thread', None) is not None and self.image_loader_thread.isRunning())
        is_passthrough = getattr(self.engine, 'is_passthrough', False)

        if is_same_file and not is_loader_busy and not is_passthrough and self.engine.texture is not None:
            self._batch_waiting_for_full_image = False
            self._proceed_batch_photo_ai()
        else:
            self._batch_waiting_for_full_image = True
            self.open_image_file(target_path)

    def _finish_batch_ai(self):
        self._batch_ai_running = False
        self._batch_skip_bg = False
        self._set_batch_ui_active(False)
        self.statusBar().showMessage("🎉 Batch AI Auto-Correction finished for all photos!")
        self.dock_ai.txt_ai_output.append("\n========================================================")
        self.dock_ai.txt_ai_output.append(f"🎉 BATCH AI PROCESSING COMPLETED: {self._batch_ai_total} photo(s) processed.")
        self.dock_ai.txt_ai_output.append("========================================================\n")

    def _proceed_batch_photo_ai(self):
        if not getattr(self, '_batch_ai_running', False):
            return
        if self._batch_ai_current_idx >= len(self._batch_ai_queue):
            self._finish_batch_ai()
            return

        photo = self._batch_ai_queue[self._batch_ai_current_idx]
        pass_tag = f"Photo {self._batch_ai_current_idx + 1}/{self._batch_ai_total}"
        photo_name = os.path.basename(photo['path'])
        stem = os.path.splitext(photo_name)[0]
        photo_dir = os.path.dirname(photo['path'])

        # Check if "Skip already processed" is enabled:
        # If project is already saved, export it now (with Auto-Denoise if ISO > threshold) and skip AI re-tuning!
        skip_processed = hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_skip_already_processed_enabled') and self.dock_ai.is_batch_skip_already_processed_enabled()
        if skip_processed:
            sidecar_candidates = [
                os.path.join(photo_dir, "LumenDeep", f"{stem}.lumendeep"),
                os.path.join(photo_dir, f"{stem}.lumendeep"),
                os.path.join(photo_dir, "LumenDeep", f"{stem}.json"),
                os.path.join(photo_dir, f"{stem}.json"),
            ]
            has_project = any(os.path.exists(c) for c in sidecar_candidates)
            if has_project:
                self.dock_ai.txt_ai_output.append(f"\n--- [{pass_tag}] {photo_name} ---")
                self.dock_ai.txt_ai_output.append(f"⏩ [Fast Export] {photo_name} already tuned in saved project. Exporting without AI re-processing...")
                if self._should_auto_denoise_current_photo(photo['path']):
                    def on_fast_export_done():
                        if not getattr(self, '_batch_ai_running', False):
                            return
                        self._synchronize_all_color_equalizers()
                        self.engine.render(to_preview=True)
                        self._save_current_image_sidecar()
                        self._refresh_filmstrip_active_thumbnail()
                        self._export_current_batch_photo()
                        self._batch_ai_current_idx += 1
                        QTimer.singleShot(300, self._run_next_batch_photo)
                    self._run_auto_denoise_then_finalize(on_done_callback=on_fast_export_done)
                    return
                else:
                    self._synchronize_all_color_equalizers()
                    self.engine.render(to_preview=True)
                    self._export_current_batch_photo()
                    self._batch_ai_current_idx += 1
                    QTimer.singleShot(150, self._run_next_batch_photo)
                    return

        self.dock_ai.txt_ai_output.append(f"\n--- [{pass_tag}] {photo_name} ---")
        self.dock_ai.txt_ai_output.append(f"• Rating: {photo.get('rating', 0)}⭐ | 3D Depth Needed: {photo.get('depth_needed', False)}")

        # Check if user requested resetting settings to neutral baseline before AI
        should_reset = hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_reset_settings_enabled') and self.dock_ai.is_batch_reset_settings_enabled()
        if should_reset:
            self.reset_settings()
            self.dock_ai.txt_ai_output.append("🔄 [Clean Baseline] Reset all adjustments and layers to neutral baseline before AI tuning.")
        else:
            # Always reset previous crop boundaries & presets to uncropped baseline before AI tuning
            # so Gemini inspects the pristine 100% full-frame uncropped scene and generates fresh crops
            self._is_updating_crop_preset = True
            self.crop_presets = []
            self.active_crop_preset_id = "default"
            self.slider_crop_l.setValue(0)
            self.slider_crop_r.setValue(0)
            self.slider_crop_t.setValue(0)
            self.slider_crop_b.setValue(0)
            self.engine.set_crop(0.0, 0.0, 0.0, 0.0)
            self._is_updating_crop_preset = False
            self._populate_crop_presets_ui()
            self._update_view()

        skip_bg = getattr(self, '_batch_skip_bg', False) or (hasattr(self, 'dock_ai') and self.dock_ai.is_batch_skip_background_enabled())
        if skip_bg:
            # Delete Background layer if present on the photo, and do not create it
            bg_layers = [l for l in self.engine.layers if l.name.lower() == "background" and not l.is_base]
            if bg_layers:
                self.engine.layers = [l for l in self.engine.layers if l.name.lower() != "background" or l.is_base]
                self.engine.active_layer_index = min(self.engine.active_layer_index, len(self.engine.layers) - 1)
                self.dock_layers.refresh_layers_list(self.engine.layers, self.engine.active_layer_index)
                self._on_layer_selection_changed(self.engine.active_layer_index)
                self.dock_ai.txt_ai_output.append("🚫 [Skip Background] Removed Background layer before AI tuning.")
            self._start_batch_gemini_on_current_photo()
        else:
            depth_needed = photo.get('depth_needed', False)
            depth_np = self._get_cached_depth_map()
            has_bg_layer = any(l.name.lower() == "background" for l in self.engine.layers)

            if depth_needed and (depth_np is None or not has_bg_layer):
                self.dock_ai.txt_ai_output.append("🧊 3D Depth needed on photo. Running 'Create Background Layer (3D Depth)' first...")
                self._pending_batch_after_depth = True
                self.auto_create_background_layer()
            else:
                self._start_batch_gemini_on_current_photo()

    def _start_batch_gemini_on_current_photo(self):
        # Apply target crop aspect ratio constraint before sending to Gemini if selected
        target_aspect = "Free"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_crop_aspect'):
            target_aspect = self.dock_ai.combo_crop_aspect.currentText()
        elif hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'combo_crop_aspect'):
            target_aspect = self.viewport_toolbar.combo_crop_aspect.currentText()

        if target_aspect and not target_aspect.startswith("Free"):
            self._on_crop_aspect_ratio_changed(target_aspect)

        skip_bg = getattr(self, '_batch_skip_bg', False) or (hasattr(self, 'dock_ai') and self.dock_ai.is_batch_skip_background_enabled())
        user_prompt = self.dock_ai.get_user_prompt()
        if skip_bg:
            user_prompt = "CRITICAL: DO NOT CREATE OR ADJUST A SEPARATE BACKGROUND LAYER. Apply all adjustments exclusively to the Base Layer (Layer 0).\n" + (user_prompt or "")
        if target_aspect and not target_aspect.startswith("Free"):
            user_prompt = f"TARGET ASPECT RATIO: {target_aspect}.\n" + (user_prompt or "")

        self._start_gemini_assistant(user_prompt=user_prompt, is_auto_correct=True)

    def trigger_full_auto_cull_and_process(self):
        """
        1-Click Unified Workflow:
        1. Runs Gemini AI Photo Culling & 3D Depth assessment on all photos in the current folder.
        2. Filters out rejected/blurry photos and activates Keepers.
        3. Sequentially performs Batch AI Auto-Correction on all keeper photos:
           - Automatically creates 3D Depth Map + Background Layer if flagged with 'depth_needed' (+3D).
           - Executes multi-pass Gemini RAW auto-correction.
           - Saves .lumendeep project sidecar and updates preview thumbnails.
        """
        if not hasattr(self, 'dock_filmstrip') or not hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            return

        fw = self.dock_filmstrip.filmstrip_widget
        folder = fw.current_folder
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(self, "No Folder Selected", "Please select a photo folder in the Folders panel first.")
            return

        paths = []
        for i in range(fw.list_widget.count()):
            item = fw.list_widget.item(i)
            if item and item.data(Qt.UserRole):
                paths.append(item.data(Qt.UserRole))

        if not paths:
            QMessageBox.information(self, "No Photos Found", "No photos found in the current folder.")
            return

        cfg = self.dock_ai.get_gemini_config()
        api_key = cfg.get("api_key")
        if not api_key:
            QMessageBox.warning(
                self, "Gemini API Key Required",
                "Google Gemini API Key is missing.\n\nPlease enter your API Key in the AI panel on the left dock or add GEMINI_API_KEY to your .env file."
            )
            return

        # Ensure Auto-Export is enabled so rated 4+ photos are automatically exported
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'chk_batch_auto_export'):
            self.dock_ai.chk_batch_auto_export.setChecked(True)

        max_passes = cfg.get("max_passes", 5)
        auto_export = self.dock_ai.is_batch_auto_export_enabled() if hasattr(self, 'dock_ai') else True
        custom_folder = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
        export_desc = f"Enabled ({custom_folder if custom_folder else 'Subfolder: LDExport'}) [JPEG 85%]" if auto_export else "Disabled"

        reply = QMessageBox.question(
            self,
            "🚀 Run Full Auto Workflow (Culling + AI Processing)",
            f"Launch Complete End-to-End Automated Workflow on {len(paths)} photo(s)?\n\n"
            f"1️⃣ Step 1: AI Photo Culling & 3D Depth Assessment\n"
            f"   • Screen all photos, evaluate sharpness, reject motion-blurred faces\n"
            f"   • De-duplicate burst sequences, rate photos (1-5★), flag portrait scenes (+3D)\n\n"
            f"2️⃣ Step 2: Batch AI Auto-Correction & Export (Top photos Rating ≥ 4★ only, Max {max_passes} passes)\n"
            f"   • Automatically filter and process ONLY photos rated 4 or 5 stars\n"
            f"   • Auto-create 3D Depth Map & Background Layer if marked +3D\n"
            f"   • Execute Gemini iterative color, tone, and geometry/crop correction\n"
            f"   • Auto-Export on Complete: {export_desc}\n"
            f"   • Save .lumendeep project sidecars and refresh thumbnails\n\n"
            "Do you want to start the full automated workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        # Automatically disable Fast Viewer when starting Full Auto workflow
        if getattr(self, 'fast_viewer_mode', False):
            self._on_viewer_mode_toggled(False)

        self._full_auto_workflow_running = True
        self._set_batch_ui_active(True, status_text="[Step 1/2] AI Culling...")
        self.dock_ai.txt_ai_output.append("\n========================================================")
        self.dock_ai.txt_ai_output.append(f"🚀 FULL AUTO WORKFLOW STARTED: {len(paths)} photos in {os.path.basename(folder)}")
        self.dock_ai.txt_ai_output.append("• Step 1/2: Running Gemini AI Culling & Micro-focus Screening...")
        self.dock_ai.txt_ai_output.append("• Criteria: Only photos with Rating ≥ 4★ (4-5 Stars) will be processed & exported.")
        self.dock_ai.txt_ai_output.append(f"• Auto-Export: {export_desc}")
        self.dock_ai.txt_ai_output.append("• Tip: Press Escape or click '⏹ Stop' at any time to cancel.")
        self.dock_ai.txt_ai_output.append("========================================================\n")

        self.statusBar().showMessage(f"🚀 [Step 1/2] Running Gemini AI Culling on {len(paths)} photos (Press Esc to stop)...")

        from gui.workers.ai_culling_worker import AICullingWorker
        self.culling_worker = AICullingWorker(
            photo_paths=paths,
            api_key=api_key,
            thinking_level=cfg.get("thinking_level", "low"),
            enable_two_phase=True,
            parent=self
        )

        self.culling_worker.photo_evaluated.connect(self._on_full_auto_culling_photo_evaluated)
        self.culling_worker.progress_updated.connect(self._on_full_auto_culling_progress)
        self.culling_worker.finished_success.connect(self._on_full_auto_culling_finished)
        self.culling_worker.error_occurred.connect(self._on_full_auto_culling_error)
        self.culling_worker.start()

    def _on_full_auto_culling_photo_evaluated(self, path: str, rating: int, is_rejected: bool, depth_needed: bool, reason: str):
        fw = self.dock_filmstrip.filmstrip_widget
        fw.set_photo_ai_assessment(path, rating, is_rejected, depth_needed, reason)

    def _on_full_auto_culling_progress(self, current: int, total: int, status_text: str):
        self.statusBar().showMessage(f"🚀 [Step 1/2 Culling] {status_text}")
        if hasattr(self.dock_ai, 'progress_ai'):
            self.dock_ai.progress_ai.setFormat(f"[Step 1/2] {status_text}")

    def _on_full_auto_culling_finished(self, results: list):
        if not getattr(self, '_full_auto_workflow_running', False):
            return

        fw = self.dock_filmstrip.filmstrip_widget
        # Ensure rejected photos are hidden
        fw.filter_hide_rejected = True
        if hasattr(fw, 'btn_filter_reject'):
            fw.btn_filter_reject.blockSignals(True)
            fw.btn_filter_reject.setChecked(True)
            fw.btn_filter_reject.blockSignals(False)

        # Set rating filter to 4+ stars in Filmstrip UI and filter
        fw.filter_rating_min = 4
        if hasattr(fw, 'combo_filter_rating'):
            fw.combo_filter_rating.blockSignals(True)
            fw.combo_filter_rating.setCurrentIndex(4)  # "★ 4+ Stars"
            fw.combo_filter_rating.blockSignals(False)

        fw._apply_filters()

        n_total = len(results)
        n_rejected = sum(1 for r in results if r.get('is_rejected', False))
        n_qualifying = sum(1 for r in results if not r.get('is_rejected', False) and r.get('rating', 0) >= 4)
        n_below_threshold = sum(1 for r in results if not r.get('is_rejected', False) and r.get('rating', 0) < 4)
        n_3d = sum(1 for r in results if r.get('depth_needed', False) and not r.get('is_rejected', False) and r.get('rating', 0) >= 4)

        self.dock_ai.txt_ai_output.append(
            f"\n✅ Step 1/2 Complete: {n_total} photos screened "
            f"({n_qualifying} rated ≥ 4★, {n_below_threshold} below 4★ [skipped], {n_rejected} rejected, {n_3d} requiring 3D Depth)."
        )
        self.dock_ai.txt_ai_output.append(
            f"• Step 2/2: Starting Batch AI Auto-Correction & Export on {n_qualifying} photo(s) with rating ≥ 4★...\n"
        )
        self.statusBar().showMessage(f"🚀 Step 1 Complete ({n_qualifying} photos rated ≥ 4★). Starting Step 2: Batch AI Auto-Correction & Export...")

        # Get the visible keeper photos strictly with rating >= 4
        photos = [p for p in fw.get_visible_non_rejected_photos() if p.get('rating', 0) >= 4]
        if not photos:
            self._full_auto_workflow_running = False
            self._set_batch_ui_active(False)
            QMessageBox.information(
                self, "Full Auto Workflow",
                f"AI Culling completed ({n_total} photo(s) screened).\n\n"
                f"• {n_qualifying} photo(s) rated 4-5★\n"
                f"• {n_below_threshold} photo(s) rated < 4★ (skipped)\n"
                f"• {n_rejected} photo(s) rejected\n\n"
                "No photos with rating ≥ 4★ were found to process and export."
            )
            return

        self._batch_ai_running = True
        self._batch_ai_queue = photos
        self._batch_ai_total = len(photos)
        self._batch_ai_current_idx = 0
        self._set_batch_ui_active(True, status_text=f"Batch AI (0/{self._batch_ai_total})...")

        self._run_next_batch_photo()

    def _on_full_auto_culling_error(self, err_msg: str):
        self._full_auto_workflow_running = False
        self._set_batch_ui_active(False)
        self.statusBar().showMessage("❌ AI Culling Error")
        self.dock_ai.txt_ai_output.append(f"\n❌ Full Auto Culling Error: {err_msg}\n")
        QMessageBox.critical(self, "AI Culling Error", f"Full Auto Workflow encountered an error during Culling:\n\n{err_msg}")

    def _finish_batch_ai(self):
        self._batch_ai_running = False
        self._batch_ai_queue = []
        self._set_batch_ui_active(False)
        self.statusBar().showMessage(f"🎉 Batch AI Auto-Correction complete! Processed {self._batch_ai_total} photos.")
        self.dock_ai.txt_ai_output.append(f"\n========================================================")
        self.dock_ai.txt_ai_output.append(f"🎉 BATCH AI COMPLETE: All {self._batch_ai_total} photos processed & saved!")
        self.dock_ai.txt_ai_output.append(f"========================================================\n")
        if getattr(self, '_full_auto_workflow_running', False):
            self._full_auto_workflow_running = False
            QMessageBox.information(
                self, "Full Auto Workflow Complete",
                f"🎉 Full Auto Workflow (Culling + AI Auto-Correction + Export) successfully finished!\n\n"
                f"• Processed & exported {self._batch_ai_total} top keeper photo(s) (Rating ≥ 4★).\n"
                f"• All 3D depth maps, background layers, project sidecars, and exported JPEGs saved."
            )
        else:
            QMessageBox.information(
                self, "Batch AI Complete",
                f"Successfully processed all {self._batch_ai_total} photos in the Filmstrip!\n\n"
                "All project sidecars and preview caches have been saved."
            )

    def _set_batch_ui_active(self, active: bool, status_text: str = ""):
        """Synchronizes loading and Stop button visibility across all panels (Dock, Viewport Toolbar, Filmstrip)."""
        if hasattr(self, 'dock_ai'):
            self.dock_ai.set_loading(active, status_text=status_text)
            if hasattr(self.dock_ai, 'btn_stop_ai'):
                self.dock_ai.btn_stop_ai.setVisible(active)
        if hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'btn_tb_stop_ai'):
            self.viewport_toolbar.btn_tb_stop_ai.setVisible(active)
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            fw = self.dock_filmstrip.filmstrip_widget
            if hasattr(fw, 'btn_stop_batch'):
                fw.btn_stop_batch.setVisible(active)
            if hasattr(fw, 'btn_ai_culling'):
                fw.btn_ai_culling.setEnabled(not active)
            if hasattr(fw, 'btn_full_auto'):
                fw.btn_full_auto.setEnabled(not active)
            if hasattr(fw, 'btn_batch_crop'):
                fw.btn_batch_crop.setEnabled(not active)
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'btn_batch_crop'):
            self.dock_ai.btn_batch_crop.setEnabled(not active)

    def stop_ai_auto_correct(self):
        """Instantly terminates all active AI operations (Batch, Gemini Assistant, AI Culling, AI Cropping)."""
        self._ai_is_running = False
        self._batch_ai_running = False
        self._batch_crop_running = False
        self._full_auto_workflow_running = False
        self._pending_batch_after_depth = False
        self._batch_ai_queue = []
        self._batch_crop_queue = []
        
        if hasattr(self, 'image_label'):
            self.image_label.hide_ai_processing()
        
        if hasattr(self, 'culling_worker') and self.culling_worker and self.culling_worker.isRunning():
            self.culling_worker.cancel()
        if hasattr(self, 'gemini_worker') and self.gemini_worker and self.gemini_worker.isRunning():
            self.gemini_worker.cancel()
        if hasattr(self, 'crop_worker') and self.crop_worker and self.crop_worker.isRunning():
            self.crop_worker.cancel()
        if hasattr(self, 'bokeh_verifier_worker') and self.bokeh_verifier_worker and self.bokeh_verifier_worker.isRunning():
            self.bokeh_verifier_worker.cancel()
        if hasattr(self, 'denoise_worker') and self.denoise_worker and self.denoise_worker.isRunning():
            self.denoise_worker.cancel()

        self._set_batch_ui_active(False)
        self.statusBar().showMessage("⏹ AI / Batch Auto-Correction stopped by user.")
        self.dock_ai.txt_ai_output.append("\n⏹ Process stopped by user.\n")

    def _on_apply_prompt_style(self):
        one_time_cmd = self.dock_ai.get_and_clear_one_time_prompt()
        if not one_time_cmd:
            QMessageBox.information(self, "AI Command", "Please enter a directive for Gemini (e.g. 'Focus on the child's face') or click one of the quick preset chips.")
            return

        target_aspect = "Free"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_crop_aspect'):
            target_aspect = self.dock_ai.combo_crop_aspect.currentText()
        elif hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'combo_crop_aspect'):
            target_aspect = self.viewport_toolbar.combo_crop_aspect.currentText()

        if target_aspect and not target_aspect.startswith("Free"):
            self._on_crop_aspect_ratio_changed(target_aspect)
            one_time_cmd = f"TARGET ASPECT RATIO: {target_aspect}.\n" + one_time_cmd

        self._start_gemini_assistant(user_prompt=one_time_cmd, is_auto_correct=False)

    def _start_gemini_assistant(self, user_prompt: str = "", is_auto_correct: bool = False):
        if not mcp_server.state.image_loaded or not self.engine.texture:
            QMessageBox.information(self, "No Image Loaded", "Please open an image first before using Gemini Assistant.")
            return

        # Ensure full 32-bit RAW sensor data has completed decoding before starting Gemini!
        if getattr(self, 'image_loader_thread', None) is not None and self.image_loader_thread.isRunning():
            self.statusBar().showMessage("⏳ Waiting for full 32-bit RAW sensor data to finish decoding before starting Gemini...")
            self.dock_ai.txt_ai_output.append("⏳ Waiting for full 32-bit RAW file to finish decoding...")
            self._pending_gemini_start = (user_prompt, is_auto_correct)
            return

        cfg = self.dock_ai.get_gemini_config()
        api_key = cfg.get("api_key")
        if not api_key:
            QMessageBox.warning(self, "Gemini API Key Required", 
                                "Google Gemini API Key is missing.\n\nPlease enter your API Key in the '🔑 Google Gemini API Key' field on the left dock or add GEMINI_API_KEY to your .env file.")
            return

        if is_auto_correct:
            should_reset = hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_reset_settings_enabled') and self.dock_ai.is_batch_reset_settings_enabled()
            if should_reset:
                self.reset_settings()
                self.dock_ai.txt_ai_output.append("🔄 [Clean Baseline] Reset all adjustments and layers to neutral baseline before AI tuning.")

        self._ai_config = cfg
        self._ai_is_auto_correct = is_auto_correct
        self._ai_user_prompt = user_prompt
        self._ai_current_pass = 0
        self._ai_max_passes = cfg.get("max_passes", 5) if is_auto_correct else 1
        self._ai_prev_state = self.serialize_state()
        
        # Inject aspect ratio and image dimensions into state
        aspect_ratio = getattr(self.image_label, 'crop_aspect_ratio', None)
        aspect_name = getattr(self.image_label, 'crop_aspect_name', 'Free')
        if aspect_ratio is not None and aspect_ratio > 0:
            self._ai_prev_state['aspect_ratio'] = aspect_ratio
            self._ai_prev_state['aspect_name'] = aspect_name
            if 'geometry_crop' not in self._ai_prev_state or not isinstance(self._ai_prev_state['geometry_crop'], dict):
                self._ai_prev_state['geometry_crop'] = {}
            self._ai_prev_state['geometry_crop']['aspect_ratio'] = aspect_ratio
            self._ai_prev_state['geometry_crop']['aspect_name'] = aspect_name
            self._ai_prev_state['geometry_crop']['image_width'] = self.engine.width
            self._ai_prev_state['geometry_crop']['image_height'] = self.engine.height

        self._ai_is_running = True

        # Push state to undo stack before starting the iterative process
        self.push_undo_state()

        task_desc = f"Auto-Correction (Max {self._ai_max_passes} passes)" if is_auto_correct else f"Prompt: '{user_prompt}'"
        self.dock_ai.txt_ai_output.append(f"\n========================================")
        self.dock_ai.txt_ai_output.append(f"✨ Gemini 3.8 Flash: {task_desc}")
        if is_auto_correct:
            self.dock_ai.txt_ai_output.append(f"• Refinement rule: Adaptive step (75% on passes 1-3, 50% on later passes) until convergence.")
        self.dock_ai.txt_ai_output.append(f"• Thinking Level: {cfg.get('thinking_level', 'medium')} | Resolution: {cfg.get('ai_image_size', 1024)}px")

        self._run_next_ai_pass()

    def _run_next_ai_pass(self):
        if not getattr(self, '_ai_is_running', False):
            return

        self._ai_current_pass += 1
        pass_tag = f"Pass {self._ai_current_pass}/{self._ai_max_passes}"
        self.dock_ai.set_loading(True, status_text=pass_tag)
        if hasattr(self, 'image_label'):
            self.image_label.show_ai_processing(pass_text=pass_tag, status_text="Thinking & generating adjustments...")
        self.loading_anim_timer.start()
        self.statusBar().showMessage(f"📸 Gemini 3.8 Flash ({pass_tag}): Capturing converged viewport photo...")

        # Downscale viewport image to 1024px (or user setting) to save tokens, time and costs
        ai_size = self._ai_config.get("ai_image_size", 1024)
        img_cropped, debug_path = self.engine.get_viewport_cropped_image(
            self.image_label,
            fully_accumulate_bokeh=True,
            save_debug_copy=False,
            max_size=ai_size,
            force_uncropped=(self._ai_current_pass == 1)
        )

        self.dock_ai.txt_ai_output.append(f"\n--- [{pass_tag}] ---")
        self.dock_ai.txt_ai_output.append(f"• Viewport image: {img_cropped.width}x{img_cropped.height}px (black borders trimmed)")
        self.dock_ai.txt_ai_output.append("🧠 Gemini is reasoning and determining adjustments...")

        self.statusBar().showMessage(f"✨ Gemini 3.8 Flash ({pass_tag}): Thinking & generating adjustments...")

        # Extract active layer mask preview if available to give Gemini multimodal mask context
        mask_image = None
        if self.engine.layers and len(self.engine.layers) > 1:
            cur_layer = self.engine.layers[self.engine.active_layer_index]
            if getattr(cur_layer, 'use_depth_mask', False):
                depth_np = self._get_cached_depth_map()
                if depth_np is not None:
                    from PIL import Image
                    d_norm = np.clip(depth_np.astype(np.float32) / (65535.0 if depth_np.dtype == np.uint16 else (255.0 if depth_np.dtype == np.uint8 else 1.0)), 0.0, 1.0)
                    d_metric = 1.0 - d_norm
                    d_min = cur_layer.mask_depth_min
                    d_max = cur_layer.mask_depth_max
                    d_soft = cur_layer.mask_depth_softness
                    u = np.clip((d_metric - d_min) / max(0.0001, d_max - d_min), 0.0, 1.0)
                    w_edge = max(0.0001, d_soft * 0.5)
                    t_up = np.clip(u / w_edge, 0.0, 1.0)
                    ramp_up = t_up * t_up * (3.0 - 2.0 * t_up)
                    t_dn = np.clip((1.0 - u) / w_edge, 0.0, 1.0)
                    ramp_down = t_dn * t_dn * (3.0 - 2.0 * t_dn)
                    mask_calc = np.where((d_metric >= d_min) & (d_metric <= d_max), ramp_up * ramp_down, 0.0)
                    mask_calc = (np.clip(mask_calc, 0.0, 1.0) * 255.0).astype(np.uint8)
                    mask_img = Image.fromarray(mask_calc)
                    mask_img.thumbnail((768, 768))
                    mask_image = mask_img

        self.gemini_worker = GeminiAssistantWorker(
            image=img_cropped,
            user_prompt=self._ai_user_prompt,
            current_state=self._ai_prev_state,
            is_auto_correct=self._ai_is_auto_correct,
            pass_num=self._ai_current_pass,
            total_passes=self._ai_max_passes,
            api_key=self._ai_config.get("api_key"),
            thinking_level=self._ai_config.get("thinking_level", "medium"),
            mask_image=mask_image,
            parent=self
        )

        self.gemini_worker.thought_received.connect(self._on_gemini_thought)
        self.gemini_worker.chunk_received.connect(self._on_gemini_chunk)
        def _update_gemini_status(s):
            self.statusBar().showMessage(f"✨ Gemini ({pass_tag}): {s}")
            if hasattr(self, 'image_label') and getattr(self.image_label, 'is_ai_running', False):
                self.image_label.update_ai_status(pass_text=pass_tag, status_text=s)
        self.gemini_worker.status_updated.connect(_update_gemini_status)
        self.gemini_worker.finished_success.connect(self._on_gemini_pass_success)
        self.gemini_worker.error_occurred.connect(self._on_gemini_error)
        self.gemini_worker.start()

    def _on_gemini_thought(self, text: str):
        self.dock_ai.txt_ai_output.moveCursor(self.dock_ai.txt_ai_output.textCursor().MoveOperation.End)
        self.dock_ai.txt_ai_output.insertPlainText(text)
        if hasattr(self, 'image_label') and getattr(self.image_label, 'is_ai_running', False):
            pass_tag = f"Pass {self._ai_current_pass}/{self._ai_max_passes}"
            self.image_label.update_ai_status(pass_text=pass_tag, status_text="Reasoning (Thinking stream)...")

    def _on_gemini_chunk(self, text: str):
        pass

    def _on_gemini_pass_success(self, parsed_data: dict, explanation: str):
        if not getattr(self, '_ai_is_running', False):
            return

        pass_num = self._ai_current_pass
        max_passes = self._ai_max_passes

        # Step factor: Adaptive annealing (75% on passes 1-3 for bold progression, 50% on later passes for smooth micro-refinement)
        if max_passes > 1:
            alpha = 0.75 if pass_num <= 3 else 0.50
        else:
            alpha = 1.0

        # Protect depth mask bounds during the first half of passes
        lock_depth = (pass_num <= max(1, max_passes // 2))

        # Inject current crop aspect ratio and dimensions into _ai_prev_state
        aspect_ratio = getattr(self.image_label, 'crop_aspect_ratio', None)
        aspect_name = getattr(self.image_label, 'crop_aspect_name', 'Free')
        if aspect_ratio is not None and aspect_ratio > 0:
            self._ai_prev_state['aspect_ratio'] = aspect_ratio
            self._ai_prev_state['aspect_name'] = aspect_name
            if 'geometry_crop' not in self._ai_prev_state or not isinstance(self._ai_prev_state['geometry_crop'], dict):
                self._ai_prev_state['geometry_crop'] = {}
            self._ai_prev_state['geometry_crop']['aspect_ratio'] = aspect_ratio
            self._ai_prev_state['geometry_crop']['aspect_name'] = aspect_name
            self._ai_prev_state['geometry_crop']['image_width'] = self.engine.width
            self._ai_prev_state['geometry_crop']['image_height'] = self.engine.height

        # Blend new suggested parameters with previous state
        blended_state = blend_state_dicts(self._ai_prev_state, parsed_data, alpha, lock_depth_bounds=lock_depth)
        
        # Calculate domain-aware deltas (separating color grading from framing/bokeh noise)
        grading_delta, total_delta, max_param = compute_state_deltas(self._ai_prev_state, blended_state)
        delta = max(grading_delta, total_delta)

        # Apply the blended state to UI and GPU Engine immediately for live ViewPort update
        self.deserialize_state(blended_state)
        self._update_view()

        self.dock_ai.txt_ai_output.append(f"\n💡 Gemini Pass {pass_num} Explanation:\n{explanation}\n")
        self.dock_ai.txt_ai_output.append(
            f"• Applied {int(alpha*100)}% step | Grading Δ: {grading_delta:.1f}, Max parameter: '{max_param}'"
        )

        # Check stopping criteria: converged if grading delta < 2.5 and total delta < 3.5
        is_converged = (pass_num >= 2 and grading_delta < 2.5 and total_delta < 3.5)
        if pass_num >= max_passes:
            self._finalize_ai_photo_tuning(pass_num, max_passes, delta, converged=False)
        elif is_converged:
            self._finalize_ai_photo_tuning(pass_num, max_passes, delta, converged=True)
        else:
            # Update state reference and schedule next iteration
            self._ai_prev_state = blended_state
            self.statusBar().showMessage(f"✨ Pass {pass_num} applied (Grading Δ: {grading_delta:.1f}). Preparing Pass {pass_num + 1}...")
            QTimer.singleShot(250, self._run_next_ai_pass)

    def _finalize_ai_photo_tuning(self, pass_num: int, max_passes: int, delta: float, converged: bool = False):
        self._ai_is_running = False
        if converged:
            self.dock_ai.txt_ai_output.append(f"\n🎉 Refinement converged (visual grading delta {delta:.1f} < 2.5) at Pass {pass_num}!\n")
        else:
            self.dock_ai.txt_ai_output.append(f"\n🎉 Completed all {max_passes} refinement passes!\n")

        # Check if Background Layer or 3D Bokeh are active and user enabled AI QC verification
        should_run_qc = hasattr(self, 'dock_ai') and getattr(self.dock_ai, 'chk_enable_ai_qc', None) and self.dock_ai.chk_enable_ai_qc.isChecked()
        
        bg_layer = None
        for l in self.engine.layers:
            if l.name.lower() == "background" and l.active and not l.is_base:
                bg_layer = l
                break
        has_bg = (bg_layer is not None)
        has_bokeh = self.check_bokeh_enable.isChecked() or getattr(self.engine, 'bokeh_enabled', False)

        if should_run_qc and (has_bg or has_bokeh):
            self._run_ai_quality_control_pass(has_bg=has_bg, has_bokeh=has_bokeh, bg_layer=bg_layer)
        else:
            self._complete_ai_photo_tuning()

    def _run_ai_quality_control_pass(self, has_bg: bool, has_bokeh: bool, bg_layer=None):
        """
        Captures proposed edit vs clean base grade (and without bokeh) to evaluate with Gemini:
        1. Background Layer naturalness vs unnatural cutout halos / edge artifacts.
        2. 3D Bokeh blur quality & subject sharpness protection.
        """
        self.statusBar().showMessage("🔍 Running AI Quality Control (Background Naturalness & Bokeh Sharpness)...")
        self.dock_ai.txt_ai_output.append("🔍 Running AI Quality Control A/B Verification Pass...")
        if hasattr(self, 'image_label'):
            self.image_label.show_ai_processing(pass_text="AI QC", status_text="Evaluating background & bokeh naturalness...")
            self.loading_anim_timer.start()

        ai_qc_res = 768
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_ai_res'):
            txt = self.dock_ai.combo_ai_res.currentText()
            if "768px" in txt: ai_qc_res = 768
            elif "512px" in txt: ai_qc_res = 512
            elif "1024px" in txt: ai_qc_res = 1024
            elif "1536px" in txt: ai_qc_res = 1536

        # 1. Fast capture Full Proposed Edit (1024px)
        if bg_layer: bg_layer.active = True
        self.engine.bokeh_enabled = has_bokeh
        self.engine.render(to_preview=True)
        img_final, _ = self.engine.get_viewport_cropped_image(
            viewport_label=self.image_label,
            fully_accumulate_bokeh=False,
            save_debug_copy=False,
            max_size=ai_qc_res
        )

        # 2. If both Background and Bokeh are active, capture intermediate (Background ONLY, without Bokeh)
        img_no_bokeh = None
        if has_bg and has_bokeh:
            if bg_layer: bg_layer.active = True
            self.engine.bokeh_enabled = False
            self.engine.render(to_preview=True)
            img_no_bokeh, _ = self.engine.get_viewport_cropped_image(
                viewport_label=self.image_label,
                fully_accumulate_bokeh=False,
                save_debug_copy=False,
                max_size=ai_qc_res
            )

        # 3. Capture Clean Base Grade (Without Background layer and Without Bokeh)
        if bg_layer: bg_layer.active = False
        self.engine.bokeh_enabled = False
        self.engine.render(to_preview=True)
        img_clean_base, _ = self.engine.get_viewport_cropped_image(
            viewport_label=self.image_label,
            fully_accumulate_bokeh=False,
            save_debug_copy=False,
            max_size=ai_qc_res
        )

        # 4. Restore live view state while worker is assessing
        if bg_layer: bg_layer.active = has_bg
        self.engine.bokeh_enabled = has_bokeh
        self.engine.render(to_preview=True)

        api_key = self._ai_config.get("api_key")
        if not api_key:
            self._complete_ai_photo_tuning()
            return

        self._qc_target_bg_layer = bg_layer
        self._qc_had_bg = has_bg
        self._qc_had_bokeh = has_bokeh

        # Fast thinking level ("low") for instant QC decision
        self.qc_verifier_worker = GeminiQualityControlWorker(
            img_final=img_final,
            img_clean_base=img_clean_base,
            img_no_bokeh=img_no_bokeh,
            has_bg=has_bg,
            has_bokeh=has_bokeh,
            api_key=api_key,
            thinking_level="low",
            parent=self
        )
        self.qc_verifier_worker.status_updated.connect(lambda s: self.statusBar().showMessage(f"🔍 AI QC: {s}"))
        self.qc_verifier_worker.finished_decision.connect(self._on_qc_verifier_success)
        self.qc_verifier_worker.error_occurred.connect(self._on_qc_verifier_error)
        self.qc_verifier_worker.start()

    def _on_qc_verifier_success(self, keep_bg: bool, keep_bokeh: bool, confidence: float, reason: str):
        bg_layer = getattr(self, '_qc_target_bg_layer', None)
        had_bg = getattr(self, '_qc_had_bg', False)
        had_bokeh = getattr(self, '_qc_had_bokeh', False)

        # 1. Apply Background decision
        if had_bg:
            if not keep_bg:
                if bg_layer:
                    bg_layer.active = False
                self.dock_layers.refresh_layers_list(self.engine.layers, self.engine.active_layer_index)
                self.dock_ai.txt_ai_output.append(f"❌ AI QC: Background Layer DISABLED (unnatural cutout/halo avoided).\n• Reason: {reason}\n")
                self.statusBar().showMessage("❌ Background layer disabled by AI QC (natural look preserved)")
            else:
                if bg_layer:
                    bg_layer.active = True
                self.dock_layers.refresh_layers_list(self.engine.layers, self.engine.active_layer_index)
                self.dock_ai.txt_ai_output.append(f"✅ AI QC: Background Layer CONFIRMED & KEPT.\n• Reason: {reason}\n")

        # 2. Apply Bokeh decision
        if had_bokeh:
            if not keep_bokeh:
                self.engine.bokeh_enabled = False
                self.check_bokeh_enable.setChecked(False)
                self.dock_ai.txt_ai_output.append(f"❌ AI QC: 3D Bokeh DISABLED (Subject sharpness protected).\n• Reason: {reason}\n")
                self.statusBar().showMessage("❌ 3D Bokeh disabled by AI QC check (subject protected)")
            else:
                self.engine.bokeh_enabled = True
                self.check_bokeh_enable.setChecked(True)
                self.dock_ai.txt_ai_output.append(f"✅ AI QC: 3D Bokeh CONFIRMED & KEPT.\n• Reason: {reason}\n")

        self.engine.render(to_preview=True)
        self._update_view()
        self._complete_ai_photo_tuning()

    def _on_qc_verifier_error(self, err_msg: str):
        self.dock_ai.txt_ai_output.append(f"⚠️ AI QC verification skipped: {err_msg}\n")
        self._complete_ai_photo_tuning()

    def _export_current_batch_photo(self):
        """
        Auto-exports the currently loaded photo after AI auto-correction
        categorized into clean aspect ratio subdirectories:
          • ./LDExport/Original/
          • ./LDExport/Photo_iPad_4x5/
          • ./LDExport/Portrait_2x3/
          • ./LDExport/Square_1x1/
          • ./LDExport/Story_3x4/
          • ./LDExport/Cinematic_16x9/
        
        ★ 5-Star Rule: Photos with 5★ rating get an additional top-picks copy directly in ./LDExport/.
        Photos with 4★ or less are saved ONLY in their respective aspect subfolders.
        """
        if not self.loaded_image_path or not self.engine.texture:
            return
        self._synchronize_all_color_equalizers()
        try:
            from core.image_io import linear_to_srgb_u8
            photo_dir = os.path.dirname(self.loaded_image_path)
            stem = os.path.splitext(os.path.basename(self.loaded_image_path))[0]
            
            custom_dir = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
            if custom_dir:
                try:
                    os.makedirs(custom_dir, exist_ok=True)
                    out_dir = custom_dir
                except Exception:
                    out_dir = os.path.join(photo_dir, "LDExport")
            else:
                out_dir = os.path.join(photo_dir, "LDExport")
                
            os.makedirs(out_dir, exist_ok=True)
            
            # Determine rating of current photo
            rating = 0
            if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
                fw = self.dock_filmstrip.filmstrip_widget
                if hasattr(fw, 'culling_cache') and self.loaded_image_path in fw.culling_cache:
                    rating = int(fw.culling_cache[self.loaded_image_path].get('rating', 0) or 0)
                elif hasattr(fw, 'photo_dict') and self.loaded_image_path in fw.photo_dict:
                    rating = int(fw.photo_dict[self.loaded_image_path].get('rating', 0) or 0)
            
            if rating == 0 and getattr(self, '_batch_ai_running', False):
                if 0 <= self._batch_ai_current_idx < len(self._batch_ai_queue):
                    rating = int(self._batch_ai_queue[self._batch_ai_current_idx].get('rating', 0) or 0)

            is_5_star = (rating == 5)

            def get_aspect_subfolder_name(aspect_name: str, preset_id: str = "", w: int = 0, h: int = 0) -> str:
                an = (aspect_name or "").lower()
                pid = (preset_id or "").lower()
                
                if any(k in pid or k in an for k in ['full_frame', 'full', 'uncropped', 'sensor', 'original']):
                    return "Original"
                if '4:5' in an or '5:4' in an or '4x5' in pid or '5x4' in pid or 'ipad' in pid or 'hero' in pid:
                    return "Photo_iPad_4x5"
                if '1:1' in an or 'square' in pid or '1x1' in pid or 'insta' in pid:
                    return "Square_1x1"
                if '2:3' in an or '3:2' in an or '2x3' in pid or '3x2' in pid:
                    return "Portrait_2x3"
                if '3:4' in an or '4:3' in an or '3x4' in pid or '4x3' in pid or 'story' in pid:
                    return "Story_3x4"
                if '16:9' in an or '9:16' in an or '16x9' in pid or '9x16' in pid or 'cinematic' in pid or 'reels' in pid:
                    return "Cinematic_16x9"

                if w > 0 and h > 0:
                    ratio = max(w, h) / min(w, h)
                    if abs(ratio - 1.0) < 0.06: return "Square_1x1"
                    if abs(ratio - 1.25) < 0.06: return "Photo_iPad_4x5"
                    if abs(ratio - 1.333) < 0.06: return "Story_3x4"
                    if abs(ratio - 1.5) < 0.06: return "Portrait_2x3"
                    if abs(ratio - 1.777) < 0.07: return "Cinematic_16x9"

                clean = aspect_name.replace(':', 'x').replace('/', '_').replace(' ', '_').strip()
                return clean if clean else "Original"

            # Save active crop state to render uncropped full frame for presets slicing
            saved_crop_l = self.engine.crop_l
            saved_crop_r = self.engine.crop_r
            saved_crop_t = self.engine.crop_t
            saved_crop_b = self.engine.crop_b

            crop_presets_to_export = getattr(self, 'crop_presets', [])
            has_multi_crops = len(crop_presets_to_export) > 0

            # ALWAYS render the 100% full uncropped sensor image for Original & Multi-Crop Presets Slicing!
            # Using set_crop() is essential so that OpenGL recalculates out_w, out_h, and MVP_export!
            self.engine.set_crop(0.0, 0.0, 0.0, 0.0)

            dual_bokeh = self.dock_ai.is_batch_dual_bokeh_export_enabled() if hasattr(self, 'dock_ai') else False
            bokeh_active = bool(self.check_bokeh_enable.isChecked() or getattr(self.engine, 'bokeh_enabled', False))

            # Render Full Frame / Original
            orig_subfolder = os.path.join(out_dir, "Original")
            os.makedirs(orig_subfolder, exist_ok=True)

            if dual_bokeh and bokeh_active:
                orig_bokeh_path = os.path.join(orig_subfolder, f"{stem}_bokeh.jpg")
                orig_clean_path = os.path.join(orig_subfolder, f"{stem}.jpg")

                # 1. Export Bokeh to Original subfolder
                self.engine.bokeh_enabled = True
                self.engine.render(to_preview=False)
                pixels_bokeh = self.engine.read_pixels(to_preview=False)
                export_image_from_linear(orig_bokeh_path, pixels_bokeh, apply_gamma=True, bit_depth=8, quality=85)

                # 2. Export Clean to Original subfolder
                self.engine.bokeh_enabled = False
                self.engine.render(to_preview=False)
                pixels_clean = self.engine.read_pixels(to_preview=False)
                export_image_from_linear(orig_clean_path, pixels_clean, apply_gamma=True, bit_depth=8, quality=85)

                # 3. If 5-Star: save extra copy directly in root LDExport
                if is_5_star:
                    export_image_from_linear(os.path.join(out_dir, f"{stem}_bokeh.jpg"), pixels_bokeh, apply_gamma=True, bit_depth=8, quality=85)
                    export_image_from_linear(os.path.join(out_dir, f"{stem}.jpg"), pixels_clean, apply_gamma=True, bit_depth=8, quality=85)
                    self.dock_ai.txt_ai_output.append(f"⭐ [5★ Hero] Saved root copy in LDExport/{stem}.jpg & {stem}_bokeh.jpg")

                main_pixels = pixels_clean
                self.dock_ai.txt_ai_output.append(f"💾 [Original] Saved Original/{stem}.jpg & {stem}_bokeh.jpg")
            else:
                self.engine.render(to_preview=False)
                main_pixels = self.engine.read_pixels(to_preview=False)
                orig_path = os.path.join(orig_subfolder, f"{stem}.jpg")
                export_image_from_linear(orig_path, main_pixels, apply_gamma=True, bit_depth=8, quality=85)

                if is_5_star:
                    root_path = os.path.join(out_dir, f"{stem}.jpg")
                    export_image_from_linear(root_path, main_pixels, apply_gamma=True, bit_depth=8, quality=85)
                    self.dock_ai.txt_ai_output.append(f"⭐ [5★ Hero] Saved root copy in LDExport/{stem}.jpg")

                self.dock_ai.txt_ai_output.append(f"💾 [Original] Saved Original/{stem}.jpg")

            # Multi-Crop Presets Slicing into respective subfolders from TRUE uncropped full-frame
            if has_multi_crops:
                main_u8 = linear_to_srgb_u8(main_pixels)
                H, W = main_u8.shape[:2]
                exported_records = []
                seen_subfolder_filenames = {}

                for p_idx, preset in enumerate(crop_presets_to_export):
                    p_id = preset.get('id', f"crop_{p_idx+1}")
                    p_asp = preset.get('aspect_name', '')
                    if p_id in ('full_frame', 'full', 'uncropped'):
                        continue # Already exported in Original/

                    subfolder_name = get_aspect_subfolder_name(p_asp, p_id)
                    is_portrait = (H >= W)
                    target_ratio = None
                    if subfolder_name == "Photo_iPad_4x5":
                        target_ratio = 4.0 / 5.0 if is_portrait else 5.0 / 4.0
                    elif subfolder_name == "Square_1x1":
                        target_ratio = 1.0
                    elif subfolder_name == "Story_3x4":
                        target_ratio = 3.0 / 4.0 if is_portrait else 4.0 / 3.0
                    elif subfolder_name == "Cinematic_16x9":
                        target_ratio = 9.0 / 16.0 if is_portrait else 16.0 / 9.0
                    elif subfolder_name == "Portrait_2x3":
                        target_ratio = 2.0 / 3.0 if is_portrait else 3.0 / 2.0
                    else:
                        target_ratio = preset.get('aspect_ratio')
                        if target_ratio is None or target_ratio <= 0:
                            target_ratio, _ = parse_aspect_ratio_text(p_asp, W, H)

                    cl = int(round(float(preset.get('crop_left', 0))))
                    cr = int(round(float(preset.get('crop_right', 0))))
                    ct = int(round(float(preset.get('crop_top', 0))))
                    cb = int(round(float(preset.get('crop_bottom', 0))))

                    if target_ratio and target_ratio > 0:
                        cl, cr, ct, cb = enforce_crop_aspect_ratio(
                            cl, cr, ct, cb, W, H, target_ratio, allow_orientation_flip=False
                        )

                    cl_norm = np.clip(float(cl) / 1000.0, 0.0, 0.49)
                    cr_norm = np.clip(float(cr) / 1000.0, 0.0, 0.49)
                    ct_norm = np.clip(float(ct) / 1000.0, 0.0, 0.49)
                    cb_norm = np.clip(float(cb) / 1000.0, 0.0, 0.49)

                    x0 = int(round(cl_norm * W))
                    x1 = max(x0 + 1, int(round((1.0 - cr_norm) * W)))
                    y0 = int(round(ct_norm * H))
                    y1 = max(y0 + 1, int(round((1.0 - cb_norm) * H)))

                    cropped_sub = main_u8[y0:y1, x0:x1]
                    sub_h, sub_w = cropped_sub.shape[:2]

                    if subfolder_name == "Original":
                        subfolder_name = get_aspect_subfolder_name(p_asp, p_id, sub_w, sub_h)
                    target_sub_dir = os.path.join(out_dir, subfolder_name)
                    os.makedirs(target_sub_dir, exist_ok=True)

                    clean_pid = p_id.replace('preset_', '').replace('crop_', '')
                    
                    # Clean filename: standard {stem}.jpg or {stem}_{variant}.jpg if multiple in same folder
                    if subfolder_name not in seen_subfolder_filenames:
                        target_filename = f"{stem}.jpg"
                        seen_subfolder_filenames[subfolder_name] = 1
                    else:
                        target_filename = f"{stem}_{clean_pid}.jpg"

                    preset_file = os.path.join(target_sub_dir, target_filename)
                    export_image_from_linear(preset_file, cropped_sub, apply_gamma=False, bit_depth=8, quality=85)
                    exported_records.append(f"{subfolder_name}/{target_filename}")

                    # If 5-star and this is the Hero 4:5 portrait, also save 4:5 variant to root LDExport if desired
                    if is_5_star and subfolder_name == "Photo_iPad_4x5" and "4x5" not in seen_subfolder_filenames:
                        export_image_from_linear(os.path.join(out_dir, f"{stem}_4x5.jpg"), cropped_sub, apply_gamma=False, bit_depth=8, quality=85)

                if exported_records:
                    self.dock_ai.txt_ai_output.append(f"✂️ [Sorted Crops] Exported {len(exported_records)} variants: {', '.join(exported_records)}")

            # Restore original crop state and render viewport preview
            self.engine.set_crop(saved_crop_l, saved_crop_r, saved_crop_t, saved_crop_b)
            self.engine.render(to_preview=True)
            self.update_preview()

        except Exception as e:
            print(f"Error auto-exporting photo: {e}")
            self.dock_ai.txt_ai_output.append(f"⚠️ Auto-Export warning: {e}\n")

    def _complete_ai_photo_tuning(self):
        if hasattr(self, 'image_label'):
            self.image_label.hide_ai_processing()
        if not getattr(self, '_batch_ai_running', False):
            self._set_batch_ui_active(False)

        # Ensure Bokeh Quality Samples is explicitly set to 256 after auto-tuning
        if hasattr(self, 'combo_bokeh_samples'):
            self.combo_bokeh_samples.blockSignals(True)
            self.combo_bokeh_samples.setCurrentText("256")
            self.combo_bokeh_samples.blockSignals(False)
        self.engine.bokeh_max_samples = 256

        self._synchronize_all_color_equalizers()
        self.statusBar().showMessage("✅ Gemini Auto-Correction complete!")

        # If Multi-Crop Presets generation is enabled, launch dedicated clean crop phase
        should_gen_crops = hasattr(self, 'dock_ai') and getattr(self.dock_ai, 'chk_generate_multi_crop', None) and self.dock_ai.chk_generate_multi_crop.isChecked()

        if should_gen_crops:
            # Run dedicated AI Multi-Crop phase for both single & batch processing
            self.trigger_ai_crop_presets()
        else:
            self._finalize_post_tuning_actions()

    @staticmethod
    def _parse_iso_value(*args) -> Optional[int]:
        """Safely parses any EXIF ISO representation into an integer."""
        if not args:
            return None
        val = args[-1]
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, (list, tuple)):
            return PhotoEditorWindow._parse_iso_value(val[0]) if val else None
        if isinstance(val, str):
            import re
            m = re.search(r'\d+', val)
            if m:
                try:
                    return int(m.group(0))
                except Exception:
                    pass
        return None

    def _get_current_photo_iso(self, target_path: Optional[str] = None) -> Optional[int]:
        """Returns the parsed integer ISO value of the given or currently loaded photo, or None if unavailable."""
        check_path = os.path.abspath(target_path) if target_path else (os.path.abspath(self.loaded_image_path) if self.loaded_image_path else None)

        # 1. If currently loaded in engine and matches target: check engine metadata
        if check_path and self.loaded_image_path and os.path.abspath(self.loaded_image_path) == check_path:
            meta = getattr(self.engine, 'metadata', None) or {}
            for k in ('iso', 'ISO', 'iso_speed', 'PhotographicSensitivity', 'ISOSpeedRatings', 'StandardOutputSensitivity'):
                if k in meta:
                    iso = self._parse_iso_value(meta[k])
                    if iso is not None:
                        return iso

        # 2. Check Filmstrip persistent cache
        if check_path:
            try:
                from core.filmstrip_cache import FilmstripCache
                rec = FilmstripCache.get_instance().get_photo(check_path)
                if rec and rec.get('iso'):
                    iso = self._parse_iso_value(rec['iso'])
                    if iso is not None:
                        return iso
            except Exception:
                pass

        # 3. Check Batch AI queue item
        if getattr(self, '_batch_ai_running', False) and 0 <= self._batch_ai_current_idx < len(self._batch_ai_queue):
            q_item = self._batch_ai_queue[self._batch_ai_current_idx]
            if isinstance(q_item, dict):
                q_p = os.path.abspath(q_item.get('path', ''))
                if not check_path or q_p == check_path:
                    iso = self._parse_iso_value(q_item.get('iso'))
                    if iso is not None:
                        return iso

        # 4. Check Filmstrip widget cached photo items
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget') and check_path:
            fw = self.dock_filmstrip.filmstrip_widget
            if hasattr(fw, 'photo_dict') and check_path in fw.photo_dict:
                iso = self._parse_iso_value(fw.photo_dict[check_path].get('iso'))
                if iso is not None:
                    return iso

        # 5. Direct extraction from file on disk via core.image_io.read_image_metadata
        if check_path and os.path.exists(check_path):
            try:
                from core.image_io import read_image_metadata
                file_meta = read_image_metadata(check_path)
                if file_meta:
                    for k in ('iso', 'ISO', 'iso_speed', 'PhotographicSensitivity', 'ISOSpeedRatings', 'StandardOutputSensitivity'):
                        if k in file_meta:
                            iso = self._parse_iso_value(file_meta[k])
                            if iso is not None:
                                return iso
            except Exception as e:
                print(f"[Auto-Denoise] Error reading metadata from {check_path}: {e}")

        return None

    def _should_auto_denoise_current_photo(self, target_path: Optional[str] = None) -> bool:
        """
        Checks whether the photo requires automated NAFNet In-Memory Denoise
        before export based on the user's ISO threshold setting.
        """
        if not hasattr(self, 'dock_ai'):
            return False

        enabled = False
        if hasattr(self.dock_ai, 'is_batch_auto_denoise_enabled'):
            enabled = self.dock_ai.is_batch_auto_denoise_enabled()
        elif hasattr(self.dock_ai, 'is_auto_denoise_enabled'):
            enabled = self.dock_ai.is_auto_denoise_enabled()
        if not enabled:
            return False

        check_path = target_path or self.loaded_image_path
        if not check_path:
            return False

        # If inspecting the actively loaded image and it was already denoised in this pass, skip
        if target_path is None or (self.loaded_image_path and os.path.abspath(target_path) == os.path.abspath(self.loaded_image_path)):
            if getattr(self.engine, 'has_ai_denoised', False):
                return False

        threshold = 200
        if hasattr(self.dock_ai, 'get_batch_auto_denoise_iso_threshold'):
            threshold = self.dock_ai.get_batch_auto_denoise_iso_threshold()
        elif hasattr(self.dock_ai, 'get_auto_denoise_iso_threshold'):
            threshold = self.dock_ai.get_auto_denoise_iso_threshold()

        iso = self._get_current_photo_iso(check_path)
        if iso is not None and iso > threshold:
            return True
        return False

    def _run_auto_denoise_then_finalize(self, on_done_callback=None):
        """
        Executes NAFNet In-Memory Denoise on the current photo before proceeding to export
        and next batch photo.
        """
        iso = self._get_current_photo_iso()
        threshold = self.dock_ai.get_auto_denoise_iso_threshold()
        photo_name = os.path.basename(self.loaded_image_path) if self.loaded_image_path else "photo"

        self.dock_ai.txt_ai_output.append(f"\n🔇 [Auto-Denoise] ISO {iso} > {threshold}: Running AI NAFNet In-Memory Denoise before export...")
        self.statusBar().showMessage(f"⏳ Auto-Denoise: {photo_name} (ISO {iso})...")

        tile_size = 512
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_tile'):
            if "1024" in self.dock_ai.combo_denoise_tile.currentText():
                tile_size = 1024

        save_to_disk = False
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'chk_denoise_save_tiff'):
            save_to_disk = self.dock_ai.chk_denoise_save_tiff.isChecked()

        strength = 1.0
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_strength'):
            strength = self.dock_ai.slider_denoise_strength.value() / 100.0

        texture_retention = 0.25
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'slider_denoise_texture'):
            texture_retention = self.dock_ai.slider_denoise_texture.value() / 100.0

        model_path = "models/NAFNet-SIDD-width64.pth"
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'combo_denoise_model'):
            if "width32" in self.dock_ai.combo_denoise_model.currentText():
                model_path = "models/NAFNet-SIDD-width32.pth"

        linear_rgb = None
        if hasattr(self, '_pristine_linear_raw') and self._pristine_linear_raw is not None:
            linear_rgb = self._pristine_linear_raw.copy()
        elif hasattr(self, 'engine') and self.engine and hasattr(self.engine, 'raw_texture') and self.engine.raw_texture:
            try:
                if hasattr(self.engine, 'get_linear_raw_numpy'):
                    linear_rgb = self.engine.get_linear_raw_numpy()
                else:
                    raw_bytes = self.engine.raw_texture.read()
                    tex = self.engine.raw_texture
                    dtype = np.float16 if getattr(tex, 'dtype', 'f4') == 'f2' else np.float32
                    arr_full = np.frombuffer(raw_bytes, dtype=dtype).reshape((self.engine.height, self.engine.width, tex.components))
                    linear_rgb = np.ascontiguousarray(arr_full[..., :3].astype(np.float32))
            except Exception as e:
                print(f"[Auto-Denoise] Could not extract from engine texture: {e}")
                linear_rgb = None

        if linear_rgb is None and self.loaded_image_path:
            try:
                from core.image_io import load_image_to_linear
                linear_rgb, _ = load_image_to_linear(self.loaded_image_path)
            except Exception as e:
                print(f"[Auto-Denoise] Failed to load linear data from path: {e}")

        from gui.workers.ai_denoise_worker import AIDenoiseWorker
        self.denoise_worker = AIDenoiseWorker(
            input_path=self.loaded_image_path,
            linear_rgb=linear_rgb,
            save_to_disk=save_to_disk,
            model_path=model_path,
            tile_size=tile_size,
            strength=strength,
            texture_retention=texture_retention,
            parent=self
        )

        self.dock_ai.progress_ai.setVisible(True)
        self.dock_ai.progress_ai.setRange(0, 100)
        self.dock_ai.progress_ai.setValue(0)

        self.denoise_worker.progress_updated.connect(self._on_denoise_progress)
        self.denoise_worker.status_updated.connect(self._on_denoise_status)

        def on_auto_denoise_done(denoised_rgb, out_file):
            self.dock_ai.progress_ai.setVisible(False)
            if denoised_rgb is not None:
                if hasattr(self.engine, 'update_image_data'):
                    self.engine.update_image_data(denoised_rgb, is_denoised=True)
                else:
                    meta = getattr(self.engine, 'metadata', None) or {}
                    self.engine.load_texture(denoised_rgb, meta)
                self._update_view()
            self.dock_ai.txt_ai_output.append(f"  ✓ [Auto-Denoise] In-Memory Denoise completed successfully for {photo_name}.")
            if callable(on_done_callback):
                on_done_callback()
            else:
                self._do_finalize_and_export()

        def on_auto_denoise_fail(err_msg):
            self.dock_ai.progress_ai.setVisible(False)
            self.dock_ai.txt_ai_output.append(f"  ⚠️ [Auto-Denoise] Warning on {photo_name}: {err_msg}. Continuing to export original...")
            if callable(on_done_callback):
                on_done_callback()
            else:
                self._do_finalize_and_export()

        self.denoise_worker.finished_in_memory.connect(on_auto_denoise_done)
        self.denoise_worker.error_occurred.connect(on_auto_denoise_fail)
        self.denoise_worker.start()

    def _finalize_post_tuning_actions(self):
        """Executes auto-denoise (if ISO > threshold), sidecar saving, auto-export, and batch step progression."""
        if self._should_auto_denoise_current_photo():
            self._run_auto_denoise_then_finalize()
        else:
            self._do_finalize_and_export()

    def _do_finalize_and_export(self):
        """Finalizes photo post-tuning: updates sidecar, refreshes thumbnail, exports, and proceeds to next photo."""
        self._synchronize_all_color_equalizers()
        self.engine.render(to_preview=True)
        self._save_current_image_sidecar()
        self._refresh_filmstrip_active_thumbnail()

        # Auto-Export if enabled in AI dock (exports main photo + all multi-crop presets)
        if hasattr(self, 'dock_ai') and self.dock_ai.is_batch_auto_export_enabled():
            self._export_current_batch_photo()

        if getattr(self, '_batch_crop_running', False):
            self._batch_crop_current_idx += 1
            # Give UI 1500ms to render clean thumbnail, refresh filmstrip, and switch to next photo smoothly
            QTimer.singleShot(1500, self._run_next_batch_crop_photo)
            return

        if getattr(self, '_batch_ai_running', False):
            self._batch_ai_current_idx += 1
            # Give UI 1800ms to render clean thumbnail, refresh filmstrip, and switch to next photo smoothly
            QTimer.singleShot(1800, self._run_next_batch_photo)

    def _on_gemini_error(self, err_msg: str):
        self._ai_is_running = False
        if hasattr(self, 'image_label'):
            self.image_label.hide_ai_processing()
        self.statusBar().showMessage("❌ Gemini Error")
        self.dock_ai.txt_ai_output.append(f"\n❌ Gemini 3.8 Flash Error:\n{err_msg}\n")
        
        is_fatal_key_error = any(k in (err_msg or "").lower() for k in [
            "api key not valid", "api_key_invalid", "invalid api key",
            "permission_denied", "permissiondenied", "403", "unregistered caller"
        ])
        if is_fatal_key_error:
            self._batch_ai_running = False
            self._ai_is_running = False
            self._set_batch_ui_active(False)
            self.statusBar().showMessage("❌ Invalid Gemini API Key")
            self.dock_ai.txt_ai_output.append("❌ BATCH STOPPED: Invalid or expired Gemini API Key. Please enter a valid API Key in the AI Assistant panel.\n")
            QMessageBox.critical(self, "Invalid Gemini API Key", f"Gemini API Key is invalid or expired:\n\n{err_msg}\n\nPlease enter a valid API key in the AI Assistant panel on the left.")
            return

        if getattr(self, '_batch_ai_running', False):
            self._batch_photo_fail_count = getattr(self, '_batch_photo_fail_count', 0) + 1
            total_photos = getattr(self, '_batch_ai_total', 1)
            cur_idx = getattr(self, '_batch_ai_current_idx', 0)
            
            if self._batch_photo_fail_count < 2:
                self.dock_ai.txt_ai_output.append(
                    f"⏳ [Batch Auto-Retry] API / Network issue on photo {cur_idx + 1}/{total_photos}. Reconnecting and retrying in 4s (Photo Retry {self._batch_photo_fail_count}/1)..."
                )
                self.statusBar().showMessage(f"⏳ Retrying photo {cur_idx + 1}/{total_photos} in 4s...")
                # Re-launch current photo AI grading cleanly with clean state
                QTimer.singleShot(4000, self._start_batch_gemini_on_current_photo)
            else:
                self.dock_ai.txt_ai_output.append(
                    f"⚠️ [Batch Skip] Photo {cur_idx + 1}/{total_photos} could not be processed after repeated attempts. Skipping to next photo in queue to keep batch workflow moving..."
                )
                self.statusBar().showMessage(f"⚠️ Photo failed. Advancing to next photo...")
                self._batch_photo_fail_count = 0
                self._batch_ai_current_idx += 1
                QTimer.singleShot(1500, self._run_next_batch_photo)
        else:
            self._set_batch_ui_active(False)
            QMessageBox.critical(self, "Gemini 3.8 Flash Error", f"Gemini Assistant encountered an error:\n\n{err_msg}")

    def _init_lensfun_database(self):
        try:
            import lensfunpy
            self.lf_db = lensfunpy.Database()
            self.lf_cams = self.lf_db.cameras
            self.lf_lenses = self.lf_db.lenses
            makers = sorted(list(set([c.maker for c in self.lf_cams if c.maker])))
            
            if hasattr(self, 'combo_cam_maker'):
                self.combo_cam_maker.blockSignals(True)
                self.combo_cam_maker.clear()
                self.combo_cam_maker.addItem("Auto / None")
                self.combo_cam_maker.addItems(makers)
                self.combo_cam_maker.blockSignals(False)
            
            if hasattr(self, 'combo_cam_model'):
                self.combo_cam_model.blockSignals(True)
                self.combo_cam_model.clear()
                self.combo_cam_model.addItem("Auto / None")
                self.combo_cam_model.blockSignals(False)
                
            if hasattr(self, 'combo_lens_profile'):
                self.combo_lens_profile.blockSignals(True)
                self.combo_lens_profile.clear()
                self.combo_lens_profile.addItem("Auto / None")
                self.combo_lens_profile.blockSignals(False)
        except Exception as e:
            print("Lensfun database initialization note:", e)

    def _on_lens_correction_toggled(self, checked: bool):
        self.engine.enable_lens_correction = bool(checked)
        if hasattr(self, 'combo_cam_maker'): self.combo_cam_maker.setEnabled(bool(checked))
        if hasattr(self, 'combo_cam_model'): self.combo_cam_model.setEnabled(bool(checked))
        if hasattr(self, 'combo_lens_profile'): self.combo_lens_profile.setEnabled(bool(checked))
        self.engine.apply_lens_correction()
        self._update_view()

    def _on_cam_maker_changed(self, idx=0):
        if not hasattr(self, 'combo_cam_maker') or not hasattr(self, 'lf_cams'): return
        maker = self.combo_cam_maker.currentText()
        self.combo_cam_model.blockSignals(True)
        self.combo_cam_model.clear()
        self.combo_cam_model.addItem("Auto / None")
        if maker != "Auto / None":
            models = sorted(list(set([c.model for c in self.lf_cams if c.maker == maker and c.model])))
            self.combo_cam_model.addItems(models)
        self.combo_cam_model.blockSignals(False)
        self._on_lens_profile_changed()

    def _on_cam_model_changed(self, idx=0):
        if not hasattr(self, 'combo_cam_maker') or not hasattr(self, 'combo_cam_model') or not hasattr(self, 'lf_db'): return
        maker = self.combo_cam_maker.currentText()
        model = self.combo_cam_model.currentText()
        self.combo_lens_profile.blockSignals(True)
        self.combo_lens_profile.clear()
        self.combo_lens_profile.addItem("Auto / None")
        if maker != "Auto / None" and model != "Auto / None":
            cams = self.lf_db.find_cameras(maker, model)
            if cams:
                cam = cams[0]
                lenses = self.lf_db.find_lenses(cam, '', '')
                lens_names = sorted(list(set([f"{l.maker} // {l.model}" for l in lenses if l.maker and l.model])))
                self.combo_lens_profile.addItems(lens_names)
        self.combo_lens_profile.blockSignals(False)
        self._on_lens_profile_changed()

    def _on_lens_profile_changed(self, idx=0):
        if not hasattr(self, 'engine') or not self.engine: return
        if not getattr(self.engine, 'metadata', None):
            self.engine.metadata = {}
        
        maker = self.combo_cam_maker.currentText() if hasattr(self, 'combo_cam_maker') else "Auto / None"
        model = self.combo_cam_model.currentText() if hasattr(self, 'combo_cam_model') else "Auto / None"
        lens_full = self.combo_lens_profile.currentText() if hasattr(self, 'combo_lens_profile') else "Auto / None"
        
        if maker != "Auto / None": self.engine.metadata['cam_maker'] = maker
        if model != "Auto / None": self.engine.metadata['cam_model'] = model
        
        if lens_full != "Auto / None" and " // " in lens_full:
            lens_maker, lens_model = lens_full.split(" // ", 1)
            self.engine.metadata['lens_maker'] = lens_maker
            self.engine.metadata['lens_model'] = lens_model
            
        self.engine.update_lens_uv_map()
        self.engine.apply_lens_correction()
        self._update_view()

    def _sync_lens_combos_from_metadata(self, metadata: dict):
        if not metadata or not hasattr(self, 'combo_cam_maker'):
            return
        import difflib

        def _set_combo_fuzzy(combo, search_text, cutoff=0.3):
            if not search_text: return
            items = [combo.itemText(i) for i in range(combo.count())]
            
            def sanitize(s):
                return s.lower().replace(" ", "").replace("-", "").replace("f/", "f").replace("canon", "").replace("nikon", "").replace("sony", "")
                
            search_san = sanitize(search_text)
            items_san = {sanitize(item): i for i, item in enumerate(items)}
            
            # Substring match first
            for i, item in enumerate(items):
                if search_text.lower() in item.lower() or search_san in sanitize(item):
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    return
                    
            matches = difflib.get_close_matches(search_san, items_san.keys(), n=1, cutoff=cutoff)
            if matches:
                combo.blockSignals(True)
                combo.setCurrentIndex(items_san[matches[0]])
                combo.blockSignals(False)

        maker = metadata.get('cam_maker', '')
        model = metadata.get('cam_model', '')
        lens_maker = metadata.get('lens_maker', '')
        lens_model = metadata.get('lens_model', '')
        
        if maker:
            _set_combo_fuzzy(self.combo_cam_maker, maker, cutoff=0.5)
            self._on_cam_maker_changed()
        if model:
            _set_combo_fuzzy(self.combo_cam_model, model, cutoff=0.3)
            self._on_cam_model_changed()
        if lens_maker or lens_model:
            search_str = f"{lens_maker} {lens_model}".strip() if (lens_maker and lens_model) else (lens_model or lens_maker)
            _set_combo_fuzzy(self.combo_lens_profile, search_str, cutoff=0.3)
            self._on_lens_profile_changed()

    def _setup_mcp_actions(self):
        pass

    # --- Serialization & Undo/Redo ---

    # =========================================================================
    # Multi-Crop Presets Management
    # =========================================================================
    def _populate_crop_presets_ui(self):
        if hasattr(self, 'dock_global') and hasattr(self.dock_global, 'update_crop_presets_list'):
            self.dock_global.update_crop_presets_list(self.crop_presets, self.active_crop_preset_id)
        if hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'update_crop_presets_list'):
            self.viewport_toolbar.update_crop_presets_list(self.crop_presets, self.active_crop_preset_id)

    def _sync_current_crop_to_active_preset(self):
        if self._is_updating_crop_preset or not self.crop_presets:
            return
        
        cl = self.slider_crop_l.value()
        cr = self.slider_crop_r.value()
        ct = self.slider_crop_t.value()
        cb = self.slider_crop_b.value()
        st = self.slider_rotation.value()
        asp = getattr(self.image_label, 'crop_aspect_name', 'Free')

        for p in self.crop_presets:
            if p.get('id') == self.active_crop_preset_id:
                p['crop_left'] = cl
                p['crop_right'] = cr
                p['crop_top'] = ct
                p['crop_bottom'] = cb
                p['straighten'] = st
                p['aspect_name'] = asp
                break

    def _on_crop_preset_changed(self, idx: int):
        if self._is_updating_crop_preset:
            return
        
        from PySide6.QtWidgets import QComboBox
        sender = self.sender()
        preset_id = None
        if sender and isinstance(sender, QComboBox):
            preset_id = sender.itemData(idx)
            
        if not preset_id:
            if hasattr(self, 'viewport_toolbar') and hasattr(self.viewport_toolbar, 'combo_crop_presets'):
                preset_id = self.viewport_toolbar.combo_crop_presets.itemData(idx)
            if not preset_id and hasattr(self, 'dock_global') and hasattr(self.dock_global, 'combo_crop_presets'):
                preset_id = self.dock_global.combo_crop_presets.itemData(idx)

        if not preset_id:
            return

        target_preset = None
        for p in self.crop_presets:
            if p.get('id') == preset_id:
                target_preset = p
                break

        if not target_preset:
            return

        self._is_updating_crop_preset = True
        try:
            self.active_crop_preset_id = preset_id

            # Synchronize both comboboxes (top toolbar + global dock) without trigger loops
            for cb_holder in [getattr(self, 'viewport_toolbar', None), getattr(self, 'dock_global', None)]:
                if cb_holder and hasattr(cb_holder, 'combo_crop_presets'):
                    cb = cb_holder.combo_crop_presets
                    if cb:
                        cb.blockSignals(True)
                        c_idx = cb.findData(preset_id)
                        if c_idx >= 0:
                            cb.setCurrentIndex(c_idx)
                        cb.blockSignals(False)

            # Determine aspect ratio and recalculate mathematically exact boundaries
            asp_ratio = target_preset.get('aspect_ratio')
            asp_name = target_preset.get('aspect_name', 'Free')
            clean_name = asp_name
            p_name = target_preset.get('name', '')

            if asp_ratio is None or asp_ratio <= 0:
                asp_ratio, clean_name = parse_aspect_ratio_text(asp_name, self.engine.width, self.engine.height)
            if asp_ratio is None or asp_ratio <= 0:
                asp_ratio, clean_name = parse_aspect_ratio_text(f"{preset_id} {p_name}", self.engine.width, self.engine.height)

            cl = int(target_preset.get('crop_left', 0))
            cr = int(target_preset.get('crop_right', 0))
            ct = int(target_preset.get('crop_top', 0))
            cb = int(target_preset.get('crop_bottom', 0))

            # Only enforce aspect ratio on completely uninitialized/empty presets
            if (cl == 0 and cr == 0 and ct == 0 and cb == 0) and asp_ratio is not None and asp_ratio > 0 and self.engine.width > 0 and self.engine.height > 0 and preset_id != 'full_frame':
                cl, cr, ct, cb = enforce_crop_aspect_ratio(
                    cl, cr, ct, cb, self.engine.width, self.engine.height, asp_ratio, allow_orientation_flip=False
                )
                target_preset['crop_left'] = cl
                target_preset['crop_right'] = cr
                target_preset['crop_top'] = ct
                target_preset['crop_bottom'] = cb
            target_preset['aspect_name'] = clean_name

            # Apply crop coordinates to sliders
            self.slider_crop_l.setValue(cl)
            self.slider_crop_r.setValue(cr)
            self.slider_crop_t.setValue(ct)
            self.slider_crop_b.setValue(cb)
            self.slider_rotation.setValue(int(target_preset.get('straighten', 0)))

            self.image_label.set_crop_aspect_ratio(asp_ratio, clean_name)

            # Match combobox by substring
            for wh, attr in [
                (getattr(self, 'viewport_toolbar', None), 'combo_crop_aspect'),
                (getattr(self, 'dock_global', None), 'combo_crop_aspect'),
                (getattr(self, 'dock_ai', None), 'combo_crop_aspect')
            ]:
                if wh and hasattr(wh, attr):
                    combo = getattr(wh, attr)
                    if combo:
                        combo.blockSignals(True)
                        target_idx = -1
                        for i in range(combo.count()):
                            it = combo.itemText(i)
                            if (clean_name in it) or (asp_name and asp_name in it) or ("1:1" in clean_name and "1:1" in it) or ("4:5" in clean_name and "4:5" in it) or ("9:16" in clean_name and "9:16" in it) or ("Original" in clean_name and "Original" in it):
                                target_idx = i
                                break
                        if target_idx >= 0:
                            combo.setCurrentIndex(target_idx)
                        combo.blockSignals(False)

            self._on_crop_changed(0)
            self._on_rotation_changed(self.slider_rotation.value())

            if hasattr(self, 'image_label') and self.image_label.is_cropping:
                self.image_label.update()

            self.statusBar().showMessage(f"✂️ Crop Preset: {target_preset.get('name', preset_id)} ({clean_name})")
        finally:
            self._is_updating_crop_preset = False

    def _on_add_crop_preset_clicked(self):
        from PySide6.QtWidgets import QInputDialog
        cur_asp = getattr(self.image_label, 'crop_aspect_name', 'Custom')
        default_name = f"Crop Preset ({cur_asp})"
        
        name, ok = QInputDialog.getText(self, "New Crop Preset", "Enter a name for this crop preset:", text=default_name)
        if not ok or not name.strip():
            return

        name = name.strip()
        import re
        slug = re.sub(r'[^a-zA-Z0-9_]+', '_', name.lower()).strip('_') or f"preset_{len(self.crop_presets)+1}"
        
        # Ensure unique ID
        existing_ids = {p.get('id') for p in self.crop_presets}
        unique_id = slug
        counter = 2
        while unique_id in existing_ids:
            unique_id = f"{slug}_{counter}"
            counter += 1

        new_preset = {
            'id': unique_id,
            'name': name,
            'aspect_name': getattr(self.image_label, 'crop_aspect_name', 'Free'),
            'crop_left': self.slider_crop_l.value(),
            'crop_right': self.slider_crop_r.value(),
            'crop_top': self.slider_crop_t.value(),
            'crop_bottom': self.slider_crop_b.value(),
            'straighten': self.slider_rotation.value()
        }

        self.crop_presets.append(new_preset)
        self.active_crop_preset_id = unique_id
        self._populate_crop_presets_ui()
        self._save_current_image_sidecar()
        self.statusBar().showMessage(f"✅ Added Crop Preset: '{name}'")

    def _on_rename_crop_preset_clicked(self):
        from PySide6.QtWidgets import QInputDialog
        target_preset = None
        for p in self.crop_presets:
            if p.get('id') == self.active_crop_preset_id:
                target_preset = p
                break

        if not target_preset:
            return

        old_name = target_preset.get('name', 'Preset')
        new_name, ok = QInputDialog.getText(self, "Rename Crop Preset", "Enter new name:", text=old_name)
        if ok and new_name.strip():
            target_preset['name'] = new_name.strip()
            self._populate_crop_presets_ui()
            self._save_current_image_sidecar()
            self.statusBar().showMessage(f"✏️ Renamed Crop Preset to '{new_name.strip()}'")

    def _on_delete_crop_preset_clicked(self):
        if len(self.crop_presets) <= 1:
            QMessageBox.information(self, "Cannot Delete", "At least one crop preset must remain.")
            return

        res = QMessageBox.question(self, "Delete Crop Preset", f"Delete selected crop preset?", QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            self.crop_presets = [p for p in self.crop_presets if p.get('id') != self.active_crop_preset_id]
            if self.crop_presets:
                self.active_crop_preset_id = self.crop_presets[0]['id']
            self._populate_crop_presets_ui()
            self._on_crop_preset_changed(0)
            self._save_current_image_sidecar()


    # =========================================================================
    # Dedicated AI Multi-Crop Presets Generation
    # =========================================================================
    def trigger_ai_crop_presets(self):
        """
        Dedicated AI Composition & Cropping Phase:
        Resets crop boundaries to full frame (preserving straighten), captures the graded
        image at 1024px, and prompts Gemini to generate 3-4 fine-art crop presets.
        """
        if not self.loaded_image_path or not self.engine.texture:
            QMessageBox.information(self, "No Image", "Please open an image first.")
            return

        api_key = self.dock_ai.edit_api_key.text().strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            QMessageBox.warning(self, "API Key Missing", "Please enter a valid Google Gemini API Key in the AI Assistant dock.")
            return

        self.statusBar().showMessage("📐 Generating AI Multi-Crop Presets (Hero 4:5, Square 1:1, Story 4:3, Full Frame)...")
        self.dock_ai.txt_ai_output.append("\n📐 [AI Framing] Capturing uncropped graded photo to generate Multi-Crop Presets...")

        # 1. Reset crop boundaries to 0 so AI sees the entire graded scene (preserve straighten)
        self._is_updating_crop_preset = True
        self.crop_presets = []
        self.active_crop_preset_id = "default"
        self.slider_crop_l.setValue(0)
        self.slider_crop_r.setValue(0)
        self.slider_crop_t.setValue(0)
        self.slider_crop_b.setValue(0)
        self.engine.set_crop(0.0, 0.0, 0.0, 0.0)
        self._is_updating_crop_preset = False
        self._populate_crop_presets_ui()
        self._update_view()

        # 2. Capture uncropped viewport photo downscaled to 1024px
        img_uncropped, _ = self.engine.get_viewport_cropped_image(
            viewport_label=self.image_label,
            fully_accumulate_bokeh=True,
            save_debug_copy=False,
            max_size=1024,
            force_uncropped=True
        )

        if hasattr(self, 'image_label'):
            self.image_label.show_ai_processing(pass_text="AI Crops", status_text="Calculating optimal composition & framing...")
            self.loading_anim_timer.start()

        from gui.gemini_worker import GeminiCropWorker
        orig_w = self.engine.width if self.engine.width > 0 else img_uncropped.width
        orig_h = self.engine.height if self.engine.height > 0 else img_uncropped.height

        self.crop_worker = GeminiCropWorker(
            image=img_uncropped,
            api_key=api_key,
            thinking_level=self._ai_config.get("thinking_level", "medium") if hasattr(self, '_ai_config') else "medium",
            orig_size=(orig_w, orig_h),
            parent=self
        )
        self.crop_worker.thought_received.connect(self._on_crop_thought)
        self.crop_worker.status_updated.connect(lambda s: self.statusBar().showMessage(f"📐 AI Crops: {s}"))
        self.crop_worker.finished_presets.connect(self._on_ai_crop_presets_success)
        self.crop_worker.error_occurred.connect(self._on_ai_crop_presets_error)
        self.crop_worker.start()

    def _on_crop_thought(self, text: str):
        if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'txt_ai_output'):
            self.dock_ai.txt_ai_output.moveCursor(self.dock_ai.txt_ai_output.textCursor().MoveOperation.End)
            self.dock_ai.txt_ai_output.insertPlainText(text)
        if hasattr(self, 'image_label') and getattr(self.image_label, 'is_ai_running', False):
            self.image_label.update_ai_status(pass_text="AI Crops", status_text="AI Thinking (Spatial composition & QC)...")

    def _on_ai_crop_presets_success(self, raw_presets: list, explanation: str):
        if hasattr(self, 'image_label'):
            self.image_label.hide_ai_processing()

        self.dock_ai.txt_ai_output.append(f"\n💡 AI Composition & Visual QC Summary:\n{explanation}\n")

        w_val = self.engine.width if self.engine.width > 0 else 8192
        h_val = self.engine.height if self.engine.height > 0 else 5464

        parsed_presets = []
        if isinstance(raw_presets, list) and len(raw_presets) > 0:
            for p_idx, p in enumerate(raw_presets):
                if not isinstance(p, dict): continue
                p_id = str(p.get('id', f"preset_{p_idx+1}")).strip()
                p_name = str(p.get('name', f"Preset {p_idx+1}")).strip()
                p_asp = str(p.get('aspect_name', p.get('aspect', 'Free'))).strip()

                p_cl = int(round(float(p.get('crop_left', p.get('left', 0)))))
                p_cr = int(round(float(p.get('crop_right', p.get('right', 0)))))
                p_ct = int(round(float(p.get('crop_top', p.get('top', 0)))))
                p_cb = int(round(float(p.get('crop_bottom', p.get('bottom', 0)))))

                # Normalize if in 0..1 scale
                if any(0.0 < v <= 1.0 for v in (p_cl, p_cr, p_ct, p_cb)):
                    p_cl = int(round(p_cl * 1000.0))
                    p_cr = int(round(p_cr * 1000.0))
                    p_ct = int(round(p_ct * 1000.0))
                    p_cb = int(round(p_cb * 1000.0))

                p_cl = int(np.clip(p_cl, 0, 480))
                p_cr = int(np.clip(p_cr, 0, 480))
                p_ct = int(np.clip(p_ct, 0, 480))
                p_cb = int(np.clip(p_cb, 0, 480))

                asp_ratio = p.get('aspect_ratio')
                clean_asp = p_asp
                if asp_ratio is None or asp_ratio <= 0:
                    asp_ratio, clean_asp = parse_aspect_ratio_text(p_asp, w_val, h_val)
                if asp_ratio is None or asp_ratio <= 0:
                    asp_ratio, clean_asp = parse_aspect_ratio_text(f"{p_id} {p_name}", w_val, h_val)

                if asp_ratio is not None and asp_ratio > 0 and p_id != 'full_frame':
                    p_cl, p_cr, p_ct, p_cb = enforce_crop_aspect_ratio(
                        p_cl, p_cr, p_ct, p_cb, w_val, h_val, asp_ratio, allow_orientation_flip=False
                    )

                parsed_presets.append({
                    'id': p_id,
                    'name': p_name,
                    'aspect_name': clean_asp,
                    'aspect_ratio': asp_ratio,
                    'crop_left': p_cl,
                    'crop_right': p_cr,
                    'crop_top': p_ct,
                    'crop_bottom': p_cb,
                    'straighten': self.slider_rotation.value()
                })

        if parsed_presets:
            self.crop_presets = parsed_presets
            self.active_crop_preset_id = parsed_presets[0]['id']
            self._populate_crop_presets_ui()
            self._on_crop_preset_changed(0)
            self._save_current_image_sidecar()
            self.statusBar().showMessage(f"✅ Generated {len(parsed_presets)} Crop Presets! Switched to '{parsed_presets[0]['name']}'")
        else:
            self.statusBar().showMessage("⚠️ No crop presets returned by AI.")

        # Finalize snapshot, auto-export all presets, and advance batch queue smoothly
        self._finalize_post_tuning_actions()

    def _on_ai_crop_presets_error(self, err_msg: str):
        if hasattr(self, 'image_label'):
            self.image_label.hide_ai_processing()
        self.dock_ai.txt_ai_output.append(f"⚠️ AI Crop generation error: {err_msg}\n")
        self.statusBar().showMessage(f"⚠️ AI Crops: {err_msg}")

        is_fatal_key_error = any(k in (err_msg or "").lower() for k in [
            "api key not valid", "api_key_invalid", "invalid api key",
            "permission_denied", "permissiondenied", "403", "unregistered caller"
        ])
        if is_fatal_key_error and getattr(self, '_batch_crop_running', False):
            self._batch_crop_running = False
            self._set_batch_ui_active(False)
            self.statusBar().showMessage("❌ Invalid Gemini API Key")
            self.dock_ai.txt_ai_output.append("❌ BATCH CROP STOPPED: Invalid or expired Gemini API Key.\n")
            QMessageBox.critical(self, "Invalid Gemini API Key", f"Gemini API Key is invalid or expired:\n\n{err_msg}\n\nPlease enter a valid API key in the AI Assistant panel.")
            return

        self._finalize_post_tuning_actions()

    def trigger_batch_ai_crop_presets(self, custom_photos_list: Optional[list] = None):
        """
        Batch processes photos using Gemini AI Auto-Cropping only.
        Clears all existing crop presets for each photo and generates new fine-art presets,
        strictly preserving all color grading, exposure, tone curves, and layer adjustments.
        If custom_photos_list is provided or 2+ photos are selected on Filmstrip,
        processes only those selected photos; otherwise processes all visible non-rejected photos.
        """
        if not hasattr(self, 'dock_filmstrip') or not hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            QMessageBox.information(self, "No Filmstrip", "Filmstrip widget is not available.")
            return

        fw = self.dock_filmstrip.filmstrip_widget
        if custom_photos_list and isinstance(custom_photos_list, list) and len(custom_photos_list) > 0:
            photos = custom_photos_list
            is_selection_batch = True
        else:
            selected_photos = fw.get_selected_photos(include_rejected=False)
            if len(selected_photos) >= 2:
                photos = selected_photos
                is_selection_batch = True
            else:
                photos = fw.get_visible_non_rejected_photos()
                is_selection_batch = False

        if not photos:
            QMessageBox.information(
                self, "No Photos Found",
                "No photos currently selected or visible in Filmstrip matching your active filter (excluding rejected photos).\n\n"
                "Please make sure photos are loaded in the Filmstrip and not marked for deletion."
            )
            return

        cfg = self.dock_ai.get_gemini_config()
        api_key = cfg.get("api_key")
        if not api_key:
            QMessageBox.warning(
                self, "Gemini API Key Required",
                "Google Gemini API Key is missing.\n\nPlease enter your API Key in the AI panel on the left dock or add GEMINI_API_KEY to your .env file."
            )
            return

        auto_export = self.dock_ai.is_batch_auto_export_enabled() if hasattr(self, 'dock_ai') else False
        custom_folder = self.dock_ai.get_batch_export_folder() if hasattr(self, 'dock_ai') else ""
        export_desc = f"Enabled ({custom_folder if custom_folder else 'Subfolder: LDExport'}) [JPEG 85%]" if auto_export else "Disabled"
        scope_label = f"{len(photos)} selected photo(s)" if is_selection_batch else f"{len(photos)} photo(s) currently displayed in the Filmstrip"

        reply = QMessageBox.question(
            self,
            "Run Batch AI Cropping & Framing",
            f"Run AI Multi-Crop Generation on {scope_label}?\n\n"
            f"• Action: Clears all existing crop presets and generates new fine-art crops (Hero 4:5/16:9, Square 1:1, Close-Up, Classic 3:4/4:3, Full Frame).\n"
            f"• Color & Grading: 100% PRESERVED (no changes to color, exposure, tone curve, or layers).\n"
            f"• Auto-Export on Complete: {export_desc}\n"
            f"• Marked for Deletion (❌): Ignored\n\n"
            "Do you want to start batch cropping?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply != QMessageBox.Yes:
            return

        # Automatically disable Fast Viewer when starting Batch AI
        if getattr(self, 'fast_viewer_mode', False):
            self._on_viewer_mode_toggled(False)

        self._batch_crop_running = True
        self._batch_crop_queue = photos
        self._batch_crop_total = len(photos)
        self._batch_crop_current_idx = 0
        self._set_batch_ui_active(True, status_text=f"Batch AI Crop (0/{self._batch_crop_total})...")

        self.dock_ai.txt_ai_output.append(f"\n========================================================")
        self.dock_ai.txt_ai_output.append(f"📐 BATCH AI MULTI-CROP PRESETS STARTED: {self._batch_crop_total} photo(s)")
        self.dock_ai.txt_ai_output.append(f"• Scope: {scope_label}")
        self.dock_ai.txt_ai_output.append(f"• Action: Reset existing presets and generate new fine-art crop variations")
        self.dock_ai.txt_ai_output.append(f"• Color & Grading: Preserved")
        self.dock_ai.txt_ai_output.append(f"• Auto-Export: {export_desc}")
        self.dock_ai.txt_ai_output.append(f"• Tip: Press Escape or click '⏹ Stop' at any time to cancel.")
        self.dock_ai.txt_ai_output.append(f"========================================================\n")

        self._run_next_batch_crop_photo()

    def _run_next_batch_crop_photo(self):
        if not getattr(self, '_batch_crop_running', False):
            return

        if self._batch_crop_current_idx >= len(self._batch_crop_queue):
            self._finish_batch_crop()
            return

        photo = self._batch_crop_queue[self._batch_crop_current_idx]
        target_path = photo['path']

        # Highlight in filmstrip
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            self.dock_filmstrip.filmstrip_widget.set_active_image(target_path)

        # Update batch UI status
        self._set_batch_ui_active(True, status_text=f"AI Crop ({self._batch_crop_current_idx + 1}/{self._batch_crop_total})...")

        # Check if the target photo is ALREADY 100% loaded as true RAW sensor data
        is_same_file = (self.loaded_image_path and os.path.abspath(self.loaded_image_path) == os.path.abspath(target_path))
        is_loader_busy = (getattr(self, 'image_loader_thread', None) is not None and self.image_loader_thread.isRunning())
        is_passthrough = getattr(self.engine, 'is_passthrough', False)

        if is_same_file and not is_loader_busy and not is_passthrough and self.engine.texture is not None:
            self._batch_waiting_for_crop_image = False
            self._proceed_batch_photo_crop()
        else:
            self._batch_waiting_for_crop_image = True
            self.open_image_file(target_path)

    def _proceed_batch_photo_crop(self):
        if not getattr(self, '_batch_crop_running', False):
            return
        if self._batch_crop_current_idx >= len(self._batch_crop_queue):
            self._finish_batch_crop()
            return

        photo = self._batch_crop_queue[self._batch_crop_current_idx]
        pass_tag = f"Photo {self._batch_crop_current_idx + 1}/{self._batch_crop_total}"
        photo_name = os.path.basename(photo['path'])

        self.dock_ai.txt_ai_output.append(f"\n--- [{pass_tag}] {photo_name} (AI Cropping) ---")

        # Reset & delete all existing crop presets for this photo
        self._is_updating_crop_preset = True
        self.crop_presets = []
        self.active_crop_preset_id = "default"
        self.slider_crop_l.setValue(0)
        self.slider_crop_r.setValue(0)
        self.slider_crop_t.setValue(0)
        self.slider_crop_b.setValue(0)
        self.engine.set_crop(0.0, 0.0, 0.0, 0.0)
        self._is_updating_crop_preset = False
        self._populate_crop_presets_ui()
        self._update_view()

        # Run dedicated AI crop presets generation (it captures uncropped graded view and queries Gemini)
        self.trigger_ai_crop_presets()

    def _finish_batch_crop(self):
        self._batch_crop_running = False
        self._set_batch_ui_active(False)
        self.statusBar().showMessage(f"🎉 Batch AI Cropping finished for all {self._batch_crop_total} photo(s)!")
        self.dock_ai.txt_ai_output.append("\n========================================================")
        self.dock_ai.txt_ai_output.append(f"🎉 BATCH AI CROPPING COMPLETED: {self._batch_crop_total} photo(s) re-cropped.")
        self.dock_ai.txt_ai_output.append("========================================================\n")

    def _synchronize_all_color_equalizers(self):
        """
        Guarantees that Point 0 (0° Red) and Point 6 (360° Red) in Color Equalizer
        and Shadow Color Equalizer are strictly synchronized across all engine layers,
        UI docks, and GPU engine textures.
        """
        if not hasattr(self, 'engine') or not self.engine:
            return

        # 1. Sanitize equalizers across all engine layers
        for layer in getattr(self.engine, 'layers', []):
            if not hasattr(layer, 'settings') or not isinstance(layer.settings, dict):
                continue
            for eq_key in ('color_equalizer', 'shadow_color_equalizer'):
                if eq_key in layer.settings and isinstance(layer.settings[eq_key], dict):
                    layer.settings[eq_key] = sanitize_equalizer_dict(layer.settings[eq_key])

        # 2. Synchronize active layer controls in dock_effects if available
        if hasattr(self, 'dock_effects') and self.dock_effects:
            active_idx = getattr(self.engine, 'active_layer_index', 0)
            if self.engine.layers and 0 <= active_idx < len(self.engine.layers):
                active_settings = self.engine.layers[active_idx].settings
                self.dock_effects.set_layer_settings(active_settings)

        # 3. Synchronize direct widget references if present
        if hasattr(self, 'color_equalizer') and self.color_equalizer:
            self.color_equalizer.set_nodes_from_ints(
                self.color_equalizer.get_hue_nodes_as_ints(),
                self.color_equalizer.get_saturation_nodes_as_ints(),
                self.color_equalizer.get_brightness_nodes_as_ints()
            )
        if hasattr(self, 'shadow_color_equalizer') and self.shadow_color_equalizer:
            self.shadow_color_equalizer.set_nodes_from_ints(
                self.shadow_color_equalizer.get_hue_nodes_as_ints(),
                self.shadow_color_equalizer.get_saturation_nodes_as_ints(),
                self.shadow_color_equalizer.get_brightness_nodes_as_ints()
            )

    def serialize_state(self) -> Dict[str, Any]:
        self._synchronize_all_color_equalizers()
        # First synchronize current active layer settings from dock_effects into self.engine.layers
        effects_dict = self.dock_effects.get_layer_settings()
        if self.engine.layers and 0 <= self.engine.active_layer_index < len(self.engine.layers):
            self.engine.layers[self.engine.active_layer_index].settings = dict(effects_dict)

        state = {
            'is_denoised': bool(getattr(self.engine, 'has_ai_denoised', False)),
            'has_ai_denoised': bool(getattr(self.engine, 'has_ai_denoised', False)),
            'bokeh': {
                'enabled': self.check_bokeh_enable.isChecked(),
                'focal_depth': self.slider_bokeh_focal_depth.value(),
                'focal_range': self.slider_bokeh_focal_range.value(),
                'fg_gamma': self.slider_bokeh_fg_gamma.value(),
                'bg_gamma': self.slider_bokeh_bg_gamma.value(),
                'subject_influence': self.slider_bokeh_subject_influence.value(),
                'radius': self.slider_bokeh_radius.value(),
                'petals': self.slider_bokeh_petals.value(),
                'rotation': self.slider_bokeh_rotation.value(),
                'boost': self.slider_bokeh_boost.value(),
                'chromatic': self.slider_bokeh_chromatic.value(),
                'vignette': self.slider_bokeh_vignette.value(),
                'radial_center': self.slider_bokeh_radial_center.value(),
                'radial_edge': self.slider_bokeh_radial_edge.value(),
                'radial_gamma': self.slider_bokeh_radial_gamma.value(),
                'samples': self.combo_bokeh_samples.currentText()
            },
            'global_output': {
                'base_curve': self.slider_base_curve.value() if getattr(self, 'slider_base_curve', None) else 30,
                'ai_blend': self.slider_ai_blend.value() if getattr(self, 'slider_ai_blend', None) else 100
            },
            'geometry_crop': {
                'straighten': self.slider_rotation.value(),
                'flip_90': self.flip_90,
                'crop_left': self.slider_crop_l.value(),
                'crop_right': self.slider_crop_r.value(),
                'crop_top': self.slider_crop_t.value(),
                'crop_bottom': self.slider_crop_b.value(),
                'aspect_name': getattr(self.image_label, 'crop_aspect_name', 'Free'),
                'aspect_ratio': getattr(self.image_label, 'crop_aspect_ratio', None),
                'image_width': self.engine.width,
                'image_height': self.engine.height
            },
            'crop_presets': copy.deepcopy(getattr(self, 'crop_presets', [])),
            'active_crop_preset': getattr(self, 'active_crop_preset_id', 'default'),
            'lens_correction': {
                'enabled': self.chk_lens_enable.isChecked(),
                'cam_maker': self.combo_cam_maker.currentText(),
                'cam_model': self.combo_cam_model.currentText(),
                'lens_profile': self.combo_lens_profile.currentText()
            },
            'layers': [
                {
                    'id': l.id, 'name': l.name, 'active': l.active, 'is_base': l.is_base,
                    'settings': l.settings, 'mask_opacity': int(l.mask_opacity * 100),
                    'mask_fill': int(l.mask_fill * 100),
                    'mask_luma_target': int(l.mask_luma_target * 100),
                    'mask_luma_range': int(l.mask_luma_range * 100),
                    'mask_hue_target': int(l.mask_hue_target * 360),
                    'mask_hue_range': int(l.mask_hue_range * 180),
                    'mask_sat_target': int(l.mask_sat_target * 100),
                    'mask_sat_range': int(l.mask_sat_range * 100),
                    'mask_softness': int(l.mask_softness * 100),
                    'mask_gradient_type': int(l.mask_gradient_type),
                    'mask_grad_black': int(l.mask_grad_black * 100),
                    'mask_grad_white': int(l.mask_grad_white * 100),
                    'mask_grad_gamma': int(l.mask_grad_gamma * 100),
                    'use_depth_mask': l.use_depth_mask,
                    'mask_depth_min': int(l.mask_depth_min * 100),
                    'mask_depth_max': int(l.mask_depth_max * 100),
                    'mask_depth_softness': int(l.mask_depth_softness * 100),
                    'mask_depth_gamma': int(l.mask_depth_gamma * 100),
                    'use_normal_mask': l.use_normal_mask,
                    'mask_normal_x': int(l.mask_normal_dir[0] * 100),
                    'mask_normal_y': int(l.mask_normal_dir[1] * 100),
                    'mask_normal_angle': int(l.mask_normal_angle * 180),
                    'mask_normal_softness': int(l.mask_normal_softness * 100),
                    'mask_normal_gamma': int(l.mask_normal_gamma * 100)
                }
                for l in self.engine.layers
            ]
        }
        # Include Base Layer effect settings at top-level for legacy and sidecar compatibility
        if self.engine.layers:
            state.update(self.engine.layers[0].settings or effects_dict)
        else:
            state.update(effects_dict)
        if hasattr(self, 'dock_global') and hasattr(self.dock_global, 'txt_photo_notes'):
            state['ai_notes'] = self.dock_global.txt_photo_notes.toPlainText()
        return sanitize_state_equalizers(state)

    def _apply_layers_data_in_place(self, layers_data: list):
        BASIC_KEYS = [
            'exposure', 'contrast', 'gamma', 'base_curve', 'temperature', 'tint',
            'highlights', 'shadows', 'whites', 'black_level', 'highlight_recovery',
            'local_contrast', 'global_chroma', 'global_vibrance',
            'denoise_luma', 'denoise_chroma', 'sharpen', 'sharpen_threshold', 'vignette'
        ]
        for i, ld in enumerate(layers_data):
            if i >= len(self.engine.layers):
                break
            layer = self.engine.layers[i]
            layer.name = ld.get('name', layer.name)
            layer.active = ld.get('active', layer.active)
            layer.is_base = ld.get('is_base', layer.is_base)

            # Extract settings
            settings = dict(ld.get('settings', {}))
            for key in ['basic_adjustments', 'tone_equalizer', 'color_equalizer', 'shadow_color_equalizer']:
                if key in ld and isinstance(ld[key], dict):
                    settings[key] = copy.deepcopy(ld[key])
            if 'basic_adjustments' not in settings or not isinstance(settings['basic_adjustments'], dict):
                settings['basic_adjustments'] = {}
            if 'basic_adjustments' in ld and isinstance(ld['basic_adjustments'], dict):
                settings['basic_adjustments'].update(ld['basic_adjustments'])

            for b_key in BASIC_KEYS:
                val = None
                if b_key in ld:
                    val = ld[b_key]
                elif 'settings' in ld and isinstance(ld['settings'], dict) and b_key in ld['settings']:
                    val = ld['settings'][b_key]
                elif 'basic_adjustments' in ld and isinstance(ld['basic_adjustments'], dict) and b_key in ld['basic_adjustments']:
                    val = ld['basic_adjustments'][b_key]
                elif 'settings' in ld and isinstance(ld['settings'], dict) and 'basic_adjustments' in ld['settings'] and isinstance(ld['settings']['basic_adjustments'], dict) and b_key in ld['settings']['basic_adjustments']:
                    val = ld['settings']['basic_adjustments'][b_key]

                if val is not None:
                    settings[b_key] = val
                    settings['basic_adjustments'][b_key] = val

            if 'tone_equalizer' in ld:
                settings['tone_equalizer'] = ld['tone_equalizer']
            if 'color_equalizer' in ld:
                settings['color_equalizer'] = sanitize_equalizer_dict(ld['color_equalizer']) if isinstance(ld['color_equalizer'], dict) else ld['color_equalizer']
            if 'shadow_color_equalizer' in ld:
                settings['shadow_color_equalizer'] = sanitize_equalizer_dict(ld['shadow_color_equalizer']) if isinstance(ld['shadow_color_equalizer'], dict) else ld['shadow_color_equalizer']

            layer.settings = settings

            # Masks
            if 'mask_opacity' in ld: layer.mask_opacity = ld['mask_opacity'] / 100.0
            if 'mask_fill' in ld: layer.mask_fill = ld['mask_fill'] / 100.0
            if 'use_depth_mask' in ld: layer.use_depth_mask = bool(ld['use_depth_mask'])
            if 'mask_depth_min' in ld: layer.mask_depth_min = ld['mask_depth_min'] / 100.0
            if 'mask_depth_max' in ld: layer.mask_depth_max = ld['mask_depth_max'] / 100.0
            if 'mask_depth_softness' in ld: layer.mask_depth_softness = ld['mask_depth_softness'] / 100.0
            if 'mask_depth_gamma' in ld: layer.mask_depth_gamma = ld['mask_depth_gamma'] / 100.0
            if 'use_normal_mask' in ld: layer.use_normal_mask = bool(ld['use_normal_mask'])
            if 'mask_normal_angle' in ld: layer.mask_normal_angle = ld['mask_normal_angle'] / 180.0
            if 'mask_normal_softness' in ld: layer.mask_normal_softness = ld['mask_normal_softness'] / 100.0
            if 'mask_normal_gamma' in ld: layer.mask_normal_gamma = ld['mask_normal_gamma'] / 100.0

    def deserialize_state(self, state: Dict[str, Any]):
        if not state:
            return

        state = sanitize_state_equalizers(state)
        self.engine.is_passthrough = False

        if not self.engine.layers:
            self.engine.layers = [Layer(id="layer_base", name="Base Layer", is_base=True, active=True)]
            self.engine.active_layer_index = 0

        current_active_idx = self.engine.active_layer_index

        skip_bg = getattr(self, '_batch_skip_bg', False) is True

        if 'layers' in state and isinstance(state['layers'], list) and len(state['layers']) > 0:
            layers_data = state['layers']
            if skip_bg:
                layers_data = [ld for ld in layers_data if isinstance(ld, dict) and (ld.get('name', '').lower() != 'background' or ld.get('is_base', False))]
            existing_layers = self.engine.layers
            can_update_in_place = (
                len(existing_layers) == len(layers_data) and
                self.dock_layers.layers_list.count() == len(existing_layers) and
                all(l.id == ld.get('id', l.id) for l, ld in zip(existing_layers, layers_data))
            )
            if can_update_in_place:
                self._apply_layers_data_in_place(layers_data)
            else:
                self._restore_layers(layers_data, self.loaded_image_path)
                self.engine.active_layer_index = min(current_active_idx, max(0, len(self.engine.layers) - 1))
        else:
            self._restore_layers([])
            self.dock_effects.set_layer_settings(state)
            if self.engine.layers:
                # If no 'layers' key in state, apply global basic adjustments to Base Layer (layer 0)
                self.engine.layers[0].settings = self.dock_effects.get_layer_settings()

        # Guarantee UI list has all items populated and check states/names are 100% in sync
        self.dock_layers.refresh_layers_list(self.engine.layers, self.engine.active_layer_index)

        # Re-apply active layer settings to ALL UI docks (Effects, Opacity, and all Mask controls)
        if self.engine.layers and 0 <= self.engine.active_layer_index < len(self.engine.layers):
            self._on_layer_selection_changed(self.engine.active_layer_index)

        self._synchronize_all_color_equalizers()

        # Optical / Lens Correction Handling in deserialize_state
        batch_lens_forced = hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_batch_lens_correction_enabled') and self.dock_ai.is_batch_lens_correction_enabled()
        chk_widget = getattr(self, 'chk_lens_enable', getattr(self.dock_global, 'chk_lens_enable', None)) if hasattr(self, 'dock_global') else None
        
        # ALWAYS force enable lens correction if batch setting is ON or if user turned it on
        en = True if batch_lens_forced else False
        if not en and 'lens_correction' in state and isinstance(state['lens_correction'], dict):
            en = bool(state['lens_correction'].get('enabled', False))
        elif not en and chk_widget and chk_widget.isChecked():
            en = True

        if 'lens_correction' in state and isinstance(state['lens_correction'], dict):
            lc = state['lens_correction']
            cam_maker = lc.get('cam_maker')
            cam_model = lc.get('cam_model')
            lens_prof = lc.get('lens_profile')
            
            if cam_maker and hasattr(self, 'combo_cam_maker') and self.combo_cam_maker.count() > 0:
                idx = self.combo_cam_maker.findText(cam_maker)
                if idx != -1 and self.combo_cam_maker.currentIndex() != idx:
                    self.combo_cam_maker.blockSignals(True)
                    self.combo_cam_maker.setCurrentIndex(idx)
                    self.combo_cam_maker.blockSignals(False)
                    
            if cam_model and hasattr(self, 'combo_cam_model') and self.combo_cam_model.count() > 0:
                idx = self.combo_cam_model.findText(cam_model)
                if idx != -1 and self.combo_cam_model.currentIndex() != idx:
                    self.combo_cam_model.blockSignals(True)
                    self.combo_cam_model.setCurrentIndex(idx)
                    self.combo_cam_model.blockSignals(False)
                    
            if lens_prof and hasattr(self, 'combo_lens_profile') and self.combo_lens_profile.count() > 0:
                idx = self.combo_lens_profile.findText(lens_prof)
                if idx != -1 and self.combo_lens_profile.currentIndex() != idx:
                    self.combo_lens_profile.blockSignals(True)
                    self.combo_lens_profile.setCurrentIndex(idx)
                    self.combo_lens_profile.blockSignals(False)

        if chk_widget:
            chk_widget.blockSignals(True)
            chk_widget.setChecked(en)
            chk_widget.blockSignals(False)
            
        self.engine.enable_lens_correction = en
        if en:
            if not self.engine.has_lens_profile:
                self.engine.update_lens_uv_map(getattr(self, 'metadata', None))
            self.engine.apply_lens_correction()

        if 'bokeh' in state:
            b = state['bokeh']
            bokeh_en = bool(b.get('enabled', False))
            # If user checked 'Forbid AI from enabling 3D Bokeh', force bokeh disabled
            if hasattr(self, 'dock_ai') and hasattr(self.dock_ai, 'is_ai_disable_bokeh_enabled') and self.dock_ai.is_ai_disable_bokeh_enabled():
                bokeh_en = False
            self.check_bokeh_enable.setChecked(bokeh_en)

            # Check if Gemini provided a focus_point [x, y] percentage on the photo
            if 'focus_point' in b and isinstance(b['focus_point'], (list, tuple)) and len(b['focus_point']) >= 2:
                fx = float(b['focus_point'][0]) / 100.0
                # Invert y because Gemini sends y from top (0..100) whereas pick_depth_at expects v=1.0 at top
                fy = 1.0 - (float(b['focus_point'][1]) / 100.0)
                cached_d = self._get_cached_depth_map()
                z_lin = self.engine.pick_depth_at(fx, fy, cached_d, sample_radius=5)
                focal_depth = int(round((1.0 - np.clip(z_lin, 0.0, 1.0)) * 100.0))
                self.slider_bokeh_focal_depth.setValue(focal_depth)
            else:
                self.slider_bokeh_focal_depth.setValue(b.get('focal_depth', 50))

            self.slider_bokeh_focal_range.setValue(int(b.get('focal_range', 12)))
            self.slider_bokeh_fg_gamma.setValue(b.get('fg_gamma', 100))
            self.slider_bokeh_bg_gamma.setValue(b.get('bg_gamma', 100))
            self.slider_bokeh_subject_influence.setValue(b.get('subject_influence', 0))
            radius_val = b.get('radius', b.get('max_radius', b.get('aperture', b.get('blur_radius', b.get('blur', 40)))))
            self.slider_bokeh_radius.setValue(int(radius_val))
            self.slider_bokeh_petals.setValue(b.get('petals', 4))
            self.slider_bokeh_rotation.setValue(b.get('rotation', 0))
            self.slider_bokeh_boost.setValue(b.get('boost', 50))
            self.slider_bokeh_chromatic.setValue(b.get('chromatic', 0))
            self.slider_bokeh_vignette.setValue(b.get('vignette', 0))
            self.slider_bokeh_radial_center.setValue(b.get('radial_center', 0))
            self.slider_bokeh_radial_edge.setValue(b.get('radial_edge', 0))
            self.slider_bokeh_radial_gamma.setValue(b.get('radial_gamma', 100))
            samples_str = str(b.get('samples', '256'))
            if samples_str in ('32', '64'):
                samples_str = '256'
            self.combo_bokeh_samples.setCurrentText(samples_str)
            self._on_bokeh_parameter_changed()
        if 'global_output' in state:
            go = state['global_output']
            if 'base_curve' in go and getattr(self, 'slider_base_curve', None): self.slider_base_curve.setValue(int(go['base_curve']))
            if 'ai_blend' in go and getattr(self, 'slider_ai_blend', None): self.slider_ai_blend.setValue(int(go['ai_blend']))
            self._on_global_parameter_changed()
        if 'base_curve' in state and getattr(self, 'slider_base_curve', None):
            self.slider_base_curve.setValue(int(state['base_curve']))
            self._on_global_parameter_changed()
        if 'ai_blend' in state and getattr(self, 'slider_ai_blend', None):
            self.slider_ai_blend.setValue(int(state['ai_blend']))
            self._on_global_parameter_changed()
        if 'geometry_crop' in state:
            g = state['geometry_crop']
            self.slider_rotation.setValue(int(g.get('straighten', 0)))
            self.flip_90 = int(g.get('flip_90', 0))
            self.engine.flip_90 = self.flip_90
            self.slider_crop_l.setValue(int(g.get('crop_left', 0)))
            self.slider_crop_r.setValue(int(g.get('crop_right', 0)))
            self.slider_crop_t.setValue(int(g.get('crop_top', 0)))
            self.slider_crop_b.setValue(int(g.get('crop_bottom', 0)))
            self._on_rotation_changed(self.slider_rotation.value())
            self._on_crop_changed(0)
            if 'aspect_name' in g:
                aspect_name = g['aspect_name']
                aspect_ratio = g.get('aspect_ratio', None)
                self.image_label.set_crop_aspect_ratio(aspect_ratio, aspect_name)
                for widget_holder, attr_name in [
                    (getattr(self, 'viewport_toolbar', None), 'combo_crop_aspect'),
                    (getattr(self, 'dock_global', None), 'combo_crop_aspect'),
                    (getattr(self, 'dock_ai', None), 'combo_crop_aspect'),
                    (getattr(getattr(self, 'dock_filmstrip', None), 'filmstrip_widget', None), 'combo_crop_aspect')
                ]:
                    if widget_holder and hasattr(widget_holder, attr_name):
                        combo = getattr(widget_holder, attr_name)
                        if combo:
                            combo.blockSignals(True)
                            idx = combo.findText(aspect_name)
                            if idx >= 0: combo.setCurrentIndex(idx)
                            combo.blockSignals(False)
        else:
            if 'straighten' in state:
                self.slider_rotation.setValue(int(state['straighten']))
                self._on_rotation_changed(self.slider_rotation.value())
            if 'crop' in state and isinstance(state['crop'], dict):
                c = state['crop']
                def _gemini_crop(v):
                    iv = int(v or 0)
                    return iv * 10 if 0 < iv <= 45 else iv
                if 'crop_left' in c or 'left' in c: self.slider_crop_l.setValue(_gemini_crop(c.get('crop_left', c.get('left', 0))))
                if 'crop_right' in c or 'right' in c: self.slider_crop_r.setValue(_gemini_crop(c.get('crop_right', c.get('right', 0))))
                if 'crop_top' in c or 'top' in c: self.slider_crop_t.setValue(_gemini_crop(c.get('crop_top', c.get('top', 0))))
                if 'crop_bottom' in c or 'bottom' in c: self.slider_crop_b.setValue(_gemini_crop(c.get('crop_bottom', c.get('bottom', 0))))
                self._on_crop_changed(0)
            for ck, slider in [('crop_left', self.slider_crop_l), ('crop_right', self.slider_crop_r), ('crop_top', self.slider_crop_t), ('crop_bottom', self.slider_crop_b)]:
                if ck in state:
                    slider.setValue(int(state[ck]))
                    self._on_crop_changed(0)
        if 'ai_notes' in state and hasattr(self, 'dock_global') and hasattr(self.dock_global, 'txt_photo_notes'):
            self.dock_global.txt_photo_notes.blockSignals(True)
            self.dock_global.txt_photo_notes.setPlainText(str(state['ai_notes'] or ""))
            self.dock_global.txt_photo_notes.blockSignals(False)

        # Restore Multi-Crop Presets
        if 'crop_presets' in state and isinstance(state['crop_presets'], list) and len(state['crop_presets']) > 0:
            self.crop_presets = copy.deepcopy(state['crop_presets'])
            self.active_crop_preset_id = state.get('active_crop_preset', self.crop_presets[0]['id'])
        else:
            # Create default preset from current geometry crop
            g = state.get('geometry_crop', {})
            cl = g.get('crop_left', self.slider_crop_l.value())
            cr = g.get('crop_right', self.slider_crop_r.value())
            ct = g.get('crop_top', self.slider_crop_t.value())
            cb = g.get('crop_bottom', self.slider_crop_b.value())
            asp = g.get('aspect_name', getattr(self.image_label, 'crop_aspect_name', 'Free'))
            self.crop_presets = [{
                'id': 'default',
                'name': f"Crop Preset ({asp})",
                'aspect_name': asp,
                'crop_left': cl,
                'crop_right': cr,
                'crop_top': ct,
                'crop_bottom': cb,
                'straighten': g.get('straighten', self.slider_rotation.value())
            }]
            self.active_crop_preset_id = 'default'

        self._populate_crop_presets_ui()

    def _on_photo_notes_changed(self, text: str):
        """Called when user edits the notes in GlobalDock."""
        if not hasattr(self, 'loaded_image_path') or not self.loaded_image_path:
            return
        self.dock_filmstrip.filmstrip_widget.set_photo_notes(self.loaded_image_path, text)
        self._save_current_image_sidecar()

    def _on_filmstrip_notes_updated(self, path: str, notes: str):
        """Called when notes are updated via Filmstrip context menu."""
        if hasattr(self, 'loaded_image_path') and self.loaded_image_path and os.path.abspath(self.loaded_image_path) == os.path.abspath(path):
            if hasattr(self, 'dock_global'):
                self.dock_global.txt_photo_notes.blockSignals(True)
                self.dock_global.txt_photo_notes.setPlainText(notes)
                self.dock_global.txt_photo_notes.blockSignals(False)
            self._save_current_image_sidecar()

    def push_undo_state(self):
        if not getattr(self, '_restoring_sidecar', False) and not getattr(self, 'is_viewer_mode', False):
            self._is_dirty = True
        self.undo_stack.append(self.serialize_state())
        if len(self.undo_stack) > 50: self.undo_stack.pop(0)

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.deserialize_state(self.undo_stack[-1])
            self._update_view()

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.deserialize_state(state)
            self._update_view()

    def reset_settings(self):
        self.dock_effects.reset_all_effects()
        self.slider_rotation.setValue(0)
        self.flip_90 = 0
        self.engine.flip_90 = 0
        self.slider_crop_l.setValue(0)
        self.slider_crop_r.setValue(0)
        self.slider_crop_t.setValue(0)
        self.slider_crop_b.setValue(0)
        self.engine.set_crop(0.0, 0.0, 0.0, 0.0)
        self.crop_presets = []
        self.active_crop_preset_id = 'default'
        self._populate_crop_presets_ui()
        self.check_bokeh_enable.setChecked(False)
        if hasattr(self, 'combo_bokeh_samples'):
            self.combo_bokeh_samples.blockSignals(True)
            self.combo_bokeh_samples.setCurrentText("256")
            self.combo_bokeh_samples.blockSignals(False)
        self.engine.bokeh_max_samples = 256
        self._restore_layers([])
        self.engine.active_layer_index = 0
        self.dock_layers.refresh_layers_list(self.engine.layers, 0)
        self._on_layer_selection_changed(0)
        self._update_view()

    def _save_preview_snapshot(self, path: str):
        """Saves a high-quality rendered viewport screenshot with black margins trimmed for 0ms future preview."""
        try:
            if not self.engine.texture or getattr(self.engine, 'is_passthrough', False):
                return
            folder = os.path.dirname(path)
            stem, _ = os.path.splitext(os.path.basename(path))
            lumen_dir = os.path.join(folder, "LumenDeep")
            os.makedirs(lumen_dir, exist_ok=True)
            preview_path = os.path.join(lumen_dir, f"{stem}_preview.jpg")

            img, _ = self.engine.get_viewport_cropped_image(viewport_label=self.image_label, fully_accumulate_bokeh=False, max_size=2560)
            if img:
                img.save(preview_path, "JPEG", quality=92)
        except Exception as e:
            print("Error saving preview snapshot:", e)

    def _is_photo_modified(self) -> bool:
        """Checks if current editor state has actually diverged from initial loaded state."""
        if not self.loaded_image_path or not hasattr(self, '_initial_saved_state') or self._initial_saved_state is None:
            return False
        if getattr(self.engine, 'is_passthrough', False) or getattr(self, 'is_viewer_mode', False):
            return False
        if not getattr(self, '_is_dirty', False):
            return False
        try:
            curr_state = self.serialize_state()
            return json.dumps(curr_state, sort_keys=True) != json.dumps(self._initial_saved_state, sort_keys=True)
        except Exception:
            return getattr(self, '_is_dirty', False)

    def _check_and_prompt_save(self) -> bool:
        """
        Prompts user to save modified photo settings if changes were made.
        Returns True if okay to proceed (Saved, Discarded, or Unmodified), False if Cancelled.
        Batch AI processing automatically saves without prompting.
        """
        if not self.loaded_image_path:
            return True

        if getattr(self, '_batch_ai_running', False):
            if self._is_photo_modified():
                self._save_current_image_sidecar()
            return True

        if not self._is_photo_modified():
            return True

        photo_name = os.path.basename(self.loaded_image_path)
        reply = QMessageBox.question(
            self,
            "Save Changes?",
            f"Save adjustments to '{photo_name}' before leaving?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save
        )
        if reply == QMessageBox.Save:
            self._save_current_image_sidecar()
            self._is_dirty = False
            self._initial_saved_state = copy.deepcopy(self.serialize_state())
            return True
        elif reply == QMessageBox.Discard:
            self._is_dirty = False
            return True
        else: # Cancel
            return False

    def _save_current_image_sidecar(self):
        if not self.loaded_image_path: return
        if getattr(self.engine, 'is_passthrough', False) or getattr(self, 'is_viewer_mode', False):
            return
        folder = os.path.dirname(self.loaded_image_path)
        stem, _ = os.path.splitext(os.path.basename(self.loaded_image_path))
        lumen_dir = os.path.join(folder, "LumenDeep")
        os.makedirs(lumen_dir, exist_ok=True)
        sidecar_path = os.path.join(lumen_dir, f"{stem}.lumendeep")
        try:
            curr_state = self.serialize_state()
            with open(sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(curr_state, f, indent=2)
            self._is_dirty = False
            self._initial_saved_state = copy.deepcopy(curr_state)
        except Exception as e:
            print("Failed to save sidecar:", e)
        self._save_preview_snapshot(self.loaded_image_path)
        self._refresh_filmstrip_active_thumbnail()

    def _restore_sidecar_or_defaults(self, path: str):
        self._restoring_sidecar = True
        try:
            folder = os.path.dirname(path)
            stem, _ = os.path.splitext(os.path.basename(path))
            lumen_dir = os.path.join(folder, "LumenDeep")
            
            candidates = [
                os.path.join(lumen_dir, f"{stem}.lumendeep"),
                os.path.join(lumen_dir, f"{stem}.json"),
                os.path.splitext(path)[0] + ".lumendeep",
                os.path.splitext(path)[0] + ".json",
            ]
            loaded = False
            for c in candidates:
                if os.path.exists(c):
                    try:
                        with open(c, 'r', encoding='utf-8') as f:
                            state = json.load(f)
                        state = sanitize_state_equalizers(state)
                        self.deserialize_state(state)
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load sidecar {c}:", e)
            if not loaded:
                self.reset_settings()
        finally:
            self._restoring_sidecar = False
            self._initial_saved_state = copy.deepcopy(self.serialize_state())
            self._is_dirty = False

    def _restore_layers(self, layers_data, image_path=None):
        if not layers_data:
            self.engine.layers = [Layer(id="layer_base", name="Base Layer", is_base=True, active=True)]
        else:
            BASIC_KEYS = [
                'exposure', 'contrast', 'gamma', 'base_curve', 'temperature', 'tint',
                'highlights', 'shadows', 'whites', 'black_level', 'highlight_recovery',
                'local_contrast', 'global_chroma', 'global_vibrance',
                'denoise_luma', 'denoise_chroma', 'sharpen', 'sharpen_threshold', 'vignette'
            ]
            restored_layers = []
            for i, ld in enumerate(layers_data):
                settings = dict(ld.get('settings', {}))
                for key in ['basic_adjustments', 'tone_equalizer', 'color_equalizer', 'shadow_color_equalizer']:
                    if key in ld and isinstance(ld[key], dict):
                        settings[key] = copy.deepcopy(ld[key])
                if 'basic_adjustments' not in settings or not isinstance(settings['basic_adjustments'], dict):
                    settings['basic_adjustments'] = {}
                if 'basic_adjustments' in ld and isinstance(ld['basic_adjustments'], dict):
                    settings['basic_adjustments'].update(ld['basic_adjustments'])

                for b_key in BASIC_KEYS:
                    val = None
                    if b_key in ld:
                        val = ld[b_key]
                    elif 'settings' in ld and isinstance(ld['settings'], dict) and b_key in ld['settings']:
                        val = ld['settings'][b_key]
                    elif 'basic_adjustments' in ld and isinstance(ld['basic_adjustments'], dict) and b_key in ld['basic_adjustments']:
                        val = ld['basic_adjustments'][b_key]
                    elif 'settings' in ld and isinstance(ld['settings'], dict) and 'basic_adjustments' in ld['settings'] and isinstance(ld['settings']['basic_adjustments'], dict) and b_key in ld['settings']['basic_adjustments']:
                        val = ld['settings']['basic_adjustments'][b_key]
                    if val is not None:
                        settings[b_key] = val
                        settings['basic_adjustments'][b_key] = val

                # Direct equalizers in ld
                if 'tone_equalizer' in ld:
                    settings['tone_equalizer'] = ld['tone_equalizer']
                if 'color_equalizer' in ld:
                    settings['color_equalizer'] = sanitize_equalizer_dict(ld['color_equalizer']) if isinstance(ld['color_equalizer'], dict) else ld['color_equalizer']
                if 'shadow_color_equalizer' in ld:
                    settings['shadow_color_equalizer'] = sanitize_equalizer_dict(ld['shadow_color_equalizer']) if isinstance(ld['shadow_color_equalizer'], dict) else ld['shadow_color_equalizer']

                # Parametric mask
                pm = ld.get('parametric_mask', {})
                luma_t = pm.get('luma_target', ld.get('mask_luma_target', 50)) / 100.0
                luma_r = pm.get('luma_range', ld.get('mask_luma_range', 0)) / 100.0
                hue_t = pm.get('hue_target', ld.get('mask_hue_target', 0)) / 360.0
                hue_r = pm.get('hue_range', ld.get('mask_hue_range', 0)) / 180.0
                sat_t = pm.get('sat_target', ld.get('mask_sat_target', 50)) / 100.0
                sat_r = pm.get('sat_range', ld.get('mask_sat_range', 0)) / 100.0
                soft = pm.get('softness', ld.get('mask_softness', 50)) / 100.0

                # Gradient mask
                gm = ld.get('gradient_mask', {})
                gt_val = gm.get('type', ld.get('mask_gradient_type', 0))
                if isinstance(gt_val, str):
                    gt_map = {"none": 0, "left to right": 1, "right to left": 2, "top to bottom": 3, "bottom to top": 4, "radial": 5}
                    gt_val = gt_map.get(gt_val.lower(), 0)
                g_type = float(gt_val)
                g_black = gm.get('black', ld.get('mask_grad_black', 0)) / 100.0
                g_white = gm.get('white', ld.get('mask_grad_white', 100)) / 100.0
                g_gamma = gm.get('gamma', ld.get('mask_grad_gamma', 100)) / 100.0

                # 3D Depth mask
                use_depth = ld.get('use_depth_mask', False)
                d_min = ld.get('mask_depth_min', 0) / 100.0
                d_max = ld.get('mask_depth_max', 110) / 100.0
                d_soft = ld.get('mask_depth_softness', 10) / 100.0
                d_gamma = ld.get('mask_depth_gamma', 100) / 100.0

                # 3D Normal mask
                use_norm = ld.get('use_normal_mask', False)
                norm_x = ld.get('mask_normal_x', 0) / 100.0
                norm_y = ld.get('mask_normal_y', 0) / 100.0
                norm_angle = ld.get('mask_normal_angle', 45) / 180.0
                norm_soft = ld.get('mask_normal_softness', 10) / 100.0
                norm_gamma = ld.get('mask_normal_gamma', 100) / 100.0

                layer = Layer(
                    id=ld.get('id', f"layer_{i+1}"),
                    name=ld.get('name', f"Layer {i+1}"),
                    is_base=ld.get('is_base', i == 0),
                    active=ld.get('active', True),
                    settings=settings,
                    mask_opacity=ld.get('mask_opacity', 100) / 100.0,
                    mask_fill=ld.get('mask_fill', 0) / 100.0,
                    mask_luma_target=luma_t,
                    mask_luma_range=luma_r,
                    mask_hue_target=hue_t,
                    mask_hue_range=hue_r,
                    mask_sat_target=sat_t,
                    mask_sat_range=sat_r,
                    mask_softness=soft,
                    mask_gradient_type=g_type,
                    mask_grad_black=g_black,
                    mask_grad_white=g_white,
                    mask_grad_gamma=g_gamma,
                    use_depth_mask=use_depth,
                    mask_depth_min=d_min,
                    mask_depth_max=d_max,
                    mask_depth_softness=d_soft,
                    mask_depth_gamma=d_gamma,
                    use_normal_mask=use_norm,
                    mask_normal_dir=(norm_x, norm_y, 1.0),
                    mask_normal_angle=norm_angle,
                    mask_normal_softness=norm_soft,
                    mask_normal_gamma=norm_gamma
                )
                restored_layers.append(layer)
            self.engine.layers = restored_layers
        self.engine.active_layer_index = 0
        self.dock_layers.refresh_layers_list(self.engine.layers, 0)
        self._on_layer_selection_changed(0)

    def save_project_sidecar(self):
        if not self.loaded_image_path:
            QMessageBox.information(self, "No Image Loaded", "Please open an image first before saving project settings.")
            return
        self._save_current_image_sidecar()
        stem = os.path.basename(self.loaded_image_path)
        self.statusBar().showMessage(f"💾 Project settings saved to sidecar for {stem}")

    def export_image(self):
        if not mcp_server.state.image_loaded or not self.loaded_image_path:
            QMessageBox.information(self, "No Image Loaded", "Please open an image first before exporting.")
            return

        from gui.dialogs.export_dialog import ExportDialog, ExportProgressDialog, ExportWorkerThread
        dlg = ExportDialog(
            self,
            source_image_path=self.loaded_image_path,
            image_width=self.engine.width,
            image_height=self.engine.height
        )
        if dlg.exec() == ExportDialog.Accepted and dlg.export_settings:
            self._synchronize_all_color_equalizers()
            settings = dlg.export_settings
            file_path = settings['file_path']
            bit_depth = settings['bit_depth']
            quality = settings['quality']
            scale = settings.get('scale', 1.0)
            save_sidecar = settings.get('save_sidecar', True)

            mp = (self.engine.width * self.engine.height) / 1_000_000.0
            dim_str = f"{self.engine.width} × {self.engine.height} px ({mp:.1f} MP)"

            progress_dlg = ExportProgressDialog(self, destination_path=file_path, dimensions=dim_str)
            progress_dlg.show()
            progress_dlg.set_progress(5, "Rendering full-resolution GPU frame...")
            QApplication.processEvents()

            try:
                from core.image_io import linear_to_srgb_u8
                dual_bokeh = settings.get('dual_bokeh', False)
                bokeh_active = bool(self.check_bokeh_enable.isChecked() or getattr(self.engine, 'bokeh_enabled', False))
                export_all_crops = settings.get('export_all_crops', False)
                crop_presets_to_export = getattr(self, 'crop_presets', [])

                base_dir = os.path.dirname(file_path)
                file_name = os.path.basename(file_path)
                stem, ext = os.path.splitext(file_name)
                clean_stem = stem
                if clean_stem.endswith("_bokeh"): clean_stem = clean_stem[:-6]
                if clean_stem.endswith("_edited"): clean_stem = clean_stem[:-7]

                if dual_bokeh and bokeh_active:
                    clean_file_path = os.path.join(base_dir, f"{clean_stem}{ext}")
                    bokeh_file_path = os.path.join(base_dir, f"{clean_stem}_bokeh{ext}")

                    # 1. Render Bokeh Pass
                    progress_dlg.set_progress(15, "Rendering full-resolution 3D Bokeh frame...")
                    QApplication.processEvents()
                    self.engine.bokeh_enabled = True
                    self.engine.render(to_preview=False, progress_callback=progress_dlg)
                    
                    if progress_dlg.is_cancelled():
                        progress_dlg.close()
                        self.statusBar().showMessage("⏹ Export cancelled by user.")
                        return

                    progress_dlg.set_progress(45, "Encoding Bokeh frame to disk...")
                    QApplication.processEvents()
                    pixels_bokeh = self.engine.read_pixels(to_preview=False)
                    export_image_from_linear(bokeh_file_path, pixels_bokeh, apply_gamma=True, bit_depth=bit_depth, quality=quality)

                    # 2. Render Clean Sharp Pass (Without Bokeh)
                    progress_dlg.set_progress(60, "Rendering clean sharp frame...")
                    QApplication.processEvents()
                    self.engine.bokeh_enabled = False
                    self.engine.render(to_preview=False)
                    pixels_clean = self.engine.read_pixels(to_preview=False)

                    # 3. Restore live view
                    self.engine.bokeh_enabled = True
                    self.engine.render(to_preview=True)

                    target_main_path = clean_file_path
                    main_pixels = pixels_clean
                else:
                    # If exporting all multi-crops, render uncropped full frame first so presets slice accurately
                    saved_crop_l = self.engine.crop_l
                    saved_crop_r = self.engine.crop_r
                    saved_crop_t = self.engine.crop_t
                    saved_crop_b = self.engine.crop_b

                    if export_all_crops and len(crop_presets_to_export) > 1:
                        self.engine.set_crop(0.0, 0.0, 0.0, 0.0)

                    progress_dlg.set_progress(20, "Rendering full-resolution frame on GPU...")
                    QApplication.processEvents()
                    self.engine.render(to_preview=False, progress_callback=progress_dlg)
                    
                    if progress_dlg.is_cancelled():
                        self.engine.set_crop(saved_crop_l, saved_crop_r, saved_crop_t, saved_crop_b)
                        progress_dlg.close()
                        self.statusBar().showMessage("⏹ Export cancelled by user.")
                        return

                    progress_dlg.set_progress(65, "Downloading pixels from GPU...")
                    QApplication.processEvents()
                    main_pixels = self.engine.read_pixels(to_preview=False)

                    if export_all_crops and len(crop_presets_to_export) > 1:
                        # Restore crop parameters for live view
                        self.engine.set_crop(saved_crop_l, saved_crop_r, saved_crop_t, saved_crop_b)
                        self.engine.render(to_preview=True)

                    target_main_path = os.path.join(base_dir, f"{clean_stem}{ext}")

                if progress_dlg.is_cancelled():
                    progress_dlg.close()
                    self.statusBar().showMessage("⏹ Export cancelled by user.")
                    return

                # Check if multi-crop presets export is requested
                if export_all_crops and len(crop_presets_to_export) > 1:
                    progress_dlg.set_progress(75, f"Encoding {len(crop_presets_to_export)} Multi-Crop Presets...")
                    QApplication.processEvents()
                    
                    # Convert full linear float frame to 8-bit sRGB uint8 ONCE for instant slicing
                    main_u8 = linear_to_srgb_u8(main_pixels)
                    H, W = main_u8.shape[:2]

                    for p_idx, preset in enumerate(crop_presets_to_export):
                        if progress_dlg.is_cancelled():
                            progress_dlg.close()
                            return
                            
                        p_id = preset.get('id', f"crop_{p_idx+1}")
                        p_asp = preset.get('aspect_name', '')
                        target_ratio = preset.get('aspect_ratio')
                        if target_ratio is None or target_ratio <= 0:
                            target_ratio, _ = parse_aspect_ratio_text(p_asp, W, H)
                        if target_ratio is None or target_ratio <= 0:
                            target_ratio, _ = parse_aspect_ratio_text(f"{p_id} {preset.get('name', '')}", W, H)

                        cl = int(round(float(preset.get('crop_left', 0))))
                        cr = int(round(float(preset.get('crop_right', 0))))
                        ct = int(round(float(preset.get('crop_top', 0))))
                        cb = int(round(float(preset.get('crop_bottom', 0))))

                        if target_ratio and target_ratio > 0 and p_id not in ('full_frame', 'full', 'uncropped'):
                            cl, cr, ct, cb = enforce_crop_aspect_ratio(
                                cl, cr, ct, cb, W, H, target_ratio, allow_orientation_flip=False
                            )

                        cl_norm = np.clip(float(cl) / 1000.0, 0.0, 0.49)
                        cr_norm = np.clip(float(cr) / 1000.0, 0.0, 0.49)
                        ct_norm = np.clip(float(ct) / 1000.0, 0.0, 0.49)
                        cb_norm = np.clip(float(cb) / 1000.0, 0.0, 0.49)

                        x0 = int(round(cl_norm * W))
                        x1 = max(x0 + 1, int(round((1.0 - cr_norm) * W)))
                        y0 = int(round(ct_norm * H))
                        y1 = max(y0 + 1, int(round((1.0 - cb_norm) * H)))

                        cropped_sub = main_u8[y0:y1, x0:x1]
                        
                        # Clean naming without _edited
                        if p_id in ('full_frame', 'full', 'uncropped'):
                            out_preset_path = os.path.join(base_dir, f"{clean_stem}{ext}")
                        else:
                            clean_pid = p_id.replace('preset_', '').replace('crop_', '')
                            out_preset_path = os.path.join(base_dir, f"{clean_stem}_{clean_pid}{ext}")

                        export_image_from_linear(out_preset_path, cropped_sub, apply_gamma=False, bit_depth=8, quality=quality)
                        
                        pct = int(75 + (p_idx + 1) / len(crop_presets_to_export) * 20)
                        progress_dlg.set_progress(pct, f"Saved '{preset.get('name', p_id)}'...")
                        QApplication.processEvents()

                # Asynchronous Encoder Thread for main file saving
                self._export_worker = ExportWorkerThread(
                    file_path=target_main_path,
                    pixels=main_pixels,
                    bit_depth=bit_depth,
                    quality=quality,
                    scale=scale,
                    parent=self
                )

                def _on_export_progress(pct, status_text):
                    if not progress_dlg.is_cancelled():
                        progress_dlg.set_progress(pct, status_text)

                def _on_export_finished(out_path, out_w, out_h):
                    progress_dlg.close()
                    if save_sidecar:
                        self._save_current_image_sidecar()
                    if dual_bokeh and bokeh_active:
                        self.statusBar().showMessage(f"✅ Dual-Exported {os.path.basename(bokeh_file_path)} & {os.path.basename(clean_file_path)}")
                        QMessageBox.information(
                            self, "Dual Export Complete",
                            f"Successfully exported 2 versions:\n• With Bokeh: {bokeh_file_path}\n• Clean Sharp: {clean_file_path}\n\nBit Depth: {bit_depth}-bit\nDimensions: {out_w} × {out_h} px"
                        )
                    else:
                        self.statusBar().showMessage(f"✅ Exported to {os.path.basename(out_path)} ({bit_depth}-bit, {out_w}×{out_h} px)")
                        QMessageBox.information(
                            self, "Export Complete",
                            f"Successfully exported:\n{out_path}\n\nBit Depth: {bit_depth}-bit\nDimensions: {out_w} × {out_h} px"
                        )

                def _on_export_error(err_str):
                    progress_dlg.close()
                    self.statusBar().showMessage("❌ Export Error")
                    QMessageBox.critical(self, "Export Failed", f"Failed to export image:\n\n{err_str}")

                self._export_worker.progress.connect(_on_export_progress)
                self._export_worker.finished.connect(_on_export_finished)
                self._export_worker.error.connect(_on_export_error)
                self._export_worker.start()

            except Exception as e:
                progress_dlg.close()
                QMessageBox.critical(self, "Export Failed", f"Failed to render full-resolution image:\n\n{e}")

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()
        
        if modifiers == Qt.ControlModifier:
            if key == Qt.Key_E:
                self.export_image()
                return
            elif key == Qt.Key_S:
                self.save_project_sidecar()
                return
            elif key == Qt.Key_O:
                self._on_open_file_dialog()
                return
            elif key == Qt.Key_Z:
                self.undo()
                return
            elif key == Qt.Key_Y:
                self.redo()
                return

        # Check Culling shortcuts (1..5 rating, 0 clear, X/Del reject)
        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_0, Qt.Key_U, Qt.Key_X, Qt.Key_Delete]:
            if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
                if self.dock_filmstrip.filmstrip_widget.handle_culling_key(key):
                    return

        if key == Qt.Key_F: self.fit_to_viewport()
        elif key == Qt.Key_Z: self.zoom_to_100()
        elif key == Qt.Key_C: self.viewport_toolbar.btn_tb_crop_tool.toggle()
        elif key in [Qt.Key_Return, Qt.Key_Enter]:
            if self.image_label.is_cropping: self.viewport_toolbar.btn_tb_crop_tool.setChecked(False)
        elif key == Qt.Key_Escape:
            if getattr(self, '_batch_ai_running', False) or getattr(self, '_full_auto_workflow_running', False) or getattr(self, '_ai_is_running', False):
                self.stop_ai_auto_correct()
                return
            if self.image_label.is_cropping: self.viewport_toolbar.btn_tb_crop_tool.setChecked(False)
        elif key == Qt.Key_V: self.viewport_toolbar.btn_tb_split_v.toggle()
        elif key == Qt.Key_H: self.viewport_toolbar.btn_tb_split_h.toggle()
        elif key == Qt.Key_B: self.viewport_toolbar.btn_tb_side_by_side.toggle()
        elif key == Qt.Key_W: self.viewport_toolbar.btn_tb_wb_pick.toggle()
        elif key == Qt.Key_F: self.viewport_toolbar.btn_tb_focus_pick.toggle()
        elif key == Qt.Key_P: self.viewport_toolbar.btn_tb_brush.toggle()
        elif key == Qt.Key_D: self.viewport_toolbar.btn_tb_depth.toggle()
        elif key == Qt.Key_N: self.viewport_toolbar.btn_tb_normals.toggle()
        elif key == Qt.Key_S: self.viewport_toolbar.btn_tb_subject.toggle()
        elif key == Qt.Key_M: self.viewport_toolbar.btn_tb_mask_prev.toggle()
        else: super().keyPressEvent(event)

    def show_ai_culling_dialog(self):
        """Opens Gemini AI Batch Photo Culling & 3D Depth Assessment Dialog."""
        if not hasattr(self, 'dock_filmstrip') or not hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            return

        fw = self.dock_filmstrip.filmstrip_widget
        folder = fw.current_folder
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(self, "No Folder Selected", "Please select a photo folder in the Folders panel first.")
            return

        # Gather photo paths from filmstrip
        paths = []
        for i in range(fw.list_widget.count()):
            item = fw.list_widget.item(i)
            if item and item.data(Qt.UserRole):
                paths.append(item.data(Qt.UserRole))

        if not paths:
            QMessageBox.information(self, "No Photos Found", "No photos found in the current folder.")
            return

        cfg = self.dock_ai.get_gemini_config()
        api_key = cfg.get("api_key")

        from gui.dialogs.ai_culling_dialog import AICullingDialog
        dlg = AICullingDialog(folder_path=folder, all_photo_paths=paths, api_key=api_key, parent=self)
        dlg.exec()

    def show_all_panels(self):
        """Restores visibility for all dock widgets."""
        for dock in [self.dock_folders, self.dock_global, self.dock_ai,
                     self.dock_filmstrip, self.dock_layers, self.dock_effects, self.dock_mask]:
            if dock:
                dock.show()
                dock.raise_()
        self.statusBar().showMessage("👁️ All panels restored.")

    def reset_panels_layout(self):
        """Resets all dock panels to their standard default positions."""
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_folders)
        self.tabifyDockWidget(self.dock_folders, self.dock_global)
        self.tabifyDockWidget(self.dock_global, self.dock_ai)
        self.dock_folders.raise_()

        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_filmstrip)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_layers)
        self.tabifyDockWidget(self.dock_layers, self.dock_effects)
        self.tabifyDockWidget(self.dock_effects, self.dock_mask)
        self.dock_effects.raise_()

        self.show_all_panels()
        self.statusBar().showMessage("🔄 Dock layout reset to default.")

    def open_documentation(self):
        """Opens the comprehensive User Guide."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        docs_path = os.path.join(base_dir, "docs", "USER_GUIDE.md")
        if os.path.exists(docs_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(docs_path))
        else:
            QDesktopServices.openUrl(QUrl("https://lumensdeep.com/docs.html"))

    def open_shortcuts_reference(self):
        """Opens the Keyboard Shortcuts cheat sheet."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sc_path = os.path.join(base_dir, "docs", "SHORTCUTS.md")
        if os.path.exists(sc_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(sc_path))
        else:
            QDesktopServices.openUrl(QUrl("https://lumensdeep.com/shortcuts.html"))

    def show_about_dialog(self):
        """Displays the rich About Lumens Deep & Open-Source Licenses Dialog."""
        try:
            from gui.dialogs.about_dialog import AboutDialog
            dlg = AboutDialog(self)
            dlg.exec()
        except Exception as e:
            QMessageBox.information(
                self, "About Lumens Deep",
                f"Lumens Deep v1.0.0 Production Release\n"
                f"Developed by Volodymyr Makovetskyi\n\n"
                f"32-Bit Linear RAW Photo Studio & 3D Neural Relighting Suite\n"
                f"https://lumensdeep.com"
            )

    def closeEvent(self, event):
        if self.loaded_image_path:
            if not self._check_and_prompt_save():
                event.ignore()
                return
        if hasattr(self, 'image_loader_thread') and self.image_loader_thread and self.image_loader_thread.isRunning():
            self.image_loader_thread.cancel()
            self.image_loader_thread.quit()
            self.image_loader_thread.wait(1000)
        if hasattr(self, 'ai_worker_thread') and self.ai_worker_thread and self.ai_worker_thread.isRunning():
            self.ai_worker_thread.cancel()
            self.ai_worker_thread.quit()
            self.ai_worker_thread.wait(1000)
        if hasattr(self, 'dock_filmstrip') and hasattr(self.dock_filmstrip, 'filmstrip_widget'):
            fw = self.dock_filmstrip.filmstrip_widget
            if hasattr(fw, 'loader_thread') and fw.loader_thread and fw.loader_thread.isRunning():
                fw.loader_thread.stop()
                fw.loader_thread.quit()
                fw.loader_thread.wait(1000)
            if hasattr(fw, 'scan_thread') and fw.scan_thread and fw.scan_thread.isRunning():
                fw.scan_thread.cancel()
                fw.scan_thread.quit()
                fw.scan_thread.wait(1000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = PhotoEditorWindow()
    window.show()
    sys.exit(app.exec())
