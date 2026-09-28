import math
import time
from typing import Optional
import numpy as np
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPainterPath,
    QPolygonF, QCursor, QTransform
)
from PySide6.QtCore import Qt, QPointF, QRectF
from core.profiler import profiler

class ImagePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.main_window = parent
        
        # Navigation
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pos = None
        self.is_dragging_split = False
        
        # Brush Mask Painting
        self.is_painting_mask = False
        self.is_picking_mask = False
        self.is_picking_wb = False
        self.is_picking_focus = False
        self.paint_radius = 50.0
        self.paint_hardness = 0.5
        self.paint_opacity = 1.0
        self.is_erasing = False
        self.hover_pos = None
        self.last_brush_uv = None
        
        # Loading Card Overlay
        self.is_loading = False
        self.loading_filename = "photo"
        self.loading_status_text = "Opening..."
        self.loading_anim_tick = 0

        # Background RAW Streaming Shimmer Bar
        self.is_background_loading = False
        self.bg_loading_filename = ""
        self.bg_loading_status_text = "Decoding full RAW in background..."

        # AI Processing Streaming Shimmer Bar & Floating Pill Badge
        self.is_ai_running = False
        self.ai_status_text = "Analyzing photo & auto-tuning..."
        self.ai_pass_text = ""
        
        # Interactive Crop & Straighten
        self.is_cropping = False
        self.crop_drag_mode = None
        self.crop_drag_start_pos = None
        self.crop_init_values = (0.0, 0.0, 0.0, 0.0)
        self.crop_init_rot = 0.0
        self.crop_aspect_ratio: Optional[float] = None
        self.crop_aspect_name: str = "Free"
        self._cached_rotate_cursor = None

        # Real-time Performance HUD
        self.show_profiler_hud = False

        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(50, 50)

    def minimumSizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(50, 50)

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(400, 300)

    def show_loading(self, filename: str, status_text: str = "Decoding RAW sensor data..."):
        self.is_loading = True
        self.is_background_loading = False
        self.loading_filename = filename
        self.loading_status_text = status_text
        self.update()

    def update_loading_status(self, status_text: str):
        self.loading_status_text = status_text
        self.update()

    def hide_loading(self):
        self.is_loading = False
        self.update()

    def show_background_loading(self, filename: str, status_text: str = "Decoding full RAW in background..."):
        self.is_background_loading = True
        self.is_loading = False
        self.bg_loading_filename = filename
        self.bg_loading_status_text = status_text
        self.update()

    def update_background_loading_status(self, status_text: str):
        self.bg_loading_status_text = status_text
        self.update()

    def hide_background_loading(self):
        self.is_background_loading = False
        self.update()

    def show_ai_processing(self, pass_text: str = "", status_text: str = "Analyzing photo & auto-tuning..."):
        self.is_ai_running = True
        self.ai_pass_text = pass_text
        self.ai_status_text = status_text
        self.update()

    def update_ai_status(self, pass_text: str = "", status_text: str = ""):
        if pass_text:
            self.ai_pass_text = pass_text
        if status_text:
            self.ai_status_text = status_text
        self.update()

    def hide_ai_processing(self):
        self.is_ai_running = False
        self.update()

    def set_crop_aspect_ratio(self, ratio: Optional[float], name: str = "Free"):
        self.crop_aspect_ratio = ratio
        self.crop_aspect_name = name
        self.update()

    def toggle_profiler_hud(self):
        self.show_profiler_hud = not self.show_profiler_hud
        self.update()
        return self.show_profiler_hud

    def set_show_profiler_hud(self, show: bool):
        self.show_profiler_hud = show
        self.update()

    def _get_crop_screen_rect(self):
        if not hasattr(self.main_window, 'engine') or not self.main_window.engine:
            return None, 0, 0, 0, 0
            
        eng = self.main_window.engine
        if not eng.texture or eng.width <= 0 or eng.height <= 0:
            return None, 0, 0, 0, 0

        angle_rad = np.radians(eng.rotation_deg - 90 * eng.flip_90)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        W_bound = eng.width * abs(c) + eng.height * abs(s)
        H_bound = eng.width * abs(s) + eng.height * abs(c)
        
        eff_prev_w = (eng.prev_w // 2) if getattr(eng, 'split_view_mode', None) == 'side_by_side' else eng.prev_w
        eff_prev_h = eng.prev_h
        
        if eff_prev_w <= 0 or eff_prev_h <= 0:
            eff_prev_w = max(1, self.width() if self.width() > 10 else 800)
            eff_prev_h = max(1, self.height() if self.height() > 10 else 600)
            
        if W_bound <= 0 or H_bound <= 0:
            return None, 0, 0, 0, 0
            
        viewport_aspect = eff_prev_w / float(eff_prev_h)
        image_aspect = W_bound / float(H_bound)
        
        if image_aspect > viewport_aspect:
            photo_w = eff_prev_w * self.zoom
            photo_h = (eff_prev_w / image_aspect) * self.zoom
        else:
            photo_h = eff_prev_h * self.zoom
            photo_w = (eff_prev_h * image_aspect) * self.zoom
            
        center_x = (self.width() / 2.0) + (self.pan_x * self.width() / 2.0)
        center_y = (self.height() / 2.0) + (self.pan_y * self.height() / 2.0)
        
        photo_x = center_x - (photo_w / 2.0)
        photo_y = center_y - (photo_h / 2.0)
        
        r = (eng.width / float(eng.height)) if eng.height > 0 else 1.0
        denom1 = r * abs(c) + abs(s)
        denom2 = abs(c) + r * abs(s)
        if denom1 > 1e-7 and denom2 > 1e-7:
            h_ins = min(eng.width / denom1, eng.height / denom2)
            w_ins = r * h_ins
        else:
            w_ins, h_ins = eng.width, eng.height
            
        auto_margin_x = max(0.0, min(0.499, (W_bound - w_ins) / (2.0 * W_bound))) if W_bound > 0 else 0.0
        auto_margin_y = max(0.0, min(0.499, (H_bound - h_ins) / (2.0 * H_bound))) if H_bound > 0 else 0.0

        crop_l = self.main_window.slider_crop_l.value() / 1000.0 if hasattr(self.main_window, 'slider_crop_l') else 0.0
        crop_r = self.main_window.slider_crop_r.value() / 1000.0 if hasattr(self.main_window, 'slider_crop_r') else 0.0
        crop_t = self.main_window.slider_crop_t.value() / 1000.0 if hasattr(self.main_window, 'slider_crop_t') else 0.0
        crop_b = self.main_window.slider_crop_b.value() / 1000.0 if hasattr(self.main_window, 'slider_crop_b') else 0.0
        
        eff_crop_l = auto_margin_x + crop_l * (1.0 - 2.0 * auto_margin_x)
        eff_crop_r = auto_margin_x + crop_r * (1.0 - 2.0 * auto_margin_x)
        eff_crop_t = auto_margin_y + crop_t * (1.0 - 2.0 * auto_margin_y)
        eff_crop_b = auto_margin_y + crop_b * (1.0 - 2.0 * auto_margin_y)

        left_px = photo_x + eff_crop_l * photo_w
        top_px = photo_y + eff_crop_t * photo_h
        right_px = photo_x + (1.0 - eff_crop_r) * photo_w
        bottom_px = photo_y + (1.0 - eff_crop_b) * photo_h
        
        rect = QRectF(left_px, top_px, max(1.0, right_px - left_px), max(1.0, bottom_px - top_px))
        return rect, photo_x, photo_y, photo_w, photo_h

    def _get_crop_hit_mode(self, pos):
        rect, pm_x, pm_y, pm_w, pm_h = self._get_crop_screen_rect()
        if not rect:
            return None
            
        x = pos.x()
        y = pos.y()
        l, r, t, b = rect.left(), rect.right(), rect.top(), rect.bottom()
        margin = 16.0
        
        # 1. Corner handles (16px box)
        if abs(x - l) <= margin and abs(y - t) <= margin: return 'TL'
        if abs(x - r) <= margin and abs(y - t) <= margin: return 'TR'
        if abs(x - l) <= margin and abs(y - b) <= margin: return 'BL'
        if abs(x - r) <= margin and abs(y - b) <= margin: return 'BR'
        
        # 2. Edge handles (8px border)
        if abs(x - l) <= 8.0 and t < y < b: return 'L'
        if abs(x - r) <= 8.0 and t < y < b: return 'R'
        if abs(y - t) <= 8.0 and l < x < r: return 'T'
        if abs(y - b) <= 8.0 and l < x < r: return 'B'
        
        # 3. Inside crop box: MOVE
        if l < x < r and t < y < b: return 'MOVE'
        
        # 4. Outside crop box: ROTATE
        return 'ROTATE'

    def _get_rotate_cursor(self):
        if self._cached_rotate_cursor is None:
            size = 32
            pix = QPixmap(size, size) if 'QPixmap' in globals() else None
            from PySide6.QtGui import QPixmap
            pix = QPixmap(size, size)
            pix.fill(Qt.transparent)
            
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            
            cx, cy, r = 16.0, 16.0, 10.0
            rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            
            path = QPainterPath()
            path.arcMoveTo(rect, 25)
            path.arcTo(rect, 25, 130)
            
            rad1 = math.radians(25)
            tip1 = QPointF(cx + r * math.cos(rad1), cy - r * math.sin(rad1))
            d1 = QPointF(math.sin(rad1), math.cos(rad1))
            n1 = QPointF(-d1.y(), d1.x())
            poly1 = QPolygonF([tip1, tip1 - 5.5 * d1 + 3.0 * n1, tip1 - 5.5 * d1 - 3.0 * n1])
            
            rad2 = math.radians(155)
            tip2 = QPointF(cx + r * math.cos(rad2), cy - r * math.sin(rad2))
            d2 = QPointF(-math.sin(rad2), -math.cos(rad2))
            n2 = QPointF(-d2.y(), d2.x())
            poly2 = QPolygonF([tip2, tip2 - 5.5 * d2 + 3.0 * n2, tip2 - 5.5 * d2 - 3.0 * n2])
            
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 255), 4.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            
            painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
            painter.setPen(QPen(QColor(0, 0, 0, 255), 1.0))
            painter.drawPolygon(poly1)
            painter.drawPolygon(poly2)
            
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 255), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            
            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0, 180), 0.5))
            painter.drawPolygon(poly1)
            painter.drawPolygon(poly2)
            
            painter.end()
            self._cached_rotate_cursor = QCursor(pix, 16, 16)
        return self._cached_rotate_cursor

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
            self.last_pos = event.position()
            
            if self.is_cropping and event.button() == Qt.MouseButton.LeftButton:
                mode = self._get_crop_hit_mode(event.position())
                if mode:
                    self.crop_drag_mode = mode
                    self.crop_drag_start_pos = event.position()
                    self.crop_init_values = (
                        self.main_window.slider_crop_l.value() / 1000.0,
                        self.main_window.slider_crop_r.value() / 1000.0,
                        self.main_window.slider_crop_t.value() / 1000.0,
                        self.main_window.slider_crop_b.value() / 1000.0
                    )
                    self.crop_init_rot = self.main_window.slider_rotation.value() / 10.0
                    self.update()
                    return

            split_mode = getattr(self.main_window, 'split_view_mode', None)
            if split_mode in ['vertical', 'horizontal']:
                pm = self.pixmap()
                if pm and not pm.isNull():
                    if split_mode == 'vertical':
                        pm_x = (self.width() - pm.width()) / 2.0
                        split_x = pm_x + getattr(self.main_window, 'split_position', 0.5) * pm.width()
                        if abs(event.position().x() - split_x) < 30:
                            self.is_dragging_split = True
                            return
                    elif split_mode == 'horizontal':
                        pm_y = (self.height() - pm.height()) / 2.0
                        split_y = pm_y + getattr(self.main_window, 'split_position', 0.5) * pm.height()
                        if abs(event.position().y() - split_y) < 30:
                            self.is_dragging_split = True
                            return

            if self.is_picking_mask or self.is_picking_wb or self.is_picking_focus:
                self._handle_picking(event)
            elif self.is_painting_mask:
                self.last_brush_uv = None
                self._handle_brush_draw(event, is_press=True)

    def _handle_picking(self, event):
        try:
            import mcp_server
            state = mcp_server.state
        except ImportError:
            from ... import mcp_server
            state = mcp_server.state
        if not state.image_loaded or not hasattr(self.main_window, 'engine'):
            return
        px, py = event.position().x(), event.position().y()
        pm = self.pixmap()
        if pm and not pm.isNull():
            pm_w, pm_h = pm.width(), pm.height()
            x_offset = (self.width() - pm_w) / 2.0
            y_offset = (self.height() - pm_h) / 2.0
            img_px = px - x_offset
            img_py = py - y_offset
            if 0 <= img_px <= pm_w and 0 <= img_py <= pm_h:
                u, v = self.main_window.engine.get_uv_from_viewport(img_px, img_py, pm_w, pm_h)
                r, g, b = self.main_window.engine.pick_color_at(u, v)
                if self.is_picking_mask:
                    self.main_window.apply_picked_mask_color(r, g, b)
                elif self.is_picking_wb:
                    self.main_window.apply_picked_wb_color(r, g, b)
                elif self.is_picking_focus:
                    cached_d = getattr(self.main_window, '_cached_depth_np', None)
                    z_lin = self.main_window.engine.pick_depth_at(u, v, cached_d)
                    self.main_window.apply_picked_focus_depth(z_lin, u, v)

    def mouseMoveEvent(self, event):
        self.hover_pos = event.position()
        self.update()
        
        if self.is_cropping:
            if self.crop_drag_mode and self.crop_drag_start_pos and (event.buttons() & Qt.MouseButton.LeftButton):
                rect, pm_x, pm_y, pm_w, pm_h = self._get_crop_screen_rect()
                if pm_w > 0 and pm_h > 0:
                    cur_x = event.position().x()
                    cur_y = event.position().y()
                    init_l, init_r, init_t, init_b = self.crop_init_values
                    mode = self.crop_drag_mode
                    ar = self.crop_aspect_ratio  # float or None

                    if ar is None:
                        # Free / Unconstrained Crop Dragging
                        if mode in ['L', 'TL', 'BL']:
                            new_l = (cur_x - pm_x) / float(pm_w)
                            new_l = max(0.0, min(0.48, min(1.0 - init_r - 0.05, new_l)))
                            self.main_window.slider_crop_l.setValue(int(new_l * 1000))
                            
                        if mode in ['R', 'TR', 'BR']:
                            new_r = (pm_x + pm_w - cur_x) / float(pm_w)
                            new_r = max(0.0, min(0.48, min(1.0 - init_l - 0.05, new_r)))
                            self.main_window.slider_crop_r.setValue(int(new_r * 1000))
                            
                        if mode in ['T', 'TL', 'TR']:
                            new_t = (cur_y - pm_y) / float(pm_h)
                            new_t = max(0.0, min(0.48, min(1.0 - init_b - 0.05, new_t)))
                            self.main_window.slider_crop_t.setValue(int(new_t * 1000))
                            
                        if mode in ['B', 'BL', 'BR']:
                            new_b = (pm_y + pm_h - cur_y) / float(pm_h)
                            new_b = max(0.0, min(0.48, min(1.0 - init_t - 0.05, new_b)))
                            self.main_window.slider_crop_b.setValue(int(new_b * 1000))
                    else:
                        # Aspect Ratio Constrained Dragging
                        if mode == 'BR':
                            x0, y0 = pm_x + init_l * pm_w, pm_y + init_t * pm_h
                            w_avail, h_avail = pm_w * (1.0 - init_l), pm_h * (1.0 - init_t)
                            w_raw = max(30.0, cur_x - x0)
                            h_raw = max(30.0, cur_y - y0)
                            w = max(w_raw, h_raw * ar)
                            if w > w_avail: w, h = w_avail, w_avail / ar
                            else: h = w / ar
                            if h > h_avail: h, w = h_avail, min(w_avail, h_avail * ar)
                            new_r = max(0.0, 1.0 - init_l - (w / float(pm_w)))
                            new_b = max(0.0, 1.0 - init_t - (h / float(pm_h)))
                            self.main_window.slider_crop_r.setValue(int(new_r * 1000))
                            self.main_window.slider_crop_b.setValue(int(new_b * 1000))

                        elif mode == 'TL':
                            x1, y1 = pm_x + (1.0 - init_r) * pm_w, pm_y + (1.0 - init_b) * pm_h
                            w_avail, h_avail = (1.0 - init_r) * pm_w, (1.0 - init_b) * pm_h
                            w_raw = max(30.0, x1 - cur_x)
                            h_raw = max(30.0, y1 - cur_y)
                            w = max(w_raw, h_raw * ar)
                            if w > w_avail: w, h = w_avail, w_avail / ar
                            else: h = w / ar
                            if h > h_avail: h, w = h_avail, min(w_avail, h_avail * ar)
                            new_l = max(0.0, (x1 - w - pm_x) / float(pm_w))
                            new_t = max(0.0, (y1 - h - pm_y) / float(pm_h))
                            self.main_window.slider_crop_l.setValue(int(new_l * 1000))
                            self.main_window.slider_crop_t.setValue(int(new_t * 1000))

                        elif mode == 'TR':
                            x0, y1 = pm_x + init_l * pm_w, pm_y + (1.0 - init_b) * pm_h
                            w_avail, h_avail = (1.0 - init_l) * pm_w, (1.0 - init_b) * pm_h
                            w_raw = max(30.0, cur_x - x0)
                            h_raw = max(30.0, y1 - cur_y)
                            w = max(w_raw, h_raw * ar)
                            if w > w_avail: w, h = w_avail, w_avail / ar
                            else: h = w / ar
                            if h > h_avail: h, w = h_avail, min(w_avail, h_avail * ar)
                            new_r = max(0.0, 1.0 - init_l - (w / float(pm_w)))
                            new_t = max(0.0, (y1 - h - pm_y) / float(pm_h))
                            self.main_window.slider_crop_r.setValue(int(new_r * 1000))
                            self.main_window.slider_crop_t.setValue(int(new_t * 1000))

                        elif mode == 'BL':
                            x1, y0 = pm_x + (1.0 - init_r) * pm_w, pm_y + init_t * pm_h
                            w_avail, h_avail = (1.0 - init_r) * pm_w, (1.0 - init_t) * pm_h
                            w_raw = max(30.0, x1 - cur_x)
                            h_raw = max(30.0, cur_y - y0)
                            w = max(w_raw, h_raw * ar)
                            if w > w_avail: w, h = w_avail, w_avail / ar
                            else: h = w / ar
                            if h > h_avail: h, w = h_avail, min(w_avail, h_avail * ar)
                            new_l = max(0.0, (x1 - w - pm_x) / float(pm_w))
                            new_b = max(0.0, 1.0 - init_t - (h / float(pm_h)))
                            self.main_window.slider_crop_l.setValue(int(new_l * 1000))
                            self.main_window.slider_crop_b.setValue(int(new_b * 1000))

                        elif mode in ['L', 'R']:
                            if mode == 'L':
                                new_l = max(0.0, min(0.48, (cur_x - pm_x) / float(pm_w)))
                                w = (1.0 - new_l - init_r) * pm_w
                                self.main_window.slider_crop_l.setValue(int(new_l * 1000))
                            else:
                                new_r = max(0.0, min(0.48, (pm_x + pm_w - cur_x) / float(pm_w)))
                                w = (1.0 - init_l - new_r) * pm_w
                                self.main_window.slider_crop_r.setValue(int(new_r * 1000))
                            h = w / ar
                            cy = (init_t + (1.0 - init_b)) / 2.0
                            new_t = max(0.0, min(0.48, cy - (h / pm_h) / 2.0))
                            new_b = max(0.0, min(0.48, 1.0 - (cy + (h / pm_h) / 2.0)))
                            self.main_window.slider_crop_t.setValue(int(new_t * 1000))
                            self.main_window.slider_crop_b.setValue(int(new_b * 1000))

                        elif mode in ['T', 'B']:
                            if mode == 'T':
                                new_t = max(0.0, min(0.48, (cur_y - pm_y) / float(pm_h)))
                                h = (1.0 - new_t - init_b) * pm_h
                                self.main_window.slider_crop_t.setValue(int(new_t * 1000))
                            else:
                                new_b = max(0.0, min(0.48, (pm_y + pm_h - cur_y) / float(pm_h)))
                                h = (1.0 - init_t - new_b) * pm_h
                                self.main_window.slider_crop_b.setValue(int(new_b * 1000))
                            w = h * ar
                            cx = (init_l + (1.0 - init_r)) / 2.0
                            new_l = max(0.0, min(0.48, cx - (w / pm_w) / 2.0))
                            new_r = max(0.0, min(0.48, 1.0 - (cx + (w / pm_w) / 2.0)))
                            self.main_window.slider_crop_l.setValue(int(new_l * 1000))
                            self.main_window.slider_crop_r.setValue(int(new_r * 1000))
                        
                    if mode == 'MOVE':
                        dx = (cur_x - self.crop_drag_start_pos.x()) / float(pm_w)
                        dy = (cur_y - self.crop_drag_start_pos.y()) / float(pm_h)
                        shift_x = max(-init_l, min(init_r, dx))
                        shift_y = max(-init_t, min(init_b, dy))
                        self.main_window.slider_crop_l.setValue(int((init_l + shift_x) * 1000))
                        self.main_window.slider_crop_r.setValue(int((init_r - shift_x) * 1000))
                        self.main_window.slider_crop_t.setValue(int((init_t + shift_y) * 1000))
                        self.main_window.slider_crop_b.setValue(int((init_b - shift_y) * 1000))
                        
                    if mode == 'ROTATE':
                        cx = pm_x + pm_w / 2.0
                        cy = pm_y + pm_h / 2.0
                        start_angle = math.atan2(self.crop_drag_start_pos.y() - cy, self.crop_drag_start_pos.x() - cx)
                        cur_angle = math.atan2(cur_y - cy, cur_x - cx)
                        delta_deg = math.degrees(cur_angle - start_angle)
                        new_rot = self.crop_init_rot + delta_deg
                        new_rot = max(-45.0, min(45.0, new_rot))
                        self.main_window.slider_rotation.setValue(int(new_rot * 10))
                        
                self.update()
                return
            else:
                hit_mode = self._get_crop_hit_mode(event.position())
                if hit_mode in ['TL', 'BR']: self.setCursor(Qt.SizeFDiagCursor)
                elif hit_mode in ['TR', 'BL']: self.setCursor(Qt.SizeBDiagCursor)
                elif hit_mode in ['T', 'B']: self.setCursor(Qt.SizeVerCursor)
                elif hit_mode in ['L', 'R']: self.setCursor(Qt.SizeHorCursor)
                elif hit_mode == 'MOVE': self.setCursor(Qt.SizeAllCursor)
                elif hit_mode == 'ROTATE': self.setCursor(self._get_rotate_cursor())
                else: self.unsetCursor()
                return

        # Split dragging & hover cursor
        split_mode = getattr(self.main_window, 'split_view_mode', None)
        if split_mode in ['vertical', 'horizontal'] and not self.is_cropping:
            pm = self.pixmap()
            if pm and not pm.isNull():
                pm_x = (self.width() - pm.width()) / 2.0
                pm_y = (self.height() - pm.height()) / 2.0
                if self.is_dragging_split and (event.buttons() & Qt.MouseButton.LeftButton):
                    if split_mode == 'vertical':
                        rel_x = (event.position().x() - pm_x) / float(pm.width())
                        self.main_window.split_position = max(0.01, min(0.99, rel_x))
                    elif split_mode == 'horizontal':
                        rel_y = (event.position().y() - pm_y) / float(pm.height())
                        self.main_window.split_position = max(0.01, min(0.99, rel_y))
                    if hasattr(self.main_window, 'engine') and self.main_window.engine:
                        self.main_window.engine.split_position = self.main_window.split_position
                        self.main_window.engine.split_view_mode = self.main_window.split_view_mode
                    self.main_window._update_view()
                    return
                elif not (event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton)):
                    if split_mode == 'vertical':
                        split_x = pm_x + getattr(self.main_window, 'split_position', 0.5) * pm.width()
                        if abs(event.position().x() - split_x) < 25:
                            self.setCursor(Qt.SizeHorCursor)
                            return
                    elif split_mode == 'horizontal':
                        split_y = pm_y + getattr(self.main_window, 'split_position', 0.5) * pm.height()
                        if abs(event.position().y() - split_y) < 25:
                            self.setCursor(Qt.SizeVerCursor)
                            return

        if not self.is_painting_mask and not self.is_picking_mask and not self.is_picking_wb and not self.is_picking_focus and not self.is_dragging_split:
            if not (event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton)):
                self.unsetCursor()

        # Viewport Navigation Pan
        if self.last_pos is not None:
            if event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.LeftButton):
                if not self.is_painting_mask:
                    delta = event.position() - self.last_pos
                    dx = (delta.x() / self.width()) * 2.0 / self.zoom
                    dy = (delta.y() / self.height()) * 2.0 / self.zoom
                    self.pan_x += dx
                    self.pan_y += dy
                    self.last_pos = event.position()
                    self.main_window._update_view()
                else:
                    self._handle_brush_draw(event, is_press=False)

    def _handle_brush_draw(self, event, is_press=False):
        if not hasattr(self.main_window, 'engine'):
            return
        px, py = event.position().x(), event.position().y()
        pm = self.pixmap()
        if pm and not pm.isNull():
            pm_w, pm_h = pm.width(), pm.height()
            x_offset = (self.width() - pm_w) / 2.0
            y_offset = (self.height() - pm_h) / 2.0
            img_px = px - x_offset
            img_py = py - y_offset
            if 0 <= img_px <= pm_w and 0 <= img_py <= pm_h:
                u, v = self.main_window.engine.get_uv_from_viewport(img_px, img_py, pm_w, pm_h)
                
                from PySide6.QtWidgets import QApplication
                mods = event.modifiers() | QApplication.keyboardModifiers()
                is_ctrl = bool(mods & Qt.ControlModifier)
                is_alt = bool(mods & Qt.AltModifier)
                is_right_btn = bool(event.buttons() & Qt.MouseButton.RightButton)
                erase = is_right_btn or is_ctrl or is_alt or self.is_erasing
                
                radius = self.paint_radius / 1000.0
                if is_press or self.last_brush_uv is None:
                    self.main_window.engine.paint_brush_mask(
                        u, v,
                        radius=radius,
                        hardness=self.paint_hardness,
                        opacity=self.paint_opacity,
                        erase=erase
                    )
                else:
                    u_prev, v_prev = self.last_brush_uv
                    self.main_window.engine.paint_brush_stroke(
                        u_prev, v_prev, u, v,
                        radius=radius,
                        hardness=self.paint_hardness,
                        opacity=self.paint_opacity,
                        erase=erase
                    )
                self.last_brush_uv = (u, v)
                self.main_window._update_view()

    def mouseReleaseEvent(self, event):
        self.last_brush_uv = None
        if self.is_cropping:
            self.crop_drag_mode = None
            self.crop_drag_start_pos = None
            self.update()
        if self.is_dragging_split:
            self.is_dragging_split = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F3:
            if hasattr(self.main_window, 'act_profiler_hud'):
                self.main_window.act_profiler_hud.setChecked(not self.show_profiler_hud)
            else:
                self.toggle_profiler_hud()
            event.accept()
            return
        if event.key() in [Qt.Key_Control, Qt.Key_Alt]:
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in [Qt.Key_Control, Qt.Key_Alt]:
            self.update()
        super().keyReleaseEvent(event)

    def wheelEvent(self, event):
        if not hasattr(self.main_window, 'engine') or not self.main_window.engine or not self.main_window.engine.texture:
            return
            
        zoom_factor = 1.15
        z_old = self.zoom
        
        if event.angleDelta().y() > 0:
            self.zoom *= zoom_factor
        else:
            self.zoom /= zoom_factor
            
        self.zoom = max(1.0, min(self.zoom, 20.0))
        z_new = self.zoom
        
        if z_new != z_old:
            # Zoom to cursor position
            cursor_ndc_x = (event.position().x() / max(1.0, float(self.width()))) * 2.0 - 1.0
            cursor_ndc_y = (event.position().y() / max(1.0, float(self.height()))) * 2.0 - 1.0
            
            self.pan_x = self.pan_x + (cursor_ndc_x - self.pan_x) * (1.0 - z_new / z_old)
            self.pan_y = self.pan_y + (cursor_ndc_y - self.pan_y) * (1.0 - z_new / z_old)
        
        # Reset pan when zoomed out fully
        if self.zoom == 1.0:
            self.pan_x = 0.0
            self.pan_y = 0.0
            
        if hasattr(self.main_window, '_update_view'):
            self.main_window._update_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self.main_window, '_update_view'):
            self.main_window._update_view()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        has_overlays = (
            self.show_profiler_hud or
            (self.is_painting_mask and self.hover_pos is not None) or
            (self.is_cropping and not self.is_loading) or
            (getattr(self.main_window, 'split_view_mode', None) in ['vertical', 'horizontal', 'side_by_side'] and not self.is_loading) or
            self.is_loading or
            self.is_background_loading or
            self.is_ai_running
        )
        if not has_overlays:
            return

        t_paint_start = time.perf_counter()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pm = self.pixmap()

            # 1. Brush Circle Preview (with Eraser mode indicator when Ctrl/Alt/Right-click is active)
            if self.is_painting_mask and self.hover_pos is not None and pm and not pm.isNull():
                screen_radius = (self.paint_radius / 1000.0) * pm.width()
                from PySide6.QtWidgets import QApplication
                mods = QApplication.keyboardModifiers()
                is_erasing_mode = bool(mods & (Qt.ControlModifier | Qt.AltModifier)) or self.is_erasing
                
                if is_erasing_mode:
                    painter.setPen(QPen(QColor(255, 70, 70, 230), 1.5))
                    painter.drawEllipse(self.hover_pos, screen_radius, screen_radius)
                    pen2 = QPen(QColor(255, 255, 255, 180), 1.5)
                    pen2.setDashPattern([3, 3])
                    painter.setPen(pen2)
                    painter.drawEllipse(self.hover_pos, screen_radius, screen_radius)
                else:
                    painter.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
                    painter.drawEllipse(self.hover_pos, screen_radius, screen_radius)
                    pen2 = QPen(QColor(0, 0, 0, 150), 1.5)
                    pen2.setDashPattern([2, 4])
                    painter.setPen(pen2)
                    painter.drawEllipse(self.hover_pos, screen_radius, screen_radius)

            # 2. Interactive Crop & Straighten Overlay
            if self.is_cropping and not self.is_loading:
                rect, pm_x, pm_y, pm_w, pm_h = self._get_crop_screen_rect()
                if rect and pm_w > 0 and pm_h > 0:
                    dim_brush = QBrush(QColor(0, 0, 0, 155))
                    if rect.top() > pm_y:
                        painter.fillRect(QRectF(pm_x, pm_y, pm_w, rect.top() - pm_y), dim_brush)
                    if pm_y + pm_h > rect.bottom():
                        painter.fillRect(QRectF(pm_x, rect.bottom(), pm_w, pm_y + pm_h - rect.bottom()), dim_brush)
                    if rect.left() > pm_x:
                        painter.fillRect(QRectF(pm_x, rect.top(), rect.left() - pm_x, rect.height()), dim_brush)
                    if pm_x + pm_w > rect.right():
                        painter.fillRect(QRectF(rect.right(), rect.top(), pm_x + pm_w - rect.right(), rect.height()), dim_brush)
                    
                    # Rule of Thirds
                    grid_pen = QPen(QColor(255, 255, 255, 65), 1, Qt.DashLine)
                    painter.setPen(grid_pen)
                    w3, h3 = rect.width() / 3.0, rect.height() / 3.0
                    painter.drawLine(rect.left() + w3, rect.top(), rect.left() + w3, rect.bottom())
                    painter.drawLine(rect.left() + 2*w3, rect.top(), rect.left() + 2*w3, rect.bottom())
                    painter.drawLine(rect.left(), rect.top() + h3, rect.right(), rect.top() + h3)
                    painter.drawLine(rect.left(), rect.top() + 2*h3, rect.right(), rect.top() + 2*h3)
                    
                    # White Box Border
                    painter.setPen(QPen(QColor(255, 255, 255, 220), 1.5, Qt.SolidLine))
                    painter.drawRect(rect)
                    
                    # 8 Corner L-Brackets & Midpoint Tabs
                    handle_pen = QPen(QColor(255, 255, 255, 255), 3.5, Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin)
                    painter.setPen(handle_pen)
                    L_len = min(22.0, min(rect.width(), rect.height()) / 3.0)
                    
                    painter.drawLine(rect.left(), rect.top(), rect.left() + L_len, rect.top())
                    painter.drawLine(rect.left(), rect.top(), rect.left(), rect.top() + L_len)
                    painter.drawLine(rect.right(), rect.top(), rect.right() - L_len, rect.top())
                    painter.drawLine(rect.right(), rect.top(), rect.right(), rect.top() + L_len)
                    painter.drawLine(rect.left(), rect.bottom(), rect.left() + L_len, rect.bottom())
                    painter.drawLine(rect.left(), rect.bottom(), rect.left(), rect.bottom() - L_len)
                    painter.drawLine(rect.right(), rect.bottom(), rect.right() - L_len, rect.bottom())
                    painter.drawLine(rect.right(), rect.bottom(), rect.right(), rect.bottom() - L_len)
                    
                    mid_x = rect.left() + rect.width() / 2.0
                    mid_y = rect.top() + rect.height() / 2.0
                    T_len = min(16.0, L_len)
                    painter.drawLine(mid_x - T_len/2, rect.top(), mid_x + T_len/2, rect.top())
                    painter.drawLine(mid_x - T_len/2, rect.bottom(), mid_x + T_len/2, rect.bottom())
                    painter.drawLine(rect.left(), mid_y - T_len/2, rect.left(), mid_y + T_len/2)
                    painter.drawLine(rect.right(), mid_y - T_len/2, rect.right(), mid_y + T_len/2)
                    
                    # Floating Angle HUD Badge
                    rot_deg = self.main_window.slider_rotation.value() / 10.0 if hasattr(self.main_window, 'slider_rotation') else 0.0
                    hud_text = f"📐 Aspect: {self.crop_aspect_name}  |  Straighten: {rot_deg:+.1f}°  |  [Enter / C] Apply"
                    font_hud = QFont("Segoe UI", 9, QFont.Bold)
                    painter.setFont(font_hud)
                    fm = painter.fontMetrics()
                    hud_w = fm.horizontalAdvance(hud_text) + 24
                    hud_h = 24.0
                    hud_x = rect.center().x() - hud_w / 2.0
                    hud_y = rect.top() - 32.0
                    if hud_y < pm_y: hud_y = rect.top() + 10.0
                    hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)
                    hud_path = QPainterPath()
                    hud_path.addRoundedRect(hud_rect, 12, 12)
                    painter.fillPath(hud_path, QBrush(QColor(20, 20, 20, 225)))
                    painter.setPen(QPen(QColor(0, 210, 255), 1.2))
                    painter.drawPath(hud_path)
                    painter.setPen(QColor(240, 240, 240))
                    painter.drawText(hud_rect, Qt.AlignCenter, hud_text)

            # 3. Split Comparison Badges & Divider Handle
            split_mode = getattr(self.main_window, 'split_view_mode', None)
            if split_mode in ['vertical', 'horizontal', 'side_by_side'] and not self.is_loading:
                if pm and not pm.isNull():
                    pm_x = (self.width() - pm.width()) / 2.0
                    pm_y = (self.height() - pm.height()) / 2.0
                    before_txt = "BEFORE"
                    after_txt = "AFTER"
                    badge_w = 80
                    
                    pm_w, pm_h = pm.width(), pm.height()
                    pm_x = (self.width() - pm_w) / 2.0
                    pm_y = (self.height() - pm_h) / 2.0
                    font_badge = QFont("Segoe UI", 9, QFont.Bold)
                    painter.setFont(font_badge)
                    
                    if split_mode in ['vertical', 'horizontal']:
                        # Badge "BEFORE"
                        b1_rect = QRectF(pm_x + 12, pm_y + 12, badge_w, 24)
                        painter.fillRect(b1_rect, QBrush(QColor(15, 15, 15, 210)))
                        painter.setPen(QPen(QColor(160, 160, 160), 1))
                        painter.drawRoundedRect(b1_rect, 4, 4)
                        painter.setPen(QColor(230, 230, 230))
                        painter.drawText(b1_rect, Qt.AlignCenter, before_txt)
                        
                        # Badge "AFTER"
                        if split_mode == 'horizontal':
                            b2_rect = QRectF(pm_x + 12, pm_y + pm_h - 36, badge_w, 24)
                        else:
                            b2_rect = QRectF(pm_x + pm_w - (badge_w + 12), pm_y + 12, badge_w, 24)
                        painter.fillRect(b2_rect, QBrush(QColor(15, 15, 15, 210)))
                        painter.setPen(QPen(QColor(0, 210, 255), 1))
                        painter.drawRoundedRect(b2_rect, 4, 4)
                        painter.setPen(QColor(0, 210, 255))
                        painter.drawText(b2_rect, Qt.AlignCenter, after_txt)
                        
                        # Split Divider & Drag Handle Icon
                        split_pos = float(getattr(self.main_window, 'split_position', 0.5))
                        if split_mode == 'vertical':
                            sx = pm_x + split_pos * pm_w
                            painter.setPen(QPen(QColor(0, 210, 255, 200), 2.0))
                            painter.drawLine(QPointF(sx, pm_y), QPointF(sx, pm_y + pm_h))
                            h_rect = QRectF(sx - 14, pm_y + pm_h / 2.0 - 14, 28, 28)
                            symbol = "⬌"
                        else:
                            sy = pm_y + split_pos * pm_h
                            painter.setPen(QPen(QColor(0, 210, 255, 200), 2.0))
                            painter.drawLine(QPointF(pm_x, sy), QPointF(pm_x + pm_w, sy))
                            h_rect = QRectF(pm_x + pm_w / 2.0 - 14, sy - 14, 28, 28)
                            symbol = "⬍"
                            
                        painter.setBrush(QBrush(QColor(25, 25, 25, 230)))
                        painter.setPen(QPen(QColor(0, 210, 255), 1.5))
                        painter.drawEllipse(h_rect)
                        painter.setPen(QColor(0, 210, 255))
                        painter.drawText(h_rect, Qt.AlignCenter, symbol)
                    elif split_mode == 'side_by_side':
                        divider_x = self.width() / 2.0
                        painter.setPen(QPen(QColor(40, 40, 40), 2.0))
                        painter.drawLine(QPointF(divider_x, 0), QPointF(divider_x, self.height()))

                        sbs_before = "BEFORE"
                        sbs_after = "AFTER"
                        
                        # Left half badge
                        b1_rect = QRectF(16, 16, badge_w, 24)
                        painter.fillRect(b1_rect, QBrush(QColor(15, 15, 15, 210)))
                        painter.setPen(QPen(QColor(160, 160, 160), 1))
                        painter.drawRoundedRect(b1_rect, 4, 4)
                        painter.setPen(QColor(230, 230, 230))
                        painter.drawText(b1_rect, Qt.AlignCenter, sbs_before)

                        # Right half badge
                        b2_rect = QRectF(divider_x + 16, 16, badge_w, 24)
                        painter.fillRect(b2_rect, QBrush(QColor(15, 15, 15, 210)))
                        painter.setPen(QPen(QColor(0, 210, 255), 1))
                        painter.drawRoundedRect(b2_rect, 4, 4)
                        painter.setPen(QColor(0, 210, 255))
                        painter.drawText(b2_rect, Qt.AlignCenter, sbs_after)

            # 4. Modal RAW Decoding / Opening Loading Card
            if self.is_loading:
                # Dim background
                painter.fillRect(self.rect(), QColor(0, 0, 0, 160))
                
                # Card
                card_w, card_h = 320.0, 100.0
                card_x = (self.width() - card_w) / 2.0
                card_y = (self.height() - card_h) / 2.0
                card_rect = QRectF(card_x, card_y, card_w, card_h)
                
                # Card Background & Border
                card_path = QPainterPath()
                card_path.addRoundedRect(card_rect, 10, 10)
                painter.fillPath(card_path, QBrush(QColor(25, 25, 25, 235)))
                painter.setPen(QPen(QColor(55, 55, 55), 1.5))
                painter.drawPath(card_path)
                
                # Title
                painter.setPen(QColor(255, 255, 255))
                font_title = QFont()
                font_title.setBold(True)
                font_title.setPixelSize(13)
                painter.setFont(font_title)
                title_rect = QRectF(card_rect.left() + 16, card_rect.top() + 16, card_rect.width() - 32, 22)
                painter.drawText(title_rect, Qt.AlignCenter, f"⏳ Opening {self.loading_filename}...")
                
                # Subtitle
                painter.setPen(QColor(0, 210, 255))
                font_sub = QFont()
                font_sub.setPixelSize(11)
                painter.setFont(font_sub)
                sub_rect = QRectF(card_rect.left() + 16, card_rect.top() + 42, card_rect.width() - 32, 20)
                painter.drawText(sub_rect, Qt.AlignCenter, self.loading_status_text)
                
                # Pulsing Bar
                bar_rect = QRectF(card_rect.left() + 30, card_rect.bottom() - 24, card_rect.width() - 60, 4)
                painter.fillRect(bar_rect, QColor(45, 45, 45))
                tick = getattr(self, 'loading_anim_tick', 0)
                t = (tick % 80) / 80.0
                pulse_w = 70.0
                pulse_x = bar_rect.left() + t * (bar_rect.width() - pulse_w)
                pulse_rect = QRectF(pulse_x, bar_rect.top(), pulse_w, 4)
                pulse_path = QPainterPath()
                pulse_path.addRoundedRect(pulse_rect, 2, 2)
                painter.fillPath(pulse_path, QBrush(QColor(0, 210, 255)))

            # 5. Background RAW Streaming Shimmer Bar & Floating Pill Badge
            if self.is_background_loading and not self.is_loading and not self.is_ai_running:
                tick = getattr(self, 'loading_anim_tick', 0)
                
                # --- Top & Bottom Neon Shimmer Bars ---
                bar_h = 5.0
                painter.fillRect(QRectF(0, 0, self.width(), bar_h), QColor(10, 15, 24, 220))
                painter.fillRect(QRectF(0, self.height() - bar_h, self.width(), bar_h), QColor(10, 15, 24, 220))
                
                shimmer_w = max(200.0, self.width() * 0.4)
                t = (tick % 75) / 75.0
                shimmer_x = -shimmer_w + t * (self.width() + 2 * shimmer_w)
                
                from PySide6.QtGui import QLinearGradient
                grad = QLinearGradient(shimmer_x, 0, shimmer_x + shimmer_w, 0)
                grad.setColorAt(0.0, QColor(0, 210, 255, 0))
                grad.setColorAt(0.3, QColor(0, 220, 255, 180))
                grad.setColorAt(0.5, QColor(255, 255, 255, 255))
                grad.setColorAt(0.7, QColor(0, 220, 255, 180))
                grad.setColorAt(1.0, QColor(0, 210, 255, 0))
                
                painter.fillRect(QRectF(shimmer_x, 0, shimmer_w, bar_h), QBrush(grad))
                painter.fillRect(QRectF(shimmer_x, self.height() - bar_h, shimmer_w, bar_h), QBrush(grad))
                
                # --- Floating Glowing Badge at Top Center ---
                status_msg = self.bg_loading_status_text or "Decoding full RAW in background..."
                badge_text = f"⚡ Instant Preview ({self.bg_loading_filename})  •  {status_msg}"
                font_bg = QFont("Segoe UI", 10, QFont.Bold)
                painter.setFont(font_bg)
                fm = painter.fontMetrics()
                bw = fm.horizontalAdvance(badge_text) + 48
                bh = 32.0
                bx = (self.width() - bw) / 2.0
                by = 16.0
                badge_rect = QRectF(bx, by, bw, bh)
                
                # Drop shadow behind badge
                shadow_rect = QRectF(bx + 2, by + 3, bw, bh)
                shadow_path = QPainterPath()
                shadow_path.addRoundedRect(shadow_rect, 16, 16)
                painter.fillPath(shadow_path, QBrush(QColor(0, 0, 0, 160)))
                
                # Main Badge body
                b_path = QPainterPath()
                b_path.addRoundedRect(badge_rect, 16, 16)
                painter.fillPath(b_path, QBrush(QColor(10, 16, 26, 245)))
                painter.setPen(QPen(QColor(0, 210, 255, 230), 1.5))
                painter.drawPath(b_path)
                
                # Pulsing Cyan Glow Indicator Dot
                pulse_alpha = int(170 + 85 * math.sin(tick * 0.2))
                painter.setBrush(QBrush(QColor(0, 230, 255, pulse_alpha)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(bx + 18, by + bh / 2.0), 5, 5)
                
                # Text
                painter.setPen(QColor(255, 255, 255))
                text_rect = QRectF(bx + 30, by, bw - 36, bh)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, badge_text)

            # 5b. AI Auto-Correction Streaming Shimmer Bar & Floating Pill Badge
            if self.is_ai_running and not self.is_loading:
                tick = getattr(self, 'loading_anim_tick', 0)
                
                # --- Top & Bottom AI Neon Shimmer Bars (Electric Violet & Cyan Gradient) ---
                bar_h = 5.0
                painter.fillRect(QRectF(0, 0, self.width(), bar_h), QColor(15, 10, 26, 220))
                painter.fillRect(QRectF(0, self.height() - bar_h, self.width(), bar_h), QColor(15, 10, 26, 220))
                
                shimmer_w = max(220.0, self.width() * 0.45)
                t = (tick % 70) / 70.0
                shimmer_x = -shimmer_w + t * (self.width() + 2 * shimmer_w)
                
                from PySide6.QtGui import QLinearGradient
                grad = QLinearGradient(shimmer_x, 0, shimmer_x + shimmer_w, 0)
                grad.setColorAt(0.0, QColor(140, 50, 255, 0))
                grad.setColorAt(0.25, QColor(180, 70, 255, 200))
                grad.setColorAt(0.5, QColor(0, 235, 255, 255))
                grad.setColorAt(0.75, QColor(180, 70, 255, 200))
                grad.setColorAt(1.0, QColor(140, 50, 255, 0))
                
                painter.fillRect(QRectF(shimmer_x, 0, shimmer_w, bar_h), QBrush(grad))
                painter.fillRect(QRectF(shimmer_x, self.height() - bar_h, shimmer_w, bar_h), QBrush(grad))
                
                # --- Floating Glowing AI Badge at Top Center ---
                badge_prefix = "✨ Gemini 3.8 Flash"
                if self.ai_pass_text:
                    badge_prefix += f" ({self.ai_pass_text})"
                status_msg = self.ai_status_text or "Auto-Correcting..."
                badge_text = f"{badge_prefix}  •  {status_msg}"
                
                font_ai = QFont("Segoe UI", 10, QFont.Bold)
                painter.setFont(font_ai)
                fm = painter.fontMetrics()
                bw = fm.horizontalAdvance(badge_text) + 48
                bh = 32.0
                bx = (self.width() - bw) / 2.0
                by = 16.0
                badge_rect = QRectF(bx, by, bw, bh)
                
                # Drop shadow behind badge
                shadow_rect = QRectF(bx + 2, by + 3, bw, bh)
                shadow_path = QPainterPath()
                shadow_path.addRoundedRect(shadow_rect, 16, 16)
                painter.fillPath(shadow_path, QBrush(QColor(0, 0, 0, 170)))
                
                # Main Badge body
                b_path = QPainterPath()
                b_path.addRoundedRect(badge_rect, 16, 16)
                painter.fillPath(b_path, QBrush(QColor(18, 12, 30, 245)))
                painter.setPen(QPen(QColor(180, 70, 255, 240), 1.5))
                painter.drawPath(b_path)
                
                # Pulsing Violet/Cyan Glow Indicator Dot
                pulse_alpha = int(170 + 85 * math.sin(tick * 0.25))
                painter.setBrush(QBrush(QColor(0, 235, 255, pulse_alpha)))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPointF(bx + 18, by + bh / 2.0), 5, 5)
                
                # Text
                painter.setPen(QColor(255, 255, 255))
                text_rect = QRectF(bx + 30, by, bw - 36, bh)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, badge_text)

            # 6. Real-time Performance Profiler HUD Overlay (F3)
            if self.show_profiler_hud and not self.is_loading:
                hud_w, hud_h = 240.0, 155.0
                margin = 16.0
                hud_x = self.width() - hud_w - margin
                hud_y = margin
                hud_rect = QRectF(hud_x, hud_y, hud_w, hud_h)

                # Card Background & Border
                hud_path = QPainterPath()
                hud_path.addRoundedRect(hud_rect, 10, 10)
                painter.fillPath(hud_path, QBrush(QColor(10, 16, 26, 235)))
                painter.setPen(QPen(QColor(0, 210, 255, 180), 1.2))
                painter.drawPath(hud_path)

                # Header: FPS & Total ms
                stats = profiler.get_summary()
                fps_val = stats['fps']
                tot_ms = stats['total_ms']
                gpu_ms = stats['gpu_render_ms']
                fbo_ms = stats['fbo_read_ms']
                pres_ms = stats['present_ms']
                paint_ms = stats['paint_ms']

                # FPS Color: Green/Cyan > 50, Yellow 30-50, Orange/Red < 30
                if fps_val >= 50.0:
                    fps_color = QColor(0, 242, 169) # Neon mint
                elif fps_val >= 30.0:
                    fps_color = QColor(255, 204, 0) # Gold
                else:
                    fps_color = QColor(255, 80, 80) # Red

                painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
                painter.setPen(fps_color)
                painter.drawText(QRectF(hud_x + 14, hud_y + 12, 130, 24), Qt.AlignLeft | Qt.AlignVCenter, f"⚡ {fps_val:.1f} FPS")

                painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
                painter.setPen(QColor(220, 220, 220))
                painter.drawText(QRectF(hud_x + 130, hud_y + 14, hud_w - 144, 20), Qt.AlignRight | Qt.AlignVCenter, f"{tot_ms:.1f} ms")

                # Divider
                painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
                painter.drawLine(QPointF(hud_x + 12, hud_y + 40), QPointF(hud_x + hud_w - 12, hud_y + 40))

                # Metric rows
                row_font = QFont("Segoe UI", 9)
                row_bold = QFont("Segoe UI", 9, QFont.Bold)
                
                rows = [
                    ("🟢 GPU Render:", f"{gpu_ms:.2f} ms", QColor(0, 230, 150)),
                    ("🔵 PCIe Readback:", f"{fbo_ms:.2f} ms", QColor(0, 200, 255)),
                    ("🟣 Qt Present:", f"{pres_ms:.2f} ms", QColor(190, 130, 255)),
                    ("🟡 Paint HUD:", f"{paint_ms:.2f} ms", QColor(255, 215, 0)),
                ]
                
                cur_y = hud_y + 46
                for label, val_str, dot_col in rows:
                    painter.setFont(row_font)
                    painter.setPen(QColor(180, 190, 205))
                    painter.drawText(QRectF(hud_x + 14, cur_y, 120, 18), Qt.AlignLeft | Qt.AlignVCenter, label)
                    
                    painter.setFont(row_bold)
                    painter.setPen(dot_col)
                    painter.drawText(QRectF(hud_x + 130, cur_y, hud_w - 144, 18), Qt.AlignRight | Qt.AlignVCenter, val_str)
                    cur_y += 20

                # Footer: Viewport dimensions
                vw = self.width()
                vh = self.height()
                painter.setFont(QFont("Segoe UI", 8))
                painter.setPen(QColor(120, 140, 160))
                painter.drawText(QRectF(hud_x + 14, cur_y + 2, hud_w - 28, 16), Qt.AlignCenter, f"Viewport: {vw}×{vh} px • [F3] HUD")
        finally:
            painter.end()
            profiler.record_metric("paint_ms", (time.perf_counter() - t_paint_start) * 1000.0)
