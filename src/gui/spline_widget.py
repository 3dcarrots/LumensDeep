import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTabWidget
from PySide6.QtGui import QPainter, QLinearGradient, QColor, QPen, QBrush, QPainterPath
from PySide6.QtCore import Qt, Signal, QRectF, QPointF

def catmull_rom_spline(P0, P1, P2, P3, num_points=10):
    """
    Computes a Catmull-Rom spline between P1 and P2.
    P0, P1, P2, P3 are (x,y) tuples or numpy arrays.
    """
    alpha = 0.5
    
    def get_t(t, p0, p1):
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = a ** (alpha * 0.5)
        return b + t

    t0 = 0.0
    t1 = get_t(t0, P0, P1)
    t2 = get_t(t1, P1, P2)
    t3 = get_t(t2, P2, P3)
    
    t = np.linspace(t1, t2, num_points)
    
    # Reshape for broadcasting
    t = t.reshape(-1, 1)
    
    A1 = (t1 - t) / (t1 - t0) * np.array(P0) + (t - t0) / (t1 - t0) * np.array(P1)
    A2 = (t2 - t) / (t2 - t1) * np.array(P1) + (t - t1) / (t2 - t1) * np.array(P2)
    A3 = (t3 - t) / (t3 - t2) * np.array(P2) + (t - t2) / (t3 - t2) * np.array(P3)
    
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    
    C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
    return C

import copy
from typing import Dict, Any, List, Optional

def sanitize_equalizer_ints(ints_list: Optional[List[int]]) -> List[int]:
    """
    Ensures a 7-point cyclic hue array (0°, 60°, 120°, 180°, 240°, 300°, 360°)
    has point 0 (0° Red) and point 6 (360° Red wrap-around) strictly synchronized and identical.
    """
    if ints_list is None:
        return [50] * 7
    res = [int(v) for v in ints_list]
    if len(res) < 7:
        if len(res) == 6:
            res.append(res[0])
        else:
            res = res + [50] * (7 - len(res))
    if len(res) >= 7:
        res = res[:7]
        res[6] = res[0]
    return [int(np.clip(round(v), 0, 100)) for v in res]

def sanitize_equalizer_dict(eq_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitizes a dictionary containing hue/saturation/brightness lists for ColorEqualizer or ShadowColorEqualizer.
    Guarantees that point 0 and point 6 are always identical for every curve.
    """
    if not isinstance(eq_dict, dict):
        return {'hue': [50] * 7, 'saturation': [50] * 7, 'brightness': [50] * 7}
    
    res = copy.deepcopy(eq_dict)
    for k_group in [
        ('hue', 'hue_nodes'),
        ('saturation', 'saturation_nodes', 'sat'),
        ('brightness', 'brightness_nodes', 'bri', 'lightness')
    ]:
        for k in k_group:
            if k in res and isinstance(res[k], (list, tuple)):
                res[k] = sanitize_equalizer_ints(res[k])
    return res

def calculate_color_equalizer_lut(hue_ints=None, sat_ints=None, bri_ints=None, size=1024) -> np.ndarray:
    """
    Computes a (size, 4) float32 LUT for Color Equalizer / Shadow Color Equalizer.
    Channel 0: Hue shift in degrees [-180, +180]
    Channel 1: Saturation multiplier [0.0, 2.0]
    Channel 2: Lightness offset [-0.5, +0.5]
    Channel 3: Unused / Alpha (0.0)
    """
    h_ints = sanitize_equalizer_ints(hue_ints)
    s_ints = sanitize_equalizer_ints(sat_ints)
    b_ints = sanitize_equalizer_ints(bri_ints)

    def _get_channel_lut(node_ints):
        nodes_y = [v / 100.0 for v in node_ints]
        w = size - 1
        nodes_x = [(i / 6.0) * w for i in range(7)]
        
        # Add phantom points for cyclic Catmull-Rom
        p_first = (nodes_x[-2] - w, nodes_y[-2])
        p_last = (nodes_x[1] + w, nodes_y[1])
        points = [p_first] + list(zip(nodes_x, nodes_y)) + [p_last]

        spline_pts = []
        for i in range(1, len(points) - 2):
            P0, P1, P2, P3 = points[i-1], points[i], points[i+1], points[i+2]
            dist = P2[0] - P1[0]
            num = max(int(dist), 2)
            segment = catmull_rom_spline(P0, P1, P2, P3, num)
            if i == len(points) - 3:
                spline_pts.extend(segment)
            else:
                spline_pts.extend(segment[:-1])

        x_vals = np.array([p[0] for p in spline_pts])
        y_vals = np.array([p[1] for p in spline_pts])
        target_x = np.linspace(0, size - 1, size)
        lut_1d = np.interp(target_x, x_vals, y_vals)
        lut_1d[-1] = lut_1d[0]
        return lut_1d

    h_norm = _get_channel_lut(h_ints)
    s_norm = _get_channel_lut(s_ints)
    b_norm = _get_channel_lut(b_ints)

    lut = np.zeros((size, 4), dtype=np.float32)
    lut[:, 0] = (h_norm - 0.5) * 360.0
    lut[:, 1] = s_norm * 2.0
    lut[:, 2] = (b_norm - 0.5)
    return lut

class SingleSplineEditor(QWidget):
    # Emits a list of dictionaries [{'hue': float, 'value': float}]
    nodes_changed = Signal(list)
    
    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.mode = mode # 'hue', 'saturation', 'brightness'
        
        # Initial nodes evenly spaced
        # value is centered at 0.5 (representing 0 offset or 1.0 scale depending on mode)
        self.nodes = [
            {'hue': 0.0, 'value': 0.5},
            {'hue': 60.0, 'value': 0.5},
            {'hue': 120.0, 'value': 0.5},
            {'hue': 180.0, 'value': 0.5},
            {'hue': 240.0, 'value': 0.5},
            {'hue': 300.0, 'value': 0.5},
            {'hue': 360.0, 'value': 0.5} # wrap around
        ]
        
        self.active_node = None
        self.margin = 15

    def sync_cyclic(self):
        """Guarantees the first (0° Red) and last (360° Red) nodes have strictly identical values."""
        if len(self.nodes) >= 2:
            self.nodes[-1]['value'] = self.nodes[0]['value']

    def paintEvent(self, event):
        self.sync_cyclic()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        # Draw gradient background (Spectrum)
        gradient = QLinearGradient(self.margin, 0, self.width() - self.margin, 0)
        gradient.setColorAt(0.0, QColor.fromHsvF(0.0, 1.0, 1.0))
        gradient.setColorAt(0.166, QColor.fromHsvF(0.166, 1.0, 1.0))
        gradient.setColorAt(0.333, QColor.fromHsvF(0.333, 1.0, 1.0))
        gradient.setColorAt(0.5, QColor.fromHsvF(0.5, 1.0, 1.0))
        gradient.setColorAt(0.666, QColor.fromHsvF(0.666, 1.0, 1.0))
        gradient.setColorAt(0.833, QColor.fromHsvF(0.833, 1.0, 1.0))
        gradient.setColorAt(1.0, QColor.fromHsvF(1.0, 1.0, 1.0))
        
        # Dim the background a bit so the line is visible
        painter.fillRect(self.margin, self.margin, w, h, QBrush(gradient))
        painter.fillRect(self.margin, self.margin, w, h, QColor(0, 0, 0, 150)) # Dark overlay
        
        # Draw Center Line
        painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.DashLine))
        painter.drawLine(self.margin, int(self.margin + h/2), self.width() - self.margin, int(self.margin + h/2))
        
        # Calculate full spline points
        pts = self.get_spline_points(w, h)
        
        # Draw Spline
        path = QPainterPath()
        if len(pts) > 0:
            path.moveTo(self.margin + pts[0][0], self.margin + h - pts[0][1])
            for p in pts[1:]:
                path.lineTo(self.margin + p[0], self.margin + h - p[1])
                
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawPath(path)
        
        # Draw Nodes
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        for i, node in enumerate(self.nodes):
            x = self.margin + (node['hue'] / 360.0) * w
            y = self.margin + h - (node['value'] * h)
            
            if i == self.active_node:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
            else:
                painter.setBrush(QBrush(QColor(50, 50, 50)))
                
            painter.drawEllipse(QPointF(x, y), 5, 5)

    def get_spline_points(self, w, h):
        """Returns physical coordinates (x, y) for drawing, where y is from bottom."""
        self.sync_cyclic()
        if len(self.nodes) < 2: return []
        
        # We need to wrap around for Catmull-Rom (P0 and Pn+1)
        nodes_x = [(n['hue'] / 360.0) * w for n in self.nodes]
        nodes_y = [n['value'] * h for n in self.nodes]
        
        # Add phantom points for cyclic Catmull-Rom
        p_first = (nodes_x[-2] - w, nodes_y[-2])
        p_last = (nodes_x[1] + w, nodes_y[1])
        
        points = [p_first] + list(zip(nodes_x, nodes_y)) + [p_last]
        
        spline_pts = []
        for i in range(1, len(points) - 2):
            P0 = points[i-1]
            P1 = points[i]
            P2 = points[i+1]
            P3 = points[i+2]
            
            # calculate distance to decide number of segments
            dist = P2[0] - P1[0]
            num = max(int(dist), 2)
            
            segment = catmull_rom_spline(P0, P1, P2, P3, num)
            
            # Exclude last point of segment to avoid duplicates, except for the very last segment
            if i == len(points) - 3:
                spline_pts.extend(segment)
            else:
                spline_pts.extend(segment[:-1])
                
        return spline_pts

    def get_lut_array(self, size=1024):
        """Generates a 1D numpy array of length `size` with the interpolated values."""
        self.sync_cyclic()
        pts = self.get_spline_points(size - 1, 1.0)
        # Extract Y values (they are already scaled 0-1 because we passed h=1.0)
        # Interpolate exactly `size` points linearly if they don't match exactly
        x_vals = np.array([p[0] for p in pts])
        y_vals = np.array([p[1] for p in pts])
        
        # Ensure strictly increasing X for np.interp
        target_x = np.linspace(0, size - 1, size)
        lut = np.interp(target_x, x_vals, y_vals)
        lut[-1] = lut[0]
        return lut

    def mousePressEvent(self, event):
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        pos = event.position()
        mx, my = pos.x(), pos.y()
        
        # Check if clicked on a node
        best_dist = 15.0
        best_node = None
        for i, node in enumerate(self.nodes):
            nx = self.margin + (node['hue'] / 360.0) * w
            ny = self.margin + h - (node['value'] * h)
            dist = np.hypot(mx - nx, my - ny)
            if dist < best_dist:
                best_dist = dist
                best_node = i
                
        if best_node is not None:
            self.active_node = best_node
            self.update()

    def mouseMoveEvent(self, event):
        if self.active_node is not None:
            w = self.width() - 2 * self.margin
            h = self.height() - 2 * self.margin
            
            # We don't allow changing hue (X axis) to keep things simple and ordered
            # We only allow changing value (Y axis)
            my = event.position().y()
            val = 1.0 - (my - self.margin) / h
            val = np.clip(val, 0.0, 1.0)
            
            # If changing the first or last node (0 and 360), sync them to keep it cyclic
            if self.active_node == 0 or self.active_node == len(self.nodes) - 1:
                self.nodes[0]['value'] = val
                self.nodes[-1]['value'] = val
            else:
                self.nodes[self.active_node]['value'] = val
                
            self.sync_cyclic()
            self.update()
            self.nodes_changed.emit(self.nodes)

    def mouseReleaseEvent(self, event):
        self.active_node = None
        self.sync_cyclic()
        self.update()
        
class ColorEqualizerWidget(QWidget):
    lut_changed = Signal(np.ndarray) # Emits the full 1024x4 LUT
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        self.hue_editor = SingleSplineEditor('hue')
        self.sat_editor = SingleSplineEditor('saturation')
        self.bri_editor = SingleSplineEditor('brightness')
        
        self.tabs.addTab(self.sat_editor, "Saturation")
        self.tabs.addTab(self.bri_editor, "Brightness")
        self.tabs.addTab(self.hue_editor, "Hue")
        
        self.hue_editor.nodes_changed.connect(self.on_nodes_changed)
        self.sat_editor.nodes_changed.connect(self.on_nodes_changed)
        self.bri_editor.nodes_changed.connect(self.on_nodes_changed)
        
    def on_nodes_changed(self, nodes):
        self.generate_and_emit_lut()
        
    def set_nodes(self, mode, nodes):
        if mode == 'hue':
            self.hue_editor.nodes = nodes
            self.hue_editor.sync_cyclic()
            self.hue_editor.update()
        elif mode == 'saturation':
            self.sat_editor.nodes = nodes
            self.sat_editor.sync_cyclic()
            self.sat_editor.update()
        elif mode == 'brightness':
            self.bri_editor.nodes = nodes
            self.bri_editor.sync_cyclic()
            self.bri_editor.update()
        self.generate_and_emit_lut()
        
    def set_nodes_from_ints(self, hue_ints, sat_ints, bri_ints):
        h_clean = sanitize_equalizer_ints(hue_ints)
        s_clean = sanitize_equalizer_ints(sat_ints)
        b_clean = sanitize_equalizer_ints(bri_ints)

        for i, val in enumerate(h_clean):
            if i < len(self.hue_editor.nodes): self.hue_editor.nodes[i]['value'] = np.clip(val / 100.0, 0.0, 1.0)
        for i, val in enumerate(s_clean):
            if i < len(self.sat_editor.nodes): self.sat_editor.nodes[i]['value'] = np.clip(val / 100.0, 0.0, 1.0)
        for i, val in enumerate(b_clean):
            if i < len(self.bri_editor.nodes): self.bri_editor.nodes[i]['value'] = np.clip(val / 100.0, 0.0, 1.0)

        self.hue_editor.sync_cyclic()
        self.sat_editor.sync_cyclic()
        self.bri_editor.sync_cyclic()

        self.hue_editor.update()
        self.sat_editor.update()
        self.bri_editor.update()
        self.generate_and_emit_lut()
        
    def get_hue_nodes_as_ints(self):
        self.hue_editor.sync_cyclic()
        raw = [int(round(n['value'] * 100)) for n in self.hue_editor.nodes]
        return sanitize_equalizer_ints(raw)
        
    def get_saturation_nodes_as_ints(self):
        self.sat_editor.sync_cyclic()
        raw = [int(round(n['value'] * 100)) for n in self.sat_editor.nodes]
        return sanitize_equalizer_ints(raw)
        
    def get_brightness_nodes_as_ints(self):
        self.bri_editor.sync_cyclic()
        raw = [int(round(n['value'] * 100)) for n in self.bri_editor.nodes]
        return sanitize_equalizer_ints(raw)
        
    def get_lut_array(self):
        self.hue_editor.sync_cyclic()
        self.sat_editor.sync_cyclic()
        self.bri_editor.sync_cyclic()
        return calculate_color_equalizer_lut(
            self.get_hue_nodes_as_ints(),
            self.get_saturation_nodes_as_ints(),
            self.get_brightness_nodes_as_ints(),
            size=1024
        )

    def generate_and_emit_lut(self):
        lut = self.get_lut_array()
        self.lut_changed.emit(lut)
def calculate_tone_equalizer_lut(node_ints, size=1024):
    """
    Computes a 1024-point 1D float32 LUT for Tone Equalizer.
    node_ints: list of 9 integers (0..100), default 50.
    Nodes at EV: -8, -7, -6, -5, -4, -3, -2, -1, 0 EV.
    Node 8 (0 EV) is the rightmost point controlling diffuse white & all blowouts.
    EV adjustment range: 0 -> -4.0 EV, 50 -> 0.0 EV, 100 -> +3.0 EV.
    """
    if node_ints is None or len(node_ints) != 9:
        node_ints = [50] * 9
        
    ev_positions = np.array([-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    nodes = [{'ev': ev, 'value': val / 100.0} for ev, val in zip(ev_positions, node_ints)]
    
    # Smooth interior nodes while preserving boundary authority
    smoothed = []
    for i in range(len(nodes)):
        n = nodes[i].copy()
        if i == 0 or i == len(nodes) - 1:
            n['value'] = nodes[i]['value']
        else:
            n['value'] = (nodes[i-1]['value'] + nodes[i]['value'] * 2.0 + nodes[i+1]['value']) / 4.0
        smoothed.append(n)
        
    w = size - 1
    h = 1.0
    nodes_x = [((n['ev'] + 8.0) / 8.0) * w for n in smoothed]
    nodes_y = [n['value'] * h for n in smoothed]
    
    dx_start = nodes_x[1] - nodes_x[0]
    dx_end = nodes_x[-1] - nodes_x[-2]
    p_first = (nodes_x[0] - dx_start, nodes_y[0])
    p_last = (nodes_x[-1] + dx_end, nodes_y[-1])
    
    points = [p_first] + list(zip(nodes_x, nodes_y)) + [p_last]
    spline_pts = []
    for i in range(1, len(points) - 2):
        seg = catmull_rom_spline(points[i-1], points[i], points[i+1], points[i+2], 25)
        if i == len(points) - 3:
            spline_pts.extend(seg)
        else:
            spline_pts.extend(seg[:-1])
            
    x_vals = np.array([p[0] for p in spline_pts])
    y_vals = np.array([p[1] for p in spline_pts])
    
    target_x = np.linspace(0, size - 1, size)
    norm_lut = np.interp(target_x, x_vals, y_vals)
    
    # Map [0, 1] to [-4.0 EV, +3.0 EV]
    lut = np.where(norm_lut < 0.5, (norm_lut - 0.5) * 8.0, (norm_lut - 0.5) * 6.0)
    return lut.astype(np.float32)

class ToneEqualizerWidget(QWidget):
    lut_changed = Signal(np.ndarray) # Emits the full 1024x1 LUT for tone EQ
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        
        # 9 Nodes from -8 EV to 0 EV
        self.nodes = [{'ev': float(i), 'value': 0.5} for i in range(-8, 1)]
        self.active_node = None
        self.margin = 15

    def set_nodes_from_ints(self, ints):
        for i, val in enumerate(ints):
            if i < len(self.nodes):
                self.nodes[i]['value'] = np.clip(val / 100.0, 0.0, 1.0)
        self.update()
        lut = self.get_lut_array(1024)
        self.lut_changed.emit(lut)
        
    def get_nodes_as_ints(self):
        return [int(round(n['value'] * 100)) for n in self.nodes]

    def get_smoothed_nodes(self):
        if len(self.nodes) < 3:
            return self.nodes
            
        smoothed = []
        for i in range(len(self.nodes)):
            n = self.nodes[i].copy()
            if i == 0 or i == len(self.nodes) - 1:
                n['value'] = self.nodes[i]['value']
            else:
                n['value'] = (self.nodes[i-1]['value'] + self.nodes[i]['value'] * 2.0 + self.nodes[i+1]['value']) / 4.0
            smoothed.append(n)
        return smoothed

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        # Draw background gradient
        gradient = QLinearGradient(self.margin, self.margin, w + self.margin, self.margin)
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(1.0, QColor(255, 255, 255))
        painter.fillRect(self.margin, self.margin, w, h, QBrush(gradient))
        painter.fillRect(self.margin, self.margin, w, h, QColor(0, 0, 0, 100)) # Dark overlay
        
        # Draw Center Line (0.0 EV)
        painter.setPen(QPen(QColor(150, 150, 150), 1, Qt.PenStyle.DashLine))
        painter.drawLine(self.margin, int(self.margin + h/2), self.width() - self.margin, int(self.margin + h/2))
        
        # Draw Grid Lines for 9 zones
        painter.setPen(QPen(QColor(100, 100, 100, 100), 1))
        for i in range(1, 9):
            x = self.margin + (i / 8.0) * w
            painter.drawLine(int(x), self.margin, int(x), self.margin + h)
        
        # Draw smoothed curve dimly
        smoothed_nodes = self.get_smoothed_nodes()
        pts_smooth = self.get_spline_points(w, h, smoothed_nodes)
        
        path_smooth = QPainterPath()
        if len(pts_smooth) > 0:
            path_smooth.moveTo(self.margin + pts_smooth[0][0], self.margin + h - pts_smooth[0][1])
            for p in pts_smooth[1:]:
                path_smooth.lineTo(self.margin + p[0], self.margin + h - p[1])
                
        painter.setPen(QPen(QColor(150, 150, 150, 180), 3))
        painter.drawPath(path_smooth)
        
        # Draw main control curve
        pts = self.get_spline_points(w, h, self.nodes)
        
        path = QPainterPath()
        if len(pts) > 0:
            path.moveTo(self.margin + pts[0][0], self.margin + h - pts[0][1])
            for p in pts[1:]:
                path.lineTo(self.margin + p[0], self.margin + h - p[1])
                
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1, Qt.PenStyle.DashLine))
        painter.drawPath(path)
        
        for i, node in enumerate(self.nodes):
            x = self.margin + ((node['ev'] + 8.0) / 8.0) * w
            y = self.margin + h - (node['value'] * h)
            
            if i == self.active_node:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
            else:
                painter.setBrush(QBrush(QColor(50, 50, 50)))
                
            painter.drawEllipse(QPointF(x, y), 5, 5)

    def get_spline_points(self, w, h, nodes_override=None):
        nodes = nodes_override if nodes_override is not None else self.nodes
        if len(nodes) < 2: return []
        
        nodes_x = [((n['ev'] + 8.0) / 8.0) * w for n in nodes]
        nodes_y = [n['value'] * h for n in nodes]
        
        # Non-cyclic catmull-rom requires distinct phantom points to avoid division by zero
        dx_start = nodes_x[1] - nodes_x[0]
        dx_end = nodes_x[-1] - nodes_x[-2]
        
        p_first = (nodes_x[0] - dx_start, nodes_y[0])
        p_last = (nodes_x[-1] + dx_end, nodes_y[-1])
        
        points = [p_first] + list(zip(nodes_x, nodes_y)) + [p_last]
        
        spline_pts = []
        for i in range(1, len(points) - 2):
            segment = catmull_rom_spline(points[i-1], points[i], points[i+1], points[i+2], 25)
            if i == len(points) - 3:
                spline_pts.extend(segment)
            else:
                spline_pts.extend(segment[:-1])
                
        return spline_pts

    def get_lut_array(self, size=1024):
        return calculate_tone_equalizer_lut(self.get_nodes_as_ints(), size)

    def mousePressEvent(self, event):
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        
        mx, my = event.position().x(), event.position().y()
        best_dist = 15.0
        best_node = None
        for i, node in enumerate(self.nodes):
            nx = self.margin + ((node['ev'] + 8.0) / 8.0) * w
            ny = self.margin + h - (node['value'] * h)
            dist = np.hypot(mx - nx, my - ny)
            if dist < best_dist:
                best_dist = dist
                best_node = i
                
        if best_node is not None:
            self.active_node = best_node
            self.update()

    def mouseMoveEvent(self, event):
        if self.active_node is not None:
            h = self.height() - 2 * self.margin
            my = event.position().y()
            val = 1.0 - (my - self.margin) / h
            self.nodes[self.active_node]['value'] = np.clip(val, 0.0, 1.0)
            self.update()
            
            lut = self.get_lut_array(1024)
            self.lut_changed.emit(lut)

    def mouseReleaseEvent(self, event):
        self.active_node = None
        self.update()
