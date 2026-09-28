import os
import sys
import threading
from pydantic import Field
try:
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from mcp.server.mcpserver import MCPServer as FastMCP
from PySide6.QtCore import QObject, Signal

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.image_io import load_image_to_linear, export_image_from_linear
from core.engine import PhotoEngine

mcp = FastMCP("PhotoEditorServer")

class AISignals(QObject):
    image_load_requested = Signal(str)
    # exp, cont, gamma, temp, chroma, vibrance, local_cont, tint, whites, shadows, highlights
    global_adjusted = Signal(float, float, float, float, float, float, float, float, float, float, float)
    
    # hue_nodes, sat_nodes, bri_nodes (lists of dicts: {'hue': float, 'value': float})
    color_equalizer_adjusted = Signal(list, list, list)
    
    # ev_nodes (list of dicts: {'ev': float, 'value': float})
    tone_equalizer_adjusted = Signal(list)
    
    geometry_adjusted = Signal(float, float, float, float, float, float)
    lens_detail_adjusted = Signal(float, float)
    render_requested = Signal()
    export_requested = Signal(str, bool)

# Global signals instance for GUI communication
signals = AISignals()

# Global state that will be initialized by GUI
class State:
    def __init__(self):
        self.engine = None
        self.image_loaded = False
        self.hue_nodes = []
        self.sat_nodes = []
        self.bri_nodes = []
        self.ev_nodes = [{'ev': float(i), 'value': 0.5} for i in range(-8, 1)]

state = State()

@mcp.tool()
def load_image(path: str) -> str:
    """Loads a photo (CR2, JPEG, PNG) into the GPU for editing."""
    if not os.path.exists(path):
        return f"Error: File not found at {path}"
    
    # Notify GUI to load the image
    signals.image_load_requested.emit(path)
    return f"Requested GUI to load {path}"

@mcp.tool()
def apply_global_adjustments(
    exposure: float = Field(1.0, description="Exposure. 1.0 is default. >1 makes brighter."),
    contrast: float = Field(1.0, description="Contrast. 1.0 is default. >1 increases contrast."),
    gamma: float = Field(1.0, description="Gamma correction for midtones. 1.0 is default. >1 brightens midtones, <1 darkens."),
    temperature: float = Field(0.0, description="Temperature. 0.0 is default. >0 warms, <0 cools."),
    global_chroma: float = Field(1.0, description="Global Chroma. 1.0 is default. Linear multiplier to chroma."),
    global_vibrance: float = Field(0.0, description="Global Vibrance. 0.0 is default. Non-linear boost to dull colors."),
    local_contrast: float = Field(0.0, description="Local Contrast. 0.0 is default. >0 increases micro-contrast (details)."),
    tint: float = Field(0.0, description="Tint. 0.0 is default. >0 adds Magenta, <0 adds Green."),
    whites: float = Field(0.0, description="Whites. 0.0 is default. >0 brightens white point."),
    shadows: float = Field(0.0, description="Shadows. 0.0 is default. >0 lifts shadows."),
    highlights: float = Field(0.0, description="Highlights. 0.0 is default. <0 recovers highlights.")
) -> str:
    """Applies global adjustments to the entire image."""
    signals.global_adjusted.emit(exposure, contrast, gamma, temperature, global_chroma, global_vibrance, local_contrast, tint, whites, shadows, highlights)
    return f"Applied global adjustments"

@mcp.tool()
def apply_geometry(
    rotation: float = Field(0.0, description="Rotation angle in degrees (-45.0 to 45.0)"),
    flip_90: int = Field(0, description="Rotation by 90 degrees in steps (0, 1, 2, 3)"),
    crop_l: float = Field(0.0, description="Crop from left (0.0 to 0.5)"),
    crop_r: float = Field(0.0, description="Crop from right (0.0 to 0.5)"),
    crop_t: float = Field(0.0, description="Crop from top (0.0 to 0.5)"),
    crop_b: float = Field(0.0, description="Crop from bottom (0.0 to 0.5)")
) -> str:
    """Applies crop and rotation to the image."""
    signals.geometry_adjusted.emit(rotation, float(flip_90), crop_l, crop_r, crop_t, crop_b)
    return "Applied geometry adjustments"

@mcp.tool()
def apply_lens_and_detail(
    vignette: float = Field(0.0, description="Vignette amount. 0.0 is default. >0 darkens edges."),
    sharpen: float = Field(0.0, description="Sharpen amount. 0.0 is default. >0 increases sharpness.")
) -> str:
    """Applies lens and detail adjustments."""
    signals.lens_detail_adjusted.emit(vignette, sharpen)
    return "Applied lens and detail adjustments"

@mcp.tool()
def apply_color_equalizer(
    mode: str = Field(..., description="Which curve to edit: 'hue', 'saturation', or 'brightness'"),
    hue_angle: float = Field(..., ge=0.0, le=360.0, description="The hue angle (0-360) where you want to add/move a point."),
    value: float = Field(..., description="The value for the point. For hue: shift in degrees. For sat: scale multiplier (1.0 is default). For bri: lightness offset (0.0 is default).")
) -> str:
    """Adds or updates a control point in the Color Equalizer curve."""
    target_nodes = None
    if mode == 'hue': target_nodes = state.hue_nodes
    elif mode == 'saturation': target_nodes = state.sat_nodes
    elif mode == 'brightness': target_nodes = state.bri_nodes
    else: return "Error: Invalid mode."
    
    # Update or add point
    replaced = False
    for i, node in enumerate(target_nodes):
        if abs(node['hue'] - hue_angle) < 5.0:
            target_nodes[i] = {'hue': hue_angle, 'value': value}
            replaced = True
            break
            
    if not replaced:
        target_nodes.append({'hue': hue_angle, 'value': value})
        target_nodes.sort(key=lambda x: x['hue'])

    # Synchronize cyclic 0° and 360° Red nodes
    first_node = next((n for n in target_nodes if abs(n.get('hue', 0) - 0.0) < 5.0), None)
    last_node = next((n for n in target_nodes if abs(n.get('hue', 0) - 360.0) < 5.0), None)
    if first_node and last_node:
        if abs(hue_angle - 0.0) <= 5.0:
            last_node['value'] = first_node['value']
        elif abs(hue_angle - 360.0) <= 5.0:
            first_node['value'] = last_node['value']
        else:
            last_node['value'] = first_node['value']
        
    signals.color_equalizer_adjusted.emit(state.hue_nodes, state.sat_nodes, state.bri_nodes)
    return f"Updated {mode} curve at hue {hue_angle} with value {value}."

@mcp.tool()
def apply_tone_equalizer(
    ev_level: float = Field(..., ge=-8.0, le=0.0, description="The exposure level to adjust (from -8 to 0). Will snap to nearest integer node."),
    ev_adjust: float = Field(..., ge=-2.0, le=2.0, description="The compensation to apply (from -2.0 EV to +2.0 EV).")
) -> str:
    """Updates a control point in the Tone Equalizer curve."""
    # Snapping to nearest integer EV node
    node_idx = int(round(ev_level)) + 8
    node_idx = max(0, min(8, node_idx))
    
    # Map EV [-2.0, 2.0] to [0.0, 1.0] value space (where 0.5 is 0 EV)
    value = (ev_adjust / 4.0) + 0.5
    value = max(0.0, min(1.0, value))
    
    state.ev_nodes[node_idx]['value'] = value
    signals.tone_equalizer_adjusted.emit(state.ev_nodes)
    return f"Updated Tone Equalizer at {ev_level} EV by {ev_adjust} EV."

try:
    from mcp.server.mcpserver import Image
except ImportError:
    try:
        from mcp.server.fastmcp import Image
    except ImportError:
        from mcp.types import ImageContent as Image

@mcp.tool()
def get_current_preview() -> Image:
    """Returns a visual preview of the current adjustments."""
    if not state.image_loaded or not state.engine:
        raise ValueError("No image loaded.")
        
    # We must ensure render is called and full bokeh is accumulated
    if getattr(state.engine, 'bokeh_enabled', False):
        state.engine.ensure_full_bokeh_accumulation()
    else:
        state.engine.render(to_preview=True)
    jpeg_bytes = state.engine.get_preview_jpeg_bytes(max_size=1024)
    return Image(data=jpeg_bytes, format="jpeg")

@mcp.tool()
def export_image(output_path: str, apply_gamma: bool = Field(True, description="Apply gamma for JPEG.")) -> str:
    """Exports the current state of the image to a file."""
    signals.export_requested.emit(output_path, apply_gamma)
    return f"Requested export to {output_path}"

def run_mcp_server(transport="sse", port=8000):
    """Runs the MCP server. Should be called in a background thread."""
    if transport == "sse":
        # FastMCP might not accept port directly on run()
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    # If run standalone, use stdio
    run_mcp_server(transport="stdio")
