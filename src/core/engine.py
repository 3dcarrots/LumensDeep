import moderngl
import numpy as np
from PIL import Image
import io
import base64
import os
from dataclasses import dataclass, field
import copy
from typing import Tuple, Optional, List, Dict, Any
from core.profiler import profiler

@dataclass
class Layer:
    id: str
    name: str
    is_base: bool = False
    active: bool = True
    
    # Layer Preset Settings
    settings: dict = field(default_factory=dict)
    
    # Parametric Mask Settings
    mask_hue_target: float = 0.0
    mask_hue_range: float = 0.0
    mask_luma_target: float = 0.5
    mask_luma_range: float = 0.0
    mask_sat_target: float = 0.5
    mask_sat_range: float = 0.0
    mask_softness: float = 0.5
    mask_opacity: float = 1.0
    mask_fill: float = 0.0
    mask_resolution: int = 4 # 1=Full, 2=Half, 4=Quarter, 8=Eighth
    
    # Gradient Mask Settings
    mask_gradient_type: float = 0.0 # 0=None, 1=L2R, 2=R2L, 3=T2B, 4=B2T, 5=Radial
    mask_grad_black: float = 0.0
    mask_grad_white: float = 1.0
    mask_grad_gamma: float = 0.5
    
    # 3D Mask Settings
    use_depth_mask: bool = False
    mask_depth_min: float = 0.0
    mask_depth_max: float = 1.0
    mask_depth_softness: float = 0.5
    mask_depth_gamma: float = 1.0
    
    use_normal_mask: bool = False
    mask_normal_dir: tuple = (1.0, 0.0, 1.0)
    mask_normal_angle: float = 0.5
    mask_normal_softness: float = 0.1
    mask_normal_gamma: float = 1.02
    
    # Texture ID mapping (set by Engine)
    mask_tex_id: int = -1

class PhotoEngine:
    def __init__(self, vertex_shader_path: str, fragment_shader_path: str, plugin_manager=None):
        self.active_layer_index = 0
        self.ctx = moderngl.create_standalone_context(require=330)
        self.plugin_manager = plugin_manager
        
        with open(vertex_shader_path, 'r') as f:
            self.vertex_shader_src = f.read()
        with open(fragment_shader_path, 'r') as f:
            fragment_shader_src = f.read()
            
        if self.plugin_manager is not None:
            fragment_shader_src = self.plugin_manager.build_fragment_shader()
            
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader_src,
            fragment_shader=fragment_shader_src
        )
        
        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'mask_blend.glsl'), 'r') as f:
                mask_blend_src = f.read()
            self.mask_blend_program = self.ctx.program(
                vertex_shader='#version 330 core\n#define VERTEX_SHADER\n' + mask_blend_src.replace('#version 330', ''),
                fragment_shader='#version 330 core\n#define FRAGMENT_SHADER\n' + mask_blend_src.replace('#version 330', '')
            )
            # Bind textures for mask_blend
            if 'accum_tex' in self.mask_blend_program: self.mask_blend_program['accum_tex'].value = 0
            if 'layer_tex' in self.mask_blend_program: self.mask_blend_program['layer_tex'].value = 1
            if 'orig_tex' in self.mask_blend_program: self.mask_blend_program['orig_tex'].value = 2
            if 'mask_tex' in self.mask_blend_program: self.mask_blend_program['mask_tex'].value = 3
            if 'global_depth_tex' in self.mask_blend_program: self.mask_blend_program['global_depth_tex'].value = 4
            if 'global_normal_tex' in self.mask_blend_program: self.mask_blend_program['global_normal_tex'].value = 5
        except Exception as e:
            print("Failed to load mask_blend.glsl", e)
            
        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'brush.glsl'), 'r') as f:
                brush_src = f.read()
            self.brush_program = self.ctx.program(
                vertex_shader='#version 330 core\n#define VERTEX_SHADER\n' + brush_src.replace('#version 330', ''),
                fragment_shader='#version 330 core\n#define FRAGMENT_SHADER\n' + brush_src.replace('#version 330', '')
            )
        except Exception as e:
            print("Failed to load brush.glsl", e)

        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'lens_correction.glsl'), 'r') as f:
                lens_src = f.read()
            self.lens_program = self.ctx.program(
                vertex_shader=self.vertex_shader_src,
                fragment_shader=lens_src
            )
            if 'image_texture' in self.lens_program: self.lens_program['image_texture'].value = 0
            if 'uv_map' in self.lens_program: self.lens_program['uv_map'].value = 1
            
            # Use identity matrix for lens correction (full screen quad)
            identity = np.eye(4, dtype='f4').tobytes()
            if 'mvp' in self.lens_program: self.lens_program['mvp'].write(identity)
            if 'mvp_base' in self.lens_program: self.lens_program['mvp_base'].write(identity)
        except Exception as e:
            print("Failed to load lens_correction.glsl", e)

        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'blur_brush.glsl'), 'r') as f:
                blur_brush_src = f.read()
            self.blur_brush_program = self.ctx.program(
                vertex_shader='#version 330 core\n#define VERTEX_SHADER\n' + blur_brush_src.replace('#version 330', ''),
                fragment_shader='#version 330 core\n#define FRAGMENT_SHADER\n' + blur_brush_src.replace('#version 330', '')
            )
            self.blur_brush_program['mask_tex'].value = 0
        except Exception as e:
            print("Failed to load blur_brush.glsl", e)
        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'bokeh_raymarch.glsl'), 'r') as f:
                raymarch_src = f.read()
            self.bokeh_raymarch_prog = self.ctx.program(
                vertex_shader='#version 330 core\n#define VERTEX_SHADER\n' + self.vertex_shader_src.replace('#version 330 core', '').replace('#version 330', ''),
                fragment_shader=raymarch_src
            )
            
            if 'origTex' in self.bokeh_raymarch_prog: self.bokeh_raymarch_prog['origTex'].value = 0
            if 'depthTex' in self.bokeh_raymarch_prog: self.bokeh_raymarch_prog['depthTex'].value = 1
            if 'maskTex' in self.bokeh_raymarch_prog: self.bokeh_raymarch_prog['maskTex'].value = 2

        except Exception as e:
            print("Failed to load bokeh_raymarch shader", e)

        try:
            with open(os.path.join(os.path.dirname(vertex_shader_path), 'preview_map.glsl'), 'r') as f:
                preview_map_src = f.read()
            
            preview_map_vs = preview_map_src.replace('#version 330 core', '#version 330 core\n#define VERTEX_SHADER')
            preview_map_fs = preview_map_src.replace('#version 330 core', '#version 330 core\n#define FRAGMENT_SHADER')
            
            self.preview_map_prog = self.ctx.program(
                vertex_shader=preview_map_vs,
                fragment_shader=preview_map_fs
            )
            if 'tex' in self.preview_map_prog: self.preview_map_prog['tex'].value = 0
        except Exception as e:
            print("Failed to load preview_map shader", e)
        # Quad geometry
        vertices = np.array([
            # x, y, u, v
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 1.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        self.mask_blend_vao = self.ctx.vertex_array(
            self.mask_blend_program,
            [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        self.brush_vao = self.ctx.vertex_array(
            self.brush_program,
            [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        if hasattr(self, 'lens_program'):
            self.lens_vao = self.ctx.vertex_array(
                self.lens_program,
                [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )
        self.blur_brush_vao = self.ctx.vertex_array(
            self.blur_brush_program,
            [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        if hasattr(self, 'bokeh_raymarch_prog'):
            self.bokeh_raymarch_vao = self.ctx.vertex_array(self.bokeh_raymarch_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        if hasattr(self, 'preview_map_prog'):
            self.preview_map_vao = self.ctx.vertex_array(self.preview_map_prog, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])
        
        # Neutral textures for Before pass
        self.neutral_ceq_texture = self.ctx.texture((1, 1), 4, np.array([0.0, 1.0, 0.0, 1.0], dtype='f4').tobytes(), dtype='f4')
        self.neutral_tone_texture = self.ctx.texture((1, 1), 1, np.array([0.0], dtype='f4').tobytes(), dtype='f4')

        # Split comparison shader (Before / After)
        split_vs = """
        #version 330 core
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        split_fs = """
        #version 330 core
        uniform sampler2D tex_before;
        uniform sampler2D tex_after;
        uniform int u_split_mode; // 1: vertical, 2: horizontal, 3: side_by_side
        uniform float u_split_pos;
        uniform vec2 u_resolution; // Viewport dimensions (prev_w, prev_h)
        uniform vec2 u_tex_size;   // FBO Texture dimensions (e.g. 4096, 4096)
        in vec2 uv;
        out vec4 fragColor;
        
        void main() {
            vec2 uv_scale = u_resolution / u_tex_size;
            
            if (u_split_mode == 1) {
                // Vertical split (Left: Before, Right: After)
                float border = 1.5 / max(1.0, u_resolution.x);
                if (abs(uv.x - u_split_pos) < border) {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                } else if (uv.x < u_split_pos) {
                    fragColor = texture(tex_before, uv * uv_scale);
                } else {
                    fragColor = texture(tex_after, uv * uv_scale);
                }
            } else if (u_split_mode == 2) {
                // Horizontal split (Top: Before, Bottom: After)
                float border = 1.5 / max(1.0, u_resolution.y);
                float split_y = u_split_pos;
                if (abs(uv.y - split_y) < border) {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                } else if (uv.y < split_y) {
                    fragColor = texture(tex_before, uv * uv_scale);
                } else {
                    fragColor = texture(tex_after, uv * uv_scale);
                }
            } else if (u_split_mode == 3) {
                // Side-by-Side (Left: Before in half-viewport, Right: After in half-viewport)
                float border = 1.5 / max(1.0, u_resolution.x);
                if (abs(uv.x - 0.5) < border) {
                    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
                } else if (uv.x < 0.5) {
                    vec2 uv_half = vec2(uv.x * 2.0, uv.y);
                    vec2 half_scale = vec2(floor(u_resolution.x * 0.5), u_resolution.y) / u_tex_size;
                    fragColor = texture(tex_before, uv_half * half_scale);
                } else {
                    vec2 uv_half = vec2((uv.x - 0.5) * 2.0, uv.y);
                    vec2 half_scale = vec2(floor(u_resolution.x * 0.5), u_resolution.y) / u_tex_size;
                    fragColor = texture(tex_after, uv_half * half_scale);
                }
            } else {
                fragColor = texture(tex_after, uv * uv_scale);
            }
        }
        """
        try:
            self.split_prog = self.ctx.program(
                vertex_shader=split_vs,
                fragment_shader=split_fs
            )
            self.split_vao = self.ctx.vertex_array(
                self.split_prog,
                [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )
        except Exception as e:
            print("Failed to compile split shader:", e)

        # Fast GPU Presentation Pass (Linear HDR Float -> IEC 61966-2-1 sRGB 8-bit Display)
        display_vs = """
        #version 330 core
        in vec2 in_position;
        in vec2 in_texcoord;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        display_fs = """
        #version 330 core
        uniform sampler2D accum_tex;
        uniform vec2 u_resolution;
        uniform vec2 u_tex_size;
        in vec2 uv;
        out vec4 fragColor;
        
        void main() {
            vec2 uv_scale = u_resolution / u_tex_size;
            vec4 raw_col = texture(accum_tex, uv * uv_scale);
            vec3 c = raw_col.rgb * 50.0;
            c = clamp(c, 0.0, 1.0);
            
            // Exact IEC 61966-2-1 standard sRGB transfer curve on GPU
            vec3 srgb_low = c * 12.92;
            vec3 srgb_high = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
            vec3 srgb = mix(srgb_high, srgb_low, lessThanEqual(c, vec3(0.0031308)));
            
            fragColor = vec4(srgb, 1.0);
        }
        """
        try:
            self.display_prog = self.ctx.program(
                vertex_shader=display_vs,
                fragment_shader=display_fs
            )
            self.display_prog['accum_tex'].value = 0
            self.display_vao = self.ctx.vertex_array(
                self.display_prog,
                [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )
        except Exception as e:
            print("Failed to compile display shader:", e)
        
        self.width = 0
        self.height = 0
        self.texture = None
        
        # Global Geometry Textures
        self.global_depth_tex = None
        self.global_normal_tex = None
        self.global_subject_mask_tex = None
        self.dummy_tex_1d = self.ctx.texture((1, 1), 1, b'\xff')
        self.dummy_tex_3d = self.ctx.texture((1, 1), 3, b'\x00\x00\x00')
        
        self.fbo = None
        self.renderbuffer = None
        self.preview_display_fbo = None
        self.preview_display_tex = None
        # Color Equalizer LUT (Texture Unit 1)
        # 1024 pixels, 4 channels (R=HueShift, G=SatScale, B=LightnessOffset, A=Unused)
        self.ceq_texture = self.ctx.texture((1024, 1), 4, dtype='f4')
        self.ceq_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        if 'color_equalizer_lut' in self.program:
            self.program['color_equalizer_lut'].value = 1
            
        # Shadow Color Equalizer LUT (Texture Unit 3)
        self.shadow_ceq_texture = self.ctx.texture((1024, 1), 4, dtype='f4')
        self.shadow_ceq_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        if 'shadow_color_equalizer_lut' in self.program:
            self.program['shadow_color_equalizer_lut'].value = 3
            # AI Blend parameter (opacity of color adjustments)
        self.ai_blend = 1.0
            
        # Tone Equalizer LUT (Texture Unit 2)
        self.tone_eq_texture = self.ctx.texture((1024, 1), 1, dtype='f4')
        self.tone_eq_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tone_eq_texture.repeat_x = False
        self.tone_eq_texture.repeat_y = False
        if 'tone_equalizer_lut' in self.program:
            self.program['tone_equalizer_lut'].value = 2

        # Neutral LUTs for before/split comparison and initial state
        neutral_ceq_data = np.zeros((1024, 4), dtype=np.float32)
        neutral_ceq_data[:, 1] = 1.0 # default saturation scale is 1.0
        self.neutral_ceq_texture = self.ctx.texture((1024, 1), 4, data=neutral_ceq_data.tobytes(), dtype='f4')
        self.neutral_ceq_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        neutral_tone_data = np.zeros(1024, dtype=np.float32) # 0.0 EV adjustment
        self.neutral_tone_texture = self.ctx.texture((1024, 1), 1, data=neutral_tone_data.tobytes(), dtype='f4')
        self.neutral_tone_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Write neutral data to active LUT textures as well
        self.ceq_texture.write(neutral_ceq_data.tobytes())
        self.shadow_ceq_texture.write(neutral_ceq_data.tobytes())
        self.tone_eq_texture.write(neutral_tone_data.tobytes())
        
        # Default uniforms
        self.layers = []
        self.mask_textures = {} # layer_id -> moderngl.Texture
        self.default_mask_tex = self.ctx.texture((1, 1), 1, data=b'\xff', dtype='f1')
        self.enable_lens_correction = False
        self.uv_map_texture = None
        self.lens_info = ""
        self._reset_uniforms()

        # We only need FBOs for accumulation, no longer need multipass bokeh FBOs
        self.preview_accum_fbo_1 = None
        self.preview_accum_fbo_2 = None
        self.render_accum_fbo_1 = None
        self.render_accum_fbo_2 = None

    def _reset_uniforms(self):
        # Geometry state
        self.crop_l = 0.0
        self.crop_r = 0.0
        self.crop_t = 0.0
        self.crop_b = 0.0
        self.rotation_deg = 0.0
        self.flip_90 = 0 # 0, 1, 2, 3 (* 90 degrees)
        self.black_level = 0.0
        self.out_w = 0
        self.out_h = 0
        self.prev_w = 0
        self.prev_h = 0
        self.base_curve_strength = 0.30
        self.is_crop_interactive_mode = False
        self.set_geometry(0.0, 0, 0.0, 0.0, 0.0, 0.0)
        
        self.view_zoom = 1.0
        self.view_pan_x = 0.0
        self.view_pan_y = 0.0
        
        # Bokeh parameters
        self.bokeh_enabled = False
        self.bokeh_focal_depth = 0.5
        self.bokeh_focal_range = 0.12
        self.bokeh_radius = 20.0
        self.bokeh_petals = 6
        self.bokeh_highlight_boost = 10.0
        self.bokeh_chromatic = 0.5
        self.bokeh_vignette = 0.5
        self.bokeh_radial_center = 0.0
        self.bokeh_radial_edge = 0.0
        self.bokeh_radial_gamma = 1.0
        self.bokeh_fg_gamma = 1.0
        self.bokeh_bg_gamma = 1.0
        self.bokeh_subject_influence = 0.0
        self.bokeh_max_samples = 256
        
        # Global Color
        self.set_global_adjustments(1.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        
        # Reset Color Equalizer
        default_lut = np.zeros((1024, 4), dtype='f4')
        default_lut[:, 1] = 1.0 # default saturation scale is 1.0
        self.update_color_equalizer_lut(default_lut)
        self.update_shadow_color_equalizer_lut(default_lut)
        
        # Split comparison mode
        self.split_view_mode = None # None, 'vertical', 'horizontal', 'side_by_side', 'before'
        self.split_position = 0.5
        
        self.frame_count = 0
        self.MVP_preview = np.eye(4, dtype=np.float32)
        self.MVP_export = np.eye(4, dtype=np.float32)
        self.source_texture = None
        self.has_ai_denoised = False

    def load_texture(self, linear_rgb_data: np.ndarray, metadata: dict = None):
        """Loads a float32 linear RGB array into the engine."""
        self.metadata = metadata
        if metadata and 'thumb_bytes' in metadata:
            self.thumb_bytes = metadata['thumb_bytes']
        else:
            self.thumb_bytes = None

        self.height, self.width = linear_rgb_data.shape[:2]
        
        # Ensure GPU finishes any pending operations before releasing/reallocating
        try:
            self.ctx.finish()
        except:
            pass
            
        # Release all old framebuffer attachments and textures
        for attr in [
            'raw_texture', 'texture', 'source_texture', 'fbo', 'renderbuffer', 'raw_fbo',
            'export_fbo_1', 'export_accum_1', 'export_fbo_2', 'export_accum_2',
            'export_layer_fbo', 'export_layer_tex',
            'preview_fbo', 'preview_renderbuffer',
            'preview_display_fbo', 'preview_display_tex',
            'preview_fbo_1', 'preview_accum_1', 'preview_fbo_2', 'preview_accum_2',
            'preview_fbo_bokeh', 'preview_accum_bokeh',
            'preview_layer_fbo', 'preview_layer_tex', 'uv_map_texture',
            'global_depth_tex', 'global_normal_tex', 'global_subject_mask_tex',
            'raw_depth_tex', 'raw_normal_tex', 'raw_subject_mask_tex'
        ]:
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if obj:
                    try: obj.release()
                    except: pass
                setattr(self, attr, None)

        self._last_layer_accum = None
        self.frame_count = 0
        self.has_ai_denoised = False

        if hasattr(self, 'mask_textures'):
            for tex in list(self.mask_textures.values()):
                if tex:
                    try: tex.release()
                    except: pass
            self.mask_textures.clear()
            
        # Upload linear RGB data directly to GPU texture without slow CPU alpha concatenation
        num_components = linear_rgb_data.shape[2]
        dtype_str = 'f2' if linear_rgb_data.dtype == np.float16 else 'f4'
        self.raw_texture = self.ctx.texture(
            (self.width, self.height), 
            num_components, 
            linear_rgb_data.tobytes(), 
            dtype=dtype_str
        )
        self.raw_texture.repeat_x = False
        self.raw_texture.repeat_y = False
        self.raw_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        self.texture = self.ctx.texture(
            (self.width, self.height), 
            4, 
            dtype=dtype_str
        )
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        
        if num_components == 4:
            self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_texture])
        else:
            self.raw_fbo = None
        
        # Generate UV Map & Apply initial lens correction
        self.update_lens_uv_map(metadata)
        
        # Create FBO for fast preview rendering (4096 max preview size)
        max_prev = 4096
        self.preview_renderbuffer = self.ctx.texture((max_prev, max_prev), 4, dtype='f2')
        self.preview_renderbuffer.repeat_x = False
        self.preview_renderbuffer.repeat_y = False
        self.preview_fbo = self.ctx.framebuffer(color_attachments=[self.preview_renderbuffer])
        
        # Create FBO for fast 8-bit preview display presentation
        self.preview_display_tex = self.ctx.texture((max_prev, max_prev), 3, dtype='f1')
        self.preview_display_tex.repeat_x = False
        self.preview_display_tex.repeat_y = False
        self.preview_display_fbo = self.ctx.framebuffer(color_attachments=[self.preview_display_tex])
        
        # Accumulation FBOs for preview
        self.preview_accum_1 = self.ctx.texture((max_prev, max_prev), 4, dtype='f2')
        self.preview_accum_1.repeat_x = False
        self.preview_accum_1.repeat_y = False
        self.preview_fbo_1 = self.ctx.framebuffer(color_attachments=[self.preview_accum_1])
        
        self.preview_accum_2 = self.ctx.texture((max_prev, max_prev), 4, dtype='f2')
        self.preview_accum_2.repeat_x = False
        self.preview_accum_2.repeat_y = False
        self.preview_fbo_2 = self.ctx.framebuffer(color_attachments=[self.preview_accum_2])
        
        self.preview_accum_bokeh = self.ctx.texture((max_prev, max_prev), 4, dtype='f2')
        self.preview_accum_bokeh.repeat_x = False
        self.preview_accum_bokeh.repeat_y = False
        self.preview_fbo_bokeh = self.ctx.framebuffer(color_attachments=[self.preview_accum_bokeh])

        self.preview_layer_tex = self.ctx.texture((max_prev, max_prev), 4, dtype='f2')
        self.preview_layer_tex.repeat_x = False
        self.preview_layer_tex.repeat_y = False
        self.preview_layer_fbo = self.ctx.framebuffer(color_attachments=[self.preview_layer_tex])

        self._update_mvp()

    def update_image_data(self, linear_rgb_data: np.ndarray, is_denoised: bool = True):
        """
        Updates the engine's image data in-memory (e.g. after AI NAFNet denoise).
        Preserves all existing layer adjustments, depth/normal maps, masks, and lens profile,
        while archiving the original pre-denoise texture as self.source_texture for
        Split V, Split H, and Side-by-Side before/after comparison.
        """
        if linear_rgb_data is None:
            return

        h, w = linear_rgb_data.shape[:2]
        if w != self.width or h != self.height:
            self.load_texture(linear_rgb_data, getattr(self, 'metadata', None))
            return

        try:
            self.ctx.finish()
        except:
            pass

        dtype_str = 'f2' if linear_rgb_data.dtype == np.float16 else 'f4'
        num_components = linear_rgb_data.shape[2]

        if is_denoised:
            if getattr(self, 'source_texture', None) is None and getattr(self, 'texture', None) is not None:
                self.source_texture = self.texture
                self.texture = None
            elif getattr(self, 'texture', None) is not None:
                try: self.texture.release()
                except: pass
                self.texture = None
            self.has_ai_denoised = True

        if getattr(self, 'raw_texture', None) is not None:
            try: self.raw_texture.release()
            except: pass
            self.raw_texture = None

        if getattr(self, 'raw_fbo', None) is not None:
            try: self.raw_fbo.release()
            except: pass
            self.raw_fbo = None

        self.raw_texture = self.ctx.texture(
            (self.width, self.height),
            num_components,
            linear_rgb_data.tobytes(),
            dtype=dtype_str
        )
        self.raw_texture.repeat_x = False
        self.raw_texture.repeat_y = False
        self.raw_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        if num_components == 4:
            self.raw_fbo = self.ctx.framebuffer(color_attachments=[self.raw_texture])

        self.texture = self.ctx.texture(
            (self.width, self.height),
            4,
            dtype=dtype_str
        )
        self.texture.repeat_x = False
        self.texture.repeat_y = False

        self.apply_lens_correction()
        try:
            self.ctx.finish()
        except:
            pass
        self._last_layer_accum = None
        self.frame_count = 0

    def _ensure_export_fbos(self):
        max_dim = int(np.ceil(np.hypot(self.width, self.height)))
        max_dim = min(8192, max(self.width, self.height, max_dim))
        if getattr(self, 'renderbuffer', None) is None or self.renderbuffer.size[0] < max_dim:
            if getattr(self, 'renderbuffer', None) is not None:
                try: self.renderbuffer.release()
                except: pass
            if getattr(self, 'export_accum_1', None) is not None:
                try: self.export_accum_1.release()
                except: pass
            if getattr(self, 'export_accum_2', None) is not None:
                try: self.export_accum_2.release()
                except: pass
            if getattr(self, 'export_layer_tex', None) is not None:
                try: self.export_layer_tex.release()
                except: pass
            if getattr(self, 'export_accum_bokeh', None) is not None:
                try: self.export_accum_bokeh.release()
                except: pass

            self.renderbuffer = self.ctx.texture((max_dim, max_dim), 4, dtype='f2')
            self.renderbuffer.repeat_x = False
            self.renderbuffer.repeat_y = False
            self.fbo = self.ctx.framebuffer(color_attachments=[self.renderbuffer])
            
            self.export_accum_1 = self.ctx.texture((max_dim, max_dim), 4, dtype='f2')
            self.export_accum_1.repeat_x = False
            self.export_accum_1.repeat_y = False
            self.export_fbo_1 = self.ctx.framebuffer(color_attachments=[self.export_accum_1])
            
            self.export_accum_2 = self.ctx.texture((max_dim, max_dim), 4, dtype='f2')
            self.export_accum_2.repeat_x = False
            self.export_accum_2.repeat_y = False
            self.export_fbo_2 = self.ctx.framebuffer(color_attachments=[self.export_accum_2])
            
            self.export_layer_tex = self.ctx.texture((max_dim, max_dim), 4, dtype='f2')
            self.export_layer_tex.repeat_x = False
            self.export_layer_tex.repeat_y = False
            self.export_layer_fbo = self.ctx.framebuffer(color_attachments=[self.export_layer_tex])

            self.export_accum_bokeh = self.ctx.texture((max_dim, max_dim), 4, dtype='f2')
            self.export_accum_bokeh.repeat_x = False
            self.export_accum_bokeh.repeat_y = False
            self.export_fbo_bokeh = self.ctx.framebuffer(color_attachments=[self.export_accum_bokeh])

    def update_lens_uv_map(self, metadata=None):
        if metadata is not None:
            self.metadata = metadata
            
        uv_map_data = self.generate_lens_uv_map(self.metadata)
        if uv_map_data:
            coords_opengl, info, grid_w, grid_h = uv_map_data
            
            if self.uv_map_texture:
                try: self.uv_map_texture.release()
                except: pass
            self.uv_map_texture = self.ctx.texture_array(
                (grid_w, grid_h, 3), 2, coords_opengl.tobytes(), dtype='f4'
            )
            self.uv_map_texture.repeat_x = False
            self.uv_map_texture.repeat_y = False
            self.uv_map_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.has_lens_profile = True
            self.lens_info = info
        else:
            if self.uv_map_texture:
                try: self.uv_map_texture.release()
                except: pass
            self.uv_map_texture = None
            self.has_lens_profile = False
            self.lens_info = "No matching lens profile found"
            
        self.apply_lens_correction()

    def generate_lens_uv_map(self, metadata):
        if not metadata: return None
        try:
            import lensfunpy
            import numpy as np
            import difflib
            db = lensfunpy.Database()
            cam_maker = str(metadata.get('cam_maker', '') or '').strip()
            cam_model = str(metadata.get('cam_model', '') or '').strip()
            lens_maker = str(metadata.get('lens_maker', '') or '').strip()
            lens_model = str(metadata.get('lens_model', '') or '').strip()
            
            # 1. Find Camera with robust fuzzy matching
            cams = db.find_cameras(cam_maker, cam_model) if (cam_maker and cam_model) else []
            if not cams and cam_model:
                all_cams = [c for c in db.cameras if (not cam_maker or cam_maker.lower() in c.maker.lower())]
                matches = [c for c in all_cams if c.model.lower() in cam_model.lower() or cam_model.lower() in c.model.lower()]
                if matches:
                    cams = matches
                else:
                    m_names = [c.model for c in all_cams]
                    close = difflib.get_close_matches(cam_model, m_names, n=1, cutoff=0.3)
                    if close:
                        cams = [c for c in all_cams if c.model == close[0]]

            if not cams:
                return None
            cam = cams[0]
            
            # 2. Find Lens with robust fuzzy matching
            lenses = db.find_lenses(cam, lens_maker, lens_model) if (lens_maker and lens_model) else []
            if not lenses and lens_model:
                all_lenses = db.find_lenses(cam, '', '')
                san = lambda s: s.lower().replace(' ', '').replace('-', '').replace('f/', 'f').replace('canon', '').replace('nikon', '').replace('sony', '')
                clean_lm = san(lens_model)
                matches = [l for l in all_lenses if clean_lm in san(l.model) or san(l.model) in clean_lm]
                if matches:
                    lenses = matches
                else:
                    m_names = [l.model for l in all_lenses]
                    close = difflib.get_close_matches(lens_model, m_names, n=1, cutoff=0.3)
                    if close:
                        lenses = [l for l in all_lenses if l.model == close[0]]

            if not lenses:
                return None
            lens = lenses[0]
            
            focal_length = metadata.get('focal_length', 50.0)
            aperture = metadata.get('aperture', 2.8)
            
            # Use smooth grid resolution (max 512) for ultra-fast, crash-proof subpixel distortion
            max_grid = 512
            aspect = float(self.width) / float(max(1, self.height))
            if aspect >= 1.0:
                grid_w = max_grid
                grid_h = max(32, int(max_grid / aspect))
            else:
                grid_h = max_grid
                grid_w = max(32, int(max_grid * aspect))

            mod = lensfunpy.Modifier(lens, cam.crop_factor, grid_w, grid_h)
            mod.initialize(focal_length, aperture, distance=10.0, 
                           flags=lensfunpy.ModifyFlags.ALL)
                           
            if hasattr(mod, 'apply_subpixel_geometry_distortion'):
                undist_coords = mod.apply_subpixel_geometry_distortion()
            elif hasattr(mod, 'apply_geometry_distortion'):
                geom = mod.apply_geometry_distortion()
                undist_coords = np.stack([geom, geom, geom], axis=2)
            else:
                undist_coords = mod.apply_subpixel_distortion()
            if undist_coords is None:
                print("Lensfun: apply_subpixel_distortion returned None")
                return None
                
            # 1. Auto-crop / Auto-scale bounds: find the valid inner rectangular bounds with a subtle 0.3% margin
            min_x = float(undist_coords[..., 0].min())
            max_x = float(undist_coords[..., 0].max())
            min_y = float(undist_coords[..., 1].min())
            max_y = float(undist_coords[..., 1].max())
            
            margin_x = (max_x - min_x) * 0.0035
            margin_y = (max_y - min_y) * 0.0035
            
            eff_min_x = min_x + margin_x
            eff_max_x = max_x - margin_x
            eff_min_y = min_y + margin_y
            eff_max_y = max_y - margin_y
            
            span_x = max(1e-6, eff_max_x - eff_min_x)
            span_y = max(1e-6, eff_max_y - eff_min_y)

            # 2. Flip Y axis of grid to match OpenGL texture coords (OpenGL (0,0) is bottom-left, Lensfun (0,0) is top-left)
            coords_opengl_grid = np.flip(undist_coords, axis=0)
            
            # 3. Scale and normalize seamlessly so the corrected image spans [0.0 .. 1.0] edge-to-edge with zero borders
            norm_coords = np.zeros_like(coords_opengl_grid, dtype=np.float32)
            norm_coords[..., 0] = np.clip((coords_opengl_grid[..., 0] - eff_min_x) / span_x, 0.0, 1.0)
            norm_coords[..., 1] = np.clip(1.0 - ((coords_opengl_grid[..., 1] - eff_min_y) / span_y), 0.0, 1.0)
            
            # 4. Transpose from (grid_h, grid_w, 3_layers, 2_components) to (3_layers, grid_h, grid_w, 2_components) for texture_array
            coords_opengl = np.ascontiguousarray(np.transpose(norm_coords, (2, 0, 1, 3)))
            
            return coords_opengl.astype(np.float32), f"{cam.maker} {cam.model} + {lens.maker} {lens.model}", grid_w, grid_h
        except Exception as e:
            print(f"Lens correction generation failed: {e}")
            return None

    def _apply_lens_to_map_texture(self, raw_tex, out_tex, w: int, h: int):
        if not raw_tex or not out_tex or not hasattr(self, 'lens_program') or not hasattr(self, 'lens_vao'):
            return
        try:
            import numpy as np
            identity = np.eye(4, dtype='f4').tobytes()
            if 'mvp' in self.lens_program: self.lens_program['mvp'].write(identity)
            if 'mvp_base' in self.lens_program: self.lens_program['mvp_base'].write(identity)
            
            fbo = self.ctx.framebuffer(color_attachments=[out_tex])
            fbo.viewport = (0, 0, w, h)
            fbo.use()
            
            raw_tex.use(0)
            
            if self.enable_lens_correction and self.has_lens_profile and self.uv_map_texture:
                self.lens_program['apply_correction'].value = 2 # Pure geometry mapping without TCA
                self.uv_map_texture.use(1)
            else:
                self.lens_program['apply_correction'].value = 0
                
            self.lens_vao.render(moderngl.TRIANGLE_STRIP)
            fbo.release()
        except Exception as e:
            print(f"Error applying lens correction to map texture: {e}")

    def apply_lens_correction(self):
        if not hasattr(self, 'raw_texture') or not self.raw_texture or self.texture is None:
            return
            
        # Always guarantee identity matrix is bound to full-screen lens quad
        import numpy as np
        identity = np.eye(4, dtype='f4').tobytes()
        if 'mvp' in self.lens_program: self.lens_program['mvp'].write(identity)
        if 'mvp_base' in self.lens_program: self.lens_program['mvp_base'].write(identity)
        
        # 1. Undistort RGB working photo (Geometry + TCA)
        fbo = self.ctx.framebuffer(color_attachments=[self.texture])
        fbo.viewport = (0, 0, self.width, self.height)
        fbo.use()
        
        self.raw_texture.use(0)
        
        if self.enable_lens_correction and self.has_lens_profile and self.uv_map_texture:
            self.lens_program['apply_correction'].value = 1
            self.uv_map_texture.use(1)
        else:
            self.lens_program['apply_correction'].value = 0
            
        if hasattr(self, 'lens_vao'):
            self.lens_vao.render(moderngl.TRIANGLE_STRIP)
            
        fbo.release()

        self.texture.build_mipmaps()
        self.texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        # 2. Undistort Depth Map (Geometry only)
        if getattr(self, 'raw_depth_tex', None) is not None and getattr(self, 'global_depth_tex', None) is not None:
            self._apply_lens_to_map_texture(self.raw_depth_tex, self.global_depth_tex, self.raw_depth_tex.width, self.raw_depth_tex.height)

        # 3. Undistort Normal Map (Geometry only)
        if getattr(self, 'raw_normal_tex', None) is not None and getattr(self, 'global_normal_tex', None) is not None:
            self._apply_lens_to_map_texture(self.raw_normal_tex, self.global_normal_tex, self.raw_normal_tex.width, self.raw_normal_tex.height)

        # 4. Undistort Subject Mask (Geometry only)
        if getattr(self, 'raw_subject_mask_tex', None) is not None and getattr(self, 'global_subject_mask_tex', None) is not None:
            self._apply_lens_to_map_texture(self.raw_subject_mask_tex, self.global_subject_mask_tex, self.raw_subject_mask_tex.width, self.raw_subject_mask_tex.height)

    def get_uv_from_viewport(self, px: float, py: float, pw: float, ph: float, to_preview=True):
        import numpy as np
        if getattr(self, 'split_view_mode', None) == 'side_by_side' and to_preview:
            half_w = pw / 2.0
            px = px - half_w if px >= half_w else px
            pw = half_w
            
        # Convert px, py (top-left origin in Qt) to OpenGL viewport coords [-1, 1]
        nx = (px / pw) * 2.0 - 1.0
        ny = 1.0 - (py / ph) * 2.0
        
        MVP = self.MVP_preview if to_preview else self.MVP_export
        try:
            inv_MVP = np.linalg.inv(MVP)
        except (np.linalg.LinAlgError, Exception):
            return 0.5, 0.5
            
        vec = np.array([nx, ny, 0.0, 1.0], dtype=np.float32)
        res = inv_MVP @ vec
        
        u = (res[0] / res[3]) * 0.5 + 0.5
        v = (res[1] / res[3]) * 0.5 + 0.5
        return u, v

    def set_geometry(self, rotation_deg: float, flip_90: int, crop_l: float, crop_r: float, crop_t: float, crop_b: float):
        self.rotation_deg = rotation_deg
        self.flip_90 = flip_90
        self.crop_l = crop_l
        self.crop_r = crop_r
        self.crop_t = crop_t
        self.crop_b = crop_b
        self._update_mvp()

    def set_crop(self, crop_l: float, crop_r: float, crop_t: float, crop_b: float):
        self.crop_l = float(crop_l)
        self.crop_r = float(crop_r)
        self.crop_t = float(crop_t)
        self.crop_b = float(crop_b)
        self.frame_count = 0
        self._update_mvp()

    def rotate_left_90(self):
        self.flip_90 = (self.flip_90 - 1) % 4
        self.frame_count = 0
        self._update_mvp()

    def rotate_right_90(self):
        self.flip_90 = (self.flip_90 + 1) % 4
        self.frame_count = 0
        self._update_mvp()

    def set_view(self, zoom: float, pan_x: float, pan_y: float):
        if self.view_zoom != zoom or self.view_pan_x != pan_x or self.view_pan_y != pan_y:
            self.frame_count = 0
            
        self.view_zoom = zoom
        self.view_pan_x = pan_x
        self.view_pan_y = pan_y
        self._update_mvp()
        
    def set_preview_size(self, w: int, h: int):
        w = max(1, w)
        h = max(1, h)
        scale = min(4096.0 / w, 4096.0 / h)
        if scale < 1.0:
            w = int(w * scale)
            h = int(h * scale)
            
        if self.prev_w != w or self.prev_h != h:
            self.frame_count = 0
            
        self.prev_w = w
        self.prev_h = h
        self._update_mvp()

    def set_crop_interactive_mode(self, enabled: bool):
        if getattr(self, 'is_crop_interactive_mode', False) != enabled:
            self.is_crop_interactive_mode = enabled
            self.frame_count = 0
            self._update_mvp()

    def _update_mvp(self):
        if 'mvp' not in self.program or self.texture is None:
            return
            
        W, H = self.width, self.height
        if W <= 0 or H <= 0:
            return

        if getattr(self, 'is_passthrough', False):
            # In passthrough mode (e.g. displaying rendered _preview.jpg or camera JPEG preview),
            # the image has already been cropped/rendered or is displayed 1:1.
            # We strictly preserve the texture's native aspect ratio with zero distortion!
            self.out_w = max(1, W)
            self.out_h = max(1, H)
            
            S_img = np.array([
                [W/2, 0, 0, 0],
                [0, H/2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            S_crop = np.array([
                [2.0/self.out_w, 0, 0, 0],
                [0, 2.0/self.out_h, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            MVP_base = S_crop @ S_img
            self.MVP_export = MVP_base
            MVP_preview_base = MVP_base
            eff_out_w, eff_out_h = self.out_w, self.out_h
        else:
            # Standard RAW / Layered edit pipeline with interactive crop & rotation
            # For OpenGL where Y is up, but our texture is mapped Top=y=-1
            # Positive angle_rad rotates CCW visually
            angle_rad = np.radians(self.rotation_deg - 90 * self.flip_90)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            
            W_bound = W * abs(c) + H * abs(s)
            H_bound = W * abs(s) + H * abs(c)
            
            # Compute maximum inscribed rectangle to auto-crop empty black corners upon Straighten / Rotation
            r = (W / float(H)) if H > 0 else 1.0
            denom1 = r * abs(c) + abs(s)
            denom2 = abs(c) + r * abs(s)
            if denom1 > 1e-7 and denom2 > 1e-7:
                h_ins = min(W / denom1, H / denom2)
                w_ins = r * h_ins
            else:
                w_ins, h_ins = W, H
                
            auto_margin_x = max(0.0, min(0.499, (W_bound - w_ins) / (2.0 * W_bound))) if W_bound > 0 else 0.0
            auto_margin_y = max(0.0, min(0.499, (H_bound - h_ins) / (2.0 * H_bound))) if H_bound > 0 else 0.0
            
            eff_crop_l = auto_margin_x + self.crop_l * (1.0 - 2.0 * auto_margin_x)
            eff_crop_r = auto_margin_x + self.crop_r * (1.0 - 2.0 * auto_margin_x)
            eff_crop_t = auto_margin_y + self.crop_t * (1.0 - 2.0 * auto_margin_y)
            eff_crop_b = auto_margin_y + self.crop_b * (1.0 - 2.0 * auto_margin_y)
            
            self.out_w = int(W_bound * (1.0 - eff_crop_l - eff_crop_r))
            self.out_h = int(H_bound * (1.0 - eff_crop_t - eff_crop_b))
            
            # Prevent zero or negative size
            self.out_w = max(1, self.out_w)
            self.out_h = max(1, self.out_h)
            
            cx = (W_bound / 2.0) * (eff_crop_l - eff_crop_r)
            cy = (H_bound / 2.0) * (eff_crop_t - eff_crop_b)
            
            S_img = np.array([
                [W/2, 0, 0, 0],
                [0, H/2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            R = np.array([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1]
            ], dtype='f4')
            
            T_crop = np.array([
                [1, 0, 0, -cx],
                [0, 1, 0, -cy],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            S_crop = np.array([
                [2.0/self.out_w, 0, 0, 0],
                [0, 2.0/self.out_h, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
            
            # Base matrix for cropping and rotating the image
            MVP_base = S_crop @ T_crop @ R @ S_img
            self.MVP_export = MVP_base
            
            # Determine whether preview displays the full rotated frame (for interactive crop editing)
            if getattr(self, 'is_crop_interactive_mode', False):
                out_w_prev = max(1, int(W_bound))
                out_h_prev = max(1, int(H_bound))
                T_crop_prev = np.eye(4, dtype='f4')
                S_crop_prev = np.array([
                    [2.0/out_w_prev, 0, 0, 0],
                    [0, 2.0/out_h_prev, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype='f4')
                MVP_preview_base = S_crop_prev @ T_crop_prev @ R @ S_img
                eff_out_w, eff_out_h = out_w_prev, out_h_prev
            else:
                MVP_preview_base = MVP_base
                eff_out_w, eff_out_h = self.out_w, self.out_h

        # For preview, fit the image into effective preview viewport
        eff_prev_w = (self.prev_w // 2) if getattr(self, 'split_view_mode', None) == 'side_by_side' else self.prev_w
        if eff_prev_w > 0 and self.prev_h > 0:
            viewport_aspect = eff_prev_w / float(self.prev_h)
            image_aspect = eff_out_w / float(eff_out_h)
            
            if image_aspect > viewport_aspect:
                scale_x = 1.0
                scale_y = viewport_aspect / image_aspect
            else:
                scale_x = image_aspect / viewport_aspect
                scale_y = 1.0
                
            S_fit = np.array([
                [scale_x, 0, 0, 0],
                [0, scale_y, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype='f4')
        else:
            S_fit = np.eye(4, dtype='f4')
            
        S_view = np.array([
            [self.view_zoom, 0, 0, 0],
            [0, self.view_zoom, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype='f4')
        
        T_view = np.array([
            [1, 0, 0, self.view_pan_x],
            [0, 1, 0, self.view_pan_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype='f4')
        
        # Preview MVP incorporates fit, zoom, and pan
        self.MVP_preview = T_view @ S_view @ S_fit @ MVP_preview_base

    def set_global_adjustments(self, exposure=1.0, contrast=1.0, gamma=1.0, temperature=0.0, global_chroma=1.0, global_vibrance=0.0, local_contrast=0.0, highlight_compression=0.0, black_level=0.0, tint=0.0, shadows_boost=0.0, highlights_boost=0.0, white_point=0.0, ai_blend=1.0):
        self.exposure = exposure
        self.contrast = contrast
        self.gamma = gamma
        self.temperature = temperature
        self.global_chroma = global_chroma
        self.global_vibrance = global_vibrance
        self.local_contrast = local_contrast
        self.highlight_comp = highlight_compression
        self.black_level = black_level
        self.tint = tint
        self.shadows_boost = shadows_boost
        self.highlights_boost = highlights_boost
        self.white_point = white_point
        self.ai_blend = ai_blend
        
    def set_lens_detail_adjustments(self, vignette_amount=0.0, sharpen_amount=0.0):
        self.vignette_amount = vignette_amount
        self.sharpen_amount = sharpen_amount
            
    def recompile_layer_shader(self, plugin_manager=None):
        """Dynamically recompiles the layer fragment shader from active plugins."""
        if plugin_manager is not None:
            self.plugin_manager = plugin_manager
            
        if not hasattr(self, 'plugin_manager') or self.plugin_manager is None:
            from plugins.manager import PluginManager
            self.plugin_manager = PluginManager()
            
        fragment_shader_src = self.plugin_manager.build_fragment_shader()
        try:
            new_program = self.ctx.program(
                vertex_shader=self.vertex_shader_src,
                fragment_shader=fragment_shader_src
            )
            self.program = new_program
            if hasattr(self, 'vbo'):
                self.vao = self.ctx.vertex_array(
                    self.program,
                    [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
                )
            if 'image_texture' in self.program: self.program['image_texture'].value = 0
            if 'color_equalizer_lut' in self.program: self.program['color_equalizer_lut'].value = 1
            if 'tone_equalizer_lut' in self.program: self.program['tone_equalizer_lut'].value = 2
            if 'shadow_color_equalizer_lut' in self.program: self.program['shadow_color_equalizer_lut'].value = 3
            print("Successfully recompiled dynamic layer fragment shader on GPU!")
            return True
        except Exception as e:
            print("Failed to recompile layer fragment shader:", e)
            return False

    def _apply_layer_settings(self, layer: Layer):
        # Apply layer settings to uniform variables
        s = layer.settings or {}
        
        # If plugin_manager is active, delegate uniform updates to active plugins
        if getattr(self, 'plugin_manager', None):
            for plugin in self.plugin_manager.get_active_plugins():
                p_settings = dict(s)
                if plugin.plugin_id in s and isinstance(s[plugin.plugin_id], dict):
                    p_settings.update(s[plugin.plugin_id])
                # Flat keys in s override subdict keys
                for k, v in s.items():
                    if k != plugin.plugin_id:
                        p_settings[k] = v
                plugin.apply_uniforms(self.program, p_settings, self)
        
        # Color equalizers LUT textures
        if 'tone_equalizer' in s and isinstance(s['tone_equalizer'], np.ndarray):
            self.update_tone_equalizer_lut(s['tone_equalizer'])
        if 'color_equalizer' in s and isinstance(s['color_equalizer'], np.ndarray):
            self.update_color_equalizer_lut(s['color_equalizer'])
        if 'shadow_color_equalizer' in s and isinstance(s['shadow_color_equalizer'], np.ndarray):
            self.update_shadow_color_equalizer_lut(s['shadow_color_equalizer'])

    def _apply_neutral_settings(self):
        """Sets neutral default adjustments on the base render program."""
        if 'exposure' in self.program: self.program['exposure'].value = 2.22
        if 'contrast' in self.program: self.program['contrast'].value = 1.0
        if 'gamma' in self.program: self.program['gamma'].value = 1.0
        if 'temperature' in self.program: self.program['temperature'].value = 0.0
        if 'tint' in self.program: self.program['tint'].value = 0.0
        if 'global_chroma' in self.program: self.program['global_chroma'].value = 1.0
        if 'global_vibrance' in self.program: self.program['global_vibrance'].value = 0.0
        if 'local_contrast' in self.program: self.program['local_contrast'].value = 0.0
        if 'white_point' in self.program: self.program['white_point'].value = 0.0
        if 'shadows_boost' in self.program: self.program['shadows_boost'].value = 0.0
        if 'highlights_boost' in self.program: self.program['highlights_boost'].value = 0.0
        if 'vignette_amount' in self.program: self.program['vignette_amount'].value = 0.0
        if 'denoise_luma' in self.program: self.program['denoise_luma'].value = 0.0
        if 'denoise_chroma' in self.program: self.program['denoise_chroma'].value = 0.0
        if 'sharpen_amount' in self.program: self.program['sharpen_amount'].value = 0.0
        if 'sharpen_threshold' in self.program: self.program['sharpen_threshold'].value = 0.0
        if 'black_level' in self.program: self.program['black_level'].value = 0.0
        if 'highlight_compression' in self.program: self.program['highlight_compression'].value = 0.0
        if 'highlight_comp' in self.program: self.program['highlight_comp'].value = 0.0
        if 'base_curve_strength' in self.program: self.program['base_curve_strength'].value = float(self.base_curve_strength)
        if 'ai_blend' in self.program: self.program['ai_blend'].value = 1.0

    def _render_before_pass(self, target_fbo, viewport, mvp):
        """Renders the neutral base RAW/image into target_fbo for before/after comparison."""
        target_fbo.viewport = viewport
        target_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        self.program['mvp'].write(mvp.T.tobytes())
        if 'mvp_base' in self.program: self.program['mvp_base'].write(self.MVP_export.T.tobytes())
        
        self._apply_neutral_settings()
        
        # When comparing, use the original pre-denoise texture if available so Before shows the true original photo with noise
        tex = self.source_texture if (getattr(self, 'has_ai_denoised', False) and getattr(self, 'source_texture', None) is not None) else self.texture
        
        tex.use(0)
        self.neutral_ceq_texture.use(1)
        self.neutral_tone_texture.use(2)
        self.neutral_ceq_texture.use(3)
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def _run_display_pass(self, source_fbo, viewport):
        """Runs the fast GPU presentation shader into preview_display_fbo."""
        if hasattr(self, 'display_prog') and getattr(self, 'preview_display_fbo', None) is not None:
            self.preview_display_fbo.viewport = viewport
            self.preview_display_fbo.use()
            source_fbo.color_attachments[0].use(0)
            self.display_prog['accum_tex'].value = 0
            self.display_prog['u_resolution'].value = (float(viewport[2]), float(viewport[3]))
            self.display_prog['u_tex_size'].value = (float(source_fbo.color_attachments[0].width), float(source_fbo.color_attachments[0].height))
            self.display_vao.render(moderngl.TRIANGLE_STRIP)

    def render(self, to_preview=True, progress_callback=None):
        if self.texture is None:
            return
            
        with profiler.scope("gpu_render_ms"):
            if not to_preview:
                self._ensure_export_fbos()
                
            if 'ai_blend' in self.program:
                self.program['ai_blend'].value = self.ai_blend

            is_side_by_side = (getattr(self, 'split_view_mode', None) == 'side_by_side')
            eff_prev_w = (self.prev_w // 2) if (to_preview and is_side_by_side) else self.prev_w
            
            fbo = self.preview_fbo if to_preview else self.fbo
            viewport = (0, 0, eff_prev_w, self.prev_h) if to_preview else (0, 0, self.out_w, self.out_h)
            final_viewport = (0, 0, self.prev_w, self.prev_h) if to_preview else (0, 0, self.out_w, self.out_h)
            mvp = self.MVP_preview if to_preview else self.MVP_export
            
            # Accumulation FBOs for multi-pass / split
            accum_1 = self.preview_fbo_1 if to_preview else self.export_fbo_1
            accum_2 = self.preview_fbo_2 if to_preview else self.export_fbo_2
            
            self.program['mvp'].write(mvp.T.tobytes())
            if 'mvp_base' in self.program: self.program['mvp_base'].write(self.MVP_export.T.tobytes())

            # Bind source textures
            self.texture.use(0)
            self.ceq_texture.use(1)
            self.tone_eq_texture.use(2)
            self.shadow_ceq_texture.use(3)
            
            current_accum = accum_1
            
            if getattr(self, 'is_passthrough', False):
                # Direct 1:1 neutral passthrough rendering for in-camera JPEG preview
                fbo.viewport = final_viewport
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                if 'exposure' in self.program: self.program['exposure'].value = 1.0
                if 'contrast' in self.program: self.program['contrast'].value = 1.0
                if 'gamma' in self.program: self.program['gamma'].value = 1.0
                if 'temperature' in self.program: self.program['temperature'].value = 0.0
                if 'tint' in self.program: self.program['tint'].value = 0.0
                if 'global_chroma' in self.program: self.program['global_chroma'].value = 1.0
                if 'global_vibrance' in self.program: self.program['global_vibrance'].value = 0.0
                if 'local_contrast' in self.program: self.program['local_contrast'].value = 0.0
                if 'white_point' in self.program: self.program['white_point'].value = 0.0
                if 'shadows_boost' in self.program: self.program['shadows_boost'].value = 0.0
                if 'highlights_boost' in self.program: self.program['highlights_boost'].value = 0.0
                if 'vignette_amount' in self.program: self.program['vignette_amount'].value = 0.0
                if 'denoise_luma' in self.program: self.program['denoise_luma'].value = 0.0
                if 'denoise_chroma' in self.program: self.program['denoise_chroma'].value = 0.0
                if 'sharpen_amount' in self.program: self.program['sharpen_amount'].value = 0.0
                if 'sharpen_threshold' in self.program: self.program['sharpen_threshold'].value = 0.0
                if 'black_level' in self.program: self.program['black_level'].value = 0.0
                if 'highlight_compression' in self.program: self.program['highlight_compression'].value = 0.0
                if 'highlight_comp' in self.program: self.program['highlight_comp'].value = 0.0
                if 'base_curve_strength' in self.program: self.program['base_curve_strength'].value = 0.0
                if 'ai_blend' in self.program: self.program['ai_blend'].value = 1.0
                
                self.program['mvp'].write(mvp.T.tobytes())
                if 'mvp_base' in self.program: self.program['mvp_base'].write(self.MVP_export.T.tobytes())
                
                self.texture.use(0)
                self.neutral_ceq_texture.use(1)
                self.neutral_tone_texture.use(2)
                self.neutral_ceq_texture.use(3)
                self.vao.render(moderngl.TRIANGLE_STRIP)
                if to_preview:
                    self._run_display_pass(fbo, final_viewport)
                return

            preview_mode = getattr(self, 'preview_mode', None)
            is_preview_mask = getattr(self, 'preview_mask', False)
            if preview_mode == 'mask':
                is_preview_mask = True

            # Fast path for global map previews
            if preview_mode in ['depth', 'normals', 'subject']:
                current_accum.viewport = final_viewport
                current_accum.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                if hasattr(self, 'preview_map_prog'):
                    prog = self.preview_map_prog
                    if 'mvp' in prog: prog['mvp'].write(self.MVP_preview.T.tobytes())
                    
                    tex_to_use = None
                    map_type = 0
                    if preview_mode == 'depth':
                        tex_to_use = self.global_depth_tex
                    elif preview_mode == 'subject':
                        tex_to_use = self.global_subject_mask_tex
                    elif preview_mode == 'normals':
                        tex_to_use = self.global_normal_tex
                        map_type = 1
                    
                    has_tex = 1 if tex_to_use is not None else 0
                    if tex_to_use is not None:
                        tex_to_use.use(0)
                    else:
                        self.dummy_tex_1d.use(0)
                        
                    if 'tex' in prog: prog['tex'].value = 0
                    if 'map_type' in prog: prog['map_type'].value = map_type
                    if 'has_tex' in prog: prog['has_tex'].value = has_tex
                    
                    self.preview_map_vao.render(moderngl.TRIANGLE_STRIP)
                
                fbo.viewport = final_viewport
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.ctx.copy_framebuffer(fbo, current_accum)
                if to_preview:
                    self._run_display_pass(fbo, final_viewport)
                return

            if 'show_mask' in self.mask_blend_program:
                self.mask_blend_program['show_mask'].value = 0

            skip_layers = (to_preview and 
                           getattr(self, 'bokeh_enabled', False) and 
                           self.frame_count > 0 and 
                           not is_preview_mask and 
                           getattr(self, '_last_layer_accum', None) is not None)

            if not self.layers:
                self.layers = [Layer(id="layer_base", name="Base Layer", is_base=True, active=True)]
                self.active_layer_index = 0

            if not skip_layers:
                for i, layer in enumerate(self.layers):
                    if not layer.active:
                        continue
                        
                    self._apply_layer_settings(layer)
                    
                    if layer.is_base:
                        # Render Base Layer directly to accum_1
                        current_accum.viewport = viewport
                        current_accum.use()
                        if is_preview_mask and self.active_layer_index == i:
                            self.ctx.clear(1.0, 1.0, 1.0, 1.0) # Base layer mask is white
                            break
                        else:
                            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                            self.vao.render(moderngl.TRIANGLE_STRIP)
                    else:
                        # Rebind source textures since previous iteration's blend step changed them
                        self.texture.use(0)
                        self.ceq_texture.use(1)
                        self.tone_eq_texture.use(2)
                        self.shadow_ceq_texture.use(3)
        
                        # 1. Render Local Layer to layer_fbo
                        layer_fbo = self.preview_layer_fbo if to_preview else self.export_layer_fbo
                        layer_tex = self.preview_layer_tex if to_preview else self.export_layer_tex
                        layer_fbo.viewport = viewport
                        layer_fbo.use()
                        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                        self.vao.render(moderngl.TRIANGLE_STRIP)
                        
                        # 2. Blend layer_fbo and current_accum into the other accum FBO
                        next_accum = accum_2 if current_accum == accum_1 else accum_1
                        next_accum.viewport = viewport
                        next_accum.use()
                        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                        # Setup mask_blend shader
                        self.mask_blend_program['mask_color_target'].value = (0.0, layer.mask_sat_target, layer.mask_hue_target * 3.14159265359 * 2.0)
                        self.mask_blend_program['mask_color_range'].value = layer.mask_hue_range
                        self.mask_blend_program['mask_sat_range'].value = layer.mask_sat_range
                        self.mask_blend_program['mask_luma_target'].value = layer.mask_luma_target
                        self.mask_blend_program['mask_luma_range'].value = layer.mask_luma_range
                        self.mask_blend_program['mask_softness'].value = layer.mask_softness
                        self.mask_blend_program['mask_opacity'].value = layer.mask_opacity
                        
                        if 'fill_mask' in self.mask_blend_program:
                            self.mask_blend_program['fill_mask'].value = layer.mask_fill
                            
                        if 'gradient_type' in self.mask_blend_program:
                            self.mask_blend_program['gradient_type'].value = layer.mask_gradient_type
                            self.mask_blend_program['grad_black'].value = layer.mask_grad_black
                            self.mask_blend_program['grad_white'].value = layer.mask_grad_white
                            self.mask_blend_program['grad_gamma'].value = layer.mask_grad_gamma
                        
                        # 3D Mask Uniforms
                        if 'use_depth_mask' in self.mask_blend_program:
                            self.mask_blend_program['use_depth_mask'].value = 1.0 if layer.use_depth_mask else 0.0                
                            if 'depth_min' in self.mask_blend_program:
                                self.mask_blend_program['depth_min'].value = layer.mask_depth_min
                                self.mask_blend_program['depth_max'].value = layer.mask_depth_max
                                self.mask_blend_program['depth_softness'].value = layer.mask_depth_softness
                                if 'depth_gamma' in self.mask_blend_program:
                                    self.mask_blend_program['depth_gamma'].value = 1.0 / max(0.01, layer.mask_depth_gamma)
                        
                        self.mask_blend_program['use_normal_mask'].value = layer.use_normal_mask
                        if 'normal_dir' in self.mask_blend_program:
                            self.mask_blend_program['normal_dir'].value = layer.mask_normal_dir
                            self.mask_normal_angle = layer.mask_normal_angle
                            self.mask_blend_program['normal_angle'].value = layer.mask_normal_angle
                            self.mask_blend_program['normal_softness'].value = layer.mask_normal_softness
                            if 'normal_gamma' in self.mask_blend_program:
                                self.mask_blend_program['normal_gamma'].value = 1.0 / max(0.01, layer.mask_normal_gamma)
                        
                        if 'inv_mvp' in self.mask_blend_program:
                            try:
                                inv = np.linalg.inv(mvp)
                                self.mask_blend_program['inv_mvp'].write(inv.T.tobytes())
                            except np.linalg.LinAlgError:
                                pass
                        
                        # Bind textures for blending
                        current_accum.color_attachments[0].use(0)
                        layer_tex.use(1)
                        self.texture.use(2) # original image
                        
                        mask_tex = self.mask_textures.get(layer.id)
                        res_div = getattr(layer, 'mask_resolution', 4) or 4
                        w_exp = max(1, self.width // res_div)
                        h_exp = max(1, self.height // res_div)
                        if mask_tex and mask_tex.width == w_exp and mask_tex.height == h_exp:
                            mask_tex.use(3)
                        else:
                            self.dummy_tex_1d.use(3) # safe dummy
                            
                        # Bind 3D Textures
                        if self.global_depth_tex:
                            self.global_depth_tex.use(4)
                        else:
                            self.dummy_tex_1d.use(4)
                            
                        if self.global_normal_tex:
                            self.global_normal_tex.use(5)
                        else:
                            self.dummy_tex_3d.use(5)
                        
                        if is_preview_mask and self.active_layer_index == i:
                            if 'show_mask' in self.mask_blend_program:
                                self.mask_blend_program['show_mask'].value = 1.0
                                
                        self.mask_blend_vao.render(moderngl.TRIANGLE_STRIP)
                        
                        # Swap accumulation buffers
                        current_accum = next_accum
                    
                    if is_preview_mask and hasattr(self, 'active_layer_index') and self.active_layer_index == i:
                        if 'show_mask' in self.mask_blend_program:
                            self.mask_blend_program['show_mask'].value = 0.0
                        break
                    
                self._last_layer_accum = current_accum
            else:
                cached_accum = getattr(self, '_last_layer_accum', None)
                if (cached_accum is not None and 
                    hasattr(cached_accum, 'color_attachments') and 
                    cached_accum.color_attachments and 
                    cached_accum.color_attachments[0] is not None):
                    current_accum = cached_accum
                else:
                    current_accum = accum_1
                
            # Apply Bokeh if enabled
            has_depth = (getattr(self, 'global_depth_tex', None) is not None)
            has_bokeh = (getattr(self, 'bokeh_enabled', False) and 
                         hasattr(self, 'bokeh_raymarch_prog') and 
                         not is_preview_mask and 
                         has_depth)
                         
            if has_bokeh:
                target_valid = (self.preview_fbo_bokeh is not None) if to_preview else (getattr(self, 'export_fbo_bokeh', None) is not None)
                if (current_accum is None or 
                    not hasattr(current_accum, 'color_attachments') or 
                    not current_accum.color_attachments or 
                    current_accum.color_attachments[0] is None or
                    not target_valid):
                    has_bokeh = False

            if has_bokeh:
                bokeh_target_fbo = self.preview_fbo_bokeh if to_preview else self.export_fbo_bokeh
                prog = self.bokeh_raymarch_prog
                
                # Setup Uniforms
                if 'mvp' in prog: prog['mvp'].write(mvp.T.tobytes())
                if 'mvp_base' in prog: prog['mvp_base'].write(self.MVP_export.T.tobytes())
                if 'inv_mvp' in prog:
                    try:
                        inv = np.linalg.inv(mvp)
                        prog['inv_mvp'].write(inv.T.tobytes())
                    except np.linalg.LinAlgError:
                        pass
                
                if 'u_focalDepth' in prog: prog['u_focalDepth'].value = getattr(self, 'bokeh_focal_depth', 0.5)
                if 'u_focalRange' in prog: prog['u_focalRange'].value = getattr(self, 'bokeh_focal_range', 0.1)
                if 'u_fgGamma' in prog: prog['u_fgGamma'].value = getattr(self, 'bokeh_fg_gamma', 1.0)
                if 'u_bgGamma' in prog: prog['u_bgGamma'].value = getattr(self, 'bokeh_bg_gamma', 1.0)
                
                if 'u_fboBounds' in prog:
                    if to_preview:
                        image_aspect = self.out_w / float(self.out_h)
                        viewport_aspect = viewport[2] / float(max(1, viewport[3]))
                        if image_aspect > viewport_aspect:
                            scale_x = 1.0
                            scale_y = viewport_aspect / image_aspect
                        else:
                            scale_x = image_aspect / viewport_aspect
                            scale_y = 1.0
                            
                        scale_x *= self.view_zoom
                        scale_y *= self.view_zoom
                        pan_x = self.view_pan_x
                        pan_y = self.view_pan_y
                        
                        min_ndc_x = -scale_x + pan_x
                        max_ndc_x =  scale_x + pan_x
                        min_ndc_y = -scale_y + pan_y
                        max_ndc_y =  scale_y + pan_y
                        
                        # Add a 1-pixel margin to prevent the bilinear interpolation from sampling the black border
                        margin_x = 1.0 / max(1, viewport[2])
                        margin_y = 1.0 / max(1, viewport[3])
                            
                        valid_min_x = float(np.clip(min_ndc_x * 0.5 + 0.5 + margin_x, 0.0, 1.0))
                        valid_min_y = float(np.clip(min_ndc_y * 0.5 + 0.5 + margin_y, 0.0, 1.0))
                        valid_max_x = float(np.clip(max_ndc_x * 0.5 + 0.5 - margin_x, 0.0, 1.0))
                        valid_max_y = float(np.clip(max_ndc_y * 0.5 + 0.5 - margin_y, 0.0, 1.0))
                    else:
                        valid_min_x, valid_min_y = 0.0, 0.0
                        valid_max_x, valid_max_y = 1.0, 1.0
                        
                    prog['u_fboBounds'].value = (valid_min_x, valid_min_y, valid_max_x, valid_max_y)
                
                accum_tex = current_accum.color_attachments[0]
                src_scale = (float(viewport[2]) / float(max(1, accum_tex.size[0])), 
                             float(viewport[3]) / float(max(1, accum_tex.size[1])))
                if 'u_srcScale' in prog: prog['u_srcScale'].value = src_scale
                
                has_mask = 1.0 if self.global_subject_mask_tex is not None else 0.0
                if 'u_hasSubjectMask' in prog: prog['u_hasSubjectMask'].value = has_mask
                if 'u_maskInfluence' in prog: prog['u_maskInfluence'].value = getattr(self, 'bokeh_subject_influence', 1.0)
                if 'u_apertureSides' in prog: prog['u_apertureSides'].value = float(getattr(self, 'bokeh_petals', 0.0))
                if 'u_apertureRotation' in prog: prog['u_apertureRotation'].value = float(getattr(self, 'bokeh_rotation', 0.0))
                if 'u_highlightBoost' in prog: prog['u_highlightBoost'].value = getattr(self, 'bokeh_highlight_boost', 0.0)
                if 'u_chromaticAberration' in prog: prog['u_chromaticAberration'].value = getattr(self, 'bokeh_chromatic', 0.0)
                if 'u_radialCenter' in prog: prog['u_radialCenter'].value = getattr(self, 'bokeh_radial_center', 0.0)
                if 'u_radialEdge' in prog: prog['u_radialEdge'].value = getattr(self, 'bokeh_radial_edge', 0.0)
                if 'u_radialGamma' in prog: prog['u_radialGamma'].value = getattr(self, 'bokeh_radial_gamma', 1.0)
                if 'u_vignette' in prog: prog['u_vignette'].value = getattr(self, 'bokeh_vignette', 0.0)
                if 'u_resolution' in prog: prog['u_resolution'].value = (viewport[2], viewport[3])
                
                max_samples = getattr(self, 'bokeh_max_samples', 256)
                if 'u_maxSamples' in prog: prog['u_maxSamples'].value = max_samples
                
                # Calculate max CoC (shift in pixels)
                if to_preview:
                    image_aspect = self.out_w / float(self.out_h)
                    viewport_aspect = viewport[2] / float(max(1, viewport[3]))
                    if image_aspect > viewport_aspect:
                        scale_x = 1.0
                        scale_y = viewport_aspect / image_aspect
                    else:
                        scale_x = image_aspect / viewport_aspect
                        scale_y = 1.0
                    ds_img_w = scale_x * self.view_zoom * viewport[2]
                    ds_img_h = scale_y * self.view_zoom * viewport[3]
                    max_ds_img_dim = max(ds_img_w, ds_img_h)
                else:
                    max_ds_img_dim = max(self.out_w, self.out_h)
                    
                radius_px = (self.bokeh_radius / 1000.0) * max_ds_img_dim
                if 'u_maxCoC' in prog: prog['u_maxCoC'].value = radius_px
                
                # Render bokeh into bokeh_target_fbo
                bokeh_target_fbo.viewport = viewport
                bokeh_target_fbo.use()
                
                # Bind Textures
                current_accum.color_attachments[0].use(0) # origTex
                if self.global_depth_tex:
                    self.global_depth_tex.use(1)
                else:
                    self.dummy_tex_1d.use(1) # fallback
                
                if self.global_subject_mask_tex:
                    self.global_subject_mask_tex.use(2)
                else:
                    self.dummy_tex_1d.use(2)

                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                
                if to_preview:
                    self.frame_count += 1
                    if 'u_frameCount' in prog: prog['u_frameCount'].value = self.frame_count
                    if self.frame_count <= 1:
                        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                    self.bokeh_raymarch_vao.render(moderngl.TRIANGLE_STRIP)
                else:
                    self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                    export_passes = min(512, max(16, max_samples))
                    step_flush = max(1, min(4, export_passes // 32))
                    
                    for pass_i in range(1, export_passes + 1):
                        if progress_callback and hasattr(progress_callback, 'is_cancelled') and progress_callback.is_cancelled():
                            break
                        if 'u_frameCount' in prog: prog['u_frameCount'].value = pass_i
                        self.bokeh_raymarch_vao.render(moderngl.TRIANGLE_STRIP)
                        
                        if progress_callback and (pass_i % step_flush == 0 or pass_i == export_passes):
                            try:
                                self.ctx.finish()
                            except Exception:
                                pass
                            pct = int(10 + (pass_i / export_passes) * 55)
                            status_msg = f"Rendering 3D Bokeh Pass {pass_i}/{export_passes} ({pct}%)..."
                            if hasattr(progress_callback, 'set_progress'):
                                progress_callback.set_progress(pct, status_msg)
                            elif callable(progress_callback):
                                progress_callback(pct, status_msg)
                        
                self.ctx.disable(moderngl.BLEND)
                
                after_tex = bokeh_target_fbo.color_attachments[0]
                after_fbo = bokeh_target_fbo
            else:
                if not to_preview and progress_callback:
                    if hasattr(progress_callback, 'set_progress'):
                        progress_callback.set_progress(70, "Finalizing GPU frame...")
                    elif callable(progress_callback):
                        progress_callback(70, "Finalizing GPU frame...")
                after_tex = current_accum.color_attachments[0] if (current_accum and current_accum.color_attachments) else None
                after_fbo = current_accum

            if after_tex is None or after_fbo is None:
                return

            # Presentation Pass: Split View / Before Mode / Normal
            # CRITICAL: Split views and Before mode are ONLY for viewport preview, NEVER for export!
            is_split = to_preview and getattr(self, 'split_view_mode', None) in ['vertical', 'horizontal', 'side_by_side'] and hasattr(self, 'split_prog')
            is_before = to_preview and getattr(self, 'split_view_mode', None) == 'before'
            
            if is_split:
                before_accum = accum_2 if current_accum == accum_1 else accum_1
                self._render_before_pass(before_accum, viewport, mvp)
                
                fbo.viewport = final_viewport
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                before_accum.color_attachments[0].use(0)
                after_tex.use(1)
                
                mode_map = {'vertical': 1, 'horizontal': 2, 'side_by_side': 3}
                mode_id = mode_map.get(self.split_view_mode, 1)
                self.split_prog['tex_before'].value = 0
                self.split_prog['tex_after'].value = 1
                self.split_prog['u_split_mode'].value = mode_id
                self.split_prog['u_split_pos'].value = float(getattr(self, 'split_position', 0.5))
                self.split_prog['u_resolution'].value = (float(final_viewport[2]), float(final_viewport[3]))
                self.split_prog['u_tex_size'].value = (float(accum_1.color_attachments[0].width), float(accum_1.color_attachments[0].height))
                self.split_vao.render(moderngl.TRIANGLE_STRIP)
            elif is_before:
                self._render_before_pass(fbo, final_viewport, mvp)
            else:
                # Normal output
                fbo.viewport = final_viewport
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                self.ctx.copy_framebuffer(fbo, after_fbo)

            if to_preview:
                self._run_display_pass(fbo, final_viewport)

    def get_or_create_mask(self, layer_id: str):
        layer = next((l for l in self.layers if l.id == layer_id), None)
        res = layer.mask_resolution if layer else 4
        
        w = max(1, self.width // res)
        h = max(1, self.height // res)
        
        if layer_id in self.mask_textures:
            tex = self.mask_textures[layer_id]
            if tex.width != w or tex.height != h or tex.dtype != 'f1':
                tex.release()
                del self.mask_textures[layer_id]
                
        if layer_id not in self.mask_textures:
            tex = self.ctx.texture((w, h), 1, dtype='f1')
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # Clear to 0
            fbo = self.ctx.framebuffer(color_attachments=[tex])
            fbo.clear(0.0)
            
            self.mask_textures[layer_id] = tex
        return self.mask_textures[layer_id]

    def clear_mask(self, layer_id: str):
        """Clears the layer mask to 0.0 (fully transparent/unmasked)."""
        tex = self.get_or_create_mask(layer_id)
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        fbo.clear(0.0)
        fbo.release()
        self.layers_dirty = True

    def clear_brush_mask(self):
        """Clears the active layer's mask texture."""
        if self.layers and hasattr(self, 'active_layer_index'):
            layer = self.layers[self.active_layer_index]
            self.clear_mask(layer.id)
            self.layers_dirty = True

    def paint_brush_mask(self, u: float, v: float, radius: float = 0.05, hardness: float = 0.5, opacity: float = 1.0, erase: bool = False):
        """
        Paints or erases a single brush stamp at normalized texture coordinate (u, v).
        """
        self.paint_brush_stroke(u, v, u, v, radius=radius, hardness=hardness, opacity=opacity, erase=erase)

    def paint_brush_stroke(self, u0: float, v0: float, u1: float, v1: float, radius: float = 0.05, hardness: float = 0.5, opacity: float = 1.0, erase: bool = False):
        """
        Paints a continuous interpolated stroke from (u0, v0) to (u1, v1) to prevent gaps on fast mouse movement.
        """
        if not self.layers or self.texture is None or not hasattr(self, 'brush_program'):
            return
        
        layer = self.layers[self.active_layer_index]
        tex = self.get_or_create_mask(layer.id)
        
        aspect = float(tex.width) / float(max(1, tex.height))
        du = (u1 - u0) * aspect
        dv = v1 - v0
        dist = float(np.hypot(du, dv))
        
        # 15% spacing between consecutive brush stamps for smooth uninterrupted strokes
        step_size = max(0.0005, radius * 0.15)
        num_steps = max(1, int(np.ceil(dist / step_size)))
        
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.viewport = (0, 0, tex.width, tex.height)
        fbo.use()
        
        self.ctx.enable(moderngl.BLEND)
        if erase:
            self.ctx.blend_func = moderngl.ZERO, moderngl.ONE_MINUS_SRC_ALPHA
        else:
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            
        prog = self.brush_program
        if 'brush_radius' in prog: prog['brush_radius'].value = radius
        if 'brush_hardness' in prog: prog['brush_hardness'].value = hardness
        if 'brush_opacity' in prog: prog['brush_opacity'].value = opacity
        if 'aspect_ratio' in prog: prog['aspect_ratio'].value = aspect
        
        for i in range(1, num_steps + 1):
            t = float(i) / float(num_steps)
            u = u0 + (u1 - u0) * t
            v = v0 + (v1 - v0) * t
            if 'brush_pos' in prog:
                prog['brush_pos'].value = (u, v)
            self.brush_vao.render(moderngl.TRIANGLE_STRIP)
            
        self.ctx.disable(moderngl.BLEND)
        fbo.release()
        self.layers_dirty = True

    def blur_brush_mask(self, u: float, v: float, radius: float = 0.05, hardness: float = 0.5):
        """Blurs the active layer mask under the brush cursor."""
        if not self.layers or self.texture is None or not hasattr(self, 'blur_brush_program'):
            return
        layer = self.layers[self.active_layer_index]
        tex = self.get_or_create_mask(layer.id)
        
        temp_tex = self.ctx.texture(tex.size, 1, dtype='f1')
        temp_fbo = self.ctx.framebuffer(color_attachments=[temp_tex])
        self.ctx.copy_framebuffer(temp_fbo, self.ctx.framebuffer(color_attachments=[tex]))
        
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.viewport = (0, 0, tex.width, tex.height)
        fbo.use()
        
        temp_tex.use(0)
        prog = self.blur_brush_program
        if 'mask_tex' in prog: prog['mask_tex'].value = 0
        if 'brush_pos' in prog: prog['brush_pos'].value = (u, v)
        if 'brush_radius' in prog: prog['brush_radius'].value = radius
        if 'brush_hardness' in prog: prog['brush_hardness'].value = hardness
        if 'aspect_ratio' in prog: prog['aspect_ratio'].value = float(tex.width) / float(max(1, tex.height))
        
        self.blur_brush_vao.render(moderngl.TRIANGLE_STRIP)
        fbo.release()
        temp_fbo.release()
        temp_tex.release()
        self.layers_dirty = True

    def get_layer_mask_numpy(self, layer_id: str) -> np.ndarray:
        """Returns 2D uint8 numpy array (H, W) for the layer's mask texture if present."""
        if not hasattr(self, 'mask_textures') or layer_id not in self.mask_textures:
            return None
        tex = self.mask_textures[layer_id]
        if tex is None:
            return None
        raw = tex.read()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((tex.height, tex.width))
        return arr

    def update_global_geometry(self, depth_np: np.ndarray, normal_np: np.ndarray):
        if depth_np is not None:
            import cv2
            # Erode depth map to fix AI dilation halos (shrink foreground, assuming 1.0/255 is near)
            k_size = max(3, int(min(depth_np.shape[:2]) * 0.0015))
            if k_size % 2 == 0: k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            depth_np = cv2.erode(depth_np, kernel, iterations=1)
            
            if self.global_depth_tex:
                self.global_depth_tex.release()
            h, w = depth_np.shape[:2]
            if depth_np.dtype == np.float32 or depth_np.dtype == np.float64:
                depth_f32 = depth_np.astype('f4').tobytes()
            elif depth_np.dtype == np.uint16:
                depth_f32 = (depth_np.astype('f4') / 65535.0).tobytes()
            else:
                depth_f32 = (depth_np.astype('f4') / 255.0).tobytes()
            self.global_depth_tex = self.ctx.texture((w, h), 1, depth_f32, dtype='f4')
            self.global_depth_tex.repeat_x = False
            self.global_depth_tex.repeat_y = False
            self.global_depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
        if normal_np is not None:
            if self.global_normal_tex:
                self.global_normal_tex.release()
            h, w = normal_np.shape[:2]
            if normal_np.dtype == np.float32 or normal_np.dtype == np.float64:
                normal_f32 = normal_np.astype('f4').tobytes()
            elif normal_np.dtype == np.uint16:
                normal_f32 = (normal_np.astype('f4') / 65535.0).tobytes()
            else:
                normal_f32 = (normal_np.astype('f4') / 255.0).tobytes()
            self.global_normal_tex = self.ctx.texture((w, h), 3, normal_f32, dtype='f4')
            self.global_normal_tex.repeat_x = False
            self.global_normal_tex.repeat_y = False
            self.global_normal_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

    def update_global_subject_mask(self, mask_np: np.ndarray):
        if mask_np is not None:
            import cv2
            # Erode subject mask to fix AI dilation halos
            k_size = max(3, int(min(mask_np.shape[:2]) * 0.0015))
            if k_size % 2 == 0: k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask_np = cv2.erode(mask_np, kernel, iterations=1)
            
            if self.global_subject_mask_tex:
                self.global_subject_mask_tex.release()
            h, w = mask_np.shape[:2]
            if mask_np.dtype == np.float32 or mask_np.dtype == np.float64:
                mask_f32 = mask_np.astype('f4').tobytes()
            elif mask_np.dtype == np.uint16:
                mask_f32 = (mask_np.astype('f4') / 65535.0).tobytes()
            else:
                mask_f32 = (mask_np.astype('f4') / 255.0).tobytes()
            self.global_subject_mask_tex = self.ctx.texture((w, h), 1, mask_f32, dtype='f4')
            self.global_subject_mask_tex.repeat_x = False
            self.global_subject_mask_tex.repeat_y = False
            self.global_subject_mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            if self.global_subject_mask_tex:
                self.global_subject_mask_tex.release()
            self.global_subject_mask_tex = None

    def update_layer_mask(self, layer_id: str, mask_array: np.ndarray):
        """Update mask texture with a 2D array (H, W)."""
        tex = self.get_or_create_mask(layer_id)
        if mask_array.shape[1] != tex.size[0] or mask_array.shape[0] != tex.size[1]:
            # Convert to float for resize to avoid PIL uint16 issues
            if mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                mask_f32 = mask_array.astype('f4')
            elif mask_array.dtype == np.uint16:
                mask_f32 = mask_array.astype('f4') / 65535.0
            else:
                mask_f32 = mask_array.astype('f4') / 255.0
                
            img = Image.fromarray(mask_f32).resize(tex.size, Image.Resampling.BILINEAR)
            mask_f32 = np.array(img, dtype='f4')
        else:
            if mask_array.dtype == np.float32 or mask_array.dtype == np.float64:
                mask_f32 = mask_array.astype('f4')
            elif mask_array.dtype == np.uint16:
                mask_f32 = (mask_array.astype('f4') / 65535.0)
            else:
                mask_f32 = (mask_array.astype('f4') / 255.0)
            
        mask_u8 = (np.clip(mask_f32, 0.0, 1.0) * 255.0).astype(np.uint8)
        tex.write(mask_u8.tobytes())

    def pick_color_at(self, u: float, v: float) -> tuple[float, float, float]:
        if not hasattr(self, 'raw_fbo') or not self.raw_fbo: return (0.0, 0.0, 0.0)
        # Convert u, v to pixel coordinates
        px = int(u * self.width)
        py = int(v * self.height)
        px = max(0, min(self.width - 1, px))
        py = max(0, min(self.height - 1, py))
        
        # Read a 1x1 area
        try:
            import struct
            data = self.raw_fbo.read(viewport=(px, py, 1, 1), components=4, dtype='f4')
            r, g, b, a = struct.unpack('4f', data)
            return (r, g, b)
        except Exception as e:
            print(f"Error picking color: {e}")
            return (0.0, 0.0, 0.0)

    def pick_depth_at(self, u: float, v: float, cached_depth_np=None, sample_radius: int = 4) -> float:
        """
        Samples the normalized depth value at normalized image coordinates (u, v) in range [0.0, 1.0].
        v=1.0 is top, v=0.0 is bottom.
        Takes a robust median neighborhood patch to avoid single-pixel depth map noise or edge artifacts.
        Returns linearized depth value in range [0.0, 1.0] matching the bokeh raymarch focal plane.
        """
        import numpy as np
        if cached_depth_np is not None and getattr(cached_depth_np, 'size', 0) > 0:
            h, w = cached_depth_np.shape[:2]
            px = int(np.clip(u * w, 0, w - 1))
            py = int(np.clip((1.0 - v) * h, 0, h - 1))

            r = max(1, sample_radius)
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            patch = cached_depth_np[y0:y1, x0:x1]

            if patch.size > 0:
                val = float(np.median(patch))
            else:
                val = float(cached_depth_np[py, px])

            if cached_depth_np.dtype == np.uint16:
                raw_d = val / 65535.0
            elif cached_depth_np.dtype == np.uint8:
                raw_d = val / 255.0
            else:
                raw_d = val
            return float(np.power(np.clip(raw_d, 0.0, 1.0), 2.0))

        if getattr(self, 'global_depth_tex', None) is None:
            return 0.5

        px = int(np.clip(u * self.global_depth_tex.width, 0, self.global_depth_tex.width - 1))
        py = int(np.clip(v * self.global_depth_tex.height, 0, self.global_depth_tex.height - 1))

        try:
            import struct
            fbo = self.ctx.framebuffer(color_attachments=[self.global_depth_tex])
            data = fbo.read(viewport=(px, py, 1, 1), components=1, dtype='f4')
            raw_depth = struct.unpack('1f', data)[0]
            z_linear = float(np.power(np.clip(raw_depth, 0.0, 1.0), 2.0))
            return z_linear
        except Exception as e:
            print(f"Error picking depth: {e}")
            return 0.5

    def paint_mask(self, layer_id: str, u: float, v: float, radius: float, hardness: float, opacity: float, mode: str = 'paint'):
        tex = self.get_or_create_mask(layer_id)
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        
        if mode == 'blur':
            # Create a temporary texture to copy current mask state
            temp_tex = self.ctx.texture(tex.size, 1, dtype='f1')
            temp_fbo = self.ctx.framebuffer(color_attachments=[temp_tex])
            
            # Copy current mask to temp
            self.ctx.copy_framebuffer(temp_fbo, fbo)
            
            # Now bind temp_tex to unit 0
            temp_tex.use(0)
            
            # Disable blending because the blur shader does the blending in fragment shader
            self.ctx.disable(moderngl.BLEND)
            
            self.blur_brush_program['brush_pos'].value = (u, v)
            self.blur_brush_program['brush_radius'].value = radius
            self.blur_brush_program['brush_hardness'].value = hardness
            self.blur_brush_program['aspect_ratio'].value = float(self.width) / float(self.height)
            
            self.blur_brush_vao.render(moderngl.TRIANGLE_STRIP)
            
            temp_tex.release()
            temp_fbo.release()
        else:
            # Enable blending to accumulate brush strokes
            self.ctx.enable(moderngl.BLEND)
            
            # Additive blending (or subtractive for erase)
            # Since texture and FBO are 8-bit normalized GL_R8 ('f1'),
            # OpenGL hardware color buffer blending strictly clamps values to [0.0, 1.0] (0..255).
            if mode == 'paint':
                # Add to mask, clamped at 1.0
                self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
                self.ctx.blend_equation = moderngl.FUNC_ADD
            else:
                # Subtract from mask, clamped at 0.0
                self.ctx.blend_equation = moderngl.FUNC_REVERSE_SUBTRACT
                self.ctx.blend_func = moderngl.ADDITIVE_BLENDING
                
            self.brush_program['brush_pos'].value = (u, v)
            self.brush_program['brush_radius'].value = radius
            self.brush_program['brush_hardness'].value = hardness
            self.brush_program['brush_opacity'].value = float(np.clip(opacity, 0.0, 1.0))
            self.brush_program['aspect_ratio'].value = float(self.width) / float(self.height)
            
            self.brush_vao.render(moderngl.TRIANGLE_STRIP)
            
            self.ctx.disable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.DEFAULT_BLENDING
            self.ctx.blend_equation = moderngl.FUNC_ADD
            
        fbo.release()


    def update_color_equalizer_lut(self, lut_data: np.ndarray):
        """
        lut_data: (1024, 4) float32 numpy array.
        R: Hue shift, G: Saturation scale, B: Lightness offset.
        """
        if lut_data.shape != (1024, 4) or lut_data.dtype != np.float32:
            raise ValueError("LUT must be a (1024, 4) float32 array")
        self.ceq_texture.write(lut_data.tobytes())

    def update_shadow_color_equalizer_lut(self, lut_data: np.ndarray):
        if lut_data.shape != (1024, 4) or lut_data.dtype != np.float32:
            raise ValueError("LUT must be a (1024, 4) float32 array")
        self.shadow_ceq_texture.write(lut_data.tobytes())

    def update_tone_equalizer_lut(self, lut_data: np.ndarray):
        """
        lut_data: (1024,) float32 numpy array of EV adjustments.
        """
        if lut_data.shape != (1024,) or lut_data.dtype != np.float32:
            raise ValueError("Tone LUT must be a (1024,) float32 array")
        self.tone_eq_texture.write(lut_data.tobytes())

    def set_depth_map(self, depth_data: np.ndarray):
        """
        depth_data: 2D numpy array (H, W) in float32 [0.0, 1.0] or uint8 [0, 255]
        """
        if depth_data is None:
            for attr in ['raw_depth_tex', 'global_depth_tex']:
                if hasattr(self, attr) and getattr(self, attr):
                    try: getattr(self, attr).release()
                    except: pass
                setattr(self, attr, None)
            return
        
        if depth_data.dtype == np.uint8:
            depth_data = depth_data.astype(np.float32) / 255.0
        elif depth_data.dtype == np.uint16:
            depth_data = depth_data.astype(np.float32) / 65535.0
        elif depth_data.dtype != np.float32:
            depth_data = depth_data.astype(np.float32)
            
        if depth_data.ndim == 3:
            depth_data = depth_data[..., 0]
            
        h, w = depth_data.shape[:2]
        for attr in ['raw_depth_tex', 'global_depth_tex']:
            if hasattr(self, attr) and getattr(self, attr):
                try: getattr(self, attr).release()
                except: pass
                
        self.raw_depth_tex = self.ctx.texture((w, h), 1, data=np.ascontiguousarray(depth_data).tobytes(), dtype='f4')
        self.raw_depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_depth_tex.repeat_x = False
        self.raw_depth_tex.repeat_y = False

        self.global_depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self.global_depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.global_depth_tex.repeat_x = False
        self.global_depth_tex.repeat_y = False

        self._apply_lens_to_map_texture(self.raw_depth_tex, self.global_depth_tex, w, h)

    def set_normal_map(self, normal_data: np.ndarray):
        """
        normal_data: (H, W, 3) numpy array in float32 [0.0, 1.0] or uint8 [0, 255]
        """
        if normal_data is None:
            for attr in ['raw_normal_tex', 'global_normal_tex']:
                if hasattr(self, attr) and getattr(self, attr):
                    try: getattr(self, attr).release()
                    except: pass
                setattr(self, attr, None)
            return
            
        if normal_data.dtype == np.uint8:
            normal_data = normal_data.astype(np.float32) / 255.0
        elif normal_data.dtype != np.float32:
            normal_data = normal_data.astype(np.float32)
            
        h, w = normal_data.shape[:2]
        for attr in ['raw_normal_tex', 'global_normal_tex']:
            if hasattr(self, attr) and getattr(self, attr):
                try: getattr(self, attr).release()
                except: pass
                
        self.raw_normal_tex = self.ctx.texture((w, h), 3, data=np.ascontiguousarray(normal_data).tobytes(), dtype='f4')
        self.raw_normal_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_normal_tex.repeat_x = False
        self.raw_normal_tex.repeat_y = False

        self.global_normal_tex = self.ctx.texture((w, h), 3, dtype='f4')
        self.global_normal_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.global_normal_tex.repeat_x = False
        self.global_normal_tex.repeat_y = False

        self._apply_lens_to_map_texture(self.raw_normal_tex, self.global_normal_tex, w, h)

    def set_subject_mask(self, mask_data: np.ndarray):
        """
        mask_data: (H, W) numpy array in float32 [0.0, 1.0] or uint8 [0, 255]
        """
        if mask_data is None:
            for attr in ['raw_subject_mask_tex', 'global_subject_mask_tex']:
                if hasattr(self, attr) and getattr(self, attr):
                    try: getattr(self, attr).release()
                    except: pass
                setattr(self, attr, None)
            return
            
        if mask_data.dtype == np.uint8:
            mask_data = mask_data.astype(np.float32) / 255.0
        elif mask_data.dtype != np.float32:
            mask_data = mask_data.astype(np.float32)
            
        if mask_data.ndim == 3:
            mask_data = mask_data[..., 0]
            
        h, w = mask_data.shape[:2]
        for attr in ['raw_subject_mask_tex', 'global_subject_mask_tex']:
            if hasattr(self, attr) and getattr(self, attr):
                try: getattr(self, attr).release()
                except: pass
                
        self.raw_subject_mask_tex = self.ctx.texture((w, h), 1, data=np.ascontiguousarray(mask_data).tobytes(), dtype='f4')
        self.raw_subject_mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.raw_subject_mask_tex.repeat_x = False
        self.raw_subject_mask_tex.repeat_y = False

        self.global_subject_mask_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self.global_subject_mask_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.global_subject_mask_tex.repeat_x = False
        self.global_subject_mask_tex.repeat_y = False

        self._apply_lens_to_map_texture(self.raw_subject_mask_tex, self.global_subject_mask_tex, w, h)

    def accumulate_bokeh_step(self) -> bool:
        """
        Performs a progressive Monte Carlo accumulation step for Bokeh raymarching.
        Returns True if more samples are needed, False if max_samples is reached.
        """
        if not getattr(self, 'bokeh_enabled', False) or self.texture is None:
            return False
        max_samples = getattr(self, 'bokeh_max_samples', 256)
        if getattr(self, 'frame_count', 0) >= max_samples:
            return False
        self.render(to_preview=True)
        return True

    def read_preview_bytes(self) -> bytes:
        """
        Ultra-fast direct readback of 8-bit sRGB viewport frame from GPU framebuffer.
        Zero CPU float mathematics, 4x lower PCIe bandwidth.
        """
        if self.texture is None or getattr(self, 'preview_display_fbo', None) is None:
            return b""
        if self.prev_w <= 0 or self.prev_h <= 0:
            return b""
        with profiler.scope("fbo_read_ms"):
            raw_bytes = self.preview_display_fbo.read(components=3, dtype='f1', viewport=(0, 0, self.prev_w, self.prev_h))
        return raw_bytes

    def read_pixels(self, to_preview: bool = True) -> np.ndarray:
        """Helper to read numpy array pixels from preview or full render FBO."""
        if to_preview:
            return self.get_preview_array()
        else:
            return self.get_result_array()

    def get_result_array(self) -> np.ndarray:
        if self.texture is None:
            return np.zeros((1,1,3), dtype=np.float32)
        if self.fbo is None:
            self.render(to_preview=False)
            
        raw_data = self.fbo.read(components=3, dtype='f4', viewport=(0, 0, self.out_w, self.out_h))
        arr = np.frombuffer(raw_data, dtype=np.float32).reshape((self.out_h, self.out_w, 3)).copy()
        np.nan_to_num(arr, copy=False)
        arr *= 50.0
        return arr
        
    def get_preview_array(self) -> np.ndarray:
        """Returns 8-bit (H, W, 3) uint8 numpy array of current preview frame."""
        if self.texture is None:
            return np.zeros((1,1,3), dtype=np.uint8)
        if getattr(self, 'preview_display_fbo', None) is None or self.prev_w <= 0 or self.prev_h <= 0:
            self.render(to_preview=True)
            
        raw_data = self.read_preview_bytes()
        if not raw_data:
            return np.zeros((max(1, self.prev_h), max(1, self.prev_w), 3), dtype=np.uint8)
            
        return np.frombuffer(raw_data, dtype=np.uint8).reshape((self.prev_h, self.prev_w, 3))
        
    def get_preview_jpeg_bytes(self, max_size=1024) -> bytes:
        if self.texture is None:
            return b""
            
        if self.prev_w <= 0 or self.prev_h <= 0 or getattr(self, 'preview_display_fbo', None) is None:
            return self.get_ai_jpeg_bytes(max_size=max_size)
            
        arr = self.get_preview_array()
        img = Image.fromarray(arr)
        
        # Save to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()
        
    def get_ai_jpeg_bytes(self, max_size=2048) -> bytes:
        if self.texture is None:
            return b""
            
        aspect = self.out_w / self.out_h
        if max_size <= 0:
            target_w = self.out_w
            target_h = self.out_h
        elif self.out_w > self.out_h:
            target_w = min(self.out_w, max_size)
            target_h = int(target_w / aspect)
        else:
            target_h = min(self.out_h, max_size)
            target_w = int(target_h * aspect)
            
        target_w = max(1, target_w)
        target_h = max(1, target_h)
            
        # Use main FBO if the target is larger than the preview FBO capacity
        use_preview = target_w <= 4096 and target_h <= 4096
        fbo = self.preview_fbo if use_preview else self.fbo
        
        # Render using MVP_export to target viewport
        fbo.viewport = (0, 0, target_w, target_h)
        self.program['mvp'].write(self.MVP_export.T.tobytes())
        if 'mvp_base' in self.program:
            self.program['mvp_base'].write(self.MVP_export.T.tobytes())
            
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Bind textures
        self.texture.use(0)
        self.ceq_texture.use(1)
        self.tone_eq_texture.use(2)
        self.shadow_ceq_texture.use(3)
        
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        raw_data = fbo.read(components=3, dtype='f4', viewport=(0, 0, target_w, target_h))
        array_f32 = np.frombuffer(raw_data, dtype=np.float32).reshape((target_h, target_w, 3)).copy()
        
        # Replace NaNs just in case
        array_f32 = np.nan_to_num(array_f32, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
        
        array_f32 *= 50.0
        
        # Clamp and gamma
        array_f32 = np.clip(array_f32, 0.0, 1.0)
        mask = array_f32 <= 0.0031308
        result_gamma = np.where(
            mask,
            array_f32 * 12.92,
            1.055 * (array_f32 ** (1.0 / 2.4)) - 0.055
        )
        
        result_8bit = (result_gamma * 255.0).astype(np.uint8)
        img = Image.fromarray(result_8bit)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        # Restore normal preview
        self.render(to_preview=True)
        
        return buffer.getvalue()

    def get_ai_image_numpy(self, high_res=False) -> np.ndarray:
        """
        Returns an 8-bit RGB numpy array (H, W, 3) representing the original, untransformed
        initial image (full initial width and height, without crop, rotation, or MVP transformation).
        This ensures generated masks and depth maps cover the entire original UV space [0, 1]x[0, 1]
        and dynamically align with any subsequent viewport cropping or rotation in the shaders.
        """
        if not hasattr(self, 'raw_texture') or self.raw_texture is None:
            if not hasattr(self, 'texture') or self.texture is None:
                return None
            
        import numpy as np
        import cv2
        
        # Read the initial uncropped texture (self.raw_texture or self.texture)
        if hasattr(self, 'raw_fbo') and self.raw_fbo:
            raw_data = self.raw_fbo.read(components=3, dtype='f4', viewport=(0, 0, self.width, self.height))
            array_f32 = np.frombuffer(raw_data, dtype=np.float32).reshape((self.height, self.width, 3)).copy()
        else:
            tex = self.raw_texture if hasattr(self, 'raw_texture') and self.raw_texture else self.texture
            raw_bytes = tex.read()
            arr_full = np.frombuffer(raw_bytes, dtype=np.float32).reshape((self.height, self.width, tex.components))
            array_f32 = arr_full[..., :3].copy()
        
        # Replace NaNs just in case
        array_f32 = np.nan_to_num(array_f32, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
        array_f32 *= 50.0
        
        # Clamp and apply standard sRGB gamma
        array_f32 = np.clip(array_f32, 0.0, 1.0)
        mask = array_f32 <= 0.0031308
        result_gamma = np.where(
            mask,
            array_f32 * 12.92,
            1.055 * (array_f32 ** (1.0 / 2.4)) - 0.055
        )
        
        result_8bit = (np.clip(result_gamma, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        if not high_res:
            max_size = 2048
            aspect = self.width / self.height
            if self.width > self.height:
                target_w = min(self.width, max_size)
                target_h = max(1, int(target_w / aspect))
            else:
                target_h = min(self.height, max_size)
                target_w = max(1, int(target_h * aspect))
                
            if target_w != self.width or target_h != self.height:
                result_8bit = cv2.resize(result_8bit, (target_w, target_h), interpolation=cv2.INTER_AREA)
                
        return result_8bit

    def get_linear_raw_numpy(self) -> Optional[np.ndarray]:
        """
        Returns the unscaled, true scene-referred linear float32 RGB data [H, W, 3]
        from raw_texture. Preserves exact sensor dynamic range and white balance without
        applying artificial gain or gamma transformations.
        """
        if not hasattr(self, 'raw_texture') or self.raw_texture is None:
            if not hasattr(self, 'texture') or self.texture is None:
                return None
        import numpy as np
        tex = self.raw_texture if hasattr(self, 'raw_texture') and self.raw_texture else self.texture
        try:
            raw_bytes = tex.read()
            dtype = np.float16 if getattr(tex, 'dtype', 'f4') == 'f2' else np.float32
            arr_full = np.frombuffer(raw_bytes, dtype=dtype).reshape((self.height, self.width, tex.components))
            arr_f32 = np.ascontiguousarray(arr_full[..., :3].astype(np.float32))
            return np.nan_to_num(arr_f32, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"Error reading linear raw data from texture: {e}")
            return None

    def ensure_full_bokeh_accumulation(self) -> None:
        """
        Forces full Monte Carlo sample accumulation for Bokeh raymarching
        until bokeh_max_samples is reached. Ensures clean noise-free output for AI and preview.
        """
        if not getattr(self, 'bokeh_enabled', False) or self.texture is None:
            return
        if getattr(self, 'global_depth_tex', None) is None:
            return
        max_samples = getattr(self, 'bokeh_max_samples', 256)
        while getattr(self, 'frame_count', 0) < max_samples:
            if not self.accumulate_bokeh_step():
                break

    def get_viewport_cropped_image(self, viewport_label=None, fully_accumulate_bokeh: bool = True, save_debug_copy: bool = False, max_size: int = 1280, force_uncropped: bool = False) -> Tuple[Image.Image, Optional[str]]:
        """
        Ultra-fast (1-2 ms) GPU hardware capture of the complete, un-split AFTER photo.
        Renders into preview FBO directly fitting the photo aspect ratio, performs hardware
        sRGB display pass on the GPU, and reads 8-bit uint8 bytes in microseconds.
        Guarantees ZERO large float copies across PCIe, ZERO main-thread CPU lag,
        and 100% full-frame capture with no Side-by-Side or viewport crop artifacts.
        When force_uncropped=True, temporarily resets crop boundaries to capture the 100% full uncropped sensor view.
        """
        if self.texture is None or self.out_w <= 0 or self.out_h <= 0:
            return Image.new('RGB', (256, 256), color=(0, 0, 0)), None

        # Temporarily save state
        orig_split_engine = getattr(self, 'split_view_mode', None)
        orig_split_mw = getattr(viewport_label.main_window, 'split_view_mode', None) if (viewport_label and hasattr(viewport_label, 'main_window')) else None
        orig_zoom, orig_pan_x, orig_pan_y = self.view_zoom, self.view_pan_x, self.view_pan_y
        orig_prev_w, orig_prev_h = max(1, self.prev_w), max(1, self.prev_h)
        orig_crop = (self.crop_l, self.crop_r, self.crop_t, self.crop_b) if force_uncropped else None

        try:
            # 1. Disable split views and reset zoom/pan to capture clean full frame
            self.split_view_mode = None
            if viewport_label and hasattr(viewport_label, 'main_window'):
                viewport_label.main_window.split_view_mode = None
            self.view_zoom = 1.0
            self.view_pan_x = 0.0
            self.view_pan_y = 0.0

            if force_uncropped:
                self.set_crop(0.0, 0.0, 0.0, 0.0)

            # 2. Set preview resolution exactly matching image aspect ratio up to max_size
            aspect = float(self.out_w) / float(max(1, self.out_h))
            if aspect >= 1.0:
                pw = min(max_size, self.out_w)
                ph = max(1, int(round(pw / aspect)))
            else:
                ph = min(max_size, self.out_h)
                pw = max(1, int(round(ph * aspect)))

            self.set_preview_size(pw, ph)

            # 3. Render preview on GPU (hardware tone-mapping, all active layers, S-curve, etc.)
            self.render(to_preview=True)

            if fully_accumulate_bokeh and getattr(self, 'bokeh_enabled', False):
                self.ensure_full_bokeh_accumulation()

            # 4. Ultra-fast hardware read of 8-bit uint8 frame (just ~3-4 MB)
            raw_bytes = self.preview_display_fbo.read(components=3, dtype='f1', viewport=(0, 0, pw, ph))
            arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((ph, pw, 3))
            img = Image.fromarray(arr)

            # 5. Save debug copy if requested
            debug_path = None
            if save_debug_copy:
                import datetime
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                debug_dir = os.path.join(root_dir, "debug_gemini_inputs")
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                debug_path = os.path.join(debug_dir, f"gemini_input_{timestamp}.jpg")
                img.save(debug_path, format="JPEG", quality=90)

            return img, debug_path

        finally:
            # 6. Restore previous viewport state & preview size instantly
            if force_uncropped and orig_crop is not None:
                self.set_crop(*orig_crop)
            self.split_view_mode = orig_split_engine
            if viewport_label and hasattr(viewport_label, 'main_window'):
                viewport_label.main_window.split_view_mode = orig_split_mw
            self.view_zoom = orig_zoom
            self.view_pan_x = orig_pan_x
            self.view_pan_y = orig_pan_y
            self.set_preview_size(orig_prev_w, orig_prev_h)
            self.render(to_preview=True)

    set_image = load_texture
    load_image = load_texture

