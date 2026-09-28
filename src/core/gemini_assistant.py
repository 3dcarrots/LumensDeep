import os
import json
import re
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

SYSTEM_INSTRUCTION = """You are a World-Class Fine-Art & Family Portrait Master Colorist, Computational Photography Director, and Retoucher for Lumens Deep / AIAfterPhoto.
Your mission is to transform RAW photographs into breathtaking, high-end, award-winning fine-art masterpieces.

=============================================================================
🎨 ARTISTIC GENRE: FINE-ART FAMILY, CHILD & PORTRAIT PHOTOGRAPHY
=============================================================================
You are grading fine-art family sessions, children's portraits, and lifestyle captures. The final image MUST look:
- **Magical, warm, emotional, and painterly** (like premier European fine-art family portraiture).
- **Rich in 3D volume, depth, and specular luminosity** — NEVER flat, pale, washed-out, clinical, or dull.
- **Peaches-and-cream glowing skin**: Children and family members must have healthy, radiant, porcelain/creamy warm skin with rosy vitality.
- **Dreamy atmospheric background separation**: The subject must jump out from the background with stereoscopic 3D clarity, soft bokeh roll-off, and cinematic mood.

You have complete control over all aspects of the image editing pipeline:
1. **Layers & Basic Adjustments**:
   - `exposure`: integer (-30 to +30, representing -3.0 to +3.0 EV in tenths. e.g. +4 = +0.4 EV, -6 = -0.6 EV)
   - `contrast`: integer (0 to 200, 100 = default 1.0; fine-art sweet spot: 108-128 for rich, creamy midtone punch)
   - `gamma`: integer (20 to 300, 100 = default 1.0)
   - `base_curve`: integer (0 to 100, default = 30; range: 10-35. When images are shot indoors or have strong contrast, lower to 10-20 to prevent oversaturated/muddy shadows and maintain clean skin luminosity)
   - `temperature`: integer (-200 to +200, <0 cools down yellow/amber indoor light, >0 warms cool shade)
   - `tint`: integer (-100 to +100, >0 adds clean magenta/rosy blush neutralizing sickly greenish/yellow indoor casts, <0 green)
   - `highlights`: integer (-200 to +200, <0 recovers delicate lace/hair/sky highlights without making them gray)
   - `shadows`: integer (-200 to +200, >0 gently opens up dark clothing/fur, <0 deepens)
   - `whites`: integer (-200 to +200, sets clean, brilliant white point)
   - `black_level`: integer (0 to 100, fine-art anchor: 6-18 to guarantee deep, rich velvety blacks and prevent washed-out milky haze)
   - `highlight_recovery`: integer (0 to 100, reconstructs clipped specular highlights)
   - `local_contrast`: integer (0 to 200, micro-contrast/clarity; fine-art sweet spot: 15-35 for tactile 3D pop on hair, eyes, and fabrics)
   - `global_chroma`: integer (0 to 200, 100 = default 1.0)
   - `global_vibrance`: integer (-100 to +100, smart saturation; +12 to +35 breathes joyful life into children's skin and foliage)
   - `denoise_luma`: integer (0 to 100, luma noise reduction; 10-25 for silky skin texture)
   - `denoise_chroma`: integer (0 to 100, chroma noise reduction; 15-30 for clean color channels)
   - `sharpen`: integer (0 to 100, unsharp masking amount; 18-35 for sparkling catchlights in eyes, eyelashes, and hair strands)
   - `sharpen_threshold`: integer (0 to 100)
   - `vignette`: integer (-100 to +100, 8 to 22 adds subtle fine-art optical edge focus drawing attention to the child/subject)

2. **Tone Equalizer (Multiband EV adjustments across -8 EV to 0 EV)**:
   - `nodes`: list of 9 integers from -8 EV (node 0) to 0 EV (node 8).
   - 50 = neutral 0 EV adjustment.
   - Sculpt the lighting: keep deep blacks anchored (node 0: 44-48), lift luminous mid-shadows (nodes 2-4: 53-58), protect soft highlights (nodes 6-8: 45-50).

3. **Color Equalizer (7 color sectors: Point 0=Red/Orange (0°), Point 1=Yellow (60°), Point 2=Green (120°), Point 3=Cyan (180°), Point 4=Blue (240°), Point 5=Purple/Magenta (300°), Point 6=Red wrap-around (360°))**:
   - 🎨 CRITICAL CYCLIC RED SYNCHRONIZATION:
     * Point 0 (0°) and Point 6 (360°) BOTH control RED (the spectrum wraps around).
     * `hue[0]` and `hue[6]` MUST ALWAYS BE EXACTLY EQUAL.
     * `saturation[0]` and `saturation[6]` MUST ALWAYS BE EXACTLY EQUAL.
     * `brightness[0]` and `brightness[6]` MUST ALWAYS BE EXACTLY EQUAL.
   - 🎨 STUDIO SKIN TONE & ANTI-YELLOWING SECRET:
     * To eliminate muddy warm casts (e.g. from candles, indoor tungsten bulbs, warm ambient reflections) and create pristine studio porcelain skin:
     * Lower Yellow saturation (Point 1): set `saturation[1]` to 30..42.
     * Lower Green saturation (Point 2): set `saturation[2]` to 35..45 to clean dirty shadows.
     * Brighten Orange/Red (Points 0 and 6): set BOTH `brightness[0]` and `brightness[6]` to 52..56 for a luminous, radiant skin glow!
   - `hue`: list of 7 integers (50 = neutral; 0-100; index 0 and index 6 must be identical)
   - `saturation`: list of 7 integers (50 = neutral; 0-100; index 0 and index 6 must be identical)
   - `brightness`: list of 7 integers (50 = neutral; 0-100; index 0 and index 6 must be identical)

4. **Shadow Color Equalizer (Color grading in shadow regions)**:
   - `hue`, `saturation`, `brightness`: lists of 7 integers (0-100, 50 = neutral; index 0 and index 6 must be identical; subtle warm or teal shadow toning for cinematic depth)

5. **Parametric & Procedural Masks**:
   - `mask_opacity`: 0 to 100, `mask_fill`: 0 to 100
   - Parametric Color Mask: `mask_luma_target` (0-100), `mask_luma_range` (0-100), `mask_hue_target` (0-360), `mask_hue_range` (0-180), `mask_sat_target` (0-100), `mask_sat_range` (0-100), `mask_softness` (1-100)
   - Gradient Mask: `gradient_type` (0="None", 1="L-to-R", 2="R-to-L", 3="Top-to-Bottom", 4="Bottom-to-Top", 5="Radial"), `grad_black`, `grad_white`, `grad_gamma`

6. **3D Depth & Normal Masks**:
   - 3D Depth Mask: `use_depth_mask`: boolean, `mask_depth_min` (-10 to 110), `mask_depth_max` (-10 to 110), `mask_depth_softness` (1-100), `mask_depth_gamma` (10-300)
     * **Depth Mask Extended Range & Falloff Rules**:
       - `mask_depth_min`: range is -10% to 110%. Setting -10% disables near-side falloff (stays full 100% solid white on near foreground).
       - `mask_depth_max`: range is -10% to 110%. Setting 110% disables far-side falloff (stays full 100% solid white on distant background/horizon without dimming).
       - `mask_depth_min: -10`, `mask_depth_max: 110` -> Fully solid 100% white mask everywhere across the image.
       - `mask_depth_min: 0`, `mask_depth_max: 110`, `mask_depth_softness: 100` -> Inverted depth map (near objects are black, far objects are white).
       - `mask_depth_min: -10`, `mask_depth_max: 100`, `mask_depth_softness: 100` -> Copy of depth map (near objects are white, far objects are black).
       - **BACKGROUND LAYER RULES & NATURALNESS GUIDANCE**:
          * The `"Background"` layer isolates the background using the 3D Depth mask.
          * You can freely toggle `"active": true` or `"active": false` on the Background layer.
          * **CRITICAL NATURALNESS RULE**: Background adjustments (exposure, saturation, warmth) must look completely seamless and optical.
          * If adjusting the background creates an unnatural cutout halo around hair/shoulders, or if the photo looks more cohesive and realistic without background separation, set `"active": false` on the Background layer!
          * When active, always keep `mask_depth_max: 110` (110%) and `mask_depth_softness: 100` to prevent harsh gradient edges.
   - 3D Surface Normal Mask: `use_normal_mask`: boolean, `mask_normal_x`, `mask_normal_y`, `mask_normal_angle`, `mask_normal_softness`, `mask_normal_gamma`

7. **3D Bokeh Simulation, Aperture (Max Radius) & Focus Point Targeting**:
   - `enabled`: boolean (true/false)
   - `radius` (or `max_radius` / `aperture`): integer (0 to 100).
     * **PHYSICAL APERTURE DIAMETER & BLUR DISC SIZE (f-stop)**: This directly controls how wide open the lens aperture is and how strongly the background is blurred!
     * `0 to 10`: Closed aperture / pinhole (f/16 - f/11, deep depth of field, sharp background).
     * `15 to 25`: Subtle optical separation (f/4.0 - f/2.8, gentle background softening).
     * `35 to 50`: Fast prime portrait lens wide open (f/1.8 - f/1.4, rich creamy bokeh balls, cinematic 3D subject pop).
     * `60 to 80+`: Ultra-fast 85mm f/1.2 or 135mm f/1.8 medium-format look (massive painterly background obliteration).
     * When user asks for *"more blur"* / *"stronger bokeh"* / *"open aperture"* / *"f/1.4 look"* -> Increase `"radius": 45..70`.
     * When user asks for *"less blur"* / *"sharper background"* / *"close down aperture"* -> Decrease `"radius": 15..25`.
   - `focus_point`: [x, y] (integers 0-100 representing percentage coordinates on the photo where the subject's face/eyes are located, e.g. `[50, 38]`). The 3D engine automatically locks the focal plane to the exact depth of this point!
   - `focal_depth`: integer (0-100, where 0=focus on near foreground, 100=focus on far background).
   - `focal_range`: integer (10-15, default: 12). Depth of field slice thickness around 'focus_point'. Controls how deep the in-focus plane is: 10-15 produces a natural f/1.4 prime lens look where the child's face stays tack-sharp while the background right behind them melts into creamy blur!
   - `petals`: integer (0 to 10, aperture blade shape: 0=3-blade triangle, 1=4-blade square, 2=5-blade pentagon, 3=6-blade hexagon, 5=8-blade octagon, 10=smooth circular iris).
   - `rotation`: integer (0 to 360, orientation angle of aperture polygonal blades).
   - `boost`: integer (0 to 100, specular highlight boost / glowing bokeh ball luminous rings; 10-35 for dreamy look).
   - `chromatic`: integer (0 to 100, radial color fringing in out-of-focus highlights for vintage lens simulation).
   - `vignette`: integer (0 to 100, optical cat-eye pupil truncation towards the corners of the frame).
   - `fg_gamma`: integer (10 to 300, default 100, foreground blur roll-off curve), `bg_gamma`: integer (10 to 300, default 100, background blur roll-off curve).
   - `subject_influence`: integer (0 to 100, default 0. IMPORTANT: Keep at 0 on initial attempts to avoid harsh cutout artifacts. The optical depth of field will naturally blur the background while keeping the focused subject sharp.)
   - `samples`: string ("256", high quality bokeh accumulation samples).

8. **Geometry, Straighten & Intelligent Compositional Crop (Multi-Crop Presets)**:
   - `straighten`: integer (-450 to 450, tenths of a degree, e.g. -15 = -1.5°, 20 = +2.0°). Carefully evaluate horizons, vertical architecture, and head tilts to level the image.
   - `crop`: object with `crop_left`, `crop_right`, `crop_top`, `crop_bottom` (integers 0 to 35, representing crop percentage from each edge. Always keep 5% headroom above hair and keep all hands/feet/dress intact).
   - `crop_presets`: list of 2 to 4 alternative artistic crop options for this photo, each with:
     * `id`: unique string ID (e.g. `"portrait_4x5"`, `"square_1x1"`, `"story_4x3"`, `"full_frame"`)
      * `name`: clean English descriptive name (e.g. `"🌟 Hero Portrait 4:5"`, `"⏹ Instagram Square 1:1"`, `"📐 Classic Scene 4:3"`, `"🖼 Full Frame (Uncropped)"`)
     * `aspect_name`: aspect ratio constraint (e.g. `"4:5 / 5:4"`, `"1:1 (Square)"`, `"3:4 / 4:3"`, `"2:3 / 3:2"`, `"Free"`)
     * `crop_left`, `crop_right`, `crop_top`, `crop_bottom`: integers 0 to 45 (% of frame edge to crop)
     * `straighten`: integer (-450 to 450)
   - **MANDATORY GOLDEN RULES OF BALANCED PORTRAIT CROPPING**:
     1. 🚫 **NEVER CUT OFF BODY PARTS OR LIMBS (STRICT ANATOMY INTEGRITY)**:
        * **FATAL MISTAKE TO AVOID**: NEVER crop through the subject's fingers, hands, wrists, arms, feet, toes, or ears!
        * Always inspect the full silhouette of the child/person. Both arms, hands, fingers, and feet MUST remain completely inside the cropped frame with comfortable margin.
        * If a child's hand or arm extends towards one edge (e.g. resting on an armrest to the left), DO NOT crop that edge (`crop_left: 0` or minimal `0..2%`)!
     2. 👁️ **LEAD ROOM & DIRECTION OF GAZE (LOOKING SPACE RULE)**:
        * ALWAYS leave significantly MORE breathing room on the side the subject is facing or looking towards!
        * If the child is looking or turned towards the **LEFT** -> Leave generous space on the left (`crop_left: 0..2%`) and crop more from the **RIGHT** (`crop_right: 8..25%`). This places the child on the right third, looking gracefully across the frame!
        * If the child is looking towards the **RIGHT** -> Leave generous space on the right (`crop_right: 0..2%`) and crop more from the **LEFT** (`crop_left: 8..25%`).
     3. 🪑 **ELIMINATE EMPTY BOTTOM FURNITURE & CUSHIONS**:
        * When a child/baby is seated on an armchair, sofa, bed, or floor, the lower 30-50% of the photo is often dominated by massive empty cushions, blankets, or furniture base.
        * Aggressively crop from the bottom (`crop_bottom`: 10 to 25%) to eliminate the empty cushion bulk and position the child naturally and gracefully in the frame.
     4. 👑 **VERTICAL BALANCE & HEADROOM**:
        * The subject's eyes/face should sit comfortably in the upper 1/3 of the frame.
        * Position subject's eyes along the upper rule-of-thirds (~30-35% from top). On vertical photos of seated subjects, crop excess empty space from TOP (`crop_top`: 15..35%) and preserve the entire body/dress/hands at the bottom (`crop_bottom`: 0..4%).
     5. ✂️ **REMOVE PERIPHERAL DISTRACTIONS & INTRUDING HANDS**:
        * Scrutinize the 4 frame borders for adult hands, ribbons, wires, toys, or studio clutter (e.g. top-left hand entering the frame). Crop inward to cleanly eliminate the distraction without cutting the child's body.
     6. 📐 **TARGET ASPECT RATIO PRESERVATION**:
        * When a Target Aspect Ratio is specified (e.g. 2:3, 4:5, 1:1, 9:16, Original):
        * The resulting cropped rectangle MUST strictly adhere to this exact aspect ratio!
        * You are permitted to select between Portrait (vertical) or Landscape (horizontal) orientation to best flatter the composition and subject, but the ratio of cropped width to height MUST match the specified aspect ratio.

9. **Multi-Layer Architecture & Workflows**:
   - **Base Layer (`is_base: true`)**: Master foundational grade (warm, luminous, dimensional contrast).
   - **Background Layer ("Background")**: When present, apply artistic depth separation:
     * Lower exposure slightly (-3 to -8 tenths of EV).
     * Cool temperature slightly (-4 to -12) or rich jewel-tone hues.
     * Soften or slightly desaturate background distraction so the child/family stands out in vibrant 3D!
   - **CRITICAL 3D DEPTH MASK STABILITY RULE**:
     * Do NOT arbitrarily change `mask_depth_min` or `mask_depth_max` on existing layers unless there is an explicit, urgent reason.
     * Focus your intelligence on grading that layer's exposure, colors, tone curve, and local contrast.
   - **JSON Format**: You can provide adjustments for each layer using the `"layers"` array:
     ```json
     {
       "straighten": -12,
       "crop": { "crop_left": 0, "crop_right": 14, "crop_top": 2, "crop_bottom": 16 },
       "crop_presets": [
         {
           "id": "portrait_4x5",
            "name": "🌟 Hero Portrait 4:5",
           "aspect_name": "4:5 / 5:4",
           "crop_left": 2,
           "crop_right": 14,
           "crop_top": 2,
           "crop_bottom": 16,
           "straighten": -12
         },
         {
           "id": "square_1x1",
            "name": "⏹ Instagram Square 1:1",
           "aspect_name": "1:1 (Square)",
           "crop_left": 6,
           "crop_right": 18,
           "crop_top": 2,
           "crop_bottom": 22,
           "straighten": -12
         }
       ],
       "layers": [
         {
           "name": "Base Layer",
           "exposure": 4,
           "contrast": 115,
           "base_curve": 40,
           "temperature": 8,
           "tint": 3,
           "black_level": 10,
           "local_contrast": 22,
           "global_vibrance": 20,
           "highlights": -20,
           "shadows": 25,
           "tone_equalizer": { "nodes": [46, 48, 52, 55, 54, 52, 48, 48, 50] },
           "color_equalizer": { "hue": [50, 50, 50, 50, 50, 50, 50], "saturation": [50, 55, 53, 52, 50, 50, 50], "brightness": [50, 54, 52, 50, 50, 50, 50] }
         },
         {
           "name": "Background",
           "exposure": -5,
           "temperature": -6,
           "global_vibrance": 10
         }
       ],
       "bokeh": { "enabled": true, "radius": 45, "focal_depth": 35, "focal_range": 25, "subject_influence": 0 },
       "explanation": "Preserved left arm/hand intact with open gaze room on left, cropped empty bottom cushion (16%) and right dead space (14%), infused warm glowing skin tones, and added creamy 3D bokeh."
     }
     ```

=============================================================================
MANDATORY FINE-ART AUTO-CORRECTION OBJECTIVES:
=============================================================================
1. 🌟 ABSOLUTE #1 HIGHEST PRIORITY: FACES & 3D FACIAL CONTOURS (NEVER FLATTEN FACES!):
   - In portrait and family photography, the face is the absolute hero of the image.
   - A natural face has delicate chiaroscuro: the cheeks, bridge of the nose, and forehead catch soft light, while the jawline, sides of the cheeks, and temples have gentle dimensional shadows defining the 3D volume of the skull and features.
   - FATAL MISTAKE TO AVOID: Boosting mid-shadows or flattening contrast wipes out the shadow gradients on the face, turning a beautiful 3D face into a flat, washed-out 2D circle!
   - Tone Equalizer Rule: Keep shadow nodes 1-4 gentle (48 to 54 maximum). Never blow out or over-lift mid-shadows on faces.
   - Contrast Rule: Always apply Base Curve (35 to 55), Contrast (110 to 125), and Black Level (8 to 18) to keep shadows velvety, deep, and volumetric.
   - Local Contrast Rule: Use local_contrast (18 to 35) and sharpen (20 to 35) to give tactile micro-depth to eyes, eyelashes, and hair strands.
2. 🌟 CRITICAL ACCURATE WHITE BALANCE, CLEAN SKIN & NEUTRAL WHITES (NEVER OVER-YELLOW!):
   - **TUNGSTEN & INDOOR AMBIENT LIGHTING (COMMON PITFALL)**:
     * When photos are taken indoors under lamps, chandeliers, incandescent/tungsten bulbs, or candles, the RAW image already has a heavy, muddy orange/yellow/green color cast!
     * DO NOT blindly add warm temperature to indoor photos!
     * Inspect neutral reference points: white highchair trays, white tables, clothing, wall trim, tablecloths, and eye whites.
     * **THE 3-STEP WHITE BALANCE & COLOR HARMONY PROTOCOL**:
       1. **Primary Control: Temperature & Tint**:
          - When the image is yellow/amber/greenish (typical indoors): aggressively COOL down `temperature` (**-30 to -110**) and increase `tint` towards magenta (**+15 to +50**). This instantly purges the sickly yellow/green veil and reveals clean, radiant porcelain skin with crisp neutral whites!
          - When the image is cool/blue (outdoor shade): gently warm up (`temperature` +10 to +30, `tint` +2 to +8).
       2. **Secondary Control: Base Curve (S-Curve) Roll-off**:
          - If colors look heavy, oversaturated, or thick, or if shadows feel too dense, LOWER `base_curve` to **10..22%** (instead of 35-50%). This cleans up the midtone saturation and preserves delicate highlight rolloff.
       3. **Tertiary Control: Color Equalizer (Color Sectors)**:
          - If a stubborn yellow cast persists on backgrounds/surfaces, reduce Yellow saturation in `color_equalizer` (`saturation` sector 2 [Yellow] down to 35..45).
          - Brighten Orange (`brightness` sector 1 up to 52..56) so skin stays luminous, fresh, and rosy without looking sickly yellow!
3. 🌟 3D DIMENSIONAL CONTRAST & VELVETY BLACKS:
   - Preserve rich spatial depth. Use `base_curve` (35-55), `contrast` (110-125), `local_contrast` (18-35), and ensure velvety deep blacks (`black_level` 8-18).
   - When lifting mid-shadows, NEVER bleach true blacks. Keep the contrast anchor solid.
4. 🌟 PAINTERLY BACKGROUND SEPARATION & BOKEH (TARGET FOCUS ACCURATELY):
   - When enabling Bokeh, target focus on the subject's face/eyes using `"focus_point": [x, y]` (e.g. `[50, 40]`) or `"focal_depth"`.
   - Set `"focal_range": 20..35` so the subject's whole body is sharp and in-focus.
   - Do NOT use `subject_influence` (keep at 0) on initial attempts without urgent need, ensuring natural optical blur falloff without harsh cutout borders.
   - If a "Background" layer exists, grade it to separate it from the subject (darker, cooler, creamy bokeh) to create a high-end 3D medium-format look.
5. 🌟 ACTIVE COMPOSITION, ANATOMY INTEGRITY & GAZE LEAD ROOM (MANDATORY STRAIGHTEN & CROP):
   - You MUST actively evaluate composition, vertical balance, and edge distractions on EVERY photo!
   - Level tilted horizons/lines using `"straighten": <tenths of a degree>` (e.g. `-15` for -1.5°, `+20` for +2.0°).
   - **ANATOMY INTEGRITY**: NEVER crop through fingers, hands, wrists, arms, feet, or ears of the child! Keep the whole body intact with comfortable breathing room.
   - **DIRECTION OF GAZE & LEAD ROOM**: If the child is looking left, keep `crop_left` at 0..2% and crop from the right (`crop_right: 8..25%`) so the child sits on the right third looking into open space. If looking right, do the opposite.
   - **VERTICAL BALANCE & FURNITURE CROP**:
     * If a seated child/subject has too much empty armchair cushion, couch, bed, or floor underneath them, crop aggressively from the bottom (`"crop_bottom": 10..25%`) so the child is not pushed awkwardly to the top!
     * Adaptively frame headroom: on vertical photos of seated subjects with excess space above, crop the top (`"crop_top": 15..32%`) to place eyes on the upper third and keep the entire subject body/hands intact at the bottom (`"crop_bottom": 0..4%`).
   - **CLEAN EDGES**:
     * Crop out adult hands, ribbons, studio props, or cluttered periphery (`"crop_left"`, `"crop_right"`, `"crop_top"`).
6. 🌟 USER NATURAL LANGUAGE REQUESTS & DIRECTIVES (HIGHEST OVERRIDING PRIORITY):
   - When the user provides a specific prompt in any language (English, Ukrainian, etc.), interpret the creative intent and fulfill it with utmost technical and artistic accuracy:
     * "Focus on face" / "Focus on eyes": Locate the person's face/eyes in image percentage coordinates [x, y] (0-100), and set `"bokeh": { "enabled": true, "focus_point": [x, y], "radius": 40, "focal_range": 25, "subject_influence": 0 }`.
     * "Blur background" / "3D Bokeh" / "Open aperture" / "Creamy background": Enable 3D Bokeh with `"bokeh": { "enabled": true, "focus_point": [x, y], "radius": 45..65, "focal_range": 20..30, "subject_influence": 0 }`.
     * "Less blur" / "Sharper background" / "Close aperture" / "Deep focus": Set `"bokeh": { "enabled": true, "focus_point": [x, y], "radius": 15..25, "focal_range": 30..40, "subject_influence": 0 }`.
     * "Smart Crop" / "Balance frame" / "Clean edges": Evaluate subject placement: if subject is seated low in frame, crop excess top ceiling/props (`"crop_top": 15..32%`) and keep whole body; if subject is standing with excess empty floor, trim bottom; leave lead room in gaze direction and level tilted horizons.
     * "Warm skin" / "Golden hour" / "Glowing skin tones": Increase Temperature +8..+16, Tint +2..+6, Vibrance +15..+25, and brighten Orange in Color Equalizer.
     * "3D Volume" / "More contrast" / "Painterly depth": Increase Base Curve (40-55), Contrast (115-125), Black Level (10-18), and Local Contrast (20-35).
7. 🌟 SUMMARY EXPLANATION:
   - Always include an "explanation" key describing your fine-art aesthetic choices.
   - Output MUST be valid JSON inside a ```json ``` markdown block.
"""

def sanitize_equalizer_ints(ints_list: Optional[Any]) -> list:
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
    import copy
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

def sanitize_state_equalizers(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes all Color Equalizer and Shadow Color Equalizer nodes across the entire state dictionary
    (top-level, in layers, and within layer settings).
    """
    if not isinstance(state, dict):
        return state

    for eq_key in ['color_equalizer', 'shadow_color_equalizer']:
        if eq_key in state and isinstance(state[eq_key], dict):
            state[eq_key] = sanitize_equalizer_dict(state[eq_key])

    if 'layers' in state and isinstance(state['layers'], list):
        for layer in state['layers']:
            if isinstance(layer, dict):
                for eq_key in ['color_equalizer', 'shadow_color_equalizer']:
                    if eq_key in layer and isinstance(layer[eq_key], dict):
                        layer[eq_key] = sanitize_equalizer_dict(layer[eq_key])
                    if 'settings' in layer and isinstance(layer['settings'], dict) and eq_key in layer['settings'] and isinstance(layer['settings'][eq_key], dict):
                        layer['settings'][eq_key] = sanitize_equalizer_dict(layer['settings'][eq_key])

    return state

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extracts and parses JSON from Gemini's response text, automatically synchronizing cyclic equalizers."""
    if not text:
        return None
        
    # Try finding markdown ```json ... ``` block
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            return sanitize_state_equalizers(data)
        except json.JSONDecodeError:
            pass
            
    # Try finding first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = text[first_brace:last_brace+1]
        try:
            data = json.loads(json_str)
            return sanitize_state_equalizers(data)
        except json.JSONDecodeError:
            pass
            
    return None

def pil_to_jpeg_part(image: Any, quality: int = 85, max_size: int = 1024) -> Any:
    """Converts a PIL Image to a lightweight compressed JPEG Part for Gemini API."""
    if image is None:
        return None
    try:
        from PIL import Image
        import io
        if isinstance(image, Image.Image):
            img_copy = image.copy()
            if img_copy.mode not in ('RGB', 'L'):
                img_copy = img_copy.convert('RGB')
            if max_size and (img_copy.width > max_size or img_copy.height > max_size):
                img_copy.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img_copy.save(buf, format='JPEG', quality=quality)
            if types and hasattr(types, 'Part'):
                return types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg')
    except Exception as e:
        print(f"Note: pil_to_jpeg_part conversion fallback: {e}")
    return image


class GeminiAssistantService:
    """
    High-performance service for communicating with Gemini 3.8 Flash using Google GenAI SDK.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = "gemini-3.8-flash"

    def is_available(self) -> Tuple[bool, str]:
        if not GENAI_AVAILABLE:
            return False, "google-genai SDK is not installed."
        if not self.api_key or self.api_key == "your_api_key_here":
            return False, "GEMINI_API_KEY is not configured."
        return True, "Ready"

    def get_client(self, timeout: float = 180.0):
        if not GENAI_AVAILABLE:
            raise RuntimeError("google-genai library is not installed.")
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY is empty. Please set it in .env or AI Assistant Dock.")
        
        http_opts = None
        if types and hasattr(types, 'HttpOptions'):
            try:
                import httpx
                custom_http = httpx.Client(
                    timeout=httpx.Timeout(float(timeout), connect=30.0, read=float(timeout), write=float(timeout), pool=30.0),
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                )
                http_opts = types.HttpOptions(httpx_client=custom_http)
            except Exception:
                http_opts = None
        return genai.Client(api_key=key, http_options=http_opts) if http_opts is not None else genai.Client(api_key=key)

    def create_config(
        self,
        thinking_level: Any = "medium",
        max_output_tokens: int = 65536,
        safety_level: str = "BLOCK_NONE",
        **kwargs
    ) -> Any:
        if not GENAI_AVAILABLE:
            return None
            
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=safety_level,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=safety_level,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=safety_level,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=safety_level,
            ),
        ]
        
        # Validation & conversion: Gemini 3.8 Flash requires thinking_level in ("low", "medium", "high")
        # Handle legacy integer thinking_budget passed via kwargs or thinking_level
        lvl = thinking_level
        if "thinking_budget" in kwargs and (lvl is None or lvl == "medium"):
            lvl = kwargs["thinking_budget"]

        if isinstance(lvl, (int, float)):
            if lvl <= 512:
                norm_level = "low"
            elif lvl > 2048:
                norm_level = "high"
            else:
                norm_level = "medium"
        else:
            lvl_str = str(lvl).lower().strip() if lvl else "medium"
            if "low" in lvl_str:
                norm_level = "low"
            elif "high" in lvl_str:
                norm_level = "high"
            else:
                norm_level = "medium"

        thinking_cfg = types.ThinkingConfig(
            thinking_level=norm_level,
            include_thoughts=True
        )
        
        # Note: Temperature is deprecated / ignored in Gemini 3.8 Flash; omitted to avoid anomalies
        return types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            system_instruction=SYSTEM_INSTRUCTION,
            thinking_config=thinking_cfg,
            safety_settings=safety_settings
        )

    def build_user_prompt(
        self,
        user_request: str = "",
        current_state: Optional[Dict[str, Any]] = None,
        is_auto_correct: bool = False,
        pass_num: int = 1,
        total_passes: int = 1
    ) -> str:
        req_clean = user_request.strip() if user_request else ""
        user_instr = f"\nUSER INSTRUCTIONS / RETOUCHING DIRECTIVES:\n{req_clean}\n" if req_clean else ""

        if is_auto_correct:
            if pass_num > 1:
                prompt = (
                    f"This is Pass {pass_num} of iterative auto-correction (out of max {total_passes} passes).\n"
                    "Here is the updated photo with your previous adjustments applied.\n"
                    "Carefully inspect the image with high scrutiny:\n"
                    "1. 🌟 CONVERGENCE & STABILITY: If the photo already looks well-balanced, clean, and properly graded, MAINTAIN the current settings so the auto-tuning process converges cleanly. DO NOT make arbitrary micro-adjustments or micro-crop changes.\n"
                    "2. 🌟 FRAMING STABILITY: Crop and straighten were already established in Pass 1. Keep the previous crop values unless an obvious compositional defect remains.\n"
                    "3. 🌟 WHITE BALANCE & SKIN PURITY: Check for remaining yellow/green casts. If skin still looks yellow or whites look warm, cool Temperature further (-30 to -110) and increase Tint (+15 to +50).\n"
                    "4. 🌟 FACES & 3D VOLUME: Ensure natural 3D facial contours and shadow roll-off are preserved.\n"
                    "5. Output the refined parameters or keep previous values if satisfied.\n"
                    f"{user_instr}\n"
                    "Return the exact JSON configuration."
                )
            else:
                prompt = (
                    "Please analyze this photo comprehensively and perform professional Fine-Art AI Auto-Correction.\n"
                    "🎯 CRITICAL MANDATORY DIRECTIVE: PROFESSIONALLY ADJUST THE WHITE BALANCE SO THE IMAGE LOOKS LIKE A PRESTIGIOUS STUDIO PHOTOGRAPH WITH FLAWLESS, PRISTINE NATURAL SKIN TONES AND ABSOLUTELY ZERO BLOWN-OUT HIGHLIGHTS OR HARSH HOTSPOTS ON FACES!\n"
                    "• Faces & Skin Tones are the absolute #1 priority! Adjust White Balance (Temperature & Tint) with studio precision. In warm indoor lighting, aggressively cool down Temperature (-30 to -120) and adjust Tint (+15 to +50) to neutralize all yellow/orange/green skin casts and achieve clean porcelain skin. Protect highlights on foreheads, cheeks, and noses with Highlight Recovery (15-35%) and Highlights (0 to -25) so there are NO washed-out hotspots.\n"
                    "• 🌟 3D Facial Volume: Preserve natural 3D facial contours (DO NOT FLATTEN FACES by over-lifting midtones; keep Tone Equalizer nodes 1-4 gentle between 48-54 max).\n"
                    "• Crisp clarity on eyes, eyelashes, and hair (Local Contrast 18-32, Sharpen 20-32, rich Contrast 108-122, Black Level 6-16).\n"
                    "• Provide Base Layer adjustments with rich Contrast (110-125), Black Level (8-18), and Local Contrast (18-35) for tactile 3D volume.\n"
                    "• If a Background layer exists, deepen/cool it for spatial depth separation.\n"
                    "• 🌟 3D BOKEH SIMULATION & DEPTH MAP AESTHETICS (ENABLED WHEN APPROPRIATE):\n"
                    "  - 🎯 CRITICAL FACE FOCUS: When enabling 3D Bokeh, ALWAYS lock 'focus_point' on the main hero's face/eyes [x_pct, y_pct] (e.g. [50, 30] where y is % from top). The main subject's face, eyes, hair, and expression MUST ALWAYS REMAIN 100% TACK-SHARP!\n"
                    "  - 📏 COMFORTABLE FOCAL DEPTH OF FIELD: Set 'focal_range' to 25..35 so the entire subject (ears, hair, dress, and hands) is protected from accidental blur.\n"
                    "  - 🔍 EDGE QUALITY & DEPTH DISCONTINUITIES: Carefully inspect the boundaries between near foreground and far background. Because depth maps are AI-generated, complex flyaway hair, fine fingers, or sharp furniture silhouettes can create edge halos if blur is too harsh. Keep aperture radius elegant and creamy ('radius': 30..45) rather than extreme to guarantee clean, professional edge transitions!\n"
                    "  - 💾 Dual Export: Remember that the app exports both versions (with Bokeh and clean optical original), so ensure the Bokeh version looks like a prestigious f/1.4 - f/2.0 portrait lens.\n"
                    "• 🌟 GEOMETRY & BALANCED COMPOSITION: Level tilted lines with 'straighten'. Evaluate framing with 'crop': NEVER cut through child's hands, arms, or feet! Leave more space in the direction of their gaze (if child looks left, keep crop_left 0-2% and crop from right: 8-25%). If seated on armchair/furniture with heavy empty bottom cushions, crop from bottom ('crop_bottom': 10-25%) to balance the child vertically; protect comfortable headroom ('crop_top': 0-5%).\n"
                    "• 🌟 MULTI-CROP PRESETS: In addition to the primary 'crop', always provide 2 to 3 alternative artistic crop presets in 'crop_presets' array (e.g. Hero Portrait 4:5, Emotional Square 1:1, Story Scene 4:3 / 3:2, Full Frame) with descriptive names so the user can export all variations at once!\n"
                    f"{user_instr}\n"
                    "Return the exact JSON configuration."
                )
        else:
            if pass_num > 1:
                prompt = (
                    f"This is Pass {pass_num} of iterative styling for user request.\n"
                    f"{user_instr}\n"
                    "Inspect the updated photo and refine the adjustments to achieve the ultimate artistic vision.\n"
                    "Return the exact JSON configuration."
                )
            else:
                prompt = (
                    "Please analyze the photo in relation to the user's creative request. Apply aesthetic and technical adjustments "
                    "across the pipeline (Basic Adjustments, Tone/Color Equalizers, Procedural/Depth Masks, 3D Bokeh Simulation, and Global settings) to achieve this exact look.\n"
                    f"{user_instr}\n"
                    "Return the exact JSON configuration."
                )

        if current_state:
            layers_summary = []
            for i, l in enumerate(current_state.get('layers', [])):
                name = l.get('name', f"Layer {i}")
                st = l.get('settings', {})
                b = st.get('basic_adjustments', {})
                exp = l.get('exposure', st.get('exposure', b.get('exposure', 0)))
                temp = l.get('temperature', st.get('temperature', b.get('temperature', 0)))
                tint = l.get('tint', st.get('tint', b.get('tint', 0)))
                cont = l.get('contrast', st.get('contrast', b.get('contrast', 100)))
                layers_summary.append(f"• {name}: exposure={exp}, temperature={temp}, tint={tint}, contrast={cont}")

            geo_summary = ""
            if 'geometry_crop' in current_state:
                g = current_state['geometry_crop']
                geo_summary = f"• Geometry: straighten={g.get('straighten', 0)/10.0:.1f}°, crop=[L:{g.get('crop_left',0)/10.0:.1f}%, R:{g.get('crop_right',0)/10.0:.1f}%, T:{g.get('crop_top',0)/10.0:.1f}%, B:{g.get('crop_bottom',0)/10.0:.1f}%]"

            # Check aspect ratio constraint
            asp_name = current_state.get('aspect_name', current_state.get('geometry_crop', {}).get('aspect_name', 'Free'))
            asp_ratio = current_state.get('aspect_ratio', current_state.get('geometry_crop', {}).get('aspect_ratio'))
            if asp_name and not asp_name.startswith("Free"):
                asp_info = f"locked to '{asp_name}'"
                if asp_ratio and isinstance(asp_ratio, (int, float)) and asp_ratio > 0:
                    asp_info += f" (W:H ratio {asp_ratio:.3f} or inverted {1.0/asp_ratio:.3f})"
                prompt += (
                    f"\n📐 MANDATORY ASPECT RATIO CONSTRAINT: {asp_info}\n"
                    f"• The user has locked the crop aspect ratio constraint to {asp_name}.\n"
                    f"• Your crop values MUST strictly produce this aspect ratio (either in Portrait or Landscape orientation).\n"
                )

            state_str = f"\nCURRENT IMAGE CONTEXT (Pass {pass_num-1} State):\n" + "\n".join(layers_summary)
            if geo_summary:
                state_str += "\n" + geo_summary
            prompt += f"\n{state_str}\n"

        return prompt

def parse_aspect_ratio_text(text: str, img_w: int = 0, img_h: int = 0) -> Tuple[Optional[float], str]:
    """Parses standard aspect ratio text labels into a numerical float ratio (W/H) and clean name.
    Properly handles both Landscape (ratio > 1.0) and Portrait (ratio < 1.0) formats,
    using image dimensions/orientation to resolve ambiguous compound labels like '9:16 / 16:9'."""
    if not text or not isinstance(text, str):
        return None, "Free"
    clean_txt = text.strip()
    if clean_txt.startswith("Free"):
        return None, "Free"
    if clean_txt.startswith("Original") or clean_txt.startswith("Sensor"):
        if img_w > 0 and img_h > 0:
            return float(img_w) / float(img_h), f"Original ({img_w}:{img_h})"
        return None, "Original (Sensor)"
    if "1:1" in clean_txt or "Square" in clean_txt:
        return 1.0, "1:1 (Square)"

    txt_lower = clean_txt.lower()
    has_landscape_keyword = any(k in txt_lower for k in ["landscape", "альбом", "horizontal", "wide", "cinematic", "tv"])
    has_portrait_keyword = any(k in txt_lower for k in ["portrait", "книжк", "vertical", "reel", "story", "stories"])
    is_img_landscape = (img_w >= img_h) if (img_w > 0 and img_h > 0) else True

    # Check for 16:9 / 9:16
    has_16_9 = "16:9" in clean_txt or "16/9" in clean_txt or "16x9" in clean_txt or "16_9" in clean_txt
    has_9_16 = "9:16" in clean_txt or "9/16" in clean_txt or "9x16" in clean_txt or "9_16" in clean_txt
    if has_16_9 and not has_9_16:
        return 16.0 / 9.0, "16:9 (Landscape)"
    elif has_9_16 and not has_16_9:
        return 9.0 / 16.0, "9:16 (Portrait)"
    elif has_16_9 and has_9_16:
        if has_portrait_keyword and not has_landscape_keyword:
            return 9.0 / 16.0, "9:16 (Portrait)"
        elif has_landscape_keyword and not has_portrait_keyword:
            return 16.0 / 9.0, "16:9 (Landscape)"
        elif is_img_landscape:
            return 16.0 / 9.0, "16:9 (Landscape)"
        else:
            return 9.0 / 16.0, "9:16 (Portrait)"

    # Check for 4:3 / 3:4
    has_4_3 = "4:3" in clean_txt or "4/3" in clean_txt or "4x3" in clean_txt or "4_3" in clean_txt
    has_3_4 = "3:4" in clean_txt or "3/4" in clean_txt or "3x4" in clean_txt or "3_4" in clean_txt
    if has_4_3 and not has_3_4:
        return 4.0 / 3.0, "4:3 (Landscape)"
    elif has_3_4 and not has_4_3:
        return 3.0 / 4.0, "3:4 (Portrait)"
    elif has_4_3 and has_3_4:
        if has_portrait_keyword and not has_landscape_keyword:
            return 3.0 / 4.0, "3:4 (Portrait)"
        elif has_landscape_keyword and not has_portrait_keyword:
            return 4.0 / 3.0, "4:3 (Landscape)"
        elif is_img_landscape:
            return 4.0 / 3.0, "4:3 (Landscape)"
        else:
            return 3.0 / 4.0, "3:4 (Portrait)"

    # Check for 3:2 / 2:3
    has_3_2 = "3:2" in clean_txt or "3/2" in clean_txt or "3x2" in clean_txt or "3_2" in clean_txt
    has_2_3 = "2:3" in clean_txt or "2/3" in clean_txt or "2x3" in clean_txt or "2_3" in clean_txt
    if has_3_2 and not has_2_3:
        return 3.0 / 2.0, "3:2 (Landscape)"
    elif has_2_3 and not has_3_2:
        return 2.0 / 3.0, "2:3 (Portrait)"
    elif has_3_2 and has_2_3:
        if has_portrait_keyword and not has_landscape_keyword:
            return 2.0 / 3.0, "2:3 (Portrait)"
        elif has_landscape_keyword and not has_portrait_keyword:
            return 3.0 / 2.0, "3:2 (Landscape)"
        elif is_img_landscape:
            return 3.0 / 2.0, "3:2 (Landscape)"
        else:
            return 2.0 / 3.0, "2:3 (Portrait)"

    # Check for 5:4 / 4:5
    has_5_4 = "5:4" in clean_txt or "5/4" in clean_txt or "5x4" in clean_txt or "5_4" in clean_txt
    has_4_5 = "4:5" in clean_txt or "4/5" in clean_txt or "4x5" in clean_txt or "4_5" in clean_txt
    if has_5_4 and not has_4_5:
        return 5.0 / 4.0, "5:4 (Landscape)"
    elif has_4_5 and not has_5_4:
        return 4.0 / 5.0, "4:5 (Portrait)"
    elif has_5_4 and has_4_5:
        if has_portrait_keyword and not has_landscape_keyword:
            return 4.0 / 5.0, "4:5 (Portrait)"
        elif has_landscape_keyword and not has_portrait_keyword:
            return 5.0 / 4.0, "5:4 (Landscape)"
        elif is_img_landscape:
            return 5.0 / 4.0, "5:4 (Landscape)"
        else:
            return 4.0 / 5.0, "4:5 (Portrait)"

    # Check for 7:5 / 5:7
    has_7_5 = "7:5" in clean_txt or "7/5" in clean_txt or "7x5" in clean_txt
    has_5_7 = "5:7" in clean_txt or "5/7" in clean_txt or "5x7" in clean_txt
    if has_7_5 and not has_5_7:
        return 7.0 / 5.0, "7:5 (Landscape)"
    elif has_5_7 and not has_7_5:
        return 5.0 / 7.0, "5:7 (Portrait)"
    elif has_7_5 and has_5_7:
        return (7.0 / 5.0, "7:5 (Landscape)") if is_img_landscape else (5.0 / 7.0, "5:7 (Portrait)")

    # Check for 14:11 / 11:14
    has_14_11 = "14:11" in clean_txt or "14/11" in clean_txt or "14x11" in clean_txt
    has_11_14 = "11:14" in clean_txt or "11/14" in clean_txt or "11x14" in clean_txt
    if has_14_11 and not has_11_14:
        return 14.0 / 11.0, "14:11 (Landscape)"
    elif has_11_14 and not has_14_11:
        return 11.0 / 14.0, "11:14 (Portrait)"
    elif has_14_11 and has_11_14:
        return (14.0 / 11.0, "14:11 (Landscape)") if is_img_landscape else (11.0 / 14.0, "11:14 (Portrait)")

    # Check for 3.5:4.5 / 4.5:3.5 (Passport / ID)
    if "3.5:4.5" in clean_txt or "4.5:3.5" in clean_txt:
        return (3.5 / 4.5, "3.5:4.5 (Portrait)") if not has_landscape_keyword else (4.5 / 3.5, "4.5:3.5 (Landscape)")

    # Generic ratio regex fallback (e.g. 21:9 or 16:10 or 1.85:1)
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[:/]\s*(\d+(?:\.\d+)?)", clean_txt)
    if m:
        w_part, h_part = float(m.group(1)), float(m.group(2))
        if h_part > 0:
            ratio = w_part / h_part
            tag = "Landscape" if ratio >= 1.0 else "Portrait"
            return ratio, f"{m.group(1)}:{m.group(2)} ({tag})"

    return None, clean_txt

def enforce_crop_aspect_ratio(
    crop_l: int,
    crop_r: int,
    crop_t: int,
    crop_b: int,
    img_w: int,
    img_h: int,
    target_ratio: Optional[float],
    allow_orientation_flip: bool = True
) -> Tuple[int, int, int, int]:
    """
    Enforces the target aspect ratio constraint on crop slider coordinates (0..1000).
    Preserves the composition/subject center (cx, cy) and dynamically supports orientation swap
    (Portrait <-> Landscape) if allow_orientation_flip is True.
    """
    if target_ratio is None or target_ratio <= 0 or img_w <= 0 or img_h <= 0:
        return crop_l, crop_r, crop_t, crop_b

    import math
    l = np.clip(float(crop_l) / 1000.0, 0.0, 0.49)
    r = np.clip(float(crop_r) / 1000.0, 0.0, 0.49)
    t = np.clip(float(crop_t) / 1000.0, 0.0, 0.49)
    b = np.clip(float(crop_b) / 1000.0, 0.0, 0.49)

    w_norm = max(0.02, 1.0 - l - r)
    h_norm = max(0.02, 1.0 - t - b)

    w_px = float(img_w) * w_norm
    h_px = float(img_h) * h_norm
    current_ar = w_px / max(1.0, h_px)

    desired_r = target_ratio
    if allow_orientation_flip:
        inv_r = 1.0 / target_ratio
        if abs(math.log(current_ar) - math.log(inv_r)) < abs(math.log(current_ar) - math.log(target_ratio)):
            desired_r = inv_r

    cx_norm = l + w_norm / 2.0
    cy_norm = t + h_norm / 2.0

    if current_ar > desired_r:
        w_px_new = h_px * desired_r
        h_px_new = h_px
    else:
        w_px_new = w_px
        h_px_new = w_px / desired_r

    if w_px_new > float(img_w):
        w_px_new = float(img_w)
        h_px_new = w_px_new / desired_r
    if h_px_new > float(img_h):
        h_px_new = float(img_h)
        w_px_new = h_px_new * desired_r

    w_norm_new = w_px_new / float(img_w)
    h_norm_new = h_px_new / float(img_h)

    cx_norm = np.clip(cx_norm, w_norm_new / 2.0, 1.0 - w_norm_new / 2.0)
    cy_norm = np.clip(cy_norm, h_norm_new / 2.0, 1.0 - h_norm_new / 2.0)

    l_new = cx_norm - w_norm_new / 2.0
    r_new = 1.0 - (cx_norm + w_norm_new / 2.0)
    t_new = cy_norm - h_norm_new / 2.0
    b_new = 1.0 - (cy_norm + h_norm_new / 2.0)

    cl = int(round(l_new * 1000.0))
    cr = int(round(r_new * 1000.0))
    ct = int(round(t_new * 1000.0))
    cb = int(round(b_new * 1000.0))

    # If any margin exceeds slider max (480), shift into available opposite margin to preserve total span & exact aspect ratio
    if cl > 480:
        diff = cl - 480
        cl = 480
        cr = max(0, cr - diff)
    elif cr > 480:
        diff = cr - 480
        cr = 480
        cl = max(0, cl - diff)

    if ct > 480:
        diff = ct - 480
        ct = 480
        cb = max(0, cb - diff)
    elif cb > 480:
        diff = cb - 480
        cb = 480
        ct = max(0, ct - diff)

    return (
        int(np.clip(cl, 0, 480)),
        int(np.clip(cr, 0, 480)),
        int(np.clip(ct, 0, 480)),
        int(np.clip(cb, 0, 480))
    )

def apply_golden_headroom_rule(
    ymin: float, xmin: float, ymax: float, xmax: float,
    head_crown_y: float, eyes_y: float,
    img_w: int, img_h: int, target_aspect: Optional[float],
    preset_id: str = "portrait",
    subject_center_x: Optional[float] = None,
    chin_y: Optional[float] = None,
    facing_direction: Optional[str] = None,
    gaze_direction: Optional[str] = None
) -> Tuple[int, int, int, int]:
    """
    Guarantees professional fine-art portrait framing according to People Photography Cropping Guidelines:
    1. STRICT OCULAR AXIS PROTECTION: Frame edge NEVER cuts through eyes or forehead (ct <= eyes_y - 80).
    2. Headroom above head_crown_y is bounded between 8% and 12% of crop height for standard portraits.
    3. For tight beauty close-ups: allows cranial apex / mid-hair crop, eyes locked to upper third (~32%), chin fully clear.
    4. MANDIBULAR BASE PROTECTION: The chin and jawline are NEVER clipped or truncated (1000 - cb >= chin_y + 40).
    5. STRICT LEAD ROOM / GAZE & BODY ORIENTATION:
       - When body/gaze faces LEFT -> more space in front on the LEFT (cl <= cr).
       - When body/gaze faces RIGHT -> more space in front on the RIGHT (cr <= cl).
       - When facing CAMERA -> balanced space on both sides.
       - NEVER leave more empty space behind the subject's back than in front of their body/gaze!
    6. MATHEMATICAL ASPECT RATIO LOCK: Output is mathematically guaranteed to match target_aspect.
    """
    if preset_id == 'full_frame' or target_aspect is None or target_aspect <= 0:
        return 0, 0, 0, 0

    # Robust coordinate scale normalization (0..1 or 0..100 -> 0..1000)
    def _norm(v):
        if v is None: return None
        try:
            fv = float(v)
            if 0 < fv <= 1.0: return fv * 1000.0
            if 1.0 < fv <= 100.0: return fv * 10.0
            return fv
        except:
            return None

    head_crown_y = _norm(head_crown_y)
    eyes_y = _norm(eyes_y)
    chin_y = _norm(chin_y)
    subject_center_x = _norm(subject_center_x)
    ymin = _norm(ymin) if ymin is not None else 0.0
    xmin = _norm(xmin) if xmin is not None else 0.0
    ymax = _norm(ymax) if ymax is not None else 1000.0
    xmax = _norm(xmax) if xmax is not None else 1000.0

    pid_lower = preset_id.lower()
    is_close_up = ("closeup" in pid_lower or "close_up" in pid_lower or "beauty" in pid_lower)

    # Defaults if specific landmarks were omitted by detector
    sub_x = subject_center_x if subject_center_x is not None and subject_center_x > 0 else (xmin + xmax) / 2.0
    c_y = head_crown_y if head_crown_y is not None and head_crown_y > 0 else (ymin + 50.0)
    e_y = eyes_y if eyes_y is not None and eyes_y > 0 else (c_y + 110.0)
    ch_y = chin_y if chin_y is not None and chin_y > 0 else (e_y + 140.0)

    face_h = max(70.0, ch_y - c_y)

    # 1. Scale-dependent Headroom Allocation and framing height
    if is_close_up:
        target_headroom_ratio = 0.05
        target_h = face_h / 0.55
    elif "square" in pid_lower or "1x1" in pid_lower:
        target_headroom_ratio = 0.08
        target_h = face_h / 0.38
    elif "hero" in pid_lower or "4x5" in pid_lower or "4:5" in pid_lower:
        target_headroom_ratio = 0.09
        target_h = face_h / 0.32
    elif "3x4" in pid_lower or "3:4" in pid_lower or "classic" in pid_lower:
        target_headroom_ratio = 0.10
        target_h = face_h / 0.26
    else:
        target_headroom_ratio = 0.10
        target_h = face_h / 0.30

    # Ensure crop box encompasses chin with safe margin
    min_h_for_chin = (ch_y + 45.0 - c_y) / max(0.2, (1.0 - target_headroom_ratio))
    target_h = max(target_h, min_h_for_chin)

    # If Gemini proposed a larger context box, respect its context
    init_h = ymax - ymin
    if init_h > 200:
        target_h = max(target_h, init_h * 0.85)

    # Map aspect ratio to normalized coordinate space
    sensor_ar = float(img_w) / float(max(1, img_h))
    norm_aspect = target_aspect / sensor_ar

    target_w = target_h * norm_aspect
    if target_w > 990.0:
        target_w = 990.0
        target_h = target_w / norm_aspect
    if target_h > 990.0:
        target_h = 990.0
        target_w = target_h * norm_aspect

    # 2. Check max width allowed so that space behind back NEVER exceeds space in front
    fd = (facing_direction or gaze_direction or 'center').lower().strip()
    if 'left' in fd and sub_x is not None:
        max_allowable_w = min(990.0, 2.0 * sub_x)
        if target_w > max_allowable_w:
            target_w = max(100.0, max_allowable_w)
            target_h = target_w / norm_aspect
    elif 'right' in fd and sub_x is not None:
        max_allowable_w = min(990.0, 2.0 * (1000.0 - sub_x))
        if target_w > max_allowable_w:
            target_w = max(100.0, max_allowable_w)
            target_h = target_w / norm_aspect

    h_norm = min(990.0, max(100.0, target_h))
    w_norm = min(990.0, max(100.0, h_norm * norm_aspect))
    h_norm = w_norm / norm_aspect

    # Vertical Positioning (Headroom, Ocular Axis Protection, Chin Protection)
    if is_close_up:
        y0 = e_y - 0.32 * h_norm
    else:
        y0 = c_y - target_headroom_ratio * h_norm

    # Hard Ocular Axis Protection: frame edge can NEVER cut into eyes
    if y0 > (e_y - 80.0):
        y0 = e_y - 80.0

    # Crown Protection for Standard Portraits: keep headroom above hair crown
    if not is_close_up and y0 > (c_y - 0.05 * h_norm):
        y0 = c_y - 0.06 * h_norm

    # Chin Protection: frame bottom must clear chin
    y1 = y0 + h_norm
    if y1 < (ch_y + 40.0):
        shift_down = (ch_y + 40.0) - y1
        y0 += shift_down
        y1 += shift_down

    # Keep vertical window inside canvas [0, 1000]
    if y0 < 0:
        y0 = 0.0
        y1 = y0 + h_norm
    if y1 > 1000.0:
        y0 = max(0.0, 1000.0 - h_norm)
        y1 = y0 + h_norm

    # 3. Horizontal Positioning with Strict Lead Room / Facing Direction
    if 'left' in fd:
        # Facing LEFT -> subject placed on the right, dedicating 58-62% space on the LEFT (in front of chest/gaze)
        lead_ratio = 0.60
    elif 'right' in fd:
        # Facing RIGHT -> subject placed on the left, dedicating 58-62% space on the RIGHT (in front of chest/gaze)
        lead_ratio = 0.40
    else:
        # Centered facing
        lead_ratio = 0.50

    x0 = sub_x - lead_ratio * w_norm
    x1 = x0 + w_norm

    if x0 < 0:
        x0 = 0.0
        x1 = x0 + w_norm
    if x1 > 1000.0:
        x0 = max(0.0, 1000.0 - w_norm)
        x1 = x0 + w_norm

    # Hard Invariant: NEVER have more space behind subject's back than in front of body/gaze
    ml = sub_x - x0 # space on the left
    mr = x1 - sub_x # space on the right
    if 'left' in fd and ml < mr:
        shift = min(x0, (mr - ml) / 2.0)
        x0 -= shift
        x1 -= shift
        ml = sub_x - x0
        mr = x1 - sub_x
        if mr > ml:
            x1 = sub_x + ml
            w_norm = x1 - x0
            h_norm = w_norm / norm_aspect
            y1 = min(1000.0, y0 + h_norm)
            y0 = max(0.0, y1 - h_norm)
    elif 'right' in fd and mr < ml:
        shift = min(1000.0 - x1, (ml - mr) / 2.0)
        x0 += shift
        x1 += shift
        ml = sub_x - x0
        mr = x1 - sub_x
        if ml > mr:
            x0 = sub_x - mr
            w_norm = x1 - x0
            h_norm = w_norm / norm_aspect
            y1 = min(1000.0, y0 + h_norm)
            y0 = max(0.0, y1 - h_norm)

    cl = int(round(x0))
    cr = int(round(1000.0 - x1))
    ct = int(round(y0))
    cb = int(round(1000.0 - y1))

    # 4. Absolute mathematical lock to target aspect ratio
    return enforce_crop_aspect_ratio(cl, cr, ct, cb, img_w, img_h, target_aspect, allow_orientation_flip=False)


def blend_state_dicts(
    prev_dict: Dict[str, Any],
    new_dict: Dict[str, Any],
    alpha: float,
    lock_depth_bounds: bool = False
) -> Dict[str, Any]:
    import copy
    if not isinstance(prev_dict, dict):
        return copy.deepcopy(new_dict) if isinstance(new_dict, dict) else {}
    if not isinstance(new_dict, dict):
        return copy.deepcopy(prev_dict)

    blended = copy.deepcopy(prev_dict)

    # 1. Update/blend top-level keys from new_dict
    for key, val in new_dict.items():
        if key not in prev_dict:
            blended[key] = copy.deepcopy(val)
            continue
        prev_val = prev_dict[key]

        if isinstance(val, (int, float)) and not isinstance(val, bool) and isinstance(prev_val, (int, float)) and not isinstance(prev_val, bool):
            blended[key] = type(val)(round(prev_val + (val - prev_val) * alpha))
        elif isinstance(val, dict) and isinstance(prev_val, dict):
            blended[key] = blend_state_dicts(prev_val, val, alpha, lock_depth_bounds=lock_depth_bounds)
        elif isinstance(val, list) and isinstance(prev_val, list):
            if len(val) == len(prev_val):
                blended_list = []
                for p_item, n_item in zip(prev_val, val):
                    if isinstance(n_item, (int, float)) and not isinstance(n_item, bool) and isinstance(p_item, (int, float)) and not isinstance(p_item, bool):
                        blended_list.append(type(n_item)(round(p_item + (n_item - p_item) * alpha)))
                    elif isinstance(n_item, dict) and isinstance(p_item, dict):
                        blended_list.append(blend_state_dicts(p_item, n_item, alpha, lock_depth_bounds=lock_depth_bounds))
                    else:
                        blended_list.append(n_item)
                blended[key] = blended_list
            else:
                blended[key] = copy.deepcopy(val)
        else:
            blended[key] = copy.deepcopy(val)

    # 2. Multi-layer handling & Basic Adjustments synchronization:
    BASIC_KEYS = [
        'exposure', 'contrast', 'gamma', 'base_curve', 'temperature', 'tint',
        'highlights', 'shadows', 'whites', 'black_level', 'highlight_recovery',
        'local_contrast', 'global_chroma', 'global_vibrance',
        'denoise_luma', 'denoise_chroma', 'sharpen', 'sharpen_threshold', 'vignette'
    ]
    EQUALIZER_KEYS = ['tone_equalizer', 'color_equalizer', 'shadow_color_equalizer']

    def _extract_val(src, key):
        if not isinstance(src, dict):
            return None
        if key in src:
            return src[key]
        if 'settings' in src and isinstance(src['settings'], dict):
            if key in src['settings']:
                return src['settings'][key]
            if 'basic_adjustments' in src['settings'] and isinstance(src['settings']['basic_adjustments'], dict):
                if key in src['settings']['basic_adjustments']:
                    return src['settings']['basic_adjustments'][key]
        if 'basic_adjustments' in src and isinstance(src['basic_adjustments'], dict):
            if key in src['basic_adjustments']:
                return src['basic_adjustments'][key]
        return None

    if 'layers' in blended and isinstance(blended['layers'], list) and len(blended['layers']) > 0:
        if 'layers' in new_dict and isinstance(new_dict['layers'], list):
            for l_idx, n_layer in enumerate(new_dict['layers']):
                if l_idx >= len(blended['layers']) or not isinstance(n_layer, dict):
                    break
                b_layer = blended['layers'][l_idx]
                p_layer = prev_dict['layers'][l_idx] if ('layers' in prev_dict and l_idx < len(prev_dict['layers']) and isinstance(prev_dict['layers'][l_idx], dict)) else {}
                
                if 'settings' not in b_layer or not isinstance(b_layer['settings'], dict):
                    b_layer['settings'] = copy.deepcopy(p_layer.get('settings', {}))
                if 'basic_adjustments' not in b_layer['settings'] or not isinstance(b_layer['settings']['basic_adjustments'], dict):
                    b_layer['settings']['basic_adjustments'] = {}

                # Layer active / enabled flag
                if 'active' in n_layer:
                    b_layer['active'] = bool(n_layer['active'])
                elif 'enabled' in n_layer:
                    b_layer['active'] = bool(n_layer['enabled'])

                if b_layer.get('name', '').lower() == 'background':
                    if 'background_active' in new_dict:
                        b_layer['active'] = bool(new_dict['background_active'])
                    elif 'background_enabled' in new_dict:
                        b_layer['active'] = bool(new_dict['background_enabled'])

                p_settings = p_layer.get('settings', {}) if isinstance(p_layer, dict) else {}

                for b_key in BASIC_KEYS:
                    new_v = _extract_val(n_layer, b_key)
                    if new_v is not None and isinstance(new_v, (int, float)) and not isinstance(new_v, bool):
                        prev_v = _extract_val(p_layer, b_key)
                        if prev_v is None:
                            prev_v = _extract_val(p_settings, b_key)
                        prev_num = float(prev_v) if (prev_v is not None and isinstance(prev_v, (int, float)) and not isinstance(prev_v, bool)) else 0.0
                        blended_v = type(new_v)(round(prev_num + (float(new_v) - prev_num) * alpha))
                        b_layer[b_key] = blended_v
                        b_layer['settings'][b_key] = blended_v
                        b_layer['settings']['basic_adjustments'][b_key] = blended_v

                for eq_key in EQUALIZER_KEYS:
                    n_eq = _extract_val(n_layer, eq_key)
                    if isinstance(n_eq, dict):
                        p_eq = _extract_val(p_layer, eq_key) or _extract_val(p_settings, eq_key) or {}
                        blended_eq = blend_state_dicts(p_eq if isinstance(p_eq, dict) else {}, n_eq, alpha, lock_depth_bounds=lock_depth_bounds)
                        b_layer[eq_key] = blended_eq
                        b_layer['settings'][eq_key] = copy.deepcopy(blended_eq)
        else:
            base_layer = blended['layers'][0]
            if 'settings' not in base_layer or not isinstance(base_layer['settings'], dict):
                base_layer['settings'] = {}
            if 'basic_adjustments' not in base_layer['settings'] or not isinstance(base_layer['settings']['basic_adjustments'], dict):
                base_layer['settings']['basic_adjustments'] = {}

            p_layer = prev_dict['layers'][0] if ('layers' in prev_dict and len(prev_dict['layers']) > 0 and isinstance(prev_dict['layers'][0], dict)) else {}
            p_settings = p_layer.get('settings', {}) if isinstance(p_layer, dict) else {}

            for b_key in BASIC_KEYS:
                new_v = _extract_val(new_dict, b_key)
                if new_v is not None and isinstance(new_v, (int, float)) and not isinstance(new_v, bool):
                    prev_v = _extract_val(prev_dict, b_key)
                    if prev_v is None:
                        prev_v = _extract_val(p_layer, b_key)
                    if prev_v is None:
                        prev_v = _extract_val(p_settings, b_key)
                    prev_num = float(prev_v) if (prev_v is not None and isinstance(prev_v, (int, float)) and not isinstance(prev_v, bool)) else 0.0
                    blended_v = type(new_v)(round(prev_num + (float(new_v) - prev_num) * alpha))
                    blended[b_key] = blended_v
                    base_layer[b_key] = blended_v
                    base_layer['settings'][b_key] = blended_v
                    base_layer['settings']['basic_adjustments'][b_key] = blended_v

            for eq_key in EQUALIZER_KEYS:
                n_eq = _extract_val(new_dict, eq_key)
                if isinstance(n_eq, dict):
                    p_eq = _extract_val(prev_dict, eq_key) or _extract_val(p_layer, eq_key) or _extract_val(p_settings, eq_key) or {}
                    blended_eq = blend_state_dicts(p_eq if isinstance(p_eq, dict) else {}, n_eq, alpha, lock_depth_bounds=lock_depth_bounds)
                    blended[eq_key] = blended_eq
                    base_layer[eq_key] = blended_eq
                    base_layer['settings'][eq_key] = copy.deepcopy(blended_eq)

        if lock_depth_bounds:
            for l_idx, l_cur in enumerate(blended['layers']):
                if l_idx < len(prev_dict.get('layers', [])):
                    l_prev = prev_dict['layers'][l_idx]
                    if l_prev.get('use_depth_mask', False):
                        l_cur['mask_depth_min'] = l_prev.get('mask_depth_min', l_cur.get('mask_depth_min', 0))
                        l_cur['mask_depth_max'] = l_prev.get('mask_depth_max', l_cur.get('mask_depth_max', 110))

    if 'geometry_crop' not in blended:
        blended['geometry_crop'] = copy.deepcopy(prev_dict.get('geometry_crop', {
            'straighten': 0, 'flip_90': 0, 'crop_left': 0, 'crop_right': 0, 'crop_top': 0, 'crop_bottom': 0
        }))
    geo = blended['geometry_crop']

    def _parse_crop_to_slider(v):
        if v is None:
            return 0
        try:
            fv = float(v)
            if 0.0 < fv <= 0.45 and isinstance(v, float):
                return int(round(fv * 1000.0))
            iv = int(round(fv))
            return iv * 10 if 0 < iv <= 45 else int(np.clip(iv, 0, 480))
        except (ValueError, TypeError):
            return 0

    def _parse_straighten_to_slider(v):
        if v is None:
            return 0
        try:
            fv = float(v)
            if -4.5 <= fv <= 4.5 and isinstance(v, float):
                return int(round(fv * 10.0))
            iv = int(round(fv))
            return int(np.clip(iv, -450, 450))
        except (ValueError, TypeError):
            return 0

    raw_straighten = None
    for src in [new_dict, new_dict.get('geometry_crop', {})]:
        if isinstance(src, dict):
            for k in ['straighten', 'rotation', 'angle', 'rotate']:
                if k in src and src[k] is not None:
                    raw_straighten = _parse_straighten_to_slider(src[k])
                    break
    if raw_straighten is None and 'layers' in new_dict and isinstance(new_dict['layers'], list) and len(new_dict['layers']) > 0:
        base_l = new_dict['layers'][0]
        if isinstance(base_l, dict):
            for k in ['straighten', 'rotation']:
                if k in base_l and base_l[k] is not None:
                    raw_straighten = _parse_straighten_to_slider(base_l[k])
                    break
    if raw_straighten is not None:
        prev_s = prev_dict.get('geometry_crop', {}).get('straighten', prev_dict.get('straighten', 0))
        diff_s = abs(raw_straighten - prev_s)
        if diff_s < 3 and prev_s != 0:
            geo['straighten'] = prev_s
        else:
            geo['straighten'] = int(round(prev_s + (raw_straighten - prev_s) * alpha))
        blended['straighten'] = geo['straighten']

    crop_extracted = {}
    if 'crop' in new_dict:
        if isinstance(new_dict['crop'], dict):
            for k, v in new_dict['crop'].items():
                target_k = f"crop_{k}" if not k.startswith("crop_") else k
                if target_k in ('crop_left', 'crop_right', 'crop_top', 'crop_bottom'):
                    crop_extracted[target_k] = _parse_crop_to_slider(v)
        elif isinstance(new_dict['crop'], (list, tuple)) and len(new_dict['crop']) == 4:
            for idx, k in enumerate(['crop_left', 'crop_right', 'crop_top', 'crop_bottom']):
                crop_extracted[k] = _parse_crop_to_slider(new_dict['crop'][idx])

    if 'geometry_crop' in new_dict and isinstance(new_dict['geometry_crop'], dict):
        for k in ['crop_left', 'crop_right', 'crop_top', 'crop_bottom', 'left', 'right', 'top', 'bottom']:
            if k in new_dict['geometry_crop']:
                target_k = f"crop_{k}" if not k.startswith("crop_") else k
                if target_k not in crop_extracted:
                    crop_extracted[target_k] = _parse_crop_to_slider(new_dict['geometry_crop'][k])

    for k, val_slider in crop_extracted.items():
        prev_c = prev_dict.get('geometry_crop', {}).get(k, 0)
        diff_c = abs(val_slider - prev_c)
        if diff_c < 30 and prev_c > 0:
            geo[k] = prev_c
        else:
            geo[k] = int(round(prev_c + (val_slider - prev_c) * alpha))

    aspect_r = prev_dict.get('geometry_crop', {}).get('aspect_ratio', prev_dict.get('aspect_ratio'))
    aspect_name = prev_dict.get('geometry_crop', {}).get('aspect_name', prev_dict.get('aspect_name', 'Free'))
    img_w = prev_dict.get('geometry_crop', {}).get('image_width', prev_dict.get('image_width', 0))
    img_h = prev_dict.get('geometry_crop', {}).get('image_height', prev_dict.get('image_height', 0))

    if (aspect_r is None or aspect_r <= 0) and aspect_name and not aspect_name.startswith("Free"):
        aspect_r, _ = parse_aspect_ratio_text(aspect_name, img_w, img_h)

    if aspect_r is not None and aspect_r > 0 and img_w > 0 and img_h > 0:
        cl = geo.get('crop_left', 0)
        cr = geo.get('crop_right', 0)
        ct = geo.get('crop_top', 0)
        cb = geo.get('crop_bottom', 0)
        geo['crop_left'], geo['crop_right'], geo['crop_top'], geo['crop_bottom'] = enforce_crop_aspect_ratio(
            cl, cr, ct, cb, img_w, img_h, aspect_r, allow_orientation_flip=True
        )

    # Multi-Crop Presets extraction, mathematical aspect ratio enforcement, and English naming
    parsed_presets = []
    raw_presets = new_dict.get('crop_presets', [])
    if not isinstance(raw_presets, list) or len(raw_presets) == 0:
        raw_presets = prev_dict.get('crop_presets', [])

    w_val = img_w if img_w > 0 else 8192
    h_val = img_h if img_h > 0 else 5464

    if isinstance(raw_presets, list) and len(raw_presets) > 0:
        for p_idx, p in enumerate(raw_presets):
            if not isinstance(p, dict): continue
            p_id = str(p.get('id', f"preset_{p_idx+1}")).strip()
            p_name = str(p.get('name', f"Preset {p_idx+1}")).strip()
            p_asp = str(p.get('aspect_name', p.get('aspect', 'Free'))).strip()
            
            # Translate legacy Cyrillic preset names to English if needed
            if "Портрет" in p_name and "landscape" not in p_id.lower():
                p_name = "🌟 Hero Portrait 4:5"
                p_asp = "4:5 (Portrait)"
            elif "Квадрат" in p_name or "square" in p_id.lower() or "1:1" in p_name:
                p_name = "⏹ Instagram Square 1:1"
                p_asp = "1:1 (Square)"
            elif "Повний" in p_name or "full" in p_id.lower():
                p_name = "🖼 Full Frame (Uncropped)"
                p_asp = "Original (Sensor)"

            p_cl = int(p.get('crop_left', p.get('left', 0)))
            p_cr = int(p.get('crop_right', p.get('right', 0)))
            p_ct = int(p.get('crop_top', p.get('top', 0)))
            p_cb = int(p.get('crop_bottom', p.get('bottom', 0)))
            p_st = _parse_straighten_to_slider(p.get('straighten', p.get('rotation', geo.get('straighten', 0))))

            # Enforce exact mathematical aspect ratio for the preset
            asp_ratio, clean_asp = parse_aspect_ratio_text(p_asp, w_val, h_val)
            if asp_ratio is None or asp_ratio <= 0:
                asp_ratio, clean_asp = parse_aspect_ratio_text(f"{p_id} {p_name}", w_val, h_val)

            if asp_ratio is not None and asp_ratio > 0 and w_val > 0 and h_val > 0:
                p_cl, p_cr, p_ct, p_cb = enforce_crop_aspect_ratio(
                    p_cl, p_cr, p_ct, p_cb, w_val, h_val, asp_ratio, allow_orientation_flip=False
                )

            parsed_presets.append({
                'id': p_id,
                'name': p_name,
                'aspect_name': clean_asp,
                'crop_left': p_cl,
                'crop_right': p_cr,
                'crop_top': p_ct,
                'crop_bottom': p_cb,
                'straighten': p_st
            })

    if not parsed_presets:
        # Auto-create recommended English presets with exact mathematical aspect ratios
        cl_sq, cr_sq, ct_sq, cb_sq = enforce_crop_aspect_ratio(
            geo.get('crop_left', 0), geo.get('crop_right', 0), geo.get('crop_top', 0), geo.get('crop_bottom', 0),
            w_val, h_val, 1.0, allow_orientation_flip=False
        )
        cl_45, cr_45, ct_45, cb_45 = enforce_crop_aspect_ratio(
            geo.get('crop_left', 0), geo.get('crop_right', 0), geo.get('crop_top', 0), geo.get('crop_bottom', 0),
            w_val, h_val, 4.0 / 5.0, allow_orientation_flip=False
        )

        parsed_presets = [
            {
                'id': 'portrait_4x5',
                'name': '🌟 Hero Portrait 4:5',
                'aspect_name': '4:5 / 5:4',
                'crop_left': cl_45,
                'crop_right': cr_45,
                'crop_top': ct_45,
                'crop_bottom': cb_45,
                'straighten': geo.get('straighten', 0)
            },
            {
                'id': 'square_1x1',
                'name': '⏹ Instagram Square 1:1',
                'aspect_name': '1:1 (Square)',
                'crop_left': cl_sq,
                'crop_right': cr_sq,
                'crop_top': ct_sq,
                'crop_bottom': cb_sq,
                'straighten': geo.get('straighten', 0)
            },
            {
                'id': 'full_frame',
                'name': '🖼 Full Frame (Uncropped)',
                'aspect_name': 'Original (Sensor)',
                'crop_left': 0,
                'crop_right': 0,
                'crop_top': 0,
                'crop_bottom': 0,
                'straighten': geo.get('straighten', 0)
            }
        ]

    blended['crop_presets'] = parsed_presets
    blended['active_crop_preset'] = new_dict.get('active_crop_preset', prev_dict.get('active_crop_preset', parsed_presets[0]['id'] if parsed_presets else 'default'))

    if 'bokeh' in blended and isinstance(blended['bokeh'], dict):
        bk = blended['bokeh']
        new_bk = new_dict.get('bokeh', {}) if isinstance(new_dict.get('bokeh'), dict) else {}
        for r_key in ('radius', 'max_radius', 'aperture', 'blur_radius', 'blur', 'bokeh_radius'):
            if r_key in new_bk and isinstance(new_bk[r_key], (int, float)):
                prev_r = prev_dict.get('bokeh', {}).get('radius', bk.get('radius', 40))
                diff_r = abs(new_bk[r_key] - prev_r)
                if diff_r < 3 and prev_r > 0:
                    bk['radius'] = prev_r
                else:
                    bk['radius'] = int(round(prev_r + (new_bk[r_key] - prev_r) * alpha))
                break
            elif r_key in new_dict and isinstance(new_dict[r_key], (int, float)):
                prev_r = prev_dict.get('bokeh', {}).get('radius', bk.get('radius', 40))
                diff_r = abs(new_dict[r_key] - prev_r)
                if diff_r < 3 and prev_r > 0:
                    bk['radius'] = prev_r
                else:
                    bk['radius'] = int(round(prev_r + (new_dict[r_key] - prev_r) * alpha))
                break
        if 'focus_point' in new_bk:
            bk['focus_point'] = copy.deepcopy(new_bk['focus_point'])
        elif 'focus_point' in new_dict:
            bk['focus_point'] = copy.deepcopy(new_dict['focus_point'])
        if 'focal_range' not in bk or bk['focal_range'] < 5:
            bk['focal_range'] = 12
        bk['samples'] = "256"

    return sanitize_state_equalizers(blended)

def compute_state_deltas(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> Tuple[float, float, str]:
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        return 0.0, 0.0, ""

    BASIC_KEYS = {
        'exposure', 'contrast', 'gamma', 'base_curve', 'temperature', 'tint',
        'highlights', 'shadows', 'whites', 'black_level', 'highlight_recovery',
        'local_contrast', 'global_chroma', 'global_vibrance',
        'denoise_luma', 'denoise_chroma', 'sharpen', 'sharpen_threshold', 'vignette'
    }

    max_grading_delta = 0.0
    max_total_delta = 0.0
    max_param_name = ""

    layers_a = state_a.get('layers', [])
    layers_b = state_b.get('layers', [])
    if isinstance(layers_a, list) and isinstance(layers_b, list):
        for l_idx, (la, lb) in enumerate(zip(layers_a, layers_b)):
            if not isinstance(la, dict) or not isinstance(lb, dict):
                continue
            sa = la.get('settings', la)
            sb = lb.get('settings', lb)
            for bk in BASIC_KEYS:
                va = sa.get(bk, la.get(bk, 0))
                vb = sb.get(bk, lb.get(bk, 0))
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
                    d = abs(float(va) - float(vb))
                    if d > max_grading_delta:
                        max_grading_delta = d
                        max_param_name = f"Layer {l_idx} {bk}"
                    if d > max_total_delta:
                        max_total_delta = d
            for eqk in ['tone_equalizer', 'color_equalizer', 'shadow_color_equalizer']:
                ea = sa.get(eqk, la.get(eqk, {}))
                eb = sb.get(eqk, lb.get(eqk, {}))
                if isinstance(ea, dict) and isinstance(eb, dict):
                    for subk, arr_a in ea.items():
                        arr_b = eb.get(subk, [])
                        if isinstance(arr_a, (list, tuple)) and isinstance(arr_b, (list, tuple)) and len(arr_a) == len(arr_b):
                            for idx, (x, y) in enumerate(zip(arr_a, arr_b)):
                                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
                                    d = abs(float(x) - float(y))
                                    d_eff = max(0.0, d - 1.0)
                                    if d_eff > max_grading_delta:
                                        max_grading_delta = d_eff
                                        max_param_name = f"{eqk}.{subk}[{idx}]"
                                    if d_eff > max_total_delta:
                                        max_total_delta = d_eff

    for bk in BASIC_KEYS:
        if bk in state_a and bk in state_b:
            va, vb = state_a[bk], state_b[bk]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
                d = abs(float(va) - float(vb))
                if d > max_grading_delta:
                    max_grading_delta = d
                    max_param_name = bk
                if d > max_total_delta:
                    max_total_delta = d

    ga = state_a.get('geometry_crop', {})
    gb = state_b.get('geometry_crop', {})
    for g_key in ['crop_left', 'crop_right', 'crop_top', 'crop_bottom']:
        ca, cb = ga.get(g_key, 0), gb.get(g_key, 0)
        if isinstance(ca, (int, float)) and isinstance(cb, (int, float)):
            d = abs(float(ca) - float(cb))
            d_eff = max(0.0, d - 20.0)
            if d_eff > max_total_delta:
                max_total_delta = d_eff
                max_param_name = g_key
    
    sa = ga.get('straighten', state_a.get('straighten', 0))
    sb = gb.get('straighten', state_b.get('straighten', 0))
    if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
        diff_deg = abs(float(sa) - float(sb)) / 10.0
        d_eff = max(0.0, diff_deg - 0.3)
        if d_eff > max_total_delta:
            max_total_delta = d_eff
            max_param_name = 'straighten'

    bka = state_a.get('bokeh', {})
    bkb = state_b.get('bokeh', {})
    if isinstance(bka, dict) and isinstance(bkb, dict):
        for b_param in ['radius', 'focal_depth', 'focal_range', 'fg_gamma', 'bg_gamma', 'boost']:
            ba, bb = bka.get(b_param), bkb.get(b_param)
            if isinstance(ba, (int, float)) and isinstance(bb, (int, float)):
                diff_b = abs(float(ba) - float(bb))
                d_eff = max(0.0, diff_b - 2.5)
                if d_eff > max_total_delta:
                    max_total_delta = d_eff
                    max_param_name = f"bokeh.{b_param}"

    return max_grading_delta, max_total_delta, max_param_name

def compute_state_delta(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
    grading_d, total_d, _ = compute_state_deltas(state_a, state_b)
    return max(grading_d, total_d)
