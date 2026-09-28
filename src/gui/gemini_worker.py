import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal

from core.gemini_assistant import GeminiAssistantService, extract_json_from_text, pil_to_jpeg_part

class GeminiAssistantWorker(QThread):
    """
    Asynchronous worker thread for streaming requests to Gemini 3.8 Flash.
    Streams reasoning thoughts and response text to UI without blocking.
    """
    thought_received = Signal(str)
    chunk_received = Signal(str)
    status_updated = Signal(str)
    finished_success = Signal(dict, str) # parsed_state, explanation
    error_occurred = Signal(str)

    def __init__(
        self,
        image: Image.Image,
        user_prompt: str = "",
        current_state: Optional[Dict[str, Any]] = None,
        is_auto_correct: bool = False,
        pass_num: int = 1,
        total_passes: int = 1,
        api_key: Optional[str] = None,
        thinking_level: str = "medium",
        mask_image: Optional[Image.Image] = None,
        parent=None,
        **kwargs
    ):
        super().__init__(parent)
        self.image = image
        self.user_prompt = user_prompt
        self.current_state = current_state
        self.is_auto_correct = is_auto_correct
        self.pass_num = pass_num
        self.total_passes = total_passes
        self.api_key = api_key
        # Accept thinking_budget for backward compatibility if passed
        if "thinking_budget" in kwargs and thinking_level == "medium":
            tb = kwargs["thinking_budget"]
            self.thinking_level = "low" if tb <= 512 else ("high" if tb > 2048 else "medium")
        else:
            self.thinking_level = thinking_level or "medium"
        self.mask_image = mask_image
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        import time, random
        try:
            self.status_updated.emit("Initializing Gemini 3.8 Flash client...")
            service = GeminiAssistantService(api_key=self.api_key)
            ok, msg = service.is_available()
            if not ok:
                self.error_occurred.emit(f"Gemini API Error: {msg}")
                return

            if self._is_cancelled:
                return

            prompt = service.build_user_prompt(
                user_request=self.user_prompt,
                current_state=self.current_state,
                is_auto_correct=self.is_auto_correct,
                pass_num=self.pass_num,
                total_passes=self.total_passes
            )

            contents = []
            if prompt:
                contents.append(prompt)
            if self.image:
                contents.append(pil_to_jpeg_part(self.image, quality=85, max_size=1024))
            if self.mask_image:
                contents.append(pil_to_jpeg_part(self.mask_image, quality=80, max_size=768))

            if not contents:
                self.error_occurred.emit("No input data provided (image or text).")
                return

            max_retries = 4
            last_err_msg = ""
            for attempt in range(1, max_retries + 1):
                if self._is_cancelled:
                    return
                try:
                    # Fresh client creation on every attempt: guarantees a fresh HTTP session, new socket connection, clean connection pool
                    client = service.get_client(timeout=180.0)
                    config = service.create_config(
                        thinking_level=self.thinking_level
                    )

                    full_text_chunks = []
                    thought_chunks = []
                    last_finish_reason = None
                    last_block_reason = None

                    # On attempt 1 & 2: use generate_content_stream for live streaming thoughts
                    # On attempt >= 3: fall back to generate_content (atomic single POST) which avoids SSE stream drop issues
                    if attempt <= 2:
                        self.status_updated.emit(f"Sending multimodal request to Gemini 3.8 Flash (Pass {self.pass_num}/{self.total_passes})...")
                        response_stream = client.models.generate_content_stream(
                            model=service.model_name,
                            contents=contents,
                            config=config
                        )

                        for chunk in response_stream:
                            if self._is_cancelled:
                                return

                            if chunk.candidates:
                                for candidate in chunk.candidates:
                                    if getattr(candidate, 'finish_reason', None):
                                        last_finish_reason = str(candidate.finish_reason)
                                    if candidate.content and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if getattr(part, 'thought', False):
                                                if part.text:
                                                    self.thought_received.emit(part.text)
                                                    thought_chunks.append(part.text)
                                            elif getattr(part, 'text', None):
                                                self.chunk_received.emit(part.text)
                                                full_text_chunks.append(part.text)
                            elif getattr(chunk, 'text', None):
                                self.chunk_received.emit(chunk.text)
                                full_text_chunks.append(chunk.text)

                        accumulated_text = "".join(full_text_chunks).strip()
                    else:
                        self.status_updated.emit(f"Using direct atomic request to Gemini 3.8 Flash (Pass {self.pass_num}/{self.total_passes}, Attempt {attempt}/{max_retries})...")
                        response = client.models.generate_content(
                            model=service.model_name,
                            contents=contents,
                            config=config
                        )
                        if response.candidates:
                            for candidate in response.candidates:
                                if getattr(candidate, 'finish_reason', None):
                                    last_finish_reason = str(candidate.finish_reason)
                                if candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if getattr(part, 'thought', False):
                                            if part.text:
                                                self.thought_received.emit(part.text)
                                                thought_chunks.append(part.text)
                                        elif getattr(part, 'text', None):
                                            full_text_chunks.append(part.text)

                        if getattr(response, 'prompt_feedback', None) and getattr(response.prompt_feedback, 'block_reason', None):
                            last_block_reason = str(response.prompt_feedback.block_reason)

                        accumulated_text = (response.text or "").strip()
                        if not accumulated_text and full_text_chunks:
                            accumulated_text = "".join(full_text_chunks).strip()

                    if not accumulated_text:
                        err_detail = "Gemini 3.8 Flash returned an empty response."
                        if last_block_reason:
                            err_detail = f"Gemini blocked the request: {last_block_reason}."
                        elif last_finish_reason and "SAFETY" in last_finish_reason.upper():
                            err_detail = f"Gemini safety filter blocked the image ({last_finish_reason})."
                        elif last_finish_reason and "MAX_TOKENS" in last_finish_reason.upper():
                            err_detail = f"Gemini exhausted token budget during thinking ({last_finish_reason})."
                        elif last_finish_reason and "STOP" not in last_finish_reason.upper():
                            err_detail = f"Gemini terminated prematurely ({last_finish_reason})."
                        raise RuntimeError(err_detail)

                    parsed_data = extract_json_from_text(accumulated_text)
                    if not parsed_data:
                        raise ValueError("Failed to parse structured JSON from Gemini output.")

                    explanation = parsed_data.get("explanation", "Adjustments applied successfully.")
                    self.finished_success.emit(parsed_data, explanation)
                    return # Successfully finished!

                except Exception as e:
                    if self._is_cancelled:
                        return
                    last_err_msg = str(e)
                    err_str = last_err_msg.lower()

                    is_fatal_key_error = any(k in err_str for k in [
                        "api key not valid", "api_key_invalid", "invalid api key",
                        "permission_denied", "permissiondenied", "403", "unregistered caller"
                    ])
                    if is_fatal_key_error:
                        if not self._is_cancelled:
                            self.error_occurred.emit(f"Invalid Gemini API Key: {last_err_msg}")
                        return

                    is_rate_limit = any(k in err_str for k in ["429", "resource_exhausted", "quota", "too many requests", "rate limit", "ratelimit"])

                    if attempt < max_retries:
                        if is_rate_limit:
                            cooldown = min(20.0, 8.0 + 3.0 * attempt + random.uniform(0.5, 1.5))
                            status_msg = f"⏳ Gemini Rate Limit / Quota on Pass {self.pass_num}. Cooling down for {cooldown:.0f}s (Attempt {attempt}/{max_retries})..."
                        else:
                            cooldown = min(12.0, 2.0 * attempt + random.uniform(0.5, 1.0))
                            status_msg = f"⏳ Gemini API busy / Network glitch on Pass {self.pass_num}. Resetting connection & retrying in {cooldown:.1f}s (Attempt {attempt}/{max_retries})..."

                        self.status_updated.emit(status_msg)

                        # Graceful sleep with cancel detection
                        slept = 0.0
                        while slept < cooldown and not self._is_cancelled:
                            time.sleep(0.1)
                            slept += 0.1
                        if self._is_cancelled:
                            return
                        continue
                    else:
                        if not self._is_cancelled:
                            self.error_occurred.emit(f"Gemini API request failed on Pass {self.pass_num} after {max_retries} attempts: {last_err_msg}")
                        return

        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))


class GeminiQualityControlWorker(QThread):
    """
    Asynchronous worker thread for comparing with/without Background layer and 3D Bokeh.
    Evaluates:
    1. Whether the Background layer creates unnatural cutout halos or degrades the image compared to clean Base Grade.
    2. Whether 3D Bokeh blurs the main subject, degrades facial sharpness, or introduces artifacts.
    """
    status_updated = Signal(str)
    finished_decision = Signal(bool, bool, float, str) # (keep_bg, keep_bokeh, confidence, reason)
    error_occurred = Signal(str)

    def __init__(
        self,
        img_final: Image.Image,
        img_clean_base: Image.Image,
        img_no_bokeh: Optional[Image.Image] = None,
        has_bg: bool = True,
        has_bokeh: bool = True,
        api_key: Optional[str] = None,
        thinking_level: str = "low",
        parent=None,
        **kwargs
    ):
        super().__init__(parent)
        self.img_final = img_final
        self.img_clean_base = img_clean_base
        self.img_no_bokeh = img_no_bokeh
        self.has_bg = has_bg
        self.has_bokeh = has_bokeh
        self.api_key = api_key
        if "thinking_budget" in kwargs and thinking_level == "low":
            tb = kwargs["thinking_budget"]
            self.thinking_level = "low" if tb <= 512 else ("high" if tb > 2048 else "medium")
        else:
            self.thinking_level = thinking_level or "low"
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.status_updated.emit("Evaluating AI Quality Control (Background Naturalness & Bokeh Sharpness)...")
            service = GeminiAssistantService(api_key=self.api_key)
            ok, msg = service.is_available()
            if not ok:
                self.error_occurred.emit(f"Gemini API Error: {msg}")
                return

            if self._is_cancelled:
                return

            client = service.get_client()

            contents = []
            if self.has_bg and self.has_bokeh:
                prompt = (
                    "You are an expert Fine-Art Master Photographer, Retoucher, and Quality Control Director.\n"
                    "Critically compare the provided versions of this photograph:\n"
                    "- Image 1: FULL EDIT (With Background Layer grading & With 3D Bokeh blur simulation).\n"
                    "- Image 2: WITH Background Layer ONLY (Without Bokeh blur simulation).\n"
                    "- Image 3: CLEAN BASE GRADE ONLY (Without Background Layer & Without Bokeh blur simulation).\n\n"
                    "TASK: Evaluate TWO critical enhancements to ensure the highest fine-art standards:\n\n"
                    "1. BACKGROUND LAYER EVALUATION (Image 2/1 vs Clean Base Grade Image 3):\n"
                    "   - Inspect the silhouette and contours of the subject (hair, skin, limbs, clothes).\n"
                    "   - Does the Background layer adjustment create an unnatural dark/bright halo, harsh cutout edge, or jarring lighting difference that looks worse/more artificial than the unified Base Grade in Image 3?\n"
                    "   - If the Background layer looks natural, harmonious, and truly enhances depth -> 'keep_background': true.\n"
                    "   - If the Background layer creates unnatural edge artifacts, haloing, or looks worse than Image 3 -> 'keep_background': false.\n\n"
                    "2. 3D BOKEH BLUR SIMULATION EVALUATION (Image 1 vs Image 2):\n"
                    "   - Inspect the main subject's face, eyes, hair, hands, and clothing in Image 1.\n"
                    "   - Does 3D Bokeh in Image 1 blur ANY part of the main subject or create distracting fuzzy artifacts?\n"
                    "   - If Image 1 keeps the subject 100% tack-sharp and natural while cleanly blurring only the distant background -> 'keep_bokeh': true.\n"
                    "   - If Image 1 blurs or softens the subject -> 'keep_bokeh': false.\n\n"
                    "DECISION FORMAT:\n"
                    "Respond ONLY with a JSON object in this exact schema:\n"
                    "```json\n"
                    "{\n"
                    "  \"keep_background\": false,\n"
                    "  \"keep_bokeh\": false,\n"
                    "  \"confidence\": 0.95,\n"
                    "  \"reason\": \"Detailed explanation in Ukrainian or English of whether Background layer and Bokeh were kept or disabled and why\"\n"
                    "}\n"
                    "```"
                )
                contents = [
                    "Image 1 (Full Edit: With Background Layer & With 3D Bokeh):",
                    pil_to_jpeg_part(self.img_final, quality=85, max_size=768),
                    "Image 2 (With Background Layer ONLY - Without Bokeh):",
                    pil_to_jpeg_part(self.img_no_bokeh if self.img_no_bokeh is not None else self.img_final, quality=85, max_size=768),
                    "Image 3 (Clean Base Grade - Without Background & Without Bokeh):",
                    pil_to_jpeg_part(self.img_clean_base, quality=85, max_size=768),
                    prompt
                ]
            elif self.has_bg:
                prompt = (
                    "You are an expert Fine-Art Master Photographer, Retoucher, and Quality Control Director.\n"
                    "Critically compare these two graded versions of the photograph:\n"
                    "- Image 1: WITH Background Layer adjustment applied (depth-isolated background grading).\n"
                    "- Image 2: CLEAN BASE GRADE (Without Background Layer adjustment).\n\n"
                    "TASK: Evaluate the BACKGROUND LAYER naturalness:\n"
                    "1. Inspect the boundary and edges around the subject (hair, skin, clothes).\n"
                    "2. Does the Background layer create an unnatural dark/bright halo, cutout edge, or look worse/more artificial than the unified Base Grade in Image 2?\n"
                    "3. If the Background adjustment looks seamless, natural, and genuinely improves the photo -> 'keep_background': true.\n"
                    "4. If the Background adjustment introduces edge halos, looks fake, or is worse than Image 2 -> 'keep_background': false.\n\n"
                    "DECISION FORMAT:\n"
                    "Respond ONLY with a JSON object in this exact schema:\n"
                    "```json\n"
                    "{\n"
                    "  \"keep_background\": false,\n"
                    "  \"keep_bokeh\": false,\n"
                    "  \"confidence\": 0.95,\n"
                    "  \"reason\": \"Detailed explanation in Ukrainian or English of whether Background layer was kept or disabled and why\"\n"
                    "}\n"
                    "```"
                )
                contents = [
                    "Image 1 (With Background Layer Adjustment):",
                    pil_to_jpeg_part(self.img_final, quality=85, max_size=768),
                    "Image 2 (Clean Base Grade - Without Background Layer):",
                    pil_to_jpeg_part(self.img_clean_base, quality=85, max_size=768),
                    prompt
                ]
            else: # has_bokeh only
                prompt = (
                    "You are an expert Fine-Art Master Photographer, Retoucher, and Quality Control Director.\n"
                    "Compare these two graded versions of the photograph:\n"
                    "- Image 1: WITH 3D Bokeh Blur Simulation applied.\n"
                    "- Image 2: WITHOUT Bokeh Blur Simulation (Clean Optical Capture).\n\n"
                    "TASK: Critically inspect the main subject (person, child, face, eyes, hair, hands, clothing) in Image 1 compared to Image 2.\n"
                    "1. Is the subject's face, eyes, hair, or body blurred, softened, or out-of-focus due to the 3D bokeh?\n"
                    "2. Are there unnatural blur halos, glowing edge artifacts, or fuzzy patches across the subject's contours?\n"
                    "3. If Image 1 keeps the main subject 100% sharp, crisp, and natural while cleanly blurring ONLY the distant background -> 'keep_bokeh': true.\n"
                    "4. If Image 1 blurs ANY part of the main subject, degrades facial sharpness, or creates distracting artifacts -> 'keep_bokeh': false (prefer clean Image 2).\n\n"
                    "DECISION FORMAT:\n"
                    "Respond ONLY with a JSON object in this exact schema:\n"
                    "```json\n"
                    "{\n"
                    "  \"keep_background\": false,\n"
                    "  \"keep_bokeh\": false,\n"
                    "  \"confidence\": 0.95,\n"
                    "  \"reason\": \"Brief explanation in Ukrainian or English of whether bokeh blurred the subject or kept it sharp\"\n"
                    "}\n"
                    "```"
                )
                contents = [
                    "Image 1 (With 3D Bokeh Blur):",
                    pil_to_jpeg_part(self.img_final, quality=85, max_size=768),
                    "Image 2 (Without Bokeh Blur - Clean Grade):",
                    pil_to_jpeg_part(self.img_clean_base, quality=85, max_size=768),
                    prompt
                ]

            config = service.create_config(thinking_level=self.thinking_level)

            import time, random
            max_retries = 3
            response = None
            for attempt in range(1, max_retries + 1):
                if self._is_cancelled:
                    return
                try:
                    client = service.get_client(timeout=180.0)
                    response = client.models.generate_content(
                        model=service.model_name,
                        contents=contents,
                        config=config
                    )
                    break
                except Exception as e:
                    if self._is_cancelled:
                        return
                    if attempt < max_retries:
                        backoff = min(15.0, 2.0 * attempt + random.uniform(0.5, 1.0))
                        time.sleep(backoff)
                        continue
                    else:
                        raise e

            if self._is_cancelled:
                return

            text = response.text or ""
            parsed = extract_json_from_text(text)
            if parsed and isinstance(parsed, dict):
                keep_bg = bool(parsed.get("keep_background", self.has_bg)) if self.has_bg else False
                keep_bokeh = bool(parsed.get("keep_bokeh", self.has_bokeh)) if self.has_bokeh else False
                conf = float(parsed.get("confidence", 1.0))
                reason = str(parsed.get("reason", "Evaluated by Gemini QC"))
                self.finished_decision.emit(keep_bg, keep_bokeh, conf, reason)
            else:
                # Fallback: inspect text for decision
                keep_bg = ("keep_background\": true" in text.lower()) if self.has_bg else False
                keep_bokeh = ("keep_bokeh\": true" in text.lower()) if self.has_bokeh else False
                reason = text.strip()[:300]
                self.finished_decision.emit(keep_bg, keep_bokeh, 0.8, reason)

        except Exception as e:
            if not self._is_cancelled:
                self.error_occurred.emit(str(e))


GeminiBokehVerifierWorker = GeminiQualityControlWorker


class GeminiCropWorker(QThread):
    """
    Dedicated 2-Step worker that:
    1. Plans orientation-aware fine-art crop presets (Landscape vs Portrait).
    2. Streams Gemini 3.8 Flash internal thoughts in real-time.
    3. Performs a Multimodal Visual QC Inspection on ALL candidate crop previews:
       evaluates anatomy clipping, visual balance, and either approves (keep),
       refines (modify with margin delta), or prunes (reject) each preset.
    """
    thought_received = Signal(str)
    status_updated = Signal(str)
    finished_presets = Signal(list, str) # presets_list, explanation
    error_occurred = Signal(str)

    def __init__(
        self,
        image: Image.Image,
        api_key: Optional[str] = None,
        thinking_level: str = "medium",
        orig_size: Optional[Tuple[int, int]] = None,
        parent=None,
        **kwargs
    ):
        super().__init__(parent)
        self.image = image
        self.api_key = api_key
        if "thinking_budget" in kwargs and thinking_level == "medium":
            tb = kwargs["thinking_budget"]
            self.thinking_level = "low" if tb <= 512 else ("high" if tb > 2048 else "medium")
        else:
            self.thinking_level = thinking_level or "medium"

        if orig_size and orig_size[0] > 0 and orig_size[1] > 0:
            self.orig_w, self.orig_h = orig_size
        else:
            self.orig_w, self.orig_h = self.image.size

        self.is_landscape = (self.orig_w >= self.orig_h)
        self.reasoning_log = []
        self.thought_text = ""
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            import os, time, json, random
            from datetime import datetime
            from core.gemini_assistant import parse_aspect_ratio_text, enforce_crop_aspect_ratio, apply_golden_headroom_rule

            self.status_updated.emit("Detecting subject bounding box & spatial composition...")
            service = GeminiAssistantService(api_key=self.api_key)
            ok, msg = service.is_available()
            if not ok:
                self.error_occurred.emit(f"Gemini API Error: {msg}")
                return

            if self._is_cancelled:
                return

            config_step1 = service.create_config(
                thinking_level=self.thinking_level
            )

            orientation_str = "LANDSCAPE (Horizontal)" if self.is_landscape else "PORTRAIT (Vertical)"
            aspect_val = float(self.orig_w) / float(self.orig_h)

            if self.is_landscape:
                presets_guideline = (
                    "MANDATORY INDEPENDENT PRESETS TO RE-COMPOSE ANEW:\n"
                    "• Preset 1: 'hero_landscape' | Name: '🌟 Hero Landscape 16:9' | Aspect: '16:9 (Landscape)'\n"
                    "  - Cinematic master landscape crop. Eye line high on upper third (~30%), eliminate dead sky/ceiling, generous lead room in gaze direction.\n"
                    "• Preset 2: 'square_1x1' | Name: '⏹ Instagram Square 1:1' | Aspect: '1:1 (Square)'\n"
                    "  - Bold medium square portrait. Re-composed closer to subject, horizontally centered, headroom 9-11%, eye line on upper third (~32%).\n"
                    "• Preset 3: 'classic_scene' | Name: '📐 Classic Scene 4:3' | Aspect: '4:3 (Landscape)'\n"
                    "  - Balanced environmental composition showing subject and contextual scene without excessive vacant ceiling.\n"
                    "• Preset 4: 'social_story' | Name: '📱 Social Reel / Story 9:16' | Aspect: '9:16 (Portrait)'\n"
                    "  - Vertical framing centered on subject with natural breathing room.\n"
                    "• Preset 5: 'full_frame' | Name: '🖼 Full Frame (Uncropped)' | Aspect: 'Original (Sensor)' | box_2d: [0, 0, 1000, 1000]\n"
                )
            else:
                presets_guideline = (
                    "MANDATORY INDEPENDENT PRESETS TO RE-COMPOSE ANEW (Each must be uniquely framed from scratch!):\n"
                    "• Preset 1: 'hero_portrait' | Name: '🌟 Hero Portrait 4:5' | Aspect: '4:5 (Portrait)'\n"
                    "  - Master editorial portrait framing. STRICT GOLDEN HEADROOM: exactly 8% to 12% space above head_crown_y. Eyes MUST sit on upper third (top ~30-33%). Sized to include head, chest, and expressive hands/props (e.g. food/table). TRIM EXCESSIVE CEILING COMPLETELY!\n"
                    "• Preset 2: 'square_1x1' | Name: '⏹ Instagram Square 1:1' | Aspect: '1:1 (Square)'\n"
                    "  - Independent medium square portrait. Crop in closer around the torso and face. Horizontally centered, tight 8-10% headroom above crown, eyes at ~32% line.\n"
                    "• Preset 3: 'intimate_closeup' | Name: '✨ Intimate Close-Up' | Aspect: '4:5 (Portrait)'\n"
                    "  - Dramatic fine-art close-up. Sized much tighter (smaller crop box height, e.g. ~300-450 units) focused purely on face, eyes, and expression. Top crop sits right at the hair crown (5-8% headroom) or subtle hair trim, eyes dead on upper third line, bottom finishes below chin/collar.\n"
                    "• Preset 4: 'story_3x4' | Name: '📐 Classic Scene 3:4' | Aspect: '3:4 (Portrait)'\n"
                    "  - Environmental portrait showing broader table/room context, but STILL strictly keeping eye-line in upper third and eliminating vacant ceiling.\n"
                    "• Preset 5: 'full_frame' | Name: '🖼 Full Frame (Uncropped)' | Aspect: 'Original (Sensor)' | box_2d: [0, 0, 1000, 1000]\n"
                )

            # Step 1 Prompt: Orientation-aware spatial composition & subject detection
            prompt_step1 = (
                "You are an Elite Commercial & Fine-Art Portrait Photography Director.\n"
                f"IMAGE SPECIFICATIONS:\n"
                f"- Sensor Dimensions: {self.orig_w}x{self.orig_h} pixels\n"
                f"- Primary Orientation: {orientation_str} (Aspect Ratio {aspect_val:.2f}:1)\n\n"
                "COORDINATE SYSTEM:\n"
                "- Normalized scale 0 to 1000 ([ymin, xmin, ymax, xmax]). 0,0 is top-left, 1000,1000 is bottom-right.\n\n"
                "🌟 BIOMECHANICAL & ANATOMICAL CROPPING LAWS (Gestalt Visual Continuity):\n"
                "1. 🚫 STRICTLY PROHIBITED ARTICULATION CROPS (Never slice through joints - looks like amputations!):\n"
                "   • Carpal Joint (Wrist): NEVER crop across the wrist! Detaches hands and leaves floating stumps.\n"
                "   • Cubital Joint (Elbow): NEVER crop across the elbow! Creates awkward stumps along borders.\n"
                "   • Distal Phalanges (Fingers): NEVER truncate fingertips, knuckles, or thumbs! Show whole hand or crop mid-forearm.\n"
                "   • Patella (Knee): NEVER slice through the knee joint! Leaves subject blocky and unsupported.\n"
                "   • Talocrural Joint (Ankle): NEVER slice through the ankle! Leaves subject floating in space without footing.\n"
                "   • Cervical Spine (Neck): NEVER crop through the neck! Visually decapitates the head.\n"
                "   • Submental Border (Chin): NEVER truncate the chin! Jawline and facial symmetry must remain clear.\n"
                "   • Hairline Perimeter: NEVER skim the hairline or clip a thin sliver of hair.\n"
                "2. ✅ PERMISSIBLE & SAFE CROPPING ZONES (Cut across tapering long-bone muscle shafts):\n"
                "   • Cranial Apex / Mid-Hair: Safe for tight beauty/close-up headshots (decisive cut through upper forehead/mid-hair).\n"
                "   • Clavicle / Mid-Chest (Sternum): Standard corporate bust crop; creates stable triangular pectoral base.\n"
                "   • Lateral Deltoid / Mid-Bicep: Accentuates shoulder width; cuts along tapering bicep shaft.\n"
                "   • Mid-Forearm (Radius/Ulna): Allows eye to follow gesture outward toward hands.\n"
                "   • Natural Waistline: Above pelvic crest, ensuring elbows/wrists are not simultaneously bisected.\n"
                "   • Femoral Shaft (Mid-Thigh): Classic 3/4 crop, well above the knee patella.\n"
                "   • Tibial Shaft (Mid-Calf / Shin): Seven-eighths crop, through gastrocnemius well above ankle.\n"
                "   • Plantar Base (Footwear): Full-length framing requires a 5% to 8% floor buffer beneath soles.\n\n"
                "🌟 DYNAMIC SCALE-DEPENDENT HEADROOM & EYE-LINE LAWS:\n"
                "• Extreme Close-Up: 0% Headroom (Decisive cranial cut through upper forehead/mid-hair, chin fully clear, eyes on upper third).\n"
                "• Classic Bust / Headshot: 5% to 8% Headroom (Exactly 1/12th frame height rule), eyes pinned to upper third line.\n"
                "• Medium Portrait (Waist-Up): 8% Headroom, eyes in upper third, lower boundary at natural waist.\n"
                "• Three-Quarter Length: 8% to 12% Headroom, lower boundary at mid-thigh (mid-femur).\n"
                "• Full-Length Environmental: 12% to 15% Headroom, plus 5% to 8% floor margin beneath footwear.\n"
                "• ELIMINATE DEAD CEILING: Never leave 20%-40% vacant ceiling, sky, or lamps above head! Distance (head_crown_y - ymin) must match the scale above.\n\n"
                "🌟 COMPOSITION, LEAD ROOM & MULTI-SUBJECT DYNAMICS:\n"
                "• 🧭 MANDATORY LEAD ROOM & GAZE/TORSO DIRECTION LAW:\n"
                "  - When subject/torso is facing or looking LEFT: Dedicate 60%-65% of horizontal space to the LEFT (in front of chest/eyes). crop_left MUST be smaller than crop_right!\n"
                "  - When subject/torso is facing or looking RIGHT: Dedicate 60%-65% of horizontal space to the RIGHT (in front of chest/eyes). crop_right MUST be smaller than crop_left!\n"
                "  - When subject faces directly at camera: Center subject horizontally (equal left/right margins).\n"
                "  - NEVER leave more empty space behind the subject's back than in front of their body or gaze!\n"
                "• Children / Props / Seated: Meet at eye level! Keep hands, food, burgers, toys, and gestures whole and complete inside frame.\n"
                "• Horizon Decapitation Prevention: Never let a sharp background horizon, curb, or window frame cut through the neck.\n"
                "• Wide-Angle Edge Buffer: Keep heads/outer limbs out of the extreme perimeter (outer 15% margin) to prevent optical distortion.\n"
                "• INDEPENDENT PRESETS: Re-compose EVERY preset from scratch with unique scale, height, and framing intent!\n\n"
                f"{presets_guideline}\n"
                "Return valid JSON:\n"
                "```json\n"
                "{\n"
                '  "subject_detection": {\n'
                '    "head_crown_y": <integer 0-1000, exact top edge of hair/hat>,\n'
                '    "eyes_y": <integer 0-1000, exact eye level>,\n'
                '    "chin_y": <integer 0-1000, bottom of chin>,\n'
                '    "subject_center_x": <integer 0-1000, horizontal center of subject>,\n'
                '    "facing_direction": "left | right | center",\n'
                '    "gaze_direction": "left | right | center",\n'
                '    "person_box": [ymin, xmin, ymax, xmax]\n'
                '  },\n'
                '  "composition_thoughts": "Detailed biomechanical explanation of joint clearance, headroom allocation, and scale",\n'
                '  "explanation": "Summary of artistic choices",\n'
                '  "crop_presets": [\n'
                '    {\n'
                '      "id": "hero_portrait",\n'
                '      "name": "🌟 Hero Portrait 4:5",\n'
                '      "aspect_name": "4:5 (Portrait)",\n'
                '      "box_2d": [ymin, xmin, ymax, xmax],\n'
                '      "headroom_pct": 10,\n'
                '      "eye_line_pct": 32\n'
                '    }\n'
                '  ]\n'
                "}\n"
                "```"
            )

            contents_step1 = [prompt_step1, pil_to_jpeg_part(self.image, quality=85, max_size=1024)]
            self.status_updated.emit("Calculating 2D composition boundaries (Step 1/2)...")

            max_retries = 3
            accumulated_text1 = ""
            step1_thoughts = []

            for attempt in range(1, max_retries + 1):
                if self._is_cancelled: return
                try:
                    client = service.get_client(timeout=180.0)
                    step1_thoughts.clear()
                    full_chunks1 = []

                    # Attempt streaming to capture thoughts in real-time
                    if attempt <= 2:
                        response_stream = client.models.generate_content_stream(
                            model=service.model_name,
                            contents=contents_step1,
                            config=config_step1
                        )
                        for chunk in response_stream:
                            if self._is_cancelled: return
                            if chunk.candidates:
                                for candidate in chunk.candidates:
                                    if candidate.content and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if getattr(part, 'thought', False) and part.text:
                                                self.thought_received.emit(part.text)
                                                step1_thoughts.append(part.text)
                                            elif getattr(part, 'text', None):
                                                full_chunks1.append(part.text)
                            elif getattr(chunk, 'text', None):
                                full_chunks1.append(chunk.text)
                        accumulated_text1 = "".join(full_chunks1).strip()
                    else:
                        response1 = client.models.generate_content(
                            model=service.model_name,
                            contents=contents_step1,
                            config=config_step1
                        )
                        accumulated_text1 = (response1.text or "").strip()
                        if response1.candidates:
                            for candidate in response1.candidates:
                                if candidate.content and candidate.content.parts:
                                    for part in candidate.content.parts:
                                        if getattr(part, 'thought', False) and part.text:
                                            self.thought_received.emit(part.text)
                                            step1_thoughts.append(part.text)

                    if accumulated_text1:
                        break
                except Exception as e:
                    if self._is_cancelled: return
                    err_str = str(e).lower()
                    is_retryable = any(k in err_str for k in ["429", "resource_exhausted", "quota", "503", "500", "timeout", "connection"])
                    if is_retryable and attempt < max_retries:
                        backoff = min(20.0, 3.0 * attempt + random.uniform(0.5, 1.5))
                        self.status_updated.emit(f"⏳ Gemini Crops busy. Retrying in {backoff:.1f}s ({attempt}/{max_retries})...")
                        slept = 0.0
                        while slept < backoff and not self._is_cancelled:
                            time.sleep(0.1)
                            slept += 0.1
                        continue
                    else:
                        raise e

            if self._is_cancelled:
                return

            full_thought_step1 = "".join(step1_thoughts).strip()
            self.thought_text += f"\n--- STEP 1 COMPOSITION PLANNING THOUGHTS ---\n{full_thought_step1}\n"
            self.reasoning_log.append({
                "phase": "step1_planning",
                "orientation": orientation_str,
                "thoughts": full_thought_step1
            })

            data1 = extract_json_from_text(accumulated_text1)
            raw_presets = data1.get("crop_presets", [])
            explanation = data1.get("explanation", "Generated fine-art composition presets.")

            # Extract subject detection landmarks
            subject_det = data1.get("subject_detection", {})
            head_crown_y = None
            eyes_y = None
            chin_y = None
            subject_center_x = None

            def _normalize_coord(val):
                if val is None: return None
                try:
                    v = float(val)
                    if 0.0 < v <= 1.0: v *= 1000.0
                    elif 1.0 < v <= 100.0: v *= 10.0
                    return v
                except:
                    return None

            facing_direction = None
            gaze_direction = None
            if isinstance(subject_det, dict):
                head_crown_y = _normalize_coord(subject_det.get("head_crown_y"))
                eyes_y = _normalize_coord(subject_det.get("eyes_y"))
                chin_y = _normalize_coord(subject_det.get("chin_y"))
                subject_center_x = _normalize_coord(subject_det.get("subject_center_x"))
                facing_direction = str(subject_det.get("facing_direction") or "").lower().strip() or None
                gaze_direction = str(subject_det.get("gaze_direction") or "").lower().strip() or None

            # Fallback landmark extraction from person_box or person_detected_box
            if head_crown_y is None:
                pbox = data1.get("person_detected_box") or (subject_det.get("person_box") if isinstance(subject_det, dict) else None)
                if isinstance(pbox, (list, tuple)) and len(pbox) >= 4:
                    py0, px0, py1, px1 = [float(v) for v in pbox[:4]]
                    if py1 <= 1.0 and px1 <= 1.0:
                        py0 *= 1000.0; px0 *= 1000.0; py1 *= 1000.0; px1 *= 1000.0
                    elif py1 <= 100.0 and px1 <= 100.0:
                        py0 *= 10.0; px0 *= 10.0; py1 *= 10.0; px1 *= 10.0
                    head_crown_y = py0
                    eyes_y = py0 + (py1 - py0) * 0.15
                    chin_y = py0 + (py1 - py0) * 0.35
                    if subject_center_x is None:
                        subject_center_x = (px0 + px1) / 2.0

            # Convert box_2d [ymin, xmin, ymax, xmax] -> crop_left, crop_right, crop_top, crop_bottom
            parsed_presets = []
            for p in raw_presets:
                if not isinstance(p, dict): continue
                p_id = str(p.get('id', 'preset')).strip()
                p_name = str(p.get('name', 'Preset')).strip()
                p_asp = str(p.get('aspect_name', p.get('aspect', 'Free'))).strip()

                ymin = 0.0; xmin = 0.0; ymax = 1000.0; xmax = 1000.0
                if 'box_2d' in p and isinstance(p['box_2d'], (list, tuple)) and len(p['box_2d']) >= 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in p['box_2d'][:4]]
                    # Normalize if 0..100 or 0..1
                    if ymax <= 100 and xmax <= 100 and (ymax > 1.0 or xmax > 1.0):
                        ymin *= 10.0; xmin *= 10.0; ymax *= 10.0; xmax *= 10.0
                    elif ymax <= 1.0 and xmax <= 1.0:
                        ymin *= 1000.0; xmin *= 1000.0; ymax *= 1000.0; xmax *= 1000.0

                    cl = int(round(np.clip(xmin, 0, 480)))
                    cr = int(round(np.clip(1000.0 - xmax, 0, 480)))
                    ct = int(round(np.clip(ymin, 0, 480)))
                    cb = int(round(np.clip(1000.0 - ymax, 0, 480)))
                else:
                    cl = int(round(float(p.get('crop_left', 0))))
                    cr = int(round(float(p.get('crop_right', 0))))
                    ct = int(round(float(p.get('crop_top', 0))))
                    cb = int(round(float(p.get('crop_bottom', 0))))
                    # Normalize if 0..1
                    if any(0.0 < v <= 1.0 for v in (cl, cr, ct, cb)):
                        cl = int(round(cl * 1000.0))
                        cr = int(round(cr * 1000.0))
                        ct = int(round(ct * 1000.0))
                        cb = int(round(cb * 1000.0))
                    cl = int(np.clip(cl, 0, 480))
                    cr = int(np.clip(cr, 0, 480))
                    ct = int(np.clip(ct, 0, 480))
                    cb = int(np.clip(cb, 0, 480))
                    ymin = float(ct)
                    xmin = float(cl)
                    ymax = float(1000 - cb)
                    xmax = float(1000 - cr)

                # Determine and enforce mathematically precise aspect ratio centered on subject
                asp_ratio, clean_asp = parse_aspect_ratio_text(p_asp, self.orig_w, self.orig_h)
                if asp_ratio is None or asp_ratio <= 0:
                    asp_ratio, clean_asp = parse_aspect_ratio_text(f"{p_id} {p_name}", self.orig_w, self.orig_h)

                if p_id == 'full_frame':
                    cl, cr, ct, cb = 0, 0, 0, 0
                elif head_crown_y is not None and head_crown_y > 0:
                    cl, cr, ct, cb = apply_golden_headroom_rule(
                        ymin=ymin,
                        xmin=xmin,
                        ymax=ymax,
                        xmax=xmax,
                        head_crown_y=head_crown_y,
                        eyes_y=eyes_y if eyes_y is not None else -1.0,
                        img_w=self.orig_w,
                        img_h=self.orig_h,
                        target_aspect=asp_ratio,
                        preset_id=p_id,
                        subject_center_x=subject_center_x,
                        chin_y=chin_y,
                        facing_direction=facing_direction,
                        gaze_direction=gaze_direction
                    )
                elif asp_ratio is not None and asp_ratio > 0:
                    cl, cr, ct, cb = enforce_crop_aspect_ratio(
                        cl, cr, ct, cb, self.orig_w, self.orig_h, asp_ratio, allow_orientation_flip=False
                    )

                parsed_presets.append({
                    'id': p_id,
                    'name': p_name,
                    'aspect_name': clean_asp,
                    'aspect_ratio': asp_ratio,
                    'crop_left': cl,
                    'crop_right': cr,
                    'crop_top': ct,
                    'crop_bottom': cb
                })

            # =========================================================================
            # Step 2: Multimodal Visual QC Inspection & Preset Pruning
            # (AI looks at the actual cropped previews to keep, refine, or reject)
            # =========================================================================
            candidates_to_qc = [p for p in parsed_presets if p['id'] != 'full_frame']
            
            if candidates_to_qc and not self._is_cancelled:
                try:
                    self.status_updated.emit("🔍 Visual QC: Reviewing all crop previews for anatomy clipping & aesthetics...")
                    W_img, H_img = self.image.size

                    cropped_thumbnails = []
                    for pr in candidates_to_qc:
                        cl_f = pr['crop_left'] / 1000.0
                        cr_f = pr['crop_right'] / 1000.0
                        ct_f = pr['crop_top'] / 1000.0
                        cb_f = pr['crop_bottom'] / 1000.0

                        x0 = int(round(cl_f * W_img))
                        x1 = max(x0 + 10, int(round((1.0 - cr_f) * W_img)))
                        y0 = int(round(ct_f * H_img))
                        y1 = max(y0 + 10, int(round((1.0 - cb_f) * H_img)))

                        patch = self.image.crop((x0, y0, x1, y1))
                        patch.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        cropped_thumbnails.append(patch)

                    prompt_qc = (
                        "You are an uncompromising Fine-Art Photography Composition Judge & Quality Control Inspector.\n"
                        "You are inspecting the ACTUAL CROPPED RESULTS of the candidate presets for this photograph.\n\n"
                        "CRITICAL EVALUATION CRITERIA FOR EACH CANDIDATE:\n"
                        "1. 👑 DEAD HEADROOM & SUNKEN SUBJECT (AMATEUR FRAMING DEFECT):\n"
                        "   - Look closely above the subject's head/hair. Is there massive empty dead space, ceiling, lights, or window frames?\n"
                        "   - Is the subject sunken into the bottom half of the frame with eyes positioned near or below the middle?\n"
                        "   - If there is excessive empty space above the head, action = 'modify' with positive adjust_ct_delta (+20 to +70) to trim down the ceiling and lift the subject, or 'reject' if unfixable!\n"
                        "2. 🚫 BIOMECHANICAL JOINT AMPUTATIONS & LIMB CLIPPING:\n"
                        "   - Does the crop slice across an articulation joint (wrist, elbow, knee, ankle, neck, chin, knuckles)? REJECT or MODIFY!\n"
                        "   - Crops must cut cleanly along tapering long-bone muscle shafts (mid-forearm, mid-bicep, mid-thigh, mid-calf, mid-chest) or keep the complete limb!\n"
                        "   - Are hands, fingers, or props (food, toys) amputated? If hands are holding something (e.g. food, props), the entire hand and object must be whole and complete!\n"
                        "3. 👀 LEAD ROOM & FACING DIRECTION (MANDATORY FRAMING LAW):\n"
                        "   - Look at the subject's torso orientation and gaze direction.\n"
                        "   - There MUST be MORE (or at minimum EQUAL) breathing space in front of where the body is turned / looking than behind their back!\n"
                        "   - If the subject faces LEFT: left margin must be larger than right margin. Never leave more space behind their back!\n"
                        "   - If the subject faces RIGHT: right margin must be larger than left margin. Never leave more space behind their back!\n"
                        "   - If the back has more empty space than the front: action = 'modify' (shift frame towards gaze direction) or 'reject'!\n"
                        "4. ⚖️ INDEPENDENT VALUE & VISUAL HARMONY:\n"
                        "   - Does each preset offer a genuinely distinct, professional composition (e.g. intimate close-up vs hero portrait vs environmental scene)?\n"
                        "   - If a preset is redundant, poorly framed, or unbalanced, reject it!\n"
                        "5. 🎯 ACTIONS:\n"
                        "   • 'keep': Preset is well-balanced, complete, and aesthetically sound. Approve as is!\n"
                        "   • 'modify': Composition is promising but needs small margin adjustments (e.g. adjust_ct_delta = +40 to trim dead ceiling above head, or adjust_cl_delta = -20 to give elbow breathing room).\n"
                        "   • 'reject': Fundamental composition failure (excessive dead ceiling that ruins the portrait, awkward joint amputation, bisected limbs, or unusable framing). THIS PRESET WILL BE DELETED!\n\n"
                        "Return JSON format:\n"
                        "```json\n"
                        "{\n"
                        '  "overall_qc_summary": "Overall assessment of crop candidates",\n'
                        '  "candidates_verdicts": [\n'
                        '    {\n'
                        '      "id": "preset_id",\n'
                        '      "action": "keep | modify | reject",\n'
                        '      "critique": "Short 1-sentence critique",\n'
                        '      "adjust_cl_delta": 0,\n'
                        '      "adjust_cr_delta": 0,\n'
                        '      "adjust_ct_delta": 0,\n'
                        '      "adjust_cb_delta": 0\n'
                        '    }\n'
                        '  ]\n'
                        "}\n"
                        "```"
                    )

                    contents_qc = [prompt_qc, pil_to_jpeg_part(self.image, quality=80, max_size=512)]
                    for idx, (pr, thumb) in enumerate(zip(candidates_to_qc, cropped_thumbnails)):
                        contents_qc.append(f"--- CANDIDATE PRESET #{idx+1} [ID: '{pr['id']}'] | Name: '{pr['name']}' | Aspect: '{pr['aspect_name']}' ---")
                        contents_qc.append(pil_to_jpeg_part(thumb, quality=85, max_size=512))

                    config_qc = service.create_config(thinking_level="low")
                    client_qc = service.get_client(timeout=120.0)

                    qc_thoughts = []
                    full_qc_chunks = []
                    try:
                        qc_stream = client_qc.models.generate_content_stream(
                            model=service.model_name,
                            contents=contents_qc,
                            config=config_qc
                        )
                        for chunk in qc_stream:
                            if self._is_cancelled: return
                            if chunk.candidates:
                                for candidate in chunk.candidates:
                                    if candidate.content and candidate.content.parts:
                                        for part in candidate.content.parts:
                                            if getattr(part, 'thought', False) and part.text:
                                                self.thought_received.emit(part.text)
                                                qc_thoughts.append(part.text)
                                            elif getattr(part, 'text', None):
                                                full_qc_chunks.append(part.text)
                            elif getattr(chunk, 'text', None):
                                full_qc_chunks.append(chunk.text)
                        accumulated_qc_text = "".join(full_qc_chunks).strip()
                    except Exception as e_stream_qc:
                        qc_resp = client_qc.models.generate_content(
                            model=service.model_name,
                            contents=contents_qc,
                            config=config_qc
                        )
                        accumulated_qc_text = (qc_resp.text or "").strip()

                    if not self._is_cancelled:
                        full_qc_thoughts_str = "".join(qc_thoughts).strip()
                        self.thought_text += f"\n--- STEP 2 VISUAL QC THOUGHTS ---\n{full_qc_thoughts_str}\n"
                        self.reasoning_log.append({
                            "phase": "step2_visual_qc",
                            "thoughts": full_qc_thoughts_str
                        })

                        qc_data = extract_json_from_text(accumulated_qc_text)
                        verdicts_list = qc_data.get("candidates_verdicts", [])
                        verdict_map = {v.get("id"): v for v in verdicts_list if isinstance(v, dict)}

                        refined_presets = []
                        qc_report_lines = []

                        for pr in parsed_presets:
                            pid = pr['id']
                            if pid == 'full_frame':
                                refined_presets.append(pr)
                                continue

                            v = verdict_map.get(pid)
                            if not v:
                                # Keep as-is if model didn't return verdict for this specific ID
                                refined_presets.append(pr)
                                continue

                            action = str(v.get("action", "keep")).lower().strip()
                            critique = str(v.get("critique", "Approved.")).strip()

                            if action == "reject":
                                qc_report_lines.append(f"🗑️ Rejected '{pr['name']}': {critique}")
                                # Pruned! Do not add to refined_presets
                                continue
                            elif action == "modify":
                                d_cl = int(v.get("adjust_cl_delta", 0))
                                d_cr = int(v.get("adjust_cr_delta", 0))
                                d_ct = int(v.get("adjust_ct_delta", 0))
                                d_cb = int(v.get("adjust_cb_delta", 0))

                                pr['crop_left'] = int(np.clip(pr['crop_left'] + d_cl, 0, 480))
                                pr['crop_right'] = int(np.clip(pr['crop_right'] + d_cr, 0, 480))
                                pr['crop_top'] = int(np.clip(pr['crop_top'] + d_ct, 0, 480))
                                pr['crop_bottom'] = int(np.clip(pr['crop_bottom'] + d_cb, 0, 480))

                                # Re-enforce aspect ratio with adjusted margins
                                asp_r = pr.get('aspect_ratio')
                                if asp_r and asp_r > 0:
                                    pr['crop_left'], pr['crop_right'], pr['crop_top'], pr['crop_bottom'] = enforce_crop_aspect_ratio(
                                        pr['crop_left'], pr['crop_right'], pr['crop_top'], pr['crop_bottom'],
                                        self.orig_w, self.orig_h, asp_r, allow_orientation_flip=False
                                    )

                                refined_presets.append(pr)
                                qc_report_lines.append(f"✏️ Refined '{pr['name']}': {critique}")
                            else: # 'keep'
                                asp_r = pr.get('aspect_ratio')
                                if asp_r and asp_r > 0:
                                    pr['crop_left'], pr['crop_right'], pr['crop_top'], pr['crop_bottom'] = enforce_crop_aspect_ratio(
                                        pr['crop_left'], pr['crop_right'], pr['crop_top'], pr['crop_bottom'],
                                        self.orig_w, self.orig_h, asp_r, allow_orientation_flip=False
                                    )
                                refined_presets.append(pr)
                                qc_report_lines.append(f"✅ Approved '{pr['name']}': {critique}")

                        # If all cropped candidates were rejected, restore at least the first one with safe margins
                        if len(refined_presets) <= 1 and len(candidates_to_qc) > 0:
                            fallback_pr = candidates_to_qc[0]
                            fallback_pr['crop_left'] = max(0, fallback_pr['crop_left'] - 20)
                            fallback_pr['crop_right'] = max(0, fallback_pr['crop_right'] - 20)
                            refined_presets.insert(0, fallback_pr)
                            qc_report_lines.append(f"⚠️ Restored fallback candidate '{fallback_pr['name']}' with expanded safe margins.")

                        parsed_presets = refined_presets
                        if qc_report_lines:
                            explanation += "\n\n[Visual QC Inspection]:\n" + "\n".join(qc_report_lines)

                except Exception as e_qc:
                    print(f"Visual QC pass note: {e_qc}")
                    explanation += f" (Visual QC note: {e_qc})"

            # Ensure logs directory exists and write persistent log
            try:
                os.makedirs("logs", exist_ok=True)
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "image_size": [self.orig_w, self.orig_h],
                    "orientation": orientation_str,
                    "thinking_level": self.thinking_level,
                    "thoughts_step1": full_thought_step1 if 'full_thought_step1' in locals() else "",
                    "thoughts_step2_qc": full_qc_thoughts_str if 'full_qc_thoughts_str' in locals() else "",
                    "final_presets": parsed_presets,
                    "explanation": explanation
                }
                with open("logs/ai_crop_reasoning.log", "a", encoding="utf-8") as f_log:
                    f_log.write(f"\n{'='*70}\n")
                    f_log.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] AI Crop Presets Generation\n")
                    f_log.write(f"Dimensions: {self.orig_w}x{self.orig_h} | Orientation: {orientation_str}\n")
                    f_log.write(f"--- Step 1 Thoughts ---\n{log_entry['thoughts_step1']}\n")
                    f_log.write(f"--- Step 2 Visual QC Thoughts ---\n{log_entry['thoughts_step2_qc']}\n")
                    f_log.write(f"--- Result Presets ({len(parsed_presets)}) ---\n")
                    for p in parsed_presets:
                        f_log.write(f"  • {p['name']} ({p['aspect_name']}): L={p['crop_left']}, R={p['crop_right']}, T={p['crop_top']}, B={p['crop_bottom']}\n")
                    f_log.write(f"--- Explanation ---\n{explanation}\n")

                with open("logs/ai_crop_latest.json", "w", encoding="utf-8") as f_json:
                    json.dump(log_entry, f_json, indent=2, ensure_ascii=False)
            except Exception as e_log:
                print(f"Note: Could not write crop log: {e_log}")

            self.finished_presets.emit(parsed_presets, explanation)

        except Exception as e:
            self.error_occurred.emit(str(e))
