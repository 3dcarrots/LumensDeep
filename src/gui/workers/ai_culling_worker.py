import os
import json
import time
from typing import List, Dict, Any, Optional
from PySide6.QtCore import QThread, Signal
from PIL import Image
from google import genai
from google.genai import types
from core.filmstrip_cache import FilmstripCache

DEFAULT_AI_CULLING_CRITERIA = (
    "CRITICAL UNCOMPROMISING RULES FOR CULLING & RATING:\n\n"
    "1. ZERO TOLERANCE FOR MOTION-BLURRED FACES (TOP PRIORITY):\n"
    "   - Always check EXIF shutter speed and scrutinize all subjects' faces, facial skin, noses, lips, and eyes.\n"
    "   - When shutter speed is slow (e.g. 1/10s, 1/15s, 1/20s, 1/30s, 1/40s, 1/60s) or subjects are moving, motion blur on faces is very common.\n"
    "   - MANDATORY REJECTION (is_rejected: true, 1-2 Stars):\n"
    "     * IF FACES, FACIAL FEATURES, SKIN TEXTURES, OR HEADS ARE BLURRED OR SMEARED BY MOTION/SLOW SHUTTER: IT MUST BE MARKED AS is_rejected: true WITH RATING 1 OR 2 STARS!\n"
    "     * NEVER keep or rate 3-5 stars any photo where the face itself is softened or smeared by motion blur, no matter how cute the pose/smile is.\n"
    "   - SPECIFIC EXCEPTION FOR SPECULAR EYE CATCHLIGHTS (KEEP THESE!):\n"
    "     * If the face itself, skin texture, nose, and lips are crisp, sharp, and distinct, but motion blur is visible ONLY as a slight light streak / elongation in the tiny specular catchlights (highlights/reflections) inside the eyes — DO NOT REJECT!\n"
    "     * KEEP such photos (3 to 5 Stars), as specular eye catchlight motion looks artistic and attractive when facial features remain sharp.\n\n"
    "2. DO NOT PENALIZE FIXABLE RAW POST-PROCESSING PROPERTIES:\n"
    "   - NEVER lower a rating for White Balance / Color Tint (e.g. warm/yellow indoor lighting, cool shadows, green tint).\n"
    "   - NEVER lower a rating for moderate Under-exposure or Over-exposure (as long as RAW sensor data is not permanently clipped).\n"
    "   - NEVER lower a rating for flat contrast, muted saturation, or dark shadows.\n"
    "   - These are easily fixed in 1-click RAW post-processing! Judge ONLY unfixable optical capture qualities: focus sharpness on eyes, motion blur, facial expression/eye contact, and burst de-duplication.\n\n"
    "3. BURST SEQUENCES & REPEATED POSES (Aggressive De-duplication):\n"
    "   - When multiple photos capture the same pose or scene (very common with children & portraits), pick ONLY the best 1–2 sharpest frames.\n"
    "   - To qualify as a keeper in a burst sequence:\n"
    "     * 100% Tack-Sharp: Zero motion blur on eyes/face, front eye crisp and in focus.\n"
    "     * Gaze & Eyes: All subjects looking at the lens with wide-open, clear eyes.\n"
    "     * Genuine Expression: Natural, flattering smile/expression (reject mid-word mouth shapes, grimaces, crying, half-blinks).\n"
    "   - All other duplicate/weaker frames in the burst sequence MUST be marked is_rejected: true (1-2★) with reason 'Duplicate in burst - inferior sharpness/expression'.\n\n"
    "4. STAR RATING TIERS & EXPLICIT JUSTIFICATION IN 'reason':\n"
    "   - 5 Stars: Flawless Masterpiece. 100% tack-sharp focus, ZERO motion blur on faces, peak genuine expression, perfect moment.\n"
    "   - 4 Stars: Sharp Keeper. Clean and sharp on subjects, strong expression and composition (state what minor factor kept it from 5★).\n"
    "   - 3 Stars: Secondary/documentary scene without motion blur (ALWAYS explicitly state what flaw prevented 4★ or 5★, e.g. awkward mouth or subject looking away; NEVER for motion-blurred faces!).\n"
    "   - 1-2 Stars / Reject (is_rejected: true): ANY photo with motion-blurred faces, camera shake, missed focus, closed eyes, or duplicate redundant frame.\n\n"
    "5. PROACTIVE 3D DEPTH MAP & BOKEH RECOMMENDATION (depth_needed: true):\n"
    "   - PROACTIVELY set depth_needed: true for ALL keeper photos featuring:\n"
    "     * People, portraits, children, babies, couples, families, and pets where background Bokeh blur / depth of field will elevate the image.\n"
    "     * Indoor, domestic, or event scenes with busy/distracting backgrounds (doors, walls, high chairs, furniture, bystanders) that need subject isolation.\n"
    "     * Close-ups and macro subjects (cakes, flowers, objects).\n"
    "   - Set depth_needed: false ONLY for flat documents, distant landscapes at infinity, or rejected blurry shots."
)

RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw'}

class AICullingWorker(QThread):
    """
    Two-Phase AI Photo Culling Worker:
    - Phase 1 (Fast Initial Screening): Analyzes all photos at 1024px for composition, lighting, obvious blur, closed eyes.
    - Phase 2 (Keeper Micro-Sharpness Verification): Re-evaluates top candidate keepers (~10-20%) using a 50% scale
      central crop (25% trimmed from all 4 borders = 16x total pixel reduction from full sensor).
      Provides ultra-fast 4x speedup and 4x lower token cost while inspecting pin-sharp eye focus and micro-motion blur.
    * Note: Full 40MP+ resolution is NEVER sent to avoid latency and excessive token consumption.
    """
    photo_evaluated = Signal(str, int, bool, bool, str) # path, rating, is_rejected, depth_needed, reason
    progress_updated = Signal(int, int, str)            # current, total, status_text
    thought_received = Signal(str)                      # reasoning text
    phase_changed = Signal(int, str)                    # phase (1 or 2), description
    finished_success = Signal(list)                     # list of evaluated dicts
    error_occurred = Signal(str)                        # error message

    def __init__(
        self,
        photo_paths: List[str],
        user_criteria: str = "",
        api_key: str = "",
        model_name: str = "gemini-3.8-flash",
        thinking_level: str = "low",
        enable_two_phase: bool = True,
        run_only_phase_2: bool = False,
        parent=None,
        **kwargs
    ):
        super().__init__(parent)
        self.photo_paths = list(photo_paths)
        self.user_criteria = user_criteria.strip() or DEFAULT_AI_CULLING_CRITERIA
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name or "gemini-3.8-flash"
        if "thinking_budget" in kwargs and thinking_level == "low":
            tb = kwargs["thinking_budget"]
            self.thinking_level = "low" if tb <= 512 else ("high" if tb > 2048 else "medium")
        else:
            self.thinking_level = thinking_level or "low"
        self.enable_two_phase = enable_two_phase
        self.run_only_phase_2 = run_only_phase_2
        self.cache = FilmstripCache.get_instance()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if not self.photo_paths:
            self.finished_success.emit([])
            return

        if not self.api_key:
            self.error_occurred.emit("Google Gemini API Key is missing. Please enter your API key in AI Assistant dock.")
            return

        try:
            client = genai.Client(api_key=self.api_key)
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize Gemini Client: {e}")
            return

        total_photos = len(self.photo_paths)
        evaluated_dict = {} # path -> result dict

        # ==========================================
        # DIRECT PHASE 2 ONLY MODE
        # ==========================================
        if self.run_only_phase_2:
            self.phase_changed.emit(2, f"Phase 2: Micro-focus inspection on {total_photos} unverified photo(s) (50% center crop)...")
            p2_batch_size = 2
            for k_idx in range(0, total_photos, p2_batch_size):
                if self._is_cancelled:
                    break

                k_batch = self.photo_paths[k_idx:k_idx + p2_batch_size]
                p2_status = f"[Phase 2/2] Inspecting eye-focus {k_idx + 1}–{min(k_idx + p2_batch_size, total_photos)} of {total_photos} (50% center crop)..."
                self.progress_updated.emit(k_idx, total_photos, p2_status)

                try:
                    p2_results = self._evaluate_batch(
                        client=client,
                        batch_paths=k_batch,
                        offset=k_idx,
                        total=total_photos,
                        max_dim=2048, # High precision for magnified central subject crop
                        is_phase_2=True
                    )
                    for res in p2_results:
                        res['phase2_verified'] = True
                        evaluated_dict[res['path']] = res
                        self.photo_evaluated.emit(
                            res['path'],
                            res['rating'],
                            res['is_rejected'],
                            res['depth_needed'],
                            f"[Center 50% Verified] {res['reason']}"
                        )
                except Exception as e:
                    print(f"AICullingWorker Phase 2 Only: Error in batch {k_idx}: {e}")

            self.progress_updated.emit(total_photos, total_photos, f"✓ Phase 2 completed: {total_photos} photo(s) inspected at 50% center crop.")
            final_list = list(evaluated_dict.values())
            if not self._is_cancelled:
                self.finished_success.emit(final_list)
            return

        # ==========================================
        # PHASE 1: Fast Screening at 1024px
        # ==========================================
        self.phase_changed.emit(1, f"Phase 1: Fast Screening ({total_photos} photos at 1024px)...")
        batch_size = 4
        current_idx = 0

        for i in range(0, total_photos, batch_size):
            if self._is_cancelled:
                break

            batch_paths = self.photo_paths[i:i + batch_size]
            status_text = f"[Phase 1/2] Screening photos {i + 1}–{min(i + batch_size, total_photos)} of {total_photos} (1024px)..."
            self.progress_updated.emit(i, total_photos, status_text)

            try:
                batch_results = self._evaluate_batch(
                    client=client,
                    batch_paths=batch_paths,
                    offset=i,
                    total=total_photos,
                    max_dim=1024,
                    is_phase_2=False
                )
                for res in batch_results:
                    res['phase2_verified'] = False
                    evaluated_dict[res['path']] = res
                    self.photo_evaluated.emit(
                        res['path'],
                        res['rating'],
                        res['is_rejected'],
                        res['depth_needed'],
                        res['reason']
                    )
            except Exception as e:
                print(f"AICullingWorker Phase 1: Error in batch {i}: {e}")
                for p in batch_paths:
                    fallback_res = {
                        'path': p,
                        'rating': 3,
                        'is_rejected': False,
                        'depth_needed': False,
                        'phase2_verified': False,
                        'reason': f"Phase 1 evaluation error: {str(e)[:100]}"
                    }
                    evaluated_dict[p] = fallback_res
                    self.photo_evaluated.emit(p, 3, False, False, fallback_res['reason'])

            current_idx += len(batch_paths)
            self.progress_updated.emit(current_idx, total_photos, f"Phase 1: {current_idx}/{total_photos} screened")

        if self._is_cancelled:
            return

        # ==========================================
        # PHASE 2: Keeper Micro-Sharpness Verification (50% scale + Center Crop)
        # ==========================================
        if self.enable_two_phase:
            # Select all candidate photos (rated >= 3 and not rejected) for microscopic center-crop inspection
            keeper_paths = [
                p for p, res in evaluated_dict.items()
                if res['rating'] >= 3 and not res['is_rejected']
            ]

            if keeper_paths:
                num_keepers = len(keeper_paths)
                self.phase_changed.emit(2, f"Phase 2: Verifying micro-focus on {num_keepers} keeper photos (50% center crop, 16x pixel reduction)...")
                
                # Check keepers in smaller batches of 2-3 with 50% scale center crop
                p2_batch_size = 2
                for k_idx in range(0, num_keepers, p2_batch_size):
                    if self._is_cancelled:
                        break

                    k_batch = keeper_paths[k_idx:k_idx + p2_batch_size]
                    p2_status = f"[Phase 2/2] Inspecting eye-focus {k_idx + 1}–{min(k_idx + p2_batch_size, num_keepers)} of {num_keepers} keepers (50% center crop)..."
                    self.progress_updated.emit(k_idx, num_keepers, p2_status)

                    try:
                        p2_results = self._evaluate_batch(
                            client=client,
                            batch_paths=k_batch,
                            offset=k_idx,
                            total=num_keepers,
                            max_dim=2048, # High precision for magnified central subject crop
                            is_phase_2=True,
                            prev_evaluations=evaluated_dict
                        )
                        for res in p2_results:
                            res['phase2_verified'] = True
                            prev = evaluated_dict.get(res['path'], {})
                            # Preserve depth_needed if either Phase 1 or Phase 2 flagged it
                            if prev.get('depth_needed') and not res.get('is_rejected', False):
                                res['depth_needed'] = True
                            evaluated_dict[res['path']] = res
                            self.photo_evaluated.emit(
                                res['path'],
                                res['rating'],
                                res['is_rejected'],
                                res['depth_needed'],
                                res['reason']
                            )
                    except Exception as e:
                        print(f"AICullingWorker Phase 2: Error evaluating batch {k_idx}: {e}")

                    self.progress_updated.emit(min(k_idx + len(k_batch), num_keepers), num_keepers, p2_status)

                self.progress_updated.emit(num_keepers, num_keepers, f"✓ Phase 2 completed: {num_keepers} keepers verified at 50% center crop.")

        # Update cache with final results and mark phase2_verified
        for p, res in evaluated_dict.items():
            self.cache.update_photo_flags(
                p,
                rating=res['rating'],
                is_rejected=res['is_rejected'],
                depth_needed=res['depth_needed'],
                ai_notes=res['reason'],
                phase2_verified=bool(res.get('phase2_verified', False))
            )

        self.progress_updated.emit(total_photos, total_photos, f"✓ AI Culling completed: {total_photos} photos evaluated.")
        final_list = list(evaluated_dict.values())
        if not self._is_cancelled:
            self.finished_success.emit(final_list)

    def _evaluate_batch(
        self,
        client: genai.Client,
        batch_paths: List[str],
        offset: int,
        total: int,
        max_dim: int,
        is_phase_2: bool = False,
        prev_evaluations: Optional[Dict[str, dict]] = None
    ) -> List[dict]:
        contents = []
        files_info = []

        # Load images (Phase 1 = 1024px full frame, Phase 2 = 50% scale center crop)
        from core.filmstrip_cache import FilmstripCache
        cache = FilmstripCache.get_instance()

        for idx, path in enumerate(batch_paths):
            base_name = os.path.basename(path)
            pil_img = self._load_photo_for_ai(
                path,
                max_dim=max_dim,
                use_half_res=is_phase_2,
                crop_center=is_phase_2
            )
            if pil_img:
                rec = cache.get_photo(path)
                exif_parts = []
                is_slow_shutter = False
                
                # Compute computer-vision edge clarity / focus energy score
                edge_score = None
                try:
                    import cv2
                    import numpy as np
                    gray_arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
                    edge_score = float(cv2.Laplacian(gray_arr, cv2.CV_64F).var())
                except Exception:
                    pass
                
                if rec:
                    if rec.get('shutter'): exif_parts.append(f"Shutter: {rec['shutter']}")
                    if rec.get('iso'): exif_parts.append(f"ISO: {rec['iso']}")
                    if rec.get('aperture'): exif_parts.append(f"f/{rec['aperture']:.1f}")
                    if rec.get('focal'): exif_parts.append(f"{rec['focal']:.0f}mm")

                exif_label = f" | {' | '.join(exif_parts)}" if exif_parts else ""

                prev_info = ""
                if is_phase_2 and prev_evaluations:
                    prev = prev_evaluations.get(path, {})
                    p_rating = prev.get('rating')
                    p_reason = prev.get('reason')
                    if p_rating:
                        prev_info = f" | Phase 1 Preliminary Score: {p_rating}★ ('{p_reason}')"

                contents.append(f"PHOTO #{idx + 1}: {base_name}{exif_label}{prev_info}")
                contents.append(pil_img)
                files_info.append({
                    'index': idx + 1,
                    'filename': base_name,
                    'path': path
                })

        if not contents:
            return []

        if is_phase_2:
            system_instruction = (
                "You are an elite master photo director performing STRICT MICROSCOPIC FOCUS & EYE INSPECTION (Phase 2).\n"
                "The attached candidate photos are high-resolution CENTRAL CROPS (outer 25% borders trimmed away to magnify subjects, faces, and eyes).\n\n"
                "MANDATORY EVALUATION RULES:\n"
                "1. SCRUTINIZE FACIAL FEATURES, SKIN, AND EYES:\n"
                "   - MANDATORY REJECTION (is_rejected: true, 1-2★): If faces, facial skin, noses, or lips have motion blur, motion smear streaks, ghosting, or camera shake from slow shutter speed — REJECT IT!\n"
                "   - SPECIFIC EXCEPTION (KEEP 4-5★): If the face itself, skin texture, nose, and lips are sharp and clear, but motion blur is visible ONLY in the specular catchlights/reflections inside the eyes, DO NOT REJECT — KEEP IT AS A KEEPER!\n"
                "2. DO NOT PENALIZE FIXABLE RAW PROPERTIES: Never reduce ratings for warm/cool White Balance, lighting tint, moderate under/over exposure, or flat contrast (these are easily corrected in RAW development).\n"
                "3. STAR RATINGS:\n"
                "   - 5 Stars: Tack-sharp focus on primary eyes and face, zero motion blur on face, peak genuine expression.\n"
                "   - 4 Stars: Sharp, clean keeper with crisp focal plane on face (explain what minor point kept it from 5★).\n"
                "   - 3 Stars: Sharp secondary scene without motion blur (explain what flaw kept it from 4-5★; NEVER for motion-blurred faces).\n"
                "   - 1-2 Stars & is_rejected: true: Visible motion smear/blur on face or skin, camera shake, or missed focus.\n\n"
                "4. DETAILED JUSTIFICATION IN 'reason':\n"
                "   - Explain precisely why this final rating was given.\n"
                "   - If confirming keeper (4-5★): confirm that magnified crop reveals crisp facial features and pupils.\n"
                "   - If demoting/rejecting: explicitly state what visible motion smear or defect was discovered under magnification.\n\n"
                "5. 3D DEPTH MAP & BOKEH OPPORTUNITY (depth_needed: true):\n"
                "   - Set depth_needed: true for ALL keeper portraits, people, children, babies, pets, or foreground subjects where background Bokeh blur, subject separation, or 3D relighting will enhance the photo!\n\n"
                "OUTPUT JSON SCHEMA:\n"
                "Return JSON with key 'evaluations': list of objects with fields:\n"
                "- photo_index (integer)\n"
                "- filename (string)\n"
                "- face_sharpness ('TACK_SHARP' | 'ACCEPTABLE' | 'SOFT' | 'MOTION_BLURRED' | 'OUT_OF_FOCUS')\n"
                "- rating (integer from 1 to 5)\n"
                "- is_rejected (boolean)\n"
                "- depth_needed (boolean)\n"
                "- reason (clear, descriptive explanation explicitly stating the sharpness verdict and rating justification)"
            )
            prompt_text = (
                f"Perform microscopic focus and sharpness inspection on the central crops of the attached {len(files_info)} keeper photo(s) (50% scale, central region).\n"
                "Reject photos with motion-blurred faces; keep photos where motion blur is only in eye catchlights. Flag keeper portraits for 3D depth bokeh!\n"
                "Return JSON with 'evaluations': [{photo_index, filename, face_sharpness, rating, is_rejected, depth_needed, reason}]."
            )
        else:
            system_instruction = (
                "You are an elite, uncompromising professional photo editor and director of photography specializing in high-speed commercial culling, portrait de-duplication, and quality control.\n\n"
                f"{self.user_criteria}\n\n"
                "MANDATORY EVALUATION PROCEDURE:\n"
                "1. EYE & FACE SHARPNESS:\n"
                "   - Scrutinize subjects' faces and facial skin. If faces/skin are smeared by motion blur/slow shutter, MARK AS is_rejected: true (1-2★).\n"
                "   - If face and features are sharp, but motion blur is visible ONLY in specular eye catchlights/reflections, KEEP IT (3-5★)!\n"
                "2. DO NOT PENALIZE FIXABLE RAW PROPERTIES: Never reduce ratings for warm/cool White Balance, lighting tint, moderate under/over exposure, or flat contrast (easily corrected in RAW processing). Focus on optical sharpness and expression.\n"
                "3. BURST DE-DUPLICATION: Pick ONLY the single sharpest frame with genuine open-eyed expression, and mark weaker duplicate frames as is_rejected: true.\n"
                "4. EXPLICIT RATING JUSTIFICATION IN 'reason': Explain EXACTLY why this rating was given. If 3★, explain what flaw prevented 4★ or 5★. If 1-2★/Reject, state the fatal blur/focus defect.\n"
                "5. PROACTIVE 3D DEPTH MAP & BOKEH DETECTION (depth_needed: true):\n"
                "   - Set depth_needed: true for ALL keeper photos of people, children, babies, portraits, pets, or subjects where blurring the background (Bokeh) or 3D depth grading will enhance the image!\n\n"
                "6. OUTPUT FORMAT: Output valid JSON with an array named 'evaluations', where each item has:\n"
                "- photo_index (integer)\n"
                "- filename (string)\n"
                "- face_sharpness ('TACK_SHARP' | 'ACCEPTABLE' | 'SOFT' | 'MOTION_BLURRED' | 'OUT_OF_FOCUS')\n"
                "- rating (integer from 1 to 5)\n"
                "- is_rejected (boolean, true if marked for deletion, false if keep)\n"
                "- depth_needed (boolean)\n"
                "- reason (concise 1-2 sentence explanation explicitly justifying why this rating was given and why it was not higher/lower)"
            )
            prompt_text = (
                f"Please evaluate and compare the attached {len(files_info)} photo(s) (Photos #{offset+1} to #{min(offset+len(files_info), total)} of {total}).\n"
                "Reject motion-blurred faces, keep eye-catchlight blur exceptions, de-duplicate burst sequences, and flag keeper portraits for 3D Depth / Bokeh!\n"
                "Return JSON with 'evaluations': [{photo_index, filename, face_sharpness, rating, is_rejected, depth_needed, reason}]."
            )

        contents.append(prompt_text)

        # Phase 1 uses "low" for ultra-fast screening, Phase 2 uses "medium" for microscopic eye focus inspection
        current_thinking = "medium" if is_phase_2 else ("low" if self.thinking_level == "low" else self.thinking_level)
        if current_thinking not in ["low", "medium", "high"]:
            current_thinking = "low"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(
                thinking_level=current_thinking
            )
        )

        import time, random
        max_retries = 5
        response = None
        for attempt in range(1, max_retries + 1):
            if self._is_cancelled:
                return []
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                break
            except Exception as e:
                if self._is_cancelled:
                    return []
                err_str = str(e).lower()
                is_retryable = any(k in err_str for k in ["429", "resource_exhausted", "quota", "503", "500", "timeout", "connection", "overloaded"])
                if is_retryable and attempt < max_retries:
                    backoff = min(35.0, (3.5 * (2 ** (attempt - 1))) + random.uniform(0.5, 1.5))
                    self.progress_updated.emit(offset, total, f"⏳ Gemini Culling rate limit/busy. Cooling down for {backoff:.1f}s (Attempt {attempt}/{max_retries})...")
                    slept = 0.0
                    while slept < backoff and not self._is_cancelled:
                        time.sleep(0.1)
                        slept += 0.1
                    continue
                else:
                    raise e

        resp_text = response.text.strip()
        data = {}
        try:
            data = json.loads(resp_text)
        except Exception:
            if resp_text.startswith("```"):
                lines = resp_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                resp_text = "\n".join(lines).strip()
                data = json.loads(resp_text)

        evals_list = data.get('evaluations', [])
        results = []

        for info in files_info:
            p_idx = info['index']
            fname = info['filename']
            path = info['path']

            match = None
            for e in evals_list:
                if e.get('photo_index') == p_idx or e.get('filename') == fname:
                    match = e
                    break

            if match:
                rating = max(1, min(5, int(match.get('rating', 3))))
                is_rej = bool(match.get('is_rejected', False))
                dp_needed = bool(match.get('depth_needed', False))
                reason = str(match.get('reason', '')).strip()
                face_sharpness = str(match.get('face_sharpness', '')).upper().strip()

                # If non-rejected keeper photo (rating >= 3) featuring people/children/portraits, ensure 3D depth is active
                if not is_rej and rating >= 3 and not dp_needed:
                    reason_lower = reason.lower()
                    portrait_keywords = ["portrait", "baby", "child", "girl", "boy", "person", "people", "face", "smile", "toddler", "pet", "cake", "eyes", "pupil"]
                    if any(kw in reason_lower for kw in portrait_keywords):
                        dp_needed = True

                # Programmatic quality gate ONLY when model explicitly identified actual blur
                blur_keywords = [
                    "motion blur", "motion-blur", "motion smear", "motion-smeared",
                    "camera shake", "shutter blur", "out of focus", "misfocused",
                    "heavy blur", "severe blur", "double contour", "blurred face", "blurry face"
                ]
                reason_lower = reason.lower()
                has_blur_complaint = (
                    any(kw in reason_lower for kw in blur_keywords)
                    and not ("no motion blur" in reason_lower or "zero motion blur" in reason_lower or "not blurry" in reason_lower or "sharp" in reason_lower)
                )

                if face_sharpness in ["MOTION_BLURRED", "OUT_OF_FOCUS"] or has_blur_complaint:
                    rating = min(rating, 2)
                    is_rej = True
            else:
                rating = 3
                is_rej = False
                dp_needed = True # Default true for un-evaluated keeper candidates
                reason = "Evaluated by AI"

            if is_phase_2 and not reason.startswith("[Phase 2"):
                reason = f"[Phase 2 Micro-Focus] {reason}"

            results.append({
                'path': path,
                'rating': rating,
                'is_rejected': is_rej,
                'depth_needed': dp_needed,
                'reason': reason
            })

        return results

    def _load_photo_for_ai(
        self,
        path: str,
        max_dim: int = 1024,
        use_half_res: bool = False,
        crop_center: bool = False
    ) -> Optional[Image.Image]:
        """
        Loads, transposes EXIF orientation upright, and scales an image for Gemini token efficiency.
        """
        from PIL import Image, ImageOps

        # 0. Check 50% cached preview first (instant load from SSD/Disk cache in ~3ms!)
        from core.filmstrip_cache import FilmstripCache
        cache = FilmstripCache.get_instance()
        cached_preview_path = cache.get_cached_preview_path(path)
        if cached_preview_path and os.path.exists(cached_preview_path):
            try:
                img = Image.open(cached_preview_path)
                img = ImageOps.exif_transpose(img).convert('RGB')
                if crop_center:
                    w, h = img.size
                    img = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
                if img.width > max_dim or img.height > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                return img
            except Exception:
                pass

        ext_lower = os.path.splitext(path)[1].lower()

        # 1. RAW formats (fallback if not in cache yet)
        if ext_lower in RAW_EXTENSIONS:
            try:
                import rawpy
                with rawpy.imread(path) as raw:
                    # Fast embedded JPEG thumbnail
                    try:
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            import io
                            img = Image.open(io.BytesIO(bytes(thumb.data)))
                            img = ImageOps.exif_transpose(img).convert('RGB')
                            cache.save_preview_jpeg(path, img)
                            if crop_center:
                                w, h = img.size
                                img = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
                            if img.width > max_dim or img.height > max_dim:
                                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                            return img
                    except Exception:
                        pass
                    # Half size decode
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    img = ImageOps.exif_transpose(Image.fromarray(rgb)).convert('RGB')
                    cache.save_preview_jpeg(path, img)
                    if crop_center:
                        w, h = img.size
                        img = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
                    if img.width > max_dim or img.height > max_dim:
                        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    return img
            except Exception:
                pass

        # 2. Standard JPEG / PNG / TIFF
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert('RGB')
            cache.save_preview_jpeg(path, img)
            if crop_center:
                w, h = img.size
                img = img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            return img
        except Exception:
            return None
