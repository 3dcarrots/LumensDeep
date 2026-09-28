import os
import io
import re
import queue
from typing import Any, Optional, Tuple, List, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QListWidget, QListWidgetItem, QStyledItemDelegate, 
    QStyle, QPushButton, QSlider, QAbstractItemView,
    QProgressBar, QSizePolicy, QMenu, QComboBox, QMessageBox, QApplication,
    QInputDialog
)
from PySide6.QtGui import (
    QPixmap, QImage, QImageReader, QPainter, 
    QColor, QPen, QBrush, QFont, QPainterPath, QAction, QKeyEvent
)
from PySide6.QtCore import (
    Qt, Signal, QSize, QRectF, QThread, QObject, QMutex, QMutexLocker, QPoint
)

from core.filmstrip_cache import FilmstripCache

RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw'}
STD_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
ALL_IMAGE_EXTENSIONS = RAW_EXTENSIONS | STD_EXTENSIONS

class FolderScanThread(QThread):
    """
    Background worker that scans the folder with zero UI blocking.
    Checks SQLite cache for instant thumbnail and metadata retrieval.
    """
    batch_found = Signal(int, list)       # generation_id, list of item_data dicts
    scan_complete = Signal(int, list)     # generation_id, full list of item_data dicts

    def __init__(self, folder_path: str, generation_id: int, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.generation_id = generation_id
        self._is_cancelled = False
        self.cache = FilmstripCache.get_instance()

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        if not self.folder_path or not os.path.isdir(self.folder_path):
            self.scan_complete.emit(self.generation_id, [])
            return

        # 1. Fetch entire folder records from SQLite in one fast query
        cached_folder_data = self.cache.get_folder_cache(self.folder_path)

        # 2. Read LumenDeep subfolder contents in one fast call for O(1) in-memory checks
        lumen_files = set()
        lumen_dir = os.path.join(self.folder_path, "LumenDeep")
        if os.path.isdir(lumen_dir):
            try:
                lumen_files = set(os.listdir(lumen_dir))
            except Exception:
                pass

        # 3. Fast directory scan using os.scandir
        raw_files = {}
        std_files = {}
        other_img_files = []
        local_sidecars = set()
        file_stats = {}

        try:
            with os.scandir(self.folder_path) as it:
                for entry in it:
                    if self._is_cancelled:
                        return
                    if not entry.is_file():
                        continue

                    fname = entry.name
                    stem, ext = os.path.splitext(fname)
                    ext_lower = ext.lower()

                    if fname.endswith(".json") or stem.endswith("_preview"):
                        local_sidecars.add(fname)

                    # Ignore standalone sidecars
                    if stem.endswith("_depth") or stem.endswith("_normals") or stem.endswith("_subject") or stem.endswith("_preview"):
                        continue

                    stat = entry.stat()
                    file_stats[entry.path] = (stat.st_mtime, stat.st_size)

                    if ext_lower in RAW_EXTENSIONS:
                        raw_files[stem.lower()] = entry.path
                    elif ext_lower in {'.jpg', '.jpeg'}:
                        std_files[stem.lower()] = entry.path
                    elif ext_lower in STD_EXTENSIONS:
                        other_img_files.append(entry.path)
        except Exception:
            self.scan_complete.emit(self.generation_id, [])
            return

        if self._is_cancelled:
            return

        # Deduplication: if RAW exists, ignore JPG with same stem
        final_file_paths = list(raw_files.values())
        for stem_lower, jpg_path in std_files.items():
            if stem_lower not in raw_files:
                final_file_paths.append(jpg_path)

        final_file_paths.extend(other_img_files)

        # Natural sort
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

        final_file_paths.sort(key=natural_sort_key)

        # Build item dicts
        results = []
        batch = []
        for path in final_file_paths:
            if self._is_cancelled:
                return

            norm_path = os.path.abspath(path)
            base_name = os.path.basename(path)
            stem, _ = os.path.splitext(base_name)
            
            has_project = bool(
                f"{stem}.lumendeep" in lumen_files or
                f"{stem}.json" in lumen_files or
                f"{stem}_preview.jpg" in lumen_files or
                f"{stem}.lumendeep" in local_sidecars or
                f"{stem}.json" in local_sidecars or
                f"{stem}_preview.jpg" in local_sidecars
            )

            has_depth_map = bool(
                f"{stem}_depth.png" in lumen_files or
                f"{stem}.depth.png" in lumen_files or
                f"{stem}_depth.npy" in lumen_files or
                f"{stem}_depth.png" in local_sidecars or
                f"{stem}.depth.png" in local_sidecars
            )

            mtime, size = file_stats.get(path, (0.0, 0))
            
            # Check cache
            cached_rec = cached_folder_data.get(norm_path)
            cached_pixmap = None
            is_cache_valid = False
            rating = 0
            is_rejected = False
            depth_needed = False
            ai_notes = ""

            if cached_rec:
                rating = cached_rec.get('rating', 0) or 0
                is_rejected = bool(cached_rec.get('is_rejected', 0))
                depth_needed = bool(cached_rec.get('depth_needed', 0))
                ai_notes = cached_rec.get('ai_notes', "") or ""
                
                # Verify mtime to ensure cache freshness
                if abs(cached_rec.get('mtime', 0.0) - mtime) < 1.0 and cached_rec.get('thumb_file'):
                    cached_pixmap = self.cache.get_cached_pixmap(cached_rec['thumb_file'], folder=self.folder_path)
                    if cached_pixmap and not cached_pixmap.isNull():
                        is_cache_valid = True

            item_dict = {
                'path': norm_path,
                'has_project': has_project,
                'has_depth_map': has_depth_map,
                'mtime': mtime,
                'size': size,
                'rating': rating,
                'is_rejected': is_rejected,
                'depth_needed': depth_needed,
                'ai_notes': ai_notes,
                'cached_pixmap': cached_pixmap,
                'is_cached': is_cache_valid
            }

            results.append(item_dict)
            batch.append(item_dict)

            # Emit in progressive batches of 40 items
            if len(batch) >= 40:
                self.batch_found.emit(self.generation_id, list(batch))
                batch.clear()

        if batch and not self._is_cancelled:
            self.batch_found.emit(self.generation_id, list(batch))

        if not self._is_cancelled:
            self.scan_complete.emit(self.generation_id, results)


class ThumbnailLoaderThread(QThread):
    """
    Dedicated worker thread that sequentially generates high-res thumbnails
    smoothly in the background and saves them into the persistent disk cache.
    """
    thumbnail_ready = Signal(int, str, QPixmap, bool, dict) # generation_id, path, pixmap, has_project, metadata
    progress_updated = Signal(int, int, int)                # generation_id, current, total

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mutex = QMutex()
        self.queue = []
        self.generation_id = 0
        self.target_size = QSize(360, 270) # High-res base for dynamic scaling
        self._is_running = True
        self._is_cancelled = False
        self.cache = FilmstripCache.get_instance()

    def set_tasks(self, file_paths: list, generation_id: int):
        with QMutexLocker(self.mutex):
            self.generation_id = generation_id
            self.queue = list(file_paths)
            self._is_cancelled = False

    def add_task_priority(self, file_path: str, generation_id: int):
        with QMutexLocker(self.mutex):
            self.generation_id = generation_id
            if file_path not in self.queue:
                self.queue.insert(0, file_path)
            self._is_cancelled = False

    def cancel(self):
        with QMutexLocker(self.mutex):
            self._is_cancelled = True
            self.queue.clear()

    def stop(self):
        with QMutexLocker(self.mutex):
            self._is_running = False
            self._is_cancelled = True
            self.queue.clear()

    def run(self):
        from concurrent.futures import ThreadPoolExecutor
        
        def process_item(item_path, gen_id):
            if self._is_cancelled or gen_id != self.generation_id:
                return
            try:
                pixmap, has_project, meta, preview_pil = self._extract_thumbnail(item_path, self.target_size)
                
                # Save to disk cache (both 360x270 thumb & 50% preview JPEG)
                if pixmap and not pixmap.isNull() and os.path.exists(item_path):
                    try:
                        mtime = os.path.getmtime(item_path)
                        size = os.path.getsize(item_path)
                        self.cache.save_thumbnail_and_meta(
                            item_path, pixmap, mtime, size,
                            metadata=meta, has_project=has_project,
                            preview_img=preview_pil
                        )
                    except Exception:
                        pass

                with QMutexLocker(self.mutex):
                    if not self._is_cancelled and gen_id == self.generation_id:
                        self.thumbnail_ready.emit(gen_id, item_path, pixmap, has_project, meta)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=6) as executor:
            while self._is_running:
                batch = []
                gen_id = 0
                with QMutexLocker(self.mutex):
                    if not self._is_cancelled and self.queue:
                        gen_id = self.generation_id
                        batch = self.queue[:12]
                        del self.queue[:12]

                if not batch:
                    self.msleep(40)
                    continue

                futures = [executor.submit(process_item, p, gen_id) for p in batch]
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass

    def _extract_thumbnail(self, path: str, target_size: QSize) -> tuple[QPixmap, bool, dict, Optional[Any]]:
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path)
        stem, ext = os.path.splitext(base_name)
        ext_lower = ext.lower()
        meta = {}
        preview_pil = None

        # Check LumenDeep preview
        lumen_preview = os.path.join(dir_name, "LumenDeep", f"{stem}_preview.jpg")
        old_preview = os.path.join(dir_name, f"{stem}_preview.jpg")
        lumen_json = os.path.join(dir_name, "LumenDeep", f"{stem}.json")
        old_json = os.path.join(dir_name, f"{stem}.json")

        has_project = (
            os.path.exists(lumen_json) or 
            os.path.exists(old_json) or 
            os.path.exists(lumen_preview) or 
            os.path.exists(old_preview)
        )

        # 1. LumenDeep rendered preview or 50% half preview
        lumen_half = os.path.join(dir_name, "LumenDeep", f"{stem}_half.jpg")
        old_half = os.path.join(dir_name, f"{stem}_half.jpg")

        preview_path = (
            lumen_preview if os.path.exists(lumen_preview) else (
                old_preview if os.path.exists(old_preview) else (
                    lumen_half if os.path.exists(lumen_half) else (
                        old_half if os.path.exists(old_half) else None
                    )
                )
            )
        )
        if preview_path:
            try:
                from PIL import Image, ImageOps
                p_img = Image.open(preview_path)
                p_img = ImageOps.exif_transpose(p_img).convert('RGB')
                meta['width'] = p_img.width
                meta['height'] = p_img.height
                try:
                    from core.image_io import read_image_metadata
                    f_meta = read_image_metadata(path)
                    if f_meta:
                        meta.update(f_meta)
                except Exception:
                    pass
                q_data = p_img.tobytes("raw", "RGB")
                q_img = QImage(q_data, p_img.width, p_img.height, p_img.width * 3, QImage.Format_RGB888)
                pm = QPixmap.fromImage(q_img).scaled(
                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                return pm, has_project, meta, p_img
            except Exception:
                pass

        # 2. RAW embedded JPEG preview
        if ext_lower in RAW_EXTENSIONS:
            try:
                import rawpy
                with rawpy.imread(path) as raw:
                    try:
                        if hasattr(raw, 'sizes'):
                            meta['width'] = raw.sizes.raw_width
                            meta['height'] = raw.sizes.raw_height
                        thumb = raw.extract_thumb()
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            safe_bytes = bytes(thumb.data)
                            try:
                                import io
                                from PIL import Image, ImageOps
                                p_img = Image.open(io.BytesIO(safe_bytes))
                                p_img = ImageOps.exif_transpose(p_img).convert('RGB')
                                q_data = p_img.tobytes("raw", "RGB")
                                q_img = QImage(q_data, p_img.width, p_img.height, p_img.width * 3, QImage.Format_RGB888)
                                pm = QPixmap.fromImage(q_img).scaled(
                                    target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                                )
                                return pm, has_project, meta, p_img
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Fallback to half-size decode if no embedded thumb
                    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
                    from PIL import Image, ImageOps
                    p_img = ImageOps.exif_transpose(Image.fromarray(rgb)).convert('RGB')
                    q_data = p_img.tobytes("raw", "RGB")
                    q_img = QImage(q_data, p_img.width, p_img.height, p_img.width * 3, QImage.Format_RGB888)
                    pm = QPixmap.fromImage(q_img).scaled(
                        target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    return pm, has_project, meta, p_img
            except Exception:
                pass

        # 3. Standard image downscaled
        try:
            from PIL import Image, ImageOps
            p_img = Image.open(path)
            p_img = ImageOps.exif_transpose(p_img).convert('RGB')
            meta['width'] = p_img.width
            meta['height'] = p_img.height
            q_data = p_img.tobytes("raw", "RGB")
            q_img = QImage(q_data, p_img.width, p_img.height, p_img.width * 3, QImage.Format_RGB888)
            pm = QPixmap.fromImage(q_img).scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            return pm, has_project, meta, p_img
        except Exception:
            pass

        empty_pm = QPixmap(target_size)
        empty_pm.fill(QColor(35, 35, 35))
        return empty_pm, has_project, meta, None


class FilmstripItemDelegate(QStyledItemDelegate):
    """
    Renders dynamically scalable filmstrip thumbnail cards with:
    - Scaled preview image matching card height
    - Filename and extension badge
    - 'LD' project badge if LumenDeep project exists
    - Active / selected glow outline
    - Rating Stars badge (★ 1..5)
    - Red Reject badge (❌) and dimmed card styling for photos marked for deletion
    - 3D Depth Needed badge (🧊 3D)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_path = None
        self.card_size = QSize(110, 95)

    def set_active_path(self, path: str):
        self.active_path = path

    def set_card_size(self, size: QSize):
        self.card_size = size

    def sizeHint(self, option, index):
        return self.card_size

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        rect = option.rect
        file_path = index.data(Qt.UserRole)
        has_project = bool(index.data(Qt.UserRole + 1))
        pixmap = index.data(Qt.UserRole + 2)
        rating = int(index.data(Qt.UserRole + 3) or 0)
        is_rejected = bool(index.data(Qt.UserRole + 4))
        depth_needed = bool(index.data(Qt.UserRole + 5))
        has_depth_map = bool(index.data(Qt.UserRole + 7))
        base_name = os.path.basename(file_path) if file_path else ""

        is_active = bool(self.active_path and file_path and os.path.abspath(self.active_path) == os.path.abspath(file_path))
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hover = bool(option.state & QStyle.State_MouseOver)

        # Card geometry with margin
        card_rect = QRectF(rect.adjusted(3, 3, -3, -3))

        # Dynamic label height and font sizes
        label_h = min(24.0, max(16.0, card_rect.height() * 0.16))
        font_label_size = min(12, max(9, int(card_rect.height() * 0.10)))

        # Background & Border styling
        bg_path = QPainterPath()
        bg_path.addRoundedRect(card_rect, 6, 6)

        if is_rejected:
            # Dimmed red-tinted background for rejected photos
            if is_active:
                painter.fillPath(bg_path, QBrush(QColor(120, 20, 20, 90)))
                painter.setPen(QPen(QColor(255, 60, 60), 2.5))
            elif is_selected:
                painter.fillPath(bg_path, QBrush(QColor(90, 20, 20, 110)))
                painter.setPen(QPen(QColor(255, 90, 90), 2.0))
            else:
                painter.fillPath(bg_path, QBrush(QColor(50, 15, 15, 140)))
                painter.setPen(QPen(QColor(180, 50, 50, 180), 1.0))
            painter.drawRoundedRect(card_rect, 6, 6)
        elif is_active:
            # Active photo glowing highlight
            painter.fillPath(bg_path, QBrush(QColor(0, 150, 255, 50 if is_selected else 40)))
            painter.setPen(QPen(QColor(0, 220, 255), 2.5))
            painter.drawRoundedRect(card_rect, 6, 6)
        elif is_selected:
            # Multi-selected photo highlight
            painter.fillPath(bg_path, QBrush(QColor(0, 120, 215, 45)))
            painter.setPen(QPen(QColor(50, 160, 255), 2.0))
            painter.drawRoundedRect(card_rect, 6, 6)
        elif is_hover:
            painter.fillPath(bg_path, QBrush(QColor(48, 48, 48)))
            painter.setPen(QPen(QColor(80, 80, 80), 1.0))
            painter.drawRoundedRect(card_rect, 6, 6)
        else:
            painter.fillPath(bg_path, QBrush(QColor(28, 28, 28)))
            painter.setPen(QPen(QColor(45, 45, 45), 1.0))
            painter.drawRoundedRect(card_rect, 6, 6)

        # Draw Thumbnail
        thumb_area = QRectF(
            card_rect.left() + 4, 
            card_rect.top() + 4, 
            card_rect.width() - 8, 
            card_rect.height() - label_h - 6
        )

        if pixmap and isinstance(pixmap, QPixmap) and not pixmap.isNull():
            pm_scaled = pixmap.scaled(
                int(thumb_area.width()), 
                int(thumb_area.height()), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            x_offset = thumb_area.left() + (thumb_area.width() - pm_scaled.width()) / 2.0
            y_offset = thumb_area.top() + (thumb_area.height() - pm_scaled.height()) / 2.0
            
            img_rect = QRectF(x_offset, y_offset, pm_scaled.width(), pm_scaled.height())
            painter.fillRect(img_rect, QColor(15, 15, 15))
            
            if is_rejected:
                painter.setOpacity(0.55) # Dim thumbnail if rejected
            painter.drawPixmap(int(x_offset), int(y_offset), pm_scaled)
            painter.setOpacity(1.0)
            
            painter.setPen(QPen(QColor(50, 50, 50, 150), 1.0))
            painter.drawRect(img_rect)
        else:
            painter.setPen(QPen(QColor(90, 90, 90)))
            font_loading = QFont(painter.font())
            font_loading.setPixelSize(min(16, max(9, int(card_rect.height() * 0.12))))
            painter.setFont(font_loading)
            painter.drawText(thumb_area, Qt.AlignCenter, "⏳")

        # Badges geometry
        badge_w = min(36.0, max(22.0, card_rect.width() * 0.22))
        badge_h = min(20.0, max(13.0, card_rect.height() * 0.14))
        font_badge_size = min(10, max(8, int(badge_h * 0.65)))

        # 1. Reject Badge ('❌')
        if is_rejected:
            rej_w = min(40.0, max(24.0, card_rect.width() * 0.24))
            rej_rect = QRectF(card_rect.left() + 5, card_rect.top() + 5, rej_w, badge_h)
            rej_path = QPainterPath()
            rej_path.addRoundedRect(rej_rect, 3, 3)
            painter.fillPath(rej_path, QBrush(QColor(220, 40, 40, 230)))
            
            painter.setPen(QColor(255, 255, 255))
            font_rej = QFont(painter.font())
            font_rej.setPixelSize(font_badge_size)
            font_rej.setBold(True)
            painter.setFont(font_rej)
            painter.drawText(rej_rect, Qt.AlignCenter, "❌")
        elif has_project:
            # Draw Project Badge ('LD') if LumenDeep project exists
            badge_rect = QRectF(card_rect.left() + 5, card_rect.top() + 5, badge_w, badge_h)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(badge_rect, 3, 3)
            painter.fillPath(badge_path, QBrush(QColor(0, 160, 230, 220)))
            
            painter.setPen(QColor(255, 255, 255))
            font_badge = QFont(painter.font())
            font_badge.setPixelSize(font_badge_size)
            font_badge.setBold(True)
            painter.setFont(font_badge)
            painter.drawText(badge_rect, Qt.AlignCenter, "LD")

        # 2. RAW Badge (Top-Right)
        _, ext = os.path.splitext(base_name)
        if ext.lower() in RAW_EXTENSIONS:
            raw_rect = QRectF(card_rect.right() - badge_w - 5, card_rect.top() + 5, badge_w, badge_h)
            raw_path = QPainterPath()
            raw_path.addRoundedRect(raw_rect, 3, 3)
            painter.fillPath(raw_path, QBrush(QColor(60, 60, 60, 210)))
            
            painter.setPen(QColor(220, 220, 220))
            font_raw = QFont(painter.font())
            font_raw.setPixelSize(font_badge_size)
            font_raw.setBold(True)
            painter.setFont(font_raw)
            painter.drawText(raw_rect, Qt.AlignCenter, "RAW")

        # 3. Rating Stars Badge (Bottom-Left above filename)
        if rating > 0:
            star_w = min(42.0, max(26.0, card_rect.width() * 0.26))
            star_rect = QRectF(card_rect.left() + 5, thumb_area.bottom() - badge_h - 2, star_w, badge_h)
            star_path = QPainterPath()
            star_path.addRoundedRect(star_rect, 3, 3)
            painter.fillPath(star_path, QBrush(QColor(20, 20, 20, 210)))
            painter.setPen(QPen(QColor(255, 190, 0, 180), 1.0))
            painter.drawRoundedRect(star_rect, 3, 3)

            painter.setPen(QColor(255, 205, 0))
            font_star = QFont(painter.font())
            font_star.setPixelSize(font_badge_size)
            font_star.setBold(True)
            painter.setFont(font_star)
            painter.drawText(star_rect, Qt.AlignCenter, f"★ {rating}")

        # 4. Depth Badges (Bottom-Right above filename)
        if has_depth_map:
            # Solid Cyan/Teal Badge with Checkmark: Depth map is already generated and ready!
            dp_w = min(46.0, max(30.0, card_rect.width() * 0.29))
            dp_rect = QRectF(card_rect.right() - dp_w - 5, thumb_area.bottom() - badge_h - 2, dp_w, badge_h)
            dp_path = QPainterPath()
            dp_path.addRoundedRect(dp_rect, 3, 3)
            painter.fillPath(dp_path, QBrush(QColor(0, 165, 220, 240)))
            painter.setPen(QPen(QColor(0, 225, 255), 1.0))
            painter.drawRoundedRect(dp_rect, 3, 3)
            
            painter.setPen(QColor(255, 255, 255))
            font_dp = QFont(painter.font())
            font_dp.setPixelSize(font_badge_size)
            font_dp.setBold(True)
            painter.setFont(font_dp)
            painter.drawText(dp_rect, Qt.AlignCenter, "✓ 3D")
        elif depth_needed:
            # Distinct Warm Amber / Gold Badge: Marked by AI / User as Needed (Pending calculation)
            dp_w = min(48.0, max(32.0, card_rect.width() * 0.30))
            dp_rect = QRectF(card_rect.right() - dp_w - 5, thumb_area.bottom() - badge_h - 2, dp_w, badge_h)
            dp_path = QPainterPath()
            dp_path.addRoundedRect(dp_rect, 3, 3)
            painter.fillPath(dp_path, QBrush(QColor(180, 85, 0, 220)))
            painter.setPen(QPen(QColor(255, 170, 0), 1.0))
            painter.drawRoundedRect(dp_rect, 3, 3)
            
            painter.setPen(QColor(255, 255, 255))
            font_dp = QFont(painter.font())
            font_dp.setPixelSize(font_badge_size)
            font_dp.setBold(True)
            painter.setFont(font_dp)
            painter.drawText(dp_rect, Qt.AlignCenter, "🧊 +3D")

        # 5. Filename text at the bottom
        label_rect = QRectF(
            card_rect.left() + 2, 
            card_rect.bottom() - label_h - 1, 
            card_rect.width() - 4, 
            label_h
        )
        if is_rejected:
            painter.setPen(QColor(220, 100, 100))
        elif is_active:
            painter.setPen(QColor(0, 210, 255))
        else:
            painter.setPen(QColor(180, 180, 180))
            
        font_label = QFont(painter.font())
        font_label.setPixelSize(font_label_size)
        if is_active:
            font_label.setBold(True)
        painter.setFont(font_label)

        metrics = painter.fontMetrics()
        elided_name = metrics.elidedText(base_name, Qt.ElideMiddle, int(label_rect.width()))
        painter.drawText(label_rect, Qt.AlignCenter, elided_name)

        painter.restore()


MODE_SINGLE_ROW = "single_row"
MODE_GRID = "grid"

class FilmstripWidget(QWidget):
    """
    Bottom Filmstrip / Thumbnail list showing photos from the selected folder.
    Features:
    - Persistent SQLite & disk caching for instant folder loading
    - Culling controls: Ratings (1-5 stars), Rejection flag (❌), 3D Depth Needed (🧊)
    - Filtering: Hide rejected photos, Filter by rating, Filter by 3D Depth
    - AI Batch Culling & Assessment dialog trigger
    - Multi-Selection (Shift + click, Ctrl + click) for batch AI & Export
    - Two View Modes: Single-Row Ribbon vs Multi-Row Grid with zoom slider
    - Hotkeys: 1-5 (ratings), 0/U (clear), X/Del (reject), D (depth)
    """
    image_selected = Signal(str)
    ai_culling_requested = Signal()
    full_auto_requested = Signal()
    stop_batch_requested = Signal()
    aspect_ratio_changed = Signal(str)
    swap_aspect_requested = Signal()
    photo_notes_updated = Signal(str, str) # (path, notes)
    batch_ai_requested = Signal(list)       # list of photo dicts
    batch_crop_requested = Signal(object)   # optional list of photo dicts (or None)
    batch_export_requested = Signal(list)   # list of photo dicts
    batch_denoise_requested = Signal(list)  # list of photo dicts or filepaths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = ""
        self.active_image_path = None
        self.view_mode = MODE_SINGLE_ROW
        self.generation_id = 0
        self.total_photos_count = 0
        self.loaded_photos_count = 0
        self.cache = FilmstripCache.get_instance()

        # Filtering state
        self.filter_hide_rejected = False
        self.filter_rating_min = 0 # 0 = All
        self.filter_depth_only = False

        # Background scan thread
        self.scan_thread = None

        # Background thumbnail loader thread
        self.loader_thread = ThumbnailLoaderThread(self)
        self.loader_thread.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.loader_thread.start()

        self._init_ui()

    def closeEvent(self, event):
        if hasattr(self, 'loader_thread') and self.loader_thread:
            self.loader_thread.stop()
            self.loader_thread.wait(300)
        if hasattr(self, 'scan_thread') and self.scan_thread:
            self.scan_thread.cancel()
            self.scan_thread.wait(300)
        super().closeEvent(event)

    def sizeHint(self):
        return QSize(400, 160)

    def minimumSizeHint(self):
        return QSize(100, 90)



    def update_thumbnail_pixmap(self, path: str, pixmap: QPixmap, preview_img: Optional[Any] = None):
        """Updates thumbnail pixmap and sets project badge for edited photo in real-time."""
        norm_path = os.path.abspath(path)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item_p = item.data(Qt.UserRole)
            if item_p and os.path.abspath(item_p) == norm_path:
                item.setData(Qt.UserRole + 1, True) # has_project
                item.setData(Qt.UserRole + 2, pixmap)
                self.list_widget.update(self.list_widget.visualItemRect(item))
                break
        
        # Persist to local folder's LumenDeep cache
        if pixmap and not pixmap.isNull():
            try:
                mtime = os.path.getmtime(norm_path) if os.path.exists(norm_path) else time.time()
                size = os.path.getsize(norm_path) if os.path.exists(norm_path) else 0
                self.cache.save_thumbnail_and_meta(
                    norm_path, pixmap, mtime, size, has_project=True, preview_img=preview_img
                )
            except Exception:
                pass

    def _init_ui(self):
        self.setMinimumHeight(95)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(0)

        # Thin sleek progress bar at the top of panel
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #222222;
                height: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0096ff, stop:1 #00d2ff);
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Header / Toolbar bar
        header_widget = QWidget()
        header_bar = QHBoxLayout(header_widget)
        header_bar.setContentsMargins(6, 2, 6, 2)
        header_bar.setSpacing(6)

        self.lbl_folder_info = QLabel("No folder selected")
        self.lbl_folder_info.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        header_bar.addWidget(self.lbl_folder_info)

        self.lbl_count_info = QLabel("")
        self.lbl_count_info.setStyleSheet("color: #888; font-size: 10px;")
        header_bar.addWidget(self.lbl_count_info)

        header_bar.addStretch()

        # Toolbar Filters
        btn_style = """
            QPushButton {
                background-color: #252b33;
                color: #d0d7de;
                border: 1px solid #3c444d;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #00d2ff;
                color: #ffffff;
            }
            QPushButton:checked {
                background-color: #1a3b5c;
                border-color: #00d2ff;
                color: #00d2ff;
                font-weight: bold;
            }
        """

        # Hide Rejected Toggle
        self.btn_filter_reject = QPushButton("🚫 Hide Rejected")
        self.btn_filter_reject.setCheckable(True)
        self.btn_filter_reject.setStyleSheet(btn_style)
        self.btn_filter_reject.setToolTip("Hide photos marked for deletion (Press 'X' or 'Del' on photo to mark)")
        self.btn_filter_reject.toggled.connect(self._on_filter_reject_toggled)
        header_bar.addWidget(self.btn_filter_reject)

        # Rating Filter Dropdown
        self.combo_filter_rating = QComboBox()
        self.combo_filter_rating.addItems(["⭐ All Stars", "★ 1+ Stars", "★ 2+ Stars", "★ 3+ Stars", "★ 4+ Stars", "★ 5 Stars", "★ Unrated (0)"])
        self.combo_filter_rating.setStyleSheet("""
            QComboBox {
                background-color: #252b33;
                color: #d0d7de;
                border: 1px solid #3c444d;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }
            QComboBox:hover {
                border-color: #00d2ff;
            }
            QComboBox QAbstractItemView {
                background-color: #1e242b;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
        """)
        self.combo_filter_rating.currentIndexChanged.connect(self._on_filter_rating_changed)
        header_bar.addWidget(self.combo_filter_rating)

        # Depth Needed Only Toggle
        self.btn_filter_depth = QPushButton("🧊 Depth Only")
        self.btn_filter_depth.setCheckable(True)
        self.btn_filter_depth.setStyleSheet(btn_style)
        self.btn_filter_depth.setToolTip("Show only photos flagged for 3D Depth / Normals / Subject generation (Press 'D' to toggle)")
        self.btn_filter_depth.toggled.connect(self._on_filter_depth_toggled)
        header_bar.addWidget(self.btn_filter_depth)

        # AI Batch Culling Button
        self.btn_ai_culling = QPushButton("✨ Gemini AI Culling")
        self.btn_ai_culling.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #ffffff;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2b5280;
                border-color: #33e0ff;
            }
        """)
        self.btn_ai_culling.setToolTip("Let Gemini 3.8 Flash review and cull all photos in this folder based on custom criteria")
        self.btn_ai_culling.clicked.connect(self._on_ai_culling_clicked)
        header_bar.addWidget(self.btn_ai_culling)

        # Full Auto Pipeline Button
        self.btn_full_auto = QPushButton("🚀 Full Auto")
        self.btn_full_auto.setStyleSheet("""
            QPushButton {
                background-color: #3b1b54;
                color: #ffffff;
                border: 1px solid #aa00ff;
                border-radius: 4px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #5a2880;
                border-color: #d500f9;
            }
        """)
        self.btn_full_auto.setToolTip("1-Click Unified Process: AI Culling -> Auto-select top photos (Rating ≥ 4★) -> 3D Depth/Background -> Multi-Pass AI Auto-Correction -> Auto-Denoise (ISO > 200) & Export")
        self.btn_full_auto.clicked.connect(lambda: self.full_auto_requested.emit())
        header_bar.addWidget(self.btn_full_auto)

        # Batch AI Crop Button
        self.btn_batch_crop = QPushButton("📐 AI Crop All")
        self.btn_batch_crop.setStyleSheet("""
            QPushButton {
                background-color: #1b3d36;
                color: #ffffff;
                border: 1px solid #00e676;
                border-radius: 4px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #25574c;
                border-color: #69f0ae;
            }
        """)
        self.btn_batch_crop.setToolTip("Generate AI Fine-Art Crop Presets for all photos in the filmstrip (deletes old presets, preserves color grading)")
        self.btn_batch_crop.clicked.connect(lambda: self.batch_crop_requested.emit(None))
        header_bar.addWidget(self.btn_batch_crop)

        # Batch Aspect Ratio Selector
        self.combo_crop_aspect = QComboBox()
        self.combo_crop_aspect.setToolTip("Target Crop Aspect Ratio for Batch Processing")
        self.combo_crop_aspect.setStyleSheet("background-color: #1a222d; color: #ffffff; border: 1px solid #3b4d61; border-radius: 4px; padding: 2px 4px; font-size: 11px;")
        self.combo_crop_aspect.addItems([
            "Free (Custom)",
            "Original (Sensor)",
            "1:1 (Square / Instagram)",
            "4:5 / 5:4 (Instagram / 8×10\")",
            "3:4 / 4:3 (Photo 6×8\" / iPad)",
            "2:3 / 3:2 (35mm / 4×6\" / 10×15cm)",
            "9:16 / 16:9 (Stories / Reels / TV)",
            "3.5:4.5 / 4.5:3.5 (Passport / ID)",
            "5:7 / 7:5 (Photo 13×18cm)",
            "11:14 / 14:11 (Large Print)"
        ])
        self.combo_crop_aspect.currentTextChanged.connect(lambda t: self.aspect_ratio_changed.emit(t))
        header_bar.addWidget(self.combo_crop_aspect)

        self.btn_crop_swap = QPushButton("🔄")
        self.btn_crop_swap.setFixedWidth(24)
        self.btn_crop_swap.setToolTip("Swap Aspect Ratio Orientation (Landscape ↔ Portrait)")
        self.btn_crop_swap.setStyleSheet("""
            QPushButton {
                background-color: #1a222d;
                border: 1px solid #3b4d61;
                border-radius: 4px;
                color: #00d2ff;
                font-weight: bold;
                padding: 2px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #24303f;
                border-color: #00d2ff;
            }
        """)
        self.btn_crop_swap.clicked.connect(lambda: self.swap_aspect_requested.emit())
        header_bar.addWidget(self.btn_crop_swap)

        # Stop Batch Process Button
        self.btn_stop_batch = QPushButton("⏹ Stop Batch (Esc)")
        self.btn_stop_batch.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: #ffffff;
                border: 1px solid #ff3333;
                border-radius: 4px;
                padding: 2px 10px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #b30000;
                border-color: #ff6666;
            }
        """)
        self.btn_stop_batch.setToolTip("Instantly stop any active AI Culling or Batch Auto-Correction (Esc)")
        self.btn_stop_batch.clicked.connect(lambda: self.stop_batch_requested.emit())
        self.btn_stop_batch.setVisible(False)
        header_bar.addWidget(self.btn_stop_batch)

        # Thumbnail Size Zoom Slider
        lbl_zoom = QLabel("🔍")
        lbl_zoom.setStyleSheet("color: #888; font-size: 11px; margin-left: 4px;")
        # View Mode Toggle Button: Single Row Ribbon vs Multi-Row Grid
        self.btn_view_mode = QPushButton("⊞ Grid View")
        self.btn_view_mode.setStyleSheet(btn_style)
        self.btn_view_mode.setToolTip("Toggle between Single-Row horizontal ribbon and Multi-Row vertical grid view")
        self.btn_view_mode.clicked.connect(self._toggle_view_mode)
        header_bar.addWidget(self.btn_view_mode)

        # Thumbnail Size Zoom Slider (Visible only in Grid View Mode)
        self.lbl_zoom = QLabel("🔍")
        self.lbl_zoom.setStyleSheet("color: #888; font-size: 11px; margin-left: 4px;")
        self.slider_thumb_zoom = QSlider(Qt.Horizontal)
        self.slider_thumb_zoom.setRange(80, 360)
        self.slider_thumb_zoom.setValue(130)
        self.slider_thumb_zoom.setFixedWidth(80)
        self.slider_thumb_zoom.setToolTip("Adjust thumbnail card size in grid view")
        self.slider_thumb_zoom.valueChanged.connect(self._on_thumb_zoom_changed)
        
        self.lbl_zoom.setVisible(False)
        self.slider_thumb_zoom.setVisible(False)
        
        header_bar.addWidget(self.lbl_zoom)
        header_bar.addWidget(self.slider_thumb_zoom)

        layout.addWidget(header_widget)

        # QListWidget with Extended Multi-Selection (Shift + click, Ctrl + click)
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setFlow(QListWidget.LeftToRight)
        self.list_widget.setWrapping(False)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(4)
        self.list_widget.setMovement(QListWidget.Static)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        self.delegate = FilmstripItemDelegate(self)
        self.list_widget.setItemDelegate(self.delegate)

        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                outline: none;
                margin: 0px 2px;
            }
            QScrollBar:horizontal {
                height: 8px;
                background: #1e1e1e;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #3e3e3e;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #555555;
            }
            QScrollBar:vertical {
                width: 8px;
                background: #1e1e1e;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3e3e3e;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555555;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
                height: 0px;
            }
        """)

        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_clicked)
        self.list_widget.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.list_widget)

    def _toggle_view_mode(self):
        new_mode = MODE_GRID if self.view_mode == MODE_SINGLE_ROW else MODE_SINGLE_ROW
        self.set_view_mode(new_mode)

    def set_view_mode(self, mode: str):
        self.view_mode = mode
        is_grid = (mode == MODE_GRID)

        self.lbl_zoom.setVisible(is_grid)
        self.slider_thumb_zoom.setVisible(is_grid)

        if is_grid:
            self.btn_view_mode.setText("═ Single Row")
            self.btn_view_mode.setToolTip("Switch to Single-Row horizontal filmstrip ribbon")
            self.list_widget.setWrapping(True)
            self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self._apply_grid_zoom(self.slider_thumb_zoom.value())
        else:
            self.btn_view_mode.setText("⊞ Grid View")
            self.btn_view_mode.setToolTip("Switch to Multi-Row vertical scrolling grid")
            self.list_widget.setWrapping(False)
            self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._update_single_row_dimensions()

    def _on_thumb_zoom_changed(self, val: int):
        if self.view_mode == MODE_GRID:
            self._apply_grid_zoom(val)

    def _apply_grid_zoom(self, val: int):
        card_h = int(val)
        card_w = max(80, int(card_h * 1.20))
        new_size = QSize(card_w, card_h)
        self.delegate.set_card_size(new_size)
        self.list_widget.setGridSize(new_size)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                item.setSizeHint(new_size)
        self.list_widget.doItemsLayout()
        self.list_widget.viewport().update()

    def _update_single_row_dimensions(self):
        viewport_h = self.list_widget.viewport().height()
        if viewport_h <= 10:
            viewport_h = max(70, self.list_widget.height() - 14)

        card_h = max(65, min(viewport_h - 4, 800))
        card_w = max(75, int(card_h * 1.18))

        new_size = QSize(card_w, card_h)
        if self.delegate.card_size != new_size:
            self.delegate.set_card_size(new_size)
            self.list_widget.setGridSize(new_size)
            
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item:
                    item.setSizeHint(new_size)

            self.list_widget.doItemsLayout()
            self.list_widget.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.view_mode == MODE_SINGLE_ROW:
            self._update_single_row_dimensions()
        else:
            self._apply_grid_zoom(self.slider_thumb_zoom.value())

    def keyPressEvent(self, event: QKeyEvent):
        """Handle culling shortcuts when filmstrip has focus."""
        if self.handle_culling_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_culling_key(self, key: int) -> bool:
        """Centralized shortcut handler for star rating, reject flag, and depth flag (supports multi-selection)."""
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            item = self.list_widget.currentItem()
            if item:
                selected_items = [item]
        if not selected_items:
            return False

        paths = [item.data(Qt.UserRole) for item in selected_items if item.data(Qt.UserRole)]
        if not paths:
            return False

        if key in [Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_0, Qt.Key_U]:
            rating_map = {Qt.Key_1: 1, Qt.Key_2: 2, Qt.Key_3: 3, Qt.Key_4: 4, Qt.Key_5: 5, Qt.Key_0: 0, Qt.Key_U: 0}
            r = rating_map[key]
            for p in paths:
                self.set_photo_rating(p, r)
            return True
        elif key in [Qt.Key_X, Qt.Key_Delete]:
            any_not_rej = any(not bool(item.data(Qt.UserRole + 4)) for item in selected_items)
            for p in paths:
                self.set_photo_rejected(p, any_not_rej)
            return True
        elif key == Qt.Key_D:
            any_not_dp = any(not bool(item.data(Qt.UserRole + 5)) for item in selected_items)
            for p in paths:
                self.set_photo_depth_needed(p, any_not_dp)
            return True
        return False

    def load_folder(self, folder_path: str):
        """Asynchronously and smoothly scan and load photos with instant cache display."""
        if not folder_path or not os.path.isdir(folder_path):
            return

        self.generation_id += 1
        current_gen = self.generation_id

        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.cancel()
            self.scan_thread.wait(50)

        self.loader_thread.cancel()

        self.current_folder = os.path.abspath(folder_path)
        self.lbl_folder_info.setText(f"📁 {os.path.basename(self.current_folder) or self.current_folder}")
        self.lbl_count_info.setText("Loading cache...")
        
        self.total_photos_count = 0
        self.loaded_photos_count = 0
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.list_widget.clear()

        # Start background scan thread
        self.scan_thread = FolderScanThread(self.current_folder, current_gen, self)
        self.scan_thread.batch_found.connect(self._on_scan_batch)
        self.scan_thread.scan_complete.connect(self._on_scan_complete)
        self.scan_thread.start()

    load_directory = load_folder

    def _on_scan_batch(self, generation_id: int, items_batch: list):
        if generation_id != self.generation_id:
            return

        self.list_widget.setUpdatesEnabled(False)
        cur_card_size = self.delegate.card_size

        for d in items_batch:
            item = QListWidgetItem()
            item.setSizeHint(cur_card_size)
            path = d['path']
            item.setData(Qt.UserRole, path)
            item.setData(Qt.UserRole + 1, d.get('has_project', False))
            item.setData(Qt.UserRole + 2, d.get('cached_pixmap'))
            item.setData(Qt.UserRole + 3, d.get('rating', 0))
            item.setData(Qt.UserRole + 4, d.get('is_rejected', False))
            item.setData(Qt.UserRole + 5, d.get('depth_needed', False))
            item.setData(Qt.UserRole + 6, d.get('ai_notes', ""))
            item.setData(Qt.UserRole + 7, d.get('has_depth_map', False))

            # Apply initial visibility based on current filters
            is_visible = self._matches_filter(
                d.get('rating', 0),
                d.get('is_rejected', False),
                d.get('depth_needed', False),
                d.get('has_depth_map', False)
            )
            item.setHidden(not is_visible)

            # Tooltip
            tooltip = f"📄 {os.path.basename(path)}"
            if d.get('rating', 0) > 0:
                tooltip += f"\n⭐ Rating: {d['rating']} stars"
            if d.get('is_rejected'):
                tooltip += "\n❌ Marked for Deletion"
            if d.get('has_depth_map'):
                tooltip += "\n✓ 3D Depth Map: Generated & Ready"
            elif d.get('depth_needed'):
                tooltip += "\n🧊 3D Depth Map: Required (+3D)"
            if d.get('ai_notes'):
                tooltip += f"\n🧠 AI Notes: {d['ai_notes']}"
            item.setToolTip(tooltip)

            self.list_widget.addItem(item)

        self.list_widget.setUpdatesEnabled(True)

    def _on_scan_complete(self, generation_id: int, all_items: list):
        if generation_id != self.generation_id:
            return

        self.total_photos_count = len(all_items)
        if self.total_photos_count == 0:
            self.lbl_count_info.setText("0 photos")
            self.progress_bar.hide()
            return

        # Queue files that lack valid cached thumbnails
        queue_paths = []
        already_cached_count = 0
        for d in all_items:
            if d.get('is_cached'):
                already_cached_count += 1
            else:
                queue_paths.append(d['path'])

        self.loaded_photos_count = already_cached_count
        self._update_progress_display()

        if queue_paths:
            self.loader_thread.set_tasks(queue_paths, self.generation_id)
        else:
            self.progress_bar.hide()

        if self.active_image_path:
            self._update_active_selection()

    def _on_thumbnail_ready(self, generation_id: int, file_path: str, pixmap: QPixmap, has_project: bool, meta: dict):
        if generation_id != self.generation_id:
            return

        self.loaded_photos_count += 1
        self._update_progress_display()

        norm_path = os.path.abspath(file_path)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and os.path.abspath(item.data(Qt.UserRole)) == norm_path:
                item.setData(Qt.UserRole + 1, has_project)
                item.setData(Qt.UserRole + 2, pixmap)
                model_idx = self.list_widget.model().index(i, 0)
                self.list_widget.update(model_idx)
                break

    def _update_progress_display(self):
        if self.total_photos_count > 0:
            pct = int((self.loaded_photos_count / self.total_photos_count) * 100)
            self.progress_bar.setValue(pct)
            
            # Compute stats across all items and visible items
            total_items = self.list_widget.count()
            visible_count = 0
            n_stars_visible = 0
            n_rej_visible = 0
            n_depth_visible = 0

            n_stars_total = 0
            n_rej_total = 0
            n_depth_total = 0

            for i in range(total_items):
                item = self.list_widget.item(i)
                if item:
                    rating = int(item.data(Qt.UserRole + 3) or 0)
                    is_rej = bool(item.data(Qt.UserRole + 4))
                    has_dp_file = bool(item.data(Qt.UserRole + 7))
                    is_depth = bool(item.data(Qt.UserRole + 5)) or has_dp_file

                    if rating > 0: n_stars_total += 1
                    if is_rej: n_rej_total += 1
                    if is_depth: n_depth_total += 1

                    if not item.isHidden():
                        visible_count += 1
                        if rating > 0: n_stars_visible += 1
                        if is_rej: n_rej_visible += 1
                        if is_depth: n_depth_visible += 1

            # Format count string
            has_filter = (
                self.filter_hide_rejected or 
                self.filter_depth_only or 
                self.filter_rating_min != 0 or 
                (visible_count != self.total_photos_count and self.loaded_photos_count == self.total_photos_count)
            )

            if has_filter:
                filter_tags = []
                if self.filter_rating_min == 5:
                    filter_tags.append(f"★ 5: {visible_count}")
                elif self.filter_rating_min > 0:
                    filter_tags.append(f"★ {self.filter_rating_min}+: {visible_count}")
                elif self.filter_rating_min == -1:
                    filter_tags.append(f"Unrated: {visible_count}")
                
                if self.filter_depth_only:
                    filter_tags.append(f"🧊 3D: {visible_count}")
                if self.filter_hide_rejected:
                    filter_tags.append("No ❌")

                tag_str = f" [{', '.join(filter_tags)}]" if filter_tags else ""
                stats_str = f"Showing {visible_count} of {self.total_photos_count} photos{tag_str}"
            else:
                stats_str = f"{self.total_photos_count} photos"
                sub_stats = []
                if n_stars_total > 0: sub_stats.append(f"{n_stars_total} ⭐")
                if n_rej_total > 0: sub_stats.append(f"{n_rej_total} ❌")
                if n_depth_total > 0: sub_stats.append(f"{n_depth_total} 🧊")
                if sub_stats:
                    stats_str += f" ({', '.join(sub_stats)})"

            if self.loaded_photos_count >= self.total_photos_count:
                self.lbl_count_info.setText(stats_str)
                self.progress_bar.hide()
            else:
                self.lbl_count_info.setText(f"{self.loaded_photos_count}/{self.total_photos_count} • {stats_str}")
                self.progress_bar.show()

    # --- Culling Operations ---
    def set_photo_rating(self, path: str, rating: int):
        """Sets star rating (0-5) for a photo and saves to cache."""
        norm_path = os.path.abspath(path)
        self.cache.update_photo_flags(norm_path, rating=rating)
        self._update_item_flag_in_ui(norm_path, role_offset=3, value=rating)

    def set_photo_rejected(self, path: str, is_rejected: bool):
        """Marks or unmarks a photo for deletion and saves to cache."""
        norm_path = os.path.abspath(path)
        self.cache.update_photo_flags(norm_path, is_rejected=is_rejected)
        self._update_item_flag_in_ui(norm_path, role_offset=4, value=is_rejected)

    def set_photo_depth_needed(self, path: str, depth_needed: bool):
        """Sets photo 3D depth requirement flag and updates UI."""
        norm_path = os.path.abspath(path)
        self.cache.update_photo_flags(norm_path, depth_needed=depth_needed)
        self._update_item_flag_in_ui(norm_path, role_offset=5, value=depth_needed)

    def set_photo_has_depth_map(self, path: str, has_depth: bool = True):
        """Marks that a 3D depth map has been generated and exists on disk."""
        norm_path = os.path.abspath(path)
        self._update_item_flag_in_ui(norm_path, role_offset=7, value=has_depth)

    def set_photo_notes(self, path: str, notes: str):
        """Sets photo user/AI notes, updates database and UI tooltip."""
        norm_path = os.path.abspath(path)
        self.cache.update_photo_flags(norm_path, ai_notes=notes)
        self._update_item_flag_in_ui(norm_path, role_offset=6, value=notes)

    def set_photo_ai_assessment(self, path: str, rating: int, is_rejected: bool, depth_needed: bool, ai_notes: str):
        """Batch update from Gemini AI assessment."""
        norm_path = os.path.abspath(path)
        is_p2 = ("[Center 50% Verified]" in ai_notes or "[50% Verified]" in ai_notes)
        self.cache.update_photo_flags(
            norm_path,
            rating=rating,
            is_rejected=is_rejected,
            depth_needed=depth_needed,
            ai_notes=ai_notes,
            phase2_verified=True if is_p2 else None
        )
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and os.path.abspath(item.data(Qt.UserRole)) == norm_path:
                item.setData(Qt.UserRole + 3, rating)
                item.setData(Qt.UserRole + 4, is_rejected)
                item.setData(Qt.UserRole + 5, depth_needed)
                item.setData(Qt.UserRole + 6, ai_notes)
                has_depth_map = bool(item.data(Qt.UserRole + 7))
                
                # Update tooltip
                tooltip = f"📄 {os.path.basename(norm_path)}\n⭐ Rating: {rating} stars"
                if is_rejected: tooltip += "\n❌ Marked for Deletion"
                if has_depth_map:
                    tooltip += "\n✓ 3D Depth Map: Generated & Ready"
                elif depth_needed:
                    tooltip += "\n🧊 3D Depth Map: Required (+3D)"
                if ai_notes: tooltip += f"\n🧠 AI: {ai_notes}"
                item.setToolTip(tooltip)

                # Filter visibility
                item.setHidden(not self._matches_filter(rating, is_rejected, depth_needed, has_depth_map))
                model_idx = self.list_widget.model().index(i, 0)
                self.list_widget.update(model_idx)
                break

    def get_selected_photos(self, include_rejected: bool = False) -> list:
        """
        Returns a list of dicts for all currently selected items in the Filmstrip.
        Respects multi-selection made via Shift+click or Ctrl+click.
        """
        photos = []
        for item in self.list_widget.selectedItems():
            if not item or item.isHidden():
                continue
            is_rejected = bool(item.data(Qt.UserRole + 4))
            if is_rejected and not include_rejected:
                continue
            path = item.data(Qt.UserRole)
            if not path:
                continue
            rating = int(item.data(Qt.UserRole + 3) or 0)
            depth_needed = bool(item.data(Qt.UserRole + 5))
            has_depth_map = bool(item.data(Qt.UserRole + 7))
            photos.append({
                'path': os.path.abspath(path),
                'rating': rating,
                'depth_needed': depth_needed,
                'has_depth_map': has_depth_map,
                'is_rejected': is_rejected
            })
        return photos

    def get_selected_or_visible_photos(self) -> list:
        """
        Returns selected photos if 2 or more photos are selected;
        otherwise returns all visible non-rejected photos in the Filmstrip.
        """
        selected = self.get_selected_photos(include_rejected=False)
        if len(selected) >= 2:
            return selected
        return self.get_visible_non_rejected_photos()

    def get_visible_non_rejected_photos(self) -> list:
        """
        Returns a list of dicts for all photos currently visible/filtered in the Filmstrip
        that are not marked for deletion (is_rejected == False).
        Respects current active rating filter, search/culling filters, etc.
        """
        photos = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item or item.isHidden():
                continue
            is_rejected = bool(item.data(Qt.UserRole + 4))
            if is_rejected:
                continue
            path = item.data(Qt.UserRole)
            if not path:
                continue
            rating = int(item.data(Qt.UserRole + 3) or 0)
            depth_needed = bool(item.data(Qt.UserRole + 5))
            has_depth_map = bool(item.data(Qt.UserRole + 7))
            path_abs = os.path.abspath(path)
            iso = None
            if hasattr(self, 'cache') and self.cache:
                rec = self.cache.get_photo(path_abs)
                if rec and rec.get('iso'):
                    iso = rec['iso']
            photos.append({
                'path': path_abs,
                'rating': rating,
                'depth_needed': depth_needed,
                'has_depth_map': has_depth_map,
                'iso': iso,
                'index': i
            })
        return photos

    def _update_item_flag_in_ui(self, norm_path: str, role_offset: int, value: Any):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and os.path.abspath(item.data(Qt.UserRole)) == norm_path:
                item.setData(Qt.UserRole + role_offset, value)
                
                rating = int(item.data(Qt.UserRole + 3) or 0)
                is_rejected = bool(item.data(Qt.UserRole + 4))
                depth_needed = bool(item.data(Qt.UserRole + 5))
                ai_notes = str(item.data(Qt.UserRole + 6) or "")
                has_depth_map = bool(item.data(Qt.UserRole + 7))

                # Update tooltip
                tooltip = f"📄 {os.path.basename(norm_path)}"
                if rating > 0: tooltip += f"\n⭐ Rating: {rating} stars"
                if is_rejected: tooltip += "\n❌ Marked for Deletion"
                if has_depth_map:
                    tooltip += "\n✓ 3D Depth Map: Generated & Ready"
                elif depth_needed:
                    tooltip += "\n🧊 3D Depth Map: Required (+3D)"
                if ai_notes: tooltip += f"\n🧠 AI: {ai_notes}"
                item.setToolTip(tooltip)

                # Update filter visibility
                item.setHidden(not self._matches_filter(rating, is_rejected, depth_needed, has_depth_map))
                model_idx = self.list_widget.model().index(i, 0)
                self.list_widget.update(model_idx)
                break
        self._update_progress_display()

    # --- Filtering ---
    def _matches_filter(self, rating: int, is_rejected: bool, depth_needed: bool, has_depth_map: bool = False) -> bool:
        if self.filter_hide_rejected and is_rejected:
            return False
        if self.filter_depth_only and not (depth_needed or has_depth_map):
            return False
        if self.filter_rating_min == -1: # Unrated only
            if rating > 0: return False
        elif self.filter_rating_min > 0:
            if rating < self.filter_rating_min: return False
        return True

    def _apply_filters(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                rating = int(item.data(Qt.UserRole + 3) or 0)
                is_rejected = bool(item.data(Qt.UserRole + 4))
                depth_needed = bool(item.data(Qt.UserRole + 5))
                has_depth_map = bool(item.data(Qt.UserRole + 7))
                item.setHidden(not self._matches_filter(rating, is_rejected, depth_needed, has_depth_map))
        self._update_progress_display()
        self.list_widget.viewport().update()

    def _on_filter_reject_toggled(self, checked: bool):
        self.filter_hide_rejected = checked
        self._apply_filters()

    def _on_filter_rating_changed(self, idx: int):
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: -1}
        self.filter_rating_min = mapping.get(idx, 0)
        self._apply_filters()

    def _on_filter_depth_toggled(self, checked: bool):
        self.filter_depth_only = checked
        self._apply_filters()

    def _on_ai_culling_clicked(self):
        self.ai_culling_requested.emit()

    # --- Context Menu ---
    def _show_context_menu(self, pos: QPoint):
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        selected_items = self.list_widget.selectedItems()
        is_multi = (len(selected_items) >= 2 and item in selected_items)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e242b;
                color: #ffffff;
                border: 1px solid #3c444d;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)

        if is_multi:
            n_sel = len(selected_items)
            paths = [it.data(Qt.UserRole) for it in selected_items if it.data(Qt.UserRole)]

            header_act = menu.addAction(f"📦 {n_sel} Photos Selected")
            header_act.setEnabled(False)
            menu.addSeparator()

            # Batch AI Auto-Correction on selected
            act_ai = menu.addAction(f"✨ Gemini AI Auto-Correct Selected ({n_sel})...")
            act_ai.triggered.connect(lambda: self.batch_ai_requested.emit(self.get_selected_photos(include_rejected=False)))

            # Batch AI Crop on selected
            act_crop = menu.addAction(f"📐 Gemini AI Auto-Crop Selected ({n_sel})...")
            act_crop.triggered.connect(lambda: self.batch_crop_requested.emit(self.get_selected_photos(include_rejected=False)))

            # Batch Export selected
            act_exp = menu.addAction(f"💾 Export Selected Photos ({n_sel})...")
            act_exp.triggered.connect(lambda: self.batch_export_requested.emit(self.get_selected_photos(include_rejected=False)))

            # Batch AI RAW Denoise on selected
            act_denoise = menu.addAction(f"🔇 AI RAW Denoise Selected ({n_sel})...")
            act_denoise.triggered.connect(lambda: self.batch_denoise_requested.emit(self.get_selected_photos(include_rejected=False)))

            menu.addSeparator()

            # Bulk Ratings
            rating_menu = menu.addMenu(f"⭐ Set Rating for Selected ({n_sel})")
            for stars in [5, 4, 3, 2, 1]:
                act = rating_menu.addAction(f"{'★' * stars}{'☆' * (5 - stars)} ({stars} Star{'s' if stars > 1 else ''})")
                act.triggered.connect(lambda checked=False, s=stars, p_list=paths: [self.set_photo_rating(p, s) for p in p_list])
            act_clear_rating = rating_menu.addAction("Clear Rating (0)")
            act_clear_rating.triggered.connect(lambda p_list=paths: [self.set_photo_rating(p, 0) for p in p_list])

            menu.addSeparator()

            # Bulk Reject
            act_reject = menu.addAction(f"❌ Mark Selected for Deletion ({n_sel})")
            act_reject.triggered.connect(lambda p_list=paths: [self.set_photo_rejected(p, True) for p in p_list])

            act_unreject = menu.addAction(f"✓ Unmark Selected from Deletion ({n_sel})")
            act_unreject.triggered.connect(lambda p_list=paths: [self.set_photo_rejected(p, False) for p in p_list])

            menu.addSeparator()

            # Bulk 3D Depth Flags
            act_depth = menu.addAction(f"🧊 Flag: 3D Depth Map Required ({n_sel})")
            act_depth.triggered.connect(lambda p_list=paths: [self.set_photo_depth_needed(p, True) for p in p_list])

            act_rm_depth = menu.addAction(f"✓ Remove 3D Depth Flag ({n_sel})")
            act_rm_depth.triggered.connect(lambda p_list=paths: [self.set_photo_depth_needed(p, False) for p in p_list])

            menu.addSeparator()

            # Delete Selected from Disk
            act_del_sel = menu.addAction(f"🗑️ Delete Selected Photos from Disk ({n_sel})...")
            act_del_sel.triggered.connect(self._delete_selected_photos_from_disk)

        else:
            path = item.data(Qt.UserRole)
            rating = int(item.data(Qt.UserRole + 3) or 0)
            is_rejected = bool(item.data(Qt.UserRole + 4))
            depth_needed = bool(item.data(Qt.UserRole + 5))
            has_depth_map = bool(item.data(Qt.UserRole + 7))

            # Star Rating Submenu
            rating_menu = menu.addMenu("⭐ Set Rating")
            for stars in [5, 4, 3, 2, 1]:
                act = rating_menu.addAction(f"{'★' * stars}{'☆' * (5 - stars)} ({stars} Star{'s' if stars > 1 else ''})")
                act.triggered.connect(lambda checked=False, s=stars: self.set_photo_rating(path, s))
            act_clear_rating = rating_menu.addAction("Clear Rating (0)")
            act_clear_rating.triggered.connect(lambda: self.set_photo_rating(path, 0))

            menu.addSeparator()

            # Reject Toggle
            rej_text = "✓ Unmark from Deletion" if is_rejected else "❌ Mark for Deletion (Reject)"
            act_reject = menu.addAction(rej_text)
            act_reject.triggered.connect(lambda: self.set_photo_rejected(path, not is_rejected))

            # Depth Status & Needed Toggle
            if has_depth_map:
                act_dp_ready = menu.addAction("✓ 3D Depth Map: Ready on disk")
                act_dp_ready.setEnabled(False)
            dp_text = "✓ Remove 3D Depth Flag" if depth_needed else "🧊 Flag: 3D Depth Map Required"
            act_depth = menu.addAction(dp_text)
            act_depth.triggered.connect(lambda: self.set_photo_depth_needed(path, not depth_needed))

            menu.addSeparator()

            # Notes / AI Assessment Edit
            act_notes = menu.addAction("📝 Edit Notes / AI Assessment...")
            act_notes.triggered.connect(lambda: self._prompt_edit_notes(path, item))

            menu.addSeparator()

            # AI Assessment
            act_ai = menu.addAction("✨ Gemini AI Culling & Assessment...")
            act_ai.triggered.connect(self._on_ai_culling_clicked)

            act_full_auto = menu.addAction("🚀 Full Auto: Cull + Auto-Correct All...")
            act_full_auto.triggered.connect(lambda: self.full_auto_requested.emit())

            ext_s = os.path.splitext(path)[1].lower() if path else ""
            if ext_s in ['.cr2', '.cr3', '.nef', '.arw', '.dng', '.raw', '.orf', '.rw2', '.pef', '.raf', '.srw']:
                act_single_denoise = menu.addAction("🔇 AI Denoise RAW Photo (NAFNet)...")
                act_single_denoise.triggered.connect(lambda p=path: self.batch_denoise_requested.emit([p]))

            menu.addSeparator()

            # Delete Rejected from Disk
            act_del_all = menu.addAction("🗑️ Delete All Marked Photos from Disk...")
            act_del_all.triggered.connect(self.delete_rejected_photos_from_disk)

        menu.exec(self.list_widget.mapToGlobal(pos))

    def _prompt_edit_notes(self, path: str, item: QListWidgetItem):
        cur_notes = str(item.data(Qt.UserRole + 6) or "")
        filename = os.path.basename(path)
        new_text, ok = QInputDialog.getMultiLineText(
            self,
            f"Photo Notes & AI Assessment - {filename}",
            f"Notes / AI Justification / Comment for '{filename}':",
            cur_notes
        )
        if ok:
            clean_text = new_text.strip()
            self.set_photo_notes(path, clean_text)
            self.photo_notes_updated.emit(path, clean_text)

    def _delete_selected_photos_from_disk(self):
        """Permanently deletes the currently selected photos from disk after confirmation."""
        selected_photos = self.get_selected_photos(include_rejected=True)
        if not selected_photos:
            return

        res = QMessageBox.warning(
            self,
            "Confirm Delete",
            f"Are you sure you want to permanently delete {len(selected_photos)} selected photo(s)?\n\nThis action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return

        deleted_count = 0
        for photo in selected_photos:
            p = photo['path']
            try:
                if os.path.exists(p):
                    os.remove(p)
                stem, _ = os.path.splitext(p)
                for ext in [".json", "_preview.jpg", "_depth.png", "_normals.png", "_subject.png"]:
                    sidecar = stem + ext
                    if os.path.exists(sidecar):
                        try: os.remove(sidecar)
                        except: pass
                self.cache.delete_photo_record(p)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {p}: {e}")

        QMessageBox.information(self, "Photos Deleted", f"Successfully deleted {deleted_count} photo(s).")
        if self.current_folder:
            self.load_folder(self.current_folder)

    def delete_rejected_photos_from_disk(self):
        """Finds all photos in the current folder marked as rejected and deletes them after confirmation."""
        rejected_paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.UserRole + 4):
                rejected_paths.append(item.data(Qt.UserRole))

        if not rejected_paths:
            QMessageBox.information(self, "No Photos Marked", "No photos are currently marked for deletion (❌).")
            return

        res = QMessageBox.warning(
            self,
            "Confirm Delete",
            f"Are you sure you want to permanently delete {len(rejected_paths)} photo(s) marked for deletion?\n\nThis action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if res != QMessageBox.Yes:
            return

        deleted_count = 0
        for p in rejected_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
                # Remove sidecars if present
                stem, _ = os.path.splitext(p)
                for ext in [".json", "_preview.jpg", "_depth.png", "_normals.png", "_subject.png"]:
                    sidecar = stem + ext
                    if os.path.exists(sidecar):
                        try: os.remove(sidecar)
                        except: pass
                self.cache.delete_photo_record(p)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {p}: {e}")

        QMessageBox.information(self, "Photos Deleted", f"Successfully deleted {deleted_count} photo(s).")
        # Reload current folder
        if self.current_folder:
            self.load_folder(self.current_folder)

    def refresh_image_preview(self, file_path: str, direct_pixmap: QPixmap = None):
        """Invalidate cache and reload thumbnail for a specific image."""
        if not file_path:
            return
        
        norm_path = os.path.abspath(file_path)

        if direct_pixmap and not direct_pixmap.isNull():
            scaled_pm = direct_pixmap.scaled(QSize(360, 270), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if os.path.exists(norm_path):
                try:
                    mtime = os.path.getmtime(norm_path)
                    size = os.path.getsize(norm_path)
                    self.cache.save_thumbnail_and_meta(norm_path, scaled_pm, mtime, size, has_project=True)
                except:
                    pass

            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item and item.data(Qt.UserRole) and os.path.abspath(item.data(Qt.UserRole)) == norm_path:
                    item.setData(Qt.UserRole + 1, True)
                    item.setData(Qt.UserRole + 2, scaled_pm)
                    model_idx = self.list_widget.model().index(i, 0)
                    self.list_widget.update(model_idx)
                    break
            self.list_widget.viewport().update()
            return

        self.loader_thread.add_task_priority(file_path, self.generation_id)

    def set_active_image(self, file_path: str):
        """Mark an image as active with the outline highlight."""
        if not file_path:
            return
        self.active_image_path = os.path.abspath(file_path)
        self.delegate.set_active_path(self.active_image_path)
        
        folder = os.path.dirname(self.active_image_path)
        if self.current_folder != folder:
            self.load_folder(folder)
        else:
            self._update_active_selection()

    def _update_active_selection(self):
        if not self.active_image_path:
            return
        norm_active = os.path.abspath(self.active_image_path)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and os.path.abspath(item.data(Qt.UserRole)) == norm_active:
                self.list_widget.blockSignals(True)
                if len(self.list_widget.selectedItems()) <= 1:
                    self.list_widget.setCurrentItem(item)
                self.list_widget.scrollToItem(item, QAbstractItemView.EnsureVisible)
                self.list_widget.blockSignals(False)
                break
        self.list_widget.viewport().update()

    def _on_item_clicked(self, item: QListWidgetItem):
        if not item:
            return
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            norm_p = os.path.abspath(file_path)
            if self.active_image_path != norm_p:
                self.active_image_path = norm_p
                self.delegate.set_active_path(norm_p)
                self.image_selected.emit(file_path)
                self.list_widget.viewport().update()

    def _on_current_item_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current and current != previous:
            file_path = current.data(Qt.UserRole)
            if file_path and os.path.exists(file_path):
                norm_p = os.path.abspath(file_path)
                if self.active_image_path != norm_p:
                    self.active_image_path = norm_p
                    self.delegate.set_active_path(norm_p)
                    self.image_selected.emit(file_path)
                    self.list_widget.viewport().update()

    def _create_context_menu(self, selected_photos: list) -> Optional[QMenu]:
        if not selected_photos:
            return None

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e222b;
                border: 1px solid #3b4d61;
                color: #e6edf3;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px 6px 12px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #007acc;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #30363d;
                margin: 4px 6px;
            }
        """)

        n = len(selected_photos)
        label_suffix = f"({n} Selected Photos)" if n > 1 else "(Current Photo)"

        # 1. Batch AI Auto-Correction
        act_ai = menu.addAction(f"⚡ Run Gemini AI Auto-Correction {label_suffix}")
        act_ai.triggered.connect(lambda: self.batch_ai_requested.emit(selected_photos))

        # 1b. Batch AI Crop
        act_crop = menu.addAction(f"📐 Run Gemini AI Auto-Cropping {label_suffix}")
        act_crop.triggered.connect(lambda: self.batch_crop_requested.emit(selected_photos))

        # 2. Batch Export
        act_export = menu.addAction(f"💾 Batch Export to JPEG 85% {label_suffix}...")
        act_export.triggered.connect(lambda: self.batch_export_requested.emit(selected_photos))

        menu.addSeparator()

        # 3. Ratings submenu
        rating_menu = menu.addMenu("⭐ Set Rating")
        for r in range(1, 6):
            act_r = rating_menu.addAction(f"{'★' * r} ({r} Stars)")
            act_r.triggered.connect(lambda checked=False, rating_val=r: [self.set_photo_rating(p['path'], rating_val) for p in selected_photos])
        act_clear_r = rating_menu.addAction("Clear Rating (0 Stars)")
        act_clear_r.triggered.connect(lambda: [self.set_photo_rating(p['path'], 0) for p in selected_photos])

        # 4. Reject Flag
        any_not_rej = any(not p['is_rejected'] for p in selected_photos)
        rej_text = "❌ Mark for Deletion (Reject)" if any_not_rej else "↩️ Unmark from Deletion"
        act_rej = menu.addAction(rej_text)
        act_rej.triggered.connect(lambda: [self.set_photo_rejected(p['path'], any_not_rej) for p in selected_photos])

        # 5. 3D Depth Needed
        any_not_dp = any(not p['depth_needed'] for p in selected_photos)
        dp_text = "🧊 Mark 3D Depth Needed (+3D)" if any_not_dp else "🧊 Clear 3D Depth Needed"
        act_dp = menu.addAction(dp_text)
        act_dp.triggered.connect(lambda: [self.set_photo_depth_needed(p['path'], any_not_dp) for p in selected_photos])

        menu.addSeparator()

        # 6. Delete from disk
        act_del = menu.addAction(f"🗑️ Delete {n} Photo(s) from Disk...")
        act_del.triggered.connect(self._delete_selected_photos_from_disk)

        return menu

    def _show_context_menu(self, pos: QPoint):
        item = self.list_widget.itemAt(pos)
        selected_items = self.list_widget.selectedItems()
        if not selected_items and item:
            selected_items = [item]
        if not selected_items:
            return

        selected_photos = []
        for it in selected_items:
            p = it.data(Qt.UserRole)
            if p:
                selected_photos.append({
                    'path': os.path.abspath(p),
                    'rating': int(it.data(Qt.UserRole + 3) or 0),
                    'is_rejected': bool(it.data(Qt.UserRole + 4)),
                    'depth_needed': bool(it.data(Qt.UserRole + 5)),
                    'has_depth_map': bool(it.data(Qt.UserRole + 7))
                })

        menu = self._create_context_menu(selected_photos)
        if menu:
            menu.exec(self.list_widget.mapToGlobal(pos))

