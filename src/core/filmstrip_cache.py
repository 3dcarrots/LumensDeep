import os
import sys
import sqlite3
import hashlib
import time
import contextlib
from typing import Dict, List, Optional, Any
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSize

class FilmstripCache:
    """
    High-performance persistent cache for Filmstrip thumbnails, EXIF metadata,
    and culling flags (star ratings, rejection marks, 3D depth requirement, AI notes).
    
    All settings, database files, and thumbnails are stored directly inside the
    local {photo_folder}/LumenDeep/ subfolder for 100% self-contained portability.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "FilmstripCache":
        if cls._instance is None:
            cls._instance = FilmstripCache()
        return cls._instance

    def __init__(self, base_dir: Optional[str] = None):
        # Default global fallback directory if no specific photo folder is provided
        if base_dir is None:
            user_home = os.path.expanduser("~")
            base_dir = os.path.join(user_home, ".lumendeep", "cache")
        
        self.fallback_dir = os.path.abspath(base_dir)
        self.fallback_thumbs_dir = os.path.join(self.fallback_dir, "thumbnails")
        self.fallback_previews_dir = os.path.join(self.fallback_dir, "previews")
        self.fallback_db_path = os.path.join(self.fallback_dir, "filmstrip.db")
        
        self._initialized_dbs = set()

    def _get_folder_lumen_dir(self, folder_or_path: str) -> str:
        """Returns the absolute path to {folder}/LumenDeep/."""
        norm = os.path.abspath(folder_or_path)
        if os.path.isdir(norm):
            folder = norm
        else:
            folder = os.path.dirname(norm)
        lumen_dir = os.path.join(folder, "LumenDeep")
        os.makedirs(lumen_dir, exist_ok=True)
        return lumen_dir

    def _get_db_path(self, folder_or_path: Optional[str] = None) -> str:
        if not folder_or_path:
            os.makedirs(self.fallback_dir, exist_ok=True)
            return self.fallback_db_path
        lumen_dir = self._get_folder_lumen_dir(folder_or_path)
        return os.path.join(lumen_dir, "filmstrip.db")

    @contextlib.contextmanager
    def _connect(self, folder_or_path: Optional[str] = None):
        db_path = self._get_db_path(folder_or_path)
        conn = sqlite3.connect(db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            if db_path not in self._initialized_dbs:
                self._init_db(conn)
                self._initialized_dbs.add(db_path)
            yield conn
        finally:
            try:
                conn.close()
            except:
                pass

    def _init_db(self, conn: sqlite3.Connection):
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS photo_cache (
                path TEXT PRIMARY KEY,
                folder TEXT,
                filename TEXT,
                mtime REAL,
                size INTEGER,
                rating INTEGER DEFAULT 0,
                is_rejected INTEGER DEFAULT 0,
                depth_needed INTEGER DEFAULT 0,
                has_project INTEGER DEFAULT 0,
                cam_model TEXT,
                lens_model TEXT,
                iso TEXT,
                shutter TEXT,
                aperture REAL,
                focal REAL,
                width INTEGER,
                height INTEGER,
                ai_notes TEXT,
                phase2_verified INTEGER DEFAULT 0,
                thumb_file TEXT,
                preview_file TEXT,
                updated_at REAL
            );
        """)
        try:
            conn.execute("ALTER TABLE photo_cache ADD COLUMN phase2_verified INTEGER DEFAULT 0;")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE photo_cache ADD COLUMN preview_file TEXT;")
        except Exception:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_folder ON photo_cache(folder);")
        conn.commit()

    def get_cached_preview_path(self, path: str) -> Optional[str]:
        """Returns the absolute file path to the 50% cached JPEG (quality 90) in the photo's LumenDeep/ folder."""
        norm_path = os.path.abspath(path)
        dir_name = os.path.dirname(norm_path)
        stem, _ = os.path.splitext(os.path.basename(norm_path))
        
        # 1. Primary: {photo_folder}/LumenDeep/{stem}_half.jpg
        lumen_half = os.path.join(dir_name, "LumenDeep", f"{stem}_half.jpg")
        if os.path.exists(lumen_half):
            return lumen_half

        # 2. Legacy/flat: {photo_folder}/{stem}_half.jpg
        old_half = os.path.join(dir_name, f"{stem}_half.jpg")
        if os.path.exists(old_half):
            return old_half

        return None

    def save_preview_jpeg(self, path: str, pil_img: Any) -> Optional[str]:
        """Saves a 50% high-quality preview (max 2560px, JPEG q=90) to {folder}/LumenDeep/{stem}_half.jpg."""
        try:
            from PIL import Image
            norm_path = os.path.abspath(path)
            dir_name = os.path.dirname(norm_path)
            stem, _ = os.path.splitext(os.path.basename(norm_path))

            lumen_dir = os.path.join(dir_name, "LumenDeep")
            os.makedirs(lumen_dir, exist_ok=True)
            preview_file = os.path.join(lumen_dir, f"{stem}_half.jpg")

            p_copy = pil_img.copy() if hasattr(pil_img, 'copy') else pil_img
            if hasattr(p_copy, 'thumbnail'):
                if p_copy.width > 2560 or p_copy.height > 2560:
                    p_copy.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                p_copy.save(preview_file, "JPEG", quality=90)

            with self._connect(norm_path) as conn:
                conn.execute(
                    "UPDATE photo_cache SET preview_file = ? WHERE path = ?",
                    (preview_file, norm_path)
                )
                conn.commit()
            return preview_file
        except Exception as e:
            print(f"FilmstripCache: Error saving preview for {path}: {e}")
            return None

    def get_folder_cache(self, folder_path: str) -> Dict[str, dict]:
        """
        Retrieves all cached records for files inside a folder in a single O(1) query.
        Returns dict keyed by normalized absolute file path.
        """
        norm_folder = os.path.abspath(folder_path)
        result = {}
        try:
            with self._connect(norm_folder) as conn:
                cur = conn.execute("SELECT * FROM photo_cache WHERE folder = ?", (norm_folder,))
                for row in cur.fetchall():
                    d = dict(row)
                    result[d['path']] = d
        except Exception as e:
            print(f"FilmstripCache: Error reading folder cache for {folder_path}: {e}")
        return result

    def get_photo_record(self, path: str) -> Optional[dict]:
        """Fetches metadata and flags for a single photo."""
        norm_path = os.path.abspath(path)
        try:
            with self._connect(norm_path) as conn:
                cur = conn.execute("SELECT * FROM photo_cache WHERE path = ?", (norm_path,))
                row = cur.fetchone()
                if row:
                    return dict(row)
        except Exception as e:
            print(f"FilmstripCache: Error reading record for {path}: {e}")
        return None

    # Alias for convenience
    get_photo = get_photo_record

    def get_cached_pixmap(self, thumb_filename: str, folder: Optional[str] = None) -> Optional[QPixmap]:
        """Loads a cached QPixmap from disk by its thumbnail filename or path."""
        if not thumb_filename:
            return None

        # 1. Absolute path check
        if os.path.isabs(thumb_filename) and os.path.exists(thumb_filename):
            pm = QPixmap(thumb_filename)
            if not pm.isNull():
                return pm

        # 2. Check local folder LumenDeep/thumbnails/
        if folder:
            local_p = os.path.join(os.path.abspath(folder), "LumenDeep", "thumbnails", thumb_filename)
            if os.path.exists(local_p):
                pm = QPixmap(local_p)
                if not pm.isNull():
                    return pm
            local_p2 = os.path.join(os.path.abspath(folder), "LumenDeep", thumb_filename)
            if os.path.exists(local_p2):
                pm = QPixmap(local_p2)
                if not pm.isNull():
                    return pm

        # 3. Fallback to global cache dir
        fallback_p = os.path.join(self.fallback_thumbs_dir, thumb_filename)
        if os.path.exists(fallback_p):
            pm = QPixmap(fallback_p)
            if not pm.isNull():
                return pm

        return None

    def save_thumbnail_and_meta(
        self,
        path: str,
        pixmap: QPixmap,
        mtime: float,
        size: int,
        metadata: Optional[dict] = None,
        has_project: bool = False,
        preview_img: Optional[Any] = None
    ) -> str:
        """
        Saves the thumbnail (360x270) into {folder}/LumenDeep/thumbnails/{stem}.jpg
        and 50% preview JPEG (max 2560px, quality 90) into {folder}/LumenDeep/{stem}_half.jpg,
        and records entry in local {folder}/LumenDeep/filmstrip.db.
        """
        norm_path = os.path.abspath(path)
        folder = os.path.dirname(norm_path)
        filename = os.path.basename(norm_path)
        stem, _ = os.path.splitext(filename)
        
        lumen_dir = os.path.join(folder, "LumenDeep")
        thumbs_dir = os.path.join(lumen_dir, "thumbnails")
        os.makedirs(thumbs_dir, exist_ok=True)

        thumb_filename = f"{stem}.jpg"
        thumb_full_path = os.path.join(thumbs_dir, thumb_filename)

        # Save pixmap to disk JPEG
        if pixmap and not pixmap.isNull():
            try:
                pixmap.save(thumb_full_path, "JPEG", quality=88)
            except Exception as e:
                print(f"FilmstripCache: Failed to save thumbnail {thumb_full_path}: {e}")

        # Save 50% preview JPEG directly to {photo_folder}/LumenDeep/{stem}_half.jpg
        preview_filename = None
        if preview_img is not None:
            try:
                from PIL import Image
                preview_full_path = os.path.join(lumen_dir, f"{stem}_half.jpg")

                p_copy = preview_img.copy() if hasattr(preview_img, 'copy') else preview_img
                if hasattr(p_copy, 'thumbnail'):
                    if p_copy.width > 2560 or p_copy.height > 2560:
                        p_copy.thumbnail((2560, 2560), Image.Resampling.LANCZOS)
                    p_copy.save(preview_full_path, "JPEG", quality=90)
                    preview_filename = preview_full_path
            except Exception as e:
                print(f"FilmstripCache: Failed to save 50% preview {path}: {e}")

        # Extract metadata fields
        meta = metadata or {}
        cam_model = f"{meta.get('cam_maker', '')} {meta.get('cam_model', '')}".strip() or None
        lens_model = meta.get('lens_model') or meta.get('lens_maker') or None
        iso = str(meta.get('iso')) if meta.get('iso') else None
        shutter = str(meta.get('shutter')) if meta.get('shutter') else None
        aperture = float(meta['aperture']) if 'aperture' in meta and meta['aperture'] else None
        focal = float(meta['focal_length']) if 'focal_length' in meta and meta['focal_length'] else None
        width = int(meta.get('width', 0)) if meta.get('width') else None
        height = int(meta.get('height', 0)) if meta.get('height') else None

        now = time.time()
        try:
            with self._connect(folder) as conn:
                conn.execute("""
                    INSERT INTO photo_cache (
                        path, folder, filename, mtime, size, has_project,
                        cam_model, lens_model, iso, shutter, aperture, focal,
                        width, height, thumb_file, preview_file, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        mtime = excluded.mtime,
                        size = excluded.size,
                        has_project = excluded.has_project,
                        cam_model = COALESCE(excluded.cam_model, photo_cache.cam_model),
                        lens_model = COALESCE(excluded.lens_model, photo_cache.lens_model),
                        iso = COALESCE(excluded.iso, photo_cache.iso),
                        shutter = COALESCE(excluded.shutter, photo_cache.shutter),
                        aperture = COALESCE(excluded.aperture, photo_cache.aperture),
                        focal = COALESCE(excluded.focal, photo_cache.focal),
                        width = COALESCE(excluded.width, photo_cache.width),
                        height = COALESCE(excluded.height, photo_cache.height),
                        thumb_file = excluded.thumb_file,
                        preview_file = COALESCE(excluded.preview_file, photo_cache.preview_file),
                        updated_at = excluded.updated_at
                """, (
                    norm_path, folder, filename, mtime, size, int(has_project),
                    cam_model, lens_model, iso, shutter, aperture, focal,
                    width, height, thumb_filename, preview_filename, now
                ))
                conn.commit()
        except Exception as e:
            print(f"FilmstripCache: Failed to insert record for {path}: {e}")

        return thumb_filename

    def update_photo_flags(
        self,
        path: str,
        rating: Optional[int] = None,
        is_rejected: Optional[bool] = None,
        depth_needed: Optional[bool] = None,
        ai_notes: Optional[str] = None,
        phase2_verified: Optional[bool] = None
    ):
        """Updates user or AI culling flags for a specific photo in its folder's LumenDeep database."""
        norm_path = os.path.abspath(path)
        folder = os.path.dirname(norm_path)
        filename = os.path.basename(norm_path)
        now = time.time()

        updates = ["updated_at = ?"]
        params = [now]

        if rating is not None:
            updates.append("rating = ?")
            params.append(int(rating))
        if is_rejected is not None:
            updates.append("is_rejected = ?")
            params.append(int(is_rejected))
        if depth_needed is not None:
            updates.append("depth_needed = ?")
            params.append(int(depth_needed))
        if ai_notes is not None:
            updates.append("ai_notes = ?")
            params.append(ai_notes)
        if phase2_verified is not None:
            updates.append("phase2_verified = ?")
            params.append(int(phase2_verified))

        params.append(norm_path)

        query = f"UPDATE photo_cache SET {', '.join(updates)} WHERE path = ?"
        try:
            with self._connect(folder) as conn:
                cur = conn.execute(query, params)
                if cur.rowcount == 0:
                    # Insert stub record if it didn't exist yet
                    conn.execute("""
                        INSERT INTO photo_cache (path, folder, filename, rating, is_rejected, depth_needed, ai_notes, phase2_verified, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        norm_path, folder, filename,
                        rating if rating is not None else 0,
                        int(is_rejected) if is_rejected is not None else 0,
                        int(depth_needed) if depth_needed is not None else 0,
                        ai_notes or "",
                        int(phase2_verified) if phase2_verified is not None else 0,
                        now
                    ))
                conn.commit()
        except Exception as e:
            print(f"FilmstripCache: Error updating flags for {path}: {e}")

    def update_batch_flags(self, records: List[dict]):
        """Efficient batch update for multiple photos from AI culling or bulk operations."""
        if not records:
            return
        
        # Group records by folder to minimize db connections
        folder_groups: Dict[str, List[dict]] = {}
        for r in records:
            p = os.path.abspath(r['path'])
            f = os.path.dirname(p)
            folder_groups.setdefault(f, []).append(r)

        now = time.time()
        for folder, group in folder_groups.items():
            try:
                with self._connect(folder) as conn:
                    for r in group:
                        path = os.path.abspath(r['path'])
                        filename = os.path.basename(path)
                        rating = r.get('rating')
                        is_rejected = r.get('is_rejected')
                        depth_needed = r.get('depth_needed')
                        ai_notes = r.get('ai_notes')
                        p2_v = r.get('phase2_verified')

                        conn.execute("""
                            INSERT INTO photo_cache (path, folder, filename, rating, is_rejected, depth_needed, ai_notes, phase2_verified, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(path) DO UPDATE SET
                                rating = CASE WHEN ? IS NOT NULL THEN ? ELSE photo_cache.rating END,
                                is_rejected = CASE WHEN ? IS NOT NULL THEN ? ELSE photo_cache.is_rejected END,
                                depth_needed = CASE WHEN ? IS NOT NULL THEN ? ELSE photo_cache.depth_needed END,
                                ai_notes = CASE WHEN ? IS NOT NULL THEN ? ELSE photo_cache.ai_notes END,
                                phase2_verified = CASE WHEN ? IS NOT NULL THEN ? ELSE photo_cache.phase2_verified END,
                                updated_at = ?
                        """, (
                            path, folder, filename,
                            rating if rating is not None else 0,
                            int(is_rejected) if is_rejected is not None else 0,
                            int(depth_needed) if depth_needed is not None else 0,
                            ai_notes or "",
                            int(p2_v) if p2_v is not None else 0,
                            now,
                            rating, rating,
                            int(is_rejected) if is_rejected is not None else None, int(is_rejected) if is_rejected is not None else None,
                            int(depth_needed) if depth_needed is not None else None, int(depth_needed) if depth_needed is not None else None,
                            ai_notes, ai_notes,
                            int(p2_v) if p2_v is not None else None, int(p2_v) if p2_v is not None else None,
                            now
                        ))
                    conn.commit()
            except Exception as e:
                print(f"FilmstripCache: Error batch updating flags in folder {folder}: {e}")

    def delete_photo_record(self, path: str):
        """Removes a photo record and its thumbnail from cache."""
        norm_path = os.path.abspath(path)
        folder = os.path.dirname(norm_path)
        try:
            with self._connect(folder) as conn:
                cur = conn.execute("SELECT thumb_file FROM photo_cache WHERE path = ?", (norm_path,))
                row = cur.fetchone()
                if row and row['thumb_file']:
                    thumb_p = os.path.join(folder, "LumenDeep", "thumbnails", row['thumb_file'])
                    if os.path.exists(thumb_p):
                        try: os.remove(thumb_p)
                        except: pass
                conn.execute("DELETE FROM photo_cache WHERE path = ?", (norm_path,))
                conn.commit()
        except Exception as e:
            print(f"FilmstripCache: Error deleting record for {path}: {e}")
