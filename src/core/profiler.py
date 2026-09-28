import time
from collections import deque
from typing import Dict, Any, Optional
from contextlib import contextmanager

class PerformanceProfiler:
    """
    High-resolution performance profiler for the real-time viewport and rendering pipeline.
    Tracks GPU render passes, PCIe buffer readbacks, Qt presentation, and overall FPS.
    """
    def __init__(self, history_size: int = 30):
        self.enabled = True
        self.history_size = history_size
        
        # Frame timing history
        self._frame_times = deque(maxlen=history_size)
        self._last_frame_start = 0.0
        
        # Current frame category times in milliseconds
        self.metrics: Dict[str, float] = {
            'gpu_render_ms': 0.0,
            'fbo_read_ms': 0.0,
            'present_ms': 0.0,
            'paint_ms': 0.0,
            'total_frame_ms': 0.0,
        }
        
        # Rolling averages for stable HUD display
        self._metrics_history: Dict[str, deque] = {
            'gpu_render_ms': deque(maxlen=history_size),
            'fbo_read_ms': deque(maxlen=history_size),
            'present_ms': deque(maxlen=history_size),
            'paint_ms': deque(maxlen=history_size),
            'total_frame_ms': deque(maxlen=history_size),
        }
        
        self.fps: float = 0.0
        self.frame_count: int = 0

    def start_frame(self):
        """Marks the start of a new frame update."""
        now = time.perf_counter()
        if self._last_frame_start > 0:
            delta = (now - self._last_frame_start) * 1000.0
            self._frame_times.append(delta)
            if self._frame_times:
                avg_delta = sum(self._frame_times) / len(self._frame_times)
                self.fps = 1000.0 / max(0.001, avg_delta)
        self._last_frame_start = now
        self.frame_count += 1

    def end_frame(self, total_time_ms: Optional[float] = None):
        """Finalizes the current frame and records rolling averages."""
        if total_time_ms is not None:
            self.metrics['total_frame_ms'] = total_time_ms
        else:
            self.metrics['total_frame_ms'] = (time.perf_counter() - self._last_frame_start) * 1000.0
            
        for k, v in self.metrics.items():
            if k in self._metrics_history:
                self._metrics_history[k].append(v)

    @contextmanager
    def scope(self, metric_name: str):
        """Context manager to measure execution time of a specific stage."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.metrics[metric_name] = elapsed_ms

    def record_metric(self, metric_name: str, duration_ms: float):
        """Directly records a metric value in milliseconds."""
        self.metrics[metric_name] = duration_ms

    def get_avg_metric(self, metric_name: str) -> float:
        """Returns the smoothed rolling average for a metric."""
        hist = self._metrics_history.get(metric_name)
        if hist and len(hist) > 0:
            return sum(hist) / len(hist)
        return self.metrics.get(metric_name, 0.0)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a dictionary summary of current performance metrics."""
        return {
            'fps': round(self.fps, 1),
            'total_ms': round(self.get_avg_metric('total_frame_ms'), 2),
            'gpu_render_ms': round(self.get_avg_metric('gpu_render_ms'), 2),
            'fbo_read_ms': round(self.get_avg_metric('fbo_read_ms'), 2),
            'present_ms': round(self.get_avg_metric('present_ms'), 2),
            'paint_ms': round(self.get_avg_metric('paint_ms'), 2),
        }

    def reset(self):
        """Resets all metrics history."""
        self._frame_times.clear()
        self._last_frame_start = 0.0
        for hist in self._metrics_history.values():
            hist.clear()
        for k in self.metrics:
            self.metrics[k] = 0.0
        self.fps = 0.0

# Global singleton profiler instance
profiler = PerformanceProfiler()
