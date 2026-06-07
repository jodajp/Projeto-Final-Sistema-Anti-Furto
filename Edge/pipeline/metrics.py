"""Metricas de runtime do pipeline."""

from dataclasses import dataclass, field
import time


@dataclass
class PipelineMetrics:
    frame_skip: int = 2
    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    detection_count: int = 0
    inference_calls: int = 0
    inference_total_ms: float = 0.0
    fps: float = 0.0
    _fps_window_start: float = field(default_factory=time.time)
    _fps_window_frames: int = 0

    def on_frame(self):
        self.frame_count += 1
        self._fps_window_frames += 1

        elapsed = time.time() - self._fps_window_start
        if elapsed >= 1.0:
            self.fps = self._fps_window_frames / elapsed
            self._fps_window_start = time.time()
            self._fps_window_frames = 0

    def on_inference(self, inference_ms: float):
        self.inference_calls += 1
        self.inference_total_ms += inference_ms

    def on_detection(self):
        self.detection_count += 1

    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    def average_inference_ms(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return self.inference_total_ms / self.inference_calls

    def success_rate(self) -> float:
        if self.inference_calls == 0:
            return 0.0
        return ((self.detection_count / self.inference_calls) * 100.0 / self.frame_skip)
