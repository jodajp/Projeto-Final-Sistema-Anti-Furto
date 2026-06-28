"""Renderizacao de pose e overlays de informacao."""

import cv2
import numpy as np


from Detecao.skeleton import SKELETON_CONNECTIONS



def _to_color(value, fallback):
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(v) for v in value)
    return fallback


# Class para renderizar pose e informações na tela
class PoseRenderer:
    def __init__(self, config: dict):
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.show_skeleton_canvas = bool(config.get("show_skeleton_canvas", True))
        self.confidence_threshold = float(config.get("confidence_threshold", 0.3))
        self.window_name = config.get("window_name", "ANTI-FURTO")

        colors = config.get("colors", {})
        self.color_line = _to_color(colors.get("line"), (0, 255, 255))
        self.color_point = _to_color(colors.get("point"), (0, 255, 0))
        self.color_canvas_line = _to_color(colors.get("canvas_line"), (0, 0, 255))
        self.color_canvas_point = _to_color(colors.get("canvas_point"), (0, 200, 0))
        self.color_text = _to_color(colors.get("text"), (0, 255, 0))
        self.color_warning = _to_color(colors.get("warning"), (0, 0, 255))
        self.color_muted = _to_color(colors.get("muted"), (200, 200, 200))
        self._cached_canvas = None

    def _draw_pose(self, image, keypoints, scores, line_color, point_color):
        for i, j in SKELETON_CONNECTIONS:
            if scores[i] > self.confidence_threshold and scores[j] > self.confidence_threshold:
                xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
                xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
                cv2.line(image, (xi, yi), (xj, yj), line_color, 2)

        for point, conf in zip(keypoints, scores):
            if conf > self.confidence_threshold:
                x, y = int(point[0]), int(point[1])
                cv2.circle(image, (x, y), 5, point_color, -1)
                cv2.circle(image, (x, y), 5, (0, 0, 0), 1)

    def _iter_poses(self, keypoints, scores):
        if keypoints is None or scores is None or len(keypoints) == 0:
            return []

        k_arr = np.asarray(keypoints)
        s_arr = np.asarray(scores)

        if k_arr.ndim == 2 and k_arr.shape == (17, 2):
            return [(k_arr, s_arr)]

        if k_arr.ndim == 3 and k_arr.shape[1:] == (17, 2):
            if s_arr.ndim == 1:
                s_arr = np.repeat(s_arr[np.newaxis, :], len(k_arr), axis=0)
            return list(zip(k_arr, s_arr))

        return []

    def render(self, frame, keypoints, scores):
        if not self.enabled:
            return frame

        frame_vis = frame.copy()
        poses = self._iter_poses(keypoints, scores)
        for pose_keypoints, pose_scores in poses:
            self._draw_pose(frame_vis, pose_keypoints, pose_scores, self.color_line, self.color_point)

        if not self.show_skeleton_canvas:
            return frame_vis

        # Re-use or initialize canvas to prevent memory thrashing
        if self._cached_canvas is None or self._cached_canvas.shape != frame.shape:
            self._cached_canvas = np.full_like(frame, 255)
        else:
            self._cached_canvas.fill(255)
            
        canvas = self._cached_canvas
        for pose_keypoints, pose_scores in poses:
            self._draw_pose(canvas, pose_keypoints, pose_scores, self.color_canvas_line, self.color_canvas_point)

        # Store last canvas for external display; return full-size frame_vis so drawing coords remain valid
        self.last_canvas = canvas
        return frame_vis

    def draw_overlay(self, image, info_lines, alert_text=None, debug=False):
        if not self.enabled:
            return

        for idx, line in enumerate(info_lines):
            y_pos = 24 + idx * 22
            cv2.putText(
                image,
                line,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.color_text,
                1,
            )

        if alert_text:
            cv2.putText(
                image,
                alert_text,
                (10, 24 + len(info_lines) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.color_warning,
                2,
            )

        if debug:
            cv2.putText(
                image,
                "DEBUG MODE (D to toggle)",
                (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.color_warning,
                1,
            )
