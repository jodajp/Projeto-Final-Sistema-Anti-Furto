"""Renderizacao de pose e overlays de informacao."""

import cv2
import numpy as np


from Detecao.skeleton import SKELETON_CONNECTIONS



def _to_color(value, fallback):
    if isinstance(value, list) and len(value) == 3:
        return tuple(int(v) for v in value)
    return fallback


def _is_single_pose(keypoints) -> bool:
    try:
        array = np.asarray(keypoints)
    except Exception:
        return False
    return array.ndim == 2 and array.shape == (17, 2)

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

    def _draw_pose(self, image, keypoints, scores, line_color, point_color):
        for i, j in SKELETON_CONNECTIONS:
            if i >= len(keypoints) or j >= len(keypoints):
                continue

            ci = scores[i] if i < len(scores) else 0.0
            cj = scores[j] if j < len(scores) else 0.0
            if ci <= self.confidence_threshold or cj <= self.confidence_threshold:
                continue

            xi, yi = map(int, keypoints[i])
            xj, yj = map(int, keypoints[j])
            cv2.line(image, (xi, yi), (xj, yj), line_color, 2)

        for idx, point in enumerate(keypoints):
            conf = scores[idx] if idx < len(scores) else 0.0
            if conf <= self.confidence_threshold:
                continue

            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 5, point_color, -1)
            cv2.circle(image, (x, y), 5, (0, 0, 0), 1)

    def _iter_poses(self, keypoints, scores):
        if keypoints is None or scores is None:
            return []

        if _is_single_pose(keypoints):
            return [(np.asarray(keypoints), np.asarray(scores))]

        keypoints_array = np.asarray(keypoints)
        scores_array = np.asarray(scores)

        if keypoints_array.ndim != 3 or keypoints_array.shape[1:] != (17, 2):
            return []

        if scores_array.ndim == 1:
            scores_array = np.repeat(scores_array[np.newaxis, :], keypoints_array.shape[0], axis=0)

        count = min(len(keypoints_array), len(scores_array))
        return [(keypoints_array[i], scores_array[i]) for i in range(count)]

    def render(self, frame, keypoints, scores):
        if not self.enabled:
            return frame

        frame_vis = frame.copy()
        poses = self._iter_poses(keypoints, scores)
        for pose_keypoints, pose_scores in poses:
            self._draw_pose(frame_vis, pose_keypoints, pose_scores, self.color_line, self.color_point)

        if not self.show_skeleton_canvas:
            return frame_vis

        # Build separate canvas for skeletons but keep main frame size unchanged
        canvas = np.full_like(frame, 255)
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
