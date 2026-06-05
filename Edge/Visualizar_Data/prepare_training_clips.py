#!/usr/bin/env python3
"""
Interactive helper to prepare short training clips from Shoplifting videos.

Features:
- List videos under Visualizar_Data/Data/Shoplifting and choose one
- Interactive playback with side-by-side normalized skeleton
- Mark start/end (S/E), adjust speed (U/D), seek, and save trimmed clip
- Writes clipped video and appends normalized keypoints to
  Visualizar_Data/Output/custom_shoplifting_dataset.pkl (same format used by training)

This reuses `SkeletonNormalizer` and the detector factory to avoid duplicated code.
"""
import sys
from pathlib import Path
import argparse
import json
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Detecao.detector_factory import create_detector
from Visualizar_Data.skeleton_normalizer import SkeletonNormalizer, SkeletonNormConfig
from pipeline.spatial_normalizer import SpatialNormalizer, NormalizationParams
from pipeline.skeleton_visualizer import SkeletonVisualizer


def list_videos(folder: Path):
    vids = sorted(list(folder.glob("*.mp4")) + list(folder.glob("*.avi")))
    return vids


def parse_video_selection(selection_text: str, max_index: int) -> List[int]:
    indices: List[int] = []
    seen = set()

    for raw_token in selection_text.split(','):
        token = raw_token.strip()
        if not token:
            continue

        if '-' in token:
            start_text, end_text = token.split('-', 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                if 1 <= value <= max_index and value not in seen:
                    indices.append(value)
                    seen.add(value)
        else:
            value = int(token)
            if 1 <= value <= max_index and value not in seen:
                indices.append(value)
                seen.add(value)

    return indices


def load_detector_and_normalizer(config_path: Path):
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_conf = yaml.safe_load(f)
    yaml_conf['runtime']['frame_skip'] = 1
    from pipeline.config import AppConfig
    app_config = AppConfig(yaml_conf)
    detector = create_detector(app_config.detector_config())

    norm_cfg = SkeletonNormConfig()
    norm_cfg.apply_rotation_90deg = False
    normalizer = SkeletonNormalizer(norm_cfg)
    return detector, normalizer


def extract_all_poses(cap, detector, normalizer):
    frames = []
    norm_kpts = []
    norm_scores = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())

        kpts, scores = detector.detect(frame)
        if kpts is None or len(kpts) == 0:
            # placeholder
            norm_kpts.append(np.zeros((17, 2), dtype=np.float32))
            norm_scores.append(np.zeros((17,), dtype=np.float32))
            continue

        kpts = np.array(kpts)
        scores = np.array(scores)

        # select best person when multiple
        if kpts.ndim == 3:
            person_conf = scores.mean(axis=1)
            best = int(np.argmax(person_conf))
            best_kpt = kpts[best]
            best_score = scores[best]
        else:
            best_kpt = kpts
            best_score = scores

        combined = np.column_stack((best_kpt, best_score))  # (17,3)
        norm = normalizer.normalize_and_center(combined)
        norm_kpts.append(norm[:, :2])
        norm_scores.append(norm[:, 2])

    return frames, norm_kpts, norm_scores


def resample_indices(start, end, speed_factor):
    orig_len = max(1, end - start + 1)
    new_len = max(1, int(round(orig_len / speed_factor)))
    idxs = np.linspace(start, end, new_len)
    idxs = np.round(idxs).astype(int)
    return idxs


def write_clip(output_path: Path, frames_list, keypoints_list, scores_list, fps):
    h, w = frames_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames_list:
        out.write(f)
    out.release()


def append_to_dataset_pkl(output_pkl: Path, kpt_array, score_array, orig_shape):
    if output_pkl.exists():
        with open(output_pkl, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {'split': {'train': []}, 'annotations': []}

    entry_id = f"clip_{int(time.time())}"
    ann = {
        'frame_dir': entry_id,
        'label': 'suspicious',
        'img_shape': orig_shape,
        'original_shape': orig_shape,
        'total_frames': kpt_array.shape[0],
        'keypoint': np.expand_dims(kpt_array, axis=0),
        'keypoint_score': np.expand_dims(score_array, axis=0)
    }
    data['annotations'].append(ann)
    data['split']['train'].append(entry_id)

    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

    return entry_id


def rotate_frame(frame: np.ndarray, rotation_mode: int) -> np.ndarray:
    if rotation_mode == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_mode == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_mode == 3:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def rotate_keypoints(keypoints: np.ndarray, rotation_mode: int, width: int, height: int) -> np.ndarray:
    rotated = keypoints.copy().astype(np.float32)

    if rotation_mode == 1:
        x_new = height - 1 - rotated[:, 1]
        y_new = rotated[:, 0]
        rotated[:, 0] = x_new
        rotated[:, 1] = y_new
    elif rotation_mode == 2:
        rotated[:, 0] = width - 1 - rotated[:, 0]
        rotated[:, 1] = height - 1 - rotated[:, 1]
    elif rotation_mode == 3:
        x_new = rotated[:, 1]
        y_new = width - 1 - rotated[:, 0]
        rotated[:, 0] = x_new
        rotated[:, 1] = y_new

    return rotated


def render_pose_canvas(
    normalized_keypoints: np.ndarray,
    scores: np.ndarray,
    visualizer: SkeletonVisualizer,
) -> np.ndarray:
    return visualizer.render(normalized_keypoints, scores, title='Torso-relative skeleton')


def extract_raw_poses(cap, detector):
    frames: List[np.ndarray] = []
    raw_kpts: List[np.ndarray] = []
    raw_scores: List[np.ndarray] = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame.copy())

        kpts, scores = detector.detect(frame)
        if kpts is None or len(kpts) == 0:
            raw_kpts.append(np.zeros((17, 2), dtype=np.float32))
            raw_scores.append(np.zeros((17,), dtype=np.float32))
            continue

        kpts = np.asarray(kpts, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if kpts.ndim == 3:
            person_conf = scores.mean(axis=1)
            best = int(np.argmax(person_conf))
            best_kpt = kpts[best]
            best_score = scores[best]
        else:
            best_kpt = kpts
            best_score = scores

        raw_kpts.append(best_kpt.astype(np.float32))
        raw_scores.append(best_score.astype(np.float32))

    return frames, raw_kpts, raw_scores


def compute_motion_scores(raw_keypoints: np.ndarray) -> np.ndarray:
    if raw_keypoints.shape[0] < 2:
        return np.zeros((raw_keypoints.shape[0],), dtype=np.float32)

    diffs = np.diff(raw_keypoints, axis=0)
    frame_motion = np.linalg.norm(diffs, axis=2).mean(axis=1)
    motion = np.concatenate([np.zeros((1,), dtype=np.float32), frame_motion.astype(np.float32)])
    return np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)


def infer_rotation_mode(
    raw_keypoints: np.ndarray,
    raw_scores: np.ndarray,
    frame_shape: Tuple[int, int],
    sample_frames: int = 10,
) -> int:
    height, width = frame_shape
    if raw_keypoints.shape[0] == 0:
        return 0

    sample_count = min(sample_frames, raw_keypoints.shape[0])
    sample_indices = np.linspace(0, raw_keypoints.shape[0] - 1, sample_count).round().astype(int)

    def anchor_y(points: np.ndarray, scores: np.ndarray, indices: Tuple[int, ...]) -> float:
        confidence_mask = scores[list(indices)] >= 0.3
        if not np.any(confidence_mask):
            return float('nan')
        valid_points = points[list(indices)][confidence_mask]
        return float(np.mean(valid_points[:, 1]))

    def uprightness(points: np.ndarray, scores: np.ndarray) -> float:
        shoulders_y = anchor_y(points, scores, (5, 6))
        hips_y = anchor_y(points, scores, (11, 12))
        ankles_y = anchor_y(points, scores, (15, 16))

        if any(np.isnan(v) for v in (shoulders_y, hips_y, ankles_y)):
            return -1e9

        vertical_order = (hips_y - shoulders_y) + (ankles_y - hips_y)
        x_span = float(np.nanmax(points[:, 0]) - np.nanmin(points[:, 0]))
        y_span = float(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))
        return vertical_order + 0.1 * (y_span - x_span)

    mode_scores = []
    for mode in range(4):
        frame_scores = []
        for frame_idx in sample_indices:
            rotated = rotate_keypoints(raw_keypoints[frame_idx], mode, width, height)
            frame_scores.append(uprightness(rotated, raw_scores[frame_idx]))
        mode_scores.append(float(np.mean(frame_scores)) if frame_scores else -1e9)

    return int(np.argmax(mode_scores))


def _safe_skeleton_normalize(
    normalizer: SpatialNormalizer,
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, bool, float]:
    pose = normalizer.normalize(keypoints, scores)
    return pose.keypoints, pose.is_valid, pose.torso_length


def compute_hand_risk_scores(raw_keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if raw_keypoints.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    wrists = raw_keypoints[:, [9, 10], :].mean(axis=1)
    torso = raw_keypoints[:, [5, 6, 11, 12], :].mean(axis=1)
    dist = np.linalg.norm(wrists - torso, axis=1)
    valid = np.isfinite(dist)
    risk = np.zeros_like(dist, dtype=np.float32)
    risk[valid] = np.clip(1.0 - (dist[valid] / 220.0), 0.0, 1.0)

    confidence_gate = np.clip(scores[:, [9, 10, 5, 6, 11, 12]].mean(axis=1), 0.0, 1.0)
    risk *= confidence_gate.astype(np.float32)
    return np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)


def suggest_clip_range(motion_scores: np.ndarray, fps: float, target_seconds: float) -> Tuple[int, int]:
    total = int(motion_scores.shape[0])
    if total == 0:
        return 0, 0

    duration_frames = max(1, int(round(target_seconds * fps)))
    peak_idx = int(np.argmax(motion_scores))
    start_idx = max(0, peak_idx - duration_frames // 2)
    end_idx = min(total - 1, start_idx + duration_frames - 1)
    start_idx = max(0, end_idx - duration_frames + 1)
    return start_idx, end_idx


def _trackbar_noop(_value: int) -> None:
    return None


def _build_info_panel(
    width: int,
    height: int,
    source_name: str,
    total_frames: int,
    fps: float,
    start_idx: int,
    end_idx: int,
    current_idx: int,
    speed_factor: float,
    rotation_mode: int,
    target_seconds: float,
    motion_score: float,
    hand_risk: float,
) -> np.ndarray:
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    border = (60, 180, 255)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border, 2)

    lines = [
        f"Source: {source_name}",
        f"Frames: {total_frames} | FPS: {fps:.2f}",
        f"Current: {current_idx + 1}/{total_frames}",
        f"Range: {start_idx + 1}-{end_idx + 1} | Len: {max(0, end_idx - start_idx + 1)} frames",
        f"Speed: {speed_factor:.2f}x | Rotation: {rotation_mode * 90}deg",
        f"Target clip: {target_seconds:.1f}s | Motion peak: {motion_score:.3f}",
        f"Hand risk: {hand_risk:.3f}",
        "",
        "Keys:",
        "Space play/pause | S set start | E set end",
        "A suggest clip | O auto-orient | X export | N next/cancel | Q quit",
        "Left/Right step frame | Z/X speed down/up",
    ]

    y = 28
    for line in lines:
        color = (235, 235, 235)
        if line == "Keys:":
            color = (255, 210, 120)
        cv2.putText(panel, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 24

    return panel


def _resize_with_aspect(frame: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_height / max(1, h)
    return cv2.resize(frame, (int(round(w * scale)), target_height))


@dataclass
class ClipExportResult:
    clip_path: Path
    poster_path: Path
    npz_path: Path
    pkl_path: Path
    manifest_path: Path
    entry_id: str


class ClipPreparationUI:
    def __init__(
        self,
        video_path: Path,
        frames: List[np.ndarray],
        raw_keypoints: List[np.ndarray],
        raw_scores: List[np.ndarray],
        detector_name: str,
        normalizer: SkeletonNormalizer,
        output_dir: Path,
        fps: float,
    ) -> None:
        self.video_path = video_path
        self.frames = frames
        self.raw_keypoints = np.asarray(raw_keypoints, dtype=np.float32)
        self.raw_scores = np.asarray(raw_scores, dtype=np.float32)
        self.detector_name = detector_name
        self.normalizer = normalizer
        self.output_dir = output_dir
        self.fps = float(fps) if fps and fps > 0 else 25.0
        self.total_frames = len(frames)
        self.orig_h, self.orig_w = frames[0].shape[:2]
        self.spatial_normalizer = SpatialNormalizer(
            NormalizationParams(
                torso_confidence_threshold=0.5,
                min_torso_length_px=10.0,
                allow_invalid_torso=True,
            )
        )
        self.skeleton_visualizer = SkeletonVisualizer(canvas_size=700)

        self.motion_scores = compute_motion_scores(self.raw_keypoints)
        self.hand_risk_scores = compute_hand_risk_scores(self.raw_keypoints, self.raw_scores)
        self.combined_scores = np.nan_to_num(self.motion_scores + (0.6 * self.hand_risk_scores), nan=0.0)

        self.rotation_mode = infer_rotation_mode(self.raw_keypoints, self.raw_scores, (self.orig_h, self.orig_w))
        self.speed_factor = 1.0
        self.target_seconds = 2.5
        self.playing = True
        self.current_idx = 0
        self.start_idx, self.end_idx = suggest_clip_range(self.combined_scores, self.fps, self.target_seconds)
        self.last_valid_normalized_pose = np.zeros((17, 2), dtype=np.float32)

        self.preview_window = 'Clip Preview'
        self.controls_window = 'Clip Controls'
        self._window_created = False

    def _make_windows(self) -> None:
        if self._window_created:
            return

        cv2.namedWindow(self.preview_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.controls_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.controls_window, 520, 720)
        cv2.resizeWindow(self.preview_window, 1400, 900)

        max_frame = max(0, self.total_frames - 1)
        cv2.createTrackbar('preview', self.controls_window, 0, max_frame, _trackbar_noop)
        cv2.createTrackbar('start', self.controls_window, self.start_idx, max_frame, _trackbar_noop)
        cv2.createTrackbar('end', self.controls_window, self.end_idx, max_frame, _trackbar_noop)
        cv2.createTrackbar('speed x100', self.controls_window, 100, 400, _trackbar_noop)
        cv2.createTrackbar('rotation', self.controls_window, self.rotation_mode, 3, _trackbar_noop)
        cv2.createTrackbar('duration x10', self.controls_window, int(round(self.target_seconds * 10)), 60, _trackbar_noop)

        cv2.setMouseCallback(self.preview_window, self._on_mouse)
        self._window_created = True

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            panel_w = 0
            if flags:
                panel_w = flags

    def _sync_from_trackbars(self) -> None:
        self.current_idx = int(cv2.getTrackbarPos('preview', self.controls_window))
        self.start_idx = int(cv2.getTrackbarPos('start', self.controls_window))
        self.end_idx = int(cv2.getTrackbarPos('end', self.controls_window))
        self.speed_factor = max(0.25, cv2.getTrackbarPos('speed x100', self.controls_window) / 100.0)
        self.rotation_mode = int(cv2.getTrackbarPos('rotation', self.controls_window))
        self.target_seconds = max(1.0, cv2.getTrackbarPos('duration x10', self.controls_window) / 10.0)

        if self.start_idx > self.end_idx:
            self.end_idx = self.start_idx
            cv2.setTrackbarPos('end', self.controls_window, self.end_idx)

        self.current_idx = max(0, min(self.current_idx, self.total_frames - 1))
        self.start_idx = max(0, min(self.start_idx, self.total_frames - 1))
        self.end_idx = max(0, min(self.end_idx, self.total_frames - 1))

    def _update_trackbars(self) -> None:
        cv2.setTrackbarPos('preview', self.controls_window, self.current_idx)
        cv2.setTrackbarPos('start', self.controls_window, self.start_idx)
        cv2.setTrackbarPos('end', self.controls_window, self.end_idx)
        cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)
        cv2.setTrackbarPos('speed x100', self.controls_window, int(round(self.speed_factor * 100)))
        cv2.setTrackbarPos('duration x10', self.controls_window, int(round(self.target_seconds * 10)))

    def _get_rotated_pose(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.frames[idx]
        height, width = frame.shape[:2]
        rotated_kpts = rotate_keypoints(self.raw_keypoints[idx], self.rotation_mode, width, height)
        rotated_frame = rotate_frame(frame, self.rotation_mode)
        return rotated_frame, rotated_kpts

    def _render_preview(self, idx: int) -> np.ndarray:
        rotated_frame, rotated_kpts = self._get_rotated_pose(idx)
        normalized_pose, is_valid, torso_length = _safe_skeleton_normalize(
            self.spatial_normalizer,
            rotated_kpts,
            self.raw_scores[idx],
        )
        if is_valid:
            self.last_valid_normalized_pose = normalized_pose
        else:
            normalized_pose = self.last_valid_normalized_pose

        pose_canvas = render_pose_canvas(normalized_pose, self.raw_scores[idx], self.skeleton_visualizer)

        frame_disp = _resize_with_aspect(rotated_frame, self.normalizer.config.canvas_height)
        if frame_disp.shape[0] != pose_canvas.shape[0]:
            pose_canvas = cv2.resize(pose_canvas, (pose_canvas.shape[1], frame_disp.shape[0]))

        composite = np.hstack([frame_disp, pose_canvas])
        return composite

    def _render_controls(self) -> np.ndarray:
        motion = float(self.combined_scores[self.current_idx]) if self.total_frames else 0.0
        hand_risk = float(self.hand_risk_scores[self.current_idx]) if self.total_frames else 0.0
        panel = _build_info_panel(
            520,
            720,
            self.video_path.name,
            self.total_frames,
            self.fps,
            self.start_idx,
            self.end_idx,
            self.current_idx,
            self.speed_factor,
            self.rotation_mode,
            self.target_seconds,
            motion,
            hand_risk,
        )
        return panel

    def _playback_delay_ms(self) -> int:
        base = 1000.0 / max(self.fps, 1.0)
        return max(1, int(round(base / max(self.speed_factor, 0.01))))

    def _step_preview(self, step: int) -> None:
        if self.start_idx < self.end_idx:
            lo, hi = self.start_idx, self.end_idx
        else:
            lo, hi = 0, self.total_frames - 1
        self.current_idx = int(np.clip(self.current_idx + step, lo, hi))
        cv2.setTrackbarPos('preview', self.controls_window, self.current_idx)

    def _resample_indices(self, start_idx: int, end_idx: int) -> np.ndarray:
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        total = end_idx - start_idx + 1
        if total <= 1:
            return np.array([start_idx], dtype=np.int32)

        target_frames = max(1, int(round(total / self.speed_factor)))
        sampled = np.linspace(start_idx, end_idx, target_frames)
        sampled = np.clip(np.round(sampled).astype(np.int32), start_idx, end_idx)
        return sampled

    def _export_clip(self) -> ClipExportResult:
        selected = self._resample_indices(self.start_idx, self.end_idx)
        rotated_frames = []
        rotated_kpts = []
        rotated_scores = []
        normalized_kpts = []
        torso_lengths = []
        valid_flags = []

        for frame_idx in selected:
            frame = self.frames[frame_idx]
            h, w = frame.shape[:2]
            rotated_frame = rotate_frame(frame, self.rotation_mode)
            rotated_pose = rotate_keypoints(self.raw_keypoints[frame_idx], self.rotation_mode, w, h)
            normalized_pose, is_valid, torso_length = _safe_skeleton_normalize(
                self.spatial_normalizer,
                rotated_pose,
                self.raw_scores[frame_idx],
            )

            rotated_frames.append(rotated_frame)
            rotated_kpts.append(rotated_pose)
            rotated_scores.append(self.raw_scores[frame_idx])
            normalized_kpts.append(normalized_pose)
            torso_lengths.append(torso_length)
            valid_flags.append(is_valid)

        rotated_kpts_arr = np.asarray(rotated_kpts, dtype=np.float32)
        rotated_scores_arr = np.asarray(rotated_scores, dtype=np.float32)
        normalized_kpts_arr = np.asarray(normalized_kpts, dtype=np.float32)
        torso_lengths_arr = np.asarray(torso_lengths, dtype=np.float32)
        valid_flags_arr = np.asarray(valid_flags, dtype=bool)

        export_root = self.output_dir / 'clips'
        export_root.mkdir(parents=True, exist_ok=True)
        record_dir = export_root / self.video_path.stem
        record_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        clip_name = f"{self.video_path.stem}_f{self.start_idx + 1:04d}_{self.end_idx + 1:04d}_s{int(round(self.speed_factor * 100)):03d}_r{self.rotation_mode}_{timestamp}"
        clip_path = record_dir / f"{clip_name}.mp4"
        poster_path = record_dir / f"{clip_name}.png"
        npz_path = record_dir / f"{clip_name}.npz"
        manifest_path = export_root / 'clips_manifest.jsonl'
        pkl_path = self.output_dir / 'custom_shoplifting_dataset.pkl'

        export_fps = max(1, int(round(self.fps)))
        h, w = rotated_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*'mp4v'), export_fps, (w, h))
        for frame in rotated_frames:
            writer.write(frame)
        writer.release()

        if rotated_frames:
            first = rotated_frames[0].copy()
            cv2.putText(first, self.video_path.name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(poster_path), first)

        np.savez_compressed(
            npz_path,
            keypoint=rotated_kpts_arr,
            keypoint_score=rotated_scores_arr,
            normalized_keypoint=normalized_kpts_arr,
            label=np.array(['suspicious']),
            source_video=np.array([str(self.video_path)]),
            start_frame=np.array([self.start_idx], dtype=np.int32),
            end_frame=np.array([self.end_idx], dtype=np.int32),
            speed_factor=np.array([self.speed_factor], dtype=np.float32),
            rotation_mode=np.array([self.rotation_mode], dtype=np.int32),
            fps=np.array([export_fps], dtype=np.float32),
            torso_length=torso_lengths_arr,
            pose_valid=valid_flags_arr,
        )

        entry_id = append_to_dataset_pkl(
            pkl_path,
            rotated_kpts_arr,
            rotated_scores_arr,
            (rotated_frames[0].shape[0], rotated_frames[0].shape[1]),
        )

        manifest_entry = {
            'entry_id': entry_id,
            'video_path': str(self.video_path),
            'clip_path': str(clip_path),
            'poster_path': str(poster_path),
            'npz_path': str(npz_path),
            'start_frame': int(self.start_idx),
            'end_frame': int(self.end_idx),
            'speed_factor': float(self.speed_factor),
            'rotation_mode': int(self.rotation_mode),
            'fps': float(export_fps),
            'label': 'suspicious',
            'detector': self.detector_name,
            'valid_frames': int(valid_flags_arr.sum()),
            'avg_torso_length': float(np.mean(torso_lengths_arr)) if len(torso_lengths_arr) else 0.0,
        }
        with open(manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')

        return ClipExportResult(
            clip_path=clip_path,
            poster_path=poster_path,
            npz_path=npz_path,
            pkl_path=pkl_path,
            manifest_path=manifest_path,
            entry_id=entry_id,
        )

    def _set_suggested_range(self) -> None:
        self.start_idx, self.end_idx = suggest_clip_range(self.combined_scores, self.fps, self.target_seconds)
        self._update_trackbars()

    def _set_markers_to_current(self) -> None:
        self.start_idx = self.current_idx
        self.end_idx = min(self.total_frames - 1, max(self.current_idx + max(1, int(round(self.target_seconds * self.fps))) - 1, self.current_idx))
        self._update_trackbars()

    def run(self) -> Optional[ClipExportResult]:
        self._make_windows()
        self._update_trackbars()

        last_time = time.time()
        last_key = -1
        export_result = None

        while True:
            self._sync_from_trackbars()

            if self.playing:
                now = time.time()
                delay = self._playback_delay_ms() / 1000.0
                if now - last_time >= delay:
                    last_time = now
                    self._step_preview(1)
                    if self.current_idx >= self.end_idx and self.end_idx > self.start_idx:
                        self.current_idx = self.start_idx
                        cv2.setTrackbarPos('preview', self.controls_window, self.current_idx)
            else:
                last_time = time.time()

            preview = self._render_preview(self.current_idx)
            controls = self._render_controls()
            cv2.imshow(self.preview_window, preview)
            cv2.imshow(self.controls_window, controls)

            key = cv2.waitKeyEx(1)
            key8 = key & 0xFF
            if key != -1:
                last_key = key8

            if key8 in (ord('q'), ord('Q')):
                break
            if key8 == ord(' '):
                self.playing = not self.playing
            elif key8 in (ord('a'), ord('A')):
                self._set_suggested_range()
            elif key8 in (ord('o'), ord('O')):
                self.rotation_mode = infer_rotation_mode(self.raw_keypoints, self.raw_scores, (self.orig_h, self.orig_w))
                cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)
            elif key8 in (ord('x'), ord('X')):
                export_result = self._export_clip()
                print(f"[OK] Exported: {export_result.clip_path}")
                print(f"[OK] Metadata: {export_result.manifest_path}")
                break
            elif key8 in (ord('n'), ord('N')):
                break
            elif key8 in (ord('s'), ord('S')):
                self.start_idx = self.current_idx
                cv2.setTrackbarPos('start', self.controls_window, self.start_idx)
            elif key8 in (ord('e'), ord('E')):
                self.end_idx = self.current_idx
                cv2.setTrackbarPos('end', self.controls_window, self.end_idx)
            elif key8 in (ord('z'), ord('Z')):
                self.speed_factor = max(0.25, self.speed_factor / 1.25)
                cv2.setTrackbarPos('speed x100', self.controls_window, int(round(self.speed_factor * 100)))
            elif key8 in (ord('c'), ord('C')):
                self.speed_factor = min(4.0, self.speed_factor * 1.25)
                cv2.setTrackbarPos('speed x100', self.controls_window, int(round(self.speed_factor * 100)))
            elif key in (81, 2424832, ord('h')):
                self._step_preview(-1)
            elif key in (83, 2555904, ord('l')):
                self._step_preview(1)
            elif key8 == ord('1'):
                self.rotation_mode = 0
                cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)
            elif key8 == ord('2'):
                self.rotation_mode = 1
                cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)
            elif key8 == ord('3'):
                self.rotation_mode = 2
                cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)
            elif key8 == ord('4'):
                self.rotation_mode = 3
                cv2.setTrackbarPos('rotation', self.controls_window, self.rotation_mode)

        cv2.destroyAllWindows()
        return export_result


def interactive_trim(video_path: Path, detector, normalizer, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Failed to open video:', video_path)
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('Extracting poses (this may take a while)...')
    frames, norm_kpts, norm_scores = extract_all_poses(cap, detector, normalizer)
    cap.release()

    total = len(frames)
    if total == 0:
        print('No frames extracted.')
        return

    start_idx = 0
    end_idx = total - 1
    cur = 0
    playing = True
    speed = 1.0

    window_name = 'Trim: S=start E=end U=up D=down Y=save N=cancel Q=quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = frames[cur]
        skel_canvas = draw_skeleton_on_canvas(norm_kpts[cur], norm_scores[cur], canvas_size=(normalizer.config.canvas_height, normalizer.config.canvas_width), conf_threshold=0.3)

        # Resize original to same height
        scale = normalizer.config.canvas_height / orig_h
        new_w = int(orig_w * scale)
        frame_resized = cv2.resize(frame, (new_w, normalizer.config.canvas_height))
        side = np.hstack([frame_resized, skel_canvas])

        info = f"Frame {cur+1}/{total} | Start={start_idx+1} End={end_idx+1} | Speed={speed:.2f}x"
        cv2.putText(side, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow(window_name, side)

        key = cv2.waitKey(int(1000 / max(1, orig_fps))) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('s'):
            start_idx = cur
        elif key == ord('e'):
            end_idx = cur
        elif key == ord('y'):
            # save trimmed clip
            if end_idx <= start_idx:
                print('Invalid range: end must be after start')
                continue
            idxs = resample_indices(start_idx, end_idx, speed)
            sel_frames = [frames[i] for i in idxs]
            sel_kpts = np.stack([norm_kpts[i] for i in idxs])
            sel_scores = np.stack([norm_scores[i] for i in idxs])

            out_clips = out_dir / 'clips'
            out_clips.mkdir(parents=True, exist_ok=True)
            out_video = out_clips / f"{video_path.stem}_clip_{start_idx+1}_{end_idx+1}_{int(speed*100)}.mp4"

            out_fps = int(max(1, round(orig_fps * speed)))
            print('Writing clip to', out_video)
            write_clip(out_video, sel_frames, sel_kpts, sel_scores, out_fps)

            # append to pkl
            out_pkl = ROOT_DIR / 'Visualizar_Data' / 'Output' / 'custom_shoplifting_dataset.pkl'
            out_pkl.parent.mkdir(parents=True, exist_ok=True)
            entry_id = append_to_dataset_pkl(out_pkl, sel_kpts, sel_scores, (orig_h, orig_w))
            print(f'Saved clip and appended to dataset as {entry_id}')
            break
        elif key == ord('n'):
            print('Cancelled for this video')
            break
        elif key == ord('u'):
            speed = min(4.0, speed * 1.25)
        elif key == ord('d'):
            speed = max(0.25, speed / 1.25)
        elif key == 81:  # left arrow
            cur = max(0, cur - 1)
        elif key == 83:  # right arrow
            cur = min(total - 1, cur + 1)

        if playing:
            cur = (cur + 1) % total

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Prepare training clips interactively')
    parser.add_argument('--shoplifting-dir', default=str(ROOT_DIR / 'Visualizar_Data' / 'Data' / 'Shoplifting'))
    args = parser.parse_args()

    shop_dir = Path(args.shoplifting_dir)
    if not shop_dir.exists():
        print('Shoplifting data folder not found:', shop_dir)
        return

    videos = list_videos(shop_dir)
    if not videos:
        print('No videos found in', shop_dir)
        return

    print('Found videos:')
    for i, v in enumerate(videos, 1):
        print(f"  [{i}] {v.name}")

    sel = input('Select video index/range(s) to open (e.g. 1,3,5-7): ')
    try:
        selected_indices = parse_video_selection(sel, len(videos))
    except Exception:
        print('Invalid selection')
        return

    if not selected_indices:
        print('No valid selection made.')
        return

    print('Selected videos:')
    for idx in selected_indices:
        print(f'  - {videos[idx - 1].name}')

    config_path = ROOT_DIR / 'config.yaml'
    detector, normalizer = load_detector_and_normalizer(config_path)
    detector_name = 'detector'
    if hasattr(detector, 'get_info'):
        try:
            detector_name = detector.get_info().get('backend', detector_name)
        except Exception:
            detector_name = detector.__class__.__name__
    else:
        detector_name = detector.__class__.__name__

    out_dir = ROOT_DIR / 'Visualizar_Data' / 'Output'
    out_dir.mkdir(exist_ok=True)

    for position, idx in enumerate(selected_indices, start=1):
        video_path = videos[idx - 1]
        print(f'\n[{position}/{len(selected_indices)}] Opening {video_path.name}')

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print('Failed to open video:', video_path)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        print('Extracting frames and poses...')
        frames, raw_kpts, raw_scores = extract_raw_poses(cap, detector)
        cap.release()

        if not frames:
            print('No frames extracted.')
            continue

        ui = ClipPreparationUI(
            video_path=video_path,
            frames=frames,
            raw_keypoints=raw_kpts,
            raw_scores=raw_scores,
            detector_name=detector_name,
            normalizer=normalizer,
            output_dir=out_dir,
            fps=fps,
        )
        result = ui.run()
        if result is not None:
            print(f'Exported clip: {result.clip_path}')
            print(f'Preview poster: {result.poster_path}')
            print(f'NPZ export: {result.npz_path}')
            print(f'PKL updated: {result.pkl_path}')
        else:
            print('No clip exported for this video.')


if __name__ == '__main__':
    main()
