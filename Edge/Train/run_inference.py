"""
Phase 4 Model Inference & Evaluation Script
============================================

Select a video and run live shoplifting prediction with debugging overlay.
Shows pose validity, confidence scores, model logits, attention weights.

Hotkeys:
  SPACE:  Play/Pause
  LEFT:   Step back 1 frame
  RIGHT:  Step forward 1 frame
  S:      Save current frame + prediction to debug output
  R:      Reset to start
  Q:      Quit
"""

import sys
import os
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import cv2
import yaml

# Setup path
EDGE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EDGE_DIR))

from Detecao.detector_factory import create_detector
from pipeline.config import AppConfig
from pipeline.kinematic_features import KinematicFeatureExtractor
from pipeline.spatial_normalizer import SpatialNormalizer, NormalizationParams
from pipeline.skeleton_visualizer import SkeletonVisualizer
from Train.phase4_model import Phase4Classifier
from Train.phase4_types import Phase4Config
from Detecao.skeleton import SKELETON_CONNECTIONS


@dataclass
class InferenceFrame:
    frame_idx: int
    raw_frame: np.ndarray  # H×W×3
    raw_keypoints: np.ndarray  # 17×2
    raw_scores: np.ndarray  # 17
    pose_valid: bool
    torso_length: float
    mean_confidence: float
    normalized_keypoints: np.ndarray  # 17×2
    kinematic_features: np.ndarray  # 66
    track_id: int = 0


class Phase4Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu", threshold: float = 0.5, temperature: float = 1.0):
        """Initialize model and components."""
        self.device = device
        self.threshold = float(threshold)
        self.temperature = max(float(temperature), 1e-3)
        
        self.feature_extractor = KinematicFeatureExtractor()
        
        # Load config (same as training)
        # input_size dynamically derived
        self.config = Phase4Config(
            sequence_length=45,
            input_size=self.feature_extractor.feature_dim(),
            hidden_size=128,
            num_layers=1,
            attention_size=64,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=16,
            epochs=30,
            weight_decay=0.0,
            confidence_weighted_loss=True,
            device=device,
        )
        
        # Check model file extension and existences
        # Auto-convert pth path to onnx path if onnx exists
        onnx_candidate = checkpoint_path.replace(".pth", ".onnx")
        self.use_onnx = checkpoint_path.endswith(".onnx") or os.path.exists(onnx_candidate)
        
        if self.use_onnx:
            actual_path = onnx_candidate if not checkpoint_path.endswith(".onnx") else checkpoint_path
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(str(actual_path), providers=['CPUExecutionProvider'])
            print(f"[INFERENCE] Loaded ONNX model from {actual_path} successfully.")
        else:
            # Create model
            self.model = Phase4Classifier(self.config).to(device)
            self.model.eval()
            
            # Load checkpoint
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                # Handle both state_dict and full checkpoint
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"[INFERENCE] Loaded PyTorch checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Initialize components using config
        config_path = EDGE_DIR / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load and fix config paths to be relative to EDGE_DIR
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Fix ONNX model path if relative
        if 'detector' in config_data and 'onnx' in config_data['detector']:
            onnx_path = config_data['detector']['onnx'].get('model_path', '')
            if onnx_path.startswith('./'):
                # Convert relative path to be relative to EDGE_DIR
                config_data['detector']['onnx']['model_path'] = str(EDGE_DIR / onnx_path[2:])
        
        app_config = AppConfig(config_data)
        self.detector = create_detector(app_config.detector_config())
        self.normalizer = SpatialNormalizer(NormalizationParams(
            torso_confidence_threshold=0.5,
            allow_invalid_torso=True
        ))
        
        # Startup contract assertion
        assert self.feature_extractor.feature_dim() == self.config.input_size, \
            f"Feature dimension mismatch! Extractor: {self.feature_extractor.feature_dim()}, Model Config: {self.config.input_size}"
        
        self.visualizer = SkeletonVisualizer(canvas_size=500, show_labels=False, show_confidence=False)
        
        # Inference state
        self.frame_buffers = {}  # track_id -> deque(maxlen=45)
        self.prediction_histories = {}  # track_id -> deque(maxlen=45)
        self.track_centroids = {}  # track_id -> centroid (x,y)
        self.track_last_seen = {}  # track_id -> frame_idx
        self.next_track_id = 1
        
    @staticmethod
    def _box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _build_bbox(keypoints, scores, frame_shape):
        h_img, w_img = frame_shape[:2]
        in_bounds = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w_img) & \
                    (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h_img) & \
                    np.isfinite(keypoints).all(axis=1)
        
        valid_mask = in_bounds & (scores > 0.2)
        if np.sum(valid_mask) < 3:
            valid_mask = in_bounds

        if not np.any(valid_mask):
            return None

        valid_kpts = keypoints[valid_mask]
        x_min, y_min = np.min(valid_kpts, axis=0)
        x_max, y_max = np.max(valid_kpts, axis=0)

        # Padding
        pad_x = max(25.0, (x_max - x_min) * 0.12)
        pad_y = max(35.0, (y_max - y_min) * 0.18)

        x1 = int(np.clip(x_min - pad_x, 0, w_img - 1))
        y1 = int(np.clip(y_min - pad_y, 0, h_img - 1))
        x2 = int(np.clip(x_max + pad_x, 0, w_img - 1))
        y2 = int(np.clip(y_max + pad_y, 0, h_img - 1))

        return [x1, y1, x2, y2, float(np.mean(scores[valid_mask]))] if x2 > x1 and y2 > y1 else None

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> list[InferenceFrame]:
        """Process single frame: detect poses, track people, normalize, buffer."""
        # Detect pose
        keypoints, scores = self.detector.detect(frame)
        
        detected_poses = []
        if keypoints is not None and len(keypoints) > 0:
            keypoints = np.asarray(keypoints, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            if keypoints.ndim == 2:
                keypoints = keypoints[np.newaxis, ...]
                scores = scores[np.newaxis, ...]
            
            for kpts, scs in zip(keypoints, scores):
                kpts = kpts.reshape(17, 2)
                scs = scs.reshape(17)
                detected_poses.append((kpts, scs))
        
        # Deduplicate overlapping detections (IoU >= 0.55)
        if len(detected_poses) >= 2:
            poses_with_boxes = []
            for kpts, scs in detected_poses:
                bbox = self._build_bbox(kpts, scs, frame.shape)
                if bbox is not None:
                    poses_with_boxes.append((kpts, scs, bbox))
            
            poses_with_boxes = sorted(poses_with_boxes, key=lambda p: p[2][4], reverse=True)
            deduped = []
            for kpts, scs, bbox in poses_with_boxes:
                is_duplicate = any(
                    self._box_iou(bbox, kept[2]) >= 0.55
                    for kept in deduped
                )
                if not is_duplicate:
                    deduped.append((kpts, scs, bbox))
            detected_poses = [(kpts, scs) for kpts, scs, _ in deduped]
        
        # Clean up dead tracks (not seen for more than 30 frames)
        dead_tracks = [tid for tid, last_idx in self.track_last_seen.items() if frame_idx - last_idx > 30]
        for tid in dead_tracks:
            self.frame_buffers.pop(tid, None)
            self.prediction_histories.pop(tid, None)
            self.track_centroids.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            
        # Compute centroids for newly detected poses
        centroids = []
        for kpts, scs in detected_poses:
            valid_kpts = kpts[scs > 0.1]
            if len(valid_kpts) > 0:
                centroids.append(valid_kpts.mean(axis=0))
            else:
                centroids.append(kpts.mean(axis=0))
                
        matched_indices = {}  # pose_idx -> track_id
        if self.track_centroids and centroids:
            track_ids = list(self.track_centroids.keys())
            for p_idx, centroid in enumerate(centroids):
                dists = [np.linalg.norm(centroid - self.track_centroids[tid]) for tid in track_ids]
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < 150.0:  # distance threshold: 150 pixels
                    matched_indices[p_idx] = track_ids[min_idx]
                    
        inf_frames = []
        for p_idx, (kpts, scs) in enumerate(detected_poses):
            if p_idx in matched_indices:
                tid = matched_indices[p_idx]
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                
            self.track_centroids[tid] = centroids[p_idx]
            self.track_last_seen[tid] = frame_idx
            
            self.frame_buffers.setdefault(tid, deque(maxlen=self.config.sequence_length))
            self.prediction_histories.setdefault(tid, deque(maxlen=self.config.sequence_length))
            
            # Normalize pose
            normalized_pose = self.normalizer.normalize(kpts, scs)
            normalized_keypoints = normalized_pose.keypoints
            pose_valid = normalized_pose.is_valid
            torso_length = normalized_pose.torso_length
            mean_confidence = np.mean(scs)
            
            inf_frame = InferenceFrame(
                frame_idx=frame_idx,
                raw_frame=frame,
                raw_keypoints=kpts,
                raw_scores=scs,
                pose_valid=pose_valid,
                torso_length=torso_length,
                mean_confidence=mean_confidence,
                normalized_keypoints=normalized_keypoints,
                kinematic_features=np.zeros(50, dtype=np.float32),
                track_id=tid
            )
            
            self.frame_buffers[tid].append((normalized_keypoints, scs))
            inf_frames.append(inf_frame)
            
        return inf_frames
    
    def predict(self, frame: InferenceFrame) -> tuple:
        """Predict shoplifting probability from sequence buffer."""
        import time
        t_start = time.perf_counter()
        
        tid = getattr(frame, 'track_id', 0)
        buffer = self.frame_buffers.get(tid, [])
        if len(buffer) < self.config.sequence_length:
            # Not enough frames yet
            return None, None, None, None
        
        # Extract kinematic features for full sequence
        normalized_keypoints_list = np.array([kp for kp, _ in buffer])  # (T, 17, 2)
        
        # Add batch dimension: (1, T, 17, 2)
        coords_batch = normalized_keypoints_list.astype(np.float32)[np.newaxis, :, :, :]
        
        # Extract features: (1, T, D)
        features_batch = self.feature_extractor.transform(coords_batch)
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: features_batch.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logit = float(ort_outs[0][0])
            prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))  # calibrated sigmoid
            attn = ort_outs[1][0] if len(ort_outs) > 1 else np.ones(self.config.sequence_length) / self.config.sequence_length
        else:
            # Prepare tensor for model
            features_tensor = torch.from_numpy(features_batch).float().to(self.device)
            
            with torch.no_grad():
                logits, attn_weights = self.model(features_tensor)
            
            logit = logits.item()
            prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))  # calibrated sigmoid
            attn = attn_weights[0].cpu().numpy()  # (T,)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        feature_seq = features_batch[0]
        feature_stats = {
            "feature_abs_mean": float(np.mean(np.abs(feature_seq))),
            "feature_std": float(np.std(feature_seq)),
            "feature_delta_mean": float(np.mean(np.abs(np.diff(feature_seq, axis=0)))) if feature_seq.shape[0] > 1 else 0.0,
            "attention_entropy": float(-np.sum(attn * np.log(np.clip(attn, 1e-8, 1.0))) / np.log(len(attn))) if len(attn) > 1 else 0.0,
            "latency_ms": latency_ms,
        }

        return logit, prob, attn, feature_stats

    def render_debug_overlay(self, frame: np.ndarray, inf_frames: list[InferenceFrame], 
                             track_predictions: dict) -> np.ndarray:
        """Add debugging overlay to frame with multi-person tracking support."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw skeletons and bounding boxes directly on the overlay image
        for inf_frame in inf_frames:
            tid = inf_frame.track_id
            preds = track_predictions.get(tid)
            prob = preds[1] if preds else None
            
            # Determine color based on prediction
            if prob is not None:
                color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
            else:
                color = (0, 165, 255) # Orange (buffering)
                
            # Compute bounding box
            valid_kpts = inf_frame.raw_keypoints[inf_frame.raw_scores > 0.1]
            if len(valid_kpts) > 0:
                x_min = int(np.min(valid_kpts[:, 0]))
                y_min = int(np.min(valid_kpts[:, 1]))
                x_max = int(np.max(valid_kpts[:, 0]))
                y_max = int(np.max(valid_kpts[:, 1]))
                
                # Expand box slightly
                padding = 10
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w - 1, x_max + padding)
                y_max = min(h - 1, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
                
                # Bounding box label
                status_str = "Buffering..." if prob is None else f"{prob:.2f} ({'SUSPICIOUS' if prob >= self.threshold else 'NORMAL'})"
                label = f"P{tid}: {status_str}"
                cv2.putText(overlay, label, (x_min, max(15, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw skeleton connections on the original frame
            for idx_from, idx_to in SKELETON_CONNECTIONS:
                pt_from = inf_frame.raw_keypoints[idx_from]
                pt_to = inf_frame.raw_keypoints[idx_to]
                if inf_frame.raw_scores[idx_from] > 0.1 and inf_frame.raw_scores[idx_to] > 0.1:
                    cv2.line(overlay, tuple(map(int, pt_from)), tuple(map(int, pt_to)), color, 2)
            
            # Draw keypoints on the original frame
            for kpt, score in zip(inf_frame.raw_keypoints, inf_frame.raw_scores):
                if score > 0.1:
                    cv2.circle(overlay, tuple(map(int, kpt)), 4, (255, 255, 255), -1)
                    
        # Paste normalized skeletons side-by-side in the top-right corner
        x_offset = w - 10
        y_offset = 10
        for inf_frame in inf_frames:
            tid = inf_frame.track_id
            try:
                skeleton_canvas = self.visualizer.render(
                    inf_frame.normalized_keypoints, 
                    inf_frame.raw_scores,
                    title=f"P{tid} Normalized"
                )
            except Exception:
                skeleton_canvas = np.zeros((500, 500, 3), dtype=np.uint8)
            
            resized = cv2.resize(skeleton_canvas, (110, 110))
            r_h, r_w = resized.shape[:2]
            if x_offset - r_w > 150: # Don't overlap too much with the center
                overlay[y_offset:y_offset+r_h, x_offset-r_w:x_offset] = resized
                x_offset -= (r_w + 10)
        
        # Draw debugging text (left side)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        x, y = 10, 20
        
        cv2.putText(overlay, "=== ACTIVE TRACKS ===", (x, y), font, font_scale, (255, 200, 0), thickness)
        y += int(line_height * 1.5)
        
        for inf_frame in inf_frames:
            tid = inf_frame.track_id
            preds = track_predictions.get(tid)
            if preds and preds[1] is not None:
                logit, prob, attn, feature_stats = preds
                decision = "SHOPLIFTING" if prob >= self.threshold else "NORMAL"
                decision_color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
                
                track_text = f"P{tid}: {decision} (P={prob:.3f})"
                cv2.putText(overlay, track_text, (x, y), font, font_scale, decision_color, thickness)
                y += line_height
                
                # Small confidence bar
                cv2.rectangle(overlay, (x, y), (x + 100, y + 8), (100, 100, 100), -1)
                cv2.rectangle(overlay, (x, y), (x + int(100 * prob), y + 8), decision_color, -1)
                y += int(line_height * 0.8)
                
                # Latency & Logit
                if feature_stats and "latency_ms" in feature_stats:
                    cv2.putText(overlay, f"  Latency: {feature_stats['latency_ms']:.1f}ms | Logit: {logit:.2f}", (x, y), font, font_scale * 0.7, (200, 200, 200), thickness)
                    y += line_height
            else:
                buffer_len = len(self.frame_buffers.get(tid, []))
                cv2.putText(overlay, f"P{tid}: Buffering ({buffer_len}/45)", (x, y), font, font_scale, (150, 150, 150), thickness)
                y += line_height
            y += 5
            
        # Hotkey hints (bottom-left)
        y_bottom = int(h - 30)
        cv2.putText(overlay, "SPACE: Play/Pause | LEFT/RIGHT: Step | S: Save | R: Reset | Q: Quit", 
                   (x, y_bottom), font, font_scale*0.7, (150, 150, 150), 1)
        
        return overlay
    
    def run_inference(self, video_path: str, output_path: str = None):
        """Run inference on video, either interactively or saving to output_path."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFERENCE] Video: {video_path}")
        print(f"[INFERENCE] FPS: {fps}, Total frames: {total_frames}, Resolution: {width}x{height}")
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (width, height))
            print(f"[INFERENCE] Saving output to {output_path}...")
            playing = True
        else:
            playing = False
            
        current_frame_idx = 0
        delay_ms = int(1000 / fps) if fps > 0 else 33
        
        # Create debug output directory
        debug_dir = Path("inference_debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        while True:
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print("[INFERENCE] End of video reached.")
                break
            
            # Process frame: returns list of InferenceFrame
            inf_frames = self.process_frame(frame, current_frame_idx)
            
            # Predict for each active track
            track_predictions = {}
            for inf_frame in inf_frames:
                tid = inf_frame.track_id
                logit, prob, attn, feature_stats = self.predict(inf_frame)
                track_predictions[tid] = (logit, prob, attn, feature_stats)
            
            # Render
            display = self.render_debug_overlay(frame, inf_frames, track_predictions)
            
            if writer:
                writer.write(display)
                current_frame_idx += 1
                if current_frame_idx % 50 == 0:
                    print(f"[INFERENCE] Processed {current_frame_idx}/{total_frames} frames...")
            else:
                cv2.imshow("Phase 4 Inference", display)
                
                # Handle input
                if playing:
                    key = cv2.waitKeyEx(delay_ms) & 0xFF
                else:
                    key = cv2.waitKeyEx(0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # SPACE
                    playing = not playing
                    print(f"[INFERENCE] {'Playing' if playing else 'Paused'} at frame {current_frame_idx}")
                elif key == 82:  # RIGHT arrow
                    current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
                    playing = False
                    print(f"[INFERENCE] Stepped forward to frame {current_frame_idx}")
                elif key == 81:  # LEFT arrow
                    current_frame_idx = max(current_frame_idx - 1, 0)
                    playing = False
                    print(f"[INFERENCE] Stepped back to frame {current_frame_idx}")
                elif key == ord('r'):  # R
                    current_frame_idx = 0
                    self.frame_buffers.clear()
                    self.prediction_histories.clear()
                    self.track_centroids.clear()
                    self.track_last_seen.clear()
                    self.next_track_id = 1
                    playing = False
                    print("[INFERENCE] Reset to start")
                elif key == ord('s'):  # S - Save debug frame
                    probs_str = "_".join([f"P{tid}_{preds[1]:.2f}" for tid, preds in track_predictions.items() if preds[1] is not None])
                    filename = debug_dir / f"frame_{current_frame_idx:04d}_{probs_str}.png"
                    cv2.imwrite(str(filename), display)
                    print(f"[INFERENCE] Saved debug frame: {filename}")
                elif playing:
                    current_frame_idx += 1
                    if current_frame_idx >= total_frames:
                        playing = False
                        print("[INFERENCE] End of video, paused")
        
        cap.release()
        if writer:
            writer.release()
        else:
            cv2.destroyAllWindows()
        print("[INFERENCE] Done.")


def list_videos(folder: str) -> list:
    """List all .mp4 files in folder."""
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    return sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in video_exts])


def select_video(folder: str) -> str:
    """Interactive video selector."""
    videos = list_videos(folder)
    if not videos:
        print(f"[ERROR] No videos found in {folder}")
        return None
    
    print(f"\n[VIDEO SELECTOR] Found {len(videos)} videos:")
    for i, v in enumerate(videos):
        print(f"  {i:3d}: {v}")
    
    while True:
        try:
            user_input = input("\nEnter video number(s) or range (e.g., '5' or '0-2' or '1,3,5'): ").strip()
            if not user_input:
                continue
            
            selected = []
            for part in user_input.replace(',', ' ').split():
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected.extend(range(start, end + 1))
                else:
                    selected.append(int(part))
            
            selected = sorted(set(selected))
            if all(0 <= idx < len(videos) for idx in selected):
                if len(selected) == 1:
                    return os.path.join(folder, videos[selected[0]])
                else:
                    print(f"\n[SELECTOR] Processing {len(selected)} videos sequentially...")
                    for idx in selected:
                        yield os.path.join(folder, videos[idx])
                    return None
        except (ValueError, IndexError):
            print("[ERROR] Invalid input. Try again.")


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Phase 4 Model Inference & Visualization")
    parser.add_argument("--video", type=str, help="Path to video file (if not specified, show selector)")
    parser.add_argument("--checkpoint", type=str, default="models/retails_clip_model_precision.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for shoplifting")
    parser.add_argument("--threshold-file", type=str, default="models/phase4_experiment_report.json", help="Optional JSON report file containing a calibrated threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output", type=str, help="Path to output video file (if specified, run offline and save results)")
    args = parser.parse_args()

    threshold = float(args.threshold)
    temperature = 1.0

    # Get checkpoint path
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = EDGE_DIR / checkpoint_path
        
    # Try to load calibrated threshold dynamically from report
    report_candidates = [
        Path(args.threshold_file) if args.threshold_file else None,
        checkpoint_path.parent / "phase4_experiment_report.json",
        checkpoint_path.with_name(checkpoint_path.stem + "_report.json"),
        checkpoint_path.with_suffix(".json")
    ]
    for r_path in report_candidates:
        if r_path and r_path.exists():
            try:
                report = json.loads(r_path.read_text(encoding="utf-8"))
                threshold = float(report.get("threshold", threshold))
                temperature = float(report.get("temperature", temperature))
                print(f"[INFERENCE] Loaded calibrated threshold {threshold:.3f} and temperature {temperature:.2f} from {r_path.name}")
                break
            except Exception as exc:
                pass
    
    # Check if checkpoint path exists or its onnx candidate exists
    onnx_candidate = str(checkpoint_path).replace(".pth", ".onnx")
    if not os.path.exists(checkpoint_path) and not os.path.exists(onnx_candidate):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return
    
    # Get video path
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"[ERROR] Video not found: {video_path}")
            return
        
        # Run inference on specified video
        inference = Phase4Inference(str(checkpoint_path), device=args.device, threshold=threshold, temperature=temperature)
        inference.run_inference(video_path, output_path=args.output)
    else:
        # Interactive video selector
        shoplifting_dir = EDGE_DIR / "Visualizar_Data" / "Data" / "Shoplifting"
        if not shoplifting_dir.exists():
            print(f"[ERROR] Shoplifting folder not found: {shoplifting_dir}")
            return
        
        video_selector = select_video(str(shoplifting_dir))
        
        # Handle both single and multiple selections
        if isinstance(video_selector, str):
            # Single video selected
            video_path = video_selector
            if not os.path.exists(video_path):
                print(f"[ERROR] Video not found: {video_path}")
                return
            
            print(f"\n[INFERENCE] Starting with video: {os.path.basename(video_path)}")
            inference = Phase4Inference(str(checkpoint_path), device=args.device, threshold=threshold, temperature=temperature)
            inference.run_inference(video_path, output_path=args.output)
        else:
            # Multiple videos selected (generator)
            for video_path in video_selector:
                if not os.path.exists(video_path):
                    print(f"[WARNING] Video not found: {video_path}, skipping...")
                    continue
                
                print(f"\n[INFERENCE] Starting with video: {os.path.basename(video_path)}")
                inference = Phase4Inference(str(checkpoint_path), device=args.device, threshold=threshold, temperature=temperature)
                inference.run_inference(video_path, output_path=args.output)


if __name__ == "__main__":
    main()
