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
import time

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


class NaiveTracker:
    """Simple IoU-based bounding box tracker."""
    def __init__(self, iou_threshold=0.55, distance_threshold=150.0, max_missing_frames=30):
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.max_missing_frames = max_missing_frames
        
        self.track_centroids = {}  # track_id -> centroid (x,y)
        self.track_last_seen = {}  # track_id -> frame_idx
        self.next_track_id = 1
        
    @staticmethod
    def _box_iou(box_a, box_b):
        x1, y1, x2, y2 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1]), min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    @staticmethod
    def _build_bbox(keypoints, scores, frame_shape):
        h, w = frame_shape[:2]
        valid = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h) & (scores > 0.2)
        if np.sum(valid) < 3: return None
        
        valid_kpts = keypoints[valid]
        x_min, y_min = np.min(valid_kpts, axis=0)
        x_max, y_max = np.max(valid_kpts, axis=0)
        
        pad_x, pad_y = max(25.0, (x_max - x_min) * 0.12), max(35.0, (y_max - y_min) * 0.18)
        x1, y1 = int(np.clip(x_min - pad_x, 0, w - 1)), int(np.clip(y_min - pad_y, 0, h - 1))
        x2, y2 = int(np.clip(x_max + pad_x, 0, w - 1)), int(np.clip(y_max + pad_y, 0, h - 1))
        
        return [x1, y1, x2, y2, float(np.mean(scores[valid]))] if x2 > x1 and y2 > y1 else None

    def clean_dead_tracks(self, frame_idx):
        dead = [tid for tid, last in self.track_last_seen.items() if frame_idx - last > self.max_missing_frames]
        for tid in dead:
            self.track_centroids.pop(tid, None)
            self.track_last_seen.pop(tid, None)
        return dead

    def deduplicate(self, detected_poses, frame_shape):
        poses_with_boxes = [(k, s, self._build_bbox(k, s, frame_shape)) for k, s in detected_poses]
        poses_with_boxes = [p for p in poses_with_boxes if p[2] is not None]
        poses_with_boxes = sorted(poses_with_boxes, key=lambda p: p[2][4], reverse=True)
        
        deduped = []
        for p in poses_with_boxes:
            if not any(self._box_iou(p[2], kept[2]) >= self.iou_threshold for kept in deduped):
                deduped.append(p)
        return [(p[0], p[1]) for p in deduped]

    def update(self, detected_poses, frame_idx):
        centroids = [k[s > 0.1].mean(axis=0) if len(k[s > 0.1]) > 0 else k.mean(axis=0) for k, s in detected_poses]
        matched_indices = {}
        
        if self.track_centroids and centroids:
            track_ids = list(self.track_centroids.keys())
            for p_idx, centroid in enumerate(centroids):
                dists = [np.linalg.norm(centroid - self.track_centroids[tid]) for tid in track_ids]
                min_idx = int(np.argmin(dists))
                if dists[min_idx] < self.distance_threshold:
                    matched_indices[p_idx] = track_ids[min_idx]
                    
        track_assignments = []
        for p_idx, _ in enumerate(detected_poses):
            if p_idx in matched_indices:
                tid = matched_indices[p_idx]
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                
            self.track_centroids[tid] = centroids[p_idx]
            self.track_last_seen[tid] = frame_idx
            track_assignments.append(tid)
            
        return track_assignments


class InferenceVisualizer:
    """Handles rendering of debug UI over the frame."""
    def __init__(self, sequence_length, threshold):
        self.seq_len = sequence_length
        self.threshold = threshold
        self.skel_viz = SkeletonVisualizer(canvas_size=500, show_labels=False, show_confidence=False)

    def draw_skeleton(self, overlay, inf_frame, color):
        kpts, scores = inf_frame.raw_keypoints, inf_frame.raw_scores
        for idx_from, idx_to in SKELETON_CONNECTIONS:
            if scores[idx_from] > 0.1 and scores[idx_to] > 0.1:
                cv2.line(overlay, tuple(map(int, kpts[idx_from])), tuple(map(int, kpts[idx_to])), color, 2)
        for kpt, score in zip(kpts, scores):
            if score > 0.1:
                cv2.circle(overlay, tuple(map(int, kpt)), 4, (255, 255, 255), -1)

    def draw_bounding_box(self, overlay, inf_frame, prob, color):
        valid_kpts = inf_frame.raw_keypoints[inf_frame.raw_scores > 0.1]
        if len(valid_kpts) == 0: return
        
        h, w = overlay.shape[:2]
        x_min, y_min = max(0, int(np.min(valid_kpts[:, 0])) - 10), max(0, int(np.min(valid_kpts[:, 1])) - 10)
        x_max, y_max = min(w - 1, int(np.max(valid_kpts[:, 0])) + 10), min(h - 1, int(np.max(valid_kpts[:, 1])) + 10)
        
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
        status = "Buffering..." if prob is None else f"{prob:.2f} ({'SUSPICIOUS' if prob >= self.threshold else 'NORMAL'})"
        cv2.putText(overlay, f"P{inf_frame.track_id}: {status}", (x_min, max(15, y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def render(self, frame, inf_frames, track_preds, frame_buffers):
        overlay = frame.copy()
        
        for inf_frame in inf_frames:
            prob = track_preds.get(inf_frame.track_id, (None, None, None, None))[1]
            color = (0, 0, 255) if (prob is not None and prob >= self.threshold) else (0, 255, 0) if prob is not None else (0, 165, 255)
            self.draw_skeleton(overlay, inf_frame, color)
            self.draw_bounding_box(overlay, inf_frame, prob, color)
            
        self._render_sidebar(overlay, inf_frames, track_preds, frame_buffers)
        return overlay

    def _render_sidebar(self, overlay, inf_frames, track_preds, frame_buffers):
        h, w = overlay.shape[:2]
        x_offset = w - 10
        for inf_frame in inf_frames:
            try:
                canvas = self.skel_viz.render(inf_frame.normalized_keypoints, inf_frame.raw_scores, title=f"P{inf_frame.track_id}")
            except Exception:
                canvas = np.zeros((500, 500, 3), dtype=np.uint8)
            resized = cv2.resize(canvas, (110, 110))
            if x_offset - 110 > 150:
                overlay[10:120, x_offset-110:x_offset] = resized
                x_offset -= 120

        x, y = 10, 20
        cv2.putText(overlay, "=== ACTIVE TRACKS ===", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        y += 30
        
        for inf_frame in inf_frames:
            tid = inf_frame.track_id
            logit, prob, attn, stats = track_preds.get(tid, (None, None, None, None))
            if prob is not None:
                color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
                cv2.putText(overlay, f"P{tid}: {'SHOPLIFTING' if prob >= self.threshold else 'NORMAL'} (P={prob:.3f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
                cv2.rectangle(overlay, (x, y), (x + 100, y + 8), (100, 100, 100), -1)
                cv2.rectangle(overlay, (x, y), (x + int(100 * prob), y + 8), color, -1)
                y += 15
                if stats:
                    cv2.putText(overlay, f"  Lat: {stats['latency_ms']:.1f}ms | Log: {logit:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    y += 20
            else:
                buf_len = len(frame_buffers.get(tid, []))
                cv2.putText(overlay, f"P{tid}: Buffering ({buf_len}/{self.seq_len})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                y += 20
            y += 5
            
        cv2.putText(overlay, "SPACE: Play | LEFT/RIGHT: Step | S: Save | R: Reset | Q: Quit", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)


class Phase4Inference:
    def __init__(self, checkpoint_path: str, device: str = "cpu", threshold: float = 0.5, temperature: float = 1.0, sequence_length: int = 60):
        # Load Config First
        config_path = EDGE_DIR / "config.yaml"
        if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_data = yaml.safe_load(f)
            
        # GPU detection from config
        if self.config_data.get('detector', {}).get('onnx', {}).get('use_gpu', False) and device == "cpu":
            device = "cuda"

        self.device = device
        self.threshold = float(threshold)
        self.temperature = max(float(temperature), 1e-3)
        self.sequence_length = int(sequence_length)
        
        self.feature_extractor = KinematicFeatureExtractor()
        self.tracker = NaiveTracker()
        self.visualizer = InferenceVisualizer(self.sequence_length, self.threshold)
        
        self.config = Phase4Config(
            sequence_length=self.sequence_length,
            input_size=self.feature_extractor.feature_dim(),
            hidden_size=128, num_layers=2, attention_size=64, dropout=0.2,
            learning_rate=1e-3, batch_size=16, epochs=30, weight_decay=0.0,
            confidence_weighted_loss=True, device=self.device,
        )
        
        self._init_model(checkpoint_path)
        self._init_pipeline()
        
        self.frame_buffers = {}  

    def _init_model(self, checkpoint_path):
        onnx_candidate = checkpoint_path.replace(".pth", ".onnx")
        self.use_onnx = checkpoint_path.endswith(".onnx") or os.path.exists(onnx_candidate)
        
        if self.use_onnx:
            actual_path = onnx_candidate if not checkpoint_path.endswith(".onnx") else checkpoint_path
            import onnxruntime as ort
            
            # Setup session options for maximum performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if hasattr(sess_options, 'intra_op_num_threads'):
                sess_options.intra_op_num_threads = 3
            
            # Priority: CUDA -> DML -> CPU
            requested_providers = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            providers = [p for p in requested_providers if p in available_providers]
            
            self.ort_session = ort.InferenceSession(str(actual_path), sess_options=sess_options, providers=providers)
            
            # Check if DML fell back to CPU for specific nodes (common with LSTMs)
            active_provider = self.ort_session.get_providers()[0]
            print(f"[INFERENCE] Loaded ONNX model successfully on {active_provider}.")
            if active_provider == 'DmlExecutionProvider':
                print("  [ONNX] Note: LSTMs natively run very fast on CPU. DirectML may offload some LSTM nodes to CPU seamlessly.")
        else:
            self.model = Phase4Classifier(self.config).to(self.device).eval()
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                print(f"[INFERENCE] Loaded PyTorch checkpoint on {self.device}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def _init_pipeline(self):
        if 'detector' in self.config_data and 'onnx' in self.config_data['detector']:
            onnx_path = self.config_data['detector']['onnx'].get('model_path', '')
            if onnx_path.startswith('./'):
                self.config_data['detector']['onnx']['model_path'] = str(EDGE_DIR / onnx_path[2:])
        
        self.detector = create_detector(AppConfig(self.config_data).detector_config())
        self.normalizer = SpatialNormalizer(NormalizationParams(torso_confidence_threshold=0.5, allow_invalid_torso=True))

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> list[InferenceFrame]:
        keypoints, scores = self.detector.detect(frame)
        
        detected_poses = []
        if keypoints is not None and len(keypoints) > 0:
            keypoints, scores = np.asarray(keypoints, dtype=np.float32), np.asarray(scores, dtype=np.float32)
            if keypoints.ndim == 2: keypoints, scores = keypoints[np.newaxis, ...], scores[np.newaxis, ...]
            detected_poses = [(k.reshape(17, 2), s.reshape(17)) for k, s in zip(keypoints, scores)]
        
        detected_poses = self.tracker.deduplicate(detected_poses, frame.shape)
        dead_tracks = self.tracker.clean_dead_tracks(frame_idx)
        for tid in dead_tracks: self.frame_buffers.pop(tid, None)
            
        track_assignments = self.tracker.update(detected_poses, frame_idx)
        
        inf_frames = []
        for (kpts, scs), tid in zip(detected_poses, track_assignments):
            self.frame_buffers.setdefault(tid, deque(maxlen=self.config.sequence_length))
            norm_pose = self.normalizer.normalize(kpts, scs)
            
            inf_frame = InferenceFrame(
                frame_idx=frame_idx, raw_frame=frame, raw_keypoints=kpts, raw_scores=scs,
                pose_valid=norm_pose.is_valid, torso_length=norm_pose.torso_length,
                mean_confidence=np.mean(scs), normalized_keypoints=norm_pose.keypoints,
                kinematic_features=np.zeros(50, dtype=np.float32), track_id=tid
            )
            
            self.frame_buffers[tid].append((norm_pose.keypoints, scs))
            inf_frames.append(inf_frame)
            
        return inf_frames
    
    def predict(self, frame: InferenceFrame) -> tuple:
        buffer = self.frame_buffers.get(frame.track_id, [])
        if len(buffer) < self.config.sequence_length: return None, None, None, None
        
        coords_batch = np.array([kp for kp, _ in buffer]).astype(np.float32)[np.newaxis, :, :, :]
        features_batch = self.feature_extractor.transform(coords_batch)
        
        t_start = time.perf_counter()
        if self.use_onnx:
            ort_outs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: features_batch.astype(np.float32)})
            logit = float(ort_outs[0][0])
            attn = ort_outs[1][0] if len(ort_outs) > 1 else np.ones(self.sequence_length) / self.sequence_length
        else:
            with torch.no_grad():
                logits, attn_weights = self.model(torch.from_numpy(features_batch).float().to(self.device))
            logit, attn = logits.item(), attn_weights[0].cpu().numpy()

        prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        
        feature_stats = {"latency_ms": latency_ms}
        return logit, prob, attn, feature_stats

    def run_inference(self, video_path: str, output_path: str = None):
        cap = cv2.VideoCapture(video_path)
        fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps or 30.0, (int(cap.get(3)), int(cap.get(4)))) if output_path else None
        
        playing, current_frame_idx = bool(output_path), 0
        debug_dir = Path("inference_debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        frame_skip = self.config_data.get('runtime', {}).get('frame_skip', 0)
        if frame_skip > 0:
            print(f"[INFERENCE] Frame skipping enabled: skipping {frame_skip} frames between checks to match production speed.")
        
        last_frame_idx = -1
        
        while True:
            # ONLY seek if we are jumping around (fixes massive 50ms+ OpenCV decode lag)
            if current_frame_idx != last_frame_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                
            ret, frame = cap.read()
            if not ret: break
            last_frame_idx = current_frame_idx
            
            # Frame skipping logic (only skip when actively playing)
            if playing and frame_skip > 0 and current_frame_idx % (frame_skip + 1) != 0:
                current_frame_idx += 1
                if current_frame_idx >= total_frames:
                    playing = False
                continue
            
            inf_frames = self.process_frame(frame, current_frame_idx)
            track_preds = {f.track_id: self.predict(f) for f in inf_frames}
            display = self.visualizer.render(frame, inf_frames, track_preds, self.frame_buffers)
            
            if writer:
                writer.write(display)
                current_frame_idx += 1
            else:
                max_w, max_h = self.config_data.get('visualization', {}).get('max_display_width', 1200), self.config_data.get('visualization', {}).get('max_display_height', 900)
                h_d, w_d = display.shape[:2]
                if w_d > max_w or h_d > max_h:
                    scale = min(max_w / w_d, max_h / h_d)
                    display = cv2.resize(display, (int(w_d * scale), int(h_d * scale)))

                cv2.namedWindow("Phase 4 Inference", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Phase 4 Inference", display)
                
                key = cv2.waitKeyEx(1 if playing else 0) & 0xFF
                if key == ord('q'): break
                elif key == ord(' '): playing = not playing
                elif key == 82: current_frame_idx, playing = min(current_frame_idx + 1, total_frames - 1), False
                elif key == 81: current_frame_idx, playing = max(current_frame_idx - 1, 0), False
                elif key == ord('r'): current_frame_idx, playing, self.frame_buffers, self.tracker = 0, False, {}, NaiveTracker()
                elif key == ord('s'): cv2.imwrite(str(debug_dir / f"frame_{current_frame_idx:04d}.png"), display)
                elif playing: current_frame_idx = current_frame_idx + 1 if current_frame_idx + 1 < total_frames else (playing := False) or current_frame_idx
        
        cap.release()
        if writer: writer.release()
        else: cv2.destroyAllWindows()


def list_videos(folder: str) -> list:
    return sorted([f for f in os.listdir(folder) if Path(f).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}])

def select_video(folder: str) -> str:
    videos = list_videos(folder)
    if not videos: return None
    for i, v in enumerate(videos): print(f"  {i:3d}: {v}")
    
    while True:
        try:
            parts = input("\nEnter video number(s) or range (e.g., '5', '0-2'): ").strip().replace(',', ' ').split()
            if not parts: continue
            
            selected = sorted(set([int(p) for p in parts if '-' not in p] + [x for p in parts if '-' in p for x in range(*(lambda s,e: (int(s), int(e)+1))(*p.split('-')))]))
            if all(0 <= idx < len(videos) for idx in selected):
                if len(selected) == 1: return os.path.join(folder, videos[selected[0]])
                else: return (os.path.join(folder, videos[idx]) for idx in selected)
        except Exception:
            print("[ERROR] Invalid input.")


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Phase 4 Inference")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, default="models/phase4_experiment_model.onnx")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, help="Output video path")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint if os.path.isabs(args.checkpoint) else EDGE_DIR / args.checkpoint
    threshold, temperature = args.threshold, 1.0
    
    report_path = Path(str(checkpoint_path).replace(".onnx", ".pth").replace(".pth", "_report.json"))
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            threshold, temperature = float(report.get("threshold", threshold)), float(report.get("temperature", temperature))
            print(f"[INFERENCE] Loaded threshold {threshold:.3f}")
        except Exception: pass

    if args.video:
        Phase4Inference(str(checkpoint_path), args.device, threshold, temperature, args.sequence_length).run_inference(args.video, args.output)
    else:
        video_selector = select_video(str(EDGE_DIR / "Visualizar_Data" / "Data" / "Shoplifting"))
        if not video_selector: return
        for video_path in ([video_selector] if isinstance(video_selector, str) else video_selector):
            Phase4Inference(str(checkpoint_path), args.device, threshold, temperature, args.sequence_length).run_inference(video_path, args.output)


if __name__ == "__main__":
    main()
