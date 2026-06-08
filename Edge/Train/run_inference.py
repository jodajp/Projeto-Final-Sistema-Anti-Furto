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
        self.frame_buffer = deque(maxlen=45)  # Keep last 45 frames for sequence
        self.prediction_history = deque(maxlen=45)  # For attention visualization
        
    def process_frame(self, frame: np.ndarray) -> InferenceFrame:
        """Process single frame: detect pose, normalize, extract features."""
        # Detect pose
        keypoints, scores = self.detector.detect(frame)
        
        # Handle detector result format - extract best person if multiple
        if keypoints is None or len(keypoints) == 0:
            raw_keypoints = np.zeros((17, 2), dtype=np.float32)
            raw_scores = np.zeros(17, dtype=np.float32)
        else:
            keypoints = np.asarray(keypoints, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            
            # Handle batched results (multiple people)
            if keypoints.ndim == 3:
                # Multiple people detected - select best
                person_conf = scores.mean(axis=1)
                best_idx = int(np.argmax(person_conf))
                raw_keypoints = keypoints[best_idx]
                raw_scores = scores[best_idx]
            else:
                raw_keypoints = keypoints
                raw_scores = scores
        
        raw_keypoints = np.asarray(raw_keypoints, dtype=np.float32).reshape(17, 2)
        raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(17)
        
        # Normalize
        normalized_pose = self.normalizer.normalize(raw_keypoints, raw_scores)
        normalized_keypoints = normalized_pose.keypoints
        pose_valid = normalized_pose.is_valid
        
        # Compute metrics
        torso_length = normalized_pose.torso_length
        mean_confidence = np.mean(raw_scores)
        
        # Kinematic features will be computed when we have a full sequence
        kinematic_features = np.zeros(50, dtype=np.float32)  # 50 dims (34 velocity + 16 limb orientation)
        
        inf_frame = InferenceFrame(
            frame_idx=len(self.frame_buffer),
            raw_frame=frame,
            raw_keypoints=raw_keypoints,
            raw_scores=raw_scores,
            pose_valid=pose_valid,
            torso_length=torso_length,
            mean_confidence=mean_confidence,
            normalized_keypoints=normalized_keypoints,
            kinematic_features=kinematic_features,
        )
        
        # Add to buffer (store normalized keypoints, raw scores for feature extraction)
        self.frame_buffer.append((normalized_keypoints, raw_scores))
        
        return inf_frame
    
    def predict(self, frame: InferenceFrame) -> tuple:
        """Predict shoplifting probability from sequence buffer."""
        import time
        t_start = time.perf_counter()
        
        if len(self.frame_buffer) < self.config.sequence_length:
            # Not enough frames yet
            return None, None, None, None
        
        # Extract kinematic features for full sequence
        # frame_buffer contains (normalized_keypoints, raw_scores) tuples
        normalized_keypoints_list = np.array([kp for kp, _ in self.frame_buffer])  # (T, 17, 2)
        
        # Add batch dimension: (1, T, 17, 2)
        coords_batch = normalized_keypoints_list.astype(np.float32)[np.newaxis, :, :, :]
        
        # Extract features: (1, T, 66)
        features_batch = self.feature_extractor.transform(coords_batch)
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: features_batch.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logit = float(ort_outs[0][0])
            prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))  # calibrated sigmoid
            attn = ort_outs[1][0] if len(ort_outs) > 1 else np.ones(self.config.sequence_length) / self.config.sequence_length
        else:
            # Prepare tensor for model
            features_tensor = torch.from_numpy(features_batch).float().to(self.device)  # (1, T, 66)
            
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
    
    def render_debug_overlay(self, frame: np.ndarray, inf_frame: InferenceFrame, 
                             logit: float = None, prob: float = None, attn: np.ndarray = None,
                             feature_stats: dict = None) -> np.ndarray:
        """Add debugging overlay to frame."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw skeleton on normalized canvas
        try:
            skeleton_canvas = self.visualizer.render(
                inf_frame.normalized_keypoints, 
                inf_frame.raw_scores,
                title=""
            )
        except Exception as e:
            # Fallback if rendering fails
            skeleton_canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Composite skeleton canvas onto frame (top-right corner)
        canvas_h, canvas_w = skeleton_canvas.shape[:2]
        scale = min((w // 3) / canvas_w, (h // 3) / canvas_h)
        resized_canvas = cv2.resize(skeleton_canvas, (int(canvas_w * scale), int(canvas_h * scale)))
        resized_h, resized_w = resized_canvas.shape[:2]
        x_offset = w - resized_w - 10
        y_offset = 10
        overlay[y_offset:y_offset+resized_h, x_offset:x_offset+resized_w] = resized_canvas
        
        # Draw debugging text (left side)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        x, y = 10, 20
        
        # Pose validity
        validity_text = "✓ VALID" if inf_frame.pose_valid else "✗ INVALID"
        validity_color = (0, 255, 0) if inf_frame.pose_valid else (0, 0, 255)
        cv2.putText(overlay, f"Pose: {validity_text}", (x, y), font, font_scale, validity_color, thickness)
        y += line_height
        
        # Torso length
        if inf_frame.torso_length > 0:
            cv2.putText(overlay, f"Torso Length: {inf_frame.torso_length:.1f}px", (x, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height
        
        # Mean confidence
        conf_color = (0, 255, 0) if inf_frame.mean_confidence > 0.5 else (0, 165, 255)
        cv2.putText(overlay, f"Avg Confidence: {inf_frame.mean_confidence:.2f}", (x, y), font, font_scale, conf_color, thickness)
        y += int(line_height * 1.5)
        
        # Model predictions
        if prob is not None:
            cv2.putText(overlay, "=== Model Prediction ===", (x, y), font, font_scale, (255, 200, 0), thickness)
            y += line_height
            
            logit_text = f"Logit: {logit:.3f}"
            cv2.putText(overlay, logit_text, (x, y), font, font_scale, (200, 200, 255), thickness)
            y += line_height
            
            prob_color = (0, 0, 255) if prob > 0.5 else (0, 255, 0)
            prob_text = f"P(Shoplifting): {prob:.3f}"
            cv2.putText(overlay, prob_text, (x, y), font, font_scale, prob_color, thickness)
            y += line_height
            
            # Confidence bar
            bar_width = 150
            bar_height = 15
            bar_x, bar_y = x, y + 5
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            filled_width = int(bar_width * prob)
            bar_color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), bar_color, -1)
            y += int(line_height * 1.5)

            decision = "SHOPLIFTING" if prob >= self.threshold else "NORMAL"
            decision_color = (0, 0, 255) if prob >= self.threshold else (0, 255, 0)
            cv2.putText(overlay, f"Decision: {decision} @ thr={self.threshold:.2f}", (x, y), font, font_scale, decision_color, thickness)
            y += line_height

            # Model engine and latency details
            engine_text = "Engine: ONNX Runtime" if self.use_onnx else "Engine: PyTorch"
            engine_color = (0, 255, 255) if self.use_onnx else (255, 100, 255)
            cv2.putText(overlay, engine_text, (x, y), font, font_scale, engine_color, thickness)
            y += line_height

            if feature_stats:
                if "latency_ms" in feature_stats:
                    cv2.putText(overlay, f"Inference Latency: {feature_stats['latency_ms']:.2f} ms", (x, y), font, font_scale, (100, 255, 100), thickness)
                    y += line_height
                cv2.putText(overlay, f"Feat |mean|: {feature_stats['feature_abs_mean']:.3f}", (x, y), font, font_scale, (220, 220, 220), thickness)
                y += line_height
                cv2.putText(overlay, f"Feat std: {feature_stats['feature_std']:.3f}", (x, y), font, font_scale, (220, 220, 220), thickness)
                y += line_height
                cv2.putText(overlay, f"Feat delta: {feature_stats['feature_delta_mean']:.4f}", (x, y), font, font_scale, (220, 220, 220), thickness)
                y += line_height
                cv2.putText(overlay, f"Attn entropy: {feature_stats['attention_entropy']:.3f}", (x, y), font, font_scale, (220, 220, 220), thickness)
                y += int(line_height * 1.2)
        
        # Frame counter & buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.config.sequence_length}"
        cv2.putText(overlay, buffer_status, (x, y), font, font_scale, (200, 200, 200), thickness)
        y += line_height
        
        # Hotkey hints (bottom-left)
        y_bottom = int(h - 100)
        cv2.putText(overlay, "SPACE: Play/Pause | LEFT/RIGHT: Step | S: Save | R: Reset | Q: Quit", 
                   (x, y_bottom), font, font_scale*0.7, (150, 150, 150), 1)
        
        return overlay
    
    def run_interactive(self, video_path: str):
        """Interactive inference loop."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFERENCE] Video: {video_path}")
        print(f"[INFERENCE] FPS: {fps}, Total frames: {total_frames}")
        
        playing = False
        current_frame_idx = 0
        delay_ms = int(1000 / fps)
        
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
            
            # Process frame
            inf_frame = self.process_frame(frame)
            
            # Predict
            logit, prob, attn, feature_stats = self.predict(inf_frame)
            
            # Render
            display = self.render_debug_overlay(frame, inf_frame, logit, prob, attn, feature_stats)
            
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
            elif key == 82:  # RIGHT arrow (cv2 key code)
                current_frame_idx = min(current_frame_idx + 1, total_frames - 1)
                playing = False
                print(f"[INFERENCE] Stepped forward to frame {current_frame_idx}")
            elif key == 81:  # LEFT arrow
                current_frame_idx = max(current_frame_idx - 1, 0)
                playing = False
                print(f"[INFERENCE] Stepped back to frame {current_frame_idx}")
            elif key == ord('r'):  # R
                current_frame_idx = 0
                self.frame_buffer.clear()
                playing = False
                print("[INFERENCE] Reset to start")
            elif key == ord('s'):  # S - Save debug frame
                filename = debug_dir / f"frame_{current_frame_idx:04d}_prob_{prob if prob else 0:.3f}.png"
                cv2.imwrite(str(filename), display)
                print(f"[INFERENCE] Saved debug frame: {filename}")
            elif playing:
                current_frame_idx += 1
                if current_frame_idx >= total_frames:
                    playing = False
                    print("[INFERENCE] End of video, paused")
        
        cap.release()
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
        inference.run_interactive(video_path)
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
            inference.run_interactive(video_path)
        else:
            # Multiple videos selected (generator)
            for video_path in video_selector:
                if not os.path.exists(video_path):
                    print(f"[WARNING] Video not found: {video_path}, skipping...")
                    continue
                
                print(f"\n[INFERENCE] Starting with video: {os.path.basename(video_path)}")
                inference = Phase4Inference(str(checkpoint_path), device=args.device, threshold=threshold, temperature=temperature)
                inference.run_interactive(video_path)


if __name__ == "__main__":
    main()
