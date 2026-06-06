from typing import List, Optional
import numpy as np
import torch
import sys
from pathlib import Path
from collections import deque

# Fix absolute imports for when loaded as a plugin
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from .base_activity import BaseActivity, SuspiciousEvent
from Train.phase4_model import Phase4Classifier
from Train.phase4_types import Phase4Config
from pipeline.kinematic_features import KinematicFeatureExtractor
from pipeline.spatial_normalizer import SpatialNormalizer, NormalizationParams


class ShopliftingActivityDetector(BaseActivity):
    """
    LSTM-based activity detector.
    Uses the trained PyTorch Phase 4 LSTM model to run inference on a rolling 
    temporal window of keypoints to classify "normal" vs "shoplifting".
    """
    def __init__(
        self, 
        model_path: str, 
        seq_length: int = 30, 
        threshold: float = 0.50, 
        cooldown_frames: int = 45,
        smoothing_window: int = 5, 
        consecutive_required: int = 3
    ):
        model_file = Path(model_path).expanduser().resolve()
        
        # Try to load calibrated threshold dynamically from report json if present
        calibrated_threshold = None
        report_candidates = [
            model_file.parent / "phase4_experiment_report.json",
            model_file.with_name(model_file.stem + "_report.json"),
            model_file.with_suffix(".json")
        ]
        import json
        for r_path in report_candidates:
            if r_path.exists():
                try:
                    with open(r_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    if "threshold" in report_data:
                        calibrated_threshold = float(report_data["threshold"])
                        print(f"[ShopliftingActivityDetector] Loaded calibrated threshold {calibrated_threshold:.3f} from {r_path.name}")
                        break
                except Exception:
                    pass
                    
        if calibrated_threshold is not None:
            threshold = calibrated_threshold

        super().__init__("shoplifting_ml", threshold=threshold)
        self.seq_length = seq_length
        self.cooldown_frames = cooldown_frames
        self.frames_since_last_alert = cooldown_frames
        self.smoothing_window = int(smoothing_window)
        self.consecutive_required = int(consecutive_required)
        self.prob_buffer = deque(maxlen=self.smoothing_window)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Preprocessors
        self.normalizer = SpatialNormalizer(NormalizationParams(
            torso_confidence_threshold=0.5,
            allow_invalid_torso=True
        ))
        self.feature_extractor = KinematicFeatureExtractor()
        
        # Initialize Config dynamically
        self.config = Phase4Config(
            sequence_length=self.seq_length,
            input_size=self.feature_extractor.feature_dim(),
            hidden_size=128,
            num_layers=1,
            attention_size=64,
            dropout=0.1
        )
        
        self.use_onnx = model_file.suffix.lower() == ".onnx"
        self.model_loaded = False
        
        if model_file.exists():
            if self.use_onnx:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(model_file), providers=['CPUExecutionProvider'])
                self.model_loaded = True
                print(f"[ShopliftingActivityDetector] Loaded ONNX LSTM model from {model_file} successfully.")
            else:
                self.model = Phase4Classifier(self.config).to(self.device)
                checkpoint = torch.load(str(model_file), map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                self.model_loaded = True
                print(f"[ShopliftingActivityDetector] Loaded PyTorch LSTM model from {model_file} successfully.")
        else:
            print(f"[WARNING] LSTM Model not found at {model_file} - Shoplifting detector disabled.")
            
        # Pose and score sequence buffers
        self.pose_buffer = deque(maxlen=self.seq_length)
        
    def detecta(
        self, 
        keypoints: List[tuple], 
        scores: List[float], 
        frame_id: int, 
        timestamp: float
    ) -> Optional[SuspiciousEvent]:
        self.frames_since_last_alert += 1
        
        if not self.model_loaded:
            return None
            
        raw_keypoints = np.asarray(keypoints, dtype=np.float32).reshape(17, 2)
        raw_scores = np.asarray(scores, dtype=np.float32).reshape(17)
        
        # Normalize frame pose
        normalized_pose = self.normalizer.normalize(raw_keypoints, raw_scores)
        
        # Append to rolling buffer
        self.pose_buffer.append((normalized_pose.keypoints, raw_scores))
        
        if len(self.pose_buffer) < self.seq_length:
            return None
            
        # Preprocess sequence for inference
        normalized_keypoints_list = np.array([kp for kp, _ in self.pose_buffer])  # (T, 17, 2)
        coords_batch = normalized_keypoints_list.astype(np.float32)[np.newaxis, :, :, :]  # (1, T, 17, 2)
        
        # Extract features
        features_batch = self.feature_extractor.transform(coords_batch)  # (1, T, input_size)
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: features_batch.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logit = float(ort_outs[0][0])
            prob = 1.0 / (1.0 + np.exp(-logit))
        else:
            features_tensor = torch.from_numpy(features_batch).float().to(self.device)
            with torch.no_grad():
                logits, _ = self.model(features_tensor)
                prob = torch.sigmoid(logits).item()
            
        # Apply smoothing / consecutive hit evaluation
        self.prob_buffer.append(prob)
        
        if frame_id % 5 == 0:
            print(f"[MLP-LSTM] Frame {frame_id} | suspicious prob: {prob:.3f} | recent: {[round(p, 3) for p in list(self.prob_buffer)]}")
            
        # Count recent windows exceeding threshold
        high_count = sum(1 for p in self.prob_buffer if p >= self.threshold)
        
        if high_count >= self.consecutive_required and self.frames_since_last_alert >= self.cooldown_frames:
            self.frames_since_last_alert = 0
            self.prob_buffer.clear()
            # Retain only half the buffer to prevent rapid consecutive alerts
            for _ in range(self.seq_length // 2):
                if self.pose_buffer:
                    self.pose_buffer.popleft()
                    
            return SuspiciousEvent(
                tipo=self.nome,
                timestamp=timestamp,
                confianca=float(prob),
                frame_id=frame_id,
                descricao=f"Classificador LSTM detectou SHOPLIFTING (Prob: {prob*100:.1f}%)",
                dados_adicionais={"model": "LSTM_attention_30f", "probability": float(prob), "recent_count": int(high_count)}
            )
            
        return None
