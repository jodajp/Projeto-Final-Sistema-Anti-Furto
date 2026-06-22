from typing import List, Optional
import numpy as np
from collections import deque
import json
from pathlib import Path

from .base_activity import BaseActivity, SuspiciousEvent
from pipeline.kinematic_features import KinematicFeatureExtractor
from pipeline.spatial_normalizer import NormalizedPose


class ShopliftingActivityDetector(BaseActivity):
    """
    LSTM-based activity detector.
    Uses the trained PyTorch/ONNX Phase 4 LSTM model to run inference on a rolling 
    temporal window of keypoints to classify "normal" vs "shoplifting".
    """
    def __init__(
        self, 
        model_path: str, 
        seq_length: int = 45, 
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
        self.smoothing_window = int(smoothing_window)
        self.consecutive_required = int(consecutive_required)
        
        # Track-specific state buffers
        self.pose_buffers = {}              # track_id -> deque of poses
        self.prob_buffers = {}              # track_id -> deque of probs
        self.frames_since_last_alerts = {}  # track_id -> frames since last alert
        
        self.feature_extractor = KinematicFeatureExtractor()
        self.use_onnx = model_file.suffix.lower() == ".onnx"
        self.model_loaded = False
        self.device = None
        self.config = None
        self.model = None
        
        # Load the model only once at initialization
        if model_file.exists():
            if self.use_onnx:
                import onnxruntime as ort
                self.ort_session = ort.InferenceSession(str(model_file), providers=['CPUExecutionProvider'])
                self.model_loaded = True
                print(f"[ShopliftingActivityDetector] Loaded ONNX LSTM model from {model_file} successfully.")
            else:
                import torch
                from Train.phase4_model import Phase4Classifier
                from Train.phase4_types import Phase4Config

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.config = Phase4Config(
                    sequence_length=self.seq_length,
                    input_size=self.feature_extractor.feature_dim(),
                    hidden_size=128,
                    num_layers=1,
                    attention_size=64,
                    dropout=0.1
                )
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
            
    def limpa_tracks_inativas(self, ids_presentes: set):
        """Limpa o estado armazenado para tracks inativas para evitar vazamentos de memória."""
        for track_id in list(self.pose_buffers.keys()):
            if track_id not in ids_presentes:
                self.pose_buffers.pop(track_id, None)
                self.prob_buffers.pop(track_id, None)
                self.frames_since_last_alerts.pop(track_id, None)

    def detecta(
        self, 
        norm_pose: NormalizedPose, 
        frame_id: int, 
        timestamp: float,
        track_id: Optional[int] = None
    ) -> Optional[SuspiciousEvent]:
        
        if not self.model_loaded or not norm_pose:
            return None
            
        tid = 0 if track_id is None else track_id
        
        # Initialize track-specific buffers if they don't exist yet
        if tid not in self.pose_buffers:
            self.pose_buffers[tid] = deque(maxlen=self.seq_length)
            self.prob_buffers[tid] = deque(maxlen=self.smoothing_window)
            self.frames_since_last_alerts[tid] = self.cooldown_frames
            
        self.frames_since_last_alerts[tid] += 1
        
        # Se a pose for inválida (NaNs), tenta usar o último frame válido se disponível
        kp_to_use = norm_pose.keypoints
        if not norm_pose.is_valid and len(self.pose_buffers[tid]) > 0:
            kp_to_use = self.pose_buffers[tid][-1][0]  # Mantém a última pose válida
            
        # Append to rolling buffer for this specific track
        self.pose_buffers[tid].append((kp_to_use, norm_pose.scores))
        
        if len(self.pose_buffers[tid]) < self.seq_length:
            return None
            
        # Preprocess sequence for inference
        normalized_keypoints_list = np.array([kp for kp, _ in self.pose_buffers[tid]])  # (T, 17, 2)
        coords_batch = normalized_keypoints_list.astype(np.float32)[np.newaxis, :, :, :]  # (1, T, 17, 2)
        
        # Extract features
        features_batch = self.feature_extractor.transform(coords_batch)  # (1, T, input_size)
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: features_batch.astype(np.float32)}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logit = float(ort_outs[0][0])
            prob = 1.0 / (1.0 + np.exp(-logit))
        else:
            import torch
            features_tensor = torch.from_numpy(features_batch).float().to(self.device)
            with torch.no_grad():
                logits, _ = self.model(features_tensor)
                prob = torch.sigmoid(logits).item()
            
        # Apply smoothing / consecutive hit evaluation
        self.prob_buffers[tid].append(prob)
        
        if frame_id % 5 == 0:
            print(f"[MLP-LSTM] Track {tid} | Frame {frame_id} | suspicious prob: {prob:.3f} | recent: {[round(p, 3) for p in list(self.prob_buffers[tid])]}")
            
        # Count recent windows exceeding threshold
        high_count = sum(1 for p in self.prob_buffers[tid] if p >= self.threshold)
        
        if high_count >= self.consecutive_required and self.frames_since_last_alerts[tid] >= self.cooldown_frames:
            self.frames_since_last_alerts[tid] = 0
            self.prob_buffers[tid].clear()
            
            # Retain only half the buffer to prevent rapid consecutive alerts
            for _ in range(self.seq_length // 2):
                if self.pose_buffers[tid]:
                    self.pose_buffers[tid].popleft()
                    
            return SuspiciousEvent(
                tipo=self.nome,
                timestamp=timestamp,
                confianca=float(prob),
                frame_id=frame_id,
                pessoa_id=track_id,
                descricao=f"Classificador LSTM detectou SHOPLIFTING (Prob: {prob*100:.1f}%)",
                dados_adicionais={
                    "model": "LSTM_attention_45f", 
                    "probability": float(prob), 
                    "recent_count": int(high_count)
                }
            )
            
        return None
