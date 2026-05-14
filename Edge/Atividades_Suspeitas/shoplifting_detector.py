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

class ActionClassifier(torch.nn.Module):
    def __init__(self, seq_len=30, input_dim=34, hidden_dim=64):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(seq_len * input_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            # No dropout needed for inference/evaluation mode, but keeping same architecture to load state_dict
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


class ShopliftingActivityDetector(BaseActivity):
    """
    ML-based activity detector.
    Uses the trained PyTorch model to run inference on a rolling temporal window
    of keypoints to classify "normal" vs "shoplifting".
    """
    def __init__(self, model_path: str, seq_length: int = 45, threshold: float = 0.65, cooldown_frames: int = 45,
                 smoothing_window: int = 5, consecutive_required: int = 3):
        super().__init__("shoplifting_ml", threshold=threshold)
        self.seq_length = seq_length
        self.buffer = []
        self.cooldown_frames = cooldown_frames
        self.frames_since_last_alert = cooldown_frames
        # Short-term probability smoothing / consecutive-hit requirements
        self.smoothing_window = int(smoothing_window)
        self.consecutive_required = int(consecutive_required)
        self.prob_buffer = deque(maxlen=self.smoothing_window)
        
        # Load Model
        self.model = ActionClassifier(seq_len=seq_length, input_dim=34)
        
        model_file = Path(model_path).expanduser().resolve()
        if model_file.exists():
            self.model.load_state_dict(torch.load(str(model_file), map_location='cpu'))
            self.model.eval()
            self.model_loaded = True
            print(f"[ShopliftingActivityDetector] Loaded model from {model_file} successfully.")
        else:
            print(f"[WARNING] ML Model not found at {model_file} - Shoplifting detector disabled.")
            self.model_loaded = False
            
    def _normalize_window(self, kpts):
        """Matches the normalization done in training script (min-max bounding box locally)"""
        normalized = np.zeros_like(kpts)
        for t in range(kpts.shape[0]):
            frame_kpts = kpts[t]
            valid_kpts = frame_kpts[~np.isnan(frame_kpts).any(axis=1)]
            if len(valid_kpts) > 0:
                min_xy = np.min(valid_kpts, axis=0)
                max_xy = np.max(valid_kpts, axis=0)
                range_xy = np.maximum(max_xy - min_xy, 1e-5)
                # Keep the same formula as training:
                normalized[t] = (frame_kpts - min_xy) / range_xy
                # MUST FLIP Y-AXIS TO MATCH OPENPOSE COCO FORMAT FROM TRAINING!
                normalized[t, :, 1] = 1.0 - normalized[t, :, 1]
            else:
                normalized[t] = frame_kpts
        return np.nan_to_num(normalized)
            
    def detecta(self, keypoints: List[tuple], scores: List[float], 
                frame_id: int, timestamp: float) -> Optional[SuspiciousEvent]:
        
        self.frames_since_last_alert += 1
        
        if not self.model_loaded:
            return None
            
        kp_array = np.array(keypoints, dtype=np.float32)
        
        self.buffer.append(kp_array)
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)
            
        if len(self.buffer) < self.seq_length:
            return None
            
        # Inference preparation
        window = np.stack(self.buffer) # (30, 17, 2)
        norm_window = self._normalize_window(window) # Normalize window
        
        # Add debugging out to see scale.
        # print("raw temp std:", window.std(), "norm temp std:", norm_window.std())
        if self.frames_since_last_alert % 5 == 0:
            print("norm_window first 3kpts of first frame:\n", norm_window[0][:3])
            
        input_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0) # (1, 30, 17, 2)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            suspicious_prob = probs[0, 1].item()

        # Append to short-term buffer and evaluate consecutive hits
        self.prob_buffer.append(suspicious_prob)

        # Debugging the inference to understand what the model sees internally
        print(f"Frame {frame_id} | suspicious prob: {suspicious_prob:.3f} | recent: {[round(p,3) for p in list(self.prob_buffer)]}")

        # Count how many recent windows exceed the per-window threshold
        high_count = sum(1 for p in self.prob_buffer if p >= self.threshold)

        if high_count >= self.consecutive_required and self.frames_since_last_alert >= self.cooldown_frames:
            self.frames_since_last_alert = 0

            # To avoid rapid re-triggering, clear the short-term buffer and half the keypoint buffer
            self.prob_buffer.clear()
            self.buffer = self.buffer[self.seq_length // 2:]

            return SuspiciousEvent(
                tipo=self.nome,
                timestamp=timestamp,
                confianca=float(suspicious_prob),
                frame_id=frame_id,
                descricao=f"Classificador ML detectou SHOPLIFTING (Prob: {suspicious_prob*100:.1f}%)",
                dados_adicionais={"model": "MLP_temporal_30f", "probability": float(suspicious_prob), "recent_count": int(high_count)}
            )

        return None
