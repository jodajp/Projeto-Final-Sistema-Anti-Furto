from dataclasses import dataclass
from typing import Optional


@dataclass
class Phase4Config:
    sequence_length: int = 30
    input_size: int = 34  # 17 keypoints * (x,y)
    hidden_size: int = 128
    num_layers: int = 1
    attention_size: int = 64
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 30
    weight_decay: float = 0.0
    confidence_weighted_loss: bool = True
    loss_type: str = "focal"  # "bce" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    device: Optional[str] = None
