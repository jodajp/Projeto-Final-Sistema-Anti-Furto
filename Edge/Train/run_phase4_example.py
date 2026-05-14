"""Example training snippet for Phase 4 classifier with a synthetic dataset.

This demonstrates the DataLoader contract and runs one training epoch.
"""
import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.kinematic_features import KinematicFeatureExtractor
from Edge.Train.phase4_types import Phase4Config
from Edge.Train.phase4_model import Phase4Classifier
from Edge.Train.train_phase4 import Trainer


class SyntheticPoseDataset(Dataset):
    def __init__(self, feats: np.ndarray, confid: np.ndarray, labels: np.ndarray) -> None:
        self.feats = feats.astype(np.float32)
        self.confid = confid.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.feats.shape[0]

    def __getitem__(self, idx: int):
        return {
            "poses": self.feats[idx],  # (T, F)
            "confidences": self.confid[idx],  # (T, K)
            "labels": self.labels[idx],
        }


def main(seed: Optional[int] = 0) -> None:
    rng = np.random.RandomState(seed)
    B = 64
    T = 30
    K = 17
    C = 2

    # synthetic coordinates (B, T, K, 2)
    coords = rng.randn(B, T, K, C).astype(np.float32)
    # synthetic confidences (B, T, K)
    confid = rng.rand(B, T, K).astype(np.float32)
    # labels
    labels = rng.randint(0, 2, size=(B,)).astype(np.float32)

    extractor = KinematicFeatureExtractor()
    feats = extractor.transform(coords)  # (B, T, F)

    config = Phase4Config(sequence_length=T, input_size=feats.shape[2], hidden_size=64, num_layers=1, attention_size=32, dropout=0.0, batch_size=16)
    model = Phase4Classifier(config)
    trainer = Trainer(model, config)

    dataset = SyntheticPoseDataset(feats, confid, labels)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    train_loss = trainer.train_epoch(dataloader)
    print(f"Example training epoch completed — loss: {train_loss:.6f}")

    # save example model
    trainer.save("./models/phase4_example.pth")


if __name__ == "__main__":
    main()
