from typing import Optional
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from .phase4_types import Phase4Config
from .losses import ConfidenceWeightedBCELoss


class Trainer:
    CRUCIAL_JOINTS = [7, 8, 9, 10]  # elbows and wrists (COCO indices)

    def __init__(self, model: nn.Module, config: Phase4Config) -> None:
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.loss_fn = ConfidenceWeightedBCELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            poses = batch["poses"].to(self.device)  # (B, T, input_size)
            confidences = batch["confidences"].to(self.device)  # (B, T, K)
            labels = batch["labels"].to(self.device)  # (B,)

            logits, _ = self.model(poses)

            # compute sample-level raw weight w_raw as mean confidence over crucial joints
            # confidences: (B, T, K)
            crucial = torch.tensor(self.CRUCIAL_JOINTS, dtype=torch.long, device=self.device)
            w_raw = confidences.index_select(dim=2, index=crucial).mean(dim=(1, 2))  # (B,)

            loss = self.loss_fn(logits, labels, w_raw)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += float(loss.detach().cpu().item())
            count += 1

        return total_loss / max(1, count)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for batch in dataloader:
            poses = batch["poses"].to(self.device)
            confidences = batch["confidences"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(poses)
            crucial = torch.tensor(self.CRUCIAL_JOINTS, dtype=torch.long, device=self.device)
            w_raw = confidences.index_select(dim=2, index=crucial).mean(dim=(1, 2))
            loss = self.loss_fn(logits, labels, w_raw)

            total_loss += float(loss.detach().cpu().item())
            count += 1

        return total_loss / max(1, count)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
