import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConfidenceWeightedBCELoss(nn.Module):
    """Binary cross-entropy with sample-level confidence weighting.

    Weighting scheme:
        w_q = 0.5 + 0.5 * w_raw
    where w_raw is the mean confidence for crucial joints in [0,1].

    The loss returned is the mean of w_q * BCE(logits, targets).
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = float(eps)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor, w_raw: Tensor) -> Tensor:
        """Compute the confidence-weighted BCE loss.

        Args:
            logits: Tensor of shape (B,) or (B,1) — raw model logits.
            targets: Tensor of shape (B,) or (B,1) — binary labels {0,1}.
            w_raw: Tensor of shape (B,) or (B,1) — sample-level raw weights in [0,1].

        Returns:
            loss: scalar Tensor if reduction != 'none', else Tensor of shape (B,)
        """
        logits = logits.reshape(-1)

        targets = targets.reshape(-1).to(dtype=logits.dtype)
        w_raw = w_raw.reshape(-1).to(dtype=logits.dtype)

        # sanitize weights: replace NaN, clamp to [0,1]
        w_raw = torch.nan_to_num(w_raw, nan=0.0, posinf=1.0, neginf=0.0)
        w_raw = torch.clamp(w_raw, 0.0, 1.0)

        w_q = 0.5 + 0.5 * w_raw

        # per-sample BCE (no reduction)
        per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        weighted = per_sample * w_q

        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        # default mean
        return weighted.mean()


class ConfidenceWeightedFocalLoss(nn.Module):
    """Binary focal loss with sample-level confidence weighting.

    Loss formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        Loss = w_q * FL(p_t)
    where:
        w_q = 0.5 + 0.5 * w_raw
        w_raw is the mean confidence for crucial joints in [0,1].
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor, w_raw: Tensor) -> Tensor:
        logits = logits.reshape(-1)

        targets = targets.reshape(-1).to(dtype=logits.dtype)
        w_raw = w_raw.reshape(-1).to(dtype=logits.dtype)

        # sanitize weights: replace NaN, clamp to [0,1]
        w_raw = torch.nan_to_num(w_raw, nan=0.0, posinf=1.0, neginf=0.0)
        w_raw = torch.clamp(w_raw, 0.0, 1.0)

        w_q = 0.5 + 0.5 * w_raw

        # probabilities
        probs = torch.sigmoid(logits)
        
        # avoid log(0) and log(1)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        loss_pos = -self.alpha * targets * ((1.0 - probs) ** self.gamma) * torch.log(probs)
        loss_neg = -(1.0 - self.alpha) * (1.0 - targets) * (probs ** self.gamma) * torch.log(1.0 - probs)
        
        per_sample = loss_pos + loss_neg
        weighted = per_sample * w_q

        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        # default mean
        return weighted.mean()
