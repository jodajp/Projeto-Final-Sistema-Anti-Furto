from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor


class CausalLSTMAttention(nn.Module):
    """Causal LSTM with temporal attention for sequence classification.

    Expects input shape (batch, seq_len, input_size).
    Returns logits shape (batch,).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        attention_size: int = 64,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_size = attention_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn_proj = nn.Linear(hidden_size, attention_size, bias=True)
        self.attn_v = nn.Linear(attention_size, 1, bias=False)

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, input_size)

        Returns:
            logits: Tensor of shape (B,) — raw logits for binary classification
            attn_weights: Tensor of shape (B, T) — attention over time
        """
        # LSTM (causal by design: no future context used)
        outputs, _ = self.lstm(x)  # outputs: (B, T, H)

        # Attention (vectorized over batch & time)
        proj = torch.tanh(self.attn_proj(outputs))  # (B, T, A)
        scores = self.attn_v(proj).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(scores, dim=1)  # (B, T)

        # Context vector: weighted sum over time
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)  # (B, H)

        logits = self.classifier(context).squeeze(-1)  # (B,)
        return logits, attn_weights
