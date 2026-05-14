from typing import Tuple

import torch
from torch import nn, Tensor

from .phase4_types import Phase4Config


class BahdanauAttention(nn.Module):
    """Bahdanau-style attention over encoder hidden states.

    energy_t = v^T tanh(W_h h_t + W_s s_last)
    where s_last is the final hidden state of the LSTM (query).
    """

    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.W_h = nn.Linear(hidden_size, attention_size, bias=False)
        self.W_s = nn.Linear(hidden_size, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, encoder_outputs: Tensor, query: Tensor) -> Tensor:
        """Compute attention weights.

        Args:
            encoder_outputs: (B, T, H)
            query: (B, H) typically the final hidden state

        Returns:
            attn_weights: (B, T) softmax over time
        """
        # project
        proj_h = self.W_h(encoder_outputs)  # (B, T, A)
        proj_s = self.W_s(query).unsqueeze(1)  # (B, 1, A)

        energies = self.v(torch.tanh(proj_h + proj_s)).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(energies, dim=1)
        return attn_weights


class Phase4Classifier(nn.Module):
    """Lightweight causal LSTM with Bahdanau attention for binary classification.

    Config-driven: pass a `Phase4Config` instance to control dimensions.
    """

    def __init__(self, config: Phase4Config) -> None:
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(config.hidden_size, config.attention_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, T, F) input kinematic features

        Returns:
            logits: (B,) raw logits
            attn_weights: (B, T) attention distribution over time
        """
        outputs, (h_n, c_n) = self.lstm(x)  # outputs: (B, T, H)

        # final hidden state from last LSTM layer
        final_hidden = h_n[-1]  # (B, H)

        attn_weights = self.attention(outputs, final_hidden)  # (B, T)

        # context vector: weighted sum over time
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)  # (B, H)

        logits = self.classifier(context).squeeze(-1)  # (B,)
        return logits, attn_weights
