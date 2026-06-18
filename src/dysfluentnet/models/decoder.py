"""
Dysfluency-conditioned SA-CTC decoder.

A 2-layer BiLSTM (512 units per direction) operating on H_hat, gated by the
detection head's utterance-level confidence p_hat through a cross-attention
gating layer (Eq. 3):

    G_t = sigmoid(W_g [h_t ; p_hat])

The gated representation (G_t * h_t) is passed to the LSTM and projected to
|V+| logits, where V+ = V (standard CTC vocabulary) union five dysfluency
token types {<BLK>, <PRO>, <SR>, <WR>, <INT>}.
"""
from __future__ import annotations

import torch
import torch.nn as nn

DYSFLUENCY_TOKENS = ["<BLK>", "<PRO>", "<SR>", "<WR>", "<INT>"]


class CrossAttentionGate(nn.Module):
    """Implements Eq. 3: G_t = sigmoid(W_g [h_t ; p_hat])."""

    def __init__(self, input_dim: int, num_classes: int = 6) -> None:
        super().__init__()
        self.gate = nn.Linear(input_dim + num_classes, input_dim)

    def forward(self, h: torch.Tensor, p_hat: torch.Tensor) -> torch.Tensor:
        """
        h: [B, T, D] encoder frames.
        p_hat: [B, K] utterance-level detection probabilities (broadcast
               to every frame, as stated in the paper).
        """
        B, T, D = h.shape
        p_broadcast = p_hat.unsqueeze(1).expand(B, T, p_hat.shape[-1])
        gate_input = torch.cat([h, p_broadcast], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))  # [B, T, D]
        return g


class SACTCDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        lstm_hidden: int = 512,
        num_lstm_layers: int = 2,
        base_vocab_size: int = 32,
        num_dys_tokens: int = len(DYSFLUENCY_TOKENS),
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.gate = CrossAttentionGate(input_dim, num_classes)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        vocab_plus_size = base_vocab_size + num_dys_tokens  # |V+|
        self.proj = nn.Linear(2 * lstm_hidden, vocab_plus_size)
        self.vocab_plus_size = vocab_plus_size

    def forward(self, h_hat: torch.Tensor, p_hat: torch.Tensor):
        """
        Returns
        -------
        ctc_logits: [B, T, |V+|] pre-log-softmax CTC emission logits.
        gated_h: [B, T, D] gated frame representations (G_t * h_t), reused
                 by the SA-CTC alignment term (Eq. 4/5).
        """
        g = self.gate(h_hat, p_hat)
        gated_h = g * h_hat
        lstm_out, _ = self.lstm(gated_h)
        ctc_logits = self.proj(lstm_out)
        return ctc_logits, gated_h
