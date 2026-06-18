"""
Multi-label dysfluency detection head.

Pools the shared encoder output H_hat via attentive statistics pooling
(Desplanques et al., 2020) into an utterance embedding u in R^512, then a
linear classifier produces logits over the six dysfluency classes:
block (BLK), prolongation (PRO), sound repetition (SR), word repetition (WR),
interjection (INT), fluent (FLU).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DYSFLUENCY_CLASSES = ["BLK", "PRO", "SR", "WR", "INT", "FLU"]


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling: produces a weighted mean and weighted
    std over the time axis, concatenated into a single utterance vector.

    The attention weights w_t (used here and reused for the SA-CTC soft
    mask in Eq. 4) are produced by a small feed-forward attention network.
    """

    def __init__(self, input_dim: int, attn_hidden_dim: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parameters
        ----------
        h: [B, T, D] frame-level features.
        mask: [B, T] boolean, True for valid (non-padded) frames.

        Returns
        -------
        u: [B, 2*D] concatenated weighted mean/std utterance embedding.
        w: [B, T] normalised attention (pooling) weights -- reused as the
           SA-CTC soft mask base in the decoder (Eq. 4).
        """
        scores = self.attn(h).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)  # [B, T]

        mu = torch.sum(w.unsqueeze(-1) * h, dim=1)  # [B, D]
        var = torch.sum(w.unsqueeze(-1) * (h - mu.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var.clamp_min(1e-8))

        u = torch.cat([mu, std], dim=-1)  # [B, 2D]
        return u, w


class DetectionHead(nn.Module):
    def __init__(self, input_dim: int = 1024, embed_dim: int = 512, num_classes: int = 6) -> None:
        super().__init__()
        self.pooling = AttentiveStatisticsPooling(input_dim)
        self.proj = nn.Linear(2 * input_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, h_hat: torch.Tensor, mask: torch.Tensor | None = None):
        """
        Parameters
        ----------
        h_hat: [B, T, D] layer-weighted encoder output.

        Returns
        -------
        logits: [B, K] raw (pre-sigmoid) class logits.
        p_hat: [B, K] sigmoid-activated detection probabilities, broadcast
               into the SA-CTC cross-attention gate (Eq. 3) and alignment
               mask (Eq. 4) by the decoder.
        w: [B, T] pooling attention weights, reused in Eq. 4.
        """
        u_stats, w = self.pooling(h_hat, mask)
        u = self.proj(u_stats)
        logits = self.classifier(u)
        p_hat = torch.sigmoid(logits)
        return logits, p_hat, w
