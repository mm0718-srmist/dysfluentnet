"""
Multi-label focal loss for the detection head (Eq. 2):

    L_det = -sum_k [ (1-p_k)^gamma * y_k * log(p_k)
                     + p_k^gamma * (1-y_k) * log(1-p_k) ]

with K=6 classes and gamma=2, as specified in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits: [B, K] raw (pre-sigmoid) detection logits.
        targets: [B, K] multi-hot ground-truth labels in {0, 1}.
        """
        p = torch.sigmoid(logits)
        p = p.clamp(min=1e-6, max=1 - 1e-6)

        pos_term = (1 - p) ** self.gamma * targets * torch.log(p)
        neg_term = p ** self.gamma * (1 - targets) * torch.log(1 - p)
        loss_per_class = -(pos_term + neg_term)  # [B, K]
        loss_per_sample = loss_per_class.sum(dim=-1)  # [B]

        if self.reduction == "mean":
            return loss_per_sample.mean()
        if self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample
