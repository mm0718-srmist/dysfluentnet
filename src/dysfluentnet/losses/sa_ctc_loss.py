"""
Stutter-Aware CTC (SA-CTC) objective.

Implements Eqs. 4-6 of the paper.

Eq. 4 (soft alignment mask), derived from the detection head's attentive
pooling weights w_t:

    M_k(t) = p_hat_k * softmax_t(w_t)

Eq. 5 (alignment consistency term per dysfluency class k):

    L_align,k = MSE( (1/T) * sum_t M_k(t) * log p(v_k | h_hat_t),  p_hat_k )

Eq. 6 (full SA-CTC objective):

    L*_CTC = L_CTC + lambda * sum_{k in V_dys} p_hat_k * L_align,k

with lambda = 0.2 (grid search) as reported in the paper.

The paper additionally fixes a SA-CTC alignment window of +-15 frames
(~0.3 s) as an implementation hyperparameter (Sec. 6 limitations / Sec. 4.4
implementation details). The exact windowing algorithm is not given as a
closed-form equation in the paper, so here we implement the natural reading:
the alignment term for class k is restricted to a +-`window` neighbourhood
around the frame of peak pooling attention for that utterance, rather than
being computed over the full utterance. Set `window=None` to disable
windowing and reproduce Eq. 5 exactly as written (full-utterance sum).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACTCLoss(nn.Module):
    def __init__(
        self,
        dysfluency_token_ids: Sequence[int],
        lambda_align: float = 0.2,
        window: int | None = 15,
        blank_id: int = 0,
        zero_infinity: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dysfluency_token_ids:
            Vocabulary indices of the five dysfluency tokens
            (<BLK>, <PRO>, <SR>, <WR>, <INT>), in the same class order used
            by the detection head's 6-way output (FLU has no CTC token and
            is excluded here, matching V_dys in the paper).
        lambda_align:
            lambda in Eq. 6. Default 0.2 (paper's grid-searched value).
        window:
            +-frames around the attention peak used to restrict the Eq. 5
            sum. Default 15, matching the paper's reported alignment
            window. Set to None to sum over the full utterance.
        """
        super().__init__()
        self.dysfluency_token_ids = list(dysfluency_token_ids)
        self.lambda_align = lambda_align
        self.window = window
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=zero_infinity)

    def _alignment_term(
        self,
        log_probs: torch.Tensor,  # [B, T, V+] log-softmax CTC outputs
        pooling_weights: torch.Tensor,  # [B, T], w_t
        p_hat: torch.Tensor,  # [B, K]
        frame_mask: torch.Tensor | None,  # [B, T] bool, True = valid frame
    ) -> torch.Tensor:
        B, T, _ = log_probs.shape
        device = log_probs.device

        # softmax_t(w_t), restricted to valid frames if a mask is given.
        w_scores = pooling_weights.clone()
        if frame_mask is not None:
            w_scores = w_scores.masked_fill(~frame_mask, float("-inf"))
        w_softmax = F.softmax(w_scores, dim=-1)  # [B, T]

        if self.window is not None:
            peak_idx = w_softmax.argmax(dim=-1)  # [B]
            t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            window_mask = (t_idx - peak_idx.unsqueeze(1)).abs() <= self.window
            if frame_mask is not None:
                window_mask = window_mask & frame_mask
        else:
            window_mask = frame_mask if frame_mask is not None else torch.ones(
                B, T, dtype=torch.bool, device=device
            )

        valid_counts = window_mask.sum(dim=-1).clamp_min(1).float()  # [B]

        align_losses = []
        for k_class, v_token_id in enumerate(self.dysfluency_token_ids):
            p_hat_k = p_hat[:, k_class]  # [B]
            log_p_vk = log_probs[..., v_token_id]  # [B, T]

            m_kt = p_hat_k.unsqueeze(1) * w_softmax  # Eq. 4: M_k(t)
            masked_term = m_kt * log_p_vk * window_mask.float()
            inner_sum = masked_term.sum(dim=-1) / valid_counts  # (1/T) sum_t [...]

            l_align_k = F.mse_loss(inner_sum, p_hat_k, reduction="none")  # [B]
            align_losses.append(p_hat_k * l_align_k)  # weight by p_hat_k (Eq. 6)

        return torch.stack(align_losses, dim=0).sum(dim=0).mean()  # scalar

    def forward(
        self,
        ctc_logits: torch.Tensor,  # [B, T, V+]
        pooling_weights: torch.Tensor,  # [B, T]
        p_hat: torch.Tensor,  # [B, K]
        targets: torch.Tensor,  # [sum(target_lengths)] flattened token ids
        input_lengths: torch.Tensor,  # [B]
        target_lengths: torch.Tensor,  # [B]
        frame_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total: scalar L*_CTC (Eq. 6).
        ctc: scalar standard CTC component.
        align: scalar weighted alignment component (already includes lambda).
        """
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V+] for nn.CTCLoss
        ctc = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)

        align = self._alignment_term(
            F.log_softmax(ctc_logits, dim=-1), pooling_weights, p_hat, frame_mask
        )
        total = ctc + self.lambda_align * align
        return total, ctc, align


class DysfluentNetLoss(nn.Module):
    """Full training objective (Eq. 7): L = L_det + beta * L*_CTC."""

    def __init__(
        self,
        dysfluency_token_ids: Sequence[int],
        focal_gamma: float = 2.0,
        lambda_align: float = 0.2,
        beta: float = 0.5,
        window: int | None = 15,
        blank_id: int = 0,
    ) -> None:
        super().__init__()
        from .focal_loss import MultiLabelFocalLoss

        self.focal = MultiLabelFocalLoss(gamma=focal_gamma)
        self.sa_ctc = SACTCLoss(
            dysfluency_token_ids=dysfluency_token_ids,
            lambda_align=lambda_align,
            window=window,
            blank_id=blank_id,
        )
        self.beta = beta

    def forward(
        self,
        detection_logits: torch.Tensor,
        detection_targets: torch.Tensor,
        ctc_logits: torch.Tensor,
        pooling_weights: torch.Tensor,
        p_hat: torch.Tensor,
        ctc_targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ):
        l_det = self.focal(detection_logits, detection_targets)
        l_ctc_star, l_ctc, l_align = self.sa_ctc(
            ctc_logits, pooling_weights, p_hat, ctc_targets, input_lengths, target_lengths, frame_mask
        )
        total = l_det + self.beta * l_ctc_star
        return {
            "loss": total,
            "l_det": l_det.detach(),
            "l_ctc": l_ctc.detach(),
            "l_align": l_align.detach(),
            "l_ctc_star": l_ctc_star.detach(),
        }
