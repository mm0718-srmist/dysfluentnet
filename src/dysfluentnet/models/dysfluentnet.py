"""
Full DysfluentNet model: shared frozen WavLM-Large encoder feeding a
multi-label detection head and a dysfluency-conditioned SA-CTC decoder
(Fig. 1 / Sec. 4.1 of the paper).

Also provides `PipelineBaseline`, the matched-encoder ablation used in the
paper to isolate the contribution of joint coupling: same frozen encoder,
detection head, and BiLSTM-CTC decoder, trained independently with no
cross-attention gate and no SA-CTC alignment term.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import LayerWeightedWavLM
from .detection_head import DetectionHead
from .decoder import SACTCDecoder, DYSFLUENCY_TOKENS


@dataclass
class DysfluentNetOutput:
    detection_logits: torch.Tensor  # [B, K]
    p_hat: torch.Tensor             # [B, K]
    ctc_logits: torch.Tensor        # [B, T, |V+|]
    pooling_weights: torch.Tensor   # [B, T]  (w_t, reused for SA-CTC mask)
    gated_h: torch.Tensor           # [B, T, D]


class DysfluentNet(nn.Module):
    def __init__(
        self,
        wavlm_model_name: str = "microsoft/wavlm-large",
        num_layers: int = 24,
        encoder_dim: int = 1024,
        detection_embed_dim: int = 512,
        num_classes: int = 6,
        lstm_hidden: int = 512,
        num_lstm_layers: int = 2,
        base_vocab_size: int = 32,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = LayerWeightedWavLM(
            model_name=wavlm_model_name, num_layers=num_layers, freeze=freeze_encoder
        )
        self.detection_head = DetectionHead(
            input_dim=encoder_dim, embed_dim=detection_embed_dim, num_classes=num_classes
        )
        self.decoder = SACTCDecoder(
            input_dim=encoder_dim,
            lstm_hidden=lstm_hidden,
            num_lstm_layers=num_lstm_layers,
            base_vocab_size=base_vocab_size,
            num_dys_tokens=len(DYSFLUENCY_TOKENS),
            num_classes=num_classes,
        )

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> DysfluentNetOutput:
        h_hat = self.encoder(waveform, attention_mask)
        frame_mask = None
        if attention_mask is not None:
            # Down-sample the sample-level mask to the frame rate of h_hat.
            frame_mask = nn.functional.interpolate(
                attention_mask.unsqueeze(1).float(), size=h_hat.shape[1], mode="nearest"
            ).squeeze(1).bool()

        det_logits, p_hat, w = self.detection_head(h_hat, frame_mask)
        ctc_logits, gated_h = self.decoder(h_hat, p_hat)

        return DysfluentNetOutput(
            detection_logits=det_logits,
            p_hat=p_hat,
            ctc_logits=ctc_logits,
            pooling_weights=w,
            gated_h=gated_h,
        )


class PipelineBaseline(nn.Module):
    """Matched-encoder ablation: detection head and a plain BiLSTM-CTC
    decoder trained independently, with no cross-attention gate and no
    SA-CTC alignment term -- isolates the benefit of joint coupling from
    the benefit of the shared encoder (Sec. 5.2 baseline)."""

    def __init__(
        self,
        wavlm_model_name: str = "microsoft/wavlm-large",
        num_layers: int = 24,
        encoder_dim: int = 1024,
        detection_embed_dim: int = 512,
        num_classes: int = 6,
        lstm_hidden: int = 512,
        num_lstm_layers: int = 2,
        vocab_plus_size: int = 37,  # base_vocab_size + 5 dysfluency tokens
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = LayerWeightedWavLM(
            model_name=wavlm_model_name, num_layers=num_layers, freeze=freeze_encoder
        )
        self.detection_head = DetectionHead(
            input_dim=encoder_dim, embed_dim=detection_embed_dim, num_classes=num_classes
        )
        self.lstm = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.ctc_proj = nn.Linear(2 * lstm_hidden, vocab_plus_size)

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None):
        h_hat = self.encoder(waveform, attention_mask)
        det_logits, p_hat, _ = self.detection_head(h_hat, None)
        lstm_out, _ = self.lstm(h_hat)  # no gating: ungated h_hat
        ctc_logits = self.ctc_proj(lstm_out)
        return det_logits, p_hat, ctc_logits
