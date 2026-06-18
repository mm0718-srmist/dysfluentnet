"""
Shared SSL encoder: frozen WavLM-Large with a learned layer-wise attention
weighted sum over its 24 transformer layers (paper Eq. 1).

    H_hat = sum_l alpha_l * H^(l),   sum_l alpha_l = 1

The detection head and the SA-CTC decoder each receive H_hat, but because
{alpha_l} is a *single* shared set of weights in the base model, the paper's
analysis in Sec. 5.1 trains the detection head and decoder branches with
their own learned projections on top of a common H_hat. We expose the raw
per-layer hidden states too, so a multi-head variant with separate alpha_l
per branch (an easy ablation extension) can be built without modifying this
module.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import WavLMModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "transformers is required for the WavLM encoder. "
        "Install with `pip install transformers`."
    ) from exc


class LayerWeightedWavLM(nn.Module):
    """Frozen WavLM-Large feature extractor with a learned softmax-weighted
    sum over its transformer layers.

    Parameters
    ----------
    model_name:
        HuggingFace checkpoint id. Defaults to the 24-layer / 1024-dim
        WavLM-Large model used in the paper (317M parameters).
    num_layers:
        Number of transformer layers to weight over. WavLM-Large has 24.
    freeze:
        If True (default, matches the paper), all WavLM parameters are
        frozen and only the layer-attention weights {alpha_l} are trained.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        num_layers: int = 24,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.num_layers = num_layers
        self.hidden_size = self.wavlm.config.hidden_size  # 1024 for Large

        # Learned per-layer logits; softmax-normalised at forward time so
        # sum_l alpha_l = 1 holds by construction (Eq. 1 constraint).
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

        if freeze:
            for p in self.wavlm.parameters():
                p.requires_grad_(False)
            self.wavlm.eval()

        self._freeze = freeze

    def train(self, mode: bool = True) -> "LayerWeightedWavLM":
        super().train(mode)
        if self._freeze:
            # Keep WavLM itself in eval mode (no dropout / running BN
            # stats updates) even when the wrapping module is in train().
            self.wavlm.eval()
        return self

    @property
    def alpha(self) -> torch.Tensor:
        """Softmax-normalised layer weights, alpha_l in the paper."""
        return F.softmax(self.layer_logits, dim=0)

    def _extract_hidden_states(self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        ctx = torch.no_grad() if self._freeze else torch.enable_grad()
        with ctx:
            outputs = self.wavlm(
                input_values=waveform,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # outputs.hidden_states: tuple of (num_layers + 1) tensors
        # [embedding output, layer_1, ..., layer_24]; we use the 24
        # transformer-layer outputs and drop the embedding layer.
        hidden_states = outputs.hidden_states[1 : self.num_layers + 1]
        return torch.stack(hidden_states, dim=0)  # [L, B, T, D]

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform: [B, T_samples] raw 16 kHz audio.
        attention_mask: [B, T_samples] optional padding mask.

        Returns
        -------
        h_hat: [B, T_frames, D] layer-weighted frame representations.
        """
        layer_stack = self._extract_hidden_states(waveform, attention_mask)  # [L, B, T, D]
        alpha = self.alpha.view(self.num_layers, 1, 1, 1)
        h_hat = (alpha * layer_stack).sum(dim=0)  # [B, T, D]
        return h_hat
