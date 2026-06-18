"""
Shape and gradient-flow sanity tests for the detection head and SA-CTC
decoder, operating on synthetic encoder outputs so these tests run without
downloading WavLM-Large or any network access.

A separate, network-requiring smoke test for the full `DysfluentNet`
(including the real WavLM-Large encoder) is provided in
`test_full_model_integration` and is skipped by default -- run it locally
with `pytest tests/ -m integration` once `microsoft/wavlm-large` is cached.
"""
import pytest
import torch

from dysfluentnet.models.detection_head import DetectionHead, AttentiveStatisticsPooling
from dysfluentnet.models.decoder import SACTCDecoder, CrossAttentionGate, DYSFLUENCY_TOKENS


D = 1024  # WavLM-Large hidden size
B, T = 3, 50
K = 6  # number of dysfluency classes


def test_attentive_pooling_shapes():
    pooling = AttentiveStatisticsPooling(input_dim=D)
    h = torch.randn(B, T, D)
    u, w = pooling(h)
    assert u.shape == (B, 2 * D)
    assert w.shape == (B, T)
    # attention weights should sum to 1 along the time axis
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_attentive_pooling_respects_mask():
    pooling = AttentiveStatisticsPooling(input_dim=D)
    h = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, T // 2 :] = False  # mask out the second half

    _, w = pooling(h, mask)
    assert torch.allclose(w[:, T // 2 :], torch.zeros(B, T - T // 2), atol=1e-6)


def test_detection_head_output_shapes():
    head = DetectionHead(input_dim=D, embed_dim=512, num_classes=K)
    h_hat = torch.randn(B, T, D)
    logits, p_hat, w = head(h_hat)

    assert logits.shape == (B, K)
    assert p_hat.shape == (B, K)
    assert w.shape == (B, T)
    assert torch.all((p_hat >= 0) & (p_hat <= 1))


def test_cross_attention_gate_shapes():
    gate = CrossAttentionGate(input_dim=D, num_classes=K)
    h = torch.randn(B, T, D)
    p_hat = torch.rand(B, K)
    g = gate(h, p_hat)
    assert g.shape == (B, T, D)
    assert torch.all((g >= 0) & (g <= 1))  # sigmoid output


def test_sa_ctc_decoder_output_shapes():
    base_vocab_size = 32
    decoder = SACTCDecoder(
        input_dim=D,
        lstm_hidden=512,
        num_lstm_layers=2,
        base_vocab_size=base_vocab_size,
        num_dys_tokens=len(DYSFLUENCY_TOKENS),
        num_classes=K,
    )
    h_hat = torch.randn(B, T, D)
    p_hat = torch.rand(B, K)

    ctc_logits, gated_h = decoder(h_hat, p_hat)

    assert ctc_logits.shape == (B, T, base_vocab_size + len(DYSFLUENCY_TOKENS))
    assert gated_h.shape == (B, T, D)


def test_gradients_flow_through_full_branch():
    head = DetectionHead(input_dim=D, embed_dim=512, num_classes=K)
    decoder = SACTCDecoder(input_dim=D, base_vocab_size=32, num_classes=K)

    h_hat = torch.randn(B, T, D, requires_grad=True)
    det_logits, p_hat, _ = head(h_hat)
    ctc_logits, _ = decoder(h_hat, p_hat)

    loss = det_logits.sum() + ctc_logits.sum()
    loss.backward()

    assert h_hat.grad is not None
    assert not torch.all(h_hat.grad == 0)


@pytest.mark.integration
def test_full_model_integration():
    """Requires network access to download microsoft/wavlm-large from the
    HuggingFace Hub. Skipped in CI by default; run explicitly with
    `pytest tests/ -m integration` once the checkpoint is cached locally."""
    from dysfluentnet.models import DysfluentNet

    model = DysfluentNet(base_vocab_size=32)
    waveform = torch.randn(2, 16000 * 3)  # 3 seconds @ 16kHz
    out = model(waveform)

    assert out.detection_logits.shape == (2, 6)
    assert out.ctc_logits.shape[0] == 2
    assert out.ctc_logits.shape[-1] == 32 + 5
