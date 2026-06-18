"""Unit tests for the loss functions. These do not require downloading
WavLM-Large or any GPU -- they operate on small synthetic tensors directly."""
import torch

from dysfluentnet.losses.focal_loss import MultiLabelFocalLoss
from dysfluentnet.losses.sa_ctc_loss import SACTCLoss
from dysfluentnet.metrics.di_wer import di_wer, wer, corpus_di_wer


def test_focal_loss_shape_and_finite():
    loss_fn = MultiLabelFocalLoss(gamma=2.0)
    logits = torch.randn(4, 6, requires_grad=True)
    targets = torch.randint(0, 2, (4, 6)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None


def test_focal_loss_perfect_prediction_is_near_zero():
    loss_fn = MultiLabelFocalLoss(gamma=2.0)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    # Large-magnitude logits matching the targets -> near-perfect sigmoid.
    logits = (targets * 2 - 1) * 20.0
    loss = loss_fn(logits, targets)
    assert loss.item() < 1e-3


def test_sa_ctc_loss_runs_end_to_end():
    B, T, vocab_plus, K = 2, 20, 10, 6
    dys_token_ids = [5, 6, 7, 8, 9]

    loss_fn = SACTCLoss(dysfluency_token_ids=dys_token_ids, lambda_align=0.2, window=5, blank_id=0)

    ctc_logits = torch.randn(B, T, vocab_plus, requires_grad=True)
    pooling_weights = torch.rand(B, T)
    p_hat = torch.rand(B, K)

    targets = torch.tensor([1, 2, 3, 1, 2])  # flattened target tokens
    input_lengths = torch.tensor([T, T])
    target_lengths = torch.tensor([3, 2])

    total, ctc, align = loss_fn(ctc_logits, pooling_weights, p_hat, targets, input_lengths, target_lengths)

    assert torch.isfinite(total)
    assert torch.isfinite(ctc)
    assert torch.isfinite(align)
    total.backward()
    assert ctc_logits.grad is not None


def test_di_wer_identical_sequences_is_zero():
    seq = ["I", "<BLK>", "want", "to", "go"]
    assert di_wer(seq, seq) == 0.0
    assert wer(seq, seq) == 0.0


def test_di_wer_penalises_dropped_dysfluency_token():
    ref = ["I", "<BLK>", "want", "to", "go"]
    hyp_dropped = ["I", "want", "to", "go"]  # silently deletes <BLK>

    assert di_wer(hyp_dropped, ref) > 0.0
    # Standard WER ignores the dropped dysfluency token since both
    # sequences are stripped of dysfluency tokens before alignment.
    assert wer(hyp_dropped, ref) == 0.0


def test_corpus_di_wer_aggregation():
    hyps = [["a", "b"], ["c", "<SR>", "d"]]
    refs = [["a", "b"], ["c", "<SR>", "d"]]
    result = corpus_di_wer(hyps, refs)
    assert result["wer"] == 0.0
    assert result["di_wer"] == 0.0
