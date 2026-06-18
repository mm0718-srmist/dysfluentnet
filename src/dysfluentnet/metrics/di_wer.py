"""
Dysfluency-inclusive Word Error Rate (DI-WER).

    DI-WER = d(y_hat_plus, y_plus) / |y_plus|

where y_plus is the reference token sequence with dysfluency tokens
included (derived from word-level disfluency annotations), y_hat_plus is
the corresponding hypothesis, and d(.,.) is the edit (Levenshtein) distance
over tokens. Under DI-WER, correctly transcribing a sound repetition as
<SR> incurs no penalty, while silently deleting it counts as a deletion
error.

Standard WER is simply DI-WER computed with dysfluency tokens stripped
from both hypothesis and reference before alignment.
"""
from __future__ import annotations

from typing import Sequence


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Token-level edit distance via classic DP (O(len(a)*len(b)))."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[m]


DYSFLUENCY_TOKEN_SET = {"<BLK>", "<PRO>", "<SR>", "<WR>", "<INT>"}


def strip_dysfluency_tokens(tokens: Sequence[str]) -> list[str]:
    return [t for t in tokens if t not in DYSFLUENCY_TOKEN_SET]


def di_wer(hypothesis: Sequence[str], reference_plus: Sequence[str]) -> float:
    """DI-WER for a single utterance. `reference_plus` must already include
    dysfluency tokens at their correct positions; `hypothesis` is the raw
    model output token sequence (already containing <BLK>/<PRO>/... tokens
    where emitted)."""
    if len(reference_plus) == 0:
        return 0.0 if len(hypothesis) == 0 else float("inf")
    dist = _levenshtein(list(hypothesis), list(reference_plus))
    return dist / len(reference_plus)


def wer(hypothesis: Sequence[str], reference_plus: Sequence[str]) -> float:
    """Standard WER: both sequences have dysfluency tokens stripped first,
    matching the paper's protocol of evaluating WER against the fluent
    reference."""
    hyp_fluent = strip_dysfluency_tokens(hypothesis)
    ref_fluent = strip_dysfluency_tokens(reference_plus)
    if len(ref_fluent) == 0:
        return 0.0 if len(hyp_fluent) == 0 else float("inf")
    dist = _levenshtein(hyp_fluent, ref_fluent)
    return dist / len(ref_fluent)


def corpus_di_wer(
    hypotheses: Sequence[Sequence[str]], references_plus: Sequence[Sequence[str]]
) -> dict[str, float]:
    """Aggregate (pooled, not averaged-per-utterance) WER and DI-WER over a
    test set, matching standard ASR corpus-level reporting."""
    assert len(hypotheses) == len(references_plus)

    total_dist_diwer, total_len_diwer = 0, 0
    total_dist_wer, total_len_wer = 0, 0

    for hyp, ref in zip(hypotheses, references_plus):
        ref = list(ref)
        hyp = list(hyp)
        total_dist_diwer += _levenshtein(hyp, ref)
        total_len_diwer += len(ref)

        hyp_f = strip_dysfluency_tokens(hyp)
        ref_f = strip_dysfluency_tokens(ref)
        total_dist_wer += _levenshtein(hyp_f, ref_f)
        total_len_wer += len(ref_f)

    return {
        "wer": total_dist_wer / max(total_len_wer, 1),
        "di_wer": total_dist_diwer / max(total_len_diwer, 1),
    }
