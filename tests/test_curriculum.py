import math

import numpy as np

from dysfluentnet.data.curriculum import fleiss_kappa, assign_tier, CurriculumTiers


def test_fleiss_kappa_full_agreement_is_one():
    # All 3 annotators agree on a single category -> perfect agreement.
    counts = np.array([3, 0, 0, 0, 0, 0])
    assert math.isclose(fleiss_kappa(counts), 1.0, abs_tol=1e-6)


def test_assign_tier_boundaries():
    assert assign_tier(0.95) == "T1"
    assert assign_tier(0.80) == "T1"
    assert assign_tier(0.79) == "T2"
    assert assign_tier(0.60) == "T2"
    assert assign_tier(0.55) == "T3"
    assert assign_tier(0.45) == "T4"
    assert assign_tier(0.10) == "T5"


def test_curriculum_active_clip_ids_expands_cumulatively():
    clip_to_tier = {
        "a": "T1",
        "b": "T2",
        "c": "T3",
        "d": "T4",
        "e": "T5",
    }
    tiers = CurriculumTiers(clip_to_tier=clip_to_tier)

    assert set(tiers.active_clip_ids(1)) == {"a"}
    assert set(tiers.active_clip_ids(2)) == {"a", "b"}
    assert set(tiers.active_clip_ids(5)) == {"a", "b", "c", "d", "e"}
    # Epochs beyond the number of tiers stay at the full set.
    assert set(tiers.active_clip_ids(10)) == {"a", "b", "c", "d", "e"}


def test_tier_counts():
    clip_to_tier = {"a": "T1", "b": "T1", "c": "T3"}
    tiers = CurriculumTiers(clip_to_tier=clip_to_tier)
    counts = tiers.tier_counts()
    assert counts["T1"] == 2
    assert counts["T3"] == 1
    assert counts["T2"] == 0
