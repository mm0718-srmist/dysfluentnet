"""
Computes the curriculum tier distribution (Sec. 4.2 table) from a prepared
SEP-28k manifest CSV, e.g. to sanity-check tier counts against the paper's
reported 4,213 / 6,881 / 4,426 / 3,894 / 3,127 split before training.

Usage
-----
    python scripts/compute_curriculum_tiers.py --manifest data/manifests/sep28k_train.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from dysfluentnet.data.curriculum import assign_tier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df["tier"] = df["fleiss_kappa"].apply(assign_tier)

    counts = df["tier"].value_counts().reindex(["T1", "T2", "T3", "T4", "T5"], fill_value=0)
    print("Tier  | N      | kappa range")
    print("------+--------+-------------------")
    ranges = {
        "T1": "kappa >= 0.80",
        "T2": "0.60 <= kappa < 0.80",
        "T3": "0.50 <= kappa < 0.60",
        "T4": "0.40 <= kappa < 0.50",
        "T5": "kappa < 0.40",
    }
    for tier, n in counts.items():
        print(f"{tier:5s} | {n:6d} | {ranges[tier]}")
    print(f"\nTotal: {counts.sum()}")


if __name__ == "__main__":
    main()
