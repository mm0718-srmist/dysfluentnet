"""
Builds the SEP-28k manifest CSVs consumed by `SEP28kDataset` from the raw
SEP-28k label files (Lea et al., 2021,
https://github.com/apple/ml-stuttering-events-dataset).

This script expects you have already cloned/downloaded SEP-28k's labels
and clipped, 3-second audio files locally, since the raw podcast audio is
not redistributed by the dataset authors and must be fetched per their
instructions.

Usage
-----
    python scripts/prepare_sep28k.py \
        --labels_csv /path/to/SEP-28k_labels.csv \
        --clips_dir /path/to/clips \
        --annotator_csv /path/to/SEP-28k_episodes.csv \
        --out_dir data/manifests

Expected raw label columns (adjust `RAW_LABEL_COLUMNS` below if your copy
of SEP-28k uses different header names):
    EpId, ClipId, Show, ... , Block, Prolongation, SoundRep, WordRep,
    Interjection, NoStutteredWords, ...
plus three per-annotator raw label sets if you are recomputing Fleiss'
kappa yourself; if your copy of SEP-28k only ships majority-vote labels,
set --skip_kappa to fall back to a fixed default tier (T3) for all clips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dysfluentnet.data.curriculum import fleiss_kappa, assign_tier

RAW_LABEL_COLUMNS = {
    "blk": "Block",
    "pro": "Prolongation",
    "sr": "SoundRep",
    "wr": "WordRep",
    "int": "Interjection",
    "flu": "NoStutteredWords",
}


def compute_kappa_from_annotator_columns(row: pd.Series, annotator_prefixes: list[str]) -> float:
    """If your SEP-28k copy ships per-annotator raw counts (as in the
    original release, columns like `Block` already aggregate 3 annotators'
    votes as 0-3 counts), Fleiss' kappa can be computed directly from the
    6-category count vector per clip."""
    counts = np.array([row[RAW_LABEL_COLUMNS[c]] for c in RAW_LABEL_COLUMNS], dtype=float)
    return fleiss_kappa(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/manifests")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_kappa", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    clips_dir = Path(args.clips_dir)

    rows = []
    for _, row in df.iterrows():
        clip_id = f"{row['EpId']}_{row['ClipId']}"
        audio_path = clips_dir / f"{clip_id}.wav"
        if not audio_path.exists():
            continue

        labels = {c: int(row[RAW_LABEL_COLUMNS[c]] > 0) for c in RAW_LABEL_COLUMNS}
        kappa = (
            compute_kappa_from_annotator_columns(row, [])
            if not args.skip_kappa
            else fleiss_kappa(np.array([3, 0, 0, 0, 0, 0]))  # forces T1 fallback
        )

        rows.append(
            {
                "clip_id": clip_id,
                "audio_path": str(audio_path),
                **labels,
                "fleiss_kappa": kappa,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest["tier"] = manifest["fleiss_kappa"].apply(assign_tier)
    print("Tier distribution:\n", manifest["tier"].value_counts())

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(manifest))
    n_val = int(len(manifest) * args.val_frac)
    n_test = int(len(manifest) * args.test_frac)

    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.iloc[train_idx].to_csv(out_dir / "sep28k_train.csv", index=False)
    manifest.iloc[val_idx].to_csv(out_dir / "sep28k_val.csv", index=False)
    manifest.iloc[test_idx].to_csv(out_dir / "sep28k_test.csv", index=False)

    print(f"Wrote {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test rows to {out_dir}")


if __name__ == "__main__":
    main()
