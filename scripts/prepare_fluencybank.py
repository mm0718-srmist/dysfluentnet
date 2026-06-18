"""
Builds the FluencyBank manifest CSV consumed by `FluencyBankDataset` from
TalkBank's word-level disfluency annotations (Ratner & MacWhinney, 2018,
https://fluency.talkbank.org).

FluencyBank's CHAT-format (.cha) transcripts encode disfluencies with
inline codes (e.g. `[/]` for repetition, `&-um` for filled pauses, `&+w`
for a fragment/block). This script provides a minimal, adjustable mapping
from common CHAT disfluency codes to this project's five dysfluency tokens
-- inspect and extend `CHAT_CODE_TO_TOKEN` for your specific transcript
conventions, since CHAT annotation style varies across corpora/coders.

Usage
-----
    python scripts/prepare_fluencybank.py \
        --cha_dir /path/to/fluencybank/transcripts \
        --audio_dir /path/to/fluencybank/audio \
        --out_csv data/manifests/fluencybank_test.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CHAT_CODE_TO_TOKEN = {
    r"&-\w+": "<INT>",      # filled pauses / interjections, e.g. &-um, &-uh
    r"&\+\w+": "<BLK>",     # fragments / blocks, e.g. &+w
    r"\[/\]": "<WR>",       # word/phrase repetition marker
    r"\[//\]": "<SR>",      # retraced/sound repetition marker
    r"&~\w+": "<PRO>",      # paper convention placeholder for prolongations;
                             # adjust to your transcript set's actual coding
}


def cha_line_to_reference_plus(line: str) -> str:
    """Converts a single CHAT utterance line into a whitespace-tokenised
    dysfluency-inclusive reference string. This is intentionally simple --
    treat it as a starting point and verify against a sample of manually
    checked transcripts before trusting it for evaluation."""
    text = line
    for pattern, token in CHAT_CODE_TO_TOKEN.items():
        text = re.sub(pattern, f" {token} ", text)
    # Strip remaining CHAT annotation punctuation/markup not mapped above.
    text = re.sub(r"[\[\]<>@:]", " ", text)
    tokens = text.split()
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cha_dir", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="data/manifests/fluencybank_test.csv")
    args = parser.parse_args()

    cha_dir = Path(args.cha_dir)
    audio_dir = Path(args.audio_dir)

    rows = []
    for cha_path in sorted(cha_dir.glob("*.cha")):
        speaker_id = cha_path.stem
        audio_path = audio_dir / f"{speaker_id}.wav"
        if not audio_path.exists():
            continue

        with open(cha_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.startswith("*")]

        for i, line in enumerate(lines):
            utterance_id = f"{speaker_id}_{i:04d}"
            reference_plus = cha_line_to_reference_plus(line)
            if not reference_plus:
                continue
            rows.append(
                {
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                    "audio_path": str(audio_path),
                    "reference_plus": reference_plus,
                }
            )

    manifest = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {len(manifest)} utterances to {out_path}")


if __name__ == "__main__":
    main()
