# DysfluentNet

**Joint Stuttering Event Detection and Dysfluency-Aware Transcription via Hierarchical Self-Supervised Learning**

Mohankumar Muthu, Sasikala E, Girirajan S — SRM Institute of Science and Technology (SRMIST), Chennai, India
Accepted at **Interspeech 2026**.

Stuttering affects roughly 70 million people worldwide, yet ASR systems perform poorly on stuttered speech, and dedicated detection systems rarely integrate transcription. DysfluentNet is a hierarchical multi-task framework that jointly performs fine-grained stuttering event detection and dysfluency-aware transcription from raw waveforms, coupling a frozen WavLM-Large encoder with a lightweight dysfluency-conditioned decoder trained under a novel **Stutter-Aware CTC (SA-CTC)** objective. A curriculum learning schedule, driven by an annotator-agreement-based difficulty partitioning of SEP-28k, progressively exposes the model to harder disfluency patterns during training.

## Results

Macro F1 (%) for 6-class stuttering event detection, and WER / DI-WER (%, lower is better) for transcription. DysfluentNet rows report mean ± std over 3 seeds.

| System | SEP-28k F1 (macro) | FluencyBank F1 |
|---|---|---|
| StutterNet | 43.2 | 61.8 |
| MC-SN | 49.7 | 66.3 |
| wav2vec 2.0 + SVM | 57.1 | 72.6 |
| Whister | 63.4 | 75.9 |
| LLM-Dys | 65.6 | 77.2 |
| Pipeline baseline (ours) | 68.9 ± 0.6 | 79.8 ± 0.5 |
| **DysfluentNet (ours)** | **72.4 ± 0.4** | **81.5 ± 0.4** |

| System | WER | DI-WER |
|---|---|---|
| Whisper large-v3 | 19.7 | 29.4† |
| Dysfluent WFST | 17.1 | 24.8† |
| SSDM 2.0 | 14.3 | 22.4† |
| Pipeline baseline (ours) | 14.6 ± 0.2 | 22.1 ± 0.3 |
| **DysfluentNet (ours)** | **13.8 ± 0.2** | **18.3 ± 0.3** |

†DI-WER computed by the authors using released models on the same FluencyBank partition (315 utterances, 32 speakers). Comparison against the matched pipeline baseline confirms that joint coupling (cross-attention gating + SA-CTC) is the primary driver of the gains over a shared-encoder-only approach.

## Architecture

A shared, frozen WavLM-Large encoder produces a layer-attention-weighted representation $\hat H = \sum_l \alpha_l H^{(l)}$. This feeds:

1. **Detection head** — attentive statistics pooling over $\hat H$ into an utterance embedding, classified into 6 dysfluency classes (block, prolongation, sound repetition, word repetition, interjection, fluent) via multi-label focal loss.
2. **SA-CTC decoder** — a 2-layer BiLSTM gated by the detection head's confidence through a cross-attention gate, projecting to an extended vocabulary that includes 5 dysfluency tokens, trained with a stuttering-aware CTC objective that adds an alignment-consistency term between the decoder's emissions and the detection head's attention.

Training uses a 5-stage curriculum over annotator-agreement (Fleiss' κ) tiers of SEP-28k:

| Tier | κ range | N (clips) | Cumulative N |
|---|---|---|---|
| T1 | κ ≥ 0.80 | 4,213 | 4,213 |
| T2 | 0.60 ≤ κ < 0.80 | 6,881 | 11,094 |
| T3 | 0.50 ≤ κ < 0.60 | 4,426 | 15,520 |
| T4 | 0.40 ≤ κ < 0.50 | 3,894 | 19,414 |
| T5 | κ < 0.40 | 3,127 | 22,541 |

See the [paper](paper/dysfluentnet_camera_ready.tex) for the full derivation, ablations, and discussion.

## Repository structure

```
dysfluentnet/
├── configs/
│   └── dysfluentnet_base.yaml      # hyperparameters reported in the paper
├── paper/
│   └── dysfluentnet_camera_ready.tex
├── scripts/
│   ├── prepare_sep28k.py           # build the SEP-28k manifest + curriculum tiers
│   ├── prepare_fluencybank.py      # build the FluencyBank manifest
│   └── compute_curriculum_tiers.py # sanity-check tier distribution
├── src/dysfluentnet/
│   ├── models/                     # encoder, detection head, SA-CTC decoder
│   ├── losses/                     # multi-label focal loss, SA-CTC loss
│   ├── data/                       # dataset loaders, curriculum sampler
│   ├── metrics/                    # WER / DI-WER
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── tests/                          # unit tests (no GPU/network required)
```

## Installation

```bash
git clone https://github.com/mm0718-srmist/dysfluentnet.git
cd dysfluentnet
pip install -e .
# or: pip install -r requirements.txt
```

Run the test suite (no GPU or model download required):

```bash
pytest tests/
```

## Data

SEP-28k and FluencyBank audio are **not redistributed** in this repository — both must be obtained from their original sources:

- SEP-28k: https://github.com/apple/ml-stuttering-events-dataset
- FluencyBank: https://fluency.talkbank.org

Once downloaded, build the manifests this codebase expects:

```bash
python scripts/prepare_sep28k.py --labels_csv <path> --clips_dir <path> --out_dir data/manifests
python scripts/prepare_fluencybank.py --cha_dir <path> --audio_dir <path> --out_csv data/manifests/fluencybank_test.csv
python scripts/compute_curriculum_tiers.py --manifest data/manifests/sep28k_train.csv
```

## Training & evaluation

```bash
python -m dysfluentnet.train --config configs/dysfluentnet_base.yaml --seed 0
python -m dysfluentnet.evaluate --config configs/dysfluentnet_base.yaml --checkpoint runs/dysfluentnet_base/seed0/best.pt
```

`configs/dysfluentnet_base.yaml` mirrors the paper's reported setup: AdamW, peak LR 3e-4 with 10% cosine warmup, batch size 16 with gradient accumulation of 3 (effective batch 48), fp16 mixed precision, SpecAugment (F=27, T=100, 2 masks each), SA-CTC alignment window of ±15 frames, λ=0.2, β=0.5, and early stopping on validation macro F1 with patience 10.

**Note on the decoder vocabulary and tokenizer:** `train.py`/`evaluate.py` assume a grapheme/BPE tokenizer producing the base CTC vocabulary; wire your tokenizer's `ctc_targets`/`input_lengths`/`target_lengths` into the training loop in place of the placeholder values currently used for the end-to-end smoke run.

## Citation

```bibtex
@inproceedings{muthu2026dysfluentnet,
  title     = {DysfluentNet: Joint Stuttering Event Detection and Dysfluency-Aware Transcription via Hierarchical Self-Supervised Learning},
  author    = {Muthu, Mohankumar and E, Sasikala and S, Girirajan},
  booktitle = {Interspeech 2026},
  year      = {2026}
}
```

## License

This code is released under the [MIT License](LICENSE).

## Contact

Mohankumar Muthu — mm0718@srmist.edu.in
Sasikala E — sasikale@srmist.edu.in (corresponding author)
