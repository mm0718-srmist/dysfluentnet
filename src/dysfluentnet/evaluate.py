"""
Evaluation entry point: reproduces the metrics reported in the paper's
results tables -- per-class and macro F1 for detection (SEP-28k test split)
and WER / DI-WER for transcription (FluencyBank test split).

Usage
-----
    python -m dysfluentnet.evaluate --config configs/dysfluentnet_base.yaml \
        --checkpoint runs/dysfluentnet_base/seed0/best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader

from dysfluentnet.data import (
    SEP28kDataset,
    collate_sep28k,
    SEP28K_CLASSES,
    FluencyBankDataset,
    collate_fluencybank,
)
from dysfluentnet.metrics import corpus_di_wer
from dysfluentnet.models import DysfluentNet
from dysfluentnet.utils import load_checkpoint


@torch.no_grad()
def evaluate_detection(model, loader, device) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(waveform, attention_mask)
        preds = (out.p_hat.cpu().numpy() > 0.5).astype(int)
        all_preds.append(preds)
        all_targets.append(batch["labels"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
    report = classification_report(
        targets, preds, target_names=SEP28K_CLASSES, zero_division=0, output_dict=True
    )

    return {
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(SEP28K_CLASSES, per_class_f1.tolist())),
        "report": report,
    }


@torch.no_grad()
def evaluate_transcription(model, loader, device, id_to_token: dict[int, str], blank_id: int = 0) -> dict:
    """Greedy-decodes the SA-CTC decoder and computes corpus-level WER and
    DI-WER against the dysfluency-inclusive reference (Eq. 8 / Sec. 5.2)."""
    model.eval()
    hypotheses, references = [], []

    for batch in loader:
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(waveform, attention_mask)

        pred_ids = out.ctc_logits.argmax(dim=-1).cpu().numpy()  # [B, T]
        for ids, ref in zip(pred_ids, batch["reference_plus"]):
            collapsed = []
            prev = None
            for tok_id in ids:
                if tok_id != blank_id and tok_id != prev:
                    collapsed.append(id_to_token.get(int(tok_id), "<unk>"))
                prev = tok_id
            hypotheses.append(collapsed)
            references.append(ref)

    return corpus_di_wer(hypotheses, references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dysfluentnet_base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--skip_transcription", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    model = DysfluentNet(
        wavlm_model_name=cfg["model"]["wavlm_model_name"],
        num_layers=cfg["model"]["num_layers"],
        encoder_dim=cfg["model"]["encoder_dim"],
        detection_embed_dim=cfg["model"]["detection"]["embed_dim"],
        num_classes=cfg["model"]["detection"]["num_classes"],
        lstm_hidden=cfg["model"]["decoder"]["lstm_hidden"],
        num_lstm_layers=cfg["model"]["decoder"]["num_lstm_layers"],
        base_vocab_size=cfg["model"]["decoder"]["base_vocab_size"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)

    test_ds = SEP28kDataset(cfg["data"]["sep28k_test_manifest"], sample_rate=cfg["data"]["sample_rate"])
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_sep28k
    )
    det_results = evaluate_detection(model, test_loader, device)
    print("=== Detection (SEP-28k test) ===")
    print(f"Macro F1: {det_results['macro_f1']:.4f}")
    for cls, f1 in det_results["per_class_f1"].items():
        print(f"  {cls:>4s}: {f1:.4f}")

    if not args.skip_transcription:
        fb_ds = FluencyBankDataset(cfg["data"]["fluencybank_test_manifest"], sample_rate=cfg["data"]["sample_rate"])
        fb_loader = DataLoader(
            fb_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fluencybank
        )
        # Build a minimal id_to_token map for the 5 dysfluency tokens; the
        # base vocabulary's id_to_token mapping comes from your tokenizer.
        base_vocab_size = cfg["model"]["decoder"]["base_vocab_size"]
        dys_tokens = cfg["model"]["decoder"]["dysfluency_tokens"]
        id_to_token = {base_vocab_size + i: tok for i, tok in enumerate(dys_tokens)}

        trans_results = evaluate_transcription(model, fb_loader, device, id_to_token)
        print("\n=== Transcription (FluencyBank test) ===")
        print(f"WER:    {trans_results['wer']:.4f}")
        print(f"DI-WER: {trans_results['di_wer']:.4f}")


if __name__ == "__main__":
    main()
