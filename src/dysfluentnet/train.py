"""
Training entry point for DysfluentNet.

Usage
-----
    python -m dysfluentnet.train --config configs/dysfluentnet_base.yaml --seed 0

This script wires together the curriculum-aware SEP-28k dataloader, the
DysfluentNet model (frozen WavLM-Large encoder + detection head + SA-CTC
decoder), and the joint loss (Eq. 7), following the implementation details
in Sec. 4.4 of the paper: AdamW, cosine LR schedule with 10% warmup,
mixed-precision (fp16) training, gradient accumulation, and early stopping
on validation macro-F1.

NOTE: SEP-28k / FluencyBank audio are not redistributed with this repo.
Build the manifests referenced in `configs/dysfluentnet_base.yaml` with
`scripts/prepare_sep28k.py` and `scripts/prepare_fluencybank.py` first.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dysfluentnet.data import SEP28kDataset, collate_sep28k
from dysfluentnet.losses import DysfluentNetLoss
from dysfluentnet.models import DysfluentNet
from dysfluentnet.utils import set_seed, save_checkpoint, spec_augment, count_trainable_parameters


def build_cosine_schedule_with_warmup(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, cfg, train: bool = True):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []
    accum_steps = cfg["train"]["grad_accum_steps"]

    for step, batch in enumerate(loader):
        waveform = batch["waveform"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=(cfg["train"]["precision"] == "fp16")):
            out = model(waveform, attention_mask)
            # NOTE: ctc_targets / input_lengths / target_lengths must come
            # from your tokenizer pipeline; placeholder zeros are used here
            # so the script runs end-to-end on dummy data for smoke testing.
            dummy_ctc_targets = torch.zeros(labels.shape[0], dtype=torch.long, device=device)
            dummy_input_lengths = torch.full((labels.shape[0],), out.ctc_logits.shape[1], device=device)
            dummy_target_lengths = torch.ones(labels.shape[0], dtype=torch.long, device=device)

            losses = loss_fn(
                detection_logits=out.detection_logits,
                detection_targets=labels,
                ctc_logits=out.ctc_logits,
                pooling_weights=out.pooling_weights,
                p_hat=out.p_hat,
                ctc_targets=dummy_ctc_targets,
                input_lengths=dummy_input_lengths,
                target_lengths=dummy_target_lengths,
                frame_mask=None,
            )
            loss = losses["loss"] / accum_steps

        if train:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += losses["loss"].item()
        n_batches += 1

        preds = (out.p_hat.detach().cpu().numpy() > 0.5).astype(int)
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    import numpy as np

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return {"loss": total_loss / max(n_batches, 1), "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dysfluentnet_base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    train_ds = SEP28kDataset(cfg["data"]["sep28k_train_manifest"], sample_rate=cfg["data"]["sample_rate"])
    val_ds = SEP28kDataset(cfg["data"]["sep28k_val_manifest"], sample_rate=cfg["data"]["sample_rate"])
    curriculum = train_ds.build_curriculum_tiers() if cfg["data"]["curriculum"]["enabled"] else None

    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_sep28k
    )

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

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    dys_token_ids = list(range(cfg["model"]["decoder"]["base_vocab_size"],
                                cfg["model"]["decoder"]["base_vocab_size"] + 5))
    loss_fn = DysfluentNetLoss(
        dysfluency_token_ids=dys_token_ids,
        focal_gamma=cfg["loss"]["focal_gamma"],
        lambda_align=cfg["loss"]["sa_ctc"]["lambda_align"],
        beta=cfg["loss"]["beta"],
        window=cfg["loss"]["sa_ctc"]["window_frames"],
    )

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["optim"]["peak_lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    steps_per_epoch = math.ceil(len(train_ds) / cfg["train"]["batch_size"])
    total_steps = steps_per_epoch * cfg["train"]["max_epochs"] // cfg["train"]["grad_accum_steps"]
    scheduler = build_cosine_schedule_with_warmup(optimizer, total_steps, cfg["optim"]["warmup_ratio"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["train"]["precision"] == "fp16"))

    best_metric, patience_counter = -1.0, 0
    out_dir = Path(cfg["experiment"]["output_dir"]) / f"seed{args.seed}"

    for epoch in range(1, cfg["train"]["max_epochs"] + 1):
        if curriculum is not None:
            active_ids = curriculum.active_clip_ids(epoch)
            epoch_train_ds = SEP28kDataset(
                cfg["data"]["sep28k_train_manifest"],
                sample_rate=cfg["data"]["sample_rate"],
                clip_ids=active_ids,
            )
        else:
            epoch_train_ds = train_ds
        train_loader = DataLoader(
            epoch_train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_sep28k
        )

        train_stats = run_epoch(model, train_loader, loss_fn, optimizer, scheduler, scaler, device, cfg, train=True)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, loss_fn, optimizer, scheduler, scaler, device, cfg, train=False)

        print(
            f"epoch {epoch:3d} | train_loss {train_stats['loss']:.4f} | "
            f"val_loss {val_stats['loss']:.4f} | val_macro_f1 {val_stats['macro_f1']:.4f}"
        )

        if val_stats["macro_f1"] > best_metric:
            best_metric = val_stats["macro_f1"]
            patience_counter = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_metric)
        else:
            patience_counter += 1

        if patience_counter >= cfg["train"]["early_stopping"]["patience"]:
            print(f"Early stopping at epoch {epoch} (best val_macro_f1={best_metric:.4f})")
            break


if __name__ == "__main__":
    main()
