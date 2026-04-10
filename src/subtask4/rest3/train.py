"""
train.py
--------
Fine-tunes JointSyllogismClassifier for Subtask 4 (multilingual multi-premise).

Same training procedure as Subtask 2:
  - Encoder + validity head from subtask2 checkpoint
  - Premise head randomly initialized
  - Differential LRs: encoder=1e-5, heads=1e-4
  - Multi-task loss: validity CE + premise BCE
  - Early stopping on combined_score
"""

import math
import os
import sys
import time
import random
from typing import Dict

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "subtask1"))
sys.path.insert(0, SCRIPT_DIR)

from src.subtask4.rest3.config import (
    NUM_EPOCHS, ENCODER_LR, HEAD_LR, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_GRAD_NORM, SEED, MODEL_SAVE_DIR, USE_FP16,
    GRADIENT_ACCUMULATION_STEPS, S2_MODEL_SAVE_DIR, S1_STEERING_VECTORS_PATH,
    STEERING_LAYERS, EARLY_STOPPING_PATIENCE,
)
from src.subtask4.rest3.model import load_joint_model, JointSyllogismClassifier
from src.subtask4.rest3.data_loader import build_dataloaders_subtask4


def set_seed(seed: int = SEED):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_val_metrics(model, val_loader, device):
    """Compute all subtask-4 validation metrics (same as subtask-2)."""
    model.eval()
    subgroups = {
        (1, 1): {"correct": 0, "total": 0},
        (1, 0): {"correct": 0, "total": 0},
        (0, 1): {"correct": 0, "total": 0},
        (0, 0): {"correct": 0, "total": 0},
    }
    total_precision = 0.0
    total_recall = 0.0
    premise_valid_count = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            premise_spans  = batch["premise_spans"].to(device)
            num_premises   = batch["num_premises"].to(device)
            labels         = batch["label"].to(device)
            premise_labels = batch["premise_labels"].to(device)
            plausibilities = batch["plausibility"]

            out        = model(input_ids, attention_mask, premise_spans, num_premises)
            val_preds  = out["logits"].argmax(dim=-1).cpu()
            prem_probs = torch.sigmoid(out["prem_logits"]).cpu()

            for b in range(labels.shape[0]):
                lbl   = labels[b].item()
                pred  = val_preds[b].item()
                plaus = plausibilities[b].item()

                key = (lbl, plaus)
                subgroups[key]["total"] += 1
                if pred == lbl:
                    subgroups[key]["correct"] += 1

                if lbl == 1:
                    n = num_premises[b].item()
                    gt_set = set()
                    for j in range(n):
                        if premise_labels[b, j].item() == 1:
                            gt_set.add(j)
                    probs_n = prem_probs[b, :n]
                    pred_set = set()
                    if n > 0:
                        top2_idx = probs_n.argsort(descending=True)[:2].tolist()
                        pred_set = set(top2_idx)
                    if gt_set:
                        tp = len(gt_set & pred_set)
                        fp = len(pred_set - gt_set)
                        fn = len(gt_set - pred_set)
                        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        total_precision += p
                        total_recall += r
                        premise_valid_count += 1

    def sg_acc(k):
        g = subgroups[k]
        return (g["correct"] / g["total"] * 100) if g["total"] > 0 else 0.0

    a_pv = sg_acc((1, 1))
    a_iv = sg_acc((1, 0))
    a_pi = sg_acc((0, 1))
    a_ii = sg_acc((0, 0))

    total_correct = sum(g["correct"] for g in subgroups.values())
    total_items   = sum(g["total"]   for g in subgroups.values())
    overall_acc   = (total_correct / total_items * 100) if total_items > 0 else 0.0

    ce_intra = (abs(a_pv - a_iv) + abs(a_pi - a_ii)) / 2.0
    ce_inter = (abs(a_pv - a_pi) + abs(a_iv - a_ii)) / 2.0
    tce = (ce_intra + ce_inter) / 2.0

    if premise_valid_count > 0:
        macro_prec = total_precision / premise_valid_count
        macro_rec  = total_recall / premise_valid_count
        premise_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                      if (macro_prec + macro_rec) > 0 else 0.0)
    else:
        macro_prec = macro_rec = premise_f1 = 0.0

    f1_pct = premise_f1 * 100
    overall_perf = (overall_acc + f1_pct) / 2.0
    log_penalty  = math.log(1 + tce)
    combined     = overall_perf / (1 + log_penalty)

    return {
        "accuracy": round(overall_acc, 4),
        "premise_precision": round(macro_prec * 100, 2),
        "premise_recall": round(macro_rec * 100, 2),
        "premise_f1": round(f1_pct, 4),
        "premise_n": premise_valid_count,
        "overall_perf": round(overall_perf, 4),
        "content_effect": round(tce, 4),
        "combined_score": round(combined, 4),
        "acc_plausible_valid": round(a_pv, 2),
        "acc_implausible_valid": round(a_iv, 2),
        "acc_plausible_invalid": round(a_pi, 2),
        "acc_implausible_invalid": round(a_ii, 2),
        "ce_intra": round(ce_intra, 4),
        "ce_inter": round(ce_inter, 4),
    }


def compute_class_weights(data_loader, device):
    counts = [0, 0]
    for batch in data_loader:
        for lbl in batch["label"].tolist():
            counts[lbl] += 1
    total = sum(counts)
    weights = [total / (2 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float).to(device)


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpu}")

    train_loader, val_loader, _, tokenizer = build_dataloaders_subtask4()

    print("\nLoading JointSyllogismClassifier from subtask2 checkpoint ...")
    model = load_joint_model(S2_MODEL_SAVE_DIR)

    # Apply subtask1 steering vectors
    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_alpha = steer_data.get("best_alpha", -2.0)
            model._steering_alpha = best_alpha
            print(f"Steering: alpha={best_alpha}")
        except Exception as e:
            print(f"[warn] Could not apply steering: {e}")

    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    class_weights = compute_class_weights(train_loader, device)
    base_model = model.module if hasattr(model, "module") else model
    base_model.set_class_weights(class_weights)

    encoder_params = [(n, p) for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad]
    head_params    = [(n, p) for n, p in model.named_parameters()
                      if "encoder" not in n and p.requires_grad]

    optimizer = AdamW([
        {"params": [p for _, p in encoder_params], "lr": ENCODER_LR},
        {"params": [p for _, p in head_params],    "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    total_steps  = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=USE_FP16)

    print(f"\n{'='*70}")
    print(f"Starting training: {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"Encoder LR: {ENCODER_LR}  |  Head LR: {HEAD_LR}")
    print(f"{'='*70}\n")

    best_combined = -1.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = epoch_val_l = epoch_prem_l = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            premise_spans  = batch["premise_spans"].to(device)
            num_premises   = batch["num_premises"].to(device)
            labels         = batch["label"].to(device)
            premise_labels = batch["premise_labels"].to(device)

            with autocast(enabled=USE_FP16):
                out  = model(input_ids, attention_mask, premise_spans,
                            num_premises, labels, premise_labels)
                loss      = out["loss"].mean() / GRADIENT_ACCUMULATION_STEPS
                val_loss  = out.get("validity_loss", torch.tensor(0.0)).mean()
                prem_loss = out.get("premise_loss",  torch.tensor(0.0)).mean()

            scaler.scale(loss).backward()
            epoch_loss   += loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_val_l  += val_loss.item()
            epoch_prem_l += prem_loss.item()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                avg = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
                      f"Step {batch_idx+1:>4}/{len(train_loader)}  "
                      f"Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}  "
                      f"AvgLoss: {avg:.4f}")
                sys.stdout.flush()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)

        metrics = compute_val_metrics(
            model.module if hasattr(model, "module") else model,
            val_loader, device)

        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch} complete in {elapsed:.1f}s")
        print(f"  Train Loss      : {avg_loss:.4f}")
        print(f"  Val Accuracy    : {metrics['accuracy']:.2f}%")
        print(f"  Val Premise F1  : {metrics['premise_f1']:.2f}%")
        print(f"  Val TCE         : {metrics['content_effect']:.4f}")
        print(f"  Val Combined    : {metrics['combined_score']:.4f}")
        print(f"  [PV={metrics['acc_plausible_valid']:.1f}  "
              f"IV={metrics['acc_implausible_valid']:.1f}  "
              f"PI={metrics['acc_plausible_invalid']:.1f}  "
              f"II={metrics['acc_implausible_invalid']:.1f}]")
        print(f"{'─'*70}\n")
        sys.stdout.flush()

        if metrics["combined_score"] > best_combined:
            best_combined = metrics["combined_score"]
            patience_counter = 0
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            base = model.module if hasattr(model, "module") else model
            torch.save(base.state_dict(),
                      os.path.join(MODEL_SAVE_DIR, "model_weights.pt"))
            print(f"  >> Saved best checkpoint (combined={best_combined:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best combined: {best_combined:.4f}")


if __name__ == "__main__":
    train()
