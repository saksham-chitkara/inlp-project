"""
train.py
--------
Fine-tunes JointSyllogismClassifier (subtask2) on the generated multi-premise
training dataset.

Initialization strategy:
  - Encoder + validity head   → loaded from subtask1 checkpoint (already trained)
  - Premise head              → randomly initialized (new task)
  - Different learning rates: encoder=ENCODER_LR, heads=HEAD_LR

Training objective:
  loss = 0.5 × CE(validity) + 0.5 × masked_BCE(per_premise_relevance)
  BCE uses pos_weight ≈ 5 to compensate for ~16% positive premise labels.

Evaluation metrics (per epoch):
  - Validity accuracy + subgroup accuracies (PV/IV/PI/II)
  - Premise F1, precision, recall (macro-averaged, valid items only)
  - Content effect: TCE = (CE_intra + CE_inter) / 2
  - Overall perf = (accuracy + F1_premises) / 2
  - Combined score = overall_perf / (1 + ln(1 + TCE))  ← checkpoint metric

Logging:
  - Detailed per-step loss (printed to stdout every 50 steps)
  - Per-epoch validation with all metrics (same format as subtask 1/3)
  - Best checkpoint saved by combined_score
  - Early stopping on combined_score (patience = 3)

Usage:
  cd /ssd_scratch/shubhamcvit/inlp/project
  python3 src/subtask2/train.py
"""

import math
import os
import sys
import json
import time
import random
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "subtask1"))
sys.path.insert(0, SCRIPT_DIR)  # subtask2 dir must come first for config import

from src.subtask2.rest3.config import (
    NUM_EPOCHS, ENCODER_LR, HEAD_LR, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_GRAD_NORM, SEED, MODEL_SAVE_DIR, USE_FP16,
    GRADIENT_ACCUMULATION_STEPS, S1_MODEL_SAVE_DIR, S1_STEERING_VECTORS_PATH,
    STEERING_LAYERS, STEERING_KNN,
)
from src.subtask2.rest3.model import load_joint_model, JointSyllogismClassifier
from src.subtask2.rest3.data_loader import build_dataloaders_subtask2

EARLY_STOPPING_PATIENCE = 7


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Validation Metrics (full, matching subtask1/3 + subtask2 specifics) ─────

def compute_val_metrics(
    model: JointSyllogismClassifier,
    val_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute all subtask-2 validation metrics:
      - Validity accuracy + 4 subgroup accuracies (PV/IV/PI/II)
      - Premise precision/recall/F1 (macro-averaged over valid items)
      - Content effect (TCE) with intra/inter components
      - Overall performance = (accuracy + F1_premises) / 2
      - Combined score = overall_perf / (1 + ln(1 + TCE))
    """
    model.eval()

    # Subgroup accumulators: (validity, plausibility) → {correct, total}
    subgroups = {
        (1, 1): {"correct": 0, "total": 0},  # valid + plausible
        (1, 0): {"correct": 0, "total": 0},  # valid + implausible
        (0, 1): {"correct": 0, "total": 0},  # invalid + plausible
        (0, 0): {"correct": 0, "total": 0},  # invalid + implausible
    }

    # Premise F1: macro-averaged per valid item
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
            plausibilities = batch["plausibility"]  # keep on CPU

            out        = model(input_ids, attention_mask, premise_spans, num_premises)
            val_preds  = out["logits"].argmax(dim=-1).cpu()            # (B,)
            prem_probs = torch.sigmoid(out["prem_logits"]).cpu()       # (B, MAX_P)

            for b in range(labels.shape[0]):
                lbl  = labels[b].item()
                pred = val_preds[b].item()
                plaus = plausibilities[b].item()

                # Subgroup accuracy
                key = (lbl, plaus)
                subgroups[key]["total"] += 1
                if pred == lbl:
                    subgroups[key]["correct"] += 1

                # Premise F1 (only ground-truth valid items)
                if lbl == 1:
                    n = num_premises[b].item()
                    gt_set = set()
                    pred_set = set()
                    for j in range(n):
                        gt_val = premise_labels[b, j].item()
                        if gt_val < 0:
                            continue
                        if gt_val == 1:
                            gt_set.add(j)
                        # Top-2 by probability for prediction
                    # Use top-2 by prob (same as inference)
                    probs_n = prem_probs[b, :n]
                    if n > 0:
                        top2_idx = probs_n.argsort(descending=True)[:2].tolist()
                        pred_set = set(top2_idx)

                    if gt_set:  # only count if there are actual relevant premises
                        tp = len(gt_set & pred_set)
                        fp = len(pred_set - gt_set)
                        fn = len(gt_set - pred_set)
                        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        total_precision += p
                        total_recall += r
                        premise_valid_count += 1

    # ── Compute accuracy ──
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

    # ── Content effect ──
    intra_valid   = abs(a_pv - a_iv)
    intra_invalid = abs(a_pi - a_ii)
    ce_intra      = (intra_valid + intra_invalid) / 2.0

    inter_plaus   = abs(a_pv - a_pi)
    inter_implaus = abs(a_iv - a_ii)
    ce_inter      = (inter_plaus + inter_implaus) / 2.0

    tce = (ce_intra + ce_inter) / 2.0

    # ── Premise F1 (macro) ──
    if premise_valid_count > 0:
        macro_prec = total_precision / premise_valid_count
        macro_rec  = total_recall / premise_valid_count
        premise_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)
                      if (macro_prec + macro_rec) > 0 else 0.0)
    else:
        macro_prec = macro_rec = premise_f1 = 0.0

    # ── Combined score (official subtask 2 ranking metric) ──
    # F1 as percentage for matching official scale
    f1_pct = premise_f1 * 100
    overall_perf = (overall_acc + f1_pct) / 2.0
    log_penalty  = math.log(1 + tce)
    combined     = overall_perf / (1 + log_penalty)

    return {
        "accuracy":               round(overall_acc, 4),
        "premise_precision":      round(macro_prec * 100, 2),
        "premise_recall":         round(macro_rec * 100, 2),
        "premise_f1":             round(f1_pct, 4),
        "premise_n":              premise_valid_count,
        "overall_perf":           round(overall_perf, 4),
        "content_effect":         round(tce, 4),
        "combined_score":         round(combined, 4),
        "acc_plausible_valid":    round(a_pv, 2),
        "acc_implausible_valid":  round(a_iv, 2),
        "acc_plausible_invalid":  round(a_pi, 2),
        "acc_implausible_invalid":round(a_ii, 2),
        "ce_intra":               round(ce_intra, 4),
        "ce_inter":               round(ce_inter, 4),
    }


# ─── Class weights helper (for validity head, same as subtask1) ─────────────

def compute_class_weights(data_loader, device: torch.device) -> torch.Tensor:
    counts = [0, 0]
    for batch in data_loader:
        for lbl in batch["label"].tolist():
            counts[lbl] += 1
    total = sum(counts)
    weights = [total / (2 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float).to(device)


# ─── Training loop ────────────────────────────────────────────────────────────

def train():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpu}")
    if torch.cuda.is_available():
        for i in range(n_gpu):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")

    # ── Build dataloaders ──
    train_loader, val_loader, _, tokenizer = build_dataloaders_subtask2()

    # ── Load model ──
    print("\nLoading JointSyllogismClassifier from subtask1 checkpoint ...")
    model = load_joint_model(S1_MODEL_SAVE_DIR)

    # ── Apply subtask1 steering vectors ──
    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_layer = steer_data.get("best_layer", STEERING_LAYERS[-1])
            best_alpha = steer_data.get("best_alpha", -2.0)
            sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "subtask1"))
            from activation_steering import ActivationSteerer
            steerer = ActivationSteerer(model, device)
            model._steering_alpha = best_alpha
            print(f"Steering: layer={best_layer}, alpha={best_alpha}")
        except Exception as e:
            print(f"[warn] Could not apply steering: {e}")

    model = model.to(device)
    if n_gpu > 1:
        print(f"[Train] Wrapping model with DataParallel ({n_gpu} GPUs)")
        model = torch.nn.DataParallel(model)

    # Class weights for validity head
    class_weights = compute_class_weights(train_loader, device)
    base_model = model.module if hasattr(model, "module") else model
    base_model.set_class_weights(class_weights)

    # ── Optimizer with differential LRs ──
    encoder_params = [
        (n, p) for n, p in model.named_parameters()
        if "encoder" in n and p.requires_grad
    ]
    head_params = [
        (n, p) for n, p in model.named_parameters()
        if "encoder" not in n and p.requires_grad
    ]

    optimizer = AdamW([
        {"params": [p for _, p in encoder_params], "lr": ENCODER_LR},
        {"params": [p for _, p in head_params],    "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    total_steps   = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=USE_FP16)

    print(f"\n{'='*70}")
    print(f"Starting training: {NUM_EPOCHS} epochs, {len(train_loader)} batches/epoch")
    print(f"Warmup steps: {warmup_steps} / {total_steps} total")
    print(f"Device: {device}  |  FP16: {USE_FP16}  |  Grad Accum: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Encoder LR: {ENCODER_LR}  |  Head LR: {HEAD_LR}")
    if n_gpu > 1:
        print(f"Multi-GPU: {n_gpu} GPUs via DataParallel")
    print(f"{'='*70}\n")

    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_premise_f1": [],
        "val_content_effect": [],
        "val_combined": [],
    }
    best_combined = -1.0
    best_metrics  = {}
    patience_counter = 0
    step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss   = 0.0
        epoch_val_l  = 0.0
        epoch_prem_l = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            premise_spans  = batch["premise_spans"].to(device)
            num_premises   = batch["num_premises"].to(device)
            labels         = batch["label"].to(device)
            premise_labels = batch["premise_labels"].to(device)

            with autocast(enabled=USE_FP16):
                out  = model(
                    input_ids, attention_mask,
                    premise_spans, num_premises,
                    labels, premise_labels,
                )
                # DataParallel gathers per-GPU losses into a vector; reduce to scalar
                loss      = out["loss"].mean() / GRADIENT_ACCUMULATION_STEPS
                val_loss  = out.get("validity_loss", torch.tensor(0.0)).mean()
                prem_loss = out.get("premise_loss",  torch.tensor(0.0)).mean()

            scaler.scale(loss).backward()
            epoch_loss   += loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_val_l  += val_loss.item() if val_loss is not None else 0
            epoch_prem_l += prem_loss.item() if prem_loss is not None else 0

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
                      f"Step {batch_idx+1:>4}/{len(train_loader)}  "
                      f"Loss: {loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}  "
                      f"AvgLoss: {avg_loss:.4f}  "
                      f"(val={val_loss.item():.4f} prem={prem_loss.item():.4f})  "
                      f"LR: {lr:.2e}")
                sys.stdout.flush()

        # ── Validation ──
        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        avg_val  = epoch_val_l / len(train_loader)
        avg_prem = epoch_prem_l / len(train_loader)

        metrics = compute_val_metrics(
            model.module if hasattr(model, "module") else model,
            val_loader, device
        )

        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(metrics["accuracy"])
        history["val_premise_f1"].append(metrics["premise_f1"])
        history["val_content_effect"].append(metrics["content_effect"])
        history["val_combined"].append(metrics["combined_score"])

        # Print epoch report (matching subtask 1/3 format)
        print(f"\n{'─'*70}")
        print(f"  Epoch {epoch} complete in {elapsed:.1f}s")
        print(f"  Train Loss      : {avg_loss:.4f}  "
              f"(validity={avg_val:.4f}  premise={avg_prem:.4f})")
        print(f"  Val Accuracy    : {metrics['accuracy']:.2f}%")
        print(f"  Val Premise F1  : {metrics['premise_f1']:.2f}%  "
              f"(P={metrics['premise_precision']:.1f}%  "
              f"R={metrics['premise_recall']:.1f}%  "
              f"n={metrics['premise_n']})")
        print(f"  Val Overall Perf: {metrics['overall_perf']:.4f}  "
              f"= (Acc + F1) / 2")
        print(f"  Val TCE         : {metrics['content_effect']:.4f}")
        print(f"  Val Combined    : {metrics['combined_score']:.4f}  ← ranking metric")
        print(f"  [PV={metrics['acc_plausible_valid']:.1f}  "
              f"IV={metrics['acc_implausible_valid']:.1f}  "
              f"PI={metrics['acc_plausible_invalid']:.1f}  "
              f"II={metrics['acc_implausible_invalid']:.1f}]")
        print(f"  CE Intra: {metrics['ce_intra']:.4f}  "
              f"CE Inter: {metrics['ce_inter']:.4f}")

        # ── Save best checkpoint ──
        if metrics["combined_score"] > best_combined:
            best_combined = metrics["combined_score"]
            best_metrics  = metrics.copy()
            m = model.module if hasattr(model, "module") else model
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(m.state_dict(), os.path.join(MODEL_SAVE_DIR, "model_weights.pt"))
            print(f"  ✓ New best combined_score: {best_combined:.4f} → checkpoint saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        print(f"{'─'*70}\n")
        sys.stdout.flush()

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"Training complete.  Best val combined_score: {best_combined:.4f}")
    print(f"  Accuracy     : {best_metrics.get('accuracy', 0):.2f}%")
    print(f"  Premise F1   : {best_metrics.get('premise_f1', 0):.2f}%")
    print(f"  Overall Perf : {best_metrics.get('overall_perf', 0):.4f}")
    print(f"  TCE          : {best_metrics.get('content_effect', 0):.4f}")
    print(f"  Combined     : {best_metrics.get('combined_score', 0):.4f}")
    print(f"Checkpoint saved to: {MODEL_SAVE_DIR}")
    print(f"{'='*70}\n")

    results_path = os.path.join(MODEL_SAVE_DIR, "train_results.json")
    with open(results_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"Results saved: {results_path}")

    return history


if __name__ == "__main__":
    train()
