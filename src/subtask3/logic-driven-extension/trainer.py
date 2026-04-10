"""
trainer.py
----------
Training loop for the LReasoner model on Subtask 3.

Key features:
  - AdamW optimizer with linear warmup + decay
  - Early stopping on combined_score = accuracy / (1 + ln(1 + TCE))
  - Best checkpoint saved when combined_score improves
  - End-of-training: saves checkpoint + config with timestamp
  - Content effect computed using 4-subgroup accuracy breakdown
  - Gradient clipping for stability
"""

import math
import time
import random
import os

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from types import SimpleNamespace
from typing import Dict, List, Optional

from config_loader import save_config_with_timestamp, save_checkpoint_with_timestamp


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Validation Metrics ───────────────────────────────────────────────────────

def compute_val_metrics(
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute accuracy, content_effect (TCE), and combined_score on validation set.

    combined_score = accuracy / (1 + ln(1 + TCE))
    This is the same formula as the official evaluation_script.py.
    """
    model.eval()

    # 4 subgroups: (validity_label, plausibility_label)
    subgroups = {
        (1, 1): {"correct": 0, "total": 0},  # valid + plausible
        (1, 0): {"correct": 0, "total": 0},  # valid + implausible
        (0, 1): {"correct": 0, "total": 0},  # invalid + plausible
        (0, 0): {"correct": 0, "total": 0},  # invalid + implausible
    }

    with torch.no_grad():
        for batch in val_loader:
            input_ids_plus = batch["input_ids_plus"].to(device)
            attention_mask_plus = batch["attention_mask_plus"].to(device)
            labels = batch["label"]  # keep on CPU, model won't need it
            plausibilities = batch["plausibility"]  # keep on CPU

            out = model(input_ids_plus, attention_mask_plus)
            preds = out["logits"].argmax(dim=-1).detach().cpu()

            for pred, label, plaus in zip(preds, labels, plausibilities):
                key = (int(label), int(plaus))
                if key in subgroups:
                    subgroups[key]["total"] += 1
                    if pred == label:
                        subgroups[key]["correct"] += 1
            
            # Free tensors explicitly
            del input_ids_plus, attention_mask_plus, out, preds

    def acc(k):
        g = subgroups[k]
        return (g["correct"] / g["total"] * 100) if g["total"] > 0 else 0.0

    a_pv = acc((1, 1))   # plausible + valid
    a_iv = acc((1, 0))   # implausible + valid
    a_pi = acc((0, 1))   # plausible + invalid
    a_ii = acc((0, 0))   # implausible + invalid

    total_correct = sum(g["correct"] for g in subgroups.values())
    total_items = sum(g["total"] for g in subgroups.values())
    overall_acc = (total_correct / total_items * 100) if total_items > 0 else 0.0

    # Content effect (same as official evaluation_script.py)
    intra_valid = abs(a_pv - a_iv)
    intra_invalid = abs(a_pi - a_ii)
    ce_intra = (intra_valid + intra_invalid) / 2.0

    inter_plaus = abs(a_pv - a_pi)
    inter_implaus = abs(a_iv - a_ii)
    ce_inter = (inter_plaus + inter_implaus) / 2.0

    tce = (ce_intra + ce_inter) / 2.0
    log_penalty = math.log(1 + tce)
    combined = overall_acc / (1 + log_penalty)

    return {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(tce, 4),
        "combined_score": round(combined, 4),
        "acc_plausible_valid": round(a_pv, 2),
        "acc_implausible_valid": round(a_iv, 2),
        "acc_plausible_invalid": round(a_pi, 2),
        "acc_implausible_invalid": round(a_ii, 2),
        "ce_intra": round(ce_intra, 4),
        "ce_inter": round(ce_inter, 4),
    }


# ─── Training Loop ─────────────────────────────────────────────────────────────

def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: SimpleNamespace,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """
    Full training procedure with early stopping on combined_score.

    Args:
        model: LReasonerModel (already on device)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        cfg: config namespace
        class_weights: optional tensor for loss weighting

    Returns:
        history dict: {"train_loss", "val_accuracy", "val_content_effect", "val_combined"}
    """
    set_seed(cfg.seed)
    device = next(model.parameters()).device

    # Set class weights
    if class_weights is not None:
        model.set_class_weights(class_weights.to(device))

    # Optimizer: AdamW with separate weight decay for bias/LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=cfg.learning_rate)

    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Best checkpoint path
    best_ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")

    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_content_effect": [],
        "val_combined": [],
    }
    best_combined = -1.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Starting training: {cfg.num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Warmup steps: {warmup_steps} / {total_steps} total")
    print(f"Device: {device}")
    print(f"Early stopping: patience={cfg.early_stopping_patience} on combined_score")
    print(f"{'='*70}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids_plus = batch["input_ids_plus"].to(device)
            attention_mask_plus = batch["attention_mask_plus"].to(device)
            input_ids_minus = batch["input_ids_minus"].to(device)
            attention_mask_minus = batch["attention_mask_minus"].to(device)
            labels = batch["label"].to(device)

            out = model(
                input_ids_plus, attention_mask_plus,
                input_ids_minus, attention_mask_minus,
                labels=labels,
            )
            loss = out["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            
            # Explicitly delete tensors to free memory on MPS
            del input_ids_plus, attention_mask_plus, input_ids_minus, attention_mask_minus, labels, out, loss
            
            if str(device) == "mps":
                torch.mps.empty_cache()

            if step % 50 == 0 or step == len(train_loader):
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step
                print(
                    f"  Epoch {epoch}/{cfg.num_epochs}  "
                    f"Step {step:>4}/{len(train_loader)}  "
                    f"Loss: {loss_val:.4f}  AvgLoss: {avg_loss:.4f}  "
                    f"LR: {lr_now:.2e}"
                )

        # ─── Validation ───────────────────────────────────────────────
        val_metrics = compute_val_metrics(model, val_loader, device)
        avg_train_loss = epoch_loss / len(train_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_content_effect"].append(val_metrics["content_effect"])
        history["val_combined"].append(val_metrics["combined_score"])

        elapsed = time.time() - epoch_start
        print(f"\n{'---'*23}")
        print(f"  Epoch {epoch} complete in {elapsed:.1f}s")
        print(f"  Train Loss    : {avg_train_loss:.4f}")
        print(f"  Val Accuracy  : {val_metrics['accuracy']:.2f}%")
        print(f"  Val TCE       : {val_metrics['content_effect']:.4f}")
        print(f"  Val Combined  : {val_metrics['combined_score']:.4f}  <- early stopping metric")
        print(f"  [PV={val_metrics['acc_plausible_valid']:.1f}  "
              f"IV={val_metrics['acc_implausible_valid']:.1f}  "
              f"PI={val_metrics['acc_plausible_invalid']:.1f}  "
              f"II={val_metrics['acc_implausible_invalid']:.1f}]")

        # ─── Best Checkpoint ──────────────────────────────────────────
        if val_metrics["combined_score"] > best_combined:
            best_combined = val_metrics["combined_score"]
            model.save(best_ckpt_path)
            print(f"  -> New best combined_score: {best_combined:.4f} -> checkpoint saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{cfg.early_stopping_patience}")

        print(f"{'---'*23}\n")

        if patience_counter >= cfg.early_stopping_patience:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    # ─── End-of-Training: Save with Timestamp ──────────────────────────
    print(f"\n{'='*70}")
    print(f"Training complete. Best val combined_score: {best_combined:.4f}")

    # Save final checkpoint with timestamp
    save_checkpoint_with_timestamp(
        cfg,
        model.state_dict(),
        extra_info={
            "best_combined_score": best_combined,
            "final_epoch": epoch,
            "history": history,
        },
    )

    # Save config with timestamp
    save_config_with_timestamp(cfg)

    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"{'='*70}\n")

    return history
