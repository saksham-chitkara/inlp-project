"""
train.py
--------
Fine-tuning loop for the SyllogismClassifier on Subtask 1 training data.

Training strategy:
  - AdamW optimiser with linear warm-up + linear decay (standard for transformers)
  - Cross-entropy loss with class weights to handle valid/invalid imbalance
  - Gradient clipping (max norm 1.0) for stability
  - Per-epoch validation; best checkpoint saved by combined_score
    (accuracy / (1 + ln(1 + content_effect)))
  - Early stopping on the combined_score (patience = 2 epochs)

Logging:
  - Detailed per-step loss logging (printed to stdout)
  - Per-epoch validation report with accuracy, content effect, combined score
"""

import math
import random
import time
import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_GRAD_NORM, SEED, MODEL_SAVE_DIR,
    USE_FP16, GRADIENT_ACCUMULATION_STEPS, EARLY_STOPPING_PATIENCE,
)
from model import SyllogismClassifier

# local import (used for inline validation scoring)
# kept lightweight to avoid circular imports
from typing import List, Dict, Optional


# --- Reproducibility ---
def set_seed(seed: int = SEED):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Validation Metric ---

def compute_val_metrics(
    model: SyllogismClassifier,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run inference on the validation set and compute:
      - accuracy
      - content_effect (TCE: total content effect)
      - combined_score = accuracy / (1 + ln(1 + TCE))

    We compute TCE using the four (validity x plausibility) subgroup accuracies,
    exactly as defined in the official evaluation script.
    """
    model.eval()

    # Accumulators for the 4 subgroups
    subgroups = {
        (1, 1): {"correct": 0, "total": 0},  # valid + plausible
        (1, 0): {"correct": 0, "total": 0},  # valid + implausible
        (0, 1): {"correct": 0, "total": 0},  # invalid + plausible
        (0, 0): {"correct": 0, "total": 0},  # invalid + implausible
    }

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            plausibilities = batch["plausibility"]  # keep on CPU for indexing

            out = model(input_ids, attention_mask)
            preds = out["logits"].argmax(dim=-1).cpu()

            for pred, label, plaus in zip(preds, labels.cpu(), plausibilities):
                key = (int(label), int(plaus))
                subgroups[key]["total"] += 1
                if pred == label:
                    subgroups[key]["correct"] += 1

    def acc(k): 
        g = subgroups[k]
        return (g["correct"] / g["total"] * 100) if g["total"] > 0 else 0.0

    a_pv  = acc((1, 1))   # plausible   + valid
    a_iv  = acc((1, 0))   # implausible + valid
    a_pi  = acc((0, 1))   # plausible   + invalid
    a_ii  = acc((0, 0))   # implausible + invalid

    total_correct = sum(g["correct"] for g in subgroups.values())
    total_items   = sum(g["total"]   for g in subgroups.values())
    overall_acc   = (total_correct / total_items * 100) if total_items > 0 else 0.0

    # Content effect (as in official evaluation_script.py)
    intra_valid    = abs(a_pv - a_iv)
    intra_invalid  = abs(a_pi - a_ii)
    ce_intra       = (intra_valid + intra_invalid) / 2.0

    inter_plaus    = abs(a_pv - a_pi)
    inter_implaus  = abs(a_iv - a_ii)
    ce_inter       = (inter_plaus + inter_implaus) / 2.0

    tce           = (ce_intra + ce_inter) / 2.0
    log_penalty   = math.log(1 + tce)
    combined      = overall_acc / (1 + log_penalty)

    return {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(tce, 4),
        "combined_score": round(combined, 4),
        "acc_plausible_valid": round(a_pv, 2),
        "acc_implausible_valid": round(a_iv, 2),
        "acc_plausible_invalid": round(a_pi, 2),
        "acc_implausible_invalid": round(a_ii, 2),
    }


# --- Training Loop ---

def train(
    model: SyllogismClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size_delta: int = 0,
    class_weights: Optional[torch.Tensor] = None,
    num_epochs: int = NUM_EPOCHS,
    save_path: str = MODEL_SAVE_DIR,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    use_fp16: bool = USE_FP16,
    grad_accum_steps: int = GRADIENT_ACCUMULATION_STEPS,
) -> Dict[str, List[float]]:
    """
    Full training procedure.

    Parameters
    ----------
    model          : SyllogismClassifier (already on device)
    train_loader   : DataLoader for training
    val_loader     : DataLoader for validation
    vocab_size_delta : number of extra tokens added by tokenizer
    class_weights  : optional tensor for loss weighting
    num_epochs     : number of training epochs
    save_path      : directory to save best checkpoint
    early_stopping_patience : stop if val combined_score doesn't improve

    Returns
    -------
    history dict: {"train_loss", "val_accuracy", "val_content_effect", "val_combined"}
    """
    set_seed()
    device = next(model.parameters()).device

    # Unwrap DataParallel for proper param access
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Note: embedding resizing was already handled during model construction
    # (SyllogismClassifier.__init__ receives vocab_size_delta directly).

    # Optimiser: AdamW with separate weight decay for bias/LayerNorm
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=LEARNING_RATE)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Store class_weights as model attribute (DataParallel-safe)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    raw_model.set_class_weights(class_weights)

    # Mixed-precision scaler
    scaler = GradScaler(enabled=use_fp16)

    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_content_effect": [],
        "val_combined": [],
    }
    best_combined = -1.0
    patience_counter = 0
    global_step = 0

    print(f"\n{'='*70}")
    print(f"Starting training: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Warmup steps: {warmup_steps} / {total_steps} total")
    print(f"Device: {device}  |  FP16: {use_fp16}  |  Grad Accum: {grad_accum_steps}")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"Multi-GPU: {n_gpu} GPUs via DataParallel")
    print(f"{'='*70}\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast(enabled=use_fp16):
                out = model(
                    input_ids,
                    attention_mask,
                    labels=labels,
                )
                loss = out["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()  # average across GPUs for DataParallel
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * (grad_accum_steps if grad_accum_steps > 1 else 1)
            global_step += 1

            if step % 50 == 0 or step == len(train_loader):
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step
                print(
                    f"  Epoch {epoch}/{num_epochs}  "
                    f"Step {step:>4}/{len(train_loader)}  "
                    f"Loss: {loss.item():.4f}  AvgLoss: {avg_loss:.4f}  "
                    f"LR: {lr_now:.2e}"
                )

        # --- Validation ---
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
        print(f"  Val Combined  : {val_metrics['combined_score']:.4f}  <- ranking metric")
        print(f"  [PV={val_metrics['acc_plausible_valid']:.1f}  "
              f"IV={val_metrics['acc_implausible_valid']:.1f}  "
              f"PI={val_metrics['acc_plausible_invalid']:.1f}  "
              f"II={val_metrics['acc_implausible_invalid']:.1f}]")

        # --- Save best checkpoint ---
        if val_metrics["combined_score"] > best_combined:
            best_combined = val_metrics["combined_score"]
            raw_model.save(save_path)
            print(f"  -> New best combined_score: {best_combined:.4f} -> checkpoint saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        print(f"{'---'*23}\n")

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n{'='*70}")
    print(f"Training complete.  Best val combined_score: {best_combined:.4f}")
    print(f"Checkpoint saved to: {save_path}")
    print(f"{'='*70}\n")

    return history
