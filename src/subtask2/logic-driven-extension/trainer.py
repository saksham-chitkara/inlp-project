"""
trainer.py
----------
Training loop for the LReasoner multi-task model on Subtask 2.

Key features:
  - AdamW optimizer with linear warmup + decay
  - Gradient accumulation for effective larger batch sizes
  - Early stopping on combined_score = (accuracy + f1_premises) / 2 / (1 + ln(1 + TCE))
  - Best checkpoint saved when combined_score improves
  - End-of-training: saves checkpoint + config with timestamp
  - Content effect computed using 4-subgroup accuracy breakdown
  - F1 for premise retrieval (macro-averaged)
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


# ─── Premise F1 Computation ──────────────────────────────────────────────────

def compute_premise_f1(
    predicted_premises: List[List[int]],
    true_premises: List[List[int]],
) -> float:
    """
    Compute macro-averaged F1 for premise retrieval.
    Matches the official evaluation_script.py for task 2 & 4.

    Args:
        predicted_premises: list of predicted premise index lists
        true_premises: list of ground-truth premise index lists

    Returns:
        F1 score (0-100 scale)
    """
    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0

    for pred_set, true_set in zip(predicted_premises, true_premises):
        tp_set = set(true_set)
        pp_set = set(pred_set)

        if len(tp_set) == 0:
            continue

        TP = len(tp_set.intersection(pp_set))
        FP = len(pp_set.difference(tp_set))
        FN = len(tp_set.difference(pp_set))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        valid_count += 1

    if valid_count == 0:
        return 0.0

    macro_precision = total_precision / valid_count
    macro_recall = total_recall / valid_count

    f1 = (2 * macro_precision * macro_recall / (macro_precision + macro_recall)
          if (macro_precision + macro_recall) > 0 else 0.0)

    return f1 * 100


# ─── Validation Metrics ───────────────────────────────────────────────────────

def compute_val_metrics(
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute accuracy, content_effect (TCE), f1_premises, and combined_score on
    the validation set.

    combined_score = (accuracy + f1_premises) / 2 / (1 + ln(1 + TCE))
    This matches the official task 2 & 4 evaluation_script.py.
    """
    model.eval()

    # 4 subgroups: (validity_label, plausibility_label)
    subgroups = {
        (1, 1): {"correct": 0, "total": 0},  # valid + plausible
        (1, 0): {"correct": 0, "total": 0},  # valid + implausible
        (0, 1): {"correct": 0, "total": 0},  # invalid + plausible
        (0, 0): {"correct": 0, "total": 0},  # invalid + implausible
    }

    all_predicted_premises = []
    all_true_premises = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids_plus = batch["input_ids_plus"].to(device)
            attention_mask_plus = batch["attention_mask_plus"].to(device)
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            premise_mask = batch["premise_mask"].to(device)
            labels = batch["label"]  # keep on CPU
            plausibilities = batch["plausibility"]  # keep on CPU
            premise_labels_batch = batch["premise_labels"]  # keep on CPU

            out = model(
                input_ids_plus, attention_mask_plus,
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                premise_mask=premise_mask,
            )
            preds = out["logits"].argmax(dim=-1).detach().cpu()

            # Premise predictions
            if "premise_logits" in out:
                batch_premises = model.predict_premises(
                    out["premise_logits"].detach().cpu(),
                    batch["premise_mask"],
                    preds,
                )
            else:
                batch_premises = [[] for _ in range(preds.size(0))]

            for i in range(preds.size(0)):
                pred = preds[i]
                label = labels[i]
                plaus = plausibilities[i]

                key = (int(label), int(plaus))
                if key in subgroups:
                    subgroups[key]["total"] += 1
                    if pred == label:
                        subgroups[key]["correct"] += 1

                # Collect premise predictions/labels
                all_predicted_premises.append(batch_premises[i])
                # Reconstruct true premises from labels tensor
                true_p = []
                p_labels = premise_labels_batch[i]
                p_mask = batch["premise_mask"][i]
                for j in range(p_labels.size(0)):
                    if p_mask[j] > 0 and p_labels[j] > 0.5:
                        true_p.append(j)
                all_true_premises.append(true_p)

            # Free tensors explicitly
            del input_ids_plus, attention_mask_plus, premise_input_ids
            del premise_attention_mask, premise_mask, out, preds

            if str(device) == "mps":
                torch.mps.empty_cache()

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

    # Content effect
    intra_valid = abs(a_pv - a_iv)
    intra_invalid = abs(a_pi - a_ii)
    ce_intra = (intra_valid + intra_invalid) / 2.0

    inter_plaus = abs(a_pv - a_pi)
    inter_implaus = abs(a_iv - a_ii)
    ce_inter = (inter_plaus + inter_implaus) / 2.0

    tce = (ce_intra + ce_inter) / 2.0

    # F1 for premise retrieval
    f1_premises = compute_premise_f1(all_predicted_premises, all_true_premises)

    # Combined score: (accuracy + f1_premises) / 2 / (1 + ln(1 + TCE))
    overall_performance = (overall_acc + f1_premises) / 2.0
    log_penalty = math.log(1 + tce)
    combined = overall_performance / (1 + log_penalty)

    return {
        "accuracy": round(overall_acc, 4),
        "f1_premises": round(f1_premises, 4),
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
    Full training procedure with gradient accumulation and early stopping
    on combined_score.

    Args:
        model: LReasonerPremiseModel (already on device)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        cfg: config namespace
        class_weights: optional tensor for loss weighting

    Returns:
        history dict
    """
    set_seed(cfg.seed)
    device = next(model.parameters()).device

    # Set class weights
    if class_weights is not None:
        model.set_class_weights(class_weights.to(device))

    # Gradient accumulation steps
    accum_steps = cfg.gradient_accumulation_steps

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

    # Scheduler accounts for gradient accumulation
    effective_steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = effective_steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Best checkpoint path
    best_ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")

    history = {
        "train_loss": [],
        "val_accuracy": [],
        "val_f1_premises": [],
        "val_content_effect": [],
        "val_combined": [],
    }
    best_combined = -1.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Starting training: {cfg.num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"Gradient accumulation: {accum_steps} steps → effective batch size = {cfg.batch_size * accum_steps}")
    print(f"Scheduler steps: warmup={warmup_steps} / total={total_steps}")
    print(f"Device: {device}")
    print(f"Early stopping: patience={cfg.early_stopping_patience} on combined_score")
    print(f"Loss: L_CE + {cfg.alpha}*L_CL + {cfg.beta}*L_BCE(premise)")
    print(f"{'='*70}\n")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_cl = 0.0
        epoch_loss_premise = 0.0
        epoch_start = time.time()

        optimizer.zero_grad()  # zero once at start of epoch

        for step, batch in enumerate(train_loader, 1):
            input_ids_plus = batch["input_ids_plus"].to(device)
            attention_mask_plus = batch["attention_mask_plus"].to(device)
            input_ids_minus = batch["input_ids_minus"].to(device)
            attention_mask_minus = batch["attention_mask_minus"].to(device)
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            premise_mask = batch["premise_mask"].to(device)
            premise_labels = batch["premise_labels"].to(device)
            labels = batch["label"].to(device)

            out = model(
                input_ids_plus, attention_mask_plus,
                input_ids_minus, attention_mask_minus,
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                premise_mask=premise_mask,
                premise_labels=premise_labels,
                labels=labels,
            )
            # Scale loss by accumulation steps
            loss = out["loss"] / accum_steps

            loss.backward()

            epoch_loss += out["loss"].item()
            epoch_loss_ce += out["loss_ce"].item()
            epoch_loss_cl += out["loss_cl"].item()
            epoch_loss_premise += out["loss_premise"].item()

            # Explicitly delete tensors to free memory
            del (input_ids_plus, attention_mask_plus, input_ids_minus,
                 attention_mask_minus, premise_input_ids, premise_attention_mask,
                 premise_mask, premise_labels, labels, out, loss)

            if str(device) == "mps":
                torch.mps.empty_cache()

            # Gradient accumulation: update every accum_steps or at end of epoch
            if step % accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 25 == 0 or step == len(train_loader):
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = epoch_loss / step
                print(
                    f"  Epoch {epoch}/{cfg.num_epochs}  "
                    f"Step {step:>4}/{len(train_loader)}  "
                    f"Loss: {epoch_loss / step:.4f}  "
                    f"LR: {lr_now:.2e}"
                )

        # ─── Validation ───────────────────────────────────────────────
        val_metrics = compute_val_metrics(model, val_loader, device)
        avg_train_loss = epoch_loss / len(train_loader)
        avg_ce = epoch_loss_ce / len(train_loader)
        avg_cl = epoch_loss_cl / len(train_loader)
        avg_prem = epoch_loss_premise / len(train_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_premises"].append(val_metrics["f1_premises"])
        history["val_content_effect"].append(val_metrics["content_effect"])
        history["val_combined"].append(val_metrics["combined_score"])

        elapsed = time.time() - epoch_start
        print(f"\n{'---'*23}")
        print(f"  Epoch {epoch} complete in {elapsed:.1f}s")
        print(f"  Train Loss     : {avg_train_loss:.4f}  "
              f"[CE={avg_ce:.4f}  CL={avg_cl:.4f}  Prem={avg_prem:.4f}]")
        print(f"  Val Accuracy   : {val_metrics['accuracy']:.2f}%")
        print(f"  Val F1 Premises: {val_metrics['f1_premises']:.2f}")
        print(f"  Val TCE        : {val_metrics['content_effect']:.4f}")
        print(f"  Val Combined   : {val_metrics['combined_score']:.4f}  <- early stopping metric")
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
