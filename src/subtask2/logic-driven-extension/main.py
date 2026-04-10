"""
main.py
-------
CLI entry point for the Logic-Driven Extension approach on Subtask 2
(Monolingual Syllogistic Reasoning + Premise Identification).

Pipeline:
  1. Load config from config.yaml
  2. Load and split data (grouping augmentations to prevent leakage)
  3. Parse English syllogisms for logic, build contrastive pairs
  4. Fine-tune XLM-RoBERTa with logic-extended context + contrastive loss
     + premise selection loss
  5. Save best checkpoint (best combined_score) + timestamped archives
  6. Evaluate on test set using official metrics (accuracy + f1_premises)

Usage:
  python main.py --mode full       # full train + evaluate pipeline
  python main.py --mode train      # train only
  python main.py --mode evaluate   # evaluate from existing predictions
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer

# Add current dir to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from config_loader import load_config
from dataset import (
    SyllogismDataset,
    load_json, train_val_split, get_class_weights,
)
from model import LReasonerPremiseModel
from trainer import train, set_seed, compute_val_metrics

# Add evaluation kit to path
EVAL_KIT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "dataset", "evaluation_kit", "task 2 & 4")
)
sys.path.insert(0, EVAL_KIT_DIR)
try:
    from evaluation_script import run_full_scoring, calculate_smooth_combined_metric
except ImportError:
    print("[Warning] Could not import evaluation_script. Evaluation will use built-in metrics.")
    run_full_scoring = None
    calculate_smooth_combined_metric = None


# ─── Device Setup ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Main] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Main] Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("[Main] Using CPU (training will be slow)")
    return device


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict(model, test_loader: DataLoader, device: torch.device) -> list:
    """Generate predictions on the test set (validity + relevant_premises)."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_plus = batch["input_ids_plus"].to(device)
            attention_mask_plus = batch["attention_mask_plus"].to(device)
            premise_input_ids = batch["premise_input_ids"].to(device)
            premise_attention_mask = batch["premise_attention_mask"].to(device)
            premise_mask = batch["premise_mask"].to(device)

            out = model(
                input_ids_plus, attention_mask_plus,
                premise_input_ids=premise_input_ids,
                premise_attention_mask=premise_attention_mask,
                premise_mask=premise_mask,
            )
            validity_preds = out["logits"].argmax(dim=-1).detach().cpu()

            # Premise predictions
            if "premise_logits" in out:
                batch_premises = model.predict_premises(
                    out["premise_logits"].detach().cpu(),
                    batch["premise_mask"],
                    validity_preds,
                )
            else:
                batch_premises = [[] for _ in range(validity_preds.size(0))]

            for i in range(validity_preds.size(0)):
                predictions.append({
                    "id": batch["id"][i],
                    "validity": bool(validity_preds[i]),
                    "relevant_premises": batch_premises[i],
                })

            # Free tensors explicitly
            del (input_ids_plus, attention_mask_plus, premise_input_ids,
                 premise_attention_mask, premise_mask, out)

            if str(device) == "mps":
                torch.mps.empty_cache()

    return predictions


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions: list,
    reference_path: str,
    output_path: str = None,
) -> dict:
    """
    Evaluate predictions against ground truth.
    Computes accuracy, f1_premises, content effect, and combined score.
    """
    with open(reference_path, "r") as f:
        ground_truth = json.load(f)

    gt_map = {item["id"]: item for item in ground_truth}

    # Overall accuracy
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt and isinstance(pred["validity"], bool):
            total += 1
            if pred["validity"] == gt["validity"]:
                correct += 1
    overall_acc = (correct / total * 100) if total > 0 else 0.0

    # F1 for premise retrieval
    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt and "relevant_premises" in gt and "relevant_premises" in pred:
            true_set = set(gt["relevant_premises"])
            pred_set = set(pred["relevant_premises"])
            if len(true_set) == 0:
                continue
            TP = len(true_set.intersection(pred_set))
            FP = len(pred_set.difference(true_set))
            FN = len(true_set.difference(pred_set))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            total_precision += precision
            total_recall += recall
            valid_count += 1

    if valid_count > 0:
        macro_p = total_precision / valid_count
        macro_r = total_recall / valid_count
        f1_premises = (2 * macro_p * macro_r / (macro_p + macro_r)
                       if (macro_p + macro_r) > 0 else 0.0) * 100
    else:
        f1_premises = 0.0

    # Subgroup accuracies
    def subgroup_acc(gt_validity, gt_plausibility):
        c = t = 0
        for pred in predictions:
            gt = gt_map.get(pred["id"])
            if gt is None:
                continue
            if gt.get("validity") == gt_validity and gt.get("plausibility") == gt_plausibility:
                t += 1
                if pred["validity"] == gt["validity"]:
                    c += 1
        return (c / t * 100) if t > 0 else 0.0

    a_pv = subgroup_acc(True, True)
    a_iv = subgroup_acc(True, False)
    a_pi = subgroup_acc(False, True)
    a_ii = subgroup_acc(False, False)

    # Content effect
    ce_intra = (abs(a_pv - a_iv) + abs(a_pi - a_ii)) / 2.0
    ce_inter = (abs(a_pv - a_pi) + abs(a_iv - a_ii)) / 2.0
    tce = (ce_intra + ce_inter) / 2.0

    # Combined score
    overall_performance = (overall_acc + f1_premises) / 2.0
    combined = overall_performance / (1 + math.log(1 + tce))

    metrics = {
        "accuracy": round(overall_acc, 4),
        "f1_premises": round(f1_premises, 4),
        "content_effect": round(tce, 4),
        "combined_score": round(combined, 4),
        "acc_plausible_valid": round(a_pv, 2),
        "acc_implausible_valid": round(a_iv, 2),
        "acc_plausible_invalid": round(a_pi, 2),
        "acc_implausible_invalid": round(a_ii, 2),
        "n_predicted": total,
    }

    # Print report
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Evaluation Report — Subtask 2 (Logic-Driven Extension)")
    print(sep)
    print(f"  Overall Accuracy    : {metrics['accuracy']:.4f}%")
    print(f"  F1 Premises         : {metrics['f1_premises']:.4f}")
    print(f"  Total Content Effect: {metrics['content_effect']:.4f}")
    print(f"  Combined Score (↑)  : {metrics['combined_score']:.4f}  ← ranking metric")
    print(f"  {'-'*56}")
    print(f"  Subgroup Accuracies:")
    print(f"    Valid   + Plausible   (PV): {a_pv:.2f}%")
    print(f"    Valid   + Implausible (IV): {a_iv:.2f}%")
    print(f"    Invalid + Plausible   (PI): {a_pi:.2f}%")
    print(f"    Invalid + Implausible (II): {a_ii:.2f}%")
    print(f"  {'-'*56}")
    print(f"  N predicted: {total}")
    print(sep)

    # Save metrics
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  [Evaluate] Metrics saved to {output_path}")

    return metrics


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Logic-Driven Extension — Subtask 2 (Monolingual Syllogisms + Premise ID)"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model checkpoint (for evaluate mode)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.num_epochs = args.epochs

    set_seed(cfg.seed)
    device = get_device()

    print(f"\n{'#'*70}")
    print(f"  Logic-Driven Extension — Subtask 2: Monolingual Syllogisms")
    print(f"  + Premise Identification")
    print(f"  Model: {cfg.model_name}")
    print(f"  Contrastive α: {cfg.alpha}, Cosine margin: {cfg.cosine_margin}")
    print(f"  Premise β: {cfg.beta}, Threshold: {cfg.premise_threshold}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation_steps} steps")
    print(f"{'#'*70}\n")

    # Paths
    best_ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
    predictions_path = os.path.join(cfg.output_dir, "predictions_subtask2.json")
    metrics_path = os.path.join(cfg.output_dir, "metrics_subtask2.json")

    if args.mode in ("full", "train"):
        # ─── Load Data ─────────────────────────────────────────────────
        print("[Main] Loading tokenizer...")
        tokenizer = XLMRobertaTokenizer.from_pretrained(cfg.model_name)

        print("[Main] Loading training data...")
        all_train_data = load_json(cfg.train_data_path)
        print(f"  Total training samples: {len(all_train_data)}")

        print("[Main] Splitting train/val...")
        train_data, val_data = train_val_split(
            all_train_data,
            val_ratio=cfg.validation_split,
            seed=cfg.seed,
        )

        print("[Main] Building datasets...")
        train_dataset = SyllogismDataset(train_data, tokenizer, cfg, has_labels=True)
        val_dataset = SyllogismDataset(val_data, tokenizer, cfg, has_labels=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        # ─── Model ─────────────────────────────────────────────────────
        print("[Main] Initializing model...")
        model = LReasonerPremiseModel(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Class weights
        class_weights = get_class_weights(all_train_data, cfg.label2id)

        # ─── Train ─────────────────────────────────────────────────────
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            class_weights=class_weights,
        )

    if args.mode in ("full", "evaluate"):
        # ─── Evaluate on Test Set ──────────────────────────────────────
        print("\n[Main] Loading test data...")
        test_data = load_json(cfg.test_data_path)
        print(f"  Test samples: {len(test_data)}")

        # Load tokenizer if not already loaded
        if args.mode == "evaluate":
            tokenizer = XLMRobertaTokenizer.from_pretrained(cfg.model_name)

        test_dataset = SyllogismDataset(test_data, tokenizer, cfg, has_labels=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Load best model
        model_path = args.model_path or best_ckpt_path
        if os.path.exists(model_path):
            print(f"[Main] Loading best model from {model_path}")
            model = LReasonerPremiseModel.load(model_path, cfg).to(device)
        else:
            if args.mode == "evaluate":
                print(f"[Main] ERROR: No model found at {model_path}")
                sys.exit(1)
            # Model already in memory from training
            print("[Main] Using in-memory model (loading best checkpoint)...")
            if os.path.exists(best_ckpt_path):
                model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        # Generate predictions
        print("[Main] Generating predictions on test set...")
        predictions = predict(model, test_loader, device)

        # Save predictions
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(predictions_path, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"  Predictions saved to {predictions_path}")

        # Evaluate
        metrics = evaluate_predictions(
            predictions=predictions,
            reference_path=cfg.test_data_path,
            output_path=metrics_path,
        )

        # Also run official evaluation script if available
        if run_full_scoring is not None:
            print("\n[Main] Running official evaluation script...")
            official_output = os.path.join(cfg.output_dir, "official_metrics.json")
            run_full_scoring(cfg.test_data_path, predictions_path, official_output)

    print("\n[Main] Pipeline complete!")


if __name__ == "__main__":
    main()
