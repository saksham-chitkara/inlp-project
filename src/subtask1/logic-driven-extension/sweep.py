#!/usr/bin/env python3
"""
Hyperparameter Sweep for LReasoner (Logic-Driven Extension) - Subtask 1.

Performs a systematic random search over key hyperparameters using stratified
k-fold cross-validation on the training set. Each configuration is evaluated
by the official combined_score metric:

    combined_score = accuracy / (1 + ln(1 + content_effect))

where content_effect captures the bias introduced by syllogism plausibility.

Results are persisted incrementally in a JSON log after every configuration,
so runs can be resumed if interrupted (e.g. Kaggle timeout).

DO NOT RUN ON CPU / LAPTOP - designed for Kaggle GPU (T4 / P100).

Usage (Kaggle - see kaggle_runner.py):
    python sweep.py --train_data /kaggle/input/... --device cuda --n_folds 5
"""

import argparse
import copy
import gc
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold

from dataset import SyllogismDataset
from model import LReasonerModel


# ---------------------------------------------------------------------------
# Evaluation helpers (inlined from official evaluation_script.py)
# ---------------------------------------------------------------------------

def _calculate_accuracy(ground_truth, predictions):
    """Overall validity accuracy (%)."""
    gt_map = {item["id"]: item for item in ground_truth}
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt and isinstance(gt["validity"], bool) and isinstance(pred["validity"], bool):
            total += 1
            if gt["validity"] == pred["validity"]:
                correct += 1
    return (correct / total * 100) if total else 0.0


def _subgroup_accuracy(gt_map, predictions, gt_validity, gt_plausibility):
    """Accuracy on a specific (validity, plausibility) subgroup."""
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if (gt
                and gt.get("validity") == gt_validity
                and gt.get("plausibility") == gt_plausibility):
            if isinstance(gt["validity"], bool) and isinstance(pred["validity"], bool):
                total += 1
                if gt["validity"] == pred["validity"]:
                    correct += 1
    return (correct / total * 100) if total else 0.0


def compute_combined_score(ground_truth, predictions):
    """
    Official combined_score = accuracy / (1 + ln(1 + content_effect)).
    Returns dict with accuracy, content_effect, combined_score.
    """
    gt_map = {item["id"]: item for item in ground_truth}
    overall_acc = _calculate_accuracy(ground_truth, predictions)

    apv = _subgroup_accuracy(gt_map, predictions, True, True)
    aiv = _subgroup_accuracy(gt_map, predictions, True, False)
    api = _subgroup_accuracy(gt_map, predictions, False, True)
    aii = _subgroup_accuracy(gt_map, predictions, False, False)

    intra = (abs(apv - aiv) + abs(api - aii)) / 2.0
    inter = (abs(apv - api) + abs(aiv - aii)) / 2.0
    tot_ce = (intra + inter) / 2.0

    combined = overall_acc / (1 + math.log(1 + tot_ce)) if tot_ce >= 0 else 0.0

    return {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(tot_ce, 4),
        "combined_score": round(combined, 4),
    }


# ---------------------------------------------------------------------------
# Training and evaluation for a single fold
# ---------------------------------------------------------------------------

def train_one_fold(model, train_loader, val_loader, optimizer, scheduler,
                   device, epochs, patience, fold_id, verbose=True):
    """Train on one CV fold. Returns (best_val_loss, best_state_dict)."""
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, loss = model(
                input_ids_plus=batch["input_ids_plus"].to(device),
                attention_mask_plus=batch["attention_mask_plus"].to(device),
                input_ids_minus=batch["input_ids_minus"].to(device),
                attention_mask_minus=batch["attention_mask_minus"].to(device),
                labels=batch["label"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()

        avg_train = train_loss_sum / len(train_loader)

        # Validate (no contrastive loss)
        model.eval()
        val_loss_sum = 0.0
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                logits, loss = model(
                    input_ids_plus=batch["input_ids_plus"].to(device),
                    attention_mask_plus=batch["attention_mask_plus"].to(device),
                    labels=batch["label"].to(device),
                )
                val_loss_sum += loss.item()
                preds = torch.argmax(logits, dim=1)
                labels = batch["label"].to(device)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val = val_loss_sum / len(val_loader)
        val_acc = correct / total if total else 0.0

        if verbose:
            print(
                f"  Fold {fold_id} | Epoch {epoch:>2}/{epochs} | "
                f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Fold {fold_id} | Early stopping at epoch {epoch}.")
                break

    return best_val_loss, best_state


def predict_fold(model, dataloader, device):
    """Return list of {id, validity} dicts from a DataLoader."""
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(
                input_ids_plus=batch["input_ids_plus"].to(device),
                attention_mask_plus=batch["attention_mask_plus"].to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i, p in enumerate(preds):
                preds_list.append({"id": batch["id"][i], "validity": bool(p)})
    return preds_list


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [8, 16],
    "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
    "dropout": [0.1, 0.2, 0.3],
    "weight_decay": [0.01, 0.05, 0.1],
    "warmup_ratio": [0.06, 0.1, 0.15],
    "max_length": [192, 256],
    "patience": [3, 5],
    "epochs": [8, 12],
}


def sample_configs(space, n_samples=60, seed=42):
    """Deterministic random sample from the Cartesian product."""
    rng = np.random.RandomState(seed)
    keys = sorted(space.keys())
    all_combos = list(itertools.product(*(space[k] for k in keys)))
    n_samples = min(n_samples, len(all_combos))
    indices = rng.choice(len(all_combos), size=n_samples, replace=False)
    configs = []
    for idx in indices:
        combo = all_combos[idx]
        configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_existing_results(output_dir):
    """Load previous sweep results for resume capability."""
    p = Path(output_dir)
    if not p.exists():
        return set(), None, [], None

    candidates = sorted(
        p.glob("sweep_results_*.json"), key=os.path.getmtime, reverse=True
    )
    for cand in candidates:
        try:
            with open(cand) as f:
                data = json.load(f)
            results = data.get("all_results", [])
            completed = {r["config_id"] for r in results}
            best = data.get("best_config")
            print(f"[Resume] Found {len(completed)} completed configs in {cand.name}")
            return completed, best, results, str(cand)
        except (json.JSONDecodeError, KeyError):
            continue
    return set(), None, [], None


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

def run_sweep(args):
    """Execute the full hyperparameter sweep."""
    device = torch.device(args.device)
    print(f"Device: {device}")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    # Pre-process dataset once per max_length in the search space
    max_lengths_in_space = sorted(set(SEARCH_SPACE["max_length"]))
    datasets_by_maxlen = {}
    for ml in max_lengths_in_space:
        print(f"Pre-processing dataset with max_length={ml} ...")
        datasets_by_maxlen[ml] = SyllogismDataset(args.train_data, tokenizer, max_length=ml)

    # Ground-truth for scoring
    with open(args.train_data, "r") as f:
        gt_full = json.load(f)
    gt_map_full = {item["id"]: item for item in gt_full}

    # Stratification labels: 2*validity + plausibility -> 4 groups
    label_array = np.array(
        [2 * int(item["validity"]) + int(item["plausibility"]) for item in gt_full]
    )

    # Generate configurations
    configs = sample_configs(SEARCH_SPACE, n_samples=args.n_configs, seed=args.seed)
    print(f"\nTotal configurations to evaluate: {len(configs)}")
    print(f"Folds per config: {args.n_folds}")
    print(f"Total training runs: {len(configs) * args.n_folds}\n")

    # Resume support
    os.makedirs(args.output_dir, exist_ok=True)
    completed_ids, best_config, results_log, existing_log = load_existing_results(
        args.output_dir
    )
    best_combined = best_config["mean_combined_score"] if best_config else -1.0

    if existing_log:
        log_path = existing_log
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path(args.output_dir) / f"sweep_results_{timestamp}.json")

    skipped = 0
    for cfg_idx, cfg in enumerate(configs):
        config_id = cfg_idx + 1
        if config_id in completed_ids:
            skipped += 1
            continue

        t0 = time.time()
        print(f"\n{'='*70}")
        print(f"Config {config_id}/{len(configs)}: {json.dumps(cfg, sort_keys=True)}")
        if skipped:
            print(f"  (Skipped {skipped} previously completed configs)")
            skipped = 0
        print(f"{'='*70}")

        dataset = datasets_by_maxlen[cfg["max_length"]]
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

        fold_scores = []

        for fold_idx, (train_ids, val_ids) in enumerate(
            skf.split(np.zeros(len(dataset)), label_array)
        ):
            train_subset = Subset(dataset, train_ids.tolist())
            val_subset = Subset(dataset, val_ids.tolist())

            train_loader = DataLoader(
                train_subset, batch_size=cfg["batch_size"], shuffle=True,
                num_workers=0, pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_subset, batch_size=cfg["batch_size"],
                num_workers=0, pin_memory=(device.type == "cuda"),
            )

            # Build model
            model = LReasonerModel(model_name="xlm-roberta-base", alpha=cfg["alpha"])
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(cfg["dropout"]),
                torch.nn.Linear(model.encoder.config.hidden_size, 2),
            )
            model = model.to(device)

            total_steps = len(train_loader) * cfg["epochs"]
            warmup_steps = int(cfg["warmup_ratio"] * total_steps)
            optimizer = AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            )
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

            # Train
            _, best_state = train_one_fold(
                model, train_loader, val_loader, optimizer, scheduler,
                device, cfg["epochs"], cfg["patience"],
                fold_id=fold_idx + 1, verbose=args.verbose,
            )

            # Load best checkpoint
            if best_state is not None:
                model.load_state_dict(best_state)

            # Predict on validation fold
            val_preds = predict_fold(model, val_loader, device)

            # Ground-truth for this fold
            gt_fold = []
            for vi in val_ids:
                item = dataset.processed_data[vi]
                gt_entry = gt_map_full.get(item["id"])
                if gt_entry:
                    gt_fold.append(gt_entry)

            scores = compute_combined_score(gt_fold, val_preds)
            fold_scores.append(scores)
            print(
                f"  Fold {fold_idx+1} => "
                f"Acc: {scores['accuracy']:.2f}% | "
                f"CE: {scores['content_effect']:.2f} | "
                f"Combined: {scores['combined_score']:.2f}"
            )

            # Free GPU memory
            del model, optimizer, scheduler, best_state
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        # Aggregate fold results
        mean_acc = np.mean([s["accuracy"] for s in fold_scores])
        std_acc = np.std([s["accuracy"] for s in fold_scores])
        mean_ce = np.mean([s["content_effect"] for s in fold_scores])
        std_ce = np.std([s["content_effect"] for s in fold_scores])
        mean_combined = np.mean([s["combined_score"] for s in fold_scores])
        std_combined = np.std([s["combined_score"] for s in fold_scores])

        entry = {
            "config_id": config_id,
            "hyperparameters": cfg,
            "mean_accuracy": round(float(mean_acc), 4),
            "std_accuracy": round(float(std_acc), 4),
            "mean_content_effect": round(float(mean_ce), 4),
            "std_content_effect": round(float(std_ce), 4),
            "mean_combined_score": round(float(mean_combined), 4),
            "std_combined_score": round(float(std_combined), 4),
            "fold_scores": fold_scores,
            "time_seconds": round(time.time() - t0, 1),
        }
        results_log.append(entry)

        print(
            f"\n  => Mean Acc: {mean_acc:.2f} +/- {std_acc:.2f} | "
            f"Mean CE: {mean_ce:.2f} +/- {std_ce:.2f} | "
            f"Mean Combined: {mean_combined:.2f} +/- {std_combined:.2f} | "
            f"Time: {entry['time_seconds']:.0f}s"
        )

        if mean_combined > best_combined:
            best_combined = mean_combined
            best_config = entry
            print("  *** New best configuration! ***")

        # Persist incrementally (resume-safe)
        with open(log_path, "w") as f:
            json.dump(
                {"best_config": best_config, "all_results": results_log},
                f, indent=2,
            )

    # Final summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"Total configs evaluated: {len(results_log)}")
    print(f"Results saved to: {log_path}")
    print(f"\nBest configuration (by mean combined_score):")
    print(json.dumps(best_config, indent=2))
    print("=" * 70)

    return best_config, results_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for LReasoner")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_configs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="sweep_results")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_sweep(args)
