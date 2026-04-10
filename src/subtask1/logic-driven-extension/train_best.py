#!/usr/bin/env python3
"""
Train with Best Hyperparameters - LReasoner (Logic-Driven Extension).

After the sweep completes, this script:
  1. Loads the best configuration from the sweep log.
  2. Trains on the FULL training set with those hyperparameters.
  3. Saves the trained model checkpoint.
  4. Optionally produces predictions on the test set.

Designed for Kaggle GPU. Called automatically by kaggle_runner.py.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup

from dataset import SyllogismDataset
from model import LReasonerModel


def train_full(cfg, train_data_path, device, verbose=True):
    """Train model on full training data with the given hyperparameters."""
    device = torch.device(device)
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

    print("\nLoading training data ...")
    train_dataset = SyllogismDataset(
        train_data_path, tokenizer, max_length=cfg["max_length"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
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
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Train (full data, no early stopping)
    print(f"Training for {cfg['epochs']} epochs on full training set ...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
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
            total_loss += loss.item()

        avg = total_loss / len(train_loader)
        if verbose:
            print(f"  Epoch {epoch}/{cfg['epochs']} | Loss: {avg:.4f}")

    return model, tokenizer


def predict_test(model, tokenizer, test_data_path, cfg, device, output_path):
    """Generate predictions on the test set."""
    device_obj = torch.device(device)
    test_dataset = SyllogismDataset(
        test_data_path, tokenizer, max_length=cfg["max_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], num_workers=0)

    model.eval()
    predictions = []
    print("Generating predictions on test set ...")
    with torch.no_grad():
        for batch in test_loader:
            logits, _ = model(
                input_ids_plus=batch["input_ids_plus"].to(device_obj),
                attention_mask_plus=batch["attention_mask_plus"].to(device_obj),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i, p in enumerate(preds):
                predictions.append({
                    "id": batch["id"][i],
                    "validity": bool(p),
                })

    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_path} ({len(predictions)} samples)")
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Train LReasoner with best hyperparameters from sweep"
    )
    parser.add_argument("--sweep_log", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_model", type=str, default="best_lreasoner_model.pt")
    parser.add_argument("--output_preds", type=str, default="predictions_subtask_1.json")
    args = parser.parse_args()

    # Load best config from sweep
    with open(args.sweep_log) as f:
        sweep_data = json.load(f)
    cfg = sweep_data["best_config"]["hyperparameters"]

    print("=" * 70)
    print("TRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    print("Best hyperparameters:")
    print(json.dumps(cfg, indent=2))
    print(f"CV combined_score: {sweep_data['best_config']['mean_combined_score']}")
    print("=" * 70)

    model, tokenizer = train_full(cfg, args.train_data, args.device)

    torch.save(model.state_dict(), args.output_model)
    print(f"\nModel saved to {args.output_model}")

    if args.test_data and os.path.exists(args.test_data):
        predict_test(model, tokenizer, args.test_data, cfg, args.device, args.output_preds)
    else:
        print("\nNo test data provided; skipping prediction.")


if __name__ == "__main__":
    main()
