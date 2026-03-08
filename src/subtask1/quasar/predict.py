"""
predict.py
----------
Inference module: runs the trained SyllogismClassifier on a DataLoader
(typically test data) and produces a JSON predictions file in the exact
format required by the official SemEval-2026 Task 11 evaluation script.

Expected output format (Subtask 1):
  [
    {"id": "<uuid>", "validity": true},
    {"id": "<uuid>", "validity": false},
    ...
  ]
"""

import os
import sys
import json
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import ID2LABEL, PREDICTIONS_PATH
from model import SyllogismClassifier


# ─── Core Inference Function ──────────────────────────────────────────────────

def run_inference(
    model: SyllogismClassifier,
    loader: DataLoader,
    device: torch.device,
    return_probabilities: bool = False,
) -> List[Dict]:
    """
    Run inference and return a list of prediction dicts.

    Parameters
    ----------
    model                : trained SyllogismClassifier (on `device`)
    loader               : DataLoader (test or validation set)
    device               : torch device
    return_probabilities : if True, include 'prob_valid' in output

    Returns
    -------
    list of dicts: [{"id": str, "validity": bool, "prob_valid": float (optional)}]
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids           = batch["id"]   # list of UUID strings

            out = model(input_ids, attention_mask)
            logits = out["logits"]                        # (B, 2)
            probs  = torch.softmax(logits, dim=-1)        # (B, 2)
            preds  = logits.argmax(dim=-1).cpu().tolist() # (B,) int

            for i, (pred, item_id) in enumerate(zip(preds, ids)):
                entry: Dict = {
                    "id":       item_id,
                    "validity": ID2LABEL[pred],   # True or False
                }
                if return_probabilities:
                    entry["prob_valid"] = round(float(probs[i, 1].item()), 6)
                predictions.append(entry)

    return predictions


# ─── Prediction + Serialisation ───────────────────────────────────────────────

def predict_and_save(
    model: SyllogismClassifier,
    loader: DataLoader,
    device: torch.device,
    output_path: str = PREDICTIONS_PATH,
    return_probabilities: bool = False,
) -> List[Dict]:
    """
    Run inference, save predictions to `output_path`, and return them.

    The output JSON strictly follows the evaluation script's expected format:
      [{"id": "...", "validity": true/false}, ...]
    """
    print(f"[Predict] Running inference on {len(loader.dataset)} examples …")
    predictions = run_inference(model, loader, device, return_probabilities)

    # Produce the clean submission format (only id + validity)
    submission = [{"id": p["id"], "validity": p["validity"]} for p in predictions]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2)

    n_valid   = sum(1 for p in submission if p["validity"] is True)
    n_invalid = len(submission) - n_valid
    print(f"[Predict] {len(submission)} predictions saved to {output_path}")
    print(f"          valid={n_valid}, invalid={n_invalid}")

    return predictions


# ─── Analysis Helper ──────────────────────────────────────────────────────────

def analyse_predictions(
    predictions: List[Dict],
    ground_truth_path: Optional[str] = None,
) -> None:
    """
    Print a brief distribution analysis of the predictions.
    Optionally compare against ground-truth labels if a path is provided.
    """
    n_valid   = sum(1 for p in predictions if p["validity"] is True)
    n_invalid = len(predictions) - n_valid
    print(f"\n[Analysis] Prediction distribution:")
    print(f"  Valid:   {n_valid:>5}  ({100*n_valid/len(predictions):.1f}%)")
    print(f"  Invalid: {n_invalid:>5}  ({100*n_invalid/len(predictions):.1f}%)")

    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, "r") as f:
            gt_data = json.load(f)
        gt_map = {item["id"]: item for item in gt_data}

        correct = sum(
            1 for p in predictions
            if p["id"] in gt_map and gt_map[p["id"]]["validity"] == p["validity"]
        )
        acc = correct / len(predictions) * 100
        print(f"  Accuracy vs ground truth: {acc:.2f}%  ({correct}/{len(predictions)})")
