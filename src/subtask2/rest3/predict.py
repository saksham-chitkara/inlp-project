"""
predict.py
----------
Inference module for JointSyllogismClassifier (Subtask 2).

Output format (Subtask 2):
  [
    {"id": "<uuid>", "validity": true, "relevant_premises": [1, 3]},
    {"id": "<uuid>", "validity": false, "relevant_premises": []},
    ...
  ]

Rules (from SemEval-2026 Task 11):
  - For valid syllogisms:   output exactly the 2 most-probable relevant premise indices
  - For invalid syllogisms: output empty list []
"""

import json
import os
import sys
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.subtask2.rest3.config import ID2LABEL, PREDICTIONS_APPROACH1_PATH, MODEL_SAVE_DIR
from src.subtask2.rest3.model import JointSyllogismClassifier


def run_inference(
    model: JointSyllogismClassifier,
    loader: DataLoader,
    device: torch.device,
    premise_threshold: float = 0.5,
    top_k: int = 2,
) -> List[Dict]:
    """
    Run inference with the joint model and return a list of prediction dicts.

    Parameters
    ----------
    model               : trained JointSyllogismClassifier
    loader              : DataLoader (collate_fn_subtask2)
    device              : torch device
    premise_threshold   : sigmoid threshold for premise relevance
    top_k               : number of premise indices to return for valid items

    Returns
    -------
    list of dicts: [{"id": str, "validity": bool, "relevant_premises": [int, int]}]
    """
    model.eval()
    predictions: List[Dict] = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            premise_spans  = batch["premise_spans"].to(device)
            num_premises   = batch["num_premises"].to(device)
            ids            = batch["id"]

            out       = model(input_ids, attention_mask, premise_spans, num_premises)
            val_preds = out["logits"].argmax(dim=-1)               # (B,)
            prem_probs= torch.sigmoid(out["prem_logits"])          # (B, MAX_P)

            for b in range(input_ids.shape[0]):
                uid   = ids[b]
                valid = ID2LABEL[val_preds[b].item()]

                if valid:
                    n    = num_premises[b].item()
                    p    = prem_probs[b, :n].cpu()                 # (n,)
                    # Sort by probability descending, take top_k
                    sorted_idxs = p.argsort(descending=True).tolist()
                    selected    = sorted(sorted_idxs[:top_k])
                    rel_premises = selected
                else:
                    rel_premises = []

                predictions.append({
                    "id":               uid,
                    "validity":         valid,
                    "relevant_premises": rel_premises,
                })

    return predictions


def predict_and_save(
    model: JointSyllogismClassifier,
    loader: DataLoader,
    device: torch.device,
    output_path: str = PREDICTIONS_APPROACH1_PATH,
    premise_threshold: float = 0.5,
    top_k: int = 2,
) -> List[Dict]:
    predictions = run_inference(model, loader, device, premise_threshold, top_k)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    n_valid = sum(1 for p in predictions if p["validity"])
    n_with  = sum(1 for p in predictions if p["relevant_premises"])
    print(f"Approach 1 predictions saved → {output_path}")
    print(f"  {len(predictions)} items  |  valid={n_valid}  invalid={len(predictions)-n_valid}  with_premises={n_with}")
    return predictions
