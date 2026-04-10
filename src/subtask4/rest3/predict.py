"""
predict.py
----------
Inference module for JointSyllogismClassifier (Subtask 4).
"""

import json
import os
import sys
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.subtask4.rest3.config import ID2LABEL, PREDICTIONS_APPROACH1_PATH
from src.subtask4.rest3.model import JointSyllogismClassifier


def run_inference(model, loader, device, top_k=2):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            premise_spans  = batch["premise_spans"].to(device)
            num_premises   = batch["num_premises"].to(device)
            ids            = batch["id"]

            out        = model(input_ids, attention_mask, premise_spans, num_premises)
            val_preds  = out["logits"].argmax(dim=-1)
            prem_probs = torch.sigmoid(out["prem_logits"])

            for b in range(input_ids.shape[0]):
                uid   = ids[b]
                valid = ID2LABEL[val_preds[b].item()]
                if valid:
                    n = num_premises[b].item()
                    p = prem_probs[b, :n].cpu()
                    sorted_idxs = p.argsort(descending=True).tolist()
                    rel_premises = sorted(sorted_idxs[:top_k])
                else:
                    rel_premises = []
                predictions.append({
                    "id": uid,
                    "validity": valid,
                    "relevant_premises": rel_premises,
                })
    return predictions


def predict_and_save(model, loader, device,
                     output_path=PREDICTIONS_APPROACH1_PATH, top_k=2):
    predictions = run_inference(model, loader, device, top_k)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Predict-S4] {len(predictions)} predictions saved → {output_path}")
    return predictions
