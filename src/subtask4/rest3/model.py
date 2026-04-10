"""
model.py
--------
JointSyllogismClassifier for Subtask 4 (multilingual multi-premise).

Same architecture as Subtask 2:
  XLM-RoBERTa-base encoder (shared)
       ├── [CLS] → validity head   (H → 2)
       └── premise spans → premise head (H → 1)

XLM-RoBERTa handles multilingual input natively.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
S1_DIR     = os.path.join(SCRIPT_DIR, "..", "subtask1")

# Load subtask1's SyllogismClassifier without polluting config
import importlib.util

_s1_config_spec = importlib.util.spec_from_file_location("config", os.path.join(S1_DIR, "config.py"))
_s1_config_mod = importlib.util.module_from_spec(_s1_config_spec)
_orig_config = sys.modules.get('config')
sys.modules['config'] = _s1_config_mod
_s1_config_spec.loader.exec_module(_s1_config_mod)

_s1_model_spec = importlib.util.spec_from_file_location("s1_model", os.path.join(S1_DIR, "model.py"))
_s1_model_mod = importlib.util.module_from_spec(_s1_model_spec)
_s1_model_spec.loader.exec_module(_s1_model_mod)
SyllogismClassifier = _s1_model_mod.SyllogismClassifier

if _orig_config is not None:
    sys.modules['config'] = _orig_config
else:
    del sys.modules['config']

sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)
from src.subtask4.rest3.config import (
    MODEL_NAME, NUM_LABELS, MAX_PREMISES, DROPOUT_RATE,
    VALIDITY_LOSS_WEIGHT, PREMISE_LOSS_WEIGHT, PREMISE_POS_WEIGHT,
    S2_MODEL_SAVE_DIR,
)


class JointSyllogismClassifier(SyllogismClassifier):
    """
    XLM-RoBERTa with two prediction heads:
      1. Validity:   inherited Linear(768 → 2), Cross-Entropy loss
      2. Premises:   new Linear(768 → 1) per premise, BCE loss
    """

    def __init__(
        self,
        model_name: str           = MODEL_NAME,
        num_labels: int           = NUM_LABELS,
        vocab_size_delta: int     = 0,
        dropout_rate: float       = DROPOUT_RATE,
        max_premises: int         = MAX_PREMISES,
        premise_pos_weight: float = PREMISE_POS_WEIGHT,
        validity_loss_weight: float = VALIDITY_LOSS_WEIGHT,
        premise_loss_weight: float  = PREMISE_LOSS_WEIGHT,
    ):
        super().__init__(
            model_name=model_name, num_labels=num_labels,
            vocab_size_delta=vocab_size_delta, dropout_rate=dropout_rate,
        )
        hidden_size = self.classifier.in_features
        self.premise_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.premise_head.weight)
        nn.init.zeros_(self.premise_head.bias)

        self.max_premises         = max_premises
        self.premise_pos_weight   = premise_pos_weight
        self.validity_loss_weight = validity_loss_weight
        self.premise_loss_weight  = premise_loss_weight

    @staticmethod
    def _pool_premise_spans(last_hidden, premise_spans):
        B, T, H = last_hidden.shape
        MAX_P   = premise_spans.shape[1]
        dev     = last_hidden.device

        tok_idx = torch.arange(T, device=dev).view(1, 1, T)
        span_s  = premise_spans[:, :, 0].unsqueeze(-1)
        span_e  = premise_spans[:, :, 1].unsqueeze(-1)

        in_span  = (tok_idx >= span_s) & (tok_idx < span_e)
        valid_sp = (span_s != -1)
        in_span  = in_span & valid_sp

        in_float = in_span.float()
        in_4d    = in_float.unsqueeze(-1)
        last_4d  = last_hidden.unsqueeze(1).expand(-1, MAX_P, -1, -1)

        sum_h = (last_4d * in_4d).sum(dim=2)
        count = in_float.sum(dim=2, keepdim=True).clamp(min=1)
        return sum_h / count

    def forward(self, input_ids, attention_mask,
                premise_spans=None, num_premises=None,
                labels=None, premise_labels=None):
        enc_out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=True)
        last_hidden = enc_out.last_hidden_state
        cls_hidden  = last_hidden[:, 0, :]

        if self._steering_alpha != 0.0 and self._steering_vectors:
            cls_hidden = self._apply_steering(cls_hidden, enc_out.hidden_states)

        val_logits = self.classifier(self.dropout(cls_hidden))

        prem_logits = None
        if premise_spans is not None:
            prem_h      = self._pool_premise_spans(last_hidden, premise_spans)
            prem_logits = self.premise_head(self.dropout(prem_h)).squeeze(-1)

        total_loss = val_loss_out = prem_loss_out = None

        if labels is not None:
            cw = self._class_weights
            val_criterion = nn.CrossEntropyLoss(
                weight=cw.to(val_logits.device) if cw is not None else None)
            val_loss_out = val_criterion(val_logits, labels)
            total_loss   = self.validity_loss_weight * val_loss_out

        if premise_labels is not None and prem_logits is not None:
            MAX_P = prem_logits.shape[1]
            dev   = prem_logits.device
            if num_premises is not None:
                prem_mask = torch.arange(MAX_P, device=dev).unsqueeze(0) < num_premises.unsqueeze(1)
            else:
                prem_mask = (premise_labels >= 0)
            valid_mask = (premise_labels >= 0) & prem_mask
            if valid_mask.any():
                pw  = torch.tensor([self.premise_pos_weight], device=dev)
                bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
                targets = premise_labels.float().clamp(0, 1)
                raw_bce = bce(prem_logits, targets)
                prem_loss_out = (raw_bce * valid_mask.float()).sum() / valid_mask.float().sum()
                if total_loss is None:
                    total_loss = self.premise_loss_weight * prem_loss_out
                else:
                    total_loss += self.premise_loss_weight * prem_loss_out

        # Always return all keys so DataParallel.gather() works
        zero = torch.tensor(0.0, device=val_logits.device)
        result = {
            "logits":        val_logits,
            "cls_hidden":    cls_hidden,
            "prem_logits":   prem_logits if prem_logits is not None else torch.zeros(val_logits.size(0), self.max_premises, device=val_logits.device),
            "loss":          total_loss if total_loss is not None else zero,
            "validity_loss": val_loss_out if val_loss_out is not None else zero,
            "premise_loss":  prem_loss_out if prem_loss_out is not None else zero,
        }
        return result

    def predict(self, input_ids, attention_mask, premise_spans,
                num_premises, premise_threshold=0.5, top_k=2):
        out      = self.forward(input_ids, attention_mask, premise_spans, num_premises)
        val_pred = out["logits"].argmax(dim=-1)
        probs    = torch.sigmoid(out["prem_logits"])

        from src.subtask4.rest3.config import ID2LABEL
        validities   = []
        rel_premises = []
        for b in range(input_ids.shape[0]):
            valid = ID2LABEL[val_pred[b].item()]
            validities.append(valid)
            if valid:
                n = num_premises[b].item()
                p = probs[b, :n]
                sorted_idxs = p.argsort(descending=True).tolist()
                rel_premises.append(sorted(sorted_idxs[:top_k]))
            else:
                rel_premises.append([])
        return validities, rel_premises


def load_joint_model(checkpoint_dir: str = S2_MODEL_SAVE_DIR) -> JointSyllogismClassifier:
    """Load JointSyllogismClassifier from subtask2 checkpoint."""
    model = JointSyllogismClassifier()
    weights_path = os.path.join(checkpoint_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[JointModel-S4] Loaded from: {weights_path}")
        print(f"  Initialized: {len(state) - len(unexpected)} params")
        if missing:
            print(f"  Missing keys: {missing[:5]}")
    else:
        print(f"[JointModel-S4] No checkpoint at {weights_path}, using random init.")
    return model
