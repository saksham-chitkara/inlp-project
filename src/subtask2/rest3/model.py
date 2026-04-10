"""
model.py
--------
JointSyllogismClassifier: extends the subtask1 XLM-RoBERTa classifier with a
second head for per-premise relevance prediction.

Architecture:
    XLM-RoBERTa-base encoder (shared)
         ├── [CLS] hidden state (B, H)
         │    └── dropout → Linear(H→2)   → validity logits     (B, 2)
         └── premise span mean-pools (B, MAX_P, H)
              └── dropout → Linear(H→1)   → relevance logits    (B, MAX_P)

Training objective:
    loss = α × CE(validity) + β × masked_BCE(per_premise_relevance)

The BCE loss is computed only over actual premise positions (not padding).
A pos_weight compensates for class imbalance (~14% positive premier labels).

Initialization:
    - Encoder + validity head loaded from subtask1 checkpoint
    - Premise head (Linear H→1) randomly initialized

Inference (Approach 1):
    1. Validity  = argmax(validity_logits)
    2. If valid: premise relevance = sigmoid(prem_logits) > 0.5
                 Return top-2 premise indices by probability
    3. If invalid: relevant_premises = []

Span pooling (vectorized, GPU-friendly):
    For each batch item i and premise j at token span [s, e):
        prem_h[i, j] = mean(last_hidden_state[i, s:e, :])
    Padded positions (span = [-1, -1]) are masked out.
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

# Import subtask1's model.py without circular import issues.
# The challenge: subtask1's model.py does `from config import ...` which
# must resolve to subtask1's config, not subtask2's.
# Solution: temporarily swap sys.modules['config'] and sys.path.
import importlib.util

# Step 1: Load subtask1's config as 'config' temporarily
_s1_config_spec = importlib.util.spec_from_file_location("config", os.path.join(S1_DIR, "config.py"))
_s1_config_mod = importlib.util.module_from_spec(_s1_config_spec)
_orig_config = sys.modules.get('config')
sys.modules['config'] = _s1_config_mod
_s1_config_spec.loader.exec_module(_s1_config_mod)

# Step 2: Load subtask1's model using the s1 config
_s1_model_spec = importlib.util.spec_from_file_location("s1_model", os.path.join(S1_DIR, "model.py"))
_s1_model_mod = importlib.util.module_from_spec(_s1_model_spec)
_s1_model_spec.loader.exec_module(_s1_model_mod)
SyllogismClassifier = _s1_model_mod.SyllogismClassifier

# Step 3: Restore config module to subtask2's
if _orig_config is not None:
    sys.modules['config'] = _orig_config
else:
    del sys.modules['config']

sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)
from src.subtask2.rest3.config import (
    MODEL_NAME, NUM_LABELS, MAX_PREMISES, DROPOUT_RATE,
    VALIDITY_LOSS_WEIGHT, PREMISE_LOSS_WEIGHT, PREMISE_POS_WEIGHT,
    S1_MODEL_SAVE_DIR,
)


class JointSyllogismClassifier(SyllogismClassifier):
    """
    XLM-RoBERTa with two prediction heads:
      1. Validity:   inherited Linear(768 → 2), Cross-Entropy loss
      2. Premises:   new Linear(768 → 1) per premise, BCE loss

    Parameters
    ----------
    max_premises : int
        Maximum number of premises per batch item. Controls padding.
    premise_pos_weight : float
        Positive weight for BCEWithLogitsLoss to handle class imbalance.
    validity_loss_weight : float
        α scaling factor for validity CE loss.
    premise_loss_weight : float
        β scaling factor for premise BCE loss.
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
            model_name=model_name,
            num_labels=num_labels,
            vocab_size_delta=vocab_size_delta,
            dropout_rate=dropout_rate,
        )
        hidden_size  = self.classifier.in_features  # 768 for xlm-roberta-base
        self.premise_head = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.premise_head.weight)
        nn.init.zeros_(self.premise_head.bias)

        self.max_premises          = max_premises
        self.premise_pos_weight    = premise_pos_weight
        self.validity_loss_weight  = validity_loss_weight
        self.premise_loss_weight   = premise_loss_weight

    # ─── Span Pooling ─────────────────────────────────────────────────────

    @staticmethod
    def _pool_premise_spans(
        last_hidden: torch.Tensor,     # (B, T, H)
        premise_spans: torch.Tensor,   # (B, MAX_P, 2)  — [start, end) inclusive; -1 = pad
    ) -> torch.Tensor:                 # (B, MAX_P, H)
        """
        Vectorised mean-pool of token hidden states over each premise's token span.
        Padded spans (-1, -1) produce zero vectors.
        """
        B, T, H   = last_hidden.shape
        MAX_P     = premise_spans.shape[1]
        dev       = last_hidden.device

        # Token position index: (1, 1, T)
        tok_idx   = torch.arange(T, device=dev).view(1, 1, T)

        # Span start/end: (B, MAX_P, 1)
        span_s = premise_spans[:, :, 0].unsqueeze(-1)   # (B, MAX_P, 1)
        span_e = premise_spans[:, :, 1].unsqueeze(-1)   # (B, MAX_P, 1)

        # in_span[b, p, t] = 1  iff  t ∈ [start_p, end_p)  and  span is valid
        in_span   = (tok_idx >= span_s) & (tok_idx < span_e)   # (B, MAX_P, T) bool
        valid_sp  = (span_s != -1)                               # (B, MAX_P, 1) bool
        in_span   = in_span & valid_sp                           # mask padding spans

        # Mean-pool: weighted sum / count
        in_float  = in_span.float()                              # (B, MAX_P, T)
        in_4d     = in_float.unsqueeze(-1)                       # (B, MAX_P, T, 1)
        last_4d   = last_hidden.unsqueeze(1).expand(-1, MAX_P, -1, -1)  # (B, MAX_P, T, H)

        sum_h     = (last_4d * in_4d).sum(dim=2)                # (B, MAX_P, H)
        count     = in_float.sum(dim=2, keepdim=True).clamp(min=1)      # (B, MAX_P, 1)
        prem_h    = sum_h / count                                # (B, MAX_P, H)

        return prem_h

    # ─── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                          # (B, T)
        attention_mask: torch.Tensor,                     # (B, T)
        premise_spans: Optional[torch.Tensor] = None,     # (B, MAX_P, 2)
        num_premises: Optional[torch.Tensor]  = None,     # (B,)
        labels: Optional[torch.Tensor]        = None,     # (B,) long
        premise_labels: Optional[torch.Tensor]= None,     # (B, MAX_P) float, -1 = pad
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
          'logits'        (B, 2)              validity logits
          'prem_logits'   (B, MAX_P)          per-premise relevance logits
          'cls_hidden'    (B, H)
          'loss'          scalar              total weighted loss (if labels or premise_labels given)
          'validity_loss' scalar
          'premise_loss'  scalar
        """
        # ── Encoder ──
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = enc_out.last_hidden_state    # (B, T, H)
        cls_hidden  = last_hidden[:, 0, :]         # (B, H)

        # ── Activation steering (inherited) ──
        if self._steering_alpha != 0.0 and self._steering_vectors:
            cls_hidden = self._apply_steering(cls_hidden, enc_out.hidden_states)

        # ── Validity head ──
        val_logits = self.classifier(self.dropout(cls_hidden))     # (B, 2)

        # ── Premise head ──
        prem_logits: Optional[torch.Tensor] = None
        if premise_spans is not None:
            prem_h    = self._pool_premise_spans(last_hidden, premise_spans)  # (B, MAX_P, H)
            prem_logits = self.premise_head(self.dropout(prem_h)).squeeze(-1) # (B, MAX_P)

        # ── Losses ──
        total_loss   = None
        val_loss_out = None
        prem_loss_out= None

        if labels is not None:
            cw  = self._class_weights
            val_criterion = nn.CrossEntropyLoss(
                weight=cw.to(val_logits.device) if cw is not None else None
            )
            val_loss_out = val_criterion(val_logits, labels)
            total_loss   = self.validity_loss_weight * val_loss_out

        if premise_labels is not None and prem_logits is not None:
            # Mask out padded positions (premise_labels == -1)
            MAX_P   = prem_logits.shape[1]
            dev     = prem_logits.device

            # Validity mask per premise slot: j < num_premises[i]
            if num_premises is not None:
                prem_mask = (
                    torch.arange(MAX_P, device=dev).unsqueeze(0)
                    < num_premises.unsqueeze(1)
                )                                                     # (B, MAX_P) bool
            else:
                prem_mask = (premise_labels >= 0)                     # (B, MAX_P) bool

            valid_mask = (premise_labels >= 0) & prem_mask

            if valid_mask.any():
                pw  = torch.tensor([self.premise_pos_weight], device=dev)
                bce = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
                # Clamp targets to [0,1] (guard against -1 leaking through)
                targets = premise_labels.float().clamp(0, 1)
                raw_bce = bce(prem_logits, targets)                   # (B, MAX_P)
                prem_loss_out = (raw_bce * valid_mask.float()).sum() / valid_mask.float().sum()

                if total_loss is None:
                    total_loss  = self.premise_loss_weight * prem_loss_out
                else:
                    total_loss  = total_loss + self.premise_loss_weight * prem_loss_out

        result: Dict[str, torch.Tensor] = {
            "logits":     val_logits,
            "cls_hidden": cls_hidden,
        }
        if prem_logits is not None:
            result["prem_logits"] = prem_logits
        if total_loss is not None:
            result["loss"]          = total_loss
        if val_loss_out is not None:
            result["validity_loss"] = val_loss_out
        if prem_loss_out is not None:
            result["premise_loss"]  = prem_loss_out

        return result

    # ─── Inference helper ─────────────────────────────────────────────────

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        premise_spans: torch.Tensor,
        num_premises: torch.Tensor,
        premise_threshold: float = 0.5,
        top_k: int = 2,
    ) -> Tuple[List[bool], List[List[int]]]:
        """
        Run inference and return:
          validities:   list of bool
          rel_premises: list of lists of int (empty for invalid items)

        For valid items: return top-k premise indices by relevance prob,
        but only those exceeding threshold. Falls back to top-k if fewer
        than top_k exceed threshold.
        """
        out     = self.forward(input_ids, attention_mask, premise_spans, num_premises)
        val_pred= out["logits"].argmax(dim=-1)      # (B,)
        probs   = torch.sigmoid(out["prem_logits"]) # (B, MAX_P)

        from src.subtask2.rest3.config import ID2LABEL
        validities: List[bool]       = []
        rel_premises: List[List[int]]= []

        for b in range(input_ids.shape[0]):
            valid = ID2LABEL[val_pred[b].item()]
            validities.append(valid)

            if valid:
                n = num_premises[b].item()
                p = probs[b, :n]                          # (n,)
                # top-k by probability
                sorted_idxs = p.argsort(descending=True).tolist()
                # require at least top_k results
                selected = sorted_idxs[:top_k]
                sorted_selected = sorted(selected)
                rel_premises.append(sorted_selected)
            else:
                rel_premises.append([])

        return validities, rel_premises


# ─── Checkpoint Loading ───────────────────────────────────────────────────────

def load_joint_model(checkpoint_dir: str = S1_MODEL_SAVE_DIR) -> JointSyllogismClassifier:
    """
    Load JointSyllogismClassifier, initializing encoder + validity head from
    the subtask1 checkpoint.  The premise head is randomly initialized.
    """
    import src.subtask2.rest3.config as s2cfg

    model = JointSyllogismClassifier()

    weights_path = os.path.join(checkpoint_dir, "model_weights.pt")
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        # Load common keys (encoder + validity head), skip premise_head
        missing, unexpected = model.load_state_dict(state, strict=False)
        s1_keys  = [k for k in missing  if "premise_head" not in k]
        new_keys = [k for k in missing  if "premise_head"     in k]
        print(f"[JointModel] Loaded from: {weights_path}")
        print(f"  Initialized from checkpoint: {len(state) - len(unexpected)} params")
        print(f"  Randomly initialized (premise_head): {new_keys}")
        if s1_keys:
            print(f"  WARNING: missing non-premise keys: {s1_keys[:5]}")
    else:
        print(f"[JointModel] No checkpoint found at {weights_path}, using random init.")

    return model
