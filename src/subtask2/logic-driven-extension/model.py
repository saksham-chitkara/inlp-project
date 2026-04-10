"""
model.py
--------
Multi-task LReasoner model for Subtask 2:
  1. Validity classification (valid/invalid)
  2. Premise identification (which premises are relevant)

Architecture:
  XLM-RoBERTa-base encoder (shared)
    → [CLS] pooled output (768-dim)
      → Dropout → Linear(768, 2) validity classification head
    → Per-premise [CLS] outputs (from separate forward passes, no grad)
      → Dropout → Linear(768, 1) premise relevance scorer

Loss:
  L = L_CE + α * L_CL + β * L_BCE
  where L_CE = CrossEntropyLoss (validity)
        L_CL = CosineEmbeddingLoss (contrastive)
        L_BCE = BCEWithLogitsLoss (premise selection, valid samples only)

Memory optimization:
  - Premise encoding uses torch.no_grad() on the shared encoder to avoid
    storing activations for all premises. Only the premise_scorer head
    is trained with gradients on premise loss.
  - Premises are encoded in small chunks to avoid OOM.
"""

import os
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from types import SimpleNamespace
from typing import Optional, Dict


class LReasonerPremiseModel(nn.Module):
    """
    XLM-RoBERTa-based multi-task model:
      - Binary validity classification with contrastive learning
      - Per-premise relevance scoring

    Parameters
    ----------
    cfg : SimpleNamespace
        Config with model_name, dropout_rate, alpha, cosine_margin,
        num_labels, beta, premise_threshold
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(cfg.model_name)
        hidden_size = self.encoder.config.hidden_size

        # ─── Validity Classification Head ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden_size, cfg.num_labels),
        )

        # ─── Premise Selection Head ────────────────────────────────────
        # A small MLP that takes [CLS] from premise encoding and scores it
        self.premise_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden_size // 4, 1),
        )

        # Loss weights
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.cosine_margin = cfg.cosine_margin
        self.premise_threshold = cfg.premise_threshold

        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=self.cosine_margin)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

        self._class_weights: Optional[torch.Tensor] = None
        self._init_heads()

    def _init_heads(self):
        """Xavier init for classification and premise heads."""
        for head in [self.classifier, self.premise_scorer]:
            for module in head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def set_class_weights(self, weights: Optional[torch.Tensor]):
        """Set class weights for cross entropy loss."""
        self._class_weights = weights
        if weights is not None:
            self.cross_entropy = nn.CrossEntropyLoss(weight=weights)

    def _encode_premises(
        self,
        premise_input_ids: torch.Tensor,
        premise_attention_mask: torch.Tensor,
        premise_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each premise independently through the shared encoder.

        Uses torch.no_grad() on encoder to save memory — only the
        premise_scorer MLP is trained with gradients from premise loss.

        Args:
            premise_input_ids:     (B, max_premises, seq_len)
            premise_attention_mask: (B, max_premises, seq_len)
            premise_mask:          (B, max_premises)

        Returns:
            premise_logits: (B, max_premises) — raw logits for each premise
        """
        B, P, S = premise_input_ids.shape
        device = premise_input_ids.device

        # Flatten to (B*P, S) for batch encoding
        flat_input_ids = premise_input_ids.view(B * P, S)
        flat_attention_mask = premise_attention_mask.view(B * P, S)

        # Only encode real premises (premise_mask == 1) to save compute
        flat_mask = premise_mask.view(B * P)
        real_indices = flat_mask.nonzero(as_tuple=True)[0]

        if real_indices.numel() == 0:
            return torch.zeros(B, P, device=device)

        real_input_ids = flat_input_ids[real_indices]
        real_attention_mask = flat_attention_mask[real_indices]

        # Encode in small chunks WITHOUT gradient on the encoder
        # to dramatically reduce memory usage
        chunk_size = 8  # small chunks to avoid OOM on MPS
        all_pooled = []

        for start in range(0, real_indices.numel(), chunk_size):
            end = min(start + chunk_size, real_indices.numel())
            with torch.no_grad():
                chunk_output = self.encoder(
                    input_ids=real_input_ids[start:end],
                    attention_mask=real_attention_mask[start:end],
                )
                chunk_pooled = chunk_output.pooler_output.detach()  # (chunk, H)
            all_pooled.append(chunk_pooled)

            # Free memory
            del chunk_output
            if device.type == "mps":
                torch.mps.empty_cache()

        # Concatenate all pooled outputs and enable grad for scorer
        all_pooled = torch.cat(all_pooled, dim=0)  # (N_real, H)
        all_pooled.requires_grad_(True)  # enable grad for premise_scorer

        # Score through premise MLP
        all_logits = self.premise_scorer(all_pooled).squeeze(-1)  # (N_real,)

        # Scatter back into full (B*P,) tensor
        full_logits = torch.full((B * P,), -1e9, device=device)
        full_logits[real_indices] = all_logits

        return full_logits.view(B, P)

    def forward(
        self,
        input_ids_plus: torch.Tensor,
        attention_mask_plus: torch.Tensor,
        input_ids_minus: Optional[torch.Tensor] = None,
        attention_mask_minus: Optional[torch.Tensor] = None,
        premise_input_ids: Optional[torch.Tensor] = None,
        premise_attention_mask: Optional[torch.Tensor] = None,
        premise_mask: Optional[torch.Tensor] = None,
        premise_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids_plus, attention_mask_plus: tokenized c_plus + conclusion
            input_ids_minus, attention_mask_minus: tokenized c_minus + conclusion
            premise_input_ids:     (B, max_premises, seq_len) per-premise encodings
            premise_attention_mask: (B, max_premises, seq_len)
            premise_mask:          (B, max_premises)
            premise_labels:        (B, max_premises) binary labels
            labels: (B,) validity labels

        Returns:
            dict with 'logits', 'premise_logits', 'loss', 'cls_hidden'
        """
        # Forward pass on positive (true logic) pair for validity
        outputs_plus = self.encoder(
            input_ids=input_ids_plus,
            attention_mask=attention_mask_plus,
        )
        pooled_plus = outputs_plus.pooler_output  # (B, H)
        logits = self.classifier(pooled_plus)  # (B, 2)

        result = {"logits": logits, "cls_hidden": pooled_plus}

        # ─── Premise Selection ─────────────────────────────────────────
        if premise_input_ids is not None and premise_mask is not None:
            premise_logits = self._encode_premises(
                premise_input_ids, premise_attention_mask, premise_mask,
            )
            result["premise_logits"] = premise_logits

        # ─── Loss Computation ──────────────────────────────────────────
        if labels is not None:
            # 1. Classification loss (CE)
            loss_ce = self.cross_entropy(logits, labels)
            loss_cl = torch.tensor(0.0, device=logits.device)
            loss_premise = torch.tensor(0.0, device=logits.device)

            # 2. Contrastive loss
            if input_ids_minus is not None:
                outputs_minus = self.encoder(
                    input_ids=input_ids_minus,
                    attention_mask=attention_mask_minus,
                )
                pooled_minus = outputs_minus.pooler_output
                target = torch.full(
                    (pooled_plus.size(0),), -1,
                    dtype=torch.float, device=pooled_plus.device,
                )
                loss_cl = self.cosine_loss(pooled_plus, pooled_minus, target)

            # 3. Premise selection loss (BCE, only for valid samples)
            if (premise_labels is not None and "premise_logits" in result
                    and premise_mask is not None):
                valid_mask = (labels == 1).float()  # (B,)
                if valid_mask.sum() > 0:
                    # (B, P) BCE loss, masked by real premises and valid samples
                    bce_raw = self.bce_loss(result["premise_logits"], premise_labels)
                    # Mask out padding premises
                    bce_masked = bce_raw * premise_mask
                    # Average per-sample, then weight by valid_mask
                    per_sample = bce_masked.sum(dim=1) / (premise_mask.sum(dim=1) + 1e-8)
                    loss_premise = (per_sample * valid_mask).sum() / (valid_mask.sum() + 1e-8)

            loss = loss_ce + self.alpha * loss_cl + self.beta * loss_premise
            result["loss"] = loss
            result["loss_ce"] = loss_ce
            result["loss_cl"] = loss_cl
            result["loss_premise"] = loss_premise

        return result

    def predict_premises(
        self,
        premise_logits: torch.Tensor,
        premise_mask: torch.Tensor,
        validity_preds: torch.Tensor,
    ) -> list:
        """
        Convert premise logits to predicted premise indices.

        For invalid predictions, returns [].
        For valid predictions, returns indices where sigmoid > threshold.
        If no premises above threshold, returns top-2.

        Args:
            premise_logits: (B, max_premises)
            premise_mask:   (B, max_premises)
            validity_preds: (B,) predicted validity (0 or 1)

        Returns:
            List of lists of premise indices
        """
        probs = torch.sigmoid(premise_logits)  # (B, P)
        batch_premises = []

        for i in range(premise_logits.size(0)):
            if validity_preds[i] == 0:
                # Invalid → no relevant premises
                batch_premises.append([])
            else:
                # Valid → select premises above threshold
                mask_i = premise_mask[i].bool()
                probs_i = probs[i]
                selected = []
                for j in range(premise_logits.size(1)):
                    if mask_i[j] and probs_i[j] > self.premise_threshold:
                        selected.append(j)
                # If none selected but predicted valid, take top-2
                if not selected:
                    masked_probs = probs_i.clone()
                    masked_probs[~mask_i] = -1.0
                    n_real = mask_i.sum().item()
                    k = min(2, n_real)
                    if k > 0:
                        topk = torch.topk(masked_probs, k)
                        selected = topk.indices.tolist()
                batch_premises.append(sorted(selected))

        return batch_premises

    def save(self, path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"  [Model] Saved to {path}")

    @classmethod
    def load(cls, path: str, cfg: SimpleNamespace) -> "LReasonerPremiseModel":
        """Load model from saved state dict."""
        model = cls(cfg)
        state_dict = torch.load(path, map_location="cpu")
        # Handle checkpoint format
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # The class weights buffer may not be present at inference time
        state_dict.pop("cross_entropy.weight", None)

        model.load_state_dict(state_dict)
        print(f"  [Model] Loaded from {path}")
        return model
