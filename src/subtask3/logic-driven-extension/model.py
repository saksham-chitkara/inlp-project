"""
model.py
--------
LReasoner model for syllogistic reasoning with contrastive learning.

Architecture:
  XLM-RoBERTa-base encoder
    → [CLS] pooled output (768-dim)
    → Dropout → Linear(768, 2) classification head

Loss:
  L = L_CE + α * L_CL
  where L_CE = CrossEntropyLoss and L_CL = CosineEmbeddingLoss
"""

import os
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from types import SimpleNamespace
from typing import Optional, Dict


class LReasonerModel(nn.Module):
    """
    XLM-RoBERTa-based binary classifier with contrastive learning support.

    Parameters
    ----------
    cfg : SimpleNamespace
        Config with model_name, dropout_rate, alpha, cosine_margin, num_labels
    """

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(cfg.model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(hidden_size, cfg.num_labels),
        )

        self.alpha = cfg.alpha
        self.cosine_margin = cfg.cosine_margin
        self.cross_entropy = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=self.cosine_margin)

        self._class_weights: Optional[torch.Tensor] = None
        self._init_head()

    def _init_head(self):
        """Xavier init for classifier head."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def set_class_weights(self, weights: Optional[torch.Tensor]):
        """Set class weights for cross entropy loss."""
        self._class_weights = weights
        if weights is not None:
            self.cross_entropy = nn.CrossEntropyLoss(weight=weights)

    def forward(
        self,
        input_ids_plus: torch.Tensor,
        attention_mask_plus: torch.Tensor,
        input_ids_minus: Optional[torch.Tensor] = None,
        attention_mask_minus: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids_plus, attention_mask_plus: tokenized c_plus + conclusion
            input_ids_minus, attention_mask_minus: tokenized c_minus + conclusion (training only)
            labels: (B,) validity labels

        Returns:
            dict with 'logits', 'loss' (if labels given), 'cls_hidden'
        """
        # Forward pass on positive (true logic) pair
        outputs_plus = self.encoder(
            input_ids=input_ids_plus,
            attention_mask=attention_mask_plus,
        )
        pooled_plus = outputs_plus.pooler_output  # (B, H)
        logits = self.classifier(pooled_plus)  # (B, 2)

        result = {"logits": logits, "cls_hidden": pooled_plus}

        if labels is not None:
            # Classification loss
            loss_ce = self.cross_entropy(logits, labels)
            loss_cl = torch.tensor(0.0, device=logits.device)

            if input_ids_minus is not None:
                # Forward pass on negative (corrupted logic) pair
                outputs_minus = self.encoder(
                    input_ids=input_ids_minus,
                    attention_mask=attention_mask_minus,
                )
                pooled_minus = outputs_minus.pooler_output  # (B, H)

                # Contrastive loss: push apart true and corrupted embeddings
                target = torch.full(
                    (pooled_plus.size(0),), -1,
                    dtype=torch.float,
                    device=pooled_plus.device,
                )
                loss_cl = self.cosine_loss(pooled_plus, pooled_minus, target)

            loss = loss_ce + self.alpha * loss_cl
            result["loss"] = loss
            result["loss_ce"] = loss_ce
            result["loss_cl"] = loss_cl

        return result

    def save(self, path: str):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"  [Model] Saved to {path}")

    @classmethod
    def load(cls, path: str, cfg: SimpleNamespace) -> "LReasonerModel":
        """Load model from saved state dict."""
        model = cls(cfg)
        state_dict = torch.load(path, map_location="cpu")
        # Handle checkpoint format (may have extra keys from save_checkpoint_with_timestamp)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            
        # The class weights buffer is saved during training but isn't present
        # dynamically when initially instantiated for inference.
        state_dict.pop("cross_entropy.weight", None)
        
        model.load_state_dict(state_dict)
        print(f"  [Model] Loaded from {path}")
        return model
