"""
model.py
--------
XLM-RoBERTa-based binary classifier for syllogistic validity prediction.

Architecture:
  XLM-RoBERTa-base
      -> [CLS] representation (768-dim)
              -> Dropout(0.1)
              -> Linear(768 -> 2)   <- valid / invalid

Activation Hooks:
  During inference, forward hooks can optionally capture and modify hidden-state
  tensors from chosen layers. This supports the Contrastive Activation Steering
  (CAA / K-CAST) approach adapted from Valentino et al. (2025).

  Note: The original paper applies activation steering to decoder-only models
  (Llama, Gemma, Qwen). We adapt the principle to encoder-only XLM-RoBERTa
  by steering the [CLS] token hidden representations, which aggregate
  sequence-level information for classification.

Key design choices from proposal:
  - XLM-RoBERTa: natively multilingual -> transfers to subtasks 3 & 4 with no
    architecture change.
  - The [CLS] token is used as the sequence representation for classification
    (standard for encoder-only models).
  - Dropout rate 0.1 is the XLM-RoBERTa default; increasing it slightly hurts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Callable

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import MODEL_NAME, NUM_LABELS, STEERING_LAYERS, HF_CACHE_DIR, USE_GRADIENT_CHECKPOINTING, DROPOUT_RATE


class SyllogismClassifier(nn.Module):
    """
    Binary classifier on top of XLM-RoBERTa.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: xlm-roberta-base).
    num_labels : int
        Number of output classes (2: invalid / valid).
    vocab_size_delta : int
        Additional token count added via tokenizer.add_special_tokens().
        The encoder embedding matrix is resized accordingly.
        (Currently 0: we use </s> as separator, which is already in the vocab.)
    dropout_rate : float
        Dropout probability before the classifier head.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = NUM_LABELS,
        vocab_size_delta: int = 0,
        dropout_rate: float = DROPOUT_RATE,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
        self.encoder = AutoModel.from_pretrained(model_name, config=config, cache_dir=HF_CACHE_DIR)

        # Gradient checkpointing: trade compute for memory on large models
        if USE_GRADIENT_CHECKPOINTING:
            self.encoder.gradient_checkpointing_enable()

        # Resize embeddings if we added special tokens (currently unused since
        # we use </s> as separator, but kept for future extensibility)
        if vocab_size_delta > 0:
            self.encoder.resize_token_embeddings(config.vocab_size + vocab_size_delta)

        hidden_size = config.hidden_size  # 768 for xlm-roberta-base
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # --- Activation Hook Infrastructure ---
        # Mutable hook storage: layer_idx -> hooked hidden states (B, T, H)
        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []
        self._steering_vectors: Dict[int, torch.Tensor] = {}   # layer -> vec (H,)
        self._steering_alpha: float = 0.0   # 0 disables steering
        self._class_weights: Optional[torch.Tensor] = None  # set externally for DataParallel compat

        self._num_labels = num_labels

        # Initialise weights for newly added modules
        self._init_head()

    def _init_head(self):
        """Xavier-uniform init for the linear head."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # --- Forward ---

    def set_class_weights(self, class_weights: Optional[torch.Tensor]):
        """Store class weights as attribute (DataParallel-safe)."""
        self._class_weights = class_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids, attention_mask : (B, T) tensors from tokenizer
        labels       : (B,) long tensor (0/1); if provided, loss is computed

        Class weights are read from self._class_weights (set via set_class_weights)
        to avoid DataParallel scatter issues with non-batch kwargs.

        Returns
        -------
        dict with keys: 'loss' (optional), 'logits' (B, 2), 'cls_hidden' (B, H)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,   # needed for activation steering
        )

        # [CLS] token representation from the last hidden state
        cls_hidden = outputs.last_hidden_state[:, 0, :]   # (B, H)

        # --- Apply activation steering if configured ---
        if self._steering_alpha != 0.0 and self._steering_vectors:
            cls_hidden = self._apply_steering(cls_hidden, outputs.hidden_states)

        logits = self.classifier(self.dropout(cls_hidden))  # (B, 2)

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "cls_hidden": cls_hidden,
        }

        if labels is not None:
            cw = self._class_weights
            criterion = nn.CrossEntropyLoss(
                weight=cw.to(logits.device) if cw is not None else None
            )
            result["loss"] = criterion(logits, labels)

        return result

    # --- Activation Extraction ---

    def get_layer_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: List[int] = STEERING_LAYERS,
    ) -> Dict[int, torch.Tensor]:
        """
        Run a forward pass and return the [CLS] hidden state at each requested layer.

        Returns: {layer_idx: tensor(B, H)}
        Used by activation_steering.py to build CAA steering vectors.
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # hidden_states is a tuple of (num_layers+1) tensors; index 0 = embedding
        layer_hidden = {}
        for layer_idx in layers:
            # layer_idx 0-11 for xlm-roberta-base; hidden_states[i+1] = layer i
            h = outputs.hidden_states[layer_idx + 1][:, 0, :]  # (B, H)
            layer_hidden[layer_idx] = h.detach().cpu()
        return layer_hidden

    # --- Steering Logic ---

    def set_steering_vectors(
        self,
        steering_vectors: Dict[int, torch.Tensor],
        alpha: float = 1.0,
    ):
        """
        Load precomputed CAA steering vector(s) into the model.

        Parameters
        ----------
        steering_vectors : {layer_idx: tensor(H,)}
        alpha            : scaling coefficient; positive = steer toward target
        """
        self._steering_vectors = {
            k: v.to(next(self.parameters()).device)
            for k, v in steering_vectors.items()
        }
        self._steering_alpha = alpha

    def _apply_steering(
        self,
        cls_hidden: torch.Tensor,
        all_hidden_states: tuple,
    ) -> torch.Tensor:
        """
        Apply the activation steering vectors to the [CLS] hidden state.

        For simplicity we accumulate the effect of each steered layer's vector
        onto the final [CLS] representation (linearly additive modification).
        This approximates the CAA-at-inference scheme from Valentino et al. (2025).
        """
        device = cls_hidden.device
        for layer_idx, vec in self._steering_vectors.items():
            vec = vec.to(device)
            cls_hidden = cls_hidden + self._steering_alpha * vec.unsqueeze(0)
        return cls_hidden

    def disable_steering(self):
        """Turn off activation steering (set alpha to 0)."""
        self._steering_alpha = 0.0

    # --- Convenience ---

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: str):
        """Save model weights and encoder to a directory."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pt"))
        self.encoder.save_pretrained(os.path.join(path, "encoder"))

    @classmethod
    def load(cls, path: str, model_name: str = MODEL_NAME, **kwargs) -> "SyllogismClassifier":
        """Load model weights from a saved checkpoint."""
        model = cls(model_name=model_name, **kwargs)
        weight_path = os.path.join(path, "model_weights.pt")
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu"), strict=False
        )
        return model
