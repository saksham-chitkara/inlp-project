"""
activation_steering.py
----------------------
Contrastive Activation Steering (CAA) and K-CAST implementation.

Based on: Valentino, Kim, Dalal, Zhao & Freitas (2025)
"Mitigating Content Effects on Reasoning in Language Models through
Fine-Grained Activation Steering" — arXiv:2505.12189.

Core intuition (from the paper):
  LLMs encode plausibility and formal validity in different layers/directions
  of their residual stream. By linearly manipulating hidden representations at
  inference time, we can "steer" the model away from content-biased judgements
  and toward logic-structure-based reasoning.

Adapted for encoder-only XLM-RoBERTa:
  The original paper targets decoder-only LLMs (Llama, Gemma, Qwen). We adapt
  the same principle to the encoder [CLS] representations, applying additive
  steering to the final hidden states at inference time.

D+ / D- construction (BEHAVIOUR-BASED, per the paper):
  D+ = examples where the fine-tuned model predicts *correctly*
       (model overcomes content bias and reasons from structure)
  D- = examples where the fine-tuned model predicts *incorrectly*
       (model is biased by plausibility → wrong answer)
  This is different from a label-based split. The steering vector captures the
  *direction in activation space* from "content-biased reasoning" to "correct
  structure-based reasoning".

Three steering strategies implemented:
─────────────────────────────────────
1. CAA (Contrastive Activation Addition):
   • Compute mean activations for D+ (correct predictions) and D- (errors).
   • Steering vector δ = mean(φ(D+)) − mean(φ(D−))
   • At inference: φ̃(x) = φ(x) + α · δ
   • Best α found by grid search on validation set.

2. CAST (Conditional Activation Steering):
   • Instead of fixed α, determine the sign conditionally:
     if the input "looks more like" D+ → use positive α
     if it "looks more like" D− → use negative α
   • Condition: similarity of φ(x) to the mean D+/D− vectors.

3. K-CAST (k-Nearest-Neighbour Conditional Steering):
   • Store individual training activation vectors (not just means).
   • At inference: find k nearest neighbours from training set.
   • Majority vote of their correctness determines steering direction.
   • Fine-grained, per-example steering modulation.
   • Most effective for models that are "unresponsive" to static CAA.

Usage:
    # After fine-tuning:
    steerer = ActivationSteerer(model, device)
    steerer.compute_steering_vectors(val_loader, layers=[9, 10, 11])
    steerer.grid_search_alpha(val_loader, layers=[9, 10, 11])
    steerer.save(STEERING_VECTORS_PATH)

    # At inference:
    steerer.load(STEERING_VECTORS_PATH)
    steerer.apply_kcast(model, k=10, alpha=best_alpha)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

sys.path.insert(0, os.path.dirname(__file__))
from config import STEERING_LAYERS, STEERING_ALPHA, STEERING_KNN, STEERING_VECTORS_PATH
from model import SyllogismClassifier
from train import compute_val_metrics


class ActivationSteerer:
    """
    Computes, stores, and applies activation steering vectors for
    content-effect mitigation in SyllogismClassifier.
    """

    def __init__(
        self,
        model: SyllogismClassifier,
        device: torch.device,
    ):
        self.model = model
        self.device = device

        # CAA steering vectors: layer → tensor(H,)
        self.steering_vectors: Dict[int, torch.Tensor] = {}

        # K-CAST: stores (activation, label) pairs per layer
        # layer → {"activations": tensor(N, H), "labels": tensor(N,)}
        self.kcast_store: Dict[int, Dict] = {}

        # Best hyperparameters found by grid search
        self.best_alpha: float = STEERING_ALPHA
        self.best_layer: int = STEERING_LAYERS[-1]

    # ─── Step 1: Collect Activations ─────────────────────────────────────────

    def _collect_activations(
        self,
        loader: DataLoader,
        layers: List[int],
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Run the fine-tuned model over `loader` and collect [CLS] hidden states
        at the specified layers, along with ground-truth labels, plausibility
        flags, and model predictions.

        The model predictions are needed for behaviour-based D+/D- construction:
        D+ = examples predicted correctly, D- = examples predicted incorrectly.

        Returns:
          {layer_idx: {"activations": (N, H), "labels": (N,), "plaus": (N,),
                       "preds": (N,), "correct": (N,)}}
        """
        self.model.eval()
        layer_data: Dict[int, Dict] = {
            l: {"acts": [], "labels": [], "plaus": [], "preds": []}
            for l in layers
        }

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"]          # (B,) on CPU
                plaus  = batch["plausibility"]    # (B,) on CPU

                # Get model predictions for behaviour-based D+/D-
                out = self.model(input_ids, attention_mask)
                preds = out["logits"].argmax(dim=-1).cpu()  # (B,)

                layer_hidden = self.model.get_layer_hidden_states(
                    input_ids, attention_mask, layers=layers
                )
                for layer_idx in layers:
                    h = layer_hidden[layer_idx]  # (B, H) on CPU
                    layer_data[layer_idx]["acts"].append(h)
                    layer_data[layer_idx]["labels"].append(labels)
                    layer_data[layer_idx]["plaus"].append(plaus)
                    layer_data[layer_idx]["preds"].append(preds)

        # Concatenate
        result = {}
        for layer_idx in layers:
            labels_cat = torch.cat(layer_data[layer_idx]["labels"], dim=0)
            preds_cat  = torch.cat(layer_data[layer_idx]["preds"],  dim=0)
            result[layer_idx] = {
                "activations": torch.cat(layer_data[layer_idx]["acts"], dim=0),   # (N, H)
                "labels":      labels_cat,                                         # (N,)
                "plaus":       torch.cat(layer_data[layer_idx]["plaus"],  dim=0), # (N,)
                "preds":       preds_cat,                                          # (N,)
                "correct":     (preds_cat == labels_cat).long(),                   # (N,) 1=correct, 0=wrong
            }
        return result

    # ─── Step 2: Compute CAA Steering Vectors ────────────────────────────────

    def compute_steering_vectors(
        self,
        train_loader: DataLoader,
        layers: List[int] = STEERING_LAYERS,
    ) -> None:
        """
        Compute contrastive activation vectors for content-effect mitigation.

        CONTENT-BASED construction (adapted from Valentino et al., 2025):
          Content-aligned    = PV (plausible+valid) or II (implausible+invalid)
                               → plausibility cue agrees with validity label
          Content-conflicting = IV (implausible+valid) or PI (plausible+invalid)
                               → plausibility cue disagrees with validity label

        δ_layer = mean(φ(conflicting)) − mean(φ(aligned))

        This captures the direction from "content-biased reasoning" toward
        "content-independent, structure-based reasoning".  Steering with
        positive α pushes representations toward the pattern seen when
        the model must ignore content cues.

        Also stores individual activations for K-CAST.
        """
        print("[Steering] Collecting activations from training set …")
        layer_data = self._collect_activations(train_loader, layers)

        for layer_idx in layers:
            acts    = layer_data[layer_idx]["activations"]  # (N, H)
            correct = layer_data[layer_idx]["correct"]       # (N,) 1=correct, 0=wrong

            # CONTENT-BASED grouping (more balanced than behavior-based):
            # Content-aligned    (PV + II): plausibility agrees with validity
            # Content-conflicting (IV + PI): plausibility disagrees with validity
            labels = layer_data[layer_idx]["labels"]
            plaus  = layer_data[layer_idx]["plaus"]
            mask_pos = ((labels == 1) & (plaus == 1)) | ((labels == 0) & (plaus == 0))  # aligned
            mask_neg = ((labels == 1) & (plaus == 0)) | ((labels == 0) & (plaus == 1))  # conflicting

            n_pos = mask_pos.sum().item()
            n_neg = mask_neg.sum().item()
            print(f"  Layer {layer_idx}: aligned={n_pos}, "
                  f"conflicting={n_neg}")

            if n_pos == 0 or n_neg == 0:
                print(f"  \u26a0 Skipping layer {layer_idx}: empty D+ or D-.")
                continue

            mean_pos = acts[mask_pos].mean(dim=0)  # (H,)
            mean_neg = acts[mask_neg].mean(dim=0)  # (H,)
            # delta = mu+ - mu-  (points toward correct reasoning)
            delta = mean_pos - mean_neg              # (H,)

            self.steering_vectors[layer_idx] = delta

            # Store for K-CAST: +1 = aligned, -1 = conflicting
            self.kcast_store[layer_idx] = {
                "activations": acts,                       # (N, H)
                "directions":  (mask_pos.float() * 2 - 1), # +1/-1
            }


    # ─── Step 3: Grid-Search Alpha ────────────────────────────────────────────

    def grid_search_alpha(
        self,
        val_loader: DataLoader,
        layers: Optional[List[int]] = None,
        alpha_range: Tuple[float, float] = (-1.0, 1.0),
        alpha_steps: int = 21,
    ) -> Tuple[int, float]:
        """
        Grid search over layer × alpha to find the combination maximising
        combined_score = Accuracy / (1 + ln(1 + TCE)) on the validation set.

        Returns (best_layer, best_alpha).
        """
        if layers is None:
            layers = list(self.steering_vectors.keys())

        alphas = torch.linspace(alpha_range[0], alpha_range[1], alpha_steps).tolist()

        best_combined = -1.0
        best_layer_found = layers[0] if layers else 9
        best_alpha_found = 0.0

        print(f"\n[Steering] Grid search over {len(layers)} layers × "
              f"{len(alphas)} alpha values …")

        for layer_idx in layers:
            if layer_idx not in self.steering_vectors:
                continue

            for alpha in alphas:
                # Temporarily set steering
                self.model.set_steering_vectors(
                    {layer_idx: self.steering_vectors[layer_idx]},
                    alpha=alpha,
                )
                metrics = compute_val_metrics(self.model, val_loader, self.device)
                combined = metrics["combined_score"]

                if combined > best_combined:
                    best_combined = combined
                    best_layer_found = layer_idx
                    best_alpha_found = alpha
                    print(f"  ↑ Layer {layer_idx} | α={alpha:+.2f} | "
                          f"Acc={metrics['accuracy']:.2f} | TCE={metrics['content_effect']:.4f} | "
                          f"Combined={combined:.4f}")

        # Disable steering (caller will re-enable with best params)
        self.model.disable_steering()

        self.best_alpha = best_alpha_found
        self.best_layer = best_layer_found

        print(f"\n[Steering] Best: layer={best_layer_found}, α={best_alpha_found:+.2f}, "
              f"combined={best_combined:.4f}")
        return best_layer_found, best_alpha_found

    # ─── K-CAST Application ───────────────────────────────────────────────────

    def apply_kcast(
        self,
        alpha: Optional[float] = None,
        k: int = STEERING_KNN,
        layer: Optional[int] = None,
    ) -> None:
        """
        Activate K-CAST steering on the model.

        With content-based D+/D- construction, K-CAST modulates steering
        *magnitude* (not direction) per example:
          - kNN neighbours that are content-aligned (direction=+1) indicate
            the example sits in a content-biased region → needs MORE steering.
          - kNN neighbours that are content-conflicting (direction=-1) indicate
            the example already reasons from structure → needs LESS steering.

        Magnitude = (fraction of aligned neighbours) * |α|
        Direction  = always +δ  (toward content-independent reasoning)
        """
        if alpha is None:
            alpha = self.best_alpha
        if layer is None:
            layer = self.best_layer

        if layer not in self.steering_vectors or layer not in self.kcast_store:
            print(f"[K-CAST] Layer {layer} not available; applying CAA instead.")
            self.model.set_steering_vectors(
                {layer: self.steering_vectors.get(layer, torch.zeros(768))},
                alpha=alpha,
            )
            return

        # Pre-normalise stored activations once for efficient cosine similarity
        stored_acts = self.kcast_store[layer]["activations"]       # (N, H)
        stored_dirs = self.kcast_store[layer]["directions"]         # (N,) ±1 aligned/conflicting
        stored_acts_norm = F.normalize(stored_acts, dim=-1)         # (N, H)

        delta = self.steering_vectors[layer]    # (H,)
        delta_unit = delta / (delta.norm() + 1e-8)

        def kcast_steering_fn(cls_hidden: torch.Tensor, all_hidden_states: tuple):
            device = cls_hidden.device
            # Get hidden state at the target layer
            h_layer = all_hidden_states[layer + 1][:, 0, :]  # (B, H)
            h_norm = F.normalize(h_layer, dim=-1).cpu()      # (B, H)

            # cosine similarity: (B, N)
            sims = h_norm @ stored_acts_norm.T

            # top-k indices: (B, k)
            _, topk_idx = sims.topk(k=min(k, sims.size(1)), dim=-1)

            # Fraction of content-aligned neighbours (+1 values)
            votes = stored_dirs[topk_idx]              # (B, k) ±1
            aligned_frac = (votes > 0).float().mean(dim=-1)  # (B,) in [0, 1]

            d = delta_unit.to(device)   # (H,)
            # Per-example magnitude: more aligned neighbours → more steering
            magnitudes = (aligned_frac.to(device) * abs(alpha)).unsqueeze(1)  # (B, 1)
            steered = cls_hidden + magnitudes * d.unsqueeze(0)
            return steered

        # Override the model's _apply_steering method
        import types
        self.model._apply_steering = types.MethodType(
            lambda self, cls_h, all_hs: kcast_steering_fn(cls_h, all_hs),
            self.model,
        )
        self.model._steering_vectors = {layer: delta.to(self.model.device)}
        self.model._steering_alpha = abs(alpha)

        print(f"[K-CAST] Applied: layer={layer}, α={alpha:+.2f}, k={k}")

    # ─── CAA Application (simple) ─────────────────────────────────────────────

    def apply_caa(
        self,
        alpha: Optional[float] = None,
        layer: Optional[int] = None,
    ) -> None:
        """Apply static CAA steering using best layer and alpha."""
        if alpha is None:
            alpha = self.best_alpha
        if layer is None:
            layer = self.best_layer
        if layer not in self.steering_vectors:
            print(f"[CAA] No steering vector for layer {layer}")
            return
        self.model.set_steering_vectors(
            {layer: self.steering_vectors[layer]},
            alpha=alpha,
        )
        print(f"[CAA] Applied: layer={layer}, α={alpha:+.2f}")

    # ─── Save / Load ──────────────────────────────────────────────────────────

    def save(self, path: str = STEERING_VECTORS_PATH) -> None:
        """Save steering vectors and K-CAST store to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "steering_vectors": self.steering_vectors,
            "kcast_store": self.kcast_store,
            "best_alpha": self.best_alpha,
            "best_layer": self.best_layer,
        }
        torch.save(payload, path)
        print(f"[Steering] Vectors saved to {path}")

    def load(self, path: str = STEERING_VECTORS_PATH) -> None:
        """Load steering vectors from disk."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.steering_vectors = payload["steering_vectors"]
        self.kcast_store      = payload["kcast_store"]
        self.best_alpha       = payload.get("best_alpha", STEERING_ALPHA)
        self.best_layer       = payload.get("best_layer", STEERING_LAYERS[-1])
        print(f"[Steering] Loaded from {path}  "
              f"(best_layer={self.best_layer}, best_alpha={self.best_alpha:+.2f})")
