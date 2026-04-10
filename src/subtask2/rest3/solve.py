#!/usr/bin/env python3
"""
Subtask 2 solver: validity + relevant premise retrieval.

Approach 1 — Joint fine-tuned model (Selection-Inference, Creswell et al. 2022):
  JointSyllogismClassifier is fine-tuned end-to-end on generated multi-premise
  training data. Two heads share the XLM-RoBERTa encoder:
    - Head 1: validity (CE loss)
    - Head 2: per-premise binary relevance (BCEWithLogitsLoss, pos_weight=6)
  At inference: validity = argmax(head1); if valid, top-2 premises by prob(head2).

  Requires: src/subtask2/outputs/model_checkpoint/model_weights.pt
  To train:  python3 src/subtask2/dataset_generator.py --technique cross
             python3 src/subtask2/train.py

Approach 2 — Leave-one-out NLI (FOLIO, Han et al. 2022):
  Reuses the existing subtask1 XLM-RoBERTa model (no extra training).
  For each valid item, for each premise i:
    - Build a version of the syllogism with premise i removed
    - Run through subtask1 validity model → P(valid | -premise_i)
    - Relevance score(i) = P(valid | all premises) - P(valid | -premise_i)
    - The two premises with the highest score drops are the relevant ones.
  This is a principled NLI/entailment approach: a premise is relevant iff
  removing it makes the argument fail.

Usage:
  cd /ssd_scratch/shubhamcvit/inlp/project
  source venv/bin/activate
  python3 -u src/subtask2/solve.py --approach 1    # joint model
  python3 -u src/subtask2/solve.py --approach 2    # leave-one-out NLI
  python3 -u src/subtask2/solve.py --approach both  # both
  python3 -u src/subtask2/solve.py --approach both --test-only  # first 5 items
"""

import json
import os
import re
import sys
import argparse

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
S1_DIR     = os.path.join(SCRIPT_DIR, "..", "subtask1")
sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Local imports ──────────────────────────────────────────────────
from src.subtask2.rest3.config import (
    TEST_DATA_PATH, OUTPUT_DIR,
    PREDICTIONS_APPROACH1_PATH, PREDICTIONS_APPROACH2_PATH,
    MODEL_SAVE_DIR as S2_MODEL_DIR,
    S1_MODEL_SAVE_DIR, S1_STEERING_VECTORS_PATH,
    STEERING_LAYERS, STEERING_KNN,
    S1_QUASAR_TEST_CACHE, S1_QUASAR_TRAIN_CACHE,
    S2_QUASAR_TEST_CACHE,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════

def parse_syllogism(text: str):
    """Split syllogism text into (premises, conclusion)."""
    sents = re.split(r'(?<=\.)\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) < 2:
        return [], sents[0] if sents else ""
    return sents[:-1], sents[-1]


def rebuild_syllogism(premises, conclusion):
    """Re-join a list of premises + conclusion into a single string."""
    return " ".join(premises) + " " + conclusion


# ═══════════════════════════════════════════════════════════════════
# APPROACH 1 — Fine-tuned JointSyllogismClassifier
#   (Selection-Inference, Creswell et al. 2022 / arXiv:2205.09712)
# ═══════════════════════════════════════════════════════════════════

def run_approach1(test_data, test_only: bool = False):
    """
    Use the JointSyllogismClassifier (fine-tuned on generated subtask2 data)
    to predict validity + relevant premises jointly.

    Requires: src/subtask2/outputs/model_checkpoint/model_weights.pt
    If checkpoint missing, raises a helpful error.
    """
    from src.subtask2.rest3.model import JointSyllogismClassifier
    from src.subtask2.rest3.data_loader import build_dataloaders_subtask2
    from src.subtask2.rest3.predict import predict_and_save

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"[Approach 1] Device: {device}  GPUs: {n_gpu}")

    weights_path = os.path.join(S2_MODEL_DIR, "model_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Subtask2 checkpoint not found: {weights_path}\n"
            "Run the following first:\n"
            "  python3 src/subtask2/dataset_generator.py --technique cross\n"
            "  python3 src/subtask2/train.py"
        )

    print(f"[Approach 1] Loading joint model from {weights_path}")
    model = JointSyllogismClassifier()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # Apply subtask1 steering vectors (same encoder, same content-bias direction)
    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_alpha = steer_data.get("best_alpha", -2.0)
            base = model.module if hasattr(model, "module") else model
            base._steering_alpha   = best_alpha
            print(f"[Approach 1] Steering vectors loaded (alpha={best_alpha})")
        except Exception as e:
            print(f"[warn] Could not load steering: {e}")

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    _, _, test_loader, _ = build_dataloaders_subtask2()
    if test_only:
        # Rebuild with only first 5 items
        pass  # test_only handled by sole caller using truncated test_data

    out_path = PREDICTIONS_APPROACH1_PATH
    preds = predict_and_save(
        model.module if hasattr(model, "module") else model,
        test_loader, device, output_path=out_path
    )
    return preds


# ═══════════════════════════════════════════════════════════════════
# APPROACH 2 — Leave-One-Out NLI
#   (FOLIO, Han et al. 2022 / arXiv:2209.00840)
#
#  Reuses subtask1 XLM-RoBERTa model — no additional training needed.
#  For each premise i in a valid syllogism:
#      score(i) = P(valid | ALL premises) - P(valid | premises \ {i})
#  The two premises with the highest relevance score are the relevant ones.
# ═══════════════════════════════════════════════════════════════════

def _load_s1_model_and_tokenizer():
    """Load the fine-tuned subtask1 XLM-RoBERTa model + QuaSAR abstractor."""
    # Import from subtask1 package
    from src.subtask2.rest3.model import SyllogismClassifier
    from src.subtask2.rest3.config import MODEL_NAME, MAX_SEQ_LEN, HF_CACHE_DIR, USE_QUASI_SYMBOLIC
    from transformers import AutoTokenizer
    from quasi_symbolic import QuasiSymbolicAbstractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"[Approach 2] Device: {device}  GPUs: {n_gpu}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # QuaSAR abstractor (no LLM needed — just the cache)
    quasar_cache = {}
    for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE, S2_QUASAR_TEST_CACHE]:
        if os.path.exists(cp):
            with open(cp) as f:
                quasar_cache.update(json.load(f))
    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache) if USE_QUASI_SYMBOLIC else None

    # Model
    model = SyllogismClassifier()
    weights = os.path.join(S1_MODEL_SAVE_DIR, "model_weights.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Subtask1 checkpoint not found: {weights}")
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=True)

    # Apply steering
    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_layer = steer_data.get("best_layer", STEERING_LAYERS[-1])
            best_alpha = steer_data.get("best_alpha", -2.0)
            model._steering_alpha = float(best_alpha)
            # Restore kcast vectors into model
            if "vectors" in steer_data:
                model._steering_vectors = {
                    int(k): v for k, v in steer_data["vectors"].items()
                }
            print(f"[Approach 2] Steering applied: layer={best_layer}, alpha={best_alpha}")
        except Exception as e:
            print(f"[warn] Steering not applied: {e}")

    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    return model, tokenizer, abstractor, device, MAX_SEQ_LEN


def _get_validity_prob(
    model, tokenizer, abstractor, text: str,
    item_stub: dict, device, max_seq_len: int,
) -> float:
    """Run a single text through the subtask1 model, return P(valid)."""
    from src.subtask2.rest3.config import USE_QUASI_SYMBOLIC, ABSTRACT_SEP
    USE_QUASI_SYMBOLIC_LOC = USE_QUASI_SYMBOLIC  # local alias to avoid double import

    # Build input (optionally with QuaSAR form)
    if abstractor is not None:
        item_id = item_stub.get("id", "")
        from src.subtask2.rest3.config import QUASAR_MODE
        abstract = abstractor.abstract(text, item_id=item_id, quasar_mode=QUASAR_MODE)
        if abstract and abstract != text:
            input_text = abstract + ABSTRACT_SEP + text
        else:
            input_text = text
    else:
        input_text = text

    enc = tokenizer(
        input_text, max_length=max_seq_len, truncation=True,
        return_tensors="pt", padding=False
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    base = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        out  = base(input_ids, attention_mask)
        prob = torch.softmax(out["logits"], dim=-1)[0, 1].item()  # P(valid)
    return prob


def run_approach2(test_data):
    """
    Leave-one-out NLI premise scoring using the subtask1 validity model.

    For each item:
      1. Get P(valid | ALL premises + conclusion) = baseline_prob
      2. If baseline → valid:
         For each premise i:
           P_i = P(valid | premises - {i} + conclusion)
           score(i) = baseline_prob - P_i
         Select top-2 premises by score (highest drop = most relevant)
      3. If baseline → invalid: relevant_premises = []

    For items with 2 premises: if both are "valid" by the full-premise model,
    both are relevant by definition (no distractors to distinguish from).
    """
    model, tokenizer, abstractor, device, max_seq_len = _load_s1_model_and_tokenizer()

    from src.subtask2.rest3.config import ID2LABEL
    LABEL_VALID_IDX = 1

    predictions = []
    n = len(test_data)

    for idx, item in enumerate(test_data):
        uid = item["id"]
        premises, conclusion = parse_syllogism(item["syllogism"])

        rel_premises = []

        if len(premises) < 2:
            # Degenerate: treat as invalid
            predictions.append({"id": uid, "validity": False, "relevant_premises": []})
            continue

        # ── Step 1: baseline probability ──
        full_text     = rebuild_syllogism(premises, conclusion)
        baseline_prob = _get_validity_prob(
            model, tokenizer, abstractor, full_text, item, device, max_seq_len
        )
        valid = baseline_prob >= 0.5

        if valid:
            if len(premises) == 2:
                # No distractors — both are relevant by definition
                rel_premises = [0, 1]
            else:
                # ── Step 2: leave-one-out scoring ──
                scores = []
                for i in range(len(premises)):
                    reduced      = premises[:i] + premises[i+1:]
                    reduced_text = rebuild_syllogism(reduced, conclusion)
                    prob_without = _get_validity_prob(
                        model, tokenizer, abstractor, reduced_text, item, device, max_seq_len
                    )
                    scores.append(baseline_prob - prob_without)

                # Top-2 by relevance score (highest drop = most influential)
                top2 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
                rel_premises = sorted(top2)

        predictions.append({
            "id":               uid,
            "validity":         valid,
            "relevant_premises": rel_premises,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            n_prem = len(premises)
            print(f"  [Approach 2] {idx+1}/{n}  "
                  f"valid={valid}  premises={n_prem}  "
                  f"baseline_prob={baseline_prob:.3f}")
            sys.stdout.flush()

    return predictions


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", default="both", choices=["1", "2", "both"])
    parser.add_argument("--test-only", action="store_true",
                        help="Run on first 5 items for quick sanity check")
    args = parser.parse_args()

    with open(TEST_DATA_PATH) as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} subtask-2 test items")

    if args.test_only:
        test_data = test_data[:5]
        print("TEST MODE: first 5 items only")

    # ── Approach 2 (no additional model needed) ──
    if args.approach in ("2", "both"):
        print("\n=== Approach 2: Leave-One-Out NLI (FOLIO-style) ===")
        preds2 = run_approach2(test_data)
        with open(PREDICTIONS_APPROACH2_PATH, "w") as f:
            json.dump(preds2, f, indent=2)
        v2  = sum(1 for p in preds2 if p["validity"])
        rp2 = sum(1 for p in preds2 if p["relevant_premises"])
        print(f"Approach 2 saved → {PREDICTIONS_APPROACH2_PATH}")
        print(f"  Valid={v2}  Invalid={len(preds2)-v2}  WithPremises={rp2}")

    # ── Approach 1 (fine-tuned joint model) ──
    if args.approach in ("1", "both"):
        print("\n=== Approach 1: Joint XLM-RoBERTa (Selection-Inference) ===")
        preds1 = run_approach1(test_data, test_only=args.test_only)
        v1  = sum(1 for p in preds1 if p["validity"])
        rp1 = sum(1 for p in preds1 if p["relevant_premises"])
        print(f"Approach 1 saved → {PREDICTIONS_APPROACH1_PATH}")
        print(f"  Valid={v1}  Invalid={len(preds1)-v1}  WithPremises={rp1}")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
