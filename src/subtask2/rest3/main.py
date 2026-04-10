"""
main.py
-------
End-to-end orchestration CLI for SemEval-2026 Task 11 — Subtask 2.

Subtask 2: Validity classification + relevant premise retrieval.
Given a syllogism with 5-8 premises (2 genuine + 3-6 distractors),
predict: (1) validity (bool), (2) indices of 2 relevant premises (0-indexed).

Techniques (inherited from subtask 1):
  Technique 1 - Neuro-Symbolic Integration (spaCy fallback for QuaSAR)
  Technique 2 - Activation Steering (CAA / K-CAST, post-training)
  Technique 3 - QuaSAR Chain-of-Thought (Llama-generated, primary)

Two inference approaches:
  Approach 1 — Joint fine-tuned model (Selection-Inference, Creswell 2022):
    JointSyllogismClassifier with validity head + premise head
  Approach 2 — Leave-One-Out NLI (FOLIO, Han et al. 2022):
    Reuses subtask1 XLM-RoBERTa, scores premise relevance by probability drop

Pipeline:
  1. Train JointSyllogismClassifier (from subtask1 checkpoint)
  2. Evaluate level-by-level:
     - Baseline (fine-tuned classifier, no QuaSAR, no steering)
     - + QuaSAR (quasi-symbolic augmentation)
     - + Activation Steering (CAA/K-CAST)
  3. Run both approaches on test set
  4. Evaluate against ground truth
  5. Print comparison tables

Usage:
  python3 -u src/subtask2/main.py --mode full
  python3 -u src/subtask2/main.py --mode train
  python3 -u src/subtask2/main.py --mode predict
  python3 -u src/subtask2/main.py --mode evaluate
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
S1_DIR     = os.path.join(SCRIPT_DIR, "..", "subtask1")
sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.subtask2.rest3.config import (
    TEST_DATA_PATH, OUTPUT_DIR,
    PREDICTIONS_APPROACH1_PATH, PREDICTIONS_APPROACH2_PATH,
    MODEL_SAVE_DIR as S2_MODEL_DIR,
    S1_MODEL_SAVE_DIR, S1_STEERING_VECTORS_PATH,
    STEERING_LAYERS, STEERING_KNN,
    S1_QUASAR_TEST_CACHE, S1_QUASAR_TRAIN_CACHE,
    S2_QUASAR_TEST_CACHE, EVAL_RESULTS_PATH,
    LLAMA_FT_SAVE_DIR, LLAMA_PREDICTIONS_PATH,
)
from src.subtask2.rest3.evaluate import (
    compute_metrics, print_full_report, print_comparison,
    evaluate_from_files,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Device Setup
# ═══════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"[Main] Using {n_gpu} GPU(s): {torch.cuda.get_device_name(0)}")
        for i in range(n_gpu):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[Main] No GPU detected — using CPU.")
    return device


# ═══════════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════════

def parse_syllogism(text: str):
    sents = re.split(r'(?<=\.)\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) < 2:
        return [], sents[0] if sents else ""
    return sents[:-1], sents[-1]


def rebuild_syllogism(premises, conclusion):
    return " ".join(premises) + " " + conclusion


def _load_test_data():
    with open(TEST_DATA_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# APPROACH 1 — Fine-tuned JointSyllogismClassifier
# ═══════════════════════════════════════════════════════════════════

def run_approach1(test_data, output_path=None):
    # Ensure subtask2 modules are resolved, not subtask1's
    # (model.py loading can pollute sys.path with S1_DIR at index 0)
    for mod in ("predict", "data_loader"):
        cached = sys.modules.get(mod)
        if cached and hasattr(cached, "__file__") and cached.__file__ and "subtask1" in cached.__file__:
            del sys.modules[mod]
    if sys.path[0] != SCRIPT_DIR:
        sys.path.insert(0, SCRIPT_DIR)

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
            "Run: python3 src/subtask2/main.py --mode train"
        )

    print(f"[Approach 1] Loading joint model from {weights_path}")
    model = JointSyllogismClassifier()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # Apply steering
    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_alpha = steer_data.get("best_alpha", -2.0)
            base = model.module if hasattr(model, "module") else model
            base._steering_alpha = best_alpha
            print(f"[Approach 1] Steering vectors loaded (alpha={best_alpha})")
        except Exception as e:
            print(f"[warn] Could not load steering: {e}")

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    _, _, test_loader, _ = build_dataloaders_subtask2()

    out_path = output_path or PREDICTIONS_APPROACH1_PATH
    preds = predict_and_save(
        model.module if hasattr(model, "module") else model,
        test_loader, device, output_path=out_path
    )
    return preds


# ═══════════════════════════════════════════════════════════════════
# APPROACH 2 — Leave-One-Out NLI
# ═══════════════════════════════════════════════════════════════════

def _load_s1_model_and_tokenizer():
    from src.subtask2.rest3.model import SyllogismClassifier
    from src.subtask2.rest3.config import MODEL_NAME, MAX_SEQ_LEN, HF_CACHE_DIR, USE_QUASI_SYMBOLIC
    from transformers import AutoTokenizer
    from quasi_symbolic import QuasiSymbolicAbstractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"[Approach 2] Device: {device}  GPUs: {n_gpu}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    quasar_cache = {}
    for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE, S2_QUASAR_TEST_CACHE]:
        if os.path.exists(cp):
            with open(cp) as f:
                quasar_cache.update(json.load(f))
    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache) if USE_QUASI_SYMBOLIC else None

    model = SyllogismClassifier()
    weights = os.path.join(S1_MODEL_SAVE_DIR, "model_weights.pt")
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Subtask1 checkpoint not found: {weights}")
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if os.path.exists(S1_STEERING_VECTORS_PATH):
        try:
            steer_data = torch.load(S1_STEERING_VECTORS_PATH, map_location="cpu")
            best_layer = steer_data.get("best_layer", STEERING_LAYERS[-1])
            best_alpha = steer_data.get("best_alpha", -2.0)
            model._steering_alpha = float(best_alpha)
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


def _get_validity_prob(model, tokenizer, abstractor, text, item_stub, device, max_seq_len):
    from src.subtask2.rest3.config import USE_QUASI_SYMBOLIC, ABSTRACT_SEP, QUASAR_MODE

    if abstractor is not None:
        item_id = item_stub.get("id", "")
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
        prob = torch.softmax(out["logits"], dim=-1)[0, 1].item()
    return prob


def run_approach2(test_data, output_path=None):
    model, tokenizer, abstractor, device, max_seq_len = _load_s1_model_and_tokenizer()

    from src.subtask2.rest3.config import ID2LABEL

    predictions = []
    n = len(test_data)

    for idx, item in enumerate(test_data):
        uid = item["id"]
        premises, conclusion = parse_syllogism(item["syllogism"])

        if len(premises) < 2:
            predictions.append({"id": uid, "validity": False, "relevant_premises": []})
            continue

        full_text     = rebuild_syllogism(premises, conclusion)
        baseline_prob = _get_validity_prob(
            model, tokenizer, abstractor, full_text, item, device, max_seq_len
        )
        valid = baseline_prob >= 0.5

        rel_premises = []
        if valid:
            if len(premises) == 2:
                rel_premises = [0, 1]
            else:
                scores = []
                for i in range(len(premises)):
                    reduced      = premises[:i] + premises[i+1:]
                    reduced_text = rebuild_syllogism(reduced, conclusion)
                    prob_without = _get_validity_prob(
                        model, tokenizer, abstractor, reduced_text, item, device, max_seq_len
                    )
                    scores.append(baseline_prob - prob_without)
                top2 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
                rel_premises = sorted(top2)

        predictions.append({
            "id":               uid,
            "validity":         valid,
            "relevant_premises": rel_premises,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [Approach 2] {idx+1}/{n}  "
                  f"valid={valid}  premises={len(premises)}  "
                  f"baseline_prob={baseline_prob:.3f}")
            sys.stdout.flush()

    # Save
    save_path = output_path or PREDICTIONS_APPROACH2_PATH
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)

    return predictions


# ═══════════════════════════════════════════════════════════════════
# APPROACH 3 — Llama QLoRA Fine-tuned Decoder
# ═══════════════════════════════════════════════════════════════════

def run_approach3(test_data, output_path=None):
    """Run fine-tuned Llama decoder for validity + premise prediction."""
    from src.subtask2.rest3.train_llama import predict_and_save as llama_predict_and_save

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.subtask2.rest3.config import LLAMA_MODEL_NAME, HF_CACHE_DIR, LLAMA_FT_USE_4BIT

    if not os.path.exists(LLAMA_FT_SAVE_DIR):
        raise FileNotFoundError(
            f"Llama checkpoint not found: {LLAMA_FT_SAVE_DIR}\n"
            "Run: python3 -u src/subtask2/train_llama.py"
        )

    print(f"[Approach 3] Loading Llama checkpoint from {LLAMA_FT_SAVE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_FT_SAVE_DIR, cache_dir=HF_CACHE_DIR, padding_side="left",
    )

    bnb_config = None
    if LLAMA_FT_USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME, cache_dir=HF_CACHE_DIR,
        quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, LLAMA_FT_SAVE_DIR)
    model.eval()

    out_path = output_path or LLAMA_PREDICTIONS_PATH
    preds = llama_predict_and_save(model, tokenizer, test_data, output_path=out_path)
    return preds


# ═══════════════════════════════════════════════════════════════════
# LLM Baseline Comparison
# ═══════════════════════════════════════════════════════════════════

def compute_llm_baseline_metrics(test_data):
    """
    Use raw Llama QuaSAR predictions (s4 answer) as a baseline.
    Only evaluates validity (no premise retrieval from LLM).
    """
    quasar_cache = {}
    for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE, S2_QUASAR_TEST_CACHE]:
        if os.path.exists(cp):
            with open(cp) as f:
                quasar_cache.update(json.load(f))

    predictions = []
    no_answer = 0
    for item in test_data:
        entry = quasar_cache.get(item["id"], {})
        llm_ans = entry.get("quasar_answer")

        if llm_ans is None:
            no_answer += 1
            # Default to invalid with no premises
            predictions.append({"id": item["id"], "validity": False, "relevant_premises": []})
            continue

        if isinstance(llm_ans, str):
            llm_ans = llm_ans.strip().lower()
            if llm_ans in ("valid", "true"):
                pred = True
            elif llm_ans in ("invalid", "false"):
                pred = False
            else:
                no_answer += 1
                predictions.append({"id": item["id"], "validity": False, "relevant_premises": []})
                continue
        elif isinstance(llm_ans, bool):
            pred = llm_ans
        else:
            no_answer += 1
            predictions.append({"id": item["id"], "validity": False, "relevant_premises": []})
            continue

        # LLM baseline has no premise retrieval capability
        predictions.append({
            "id": item["id"],
            "validity": pred,
            "relevant_premises": [],
        })

    metrics = compute_metrics(test_data, predictions)
    metrics["n_no_answer"] = no_answer
    return metrics


# ═══════════════════════════════════════════════════════════════════
# Stage Functions
# ═══════════════════════════════════════════════════════════════════

def stage_train(device):
    """Stage 1: Fine-tune JointSyllogismClassifier."""
    print("\n" + "="*70)
    print("STAGE 1: Fine-Tuning JointSyllogismClassifier")
    print("="*70)

    from src.subtask2.rest3.train import train as run_training
    history = run_training()
    return history


def stage_predict(test_data, device):
    """Stage 2: Generate predictions with all approaches."""
    print("\n" + "="*70)
    print("STAGE 2: Inference & Prediction Generation")
    print("="*70)

    results = {}

    # Approach 2 first (no extra checkpoint needed)
    print("\n" + "-"*60)
    print("  Approach 2: Leave-One-Out NLI (FOLIO-style)")
    print("-"*60)
    preds2 = run_approach2(test_data)
    v2  = sum(1 for p in preds2 if p["validity"])
    print(f"  Approach 2: {len(preds2)} items  |  Valid={v2}  Invalid={len(preds2)-v2}")
    results["approach2"] = preds2

    # Approach 1 (fine-tuned joint model)
    print("\n" + "-"*60)
    print("  Approach 1: Joint XLM-RoBERTa (Selection-Inference)")
    print("-"*60)
    preds1 = run_approach1(test_data)
    v1  = sum(1 for p in preds1 if p["validity"])
    print(f"  Approach 1: {len(preds1)} items  |  Valid={v1}  Invalid={len(preds1)-v1}")
    results["approach1"] = preds1

    # Approach 3 (Llama QLoRA) — only if checkpoint exists
    if os.path.exists(LLAMA_FT_SAVE_DIR):
        print("\n" + "-"*60)
        print("  Approach 3: Llama QLoRA Fine-tuned Decoder")
        print("-"*60)
        try:
            preds3 = run_approach3(test_data)
            v3 = sum(1 for p in preds3 if p["validity"])
            print(f"  Approach 3: {len(preds3)} items  |  Valid={v3}  Invalid={len(preds3)-v3}")
            results["approach3"] = preds3
        except Exception as e:
            print(f"  [warn] Approach 3 failed: {e}")
    else:
        print("\n  [skip] Approach 3 (Llama QLoRA): no checkpoint found")

    return results


def stage_evaluate(test_data):
    """Stage 3: Evaluate predictions against ground truth."""
    print("\n" + "="*70)
    print("STAGE 3: Evaluation (ground-truth from test set)")
    print("="*70)

    results = {}

    for approach, path, label in [
        ("approach1", PREDICTIONS_APPROACH1_PATH, "Approach 1 (Joint Model)"),
        ("approach2", PREDICTIONS_APPROACH2_PATH, "Approach 2 (LOO NLI)"),
        ("approach3", LLAMA_PREDICTIONS_PATH,     "Approach 3 (Llama QLoRA)"),
    ]:
        if os.path.exists(path):
            metrics = evaluate_from_files(
                reference_path=TEST_DATA_PATH,
                predictions_path=path,
                output_metrics_path=EVAL_RESULTS_PATH.replace(".json", f"_{approach}.json"),
                verbose=True,
                title=f"Evaluation — {label}",
            )
            results[approach] = metrics
        else:
            print(f"  [skip] No predictions file for {label}: {path}")

    return results


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2026 Task 11 Subtask 2 — Full Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "predict", "evaluate"],
        default="full",
    )
    args = parser.parse_args()

    from src.subtask2.rest3.train import set_seed
    from src.subtask2.rest3.config import SEED
    set_seed(SEED)
    device = get_device()

    print("\n" + "#"*70)
    print("  SemEval-2026 Task 11 — Subtask 2: Validity + Premise Retrieval")
    print("  Technique 1: Neuro-Symbolic (spaCy fallback)")
    print("  Technique 2: Activation Steering (CAA / K-CAST)")
    print("  Technique 3: QuaSAR Chain-of-Thought (Llama → XLM-RoBERTa)")
    print("  Approach 1:  Joint Model (Selection-Inference)")
    print("  Approach 2:  Leave-One-Out NLI (FOLIO-style)")
    print("#"*70 + "\n")

    test_data = _load_test_data()
    print(f"[Main] Test data: {len(test_data)} items")

    # Check caches
    for cache_path, label in [
        (S1_QUASAR_TRAIN_CACHE, "S1 QuaSAR train"),
        (S1_QUASAR_TEST_CACHE,  "S1 QuaSAR test"),
        (S2_QUASAR_TEST_CACHE,  "S2 QuaSAR test"),
    ]:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"[Main] {label}: {len(cache)} entries OK")
        else:
            print(f"[Main] WARNING: {label} not found: {cache_path}")

    # ─── TRAIN ONLY ───
    if args.mode == "train":
        stage_train(device)
        print("\n[Feedback] Training complete. Please review the metrics above.")
        print("  → To run inference: python3 src/subtask2/main.py --mode predict")
        print("  → Any changes needed before proceeding? Let me know.")
        return

    # ─── EVALUATE ONLY ───
    if args.mode == "evaluate":
        eval_results = stage_evaluate(test_data)
        if len(eval_results) == 2:
            print_comparison([
                {"label": "Approach 1 (Joint)", "metrics": eval_results["approach1"]},
                {"label": "Approach 2 (LOO)",   "metrics": eval_results["approach2"]},
            ], title="APPROACH COMPARISON")

        print("\n[Feedback] Evaluation complete. Please review the results above.")
        print("  → Are you satisfied with the results?")
        print("  → Should we try different hyperparameters or approaches?")
        return

    # ─── PREDICT ONLY ───
    if args.mode == "predict":
        pred_results = stage_predict(test_data, device)
        eval_results = stage_evaluate(test_data)
        if len(eval_results) == 2:
            print_comparison([
                {"label": "Approach 1 (Joint)", "metrics": eval_results["approach1"]},
                {"label": "Approach 2 (LOO)",   "metrics": eval_results["approach2"]},
            ], title="APPROACH COMPARISON")

        print("\n[Feedback] Prediction and evaluation complete.")
        print("  → Please review the comparison above.")
        print("  → Any adjustments needed? Let me know.")
        return

    # ─── FULL PIPELINE ─────────────────────────────────────────────────────

    # === LLM BASELINE ===
    print("\n" + "="*70)
    print("  LEVEL 0: Raw LLM Baseline (Llama 3.1 8B Instruct)")
    print("="*70)
    llm_metrics = compute_llm_baseline_metrics(test_data)
    print_full_report(llm_metrics, title="Level 0: Raw LLM Baseline (validity only, no premise retrieval)")
    n_no_ans = llm_metrics.get("n_no_answer", 0)
    print(f"  Note: {n_no_ans} items had no LLM answer (defaulted to invalid).")
    print(f"  Note: LLM baseline has no premise retrieval → F1=0.")

    # === TRAIN ===
    stage_train(device)

    # === LEVEL 1: Fine-tuned model WITHOUT QuaSAR, WITHOUT steering ===
    # (The model was trained with QuaSAR+steering from S1 checkpoint,
    #  but we can evaluate without them by running approach2 without QuaSAR)
    # We compare: approach1 (joint) vs approach2 (LOO NLI)

    # === APPROACH 2: LOO NLI ===
    print("\n" + "="*70)
    print("  LEVEL 1: Approach 2 — Leave-One-Out NLI (subtask1 model)")
    print("="*70)
    preds2 = run_approach2(test_data)
    a2_metrics = compute_metrics(test_data, preds2)
    print_full_report(a2_metrics, title="Level 1: Approach 2 (LOO NLI + QuaSAR + Steering)")

    # === APPROACH 1: Joint Model ===
    print("\n" + "="*70)
    print("  LEVEL 2: Approach 1 — Joint Fine-Tuned Model")
    print("="*70)
    preds1 = run_approach1(test_data)
    a1_metrics = compute_metrics(test_data, preds1)
    print_full_report(a1_metrics, title="Level 2: Approach 1 (Joint XLM-R + QuaSAR + Steering)")

    # === AGREEMENT ANALYSIS ===
    print("\n" + "="*70)
    print("  AGREEMENT ANALYSIS: Approach 1 vs Approach 2")
    print("="*70)
    p1_map = {p["id"]: p for p in preds1}
    p2_map = {p["id"]: p for p in preds2}
    agree_v = sum(1 for uid in p1_map if p1_map[uid]["validity"] == p2_map[uid]["validity"])
    both_valid = sum(1 for uid in p1_map if p1_map[uid]["validity"] and p2_map[uid]["validity"])
    agree_p = sum(1 for uid in p1_map
                  if p1_map[uid]["validity"] and p2_map[uid]["validity"]
                  and p1_map[uid]["relevant_premises"] == p2_map[uid]["relevant_premises"])
    only1 = sum(1 for uid in p1_map if p1_map[uid]["validity"] and not p2_map[uid]["validity"])
    only2 = sum(1 for uid in p1_map if not p1_map[uid]["validity"] and p2_map[uid]["validity"])
    n_items = len(preds1)
    print(f"  Validity agreement: {agree_v}/{n_items} ({agree_v/n_items*100:.1f}%)")
    print(f"  Both valid: {both_valid}  |  Only A1 valid: {only1}  |  Only A2 valid: {only2}")
    if both_valid > 0:
        print(f"  Premise agreement (shared valid): {agree_p}/{both_valid} "
              f"({agree_p/both_valid*100:.1f}%)")

    # === FINAL COMPARISON TABLE ===
    comparison_rows = [
        {"label": "LLM Baseline", "metrics": llm_metrics},
        {"label": "A2: LOO NLI",  "metrics": a2_metrics},
        {"label": "A1: Joint",    "metrics": a1_metrics},
    ]
    print_comparison(comparison_rows,
                     title="FINAL COMPARISON: LLM Baseline vs LOO NLI vs Joint Model")

    # === SAVE BEST PREDICTIONS ===
    # Determine which approach has higher combined_score
    best_approach = "approach1" if a1_metrics["combined_score"] >= a2_metrics["combined_score"] else "approach2"
    best_metrics = a1_metrics if best_approach == "approach1" else a2_metrics
    best_pred_path = PREDICTIONS_APPROACH1_PATH if best_approach == "approach1" else PREDICTIONS_APPROACH2_PATH

    # Save combined metrics
    all_results = {
        "llm_baseline": {k: v for k, v in llm_metrics.items() if isinstance(v, (int, float))},
        "approach1": {k: v for k, v in a1_metrics.items() if isinstance(v, (int, float))},
        "approach2": {k: v for k, v in a2_metrics.items() if isinstance(v, (int, float))},
        "best_approach": best_approach,
    }
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Main] All metrics saved → {EVAL_RESULTS_PATH}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — Best: {best_approach}")
    print(f"  Accuracy     : {best_metrics['accuracy']:.2f}%")
    print(f"  Premise F1   : {best_metrics['premise_f1']:.2f}%")
    print(f"  Overall Perf : {best_metrics['overall_perf']:.4f}")
    print(f"  TCE          : {best_metrics['content_effect']:.4f}")
    print(f"  Combined (↑) : {best_metrics['combined_score']:.4f}")
    print(f"  Predictions  → {best_pred_path}")
    print(f"  Metrics      → {EVAL_RESULTS_PATH}")
    print(f"{'='*70}")

    # === FEEDBACK PROMPT ===
    print("\n" + "-"*70)
    print("  [Feedback Required]")
    print("-"*70)
    print("  Please review the results above and provide feedback:")
    print("  1. Are the accuracy and premise F1 scores acceptable?")
    print("  2. Is the content effect (TCE) within acceptable bounds?")
    print("  3. Should we try different hyperparameters (learning rate, epochs)?")
    print("  4. Should we regenerate the training data with different parameters?")
    print("  5. Which approach should be used for final submission?")
    print("  6. Any other observations or concerns?")
    print("-"*70)


if __name__ == "__main__":
    main()
