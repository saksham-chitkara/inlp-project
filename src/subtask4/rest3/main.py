"""
main.py
-------
End-to-end orchestration CLI for SemEval-2026 Task 11 — Subtask 4.

Subtask 4: Multilingual Validity Classification + Relevant Premise Retrieval.
Same as Subtask 2 but on translated syllogisms (11 languages).

Three approaches:
  Approach 1 — Joint fine-tuned JointSyllogismClassifier (XLM-RoBERTa)
  Approach 2 — Leave-One-Out NLI (reuses subtask1 model)
  Approach 3 — Llama QLoRA fine-tuned decoder

Pipeline:
  0. [generate]  QuaSAR generation for subtask4 test items (Llama, offline)
  1. [train]     Fine-tune JointSyllogismClassifier on translated data
  2. [predict]   Run all 3 approaches on multilingual test set
  3. [evaluate]  Evaluate against ground truth

Usage:
  python3 -u src/subtask4/main.py --mode full
  python3 -u src/subtask4/main.py --mode generate
  python3 -u src/subtask4/main.py --mode train
  python3 -u src/subtask4/main.py --mode predict
  python3 -u src/subtask4/main.py --mode evaluate
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
S2_DIR     = os.path.join(SCRIPT_DIR, "..", "subtask2")
sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.subtask4.rest3.config import (
    TEST_DATA_PATH, OUTPUT_DIR,
    PREDICTIONS_APPROACH1_PATH, PREDICTIONS_APPROACH2_PATH,
    MODEL_SAVE_DIR as S4_MODEL_DIR,
    S2_MODEL_SAVE_DIR,
    S1_MODEL_SAVE_DIR, S1_STEERING_VECTORS_PATH,
    STEERING_LAYERS, STEERING_KNN,
    S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
    S2_QUASAR_TEST_CACHE, S4_QUASAR_TEST_CACHE,
    EVAL_RESULTS_PATH,
    LLAMA_FT_SAVE_DIR, LLAMA_PREDICTIONS_PATH,
)
from src.subtask4.rest3.evaluate import (
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
        print(f"[Main-S4] Using {n_gpu} GPU(s): {torch.cuda.get_device_name(0)}")
        for i in range(n_gpu):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[Main-S4] No GPU detected — using CPU.")
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
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# STAGE 0: QuaSAR Generation for subtask4 test items
# ═══════════════════════════════════════════════════════════════════

def stage_generate(device):
    """Generate QuaSAR cache for subtask4 test items using Llama."""
    print("\n" + "=" * 70)
    print("STAGE 0: QuaSAR Generation for Subtask 4 Test Items")
    print("=" * 70)

    from src.subtask4.rest3.quasar_generator import (
        generate_quasar_batch, load_quasar_cache, load_llama_model,
    )

    test_data = _load_test_data()
    existing = load_quasar_cache(S4_QUASAR_TEST_CACHE)

    # Check how many unique English syllogisms still need generation
    unique_eng = {item["syllogism"] for item in test_data}
    cached_eng = set()
    for item in test_data:
        if item["id"] in existing:
            cached_eng.add(item["syllogism"])
    missing = unique_eng - cached_eng

    if not missing:
        print(f"[Generate] All {len(unique_eng)} unique English syllogisms cached. Skipping.")
        return

    print(f"[Generate] {len(missing)} of {len(unique_eng)} unique syllogisms need generation.")
    model, tokenizer = load_llama_model()
    generate_quasar_batch(test_data, model, tokenizer, S4_QUASAR_TEST_CACHE, existing)
    del model, tokenizer
    torch.cuda.empty_cache()
    print("[Generate] Llama unloaded. GPU memory freed.")


# ═══════════════════════════════════════════════════════════════════
# APPROACH 1 — Fine-tuned JointSyllogismClassifier
# ═══════════════════════════════════════════════════════════════════

def run_approach1(test_data, output_path=None):
    """Run Approach 1: Joint XLM-RoBERTa model on subtask4 test data."""
    # Ensure subtask4 modules are resolved
    for mod in ("predict", "data_loader"):
        cached = sys.modules.get(mod)
        if cached and hasattr(cached, "__file__") and cached.__file__ and "subtask4" not in cached.__file__:
            del sys.modules[mod]
    if sys.path[0] != SCRIPT_DIR:
        sys.path.insert(0, SCRIPT_DIR)

    from src.subtask4.rest3.model import JointSyllogismClassifier
    from src.subtask4.rest3.data_loader import build_dataloaders_subtask4
    from src.subtask4.rest3.predict import predict_and_save

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"[Approach 1] Device: {device}  GPUs: {n_gpu}")

    weights_path = os.path.join(S4_MODEL_DIR, "model_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Subtask4 checkpoint not found: {weights_path}\n"
            "Run: python3 src/subtask4/main.py --mode train"
        )

    print(f"[Approach 1] Loading joint model from {weights_path}")
    model = JointSyllogismClassifier()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)

    # Apply steering from subtask1 (if available)
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

    _, _, test_loader, _ = build_dataloaders_subtask4()

    out_path = output_path or PREDICTIONS_APPROACH1_PATH
    preds = predict_and_save(
        model.module if hasattr(model, "module") else model,
        test_loader, device, output_path=out_path
    )
    return preds


# ═══════════════════════════════════════════════════════════════════
# APPROACH 2 — Leave-One-Out NLI (subtask1 model on translated text)
# ═══════════════════════════════════════════════════════════════════

def _load_s1_model_and_tokenizer():
    """Load subtask1 XLM-RoBERTa classifier for LOO NLI."""
    # Need subtask1's model and quasi_symbolic
    sys.path.insert(0, S1_DIR)
    from src.subtask4.rest3.model import SyllogismClassifier
    from src.subtask4.rest3.config import MODEL_NAME, MAX_SEQ_LEN, HF_CACHE_DIR, USE_QUASI_SYMBOLIC
    from transformers import AutoTokenizer
    from quasi_symbolic import QuasiSymbolicAbstractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu  = torch.cuda.device_count()
    print(f"[Approach 2] Device: {device}  GPUs: {n_gpu}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # Merge all QuaSAR caches (S1 train/test + S2 test + S4 test)
    quasar_cache = {}
    for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
               S2_QUASAR_TEST_CACHE, S4_QUASAR_TEST_CACHE]:
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
    """Get P(valid) for a given text using the subtask1 model."""
    from src.subtask4.rest3.config import USE_QUASI_SYMBOLIC, ABSTRACT_SEP, QUASAR_MODE

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
        return_tensors="pt", padding=False)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    base = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        out  = base(input_ids, attention_mask)
        prob = torch.softmax(out["logits"], dim=-1)[0, 1].item()
    return prob


def run_approach2(test_data, output_path=None):
    """
    Leave-One-Out NLI on subtask4 test data.

    For multilingual items, the LOO analysis operates on the TRANSLATED
    text (syllogism_t). QuaSAR augmentation uses the English syllogism.
    """
    model, tokenizer, abstractor, device, max_seq_len = _load_s1_model_and_tokenizer()

    predictions = []
    n = len(test_data)

    for idx, item in enumerate(test_data):
        uid = item["id"]
        # Use translated text for premises/conclusion
        syllogism_t = item.get("syllogism_t", item["syllogism"])
        premises, conclusion = parse_syllogism(syllogism_t)

        if len(premises) < 2:
            predictions.append({"id": uid, "validity": False, "relevant_premises": []})
            continue

        full_text     = rebuild_syllogism(premises, conclusion)
        baseline_prob = _get_validity_prob(
            model, tokenizer, abstractor, full_text, item, device, max_seq_len)
        valid = baseline_prob >= 0.5

        rel_premises = []
        if valid:
            if len(premises) == 2:
                rel_premises = [0, 1]
            else:
                scores = []
                for i in range(len(premises)):
                    reduced      = premises[:i] + premises[i + 1:]
                    reduced_text = rebuild_syllogism(reduced, conclusion)
                    prob_without = _get_validity_prob(
                        model, tokenizer, abstractor, reduced_text, item, device, max_seq_len)
                    scores.append(baseline_prob - prob_without)
                top2 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:2]
                rel_premises = sorted(top2)

        predictions.append({
            "id":                uid,
            "validity":          valid,
            "relevant_premises": rel_premises,
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [Approach 2] {idx + 1}/{n}  "
                  f"valid={valid}  lang={item.get('lang', '?')}  "
                  f"premises={len(premises)}  "
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
    from src.subtask4.rest3.train_llama import predict_and_save as llama_predict_and_save

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.subtask4.rest3.config import LLAMA_MODEL_NAME, HF_CACHE_DIR, LLAMA_FT_USE_4BIT

    if not os.path.exists(LLAMA_FT_SAVE_DIR):
        raise FileNotFoundError(
            f"Llama checkpoint not found: {LLAMA_FT_SAVE_DIR}\n"
            "Run: python3 -u src/subtask4/train_llama.py")

    print(f"[Approach 3] Loading Llama checkpoint from {LLAMA_FT_SAVE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_FT_SAVE_DIR, cache_dir=HF_CACHE_DIR, padding_side="left")

    bnb_config = None
    if LLAMA_FT_USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME, cache_dir=HF_CACHE_DIR,
        quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, LLAMA_FT_SAVE_DIR)
    model.eval()

    out_path = output_path or LLAMA_PREDICTIONS_PATH
    preds = llama_predict_and_save(model, tokenizer, test_data, output_path=out_path)
    return preds


# ═══════════════════════════════════════════════════════════════════
# LLM Baseline (raw QuaSAR answers as validity predictions)
# ═══════════════════════════════════════════════════════════════════

def compute_llm_baseline_metrics(test_data):
    quasar_cache = {}
    for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
               S2_QUASAR_TEST_CACHE, S4_QUASAR_TEST_CACHE]:
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
    """Fine-tune JointSyllogismClassifier on translated training data."""
    print("\n" + "=" * 70)
    print("STAGE 1: Fine-Tuning JointSyllogismClassifier (Multilingual)")
    print("=" * 70)

    from src.subtask4.rest3.train import train as run_training
    history = run_training()
    return history


def stage_predict(test_data, device):
    """Generate predictions with all 3 approaches."""
    print("\n" + "=" * 70)
    print("STAGE 2: Inference & Prediction Generation")
    print("=" * 70)

    results = {}

    # Approach 2: LOO NLI
    print("\n" + "-" * 60)
    print("  Approach 2: Leave-One-Out NLI (FOLIO-style)")
    print("-" * 60)
    preds2 = run_approach2(test_data)
    v2 = sum(1 for p in preds2 if p["validity"])
    print(f"  Approach 2: {len(preds2)} items  |  Valid={v2}  Invalid={len(preds2) - v2}")
    results["approach2"] = preds2

    # Approach 1: Joint Model
    print("\n" + "-" * 60)
    print("  Approach 1: Joint XLM-RoBERTa (Selection-Inference)")
    print("-" * 60)
    preds1 = run_approach1(test_data)
    v1 = sum(1 for p in preds1 if p["validity"])
    print(f"  Approach 1: {len(preds1)} items  |  Valid={v1}  Invalid={len(preds1) - v1}")
    results["approach1"] = preds1

    # Approach 3: Llama QLoRA
    if os.path.exists(LLAMA_FT_SAVE_DIR):
        print("\n" + "-" * 60)
        print("  Approach 3: Llama QLoRA Fine-tuned Decoder")
        print("-" * 60)
        try:
            preds3 = run_approach3(test_data)
            v3 = sum(1 for p in preds3 if p["validity"])
            print(f"  Approach 3: {len(preds3)} items  |  Valid={v3}  Invalid={len(preds3) - v3}")
            results["approach3"] = preds3
        except Exception as e:
            print(f"  [warn] Approach 3 failed: {e}")
    else:
        print("\n  [skip] Approach 3 (Llama QLoRA): no checkpoint found")

    return results


def stage_evaluate(test_data):
    """Evaluate predictions against ground truth."""
    print("\n" + "=" * 70)
    print("STAGE 3: Evaluation (ground-truth from test set)")
    print("=" * 70)

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
# Per-Language Analysis
# ═══════════════════════════════════════════════════════════════════

def per_language_analysis(test_data, predictions):
    """Break down accuracy and premise F1 by language."""
    print("\n" + "=" * 70)
    print("  PER-LANGUAGE ANALYSIS")
    print("=" * 70)

    # Group by language
    by_lang = {}
    pred_map = {p["id"]: p for p in predictions}
    for item in test_data:
        lang = item.get("lang", "en")
        by_lang.setdefault(lang, {"items": [], "preds": []})
        by_lang[lang]["items"].append(item)
        by_lang[lang]["preds"].append(pred_map.get(item["id"], {
            "id": item["id"], "validity": False, "relevant_premises": []}))

    print(f"  {'Language':<8} {'N':>4} {'Acc%':>7} {'Prem_F1%':>9} {'Combined':>10}")
    print(f"  {'-' * 42}")

    lang_metrics = {}
    for lang in sorted(by_lang.keys()):
        group = by_lang[lang]
        m = compute_metrics(group["items"], group["preds"])
        lang_metrics[lang] = m
        print(f"  {lang:<8} {len(group['items']):>4} "
              f"{m['accuracy']:>6.1f}% {m['premise_f1']:>8.1f}% "
              f"{m['combined_score']:>9.2f}")

    # Overall (macro average across languages)
    avg_acc   = sum(m["accuracy"] for m in lang_metrics.values()) / len(lang_metrics)
    avg_pf1   = sum(m["premise_f1"] for m in lang_metrics.values()) / len(lang_metrics)
    avg_comb  = sum(m["combined_score"] for m in lang_metrics.values()) / len(lang_metrics)
    print(f"  {'-' * 42}")
    print(f"  {'MACRO':>8} {'':>4} {avg_acc:>6.1f}% {avg_pf1:>8.1f}% {avg_comb:>9.2f}")
    print("=" * 70)

    return lang_metrics


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2026 Task 11 Subtask 4 — Full Pipeline (Multilingual)")
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "train", "predict", "evaluate"],
        default="full",
    )
    args = parser.parse_args()

    from src.subtask4.rest3.train import set_seed
    from src.subtask4.rest3.config import SEED
    set_seed(SEED)
    device = get_device()

    print("\n" + "#" * 70)
    print("  SemEval-2026 Task 11 — Subtask 4: Multilingual Validity + Premise Retrieval")
    print("  Approach 1:  Joint XLM-RoBERTa (Selection-Inference)")
    print("  Approach 2:  Leave-One-Out NLI (FOLIO-style)")
    print("  Approach 3:  Llama QLoRA Fine-tuned Decoder")
    print("#" * 70 + "\n")

    # ─── GENERATE ONLY ────────────────────────────────────────────
    if args.mode == "generate":
        stage_generate(device)
        print("\n[Done] QuaSAR generation complete.")
        return

    test_data = _load_test_data()
    print(f"[Main-S4] Test data: {len(test_data)} items")

    # Show language distribution
    langs = {}
    for item in test_data:
        lang = item.get("lang", "?")
        langs[lang] = langs.get(lang, 0) + 1
    print(f"[Main-S4] Languages: {dict(sorted(langs.items()))}")

    # Check QuaSAR caches
    for cache_path, label in [
        (S1_QUASAR_TRAIN_CACHE, "S1 QuaSAR train"),
        (S1_QUASAR_TEST_CACHE,  "S1 QuaSAR test"),
        (S2_QUASAR_TEST_CACHE,  "S2 QuaSAR test"),
        (S4_QUASAR_TEST_CACHE,  "S4 QuaSAR test"),
    ]:
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"[Main-S4] {label}: {len(cache)} entries OK")
        else:
            print(f"[Main-S4] WARNING: {label} not found: {cache_path}")

    # ─── TRAIN ONLY ──────────────────────────────────────────────
    if args.mode == "train":
        stage_train(device)
        print("\n[Done] Training complete.")
        return

    # ─── EVALUATE ONLY ───────────────────────────────────────────
    if args.mode == "evaluate":
        eval_results = stage_evaluate(test_data)

        # Per-language analysis for each approach
        for approach, path in [
            ("approach1", PREDICTIONS_APPROACH1_PATH),
            ("approach2", PREDICTIONS_APPROACH2_PATH),
            ("approach3", LLAMA_PREDICTIONS_PATH),
        ]:
            if os.path.exists(path):
                with open(path) as f:
                    preds = json.load(f)
                print(f"\n  --- {approach.upper()} ---")
                per_language_analysis(test_data, preds)

        if len(eval_results) >= 2:
            rows = []
            for k in sorted(eval_results.keys()):
                rows.append({"label": k, "metrics": eval_results[k]})
            print_comparison(rows, title="APPROACH COMPARISON (Subtask 4)")

        print("\n[Done] Evaluation complete.")
        return

    # ─── PREDICT ONLY ────────────────────────────────────────────
    if args.mode == "predict":
        pred_results = stage_predict(test_data, device)
        eval_results = stage_evaluate(test_data)

        # Per-language for best approach
        best_key = max(eval_results, key=lambda k: eval_results[k].get("combined_score", 0))
        best_path = {
            "approach1": PREDICTIONS_APPROACH1_PATH,
            "approach2": PREDICTIONS_APPROACH2_PATH,
            "approach3": LLAMA_PREDICTIONS_PATH,
        }.get(best_key)
        if best_path and os.path.exists(best_path):
            with open(best_path) as f:
                best_preds = json.load(f)
            per_language_analysis(test_data, best_preds)

        print("\n[Done] Prediction and evaluation complete.")
        return

    # ─── FULL PIPELINE ───────────────────────────────────────────────

    # === LLM BASELINE ===
    if os.path.exists(S4_QUASAR_TEST_CACHE):
        print("\n" + "=" * 70)
        print("  LEVEL 0: Raw LLM Baseline (Llama QuaSAR answers)")
        print("=" * 70)
        llm_metrics = compute_llm_baseline_metrics(test_data)
        print_full_report(llm_metrics, title="Level 0: Raw LLM Baseline (validity only, no premise retrieval)")
        n_no_ans = llm_metrics.get("n_no_answer", 0)
        print(f"  Note: {n_no_ans} items had no LLM answer (defaulted to invalid).")
    else:
        llm_metrics = None
        print("\n  [skip] LLM baseline (no S4 QuaSAR cache found)")

    # === TRAIN ===
    stage_train(device)

    # === APPROACH 2: LOO NLI ===
    print("\n" + "=" * 70)
    print("  Approach 2: Leave-One-Out NLI (subtask1 model on translated text)")
    print("=" * 70)
    preds2 = run_approach2(test_data)
    a2_metrics = compute_metrics(test_data, preds2)
    print_full_report(a2_metrics, title="Approach 2 (LOO NLI + QuaSAR + Steering)")
    per_language_analysis(test_data, preds2)

    # === APPROACH 1: Joint Model ===
    print("\n" + "=" * 70)
    print("  Approach 1: Joint XLM-RoBERTa (Selection-Inference)")
    print("=" * 70)
    preds1 = run_approach1(test_data)
    a1_metrics = compute_metrics(test_data, preds1)
    print_full_report(a1_metrics, title="Approach 1 (Joint XLM-R + QuaSAR + Steering)")
    per_language_analysis(test_data, preds1)

    # === AGREEMENT ANALYSIS ===
    print("\n" + "=" * 70)
    print("  AGREEMENT ANALYSIS: Approach 1 vs Approach 2")
    print("=" * 70)
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
    print(f"  Validity agreement: {agree_v}/{n_items} ({agree_v / n_items * 100:.1f}%)")
    print(f"  Both valid: {both_valid}  |  Only A1 valid: {only1}  |  Only A2 valid: {only2}")
    if both_valid > 0:
        print(f"  Premise agreement (shared valid): {agree_p}/{both_valid} "
              f"({agree_p / both_valid * 100:.1f}%)")

    # === FINAL COMPARISON TABLE ===
    comparison_rows = []
    if llm_metrics:
        comparison_rows.append({"label": "LLM Baseline", "metrics": llm_metrics})
    comparison_rows.extend([
        {"label": "A2: LOO NLI", "metrics": a2_metrics},
        {"label": "A1: Joint",   "metrics": a1_metrics},
    ])
    print_comparison(comparison_rows,
                     title="FINAL COMPARISON: Subtask 4 (Multilingual)")

    # === SAVE METRICS ===
    best_approach = "approach1" if a1_metrics["combined_score"] >= a2_metrics["combined_score"] else "approach2"
    best_metrics  = a1_metrics if best_approach == "approach1" else a2_metrics
    best_pred_path = PREDICTIONS_APPROACH1_PATH if best_approach == "approach1" else PREDICTIONS_APPROACH2_PATH

    all_results = {
        "approach1": {k: v for k, v in a1_metrics.items() if isinstance(v, (int, float))},
        "approach2": {k: v for k, v in a2_metrics.items() if isinstance(v, (int, float))},
        "best_approach": best_approach,
    }
    if llm_metrics:
        all_results["llm_baseline"] = {k: v for k, v in llm_metrics.items() if isinstance(v, (int, float))}
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Main-S4] All metrics saved → {EVAL_RESULTS_PATH}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — Best: {best_approach}")
    print(f"  Accuracy     : {best_metrics['accuracy']:.2f}%")
    print(f"  Premise F1   : {best_metrics['premise_f1']:.2f}%")
    print(f"  Overall Perf : {best_metrics['overall_perf']:.4f}")
    print(f"  TCE          : {best_metrics['content_effect']:.4f}")
    print(f"  Combined (↑) : {best_metrics['combined_score']:.4f}")
    print(f"  Predictions  → {best_pred_path}")
    print(f"  Metrics      → {EVAL_RESULTS_PATH}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
