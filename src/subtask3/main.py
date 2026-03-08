"""
main.py
-------
End-to-end orchestration CLI for SemEval-2026 Task 11 — Subtask 1.

Three integrated techniques:
  Technique 1 - Neuro-Symbolic Integration (spaCy, fallback when no LLM)
  Technique 2 - Activation Steering (CAA / K-CAST, post-training)
  Technique 3 - QuaSAR Chain-of-Thought (Llama-generated, primary)

Pipeline:
  0. [generate] (offline) Run Llama to produce QuaSAR cache for all data
  1. Load and preprocess training/test data
  2. Build quasi-symbolic augmented inputs:
       Primary:  QuaSAR s2 formalisation from Llama cache (technique 3)
       Fallback: spaCy X/Y/Z extraction (technique 1)
  3. Fine-tune XLM-RoBERTa classifier
  4. Compute activation steering vectors (technique 2;
     behaviour-based D+/D- from correct/incorrect predictions)
  5. Grid-search best steering hyperparameters on validation set
  6. Generate predictions on test set with steering applied
  7. Evaluate and write final predictions JSON

Usage (full pipeline, after QuaSAR caches exist):
  python main.py --mode full

Usage (individual stages):
  python main.py --mode generate   # Llama QuaSAR generation (offline, GPU-heavy)
  python main.py --mode train
  python main.py --mode steer
  python main.py --mode predict
  python main.py --mode evaluate

Flags:
  --no_quasi_symbolic   Disable quasi-symbolic text augmentation
  --no_steering         Disable activation steering
  --epochs N            Override number of training epochs
  --alpha F             Manually set steering alpha (skips grid search)
  --model_path PATH     Load existing checkpoint (skip training)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_SAVE_DIR,
    STEERING_VECTORS_PATH, PREDICTIONS_PATH, EVAL_RESULTS_PATH,
    QUASAR_TRAIN_CACHE, QUASAR_TEST_CACHE,
    USE_QUASI_SYMBOLIC, USE_ACTIVATION_STEERING,
    STEERING_LAYERS, STEERING_ALPHA, STEERING_KNN,
    SEED, NUM_EPOCHS, EARLY_STOPPING_PATIENCE,
)
from data_loader import build_dataloaders, get_class_weights, load_json
from model import SyllogismClassifier
from train import train, set_seed
from activation_steering import ActivationSteerer
from predict import predict_and_save, analyse_predictions
from evaluate import evaluate_from_files, print_full_report
from quasar_generator import generate_quasar_batch, load_quasar_cache, load_llama_model


# ─── Raw LLM Baseline ────────────────────────────────────────────────────────

def print_llm_baseline_comparison(pipeline_metrics: dict):
    """
    Compare our full pipeline results with raw Llama QuaSAR predictions
    (no fine-tuning, no steering — just Llama's answer from s4).
    """
    import math

    test_data = load_json(TEST_DATA_PATH)
    quasar_test_cache = {}
    if os.path.exists(QUASAR_TEST_CACHE):
        import json as json_mod
        with open(QUASAR_TEST_CACHE, "r") as f:
            quasar_test_cache = json_mod.load(f)

    # Compute Llama baseline
    correct = total = no_answer = 0
    subgroups = {(True, True): [0, 0], (True, False): [0, 0],
                 (False, True): [0, 0], (False, False): [0, 0]}

    for item in test_data:
        entry = quasar_test_cache.get(item["id"], {})
        llm_ans = entry.get("quasar_answer")
        gt_validity = item["validity"]
        gt_plaus = item.get("plausibility", None)

        if llm_ans is None:
            no_answer += 1
            continue
        if isinstance(llm_ans, str):
            llm_ans = llm_ans.strip().lower()
            if llm_ans in ("valid", "true"):
                pred = True
            elif llm_ans in ("invalid", "false"):
                pred = False
            else:
                no_answer += 1
                continue
        elif isinstance(llm_ans, bool):
            pred = llm_ans
        else:
            no_answer += 1
            continue

        total += 1
        if pred == gt_validity:
            correct += 1
        key = (gt_validity, gt_plaus)
        if key in subgroups:
            subgroups[key][1] += 1
            if pred == gt_validity:
                subgroups[key][0] += 1

    llm_acc = (correct / total * 100) if total > 0 else 0.0

    def sg_acc(k):
        c, t = subgroups[k]
        return (c / t * 100) if t > 0 else 0.0

    a_pv, a_iv = sg_acc((True, True)), sg_acc((True, False))
    a_pi, a_ii = sg_acc((False, True)), sg_acc((False, False))
    ce_intra = (abs(a_pv - a_iv) + abs(a_pi - a_ii)) / 2.0
    ce_inter = (abs(a_pv - a_pi) + abs(a_iv - a_ii)) / 2.0
    llm_tce = (ce_intra + ce_inter) / 2.0
    llm_combined = llm_acc / (1 + math.log(1 + llm_tce)) if llm_tce > 0 else llm_acc

    # Print comparison
    print("\n" + "="*70)
    print("  COMPARISON: Our Pipeline vs Raw LLM (Llama 3.1 8B Instruct)")
    print("="*70)
    print(f"  {'Metric':<25} {'Raw LLM':>12} {'Our Pipeline':>14} {'Improvement':>14}")
    print(f"  {'-'*65}")
    print(f"  {'Accuracy':<25} {llm_acc:>11.2f}% {pipeline_metrics['accuracy']:>13.4f}% "
          f"{pipeline_metrics['accuracy'] - llm_acc:>+13.2f}%")
    print(f"  {'Content Effect (TCE)':<25} {llm_tce:>12.2f} {pipeline_metrics['content_effect']:>14.4f} "
          f"{pipeline_metrics['content_effect'] - llm_tce:>+14.2f}")
    print(f"  {'Combined Score':<25} {llm_combined:>12.4f} {pipeline_metrics['combined_score']:>14.4f} "
          f"{pipeline_metrics['combined_score'] - llm_combined:>+14.4f}")
    print(f"  {'-'*65}")
    print(f"  LLM subgroups: PV={a_pv:.1f}  IV={a_iv:.1f}  PI={a_pi:.1f}  II={a_ii:.1f}")
    print(f"  LLM: {total} predictions, {no_answer} no-answer")
    print("="*70)


# ─── Device Setup ────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        print(f"[Main] Using {n_gpu} GPU(s): {torch.cuda.get_device_name(0)}")
        for i in range(n_gpu):
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Main] Using Apple MPS (M-series GPU)")
    else:
        device = torch.device("cpu")
        print("[Main] No GPU detected — using CPU (training will be slow).")
    return device


# ─── Stage Functions ─────────────────────────────────────────────────────────

def stage_generate(args, device: torch.device):
    """Stage 0 (offline): Generate QuaSAR 4-step outputs using Llama.

    Produces cached JSON files mapping item IDs to s2 formalisations.
    Must run once before training (requires Llama access).
    """
    print("\n" + "="*70)
    print("STAGE 0: QuaSAR Generation (Llama -> 4-step Chain-of-Thought)")
    print("="*70)

    # Check which datasets need generation
    tasks = []
    for data_path, cache_path, label in [
        (TRAIN_DATA_PATH, QUASAR_TRAIN_CACHE, "train"),
        (TEST_DATA_PATH,  QUASAR_TEST_CACHE,  "test"),
    ]:
        data = load_json(data_path)
        existing = load_quasar_cache(cache_path)
        missing = [item for item in data if item.get("id") not in existing]

        if not missing:
            print(f"[Generate] {label}: all {len(data)} items cached. Skipping.")
        else:
            print(f"[Generate] {label}: {len(missing)} of {len(data)} items need generation.")
            tasks.append((data, cache_path, label, existing))

    if not tasks:
        print("[Generate] All QuaSAR caches are complete. Nothing to do.")
        return

    # Load Llama (only once, shared across train/test)
    llama_model, llama_tokenizer = load_llama_model()

    for data, cache_path, label, existing in tasks:
        print(f"\n[Generate] Generating QuaSAR for {label} ({len(data)} items)...")
        generate_quasar_batch(
            data=data,
            model=llama_model,
            tokenizer=llama_tokenizer,
            output_path=cache_path,
            existing_cache=existing,
        )
        cache = load_quasar_cache(cache_path)
        print(f"[Generate] {label}: done. {len(cache)} items cached.")

    # Free Llama from GPU memory before training
    del llama_model, llama_tokenizer
    torch.cuda.empty_cache()
    print("[Generate] Llama unloaded. GPU memory freed for training.")


def stage_train(args, device: torch.device):
    """Stage 1: Fine-tune XLM-RoBERTa on the training data."""
    print("\n" + "="*70)
    print("STAGE 1/4: Data Loading & Preprocessing")
    print("="*70)

    use_qs = (not args.no_quasi_symbolic) and USE_QUASI_SYMBOLIC
    train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta = build_dataloaders(
        TRAIN_DATA_PATH, TEST_DATA_PATH, use_quasi_symbolic=use_qs
    )

    print("\n" + "="*70)
    print("STAGE 2/4: Model Initialisation")
    print("="*70)
    model = SyllogismClassifier(vocab_size_delta=vocab_delta)
    model = model.to(device)
    print(f"[Main] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # NOTE: DataParallel disabled — corrupts activation steering.
    # Single GPU is sufficient for XLM-RoBERTa-base (278M params).

    train_data = load_json(TRAIN_DATA_PATH)
    class_weights = get_class_weights(train_data)

    print("\n" + "="*70)
    print("STAGE 3/4: Fine-Tuning")
    print("="*70)
    num_epochs = args.epochs if args.epochs else NUM_EPOCHS
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size_delta=vocab_delta,
        class_weights=class_weights,
        num_epochs=num_epochs,
        save_path=MODEL_SAVE_DIR,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
    )

    return train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta


def stage_steer(args, model, train_loader, val_loader, device):
    """Stage 2: Compute steering vectors + grid-search alpha."""
    print("\n" + "="*70)
    print("STAGE 3/4: Activation Steering (CAA / K-CAST)")
    print("="*70)

    steerer = ActivationSteerer(model, device)
    steerer.compute_steering_vectors(train_loader, layers=STEERING_LAYERS)

    if args.alpha is not None:
        # Manual alpha — skip grid search
        steerer.best_alpha = args.alpha
        steerer.best_layer = STEERING_LAYERS[-1]
        print(f"[Main] Using manual alpha={args.alpha}")
    else:
        steerer.grid_search_alpha(val_loader, layers=STEERING_LAYERS)

    steerer.save(STEERING_VECTORS_PATH)
    return steerer


def stage_predict(args, model, test_loader, device, use_kcast: bool = True):
    """Stage 3: Generate predictions on test set."""
    print("\n" + "="*70)
    print("STAGE 4/4: Inference & Prediction Generation")
    print("="*70)

    preds = predict_and_save(
        model=model,
        loader=test_loader,
        device=device,
        output_path=PREDICTIONS_PATH,
        return_probabilities=True,
    )
    return preds


def stage_evaluate(preds):
    """Stage 4: Evaluate predictions against ground-truth test labels."""
    print("\n" + "="*70)
    print("EVALUATION (in-process, using ground-truth from test set)")
    print("="*70)

    metrics = evaluate_from_files(
        reference_path=TEST_DATA_PATH,
        predictions_path=PREDICTIONS_PATH,
        output_metrics_path=EVAL_RESULTS_PATH,
        verbose=True,
    )
    return metrics


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SemEval-2026 Task 11 Subtask 1 — Full Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generate", "train", "steer", "predict", "evaluate"],
        default="full",
        help="Pipeline stage to run (default: full). 'generate' runs Llama QuaSAR.",
    )
    parser.add_argument(
        "--no_quasi_symbolic",
        action="store_true",
        help="Disable quasi-symbolic text augmentation.",
    )
    parser.add_argument(
        "--no_steering",
        action="store_true",
        help="Disable activation steering (run plain fine-tuned model).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Steering alpha (overrides grid search if provided).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_SAVE_DIR,
        help="Path to saved model checkpoint (skips training in steer/predict modes).",
    )
    parser.add_argument(
        "--use_kcast",
        action="store_true",
        default=True,
        help="Use K-CAST steering (default) vs plain CAA.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()

    print("\n" + "#"*70)
    print("  SemEval-2026 Task 11 - Subtask 3: Syllogistic Reasoning (Multilingual)")
    print("  Technique 1: Neuro-Symbolic Integration (spaCy fallback)")
    print("  Technique 2: Activation Steering (CAA / K-CAST)")
    print("  Technique 3: QuaSAR Chain-of-Thought (Llama -> XLM-RoBERTa)")
    print("#"*70 + "\n")

    use_qs       = (not args.no_quasi_symbolic) and USE_QUASI_SYMBOLIC
    use_steering = (not args.no_steering)       and USE_ACTIVATION_STEERING

    # ─── GENERATE ONLY ────────────────────────────────────────────────────────
    if args.mode == "generate":
        stage_generate(args, device)
        return

    # ─── FULL PIPELINE ────────────────────────────────────────────────────────
    if args.mode == "full":
        # Check QuaSAR caches
        for cache_path, label in [(QUASAR_TRAIN_CACHE, "train"), (QUASAR_TEST_CACHE, "test")]:
            if os.path.exists(cache_path):
                cache = load_quasar_cache(cache_path)
                print(f"[Main] QuaSAR cache ({label}): {len(cache)} entries OK")
            else:
                print(f"[Main] WARNING: QuaSAR cache ({label}) not found: {cache_path}")
                print(f"  Will use spaCy fallback. Run --mode generate first for best results.")

        # --- Train ---
        train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta = \
            stage_train(args, device)

        # --- Baseline: Untrained XLM-RoBERTa ---
        print("\n" + "="*70)
        print("  BASELINE: Evaluating UNTRAINED XLM-RoBERTa (random classifier head)")
        print("="*70)
        baseline_model = SyllogismClassifier(vocab_size_delta=vocab_delta)
        baseline_model = baseline_model.to(device)
        # DataParallel disabled (corrupts steering activations)
        baseline_preds = predict_and_save(
            model=baseline_model,
            loader=test_loader,
            device=device,
            output_path=PREDICTIONS_PATH.replace(".json", "_baseline.json"),
            return_probabilities=True,
        )
        baseline_metrics = evaluate_from_files(
            reference_path=TEST_DATA_PATH,
            predictions_path=PREDICTIONS_PATH.replace(".json", "_baseline.json"),
            output_metrics_path=EVAL_RESULTS_PATH.replace(".json", "_baseline.json"),
            verbose=True,
        )
        del baseline_model
        torch.cuda.empty_cache()

        # Load the best checkpoint saved during training
        print(f"\n[Main] Loading best checkpoint from {args.model_path} …")
        model = SyllogismClassifier.load(args.model_path, vocab_size_delta=vocab_delta)
        model = model.to(device)
        # NOTE: Do NOT use DataParallel for steering/prediction.
        # DataParallel corrupts get_layer_hidden_states() activations,
        # causing steering vectors to be ~150x too small (effectively zero).

        # --- Pre-Steering Test Evaluation ---
        print("\n" + "="*70)
        print("  FINE-TUNED: Evaluating fine-tuned model WITHOUT activation steering")
        print("="*70)
        pre_steer_preds = predict_and_save(
            model=model,
            loader=test_loader,
            device=device,
            output_path=PREDICTIONS_PATH.replace(".json", "_pre_steering.json"),
            return_probabilities=True,
        )
        pre_steer_metrics = evaluate_from_files(
            reference_path=TEST_DATA_PATH,
            predictions_path=PREDICTIONS_PATH.replace(".json", "_pre_steering.json"),
            output_metrics_path=EVAL_RESULTS_PATH.replace(".json", "_pre_steering.json"),
            verbose=True,
        )
        analyse_predictions(pre_steer_preds, TEST_DATA_PATH)

        # --- Steer ---
        if use_steering:
            steerer = stage_steer(args, model, train_loader, val_loader, device)
            steerer.apply_caa(alpha=steerer.best_alpha, layer=steerer.best_layer)
        else:
            print("[Main] Activation steering disabled.")

        # --- Predict (with steering) ---
        preds = stage_predict(args, model, test_loader, device)

        # --- Evaluate (with steering) ---
        metrics = stage_evaluate(preds)
        analyse_predictions(preds, TEST_DATA_PATH)

        # --- Final 3-Way Comparison ---
        print("\n" + "="*70)
        print("  FINAL COMPARISON: Untrained vs Fine-Tuned vs Fine-Tuned + Steering")
        print("="*70)
        print(f"  {'Metric':<25} {'Untrained':>14} {'Fine-Tuned':>14} {'+ Steering':>14}")
        print(f"  {'-'*67}")
        print(f"  {'Accuracy':<25} {baseline_metrics['accuracy']:>13.2f}% {pre_steer_metrics['accuracy']:>13.2f}% {metrics['accuracy']:>13.2f}%")
        print(f"  {'Content Effect (TCE)':<25} {baseline_metrics['content_effect']:>14.4f} {pre_steer_metrics['content_effect']:>14.4f} {metrics['content_effect']:>14.4f}")
        print(f"  {'Combined Score':<25} {baseline_metrics['combined_score']:>14.4f} {pre_steer_metrics['combined_score']:>14.4f} {metrics['combined_score']:>14.4f}")
        print("="*70)

        print("\n[Main] Full pipeline complete!")
        print(f"  Predictions → {PREDICTIONS_PATH}")
        print(f"  Metrics     → {EVAL_RESULTS_PATH}")

    # ─── TRAIN ONLY ───────────────────────────────────────────────────────────
    elif args.mode == "train":
        stage_train(args, device)

    # ─── STEER ONLY ───────────────────────────────────────────────────────────
    elif args.mode == "steer":
        print(f"[Main] Loading model from {args.model_path} …")
        train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta = \
            build_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, use_quasi_symbolic=use_qs)
        model = SyllogismClassifier.load(args.model_path, vocab_size_delta=vocab_delta)
        model = model.to(device)
        stage_steer(args, model, train_loader, val_loader, device)

    # ─── PREDICT ONLY ─────────────────────────────────────────────────────────
    elif args.mode == "predict":
        train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta = \
            build_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, use_quasi_symbolic=use_qs)
        model = SyllogismClassifier.load(args.model_path, vocab_size_delta=vocab_delta)
        model = model.to(device)

        if use_steering and os.path.exists(STEERING_VECTORS_PATH):
            steerer = ActivationSteerer(model, device)
            steerer.load(STEERING_VECTORS_PATH)
            steerer.apply_kcast(
                alpha=args.alpha if args.alpha else steerer.best_alpha,
                k=STEERING_KNN,
                layer=steerer.best_layer,
            )

        preds = stage_predict(args, model, test_loader, device)
        analyse_predictions(preds, TEST_DATA_PATH)

    # ─── EVALUATE ONLY ────────────────────────────────────────────────────────
    elif args.mode == "evaluate":
        if not os.path.exists(PREDICTIONS_PATH):
            print(f"[Main] ERROR: No predictions file found at {PREDICTIONS_PATH}")
            print("  Run with --mode predict (or --mode full) first.")
            sys.exit(1)
        evaluate_from_files(
            reference_path=TEST_DATA_PATH,
            predictions_path=PREDICTIONS_PATH,
            output_metrics_path=EVAL_RESULTS_PATH,
        )


if __name__ == "__main__":
    main()
