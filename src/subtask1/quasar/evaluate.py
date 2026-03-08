"""
evaluate.py
-----------
Wrapper around the official SemEval-2026 Task 11 evaluation functions,
plus convenience utilities for in-process evaluation during development.

The official evaluation script lives at:
  semeval_2026_task_11/evaluation_kit/task 1 & 3/evaluation_script.py

This module provides:
1. run_official_eval()    — calls the official script via subprocess (for final submission checks)
2. compute_metrics()      — runs the same logic in-process (fast, for development + validation)
3. print_full_report()    — pretty-printed report with all subgroup accuracies
"""

import os
import sys
import json
import math
import subprocess
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from config import EVAL_KIT_REFERENCE, EVAL_RESULTS_PATH, PREDICTIONS_PATH, EVAL_SCRIPT_PATH


# ─── In-Process Evaluation (mirrors official script exactly) ─────────────────

def compute_subgroup_acc(
    gt_map: Dict[str, Any],
    predictions: List[Dict],
    gt_validity: bool,
    gt_plausibility: bool,
) -> float:
    """Accuracy for one (validity × plausibility) subgroup."""
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is None:
            continue
        if gt.get("validity") == gt_validity and gt.get("plausibility") == gt_plausibility:
            total += 1
            if pred["validity"] == gt["validity"]:
                correct += 1
    return (correct / total * 100) if total > 0 else 0.0


def compute_metrics(
    ground_truth: List[Dict],
    predictions: List[Dict],
) -> Dict[str, float]:
    """
    Compute accuracy, content effect (TCE), and combined_score.

    Mirrors the logic in the official evaluation_script.py exactly.

    Returns:
      {
        "accuracy": float,
        "content_effect": float,
        "combined_score": float,
        "acc_plausible_valid": float,
        "acc_implausible_valid": float,
        "acc_plausible_invalid": float,
        "acc_implausible_invalid": float,
        "n_predicted": int,
        "n_missing": int,
      }
    """
    gt_map = {item["id"]: item for item in ground_truth}
    pred_ids = {p["id"] for p in predictions}
    gt_ids   = set(gt_map.keys())
    missing  = len(gt_ids - pred_ids)

    if missing > 0:
        print(f"  ⚠ {missing} ground-truth examples have no prediction!")

    # Overall accuracy
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is not None and isinstance(pred["validity"], bool):
            total += 1
            if pred["validity"] == gt["validity"]:
                correct += 1
    overall_acc = (correct / total * 100) if total > 0 else 0.0

    # Subgroup accuracies (4 cells)
    a_pv = compute_subgroup_acc(gt_map, predictions, True,  True)   # valid + plausible
    a_iv = compute_subgroup_acc(gt_map, predictions, True,  False)  # valid + implausible
    a_pi = compute_subgroup_acc(gt_map, predictions, False, True)   # invalid + plausible
    a_ii = compute_subgroup_acc(gt_map, predictions, False, False)  # invalid + implausible

    # Content effect components
    intra_valid   = abs(a_pv - a_iv)
    intra_invalid = abs(a_pi - a_ii)
    ce_intra      = (intra_valid + intra_invalid) / 2.0

    inter_plaus   = abs(a_pv - a_pi)
    inter_implaus = abs(a_iv - a_ii)
    ce_inter      = (inter_plaus + inter_implaus) / 2.0

    tce      = (ce_intra + ce_inter) / 2.0
    combined = overall_acc / (1 + math.log(1 + tce))

    return {
        "accuracy":               round(overall_acc, 4),
        "content_effect":         round(tce, 4),
        "combined_score":         round(combined, 4),
        "acc_plausible_valid":    round(a_pv, 2),
        "acc_implausible_valid":  round(a_iv, 2),
        "acc_plausible_invalid":  round(a_pi, 2),
        "acc_implausible_invalid":round(a_ii, 2),
        "ce_intra":               round(ce_intra, 4),
        "ce_inter":               round(ce_inter, 4),
        "n_predicted":            total,
        "n_missing":              missing,
    }


# ─── Pretty Report ────────────────────────────────────────────────────────────

def print_full_report(
    metrics: Dict[str, float],
    title: str = "Evaluation Report — Subtask 1",
) -> None:
    """Print a nicely formatted evaluation report."""
    sep = "=" * 60
    thin = "-" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Overall Accuracy    : {metrics['accuracy']:.4f}%")
    print(f"  Total Content Effect: {metrics['content_effect']:.4f}")
    print(f"  Combined Score (↑)  : {metrics['combined_score']:.4f}  ← ranking metric")
    print(thin)
    print("  Subgroup Accuracies:")
    print(f"    Valid   + Plausible   (PV): {metrics['acc_plausible_valid']:.2f}%")
    print(f"    Valid   + Implausible (IV): {metrics['acc_implausible_valid']:.2f}%")
    print(f"    Invalid + Plausible   (PI): {metrics['acc_plausible_invalid']:.2f}%")
    print(f"    Invalid + Implausible (II): {metrics['acc_implausible_invalid']:.2f}%")
    print(thin)
    print(f"  CE Intra (|PV-IV| + |PI-II|)/2 : {metrics['ce_intra']:.4f}")
    print(f"  CE Inter (|PV-PI| + |IV-II|)/2 : {metrics['ce_inter']:.4f}")
    print(thin)
    print(f"  N predicted: {metrics['n_predicted']}  |  N missing: {metrics['n_missing']}")
    print(sep)
    print("  Content Effect Legend:")
    print("    PV = Plausible+Valid  IV = Implausible+Valid")
    print("    PI = Plausible+Invalid  II = Implausible+Invalid")
    print(f"{sep}\n")


# ─── Evaluate from JSON Files ─────────────────────────────────────────────────

def evaluate_from_files(
    reference_path: str,
    predictions_path: str,
    output_metrics_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Load ground-truth and predictions from JSON, compute metrics, optionally save.

    Parameters
    ----------
    reference_path      : path to ground-truth JSON (with validity + plausibility)
    predictions_path    : path to predictions JSON (with id + validity only)
    output_metrics_path : optional path to write metric results JSON
    verbose             : if True, print the full report

    Returns
    -------
    dict of metrics
    """
    with open(reference_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    metrics = compute_metrics(ground_truth, predictions)

    if verbose:
        print_full_report(metrics)

    if output_metrics_path:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        # Save only the 3 keys expected by the Codabench evaluation script
        out = {
            "accuracy":      metrics["accuracy"],
            "content_effect": metrics["content_effect"],
            "combined_score": metrics["combined_score"],
        }
        with open(output_metrics_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Evaluate] Metrics saved to {output_metrics_path}")

    return metrics


# ─── Official Script Invocation ───────────────────────────────────────────────

def run_official_eval(
    reference_path: str = EVAL_KIT_REFERENCE,
    predictions_path: str = PREDICTIONS_PATH,
    output_path: str = EVAL_RESULTS_PATH,
) -> None:
    """
    Invoke the official evaluation_script.py as a subprocess.
    Useful for a final sanity-check before Codabench submission.
    """
    cmd = [
        sys.executable,
        EVAL_SCRIPT_PATH,
        "--reference_data_path", reference_path,
        "--prediction_path",     predictions_path,
        "--output_path",         output_path,
    ]
    print(f"[Evaluate] Running official eval script …")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if os.path.exists(output_path):
            with open(output_path) as f:
                print("Official results:", json.load(f))
    except Exception as e:
        print(f"[Evaluate] Could not run official script: {e}")
        print("  → Use evaluate_from_files() for in-process evaluation instead.")
