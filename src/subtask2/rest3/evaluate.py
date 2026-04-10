"""
evaluate.py
-----------
Evaluation functions for SemEval-2026 Task 11 — Subtask 2.

Mirrors the official evaluation_script.py (task 2 & 4) exactly,
plus convenience utilities for in-process evaluation during development.

Official metric hierarchy (ranking = combined_score):
  1. Accuracy:       overall validity classification accuracy
  2. F1 (premises):  macro-averaged F1 for premise retrieval (valid items only)
  3. Content Effect:  TCE = (CE_intra + CE_inter) / 2
     CE_intra = (|acc_PV - acc_IV| + |acc_PI - acc_II|) / 2
     CE_inter = (|acc_PV - acc_PI| + |acc_IV - acc_II|) / 2
  4. Overall Perf:   (accuracy + f1_premises) / 2
  5. Combined Score:  overall_perf / (1 + ln(1 + TCE))   ← ranking metric
"""

import os
import sys
import json
import math
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from src.subtask2.rest3.config import EVAL_RESULTS_PATH, TEST_DATA_PATH


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


def compute_f1_premises(
    gt_map: Dict[str, Any],
    predictions: List[Dict],
) -> Dict[str, float]:
    """
    Macro-averaged F1 for premise retrieval (official definition).
    Only computed over items where ground truth is valid AND has relevant_premises.

    Returns dict with precision, recall, f1 (all as percentages).
    """
    total_precision = 0.0
    total_recall = 0.0
    valid_count = 0

    for pred in predictions:
        item_id = pred["id"]
        gt = gt_map.get(item_id)
        if gt is None:
            continue
        gt_premises = gt.get("relevant_premises", [])
        pred_premises = pred.get("relevant_premises", [])

        if not gt_premises:  # skip invalid items (empty relevant_premises)
            continue

        true_positives = set(gt_premises)
        predicted_positives = set(pred_premises)

        tp = len(true_positives.intersection(predicted_positives))
        fp = len(predicted_positives.difference(true_positives))
        fn = len(true_positives.difference(predicted_positives))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        valid_count += 1

    if valid_count == 0:
        return {"premise_precision": 0.0, "premise_recall": 0.0, "premise_f1": 0.0, "premise_n": 0}

    macro_precision = total_precision / valid_count
    macro_recall = total_recall / valid_count

    f1 = (2 * macro_precision * macro_recall / (macro_precision + macro_recall)
          if (macro_precision + macro_recall) > 0 else 0.0)

    return {
        "premise_precision": round(macro_precision * 100, 4),
        "premise_recall": round(macro_recall * 100, 4),
        "premise_f1": round(f1 * 100, 4),
        "premise_n": valid_count,
    }


def compute_metrics(
    ground_truth: List[Dict],
    predictions: List[Dict],
) -> Dict[str, float]:
    """
    Compute all subtask 2 metrics:
      - accuracy, f1_premises, content_effect, combined_score
      - subgroup accuracies (PV, IV, PI, II)
      - CE intra/inter components

    Returns a flat dict of all metrics.
    """
    gt_map = {item["id"]: item for item in ground_truth}
    pred_ids = {p["id"] for p in predictions}
    gt_ids = set(gt_map.keys())
    missing = len(gt_ids - pred_ids)

    if missing > 0:
        print(f"  ⚠ {missing} ground-truth examples have no prediction!")

    # ── Overall accuracy ──
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is not None and isinstance(pred.get("validity"), bool):
            total += 1
            if pred["validity"] == gt["validity"]:
                correct += 1
    overall_acc = (correct / total * 100) if total > 0 else 0.0

    # ── Subgroup accuracies ──
    a_pv = compute_subgroup_acc(gt_map, predictions, True, True)
    a_iv = compute_subgroup_acc(gt_map, predictions, True, False)
    a_pi = compute_subgroup_acc(gt_map, predictions, False, True)
    a_ii = compute_subgroup_acc(gt_map, predictions, False, False)

    # ── Content effect ──
    intra_valid = abs(a_pv - a_iv)
    intra_invalid = abs(a_pi - a_ii)
    ce_intra = (intra_valid + intra_invalid) / 2.0

    inter_plaus = abs(a_pv - a_pi)
    inter_implaus = abs(a_iv - a_ii)
    ce_inter = (inter_plaus + inter_implaus) / 2.0

    tce = (ce_intra + ce_inter) / 2.0

    # ── Premise F1 ──
    f1_info = compute_f1_premises(gt_map, predictions)

    # ── Combined score (official ranking metric for task 2) ──
    overall_perf = (overall_acc + f1_info["premise_f1"]) / 2.0
    log_penalty = math.log(1 + tce)
    combined = overall_perf / (1 + log_penalty)

    return {
        "accuracy": round(overall_acc, 4),
        "premise_precision": f1_info["premise_precision"],
        "premise_recall": f1_info["premise_recall"],
        "premise_f1": f1_info["premise_f1"],
        "premise_n": f1_info["premise_n"],
        "content_effect": round(tce, 4),
        "combined_score": round(combined, 4),
        "overall_perf": round(overall_perf, 4),
        "acc_plausible_valid": round(a_pv, 2),
        "acc_implausible_valid": round(a_iv, 2),
        "acc_plausible_invalid": round(a_pi, 2),
        "acc_implausible_invalid": round(a_ii, 2),
        "ce_intra": round(ce_intra, 4),
        "ce_inter": round(ce_inter, 4),
        "n_predicted": total,
        "n_missing": missing,
    }


# ─── Pretty Report ────────────────────────────────────────────────────────────

def print_full_report(
    metrics: Dict[str, float],
    title: str = "Evaluation Report — Subtask 2",
) -> None:
    """Print a nicely formatted evaluation report."""
    sep = "=" * 65
    thin = "-" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Overall Accuracy       : {metrics['accuracy']:.4f}%")
    print(f"  Premise F1             : {metrics['premise_f1']:.4f}%  "
          f"(P={metrics['premise_precision']:.2f}%  R={metrics['premise_recall']:.2f}%  "
          f"n={metrics['premise_n']})")
    print(f"  Overall Performance    : {metrics['overall_perf']:.4f}  "
          f"= (Acc + F1) / 2")
    print(f"  Total Content Effect   : {metrics['content_effect']:.4f}")
    print(f"  Combined Score (↑)     : {metrics['combined_score']:.4f}  "
          f"← ranking metric")
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
    title: str = "Evaluation Report — Subtask 2",
) -> Dict[str, float]:
    """
    Load ground-truth and predictions from JSON, compute metrics, optionally save.
    """
    with open(reference_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    metrics = compute_metrics(ground_truth, predictions)

    if verbose:
        print_full_report(metrics, title=title)

    if output_metrics_path:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        out = {
            "accuracy": metrics["accuracy"],
            "f1_premises": metrics["premise_f1"],
            "content_effect": metrics["content_effect"],
            "combined_score": metrics["combined_score"],
        }
        with open(output_metrics_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Evaluate] Metrics saved to {output_metrics_path}")

    return metrics


# ─── Print Comparison Table ────────────────────────────────────────────────────

def print_comparison(
    rows: List[Dict],
    title: str = "COMPARISON",
) -> None:
    """
    Print a comparison table for multiple evaluation runs.

    Parameters
    ----------
    rows : list of dicts, each with keys:
      "label"           : str (e.g. "Baseline", "+QuaSAR", "+Steering")
      "metrics"         : dict from compute_metrics()
    """
    sep = "=" * 80
    thin = "-" * 80
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    labels = [r["label"] for r in rows]
    max_lbl = max(len(l) for l in labels)
    header = f"  {'Metric':<25}"
    for lbl in labels:
        header += f" {lbl:>{max(14, max_lbl+2)}}"
    print(header)
    print(f"  {thin[:len(header)-2]}")

    for key, fmt, suffix in [
        ("accuracy", ".2f", "%"),
        ("premise_f1", ".2f", "%"),
        ("overall_perf", ".2f", ""),
        ("content_effect", ".4f", ""),
        ("combined_score", ".4f", " ←"),
    ]:
        row_str = f"  {key:<25}"
        for r in rows:
            m = r["metrics"]
            val = m.get(key, 0.0)
            row_str += f" {val:{fmt}}{suffix}".rjust(max(14, max_lbl + 2))
        print(row_str)

    print(f"  {thin[:len(header)-2]}")

    # Subgroups for each
    for r in rows:
        m = r["metrics"]
        print(f"  {r['label']}: PV={m['acc_plausible_valid']:.1f}  "
              f"IV={m['acc_implausible_valid']:.1f}  "
              f"PI={m['acc_plausible_invalid']:.1f}  "
              f"II={m['acc_implausible_invalid']:.1f}")
    print(sep)
