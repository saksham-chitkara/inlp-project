"""
evaluate.py
-----------
Evaluation for Subtask 4 (same metrics as Subtask 2: accuracy + premise F1 + TCE).
"""

import os
import sys
import json
import math
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(__file__))
from src.subtask4.rest3.config import EVAL_RESULTS_PATH, TEST_DATA_PATH


def compute_subgroup_acc(gt_map, predictions, gt_validity, gt_plausibility):
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


def compute_f1_premises(gt_map, predictions):
    total_precision = total_recall = 0.0
    valid_count = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is None:
            continue
        gt_premises = gt.get("relevant_premises", [])
        if not gt_premises:
            continue
        pred_premises = set(pred.get("relevant_premises", []))
        true_set = set(gt_premises)
        tp = len(true_set & pred_premises)
        fp = len(pred_premises - true_set)
        fn = len(true_set - pred_premises)
        total_precision += tp / (tp + fp) if (tp + fp) > 0 else 0.0
        total_recall    += tp / (tp + fn) if (tp + fn) > 0 else 0.0
        valid_count += 1

    if valid_count == 0:
        return {"premise_precision": 0.0, "premise_recall": 0.0,
                "premise_f1": 0.0, "premise_n": 0}

    mp = total_precision / valid_count
    mr = total_recall / valid_count
    f1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    return {
        "premise_precision": round(mp * 100, 4),
        "premise_recall": round(mr * 100, 4),
        "premise_f1": round(f1 * 100, 4),
        "premise_n": valid_count,
    }


def compute_metrics(ground_truth, predictions):
    gt_map   = {item["id"]: item for item in ground_truth}
    pred_ids = {p["id"] for p in predictions}
    missing  = len(set(gt_map.keys()) - pred_ids)

    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt is not None and isinstance(pred.get("validity"), bool):
            total += 1
            if pred["validity"] == gt["validity"]:
                correct += 1
    overall_acc = (correct / total * 100) if total > 0 else 0.0

    a_pv = compute_subgroup_acc(gt_map, predictions, True, True)
    a_iv = compute_subgroup_acc(gt_map, predictions, True, False)
    a_pi = compute_subgroup_acc(gt_map, predictions, False, True)
    a_ii = compute_subgroup_acc(gt_map, predictions, False, False)

    ce_intra = (abs(a_pv - a_iv) + abs(a_pi - a_ii)) / 2.0
    ce_inter = (abs(a_pv - a_pi) + abs(a_iv - a_ii)) / 2.0
    tce = (ce_intra + ce_inter) / 2.0

    f1_info = compute_f1_premises(gt_map, predictions)
    overall_perf = (overall_acc + f1_info["premise_f1"]) / 2.0
    combined = overall_perf / (1 + math.log(1 + tce))

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


def print_full_report(metrics, title="Evaluation Report — Subtask 4"):
    sep  = "=" * 65
    thin = "-" * 65
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Overall Accuracy       : {metrics['accuracy']:.4f}%")
    print(f"  Premise F1             : {metrics['premise_f1']:.4f}%  "
          f"(P={metrics['premise_precision']:.2f}%  R={metrics['premise_recall']:.2f}%  "
          f"n={metrics['premise_n']})")
    print(f"  Overall Performance    : {metrics['overall_perf']:.4f}")
    print(f"  Total Content Effect   : {metrics['content_effect']:.4f}")
    print(f"  Combined Score (↑)     : {metrics['combined_score']:.4f}")
    print(thin)
    print("  Subgroup Accuracies:")
    print(f"    PV: {metrics['acc_plausible_valid']:.2f}%  "
          f"IV: {metrics['acc_implausible_valid']:.2f}%  "
          f"PI: {metrics['acc_plausible_invalid']:.2f}%  "
          f"II: {metrics['acc_implausible_invalid']:.2f}%")
    print(f"  CE Intra: {metrics['ce_intra']:.4f}  CE Inter: {metrics['ce_inter']:.4f}")
    print(f"  N predicted: {metrics['n_predicted']}  |  N missing: {metrics['n_missing']}")
    print(f"{sep}\n")


def print_comparison(rows, title="APPROACH COMPARISON"):
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    header = f"  {'Approach':<30} {'Acc':>8} {'PremF1':>8} {'TCE':>8} {'Combined':>10}"
    print(header)
    print(f"  {'-'*70}")
    for row in rows:
        m = row["metrics"]
        print(f"  {row['label']:<30} {m['accuracy']:>7.2f}% "
              f"{m['premise_f1']:>7.2f}% {m['content_effect']:>8.4f} "
              f"{m['combined_score']:>10.4f}")
    print(f"{sep}\n")


def evaluate_from_files(reference_path, predictions_path,
                        output_metrics_path=None, verbose=True,
                        title="Evaluation Report — Subtask 4"):
    with open(reference_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    metrics = compute_metrics(ground_truth, predictions)
    if verbose:
        print_full_report(metrics, title=title)

    if output_metrics_path:
        os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
        with open(output_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics
