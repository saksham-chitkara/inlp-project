"""
per_sample_eval.py
------------------
Generate per-sample comparison of all 3 approaches against ground truth.
Output: per_sample_results.json in the outputs directory.
"""
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_PATH = os.path.join(BASE_DIR, "semeval_2026_task_11", "test_data", "subtask 2", "test_data_subtask_2.json")
OUT_DIR = os.path.join(BASE_DIR, "src", "subtask2", "outputs")

def main():
    with open(TEST_PATH) as f:
        gt = json.load(f)

    preds = {}
    for label, fname in [("approach1", "predictions_approach1.json"),
                         ("approach2", "predictions_approach2.json"),
                         ("approach3", "predictions_llama.json")]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                pred_list = json.load(f)
            preds[label] = {p["id"]: p for p in pred_list}
            print(f"Loaded {label}: {len(pred_list)} items")
        else:
            print(f"SKIP {label}: {path} not found")

    results = []
    for item in gt:
        uid = item["id"]
        # Count premises by splitting syllogism
        sents = [s.strip() for s in item["syllogism"].split(".") if s.strip()]
        num_premises = len(sents) - 1 if len(sents) > 1 else len(sents)

        row = {
            "id": uid,
            "gt_validity": item["validity"],
            "gt_plausibility": item.get("plausibility"),
            "gt_relevant_premises": item.get("relevant_premises", []),
            "num_premises": num_premises,
        }
        for label, pred_map in preds.items():
            p = pred_map.get(uid, {})
            row[f"{label}_validity"] = p.get("validity")
            row[f"{label}_premises"] = p.get("relevant_premises", [])
            row[f"{label}_validity_correct"] = (p.get("validity") == item["validity"])
            if item["validity"] and item.get("relevant_premises"):
                row[f"{label}_premises_correct"] = (
                    sorted(p.get("relevant_premises", [])) == sorted(item["relevant_premises"])
                )
            else:
                row[f"{label}_premises_correct"] = None
        results.append(row)

    out_path = os.path.join(OUT_DIR, "per_sample_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Per-sample results: {len(results)} items saved to {out_path}")


if __name__ == "__main__":
    main()
