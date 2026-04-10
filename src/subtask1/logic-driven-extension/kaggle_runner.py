#!/usr/bin/env python3
"""
Kaggle Runner for LReasoner Hyperparameter Sweep.

This is the ONLY script you need to run on Kaggle.
Paste this entire file into a single Kaggle Notebook code cell and run.

Steps:
  1. Upload logic_sweep_kaggle.zip as a Kaggle Dataset ("logic-sweep-src")
  2. Create a new Kaggle Notebook, Accelerator: GPU T4 x2
  3. Add your uploaded dataset
  4. Paste this script into the first code cell and run
"""

import subprocess
import sys
import os
import json
import glob

# ======================================================================
# 1. Install spaCy model (needed for NLP preprocessing)
# ======================================================================
print("[Setup] Installing spaCy English model ...")
subprocess.check_call(
    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
    stdout=subprocess.DEVNULL,
)
print("[Setup] spaCy model installed.\n")

# ======================================================================
# 2. Paths (adjust DATASET_NAME if you named it differently)
# ======================================================================
DATASET_NAME = "logic-sweep-src"
DATASET_INPUT = f"/kaggle/input/{DATASET_NAME}"
WORKING_DIR = "/kaggle/working"

SRC_DIR = os.path.join(DATASET_INPUT, "src")
TRAIN_DATA = os.path.join(DATASET_INPUT, "data", "train_data.json")
OUTPUT_DIR = os.path.join(WORKING_DIR, "sweep_results")

# Verify dataset is accessible
assert os.path.isdir(SRC_DIR), (
    f"Source directory not found at {SRC_DIR}.\n"
    f"Make sure you uploaded the zip as a dataset named '{DATASET_NAME}'.\n"
    f"Contents of /kaggle/input/: {os.listdir('/kaggle/input/')}"
)
assert os.path.isfile(TRAIN_DATA), (
    f"Training data not found at {TRAIN_DATA}.\n"
    f"Contents of dataset: {os.listdir(DATASET_INPUT)}"
)

# ======================================================================
# 3. Add source to Python path
# ======================================================================
sys.path.insert(0, SRC_DIR)
os.chdir(WORKING_DIR)

# ======================================================================
# 4. Verify GPU
# ======================================================================
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Setup] Device: {device}")
if device == "cuda":
    print(f"[Setup] GPU   : {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[Setup] VRAM  : {gpu_mem:.1f} GB")
else:
    print("[WARNING] No GPU detected! Sweep will be extremely slow.")
print()

# ======================================================================
# 5. Set seeds
# ======================================================================
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ======================================================================
# 6. Configuration
# ======================================================================
N_CONFIGS = 60    # number of random HP configs to evaluate
N_FOLDS = 5       # stratified CV folds
VERBOSE = True    # per-epoch logs (set False to reduce output)

# ======================================================================
# 7. Run the sweep
# ======================================================================
from sweep import run_sweep

class SweepArgs:
    """Mimics argparse namespace for run_sweep()."""
    train_data = TRAIN_DATA
    device = device
    n_folds = N_FOLDS
    n_configs = N_CONFIGS
    seed = SEED
    output_dir = OUTPUT_DIR
    verbose = VERBOSE

args = SweepArgs()

print("=" * 70)
print(f"  HYPERPARAMETER SWEEP")
print(f"  Configs   : {N_CONFIGS}")
print(f"  Folds     : {N_FOLDS}")
print(f"  Total runs: {N_CONFIGS * N_FOLDS}")
print(f"  Train data: {TRAIN_DATA}")
print(f"  Output    : {OUTPUT_DIR}")
print("=" * 70)
print()

best_config, all_results = run_sweep(args)

# ======================================================================
# 8. Save best config summary
# ======================================================================
summary_path = os.path.join(OUTPUT_DIR, "best_config_summary.json")
with open(summary_path, "w") as f:
    json.dump(best_config, f, indent=2)
print(f"\n[Result] Best config saved to: {summary_path}")

# ======================================================================
# 9. Train final model with best hyperparameters
# ======================================================================
print("\n" + "=" * 70)
print("  TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
print("=" * 70)

from train_best import train_full, predict_test

cfg = best_config["hyperparameters"]
print(f"\nBest hyperparameters:")
print(json.dumps(cfg, indent=2))
print(f"CV combined_score: {best_config['mean_combined_score']}")
print()

model, tokenizer = train_full(cfg, TRAIN_DATA, device)

# Save model
model_path = os.path.join(WORKING_DIR, "best_lreasoner_model.pt")
torch.save(model.state_dict(), model_path)
print(f"\n[Result] Model saved to: {model_path}")

# ======================================================================
# 10. Generate test predictions if test data is available
# ======================================================================
test_data_path = os.path.join(DATASET_INPUT, "data", "test_data_subtask_1.json")
if os.path.isfile(test_data_path):
    preds_path = os.path.join(WORKING_DIR, "predictions_subtask_1.json")
    predict_test(model, tokenizer, test_data_path, cfg, device, preds_path)
    print(f"[Result] Predictions saved to: {preds_path}")
else:
    print("[Info] No test data found; skipping prediction.")

# ======================================================================
# 11. Final summary
# ======================================================================
print("\n" + "=" * 70)
print("  ALL DONE!")
print("=" * 70)
print(f"\nFiles in /kaggle/working/:")
for f in sorted(glob.glob(os.path.join(WORKING_DIR, "**"), recursive=True)):
    if os.path.isfile(f):
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.relpath(f, WORKING_DIR):50s} ({size_mb:.1f} MB)")

print(f"\nBest combined_score (CV): {best_config['mean_combined_score']}")
print(f"Best accuracy (CV)      : {best_config['mean_accuracy']}")
print(f"Best content_effect (CV): {best_config['mean_content_effect']}")
print("\nDownload results from the Output tab on the right.")
print("=" * 70)
