"""
generate_quasar_subtask2.py
---------------------------
Generate QuaSAR cache for subtask2 test data using Llama 3.1-8B-Instruct.
Reuses the quasar_generator.py from subtask1.

Usage:
    cd /ssd_scratch/shubhamcvit/inlp/project
    python3 src/subtask2/generate_quasar_subtask2.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
S1_DIR = os.path.join(SCRIPT_DIR, "..", "subtask1")
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Hardcode paths to avoid config import conflicts
TEST_DATA_PATH = os.path.join(
    BASE_DIR, "semeval_2026_task_11", "test_data", "subtask 2", "test_data_subtask_2.json"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "subtask2", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
QUASAR_S2_TEST_CACHE = os.path.join(OUTPUT_DIR, "quasar_s2_test_cache.json")
HF_CACHE_DIR = os.environ.get("HF_HOME", None)

# Import quasar_generator from subtask1 (needs subtask1's config on path)
sys.path.insert(0, S1_DIR)
from quasar_generator import load_llama_model, generate_quasar_batch


def main():
    # Load subtask2 test data
    print(f"Loading subtask2 test data from {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH) as f:
        test_data = json.load(f)
    print(f"  {len(test_data)} items")

    # Load existing cache if any
    existing = {}
    if os.path.exists(QUASAR_S2_TEST_CACHE):
        with open(QUASAR_S2_TEST_CACHE) as f:
            existing = json.load(f)
        print(f"  Existing cache: {len(existing)} entries")

    # Load Llama
    print("Loading Llama model...")
    model, tokenizer = load_llama_model(
        cache_dir=HF_CACHE_DIR,
        use_4bit=False,
    )

    # Generate
    cache = generate_quasar_batch(
        data=test_data,
        model=model,
        tokenizer=tokenizer,
        output_path=QUASAR_S2_TEST_CACHE,
        existing_cache=existing,
        max_new_tokens=512,
        save_every=25,
    )

    # Save final
    with open(QUASAR_S2_TEST_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"\nQuaSAR cache saved: {QUASAR_S2_TEST_CACHE} ({len(cache)} entries)")


if __name__ == "__main__":
    main()
