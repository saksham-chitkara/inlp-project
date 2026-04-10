#!/usr/bin/env python3
"""
translate_dataset.py
--------------------
Translates subtask2 generated training data into 11 target languages,
producing a multilingual training dataset for subtask4.

Input:  1920 English items from subtask2/outputs/generated_train_data.json
Output: 1920 × 11 = 21,120 items in subtask4 format:
        {id, syllogism, syllogism_t, lang, validity, plausibility, relevant_premises}

Uses deep_translator (Google Translate) with rate-limiting and retry logic.

Usage (local machine with internet):
  pip install deep-translator
  python final/src/subtask4/translate_dataset.py
"""

import json
import os
import sys
import time
import uuid
import hashlib
from typing import List, Dict

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("Installing deep-translator ...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
    from deep_translator import GoogleTranslator

# ─── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Input: generated subtask2 training data
S2_GENERATED_PATH = os.path.join(
    BASE_DIR, "src", "subtask2", "outputs", "generated_train_data.json"
)

# Output: translated training data for subtask4
OUTPUT_DIR = os.path.join(BASE_DIR, "semeval_2026_task_11", "train_data", "subtask 4")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_data_translated.json")

# Checkpoint path (for resuming interrupted translations)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "translation_checkpoint.json")

# Target languages (same as subtask3/subtask4 test set)
TARGET_LANGUAGES = [
    "bn",     # Bengali
    "de",     # German
    "es",     # Spanish
    "fr",     # French
    "it",     # Italian
    "nl",     # Dutch
    "pt",     # Portuguese
    "ru",     # Russian
    "sw",     # Swahili
    "te",     # Telugu
    "zh-CN",  # Chinese (Simplified)
]

# deep_translator uses different language codes for some
LANG_MAP = {
    "bn": "bn",
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "nl": "nl",
    "pt": "pt",
    "ru": "ru",
    "sw": "sw",
    "te": "te",
    "zh-CN": "zh-CN",
}

# Rate limiting
DELAY_BETWEEN_REQUESTS = 0.05  # seconds between API calls (Google allows ~5 qps)
BATCH_SAVE_INTERVAL    = 500   # save checkpoint every N translations


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_id(source_id: str, lang: str) -> str:
    """Generate a deterministic UUID for a (source_item, language) pair."""
    seed = f"{source_id}_{lang}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def translate_text(text: str, target_lang: str, max_retries: int = 3) -> str:
    """Translate text to target language with retry logic."""
    lang_code = LANG_MAP.get(target_lang, target_lang)
    for attempt in range(max_retries):
        try:
            translator = GoogleTranslator(source="en", target=lang_code)
            result = translator.translate(text)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 2
                print(f"    [retry {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    [FAIL] Could not translate to {target_lang}: {e}")
                return text  # fallback to English


def load_checkpoint() -> Dict[str, str]:
    """Load checkpoint of already-translated items."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(translations: Dict[str, str]):
    """Save checkpoint mapping (source_id + lang) → translated text."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load source data
    if not os.path.exists(S2_GENERATED_PATH):
        print(f"ERROR: Source file not found: {S2_GENERATED_PATH}")
        print("Run subtask2 dataset_generator first.")
        sys.exit(1)

    with open(S2_GENERATED_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    print(f"Loaded {len(source_data)} items from subtask2 generated data")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} translations already done")

    # Translate
    translated_items: List[Dict] = []
    total = len(source_data) * len(TARGET_LANGUAGES)
    done = 0
    new_translations = 0

    for item_idx, item in enumerate(source_data):
        source_id = item["id"]
        syllogism_en = item["syllogism"]

        for lang in TARGET_LANGUAGES:
            cache_key = f"{source_id}_{lang}"
            new_id = make_id(source_id, lang)

            # Check checkpoint
            if cache_key in checkpoint:
                translated_text = checkpoint[cache_key]
            else:
                translated_text = translate_text(syllogism_en, lang)
                checkpoint[cache_key] = translated_text
                new_translations += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

                # Periodic checkpoint save
                if new_translations % BATCH_SAVE_INTERVAL == 0:
                    save_checkpoint(checkpoint)
                    print(f"  [checkpoint] Saved at {new_translations} new translations")

            translated_items.append({
                "id": new_id,
                "syllogism": syllogism_en,
                "syllogism_t": translated_text,
                "lang": lang,
                "validity": item["validity"],
                "plausibility": item.get("plausibility", None),
                "relevant_premises": item["relevant_premises"],
                "source_id": source_id,  # maps back to subtask2 item
            })

            done += 1
            if done % 500 == 0:
                pct = done / total * 100
                print(f"  [{done}/{total}]  {pct:.1f}%  "
                      f"(new: {new_translations})  "
                      f"item {item_idx+1}/{len(source_data)}, lang={lang}")

    # Final save
    save_checkpoint(checkpoint)

    # Save translated dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(translated_items)} translated items saved to:")
    print(f"  {OUTPUT_PATH}")
    print(f"  New translations: {new_translations}")
    print(f"  From checkpoint:  {done - new_translations}")

    # Summary
    lang_counts = {}
    for item in translated_items:
        lang_counts[item["lang"]] = lang_counts.get(item["lang"], 0) + 1
    print("\nPer-language counts:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
