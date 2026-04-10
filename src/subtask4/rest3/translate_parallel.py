#!/usr/bin/env python3
"""
translate_parallel.py  –  Concurrent translation using ThreadPoolExecutor.
Resumes from the existing checkpoint saved by translate_dataset.py.
Uses 10 workers for ~10x throughput.
"""

import json, os, sys, time, uuid, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from deep_translator import GoogleTranslator
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
    from deep_translator import GoogleTranslator

# ─── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

S2_GENERATED_PATH = os.path.join(
    BASE_DIR, "src", "subtask2", "outputs", "generated_train_data.json"
)
OUTPUT_DIR  = os.path.join(BASE_DIR, "semeval_2026_task_11", "train_data", "subtask 4")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "train_data_translated.json")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "translation_checkpoint.json")

TARGET_LANGUAGES = ["bn","de","es","fr","it","nl","pt","ru","sw","te","zh-CN"]

NUM_WORKERS = 10          # parallel threads
SAVE_EVERY  = 500         # checkpoint interval (new translations)

# ─── Shared state ────────────────────────────────────────────────────────────
lock = threading.Lock()
checkpoint: dict = {}
new_count = 0
last_saved = 0

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_checkpoint_safe():
    global last_saved
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False)
    last_saved = new_count

def translate_one(cache_key: str, text: str, lang: str):
    """Translate a single text; returns (cache_key, translated_text)."""
    global new_count
    for attempt in range(3):
        try:
            result = GoogleTranslator(source="en", target=lang).translate(text)
            with lock:
                checkpoint[cache_key] = result
                new_count += 1
                if new_count % SAVE_EVERY == 0:
                    save_checkpoint_safe()
                    print(f"  [checkpoint] {new_count} new  |  total cached: {len(checkpoint)}")
            return cache_key, result
        except Exception as e:
            wait = (attempt + 1) * 2
            if attempt < 2:
                time.sleep(wait)
            else:
                print(f"  [FAIL] {lang} — {str(e)[:80]}")
                with lock:
                    checkpoint[cache_key] = text   # fallback to English
                    new_count += 1
                return cache_key, text

def main():
    global checkpoint, new_count
    # Load source
    with open(S2_GENERATED_PATH, "r", encoding="utf-8") as f:
        source_data = json.load(f)
    print(f"Source: {len(source_data)} items  ×  {len(TARGET_LANGUAGES)} langs  =  {len(source_data)*len(TARGET_LANGUAGES)} total")

    checkpoint = load_checkpoint()
    already = len(checkpoint)
    print(f"Checkpoint: {already} translations cached")

    # Build work list (skip already cached)
    work = []
    for item in source_data:
        sid = item["id"]
        for lang in TARGET_LANGUAGES:
            ck = f"{sid}_{lang}"
            if ck not in checkpoint:
                work.append((ck, item["syllogism"], lang))

    remaining = len(work)
    print(f"Remaining: {remaining} translations to do")
    if remaining == 0:
        print("Nothing to do, building output file...")
    else:
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            futures = {pool.submit(translate_one, ck, txt, lg): ck for ck, txt, lg in work}
            for fut in as_completed(futures):
                done += 1
                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (remaining - done) / rate / 60
                    print(f"  [{done}/{remaining}]  {done/remaining*100:.1f}%  rate={rate:.1f}/s  ETA={eta:.0f}min")

        # Final checkpoint save
        save_checkpoint_safe()
        elapsed = time.time() - t0
        print(f"\nTranslation done: {done} in {elapsed/60:.1f} min ({done/elapsed:.1f}/s)")

    # ─── Build output ────────────────────────────────────────────────────────
    translated_items = []
    for item in source_data:
        sid = item["id"]
        syl_en = item["syllogism"]
        for lang in TARGET_LANGUAGES:
            ck = f"{sid}_{lang}"
            new_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{sid}_{lang}"))
            translated_items.append({
                "id": new_id,
                "syllogism": syl_en,
                "syllogism_t": checkpoint.get(ck, syl_en),
                "lang": lang,
                "validity": item["validity"],
                "plausibility": item.get("plausibility", None),
                "relevant_premises": item["relevant_premises"],
                "source_id": sid,
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(translated_items, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(translated_items)} items to {OUTPUT_PATH}")
    lang_counts = {}
    for item in translated_items:
        lang_counts[item["lang"]] = lang_counts.get(item["lang"], 0) + 1
    for lang, c in sorted(lang_counts.items()):
        print(f"  {lang}: {c}")


if __name__ == "__main__":
    main()
