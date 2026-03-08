"""
subtask1_to_subtask3.py -- Multilingual translation pipeline for Subtask 3.

Translates Subtask 1 training data (train_data.json) into 11 languages
to produce train_data_subtask_3_format.json for multilingual syllogistic
reasoning (Subtask 3).

Input:  train_data.json  (960 entries: id, syllogism, validity, plausibility)
Output: train_data_subtask_3_format.json  (10,560 entries: 960 x 11 langs)

Uses deep-translator (Google) with 2 concurrent workers and 0.5s throttle.
"""

import json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
INPUT   = os.path.join(_PROJECT_ROOT, "train_data.json")
OUTPUT  = os.path.join(_PROJECT_ROOT, "train_data_subtask_3_format.json")
CKPT    = os.path.join(_PROJECT_ROOT, ".translate_subtask3_checkpoint.json")

LANGS = ["it", "es", "fr", "zh-CN", "de", "te", "sw", "pt", "nl", "bn", "ru"]

WORKERS    = 2
RETRIES    = 3
CHUNK      = 50
THROTTLE   = 0.5    # seconds between submissions

# ------------------------------------------------------------------ #
def translate_one(text, dest):
    """Translate a single text with retries."""
    for att in range(RETRIES):
        try:
            r = GoogleTranslator(source="en", target=dest).translate(text)
            if r:
                return r
        except Exception:
            wait = (2 ** att) + random.uniform(1, 3)
            if att < RETRIES - 1:
                time.sleep(wait)
    return None


def _load_ckpt():
    if os.path.exists(CKPT):
        try:
            with open(CKPT) as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": {}, "results": []}


def _save_ckpt(ck):
    tmp = CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ck, f, ensure_ascii=False)
    os.replace(tmp, CKPT)


# ------------------------------------------------------------------ #
def main():
    with open(INPUT) as f:
        data = json.load(f)

    N     = len(data)
    total = N * len(LANGS)

    ck      = _load_ckpt()
    done    = ck["done"]
    results = ck["results"]

    sys.stdout.write("=== Subtask 3 Translation Pipeline (deep-translator) ===\n")
    sys.stdout.write(f"Entries: {N} | Langs: {len(LANGS)} | Total: {total}\n")
    sys.stdout.write(f"Workers: {WORKERS} | Chunk: {CHUNK} | Throttle: {THROTTLE}s\n")
    if done:
        sys.stdout.write(f"Resuming: {len(done)} already done\n")
    sys.stdout.flush()

    t0 = time.time()

    for li, lang in enumerate(LANGS):
        todo = [i for i in range(N) if f"{data[i]['id']}_{lang}" not in done]
        ndone_lang = N - len(todo)

        sys.stdout.write(f"\n{'='*55}\n"
                         f"  [{li+1}/11] lang={lang}  "
                         f"done={ndone_lang}  todo={len(todo)}\n"
                         f"{'='*55}\n")
        sys.stdout.flush()

        if not todo:
            continue

        for cs in range(0, len(todo), CHUNK):
            chunk_idx = todo[cs:cs+CHUNK]
            chunk_texts = [data[i]["syllogism"] for i in chunk_idx]

            translations = [None] * len(chunk_idx)
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                future_map = {}
                for j, text in enumerate(chunk_texts):
                    f = pool.submit(translate_one, text, lang)
                    future_map[f] = j
                    time.sleep(THROTTLE)

                for f in as_completed(future_map):
                    j = future_map[f]
                    try:
                        translations[j] = f.result()
                    except Exception:
                        translations[j] = None

            chunk_failed = 0
            for j, idx in enumerate(chunk_idx):
                e = data[idx]
                t_text = translations[j]
                if not t_text:
                    t_text = e["syllogism"]
                    chunk_failed += 1

                results.append({
                    "id":           e["id"],
                    "syllogism":    e["syllogism"],
                    "validity":     e["validity"],
                    "plausibility": e["plausibility"],
                    "syllogism_t":  t_text,
                    "lang":         lang,
                })
                done[f"{e['id']}_{lang}"] = 1

            ndone_lang += len(chunk_idx)
            overall = len(done)
            elapsed = time.time() - t0
            rate = overall / elapsed if elapsed > 0 else 0
            remaining = total - overall
            eta = remaining / rate / 60 if rate > 0 else 0
            fb_str = f" [{chunk_failed} fb]" if chunk_failed else ""

            sys.stdout.write(
                f"  {lang} {ndone_lang:>4}/{N}  "
                f"| {overall:>5}/{total} ({overall/total*100:.1f}%)  "
                f"| ETA {eta:.0f}m  | {rate:.1f}/s{fb_str}\n"
            )
            sys.stdout.flush()

            ck["done"], ck["results"] = done, results
            _save_ckpt(ck)

    # ---- REPAIR PASS ----
    fb_indices = [i for i, e in enumerate(results)
                  if e["syllogism"] == e["syllogism_t"]]

    if fb_indices:
        sys.stdout.write(f"\n{'='*55}\n"
                         f"  REPAIR: {len(fb_indices)} fallback entries\n"
                         f"{'='*55}\n")
        sys.stdout.write("  Cooling 30s...\n")
        sys.stdout.flush()
        time.sleep(30)

        repaired = 0
        still_fb = 0
        for ri, idx in enumerate(fb_indices):
            e = results[idx]
            translated = translate_one(e["syllogism"], e["lang"])
            if translated:
                results[idx]["syllogism_t"] = translated
                repaired += 1
            else:
                still_fb += 1
            time.sleep(1.0)

            if (ri + 1) % 20 == 0:
                sys.stdout.write(f"  Repair {ri+1}/{len(fb_indices)}  "
                                 f"fixed={repaired} failed={still_fb}\n")
                sys.stdout.flush()
                ck["results"] = results
                _save_ckpt(ck)

        sys.stdout.write(f"  Repair: {repaired} fixed, {still_fb} remain\n")
        sys.stdout.flush()

    # ---- WRITE OUTPUT ----
    sys.stdout.write(f"\nWriting {len(results)} entries -> {OUTPUT}\n")
    sys.stdout.flush()
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if os.path.exists(CKPT):
        os.remove(CKPT)

    # ---- REPORT ----
    from collections import Counter
    lc = Counter(e["lang"] for e in results)
    fb_entries = [e for e in results if e["syllogism"] == e["syllogism_t"]]
    fb_by_lang = Counter(e["lang"] for e in fb_entries)

    sys.stdout.write("\n=== FINAL REPORT ===\n")
    sys.stdout.write(f"Total: {len(results)} entries\n")
    sys.stdout.write(f"Fallback: {len(fb_entries)}\n\n")
    for lang in LANGS:
        c = lc.get(lang, 0)
        fb = fb_by_lang.get(lang, 0)
        pct = (c - fb) / c * 100 if c > 0 else 0
        sys.stdout.write(f"  {lang:>5}: {c} entries, {fb} fallback ({pct:.1f}% ok)\n")

    sys.stdout.write("\n-- Samples --\n")
    seen = set()
    for e in results:
        if e["lang"] not in seen and e["syllogism"] != e["syllogism_t"]:
            seen.add(e["lang"])
            t = e["syllogism_t"][:150]
            if len(e["syllogism_t"]) > 150:
                t += "..."
            sys.stdout.write(f"  [{e['lang']:>5}] {t}\n")
        if len(seen) == len(LANGS):
            break

    elapsed = time.time() - t0
    sys.stdout.write(f"\nTotal: {elapsed/60:.1f} min\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
