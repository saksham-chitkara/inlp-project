import sys; sys.stdout.reconfigure(line_buffering=True)
"""
quasar_generator.py
-------------------
QuaSAR generation for Subtask 4 test items.

Subtask 4 test items contain 192 NEW English syllogisms (not in subtask2).
Each item has:
  - "syllogism"   : English original (ALWAYS use this for QuaSAR)
  - "syllogism_t" : Translated text (for model input, not QuaSAR)
  - "lang"        : Target language code

Since all 11 translations of the same syllogism share the same English
original, we de-duplicate and run QuaSAR once per unique English syllogism,
then all translations share the same cache entry.

Usage:
  python3 -u src/subtask4/quasar_generator.py
  python3 -u src/subtask4/quasar_generator.py --input <path> --output <path>
"""

import argparse
import json
import os
import re
import sys
import time
import logging
from typing import List, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.subtask4.rest3.config import (
    LLAMA_MODEL_NAME, HF_CACHE_DIR, USE_4BIT,
    TEST_DATA_PATH, OUTPUT_DIR, S4_QUASAR_TEST_CACHE,
    LLM_MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)


# ─── QuaSAR Prompt Template ──────────────────────────────────────────────────

QUASAR_PROMPT_TEMPLATE = """Analyze this syllogism for formal validity. Be concise and brief in each step.

Syllogism: {syllogism}

Complete ALL 4 steps below. Keep each step SHORT (2-4 lines max).

s1 (Abstraction): Identify predicates and replace terms with X, Y, Z.

s2 (Formalisation): Write the formal symbolic structure using X, Y, Z (e.g., All X are Y).

s3 (Explanation): Briefly check if the conclusion follows logically from the premises.

s4 (Answering): The answer is: [valid/invalid]"""


# ─── Extraction Helpers ──────────────────────────────────────────────────────

def extract_formalisation(quasar_output: str) -> str:
    """Extract the Formalisation (s2) section from QuaSAR output."""
    s2_headers = [
        r"(?:Formalisation|Formalization)\s*\(s2\)",
        r"(?:Step\s*2|s2)\s*[:\-]\s*(?:Formalisation|Formalization)",
        r"\*\*(?:Formalisation|Formalization)\s*\(s2\)\*\*",
        r"#{1,3}\s*(?:Formalisation|Formalization)\s*\(s2\)",
        r"2\)\s*(?:Formalisation|Formalization)",
        r"\*\*s2\*\*",
        r"(?:Formalisation|Formalization)\s*:",
    ]
    s3_headers = [
        r"(?:Explanation|Explaination)\s*\(s3\)",
        r"(?:Step\s*3|s3)\s*[:\-]\s*(?:Explanation|Explaination)",
        r"\*\*(?:Explanation|Explaination)\s*\(s3\)\*\*",
        r"#{1,3}\s*(?:Explanation|Explaination)\s*\(s3\)",
        r"3\)\s*(?:Explanation|Explaination)",
        r"\*\*s3\*\*",
        r"(?:Explanation|Explaination)\s*:",
    ]

    for s2_pat in s2_headers:
        for s3_pat in s3_headers:
            pattern = re.compile(
                s2_pat + r"\s*[:\-]?\s*(.*?)" + s3_pat,
                re.DOTALL | re.IGNORECASE
            )
            match = pattern.search(quasar_output)
            if match:
                s2_text = match.group(1).strip()
                if len(s2_text) > 10:
                    return re.sub(r"^[\d\)\.]+\s*", "", s2_text).strip()

    lines = quasar_output.split("\n")
    s2_started = False
    s2_lines = []
    for line in lines:
        if re.search(r"(?:Formalisation|Formalization|s2)", line, re.IGNORECASE) and not s2_started:
            s2_started = True
            after = re.sub(r".*(?:Formalisation|Formalization|s2)\s*[:\-\)]*\s*",
                           "", line, flags=re.IGNORECASE).strip()
            if after:
                s2_lines.append(after)
            continue
        if s2_started:
            if re.search(r"(?:Explanation|Explaination|s3|Step\s*3|Answering|s4)", line, re.IGNORECASE):
                break
            s2_lines.append(line)

    if s2_lines:
        text = "\n".join(s2_lines).strip()
        if len(text) > 10:
            return text

    symbolic_lines = []
    for line in lines:
        if re.search(r"(?:All|Some|No)\s+[A-Z]\b|[A-Z]\s*(?:→|->|⊂|⊆|∀|∃)|∀|∃|Premise|Conclusion", line):
            symbolic_lines.append(line.strip())
    if symbolic_lines:
        return "\n".join(symbolic_lines[:10]).strip()

    logger.warning("Could not extract s2 from QuaSAR output, using truncated output.")
    return quasar_output[:300].strip()


def extract_answer(quasar_output: str) -> Optional[bool]:
    """Extract the validity answer from QuaSAR's s4 section."""
    answer_pattern = re.compile(
        r"[Tt]he\s+answer\s+is\s*:?\s*(valid|invalid)", re.IGNORECASE)
    match = answer_pattern.search(quasar_output)
    if match:
        return match.group(1).lower() == "valid"
    last_200 = quasar_output[-200:].lower()
    if "invalid" in last_200:
        return False
    if "valid" in last_200:
        return True
    return None


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_llama_model(
    model_name: str = LLAMA_MODEL_NAME,
    use_4bit: bool = USE_4BIT,
    cache_dir: Optional[str] = HF_CACHE_DIR,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[QuaSAR-S4] Loading {model_name}...")
    token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
            cache_dir=cache_dir, token=token, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir,
            token=token, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[QuaSAR-S4] Model loaded: {n_params / 1e9:.1f}B params")
    return model, tokenizer


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_quasar_single(model, tokenizer, syllogism: str,
                           max_new_tokens: int = 512) -> str:
    prompt = QUASAR_PROMPT_TEMPLATE.format(syllogism=syllogism)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": "You are an expert logician. Follow the requested output format exactly."},
                {"role": "user", "content": prompt},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.1,
            top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_quasar_batch(
    data: List[Dict], model, tokenizer,
    output_path: str,
    existing_cache: Optional[Dict[str, Dict]] = None,
    max_new_tokens: int = 512, save_every: int = 10,
) -> Dict[str, Dict]:
    """
    Generate QuaSAR for subtask4 test items.

    De-duplicates by English syllogism text: all language variants of the
    same syllogism share one QuaSAR entry keyed by the first item's ID.
    """
    cache = existing_cache or {}

    # De-duplicate: group by English syllogism text
    syl_to_ids: Dict[str, List[str]] = {}
    syl_to_text: Dict[str, str] = {}
    for item in data:
        eng = item["syllogism"]
        syl_to_ids.setdefault(eng, []).append(item["id"])
        syl_to_text[eng] = eng

    total_unique = len(syl_to_ids)
    generated = 0
    skipped   = 0
    start_time = time.time()

    print(f"[QuaSAR-S4] {len(data)} items → {total_unique} unique English syllogisms")
    print(f"  Already cached: {len(cache)}")

    for i, (eng_syl, item_ids) in enumerate(syl_to_ids.items()):
        # Check if ANY id for this syllogism is already cached
        if any(iid in cache for iid in item_ids):
            skipped += 1
            # Ensure ALL ids for this syllogism are cached (copy entry)
            src_entry = next(cache[iid] for iid in item_ids if iid in cache)
            for iid in item_ids:
                if iid not in cache:
                    cache[iid] = src_entry.copy()
            continue

        quasar_output = generate_quasar_single(
            model, tokenizer, eng_syl, max_new_tokens=max_new_tokens)
        s2_text = extract_formalisation(quasar_output)
        answer  = extract_answer(quasar_output)

        entry = {
            "syllogism":     eng_syl,
            "quasar_full":   quasar_output,
            "quasar_s2":     s2_text,
            "quasar_answer": answer,
        }
        # Store under ALL item IDs sharing this English syllogism
        for iid in item_ids:
            cache[iid] = entry
        generated += 1

        if generated <= 3:
            print(f"\n{'='*60}")
            print(f"  DEBUG (#{generated}): ids={item_ids[:3]}")
            print(f"  Syllogism: {eng_syl[:120]}...")
            print(f"  --- Extracted s2 ---")
            print(f"  {s2_text[:200]}")
            print(f"  --- Extracted answer: {answer} ---")
            print(f"{'='*60}\n")

        elapsed = time.time() - start_time
        rate = generated / elapsed if elapsed > 0 else 0
        eta  = (total_unique - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 5 == 0 or (i + 1) == total_unique:
            print(f"  [{i+1}/{total_unique}] Generated: {generated}, "
                  f"Skipped: {skipped}, Rate: {rate:.1f} ex/s, ETA: {eta/60:.1f} min")

        if (generated > 0 and generated % save_every == 0) or (i + 1) == total_unique:
            _save_cache(cache, output_path)

    print(f"[QuaSAR-S4] Done. Generated: {generated}, Skipped: {skipped}, "
          f"Total cached: {len(cache)}")
    return cache


def load_quasar_cache(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache: Dict[str, Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"  [QuaSAR-S4] Cache saved: {output_path} ({len(cache)} entries)")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QuaSAR generation for Subtask 4")
    parser.add_argument("--input", default=TEST_DATA_PATH)
    parser.add_argument("--output", default=S4_QUASAR_TEST_CACHE)
    args = parser.parse_args()

    print(f"[QuaSAR-S4] Input:  {args.input}")
    print(f"[QuaSAR-S4] Output: {args.output}")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    print(f"[QuaSAR-S4] Loaded {len(data)} items")

    existing = load_quasar_cache(args.output)
    model, tokenizer = load_llama_model()
    generate_quasar_batch(data, model, tokenizer, args.output, existing)

    del model, tokenizer
    torch.cuda.empty_cache()
    print("[QuaSAR-S4] Done. GPU memory freed.")


if __name__ == "__main__":
    main()
