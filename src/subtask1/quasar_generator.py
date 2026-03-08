import sys; sys.stdout.reconfigure(line_buffering=True)
"""
quasar_generator.py
-------------------
Generates Quasi-Symbolic Abstraction and Reasoning (QuaSAR) outputs using
Llama 3.1 8B for syllogistic reasoning problems.

Based on: Ranaldi, Valentino & Freitas (2025)
"Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions"
ACL 2025 / arXiv:2502.12616

QuaSAR structures reasoning as (Q, S, R, A) where S = (s1, s2, s3, s4):
  s1: Abstraction  — identify predicates, variables, constants
  s2: Formalisation — translate into quasi-symbolic representation
  s3: Explanation   — step-by-step reasoning using symbolic structure
  s4: Answering     — final answer

The generated QuaSAR output is saved to a JSON cache file. During training,
the formalisation (s2) is extracted and concatenated with the original
syllogism text as input to XLM-RoBERTa:

  Input to XLM-R = "<s2_formalisation> </s> <original_syllogism>"

This gives the encoder both the logical structure (content-free symbols)
and the original content, encouraging it to attend to logical form over
plausibility — directly implementing the proposal's technique 3.

Usage:
    # Generate QuaSAR for all training data
    python quasar_generator.py --input train_data.json --output quasar_train.json

    # Or called from main.py --mode generate
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import List, Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    LLAMA_MODEL_NAME, HF_CACHE_DIR, USE_4BIT,
    TRAIN_DATA_PATH, TEST_DATA_PATH, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# ─── QuaSAR Prompt Template (from Appendix A of Ranaldi et al., 2025) ─────────
# Adapted for syllogistic reasoning: the {question} is the syllogism text,
# and we ask the model to determine formal validity.

QUASAR_PROMPT_TEMPLATE = """Analyze this syllogism for formal validity. Be concise and brief in each step.

Syllogism: {syllogism}

Complete ALL 4 steps below. Keep each step SHORT (2-4 lines max).

s1 (Abstraction): Identify predicates and replace terms with X, Y, Z.

s2 (Formalisation): Write the formal symbolic structure using X, Y, Z (e.g., All X are Y).

s3 (Explanation): Briefly check if the conclusion follows logically from the premises.

s4 (Answering): The answer is: [valid/invalid]"""


# ─── Extraction Helpers ──────────────────────────────────────────────────────

def extract_formalisation(quasar_output: str) -> str:
    """
    Extract the Formalisation (s2) section from QuaSAR output.
    This is the quasi-symbolic representation that gets fed to XLM-RoBERTa.

    Tries multiple strategies to find s2 content, from strict to lenient.
    """
    import re

    # Strategy 1: Find text between s2-header and s3-header (various formats)
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

    # Strategy 2: Line-by-line scan for s2 section
    lines = quasar_output.split("\n")
    s2_started = False
    s2_lines = []
    for line in lines:
        if re.search(r"(?:Formalisation|Formalization|s2)", line, re.IGNORECASE) and not s2_started:
            s2_started = True
            # Include content on the same line after the header
            after_header = re.sub(r".*(?:Formalisation|Formalization|s2)\s*[:\-\)]*\s*", "", line, flags=re.IGNORECASE).strip()
            if after_header:
                s2_lines.append(after_header)
            continue
        if s2_started:
            if re.search(r"(?:Explanation|Explaination|s3|Step\s*3|Answering|s4)", line, re.IGNORECASE):
                break
            s2_lines.append(line)

    if s2_lines:
        text = "\n".join(s2_lines).strip()
        if len(text) > 10:
            return text

    # Strategy 3: Find lines with symbolic patterns (X, Y, Z, All, Some, No, ∀, →)
    symbolic_lines = []
    for line in lines:
        if re.search(r"(?:All|Some|No)\s+[A-Z]\b|[A-Z]\s*(?:→|->|⊂|⊆|∀|∃)|∀|∃|Premise|Conclusion", line):
            symbolic_lines.append(line.strip())
    if symbolic_lines:
        return "\n".join(symbolic_lines[:10]).strip()

    # Last fallback: return first 300 chars
    logger.warning("Could not extract s2 from QuaSAR output, using truncated output.")
    return quasar_output[:300].strip()


def extract_answer(quasar_output: str) -> Optional[bool]:
    """
    Extract the validity answer from QuaSAR's s4 section.
    Returns True (valid), False (invalid), or None if unparseable.
    """
    import re

    # Look for "The answer is: valid/invalid"
    answer_pattern = re.compile(
        r"[Tt]he\s+answer\s+is\s*:?\s*(valid|invalid)",
        re.IGNORECASE
    )
    match = answer_pattern.search(quasar_output)
    if match:
        return match.group(1).lower() == "valid"

    # Fallback: look for "valid" or "invalid" near end
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
    """
    Load Llama 3.1 8B with optional 4-bit quantization for memory efficiency.

    With 4-bit quantization (~5GB VRAM), fits on a single 11GB GPU.
    Without quantization (~16GB FP16), needs 2x 11GB GPUs or 1x 24GB GPU.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"[QuaSAR] Loading {model_name}...")
    print(f"  4-bit quantization: {use_4bit}")
    print(f"  Cache dir: {cache_dir}")

    token = os.environ.get("HF_TOKEN", None)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            token=token,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            token=token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[QuaSAR] Model loaded: {n_params / 1e9:.1f}B params")

    return model, tokenizer


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_quasar_single(
    model,
    tokenizer,
    syllogism: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Generate QuaSAR output for a single syllogism.

    Uses low temperature (0.1) for deterministic, structured output.
    For Instruct models, wraps the prompt in the chat template.
    """
    prompt = QUASAR_PROMPT_TEMPLATE.format(syllogism=syllogism)

    # Use chat template if available (Llama-Instruct models)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": "You are an expert logician. Follow the requested output format exactly."},
                {"role": "user", "content": prompt},
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        except Exception:
            # Fallback to raw prompt if chat template fails
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result.strip()


def generate_quasar_batch(
    data: List[Dict],
    model,
    tokenizer,
    output_path: str,
    existing_cache: Optional[Dict[str, Dict]] = None,
    max_new_tokens: int = 512,
    save_every: int = 25,
) -> Dict[str, Dict]:
    """
    Generate QuaSAR outputs for a list of syllogisms.

    Saves incrementally every `save_every` examples to avoid losing progress.

    Returns:
      {id: {"syllogism": str, "quasar_full": str, "quasar_s2": str,
            "quasar_answer": bool|None}}
    """
    cache = existing_cache or {}
    total = len(data)
    skipped = 0
    generated = 0

    print(f"[QuaSAR] Generating for {total} examples...")
    print(f"  Already cached: {len(cache)}")

    start_time = time.time()

    for i, item in enumerate(data):
        item_id = item["id"]

        # Skip if already generated
        if item_id in cache:
            skipped += 1
            continue

        syllogism = item["syllogism"]
        quasar_output = generate_quasar_single(
            model, tokenizer, syllogism, max_new_tokens=max_new_tokens
        )

        # Extract the formalisation (s2) for use as input to XLM-RoBERTa
        s2_text = extract_formalisation(quasar_output)
        answer = extract_answer(quasar_output)

        cache[item_id] = {
            "syllogism": syllogism,
            "quasar_full": quasar_output,
            "quasar_s2": s2_text,
            "quasar_answer": answer,
        }
        generated += 1

        # Debug: print first 3 outputs to verify format
        if generated <= 3:
            print(f"\n{'='*60}")
            print(f"  DEBUG (item {generated}): id={item_id}")
            print(f"  Syllogism: {syllogism[:120]}...")
            print(f"  --- Raw Llama output (first 500 chars) ---")
            print(f"  {quasar_output[:500]}")
            print(f"  --- Extracted s2 ---")
            print(f"  {s2_text[:200]}")
            print(f"  --- Extracted answer: {answer} ---")
            print(f"{'='*60}\n")

        # Progress update
        elapsed = time.time() - start_time
        rate = generated / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] Generated: {generated}, "
                  f"Skipped (cached): {skipped}, "
                  f"Rate: {rate:.1f} ex/s, ETA: {eta/60:.1f} min")

        # Incremental save
        if (generated > 0 and generated % save_every == 0) or (i + 1) == total:
            _save_cache(cache, output_path)

    print(f"[QuaSAR] Done. Generated: {generated}, Skipped: {skipped}, "
          f"Total cached: {len(cache)}")

    return cache


def _save_cache(cache: Dict[str, Dict], output_path: str):
    """Save QuaSAR cache to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"  [QuaSAR] Cache saved: {output_path} ({len(cache)} entries)")


def load_quasar_cache(path: str) -> Dict[str, Dict]:
    """Load QuaSAR cache from JSON. Returns empty dict if file doesn't exist."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate QuaSAR abstractions using Llama for syllogisms"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input JSON (train or test data)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output QuaSAR cache JSON"
    )
    parser.add_argument(
        "--model", type=str, default=LLAMA_MODEL_NAME,
        help=f"HuggingFace model name (default: {LLAMA_MODEL_NAME})"
    )
    parser.add_argument(
        "--no_4bit", action="store_true",
        help="Disable 4-bit quantization (uses FP16 instead)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Max new tokens to generate per example"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing cache file (skip already-generated examples)"
    )
    args = parser.parse_args()

    # Load input data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[QuaSAR] Loaded {len(data)} examples from {args.input}")

    # Load existing cache if resuming
    existing = load_quasar_cache(args.output) if args.resume else {}

    # Load Llama
    model, tokenizer = load_llama_model(
        model_name=args.model,
        use_4bit=(not args.no_4bit),
    )

    # Generate
    cache = generate_quasar_batch(
        data=data,
        model=model,
        tokenizer=tokenizer,
        output_path=args.output,
        existing_cache=existing,
        max_new_tokens=args.max_tokens,
    )

    print(f"\n[QuaSAR] All done. Cache at: {args.output}")
    print(f"  Total entries: {len(cache)}")

    # Show a sample
    sample_id = list(cache.keys())[0]
    sample = cache[sample_id]
    print(f"\n{'='*70}")
    print(f"SAMPLE (id={sample_id}):")
    print(f"  Syllogism: {sample['syllogism'][:100]}...")
    print(f"  s2 (Formalisation): {sample['quasar_s2'][:200]}...")
    print(f"  Answer: {sample['quasar_answer']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
