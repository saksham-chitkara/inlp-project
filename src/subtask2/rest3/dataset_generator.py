#!/usr/bin/env python3
"""
dataset_generator.py
---------------------
Generates training data for Subtask 2 by injecting distractor premises into
the existing Subtask 1 training data.

Input:  960 items (480 valid, 480 invalid) -- each has exactly 2 premises + 1 conclusion
Output: SAMPLES_PER_ITEM x 960 items -- each has 6-7 premises (2 original + 4-5 distractors)
        shuffled in random order, with relevant_premises tracking the original indices.

Two techniques:
  --technique llm (default, recommended):
        Llama 3.1-8B-Instruct (fp16 across 3 GPUs) generates semantically
        plausible but logically irrelevant distractors that:
          - Use the SAME entities/domain as the source syllogism
          - Match the plausibility style (absurd for implausible, sensible for plausible)
          - Use varied formal syllogistic quantifier phrasing
          - Are confusingly similar to the real premises
        Produces harder negatives (better for model generalization).
        Requires GPU (about 30-40 min for 960 items x 2 samples on 3x RTX 2080 Ti).

  --technique cross (fast fallback, no GPU):
        Cross-contamination: sample premises from other syllogisms in the same
        domain. Lower quality but instant. Also used as automatic fallback
        when LLM fails to produce enough distractors.

Class balance:
  - Same 50/50 valid/invalid as subtask1 source
  - For VALID items:  relevant_premises = [shuffled indices of the 2 original premises]
  - For INVALID items: relevant_premises = []

Usage:
  cd /ssd_scratch/shubhamcvit/inlp/project
  python3 src/subtask2/dataset_generator.py --technique llm --samples 2
  python3 src/subtask2/dataset_generator.py --technique cross --samples 2
"""

import json
import os
import re
import sys
import math
import random
import argparse
import hashlib
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from src.subtask2.rest3.config import (
    S1_TRAIN_DATA_PATH, GENERATED_TRAIN_PATH,
    MIN_DISTRACTORS, MAX_DISTRACTORS, SAMPLES_PER_ITEM,
    LLAMA_MODEL_NAME, USE_4BIT, LLM_MAX_NEW_TOKENS, HF_CACHE_DIR, SEED,
)


# ===================================================================
# Sentence splitting
# ===================================================================

def split_sentences(text: str) -> List[str]:
    """Split text into sentences on '. ' boundaries, stripping whitespace."""
    parts = re.split(r'(?<=\.)\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_item(item: Dict) -> Tuple[List[str], str]:
    """Return (premises, conclusion) for a syllogism item."""
    sents = split_sentences(item["syllogism"])
    assert len(sents) >= 2, f"Expected >=2 sentences, got {len(sents)}: {item['syllogism'][:80]}"
    return sents[:-1], sents[-1]


# ===================================================================
# Domain bucketing for cross-contamination
# ===================================================================

_DOMAIN_KEYWORDS = {
    "animals": ["cat", "dog", "lion", "tiger", "bird", "fish", "animal",
                "mammal", "feline", "canine", "reptile", "insect", "horse",
                "rabbit", "snake", "wolf", "bear", "eagle", "duck", "cow",
                "pig", "sheep", "mouse", "rat", "monkey", "whale", "shark",
                "hound", "retriever", "poodle", "collie", "puppy", "kitten",
                "jaguar", "cougar", "lynx", "predator", "creature"],
    "plants":  ["plant", "flower", "tree", "rose", "oak", "pine", "grass",
                "vegetable", "fruit", "carrot", "pea", "bean", "radish",
                "tulip", "orchid", "fern", "moss", "algae", "cactus",
                "beet", "turnip", "potato", "berry", "lemon", "lime",
                "orange", "citrus"],
    "vehicles":["car", "truck", "bus", "bike", "bicycle", "vehicle", "scooter",
                "motorcycle", "train", "ship", "boat", "plane", "aircraft",
                "helicopter", "submarine", "tractor", "tricycle", "wheel",
                "steering", "transport"],
    "places":  ["city", "country", "town", "village", "river", "mountain",
                "ocean", "lake", "forest", "desert", "island", "continent",
                "capital", "region", "valley", "hill", "bay", "gulf",
                "stream", "geographic"],
    "people":  ["person", "man", "woman", "student", "teacher", "doctor",
                "lawyer", "worker", "farmer", "artist", "scientist", "king",
                "queen", "soldier", "engineer", "athlete", "bachelor",
                "husband", "human", "nurse", "surgeon", "dentist",
                "pharmacist", "patient"],
    "objects": ["table", "chair", "book", "pen", "pencil", "lamp", "box",
                "bottle", "stone", "metal", "plastic", "glass", "paper",
                "wood", "tool", "machine", "device", "computer", "phone",
                "tablet", "furniture", "television"],
    "food":    ["food", "bread", "milk", "cheese", "meat", "sugar", "salt",
                "coffee", "tea", "juice", "soda", "wine", "beer", "cake",
                "pizza", "rice", "soup", "beverage"],
    "music":   ["piano", "drum", "violin", "guitar", "flute", "trumpet",
                "musical", "instrument", "song", "melody", "orchestra"],
    "buildings":["house", "building", "apartment", "skyscraper", "cottage",
                 "shed", "palace", "structure", "home"],
    "astronomy":["planet", "star", "sun", "moon", "comet", "galaxy",
                 "asteroid", "celestial", "orbit", "meteor"],
    "concepts":["thing", "object", "entity", "creature", "being", "item",
                "element", "category", "class", "group", "set", "type"],
}

def _detect_domain(text: str) -> str:
    """Return the most prominent domain keyword found in text, else 'concepts'."""
    low = text.lower()
    best_domain, best_count = "concepts", 0
    for domain, kws in _DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in kws if kw in low)
        if count > best_count:
            best_count, best_domain = count, domain
    return best_domain


def _build_domain_index(data: List[Dict]) -> Dict[str, List[Tuple[str, str]]]:
    """Returns {domain -> list of (premise, source_id)} pairs."""
    index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for item in data:
        premises, _ = parse_item(item)
        domain = _detect_domain(item["syllogism"])
        for p in premises:
            index[domain].append((p, item["id"]))
    return index


# ===================================================================
# Technique 2: Cross-contamination (fast fallback)
# ===================================================================

def generate_distractors_cross(
    item: Dict,
    n_distractors: int,
    domain_index: Dict[str, List[Tuple[str, str]]],
    original_premises: List[str],
    rng: random.Random,
) -> List[str]:
    """Sample n_distractors premises from other items in the same domain."""
    domain   = _detect_domain(item["syllogism"])
    orig_set = set(original_premises)
    candidates: List[str] = []

    for p, src_id in domain_index.get(domain, []):
        if src_id != item["id"] and p not in orig_set:
            candidates.append(p)

    if len(candidates) < n_distractors:
        for d, pairs in domain_index.items():
            if d == domain:
                continue
            for p, src_id in pairs:
                if src_id != item["id"] and p not in orig_set:
                    candidates.append(p)

    rng.shuffle(candidates)
    return candidates[:n_distractors]


# ===================================================================
# Technique 1: LLM (Llama 3.1-8B-Instruct, fp16 / 4-bit)
# ===================================================================

# --- Entity extraction ---

_STOP_WORDS = frozenset({
    "all", "any", "are", "can", "the", "a", "an", "is", "it",
    "that", "this", "they", "not", "no", "some", "have", "has",
    "been", "was", "were", "and", "or", "but", "so", "also",
    "its", "their", "be", "do", "does", "did", "will", "which",
    "who", "with", "for", "from", "of", "to", "in", "on", "at",
    "as", "by", "if", "then", "thus", "every", "single", "fact",
    "known", "things", "classified", "there", "therefore", "implies",
    "like", "such", "each", "even", "more", "most", "other",
    "into", "out", "up", "about", "these", "those", "than",
    "what", "when", "how", "ones", "one", "true", "false",
    "everything", "anything", "nothing", "something", "someone",
    "anyone", "everyone", "nobody", "somebody", "thing", "entity",
    "object", "item", "kind", "type", "sort", "way", "part",
    "means", "follows", "case", "implies", "concluded", "says",
    "must", "only", "logical", "conclusion", "consequently",
    "without", "exception", "entire", "contained", "within",
    "defined", "described", "overlap", "common", "members",
    "group", "category", "set", "number", "select", "few",
    "portion", "among",
})


def _extract_entities(premises: List[str], conclusion: str) -> List[str]:
    """
    Extract key content nouns (2+ chars) from the original premises + conclusion.
    Returns up to 10 unique nouns, preserving order of first appearance.
    """
    combined = " ".join(premises + [conclusion]).lower()
    for phrase in (
        "every single ", "it is a known fact that ", "any and all ",
        "everything that is ", "there are no ", "some of the ",
        "it is true that ", "it is a fact that ", "it is undeniable that ",
        "it is also true that ", "the only logical conclusion is ",
        "from this, it follows that ", "therefore, it is concluded that ",
        "it must be the case that ", "it must be true that ",
        "consequently, ", "therefore, ", "this means that ",
        "it follows that ", "this implies that ",
    ):
        combined = combined.replace(phrase, " ")
    tokens = re.findall(r"\b[a-z][a-z]{1,}\b", combined)
    seen: Dict[str, int] = {}
    for tok in tokens:
        if tok not in _STOP_WORDS and tok not in seen:
            seen[tok] = len(seen)
    freq: Dict[str, int] = {}
    for tok in tokens:
        if tok in seen:
            freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(seen.keys(), key=lambda t: (-freq.get(t, 0), seen[t]))
    return ranked[:10]


# --- Comprehensive quantifier templates from SemEval test set analysis ---
# These are the EXACT phrasings observed across 190 test items.

_QUANTIFIER_TEMPLATES = (
    # Universal affirmative
    "Every single X is a Y.",
    "All X are Y.",
    "Any X is a Y.",
    "Anything that is a X is also a Y.",
    "Everything that is a X is also a Y.",
    "It is a known fact that all X are Y.",
    "It is a fact that all X are Y.",
    "It is true that every X is a Y.",
    "It is undeniable that all X are Y.",
    "All X are defined as Y.",
    "All of those who are X are Y.",
    "Any and all X are also Y.",
    "The entire set of X is contained within the set of Y.",
    "X are, without exception, Y.",
    # Universal negative
    "No X are Y.",
    "There are no X that are Y.",
    "There is not a single X which is a Y.",
    "A X is never a Y.",
    "X and Y have no members in common.",
    "There are no X that are not Y.",
    "There exist no X who also are Y.",
    "There is no overlap between X and Y.",
    # Particular affirmative
    "Some X are Y.",
    "Some of the X are Y.",
    "A portion of the things that are X are Y.",
    "Among the items that are X, some of them are Y.",
    "There are many X that are Y.",
    "There is at least a X that is a Y.",
    "A number of X are Y.",
    "A select few X are classified as Y.",
    "Among the group of things that are X, some are Y.",
    # Particular negative
    "Some X are not Y.",
    "There are some things which are X that are not Y.",
    "A portion of X are not Y.",
    "It is the case that some X are not Y.",
    # Relational / verbal
    "Any X that verbs is a Y.",
    "It is a fact that all X verb.",
    "Some X verb in Z.",
    "Every X is a creature with Y.",
)


# ---- THE DETAILED DISTRACTOR GENERATION PROMPT ----
# Crafted based on deep analysis of all 190 test items from subtask2.
# Encodes EVERY pattern the LLM needs to follow.

_DISTRACTOR_PROMPT = """\
You are a formal logic expert helping create a challenging syllogistic reasoning dataset.

=== TASK ===
Given a 2-premise syllogism and its conclusion, generate exactly {n} DISTRACTOR \
premises. These distractors will be shuffled with the original 2 real premises. \
A machine learning model must then identify which premises are relevant to the \
conclusion. Your job is to make distractors that are HARD to distinguish from \
the real premises.

=== ORIGINAL SYLLOGISM ===
Premise A: {p0}
Premise B: {p1}
Conclusion: {conclusion}

=== ENTITIES AND DOMAIN ===
Key entities/concepts extracted from this syllogism: {entity_list}
Detected semantic domain: {domain}

=== PLAUSIBILITY STYLE ===
This syllogism is {plausibility_desc}.
{plausibility_instruction}

=== DETAILED REQUIREMENTS ===

REQUIREMENT 1 - SAME SEMANTIC DOMAIN:
Every distractor MUST use nouns and entities from the SAME semantic domain as \
the original. The distractors should feel like they belong in the same paragraph \
as the original premises.
- If about cats/dogs/felines -> use: lions, tigers, rabbits, horses, bears, wolves, \
eagles, fish, reptiles, mammals, canines, hounds, retrievers, etc.
- If about cars/bicycles/vehicles -> use: trucks, buses, scooters, motorcycles, \
trains, ships, planes, wheels, steering, transport, etc.
- If about rivers/mountains -> use: oceans, lakes, valleys, islands, streams, \
bays, continents, geographic features, etc.
- If about planets/stars -> use: moons, comets, galaxies, asteroids, celestial, \
orbits, meteors, suns, etc.
- If about houses/buildings -> use: apartments, skyscrapers, cottages, sheds, \
palaces, structures, etc.
- If about pens/pencils -> use: erasers, markers, notebooks, paper, writing \
instruments, ink, etc.
- If about doctors/nurses -> use: surgeons, dentists, pharmacists, patients, \
hospitals, medical, etc.
NEVER introduce entities from a completely unrelated domain.

REQUIREMENT 2 - SINGLE CATEGORICAL CLAIM PER SENTENCE:
Each distractor must be EXACTLY one sentence that states a single categorical \
relationship. Valid forms:
- Class membership: "X is a Y"
- Universal: "All X are Y" / "No X are Y" / "Every X is a Y"
- Particular: "Some X are Y" / "Some X are not Y"
- Property: "X has property Y" / "X can do Y"
FORBIDDEN: compound sentences with "and"/"but"/"because"/"however", \
multi-clause sentences, questions, definitions, or explanations.

REQUIREMENT 3 - VARIED QUANTIFIER PHRASING:
Use formal syllogistic language with VARIED phrasing across all {n} distractors. \
Each distractor should use a DIFFERENT quantifier pattern. Patterns to choose from:
{quantifier_examples}
You MUST use a mix of universal affirmative, universal negative, particular \
affirmative, and particular negative forms. Do NOT use the same pattern twice.

REQUIREMENT 4 - LOGICALLY IRRELEVANT TO CONCLUSION:
The distractors MUST NOT logically contribute to proving or disproving the \
conclusion: "{conclusion}"
They should make claims about the same domain entities but about DIFFERENT \
relationships that have no bearing on the conclusion's logical chain. The \
conclusion follows ONLY from Premise A and Premise B; your distractors are \
noise.

REQUIREMENT 5 - NOT PARAPHRASES OF ORIGINALS:
Each distractor must be a genuinely NEW claim. Do NOT rephrase, restate, \
or closely paraphrase Premise A or Premise B. Do NOT use the exact same \
subject-predicate combination as either original premise with just a different \
quantifier.

REQUIREMENT 6 - CONFUSINGLY SIMILAR SURFACE FORM:
The distractors should LOOK like they could be relevant premises. They should \
share the same vocabulary, same formal academic tone, and same domain. A \
surface-level reader should struggle to tell them apart from the real premises. \
This is the key challenge: same domain + same style + different logic.

=== OUTPUT FORMAT ===
Output EXACTLY {n} sentences, one per line. Rules:
- NO numbering (no "1.", "2.", "-", "*")
- NO blank lines between sentences
- NO explanations, labels, headers, or commentary
- Each sentence MUST end with a period
- Output ONLY the {n} distractor sentences and nothing else"""


def _format_quantifier_examples() -> str:
    """Return a formatted block of quantifier pattern examples."""
    return "\n".join(f"   {q}" for q in _QUANTIFIER_TEMPLATES)


def _describe_plausibility(is_plausible: bool) -> tuple:
    """Return (short_desc, detailed_instruction) for plausibility matching."""
    if is_plausible:
        return (
            "PLAUSIBLE (real-world sensible)",
            "Your distractors MUST also be plausible real-world claims. "
            "Examples of plausible distractors:\n"
            "  - 'Every single car has a steering wheel.'\n"
            "  - 'It is a known fact that all trucks transport goods.'\n"
            "  - 'Some parrots can mimic speech.'\n"
            "  - 'Any rabbit has long ears.'\n"
            "  - 'It is a known fact that all goldfish live in water.'\n"
            "These are factually reasonable statements that could be true."
        )
    else:
        return (
            "IMPLAUSIBLE (absurd / counterfactual)",
            "Your distractors MUST also be deliberately ABSURD or "
            "counterfactual, matching the nonsensical style. "
            "Examples of implausible distractors:\n"
            "  - 'It is true that all horses are insects.'\n"
            "  - 'Every single phone is a banana.'\n"
            "  - 'Some pianos walk around on two legs.'\n"
            "  - 'It is a fact that all dentists are ghosts.'\n"
            "  - 'There are no persians that are not golden retrievers.'\n"
            "  - 'Any animal that roars is a fish.'\n"
            "These are clearly false, absurd claims."
        )


# --- LLM loading and generation ---

def load_llama():
    """Load Llama 3.1-8B-Instruct. Uses fp16 across all GPUs by default, or 4-bit if configured."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if USE_4BIT:
        print(f"[LLM] Loading {LLAMA_MODEL_NAME} (4-bit quantized) ...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        print(f"[LLM] Loading {LLAMA_MODEL_NAME} (fp16 across all GPUs) ...")
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME, cache_dir=HF_CACHE_DIR, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()
    n_gpus = torch.cuda.device_count()
    print(f"[LLM] Model ready, distributed across {n_gpus} GPU(s)")
    return model, tokenizer


def _llm_generate(model, tokenizer, prompt: str) -> str:
    """Generate text from the LLM given a prompt."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return resp.strip()


def generate_distractors_llm(
    model, tokenizer,
    item: Dict,
    premises: List[str],
    conclusion: str,
    n_distractors: int,
) -> List[str]:
    """
    Use Llama to generate entity-aware, plausibility-matched distractors.
    Same entities, same style, logically irrelevant.
    """
    entities = _extract_entities(premises, conclusion)
    entity_list = ", ".join(entities) if entities else "the same concepts"
    domain = _detect_domain(item["syllogism"])
    plaus_desc, plaus_instr = _describe_plausibility(item["plausibility"])

    assert len(premises) >= 2, "Need at least 2 original premises"
    prompt = _DISTRACTOR_PROMPT.format(
        p0=premises[0],
        p1=premises[1],
        conclusion=conclusion,
        entity_list=entity_list,
        domain=domain,
        n=n_distractors,
        plausibility_desc=plaus_desc,
        plausibility_instruction=plaus_instr,
        quantifier_examples=_format_quantifier_examples(),
    )
    raw = _llm_generate(model, tokenizer, prompt)

    # Parse: one sentence per line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    # Remove numbered prefixes (e.g. "1.", "- ", "* ")
    lines = [re.sub(r'^[\d\-\*\.]+[\.\)\s]+', '', l).strip() for l in lines]
    # Keep only non-empty non-duplicate lines that look like real sentences
    orig_set = set(premises + [conclusion])
    result = []
    seen = set()
    for line in lines:
        if not line:
            continue
        # Ensure ends with period
        if not line.endswith('.'):
            line = line.rstrip('.,:;') + '.'
        # Skip if too short, duplicate, or same as original
        if len(line) < 15:
            continue
        if line in orig_set or line in seen:
            continue
        # Skip lines that look like labels/headers
        if line.startswith(('REQUIREMENT', 'Note:', 'Example:', 'Output:')):
            continue
        result.append(line)
        seen.add(line)
    return result[:n_distractors]


# ===================================================================
# Core: build one augmented sample from a source item
# ===================================================================

def build_sample(
    item: Dict,
    distractors: List[str],
    sample_idx: int,
    rng: random.Random,
) -> Dict:
    """
    Given a source subtask1 item and a list of distractor premises,
    build one subtask2-format training sample by shuffling all premises.

    For VALID items:   relevant_premises = [shuffled positions of original 2 premises]
    For INVALID items: relevant_premises = []
    """
    orig_premises, conclusion = parse_item(item)
    all_premises = list(orig_premises) + list(distractors)
    rng.shuffle(all_premises)

    orig_set = set(orig_premises)
    if item["validity"]:
        relevant_idxs = sorted(
            i for i, p in enumerate(all_premises) if p in orig_set
        )
    else:
        relevant_idxs = []

    syllogism_text = " ".join(all_premises) + " " + conclusion

    src_id = item["id"]
    new_id = f"{src_id}_aug{sample_idx}"

    return {
        "id":               new_id,
        "source_id":        src_id,
        "syllogism":        syllogism_text,
        "validity":         item["validity"],
        "plausibility":     item["plausibility"],
        "relevant_premises": relevant_idxs,
        "num_premises":     len(all_premises),
    }


# ===================================================================
# Main generation loop
# ===================================================================

def generate_dataset(
    technique: str = "llm",
    samples_per_item: int = SAMPLES_PER_ITEM,
    min_dist: int = MIN_DISTRACTORS,
    max_dist: int = MAX_DISTRACTORS,
    output_path: str = GENERATED_TRAIN_PATH,
    seed: int = SEED,
):
    rng = random.Random(seed)
    random.seed(seed)

    print(f"Loading subtask1 training data from: {S1_TRAIN_DATA_PATH}")
    with open(S1_TRAIN_DATA_PATH) as f:
        data = json.load(f)
    print(f"  {len(data)} items (valid={sum(d['validity'] for d in data)}, "
          f"invalid={sum(not d['validity'] for d in data)})")

    domain_index: Optional[Dict] = None
    llama_model  = None
    llama_tok    = None

    if technique == "cross":
        print("Building domain index for cross-contamination ...")
        domain_index = _build_domain_index(data)
        for dom, pairs in domain_index.items():
            print(f"  {dom}: {len(pairs)} premises")
    elif technique == "llm":
        llama_model, llama_tok = load_llama()
    else:
        raise ValueError(f"Unknown technique '{technique}'. Use 'cross' or 'llm'.")

    print(f"\nGenerating {samples_per_item} samples per item "
          f"({samples_per_item * len(data)} total) ...")
    print(f"Distractors per item: {min_dist}-{max_dist}")
    sys.stdout.flush()

    generated: List[Dict] = []
    llm_failures = 0

    for idx, item in enumerate(data):
        premises, conclusion = parse_item(item)

        for s in range(samples_per_item):
            n_dist = rng.randint(min_dist, max_dist)

            if technique == "cross":
                distractors = generate_distractors_cross(
                    item, n_dist, domain_index, premises, rng
                )
            else:
                distractors = generate_distractors_llm(
                    llama_model, llama_tok, item, premises, conclusion, n_dist
                )

            # Fallback to cross-contamination if LLM didn't produce enough
            if len(distractors) < min_dist:
                llm_failures += 1
                if domain_index is None:
                    domain_index = _build_domain_index(data)
                extra = generate_distractors_cross(
                    item, n_dist - len(distractors), domain_index, premises, rng
                )
                distractors.extend(extra)

            if not distractors:
                continue

            sample = build_sample(item, distractors, s, rng)
            generated.append(sample)

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(data)}] generated {len(generated)} samples "
                  f"(LLM fallbacks so far: {llm_failures})")
            sys.stdout.flush()

    # Summary stats
    n_valid   = sum(1 for g in generated if g["validity"])
    n_invalid = sum(1 for g in generated if not g["validity"])
    print(f"\nGenerated {len(generated)} samples: valid={n_valid}, invalid={n_invalid}")
    if llm_failures:
        print(f"LLM fallback to cross-contamination: {llm_failures} times")

    prem_counts = Counter(g["num_premises"] for g in generated)
    print(f"Premise counts: {dict(sorted(prem_counts.items()))}")

    errors = 0
    for g in generated:
        np_count = g["num_premises"]
        for ri in g["relevant_premises"]:
            if ri < 0 or ri >= np_count:
                errors += 1
    if errors:
        print(f"WARNING: {errors} out-of-bounds relevant_premise indices!")
    else:
        print("Validation: all relevant_premise indices are within bounds OK")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(generated, f, indent=2)
    print(f"\nDataset saved to: {output_path}")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Subtask 2 training data with distractor premises"
    )
    parser.add_argument("--technique", default="llm", choices=["cross", "llm"],
                        help="'llm' = Llama-3.1 entity-aware distractors (default); "
                             "'cross' = cross-contamination (fast, no GPU)")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_ITEM,
                        help=f"Augmented samples per source item (default: {SAMPLES_PER_ITEM})")
    parser.add_argument("--min-dist", type=int, default=MIN_DISTRACTORS)
    parser.add_argument("--max-dist", type=int, default=MAX_DISTRACTORS)
    parser.add_argument("--output", default=GENERATED_TRAIN_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    generate_dataset(
        technique=args.technique,
        samples_per_item=args.samples,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        output_path=args.output,
        seed=args.seed,
    )
