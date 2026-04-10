"""
data_loader.py
--------------
Dataset and DataLoader for Subtask 4 (multilingual multi-premise syllogisms).

Same as Subtask 2 but with multilingual support:
  - Model input uses syllogism_t (translated text)
  - QuaSAR uses the English syllogism field (Llama works best in English)
  - Train/val split groups by English syllogism to prevent data leakage
    across translations of the same logical structure
"""

import json
import os
import re
import sys
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
S1_DIR     = os.path.join(SCRIPT_DIR, "..", "subtask1")
sys.path.insert(0, S1_DIR)
sys.path.insert(0, SCRIPT_DIR)

from src.subtask4.rest3.config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    MODEL_NAME, MAX_SEQ_LEN, MAX_PREMISES, BATCH_SIZE, EVAL_BATCH_SIZE,
    VALIDATION_SPLIT, SEED, LABEL2ID, HF_CACHE_DIR,
    USE_QUASI_SYMBOLIC, ABSTRACT_SEP,
    S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
    S2_QUASAR_TEST_CACHE, S4_QUASAR_TEST_CACHE,
    QUASAR_MODE,
)

# QuasiSymbolicAbstractor from subtask1
from quasi_symbolic import QuasiSymbolicAbstractor


# ─── Sentence splitting ──────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=\.)\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_premises_and_conclusion(text: str) -> Tuple[List[str], str]:
    sents = split_sentences(text)
    if len(sents) < 2:
        return [], sents[0] if sents else ""
    return sents[:-1], sents[-1]


# ─── Token span detection ─────────────────────────────────────────────────────

def _find_premise_token_spans(
    full_text: str,
    premises: List[str],
    encoding,
) -> List[Tuple[int, int]]:
    """Find (start_tok, end_tok) span for each premise in the tokenized text."""
    offsets = encoding.offset_mapping

    sep = ABSTRACT_SEP.strip()
    sep_pos = full_text.rfind(sep)
    orig_start_char = sep_pos + len(sep) + 1 if sep_pos != -1 else 0

    spans: List[Tuple[int, int]] = []
    search_from = orig_start_char

    for premise in premises:
        char_s = full_text.find(premise, search_from)
        if char_s == -1:
            char_s = full_text.find(premise)
        if char_s == -1:
            spans.append((-1, -1))
            continue
        char_e = char_s + len(premise)
        search_from = char_s + 1

        tok_s, tok_e = None, None
        for ti, (cs, ce) in enumerate(offsets):
            if ce == 0 and cs == 0:
                continue
            if ce <= char_s:
                continue
            if cs >= char_e:
                break
            if tok_s is None:
                tok_s = ti
            tok_e = ti

        if tok_s is None:
            spans.append((-1, -1))
        else:
            spans.append((tok_s, tok_e + 1))

    return spans


# ─── Dataset ─────────────────────────────────────────────────────────────────

class Subtask4Dataset(Dataset):
    """
    Dataset for JointSyllogismClassifier on multilingual data.

    Key difference from Subtask2Dataset:
      - Uses syllogism_t (translated) for model input
      - Uses syllogism (English) for QuaSAR lookup
      - Premise spans are detected in the translated text
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        abstractor: Optional[QuasiSymbolicAbstractor] = None,
        max_premises: int = MAX_PREMISES,
        has_labels: bool = True,
    ):
        self.data         = data
        self.tokenizer    = tokenizer
        self.abstractor   = abstractor
        self.max_premises = max_premises
        self.has_labels   = has_labels

    def __len__(self) -> int:
        return len(self.data)

    def _build_text(self, item: Dict) -> str:
        """
        Build model input:
          <English QuaSAR s2 formalisation> </s> <translated syllogism_t>

        QuaSAR is looked up by the source_id (maps back to subtask2/subtask1 item)
        or by the item's own ID (for test data).
        """
        # Use translated text for model input, fall back to English
        syllogism_t = item.get("syllogism_t", item["syllogism"])

        if self.abstractor is None:
            return syllogism_t

        # Look up QuaSAR by source_id (training) or id (test)
        item_id = item.get("source_id", item.get("id", ""))
        # Strip _aug suffix for subtask2 generated data
        base_id = re.sub(r'_aug\d+$', '', item_id)

        abstract = self.abstractor.abstract(
            item["syllogism"],  # English text for QuaSAR
            item_id=base_id,
            quasar_mode=QUASAR_MODE,
        )
        if abstract and abstract != item["syllogism"]:
            return abstract + ABSTRACT_SEP + syllogism_t
        return syllogism_t

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        # Parse premises from the TRANSLATED text (for span detection)
        syllogism_t = item.get("syllogism_t", item["syllogism"])
        premises, conclusion = parse_premises_and_conclusion(syllogism_t)
        n_prem = min(len(premises), self.max_premises)

        full_text = self._build_text(item)

        # Tokenize with offset mapping for span detection
        enc = self.tokenizer(
            full_text,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
            return_tensors=None,
            return_offsets_mapping=True,
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Premise spans in the translated portion
        raw_spans = _find_premise_token_spans(full_text, premises[:n_prem], enc)
        spans = []
        for i in range(self.max_premises):
            if i < n_prem and i < len(raw_spans):
                spans.append(raw_spans[i])
            else:
                spans.append((-1, -1))

        # Labels
        label        = LABEL2ID.get(item.get("validity", False), 0)
        plausibility = LABEL2ID.get(item.get("plausibility", False), 0)

        relevant_set   = set(item.get("relevant_premises", []))
        premise_labels = []
        for i in range(self.max_premises):
            if i < n_prem:
                premise_labels.append(1.0 if i in relevant_set else 0.0)
            else:
                premise_labels.append(-1.0)

        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "premise_spans":  torch.tensor(spans, dtype=torch.long),
            "num_premises":   torch.tensor(n_prem, dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
            "premise_labels": torch.tensor(premise_labels, dtype=torch.float),
            "plausibility":   torch.tensor(plausibility, dtype=torch.long),
            "id":             item["id"],
        }


# ─── Collate function ────────────────────────────────────────────────────────

def collate_fn_subtask4(batch: List[Dict]) -> Dict:
    """Pad input_ids and attention_mask to batch max length."""
    pad_id = 1  # XLM-RoBERTa padding token id

    max_len = max(b["input_ids"].shape[0] for b in batch)
    B       = len(batch)
    MP      = batch[0]["premise_spans"].shape[0]

    input_ids      = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len),         dtype=torch.long)
    premise_spans  = torch.full((B, MP, 2), -1,        dtype=torch.long)
    num_premises   = torch.zeros(B,                    dtype=torch.long)
    labels         = torch.zeros(B,                    dtype=torch.long)
    premise_labels = torch.full((B, MP), -1.0,         dtype=torch.float)
    plausibility   = torch.zeros(B,                    dtype=torch.long)
    ids: List[str] = []

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L]      = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        premise_spans[i]      = b["premise_spans"]
        num_premises[i]       = b["num_premises"]
        labels[i]             = b["label"]
        premise_labels[i]     = b["premise_labels"]
        plausibility[i]       = b["plausibility"]
        ids.append(b["id"])

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "premise_spans":  premise_spans,
        "num_premises":   num_premises,
        "label":          labels,
        "premise_labels": premise_labels,
        "plausibility":   plausibility,
        "id":             ids,
    }


# ─── Data Loading Utilities ──────────────────────────────────────────────────

def _load_json(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _train_val_split_by_english(
    data: List[Dict],
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split by (validity × plausibility), grouped by English
    syllogism text so that ALL translations of the same logical structure
    go into the same split. Prevents data leakage.
    """
    rng = random.Random(seed)

    # Group items by English syllogism text
    syl_groups: Dict[str, List[Dict]] = {}
    for item in data:
        eng_syl = item["syllogism"]
        syl_groups.setdefault(eng_syl, []).append(item)

    # Stratify at the syllogism level
    buckets: Dict[Tuple, List[str]] = {}
    for eng_syl, items in syl_groups.items():
        key = (items[0]["validity"], items[0].get("plausibility", None))
        buckets.setdefault(key, []).append(eng_syl)

    val_syls, train_syls = set(), set()
    for key, syls in buckets.items():
        rng.shuffle(syls)
        n_val = max(1, int(len(syls) * val_ratio))
        val_syls.update(syls[:n_val])
        train_syls.update(syls[n_val:])

    train_data = [item for item in data if item["syllogism"] in train_syls]
    val_data   = [item for item in data if item["syllogism"] in val_syls]

    rng.shuffle(train_data)
    rng.shuffle(val_data)

    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  (Unique English syllogisms: train={len(train_syls)}, "
          f"val={len(val_syls)}, no overlap)")
    return train_data, val_data


# ─── DataLoader Factory ──────────────────────────────────────────────────────

def build_dataloaders_subtask4(
    train_path: str = TRAIN_DATA_PATH,
    test_path: str = TEST_DATA_PATH,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Returns: train_loader, val_loader, test_loader, tokenizer
    """
    print("[DataLoader-S4] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # QuaSAR cache (merge subtask1 + subtask2 + subtask4 caches)
    quasar_cache = {}
    if USE_QUASI_SYMBOLIC:
        for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
                   S2_QUASAR_TEST_CACHE, S4_QUASAR_TEST_CACHE]:
            if os.path.exists(cp):
                with open(cp) as f:
                    chunk = json.load(f)
                quasar_cache.update(chunk)
                print(f"[DataLoader-S4] QuaSAR cache: {cp} ({len(chunk)} entries)")

    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache) if USE_QUASI_SYMBOLIC else None

    print("[DataLoader-S4] Loading translated training data ...")
    all_data = _load_json(train_path)

    if VALIDATION_SPLIT > 0:
        train_data, val_data = _train_val_split_by_english(all_data, VALIDATION_SPLIT)
    else:
        train_data = all_data
        val_data = None
        print(f"  Train: {len(train_data)} (all translated, val=test)")

    print("[DataLoader-S4] Loading subtask4 test data ...")
    test_data = _load_json(test_path)
    print(f"  Test:  {len(test_data)}")

    if val_data is None:
        val_data = test_data
        print(f"  Val: {len(val_data)} (= test data, real distribution)")

    train_ds = Subtask4Dataset(train_data, tokenizer, abstractor, has_labels=True)
    val_ds   = Subtask4Dataset(val_data,   tokenizer, abstractor, has_labels=True)
    test_ds  = Subtask4Dataset(test_data,  tokenizer, abstractor, has_labels=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask4,
    )
    test_loader = DataLoader(
        test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask4,
    )

    return train_loader, val_loader, test_loader, tokenizer
