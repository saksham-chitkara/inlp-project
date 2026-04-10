"""
data_loader.py
--------------
Dataset and DataLoader for Subtask 2 (multi-premise syllogisms).

Key difference from Subtask 1:
  Each item has 5-8 premises (not exactly 2). We need:
    1. Standard tokenized input (same as subtask1, with optional QuaSAR prefix)
    2. Premise token spans: for the premise_head to pool hidden states over
       each premise's tokens in the sequence.
    3. Per-premise binary labels for training.

Premise span detection:
  We tokenize with offset_mapping.  For each premise sentence, we locate its
  character position in the ORIGINAL syllogism text (not the QuaSAR prefix)
  and map to token indices.  The QuaSAR part is prepended as-is, so we shift
  character offsets accordingly.

  Spans are returned as (start_token_idx, end_token_idx) — where end is
  exclusive (Python slice convention).  Padded to MAX_PREMISES with (-1, -1).

Batch collation:
  - input_ids / attention_mask: standard padding to batch max length
  - premise_spans: (B, MAX_PREMISES, 2) — padded with -1
  - num_premises:  (B,) int
  - premise_labels (B, MAX_PREMISES) float — padded with -1 for masked loss
  - label, plausibility, id: same as subtask1
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

from src.subtask2.rest3.config import (
    S1_TRAIN_DATA_PATH, TEST_DATA_PATH, GENERATED_TRAIN_PATH,
    MODEL_NAME, MAX_SEQ_LEN, MAX_PREMISES, BATCH_SIZE, EVAL_BATCH_SIZE,
    VALIDATION_SPLIT, SEED, LABEL2ID, HF_CACHE_DIR,
    USE_QUASI_SYMBOLIC, ABSTRACT_SEP, S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE,
    S2_QUASAR_TEST_CACHE, QUASAR_MODE,
)

# QuasiSymbolicAbstractor from subtask1
from quasi_symbolic import QuasiSymbolicAbstractor


# ─── Sentence splitting (same as dataset_generator) ──────────────────────────

def split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=\.)\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def parse_premises_and_conclusion(item: Dict) -> Tuple[List[str], str]:
    sents = split_sentences(item["syllogism"])
    if len(sents) < 2:
        return [], sents[0] if sents else ""
    return sents[:-1], sents[-1]


# ─── Token span detection ─────────────────────────────────────────────────────

def _find_premise_token_spans(
    full_text: str,
    premises: List[str],
    encoding,                # HuggingFace tokenizer output with offset_mapping
) -> List[Tuple[int, int]]:
    """
    For each premise, find the (start_tok, end_tok) span in the tokenized
    `full_text`.  `end_tok` is exclusive.

    Strategy:
      1. Find char span of each premise in full_text (first occurrence from
         the right half of the text, since QuaSAR prefix may also contain
         similar words).
      2. Map char span → token span using offset_mapping.

    Returns list of (start, end) tuples; (-1, -1) if not found.
    """
    offsets = encoding.offset_mapping  # list of (char_start, char_end) per token

    # Find where the original syllogism starts in full_text (after ABSTRACT_SEP)
    sep = ABSTRACT_SEP.strip()
    sep_pos = full_text.rfind(sep)
    orig_start_char = sep_pos + len(sep) + 1 if sep_pos != -1 else 0

    spans: List[Tuple[int, int]] = []
    search_from = orig_start_char

    for premise in premises:
        # Locate premise in the original portion of the text
        char_s = full_text.find(premise, search_from)
        if char_s == -1:
            # Fallback: search entire text
            char_s = full_text.find(premise)
        if char_s == -1:
            spans.append((-1, -1))
            continue
        char_e = char_s + len(premise)
        # Advance search_from to avoid matching the same premise twice
        search_from = char_s + 1

        # Map char span to token span
        tok_s, tok_e = None, None
        for ti, (cs, ce) in enumerate(offsets):
            if ce == 0 and cs == 0:
                continue  # special token
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
            spans.append((tok_s, tok_e + 1))  # exclusive end

    return spans


# ─── Dataset ─────────────────────────────────────────────────────────────────

class Subtask2Dataset(Dataset):
    """
    Dataset for JointSyllogismClassifier training and evaluation.

    Each item yields:
      input_ids        (T,)
      attention_mask   (T,)
      premise_spans    (MAX_PREMISES, 2) — token spans; -1 = pad
      num_premises     int
      label            int  (0/1)
      premise_labels   (MAX_PREMISES,) float  — 1.0 for relevant, 0.0 for not; -1.0 = pad
      plausibility     int  (0/1)
      id               str
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
        """Build model input text: optionally prepend QuaSAR abstract form."""
        syllogism = item["syllogism"]
        if self.abstractor is None:
            return syllogism
        # Strip _aug0/_aug1 suffix to look up the base ID in QuaSAR cache
        item_id = item.get("id", "")
        base_id = re.sub(r'_aug\d+$', '', item_id)
        abstract = self.abstractor.abstract(syllogism, item_id=base_id, quasar_mode=QUASAR_MODE)
        if abstract and abstract != syllogism:
            return abstract + ABSTRACT_SEP + syllogism
        return syllogism

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        premises, conclusion = parse_premises_and_conclusion(item)
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
        offset_mapping = enc["offset_mapping"]

        # ── Premise spans ──
        raw_spans = _find_premise_token_spans(
            full_text, premises[:n_prem], enc
        )
        # Pad to max_premises
        spans = []
        for i in range(self.max_premises):
            if i < n_prem and i < len(raw_spans):
                spans.append(raw_spans[i])
            else:
                spans.append((-1, -1))

        # ── Labels ──
        label        = LABEL2ID.get(item.get("validity", False), 0)
        plausibility = LABEL2ID.get(item.get("plausibility", False), 0)

        # Premise labels (MAX_PREMISES,), -1.0 = padding
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
            "premise_spans":  torch.tensor(spans, dtype=torch.long),     # (MAX_P, 2)
            "num_premises":   torch.tensor(n_prem, dtype=torch.long),
            "label":          torch.tensor(label, dtype=torch.long),
            "premise_labels": torch.tensor(premise_labels, dtype=torch.float),
            "plausibility":   torch.tensor(plausibility, dtype=torch.long),
            "id":             item["id"],
        }


# ─── Collate function (variable-length padding) ───────────────────────────────

def collate_fn_subtask2(batch: List[Dict]) -> Dict:
    """Pad input_ids and attention_mask to batch max length."""
    pad_id = 1  # XLM-RoBERTa padding token id

    max_len = max(b["input_ids"].shape[0] for b in batch)
    B       = len(batch)
    MP      = batch[0]["premise_spans"].shape[0]  # MAX_PREMISES

    input_ids      = torch.full((B, max_len), pad_id,  dtype=torch.long)
    attention_mask = torch.zeros((B, max_len),          dtype=torch.long)
    premise_spans  = torch.full((B, MP, 2), -1,         dtype=torch.long)
    num_premises   = torch.zeros(B,                     dtype=torch.long)
    labels         = torch.zeros(B,                     dtype=torch.long)
    premise_labels = torch.full((B, MP), -1.0,          dtype=torch.float)
    plausibility   = torch.zeros(B,                     dtype=torch.long)
    ids:List[str]  = []

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L]       = b["input_ids"]
        attention_mask[i, :L]  = b["attention_mask"]
        premise_spans[i]       = b["premise_spans"]
        num_premises[i]        = b["num_premises"]
        labels[i]              = b["label"]
        premise_labels[i]      = b["premise_labels"]
        plausibility[i]        = b["plausibility"]
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


# ─── DataLoader builders ─────────────────────────────────────────────────────

def _load_json(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def _train_val_split(
    data: List[Dict],
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    n_val = max(1, int(len(data) * val_ratio))
    val_idx   = set(indices[:n_val])
    train_idx = set(indices[n_val:])
    return (
        [data[i] for i in range(len(data)) if i in train_idx],
        [data[i] for i in range(len(data)) if i in val_idx],
    )


def build_dataloaders_subtask2(
    generated_train_path: str = GENERATED_TRAIN_PATH,
    test_path: str = TEST_DATA_PATH,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """
    Returns: train_loader, val_loader, test_loader, tokenizer
    """
    print("[DataLoader-S2] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # QuaSAR cache (reuse subtask1 caches for any overlapping items)
    quasar_cache = {}
    if USE_QUASI_SYMBOLIC:
        for cp in [S1_QUASAR_TRAIN_CACHE, S1_QUASAR_TEST_CACHE, S2_QUASAR_TEST_CACHE]:
            if os.path.exists(cp):
                with open(cp) as f:
                    chunk = json.load(f)
                quasar_cache.update(chunk)
                print(f"[DataLoader-S2] QuaSAR cache: {cp} ({len(chunk)} entries)")

    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache) if USE_QUASI_SYMBOLIC else None

    print("[DataLoader-S2] Loading generated training data ...")
    all_data = _load_json(generated_train_path)

    if VALIDATION_SPLIT > 0:
        train_data, val_data = _train_val_split(all_data, VALIDATION_SPLIT)
        print(f"  Train: {len(train_data)} | Val (from generated): {len(val_data)}")
    else:
        # Use ALL generated data for training, test data as validation
        # Avoids validating on generated data that has a different distribution
        train_data = all_data
        val_data = None
        print(f"  Train: {len(train_data)} (all generated, val=test)")

    print("[DataLoader-S2] Loading subtask2 test data ...")
    test_data = _load_json(test_path)
    print(f"  Test:  {len(test_data)}")

    # If VALIDATION_SPLIT=0, use the real test data as validation
    if val_data is None:
        val_data = test_data
        print(f"  Val: {len(val_data)} (= test data, real distribution)")

    train_ds = Subtask2Dataset(train_data, tokenizer, abstractor, has_labels=True)
    val_ds   = Subtask2Dataset(val_data,   tokenizer, abstractor, has_labels=True)
    test_ds  = Subtask2Dataset(test_data,  tokenizer, abstractor, has_labels=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_subtask2,
    )

    return train_loader, val_loader, test_loader, tokenizer
