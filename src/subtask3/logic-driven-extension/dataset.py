"""
dataset.py
----------
PyTorch Dataset for the logic-driven extension approach on Subtask 3
(multilingual syllogisms).

Key design:
  - English `syllogism` field is used for logic parsing (noun/quantifier extraction)
  - Translated `syllogism_t` field is used for model input (XLM-RoBERTa)
  - Extended context (verbalized inferred relations) is appended in English
  - Conclusion is taken from the translated text
  - Negative (corrupted) samples are generated for contrastive learning
  - Train/val split groups all translations of the same syllogism together
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from parser import parse_syllogism, split_syllogism
from logic_engine import infer_implicit_relations, augment_relations, verbalize


# ─── Dataset Class ─────────────────────────────────────────────────────────────

class SyllogismDataset(Dataset):
    """
    PyTorch Dataset for syllogistic reasoning with contrastive pairs.

    Each item contains:
      - input_ids_plus, attention_mask_plus:  tokenized (c_plus, conclusion)
      - input_ids_minus, attention_mask_minus: tokenized (c_minus, conclusion)
      - label:        0=invalid, 1=valid
      - plausibility: 0=implausible, 1=plausible
      - id:           original UUID string
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        cfg: SimpleNamespace,
        has_labels: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.has_labels = has_labels
        self.processed_data = self._process_all()

    def _extract_translated_conclusion(self, syllogism_t: str) -> str:
        """
        Extract the conclusion from translated text.
        Uses the same splitting heuristic — for translated text we can't
        rely on English markers, so we fall back to period-based splitting
        and take the last sentence.
        """
        # Try period-based splitting (works across languages)
        sentences = [s.strip() for s in syllogism_t.split(".") if s.strip()]
        if len(sentences) >= 3:
            return sentences[-1] + "."
        elif len(sentences) >= 1:
            return sentences[-1] + "."
        return syllogism_t

    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single item through the logic pipeline.

        Pipeline:
          1. Parse English syllogism → entities, relations
          2. Infer implicit relations
          3. Verbalize extended relations (English)
          4. Build c_plus = translated_text + extended_context
          5. Generate negatives → c_minus = translated_text + neg_context
          6. Extract translated conclusion
        """
        syllogism_en = item["syllogism"]
        syllogism_t = item.get("syllogism_t", syllogism_en)

        # Step 1: Parse English text for logic
        parsed = parse_syllogism(syllogism_en, self.cfg.spacy_model)

        # Step 2: Infer implicit relations
        extended_relations = infer_implicit_relations(parsed["relations"])

        # Step 3: Verbalize extended context (in English)
        extended_context = verbalize(extended_relations, parsed["rev_sym_map"])

        # Step 4: Build c_plus
        c_plus = f"{syllogism_t} {extended_context}" if extended_context else syllogism_t

        # Step 5: Generate negative sample
        if extended_relations:
            neg_relations = augment_relations(extended_relations)
            neg_context = verbalize(neg_relations, parsed["rev_sym_map"])
        else:
            neg_context = ""
        c_minus = f"{syllogism_t} {neg_context}" if neg_context else syllogism_t

        # Step 6: Extract conclusion from translated text
        conclusion_t = self._extract_translated_conclusion(syllogism_t)

        result = {
            "id": item["id"],
            "c_plus": c_plus,
            "c_minus": c_minus,
            "conclusion": conclusion_t,
        }

        if self.has_labels:
            result["label"] = self.cfg.label2id[item["validity"]]
            result["plausibility"] = 1 if item.get("plausibility", False) else 0

        return result

    def _process_all(self) -> List[Dict]:
        """Process all items."""
        processed = []
        n_with_relations = 0
        for item in self.data:
            p = self._process_item(item)
            processed.append(p)
            # Track how many items got extended
            if p["c_plus"] != item.get("syllogism_t", item["syllogism"]):
                n_with_relations += 1

        print(f"  [Dataset] Processed {len(processed)} items, "
              f"{n_with_relations} enriched with extended logic context.")
        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_data[idx]
        max_len = self.cfg.max_seq_len

        # Tokenize c_plus with conclusion
        enc_plus = self.tokenizer(
            item["c_plus"], item["conclusion"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        # Tokenize c_minus with conclusion
        enc_minus = self.tokenizer(
            item["c_minus"], item["conclusion"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

        result = {
            "id": item["id"],
            "input_ids_plus": enc_plus["input_ids"].squeeze(0),
            "attention_mask_plus": enc_plus["attention_mask"].squeeze(0),
            "input_ids_minus": enc_minus["input_ids"].squeeze(0),
            "attention_mask_minus": enc_minus["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            result["label"] = torch.tensor(item["label"], dtype=torch.long)
            result["plausibility"] = torch.tensor(item["plausibility"], dtype=torch.long)

        return result


# ─── Data Utilities ────────────────────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_val_split(
    data: List[Dict],
    val_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split by (validity × plausibility), grouping all translations
    of the same English syllogism into the same split to prevent data leakage.
    """
    random.seed(seed)

    # Group by English syllogism
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
        random.shuffle(syls)
        n_val = max(1, int(len(syls) * val_ratio))
        val_syls.update(syls[:n_val])
        train_syls.update(syls[n_val:])

    train_data = [item for item in data if item["syllogism"] in train_syls]
    val_data = [item for item in data if item["syllogism"] in val_syls]

    random.shuffle(train_data)
    random.shuffle(val_data)

    n_train_syls = len(train_syls)
    n_val_syls = len(val_syls)
    print(f"  [Split] Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  [Split] Unique syllogisms: train={n_train_syls}, val={n_val_syls}, no overlap")
    return train_data, val_data


def get_class_weights(data: List[Dict], label2id: dict) -> torch.Tensor:
    """Compute per-class weights for imbalanced datasets."""
    n_valid = sum(1 for d in data if d["validity"] is True)
    n_invalid = len(data) - n_valid
    total = len(data)

    w_valid = total / (2 * n_valid) if n_valid > 0 else 1.0
    w_invalid = total / (2 * n_invalid) if n_invalid > 0 else 1.0

    print(f"  [Weights] valid: {n_valid}, invalid: {n_invalid}")
    print(f"  [Weights] w_valid: {w_valid:.3f}, w_invalid: {w_invalid:.3f}")
    return torch.tensor([w_invalid, w_valid], dtype=torch.float)
