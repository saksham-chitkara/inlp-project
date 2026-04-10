"""
dataset.py
----------
PyTorch Dataset for the logic-driven extension approach on Subtask 2
(monolingual English syllogisms + premise identification).

Key design:
  - Uses English `syllogism` field directly (no multilingual handling)
  - Extended context (verbalized inferred relations) is appended
  - Negative (corrupted) samples are generated for contrastive learning
  - Per-premise encoding for premise selection head
  - Train/val split groups augmentations of the same syllogism (via source_id)
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
    PyTorch Dataset for syllogistic reasoning with contrastive pairs
    and premise-level encoding for premise selection.

    Each item contains:
      - input_ids_plus, attention_mask_plus:  tokenized (c_plus, conclusion)
      - input_ids_minus, attention_mask_minus: tokenized (c_minus, conclusion)
      - premise_input_ids:     (max_premises, max_seq_len) per-premise encodings
      - premise_attention_mask: (max_premises, max_seq_len) per-premise masks
      - premise_labels:        (max_premises,) binary labels for relevant premises
      - premise_mask:          (max_premises,) which premise slots are real
      - num_premises:          number of actual premises
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
        self.max_premises = cfg.max_premises
        self.processed_data = self._process_all()

    def _process_item(self, item: Dict) -> Dict:
        """
        Process a single item through the logic pipeline.

        Pipeline:
          1. Parse English syllogism → entities, relations
          2. Infer implicit relations
          3. Verbalize extended relations
          4. Build c_plus = syllogism + extended_context
          5. Generate negatives → c_minus = syllogism + neg_context
          6. Extract conclusion from syllogism
          7. Split syllogism into per-premise sentences
        """
        syllogism = item["syllogism"]

        # Step 1: Parse English text for logic
        parsed = parse_syllogism(syllogism, self.cfg.spacy_model)

        # Step 2: Infer implicit relations
        extended_relations = infer_implicit_relations(parsed["relations"])

        # Step 3: Verbalize extended context
        extended_context = verbalize(extended_relations, parsed["rev_sym_map"])

        # Step 4: Build c_plus
        c_plus = f"{syllogism} {extended_context}" if extended_context else syllogism

        # Step 5: Generate negative sample
        if extended_relations:
            neg_relations = augment_relations(extended_relations)
            neg_context = verbalize(neg_relations, parsed["rev_sym_map"])
        else:
            neg_context = ""
        c_minus = f"{syllogism} {neg_context}" if neg_context else syllogism

        # Step 6: Extract conclusion (from parsed output — more reliable for English)
        conclusion = parsed["conclusion"] if parsed["conclusion"] else ""
        if not conclusion:
            # Fallback: last sentence
            sentences = [s.strip() for s in syllogism.split(".") if s.strip()]
            conclusion = sentences[-1] + "." if sentences else syllogism

        # Step 7: Split into per-premise sentences
        # Use the parsed premises directly
        premise_sentences = parsed["premises"] if parsed["premises"] else []
        # If parser failed, fall back to sentence splitting
        if not premise_sentences:
            sentences = [s.strip() + "." for s in syllogism.split(".") if s.strip()]
            if len(sentences) > 1:
                premise_sentences = sentences[:-1]
            else:
                premise_sentences = sentences

        result = {
            "id": item["id"],
            "c_plus": c_plus,
            "c_minus": c_minus,
            "conclusion": conclusion,
            "premise_sentences": premise_sentences,
        }

        if self.has_labels:
            result["label"] = self.cfg.label2id[item["validity"]]
            result["plausibility"] = 1 if item.get("plausibility", False) else 0
            # Premise labels: list of relevant premise indices (0-indexed)
            result["relevant_premises"] = item.get("relevant_premises", [])

        return result

    def _process_all(self) -> List[Dict]:
        """Process all items."""
        processed = []
        n_with_relations = 0
        for item in self.data:
            p = self._process_item(item)
            processed.append(p)
            if p["c_plus"] != item["syllogism"]:
                n_with_relations += 1

        print(f"  [Dataset] Processed {len(processed)} items, "
              f"{n_with_relations} enriched with extended logic context.")
        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_data[idx]
        max_len = self.cfg.max_seq_len
        max_p = self.max_premises

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

        # ─── Per-premise encoding for premise selection ───────────────
        premise_sentences = item["premise_sentences"]
        num_premises = min(len(premise_sentences), max_p)

        premise_input_ids = torch.zeros(max_p, max_len, dtype=torch.long)
        premise_attention_mask = torch.zeros(max_p, max_len, dtype=torch.long)
        premise_mask = torch.zeros(max_p, dtype=torch.float)
        premise_labels = torch.zeros(max_p, dtype=torch.float)

        conclusion = item["conclusion"]
        for i in range(num_premises):
            enc_premise = self.tokenizer(
                premise_sentences[i], conclusion,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            premise_input_ids[i] = enc_premise["input_ids"].squeeze(0)
            premise_attention_mask[i] = enc_premise["attention_mask"].squeeze(0)
            premise_mask[i] = 1.0

        result["premise_input_ids"] = premise_input_ids
        result["premise_attention_mask"] = premise_attention_mask
        result["premise_mask"] = premise_mask
        result["num_premises"] = torch.tensor(num_premises, dtype=torch.long)

        if self.has_labels:
            result["label"] = torch.tensor(item["label"], dtype=torch.long)
            result["plausibility"] = torch.tensor(item["plausibility"], dtype=torch.long)

            # Set premise labels
            for idx_p in item.get("relevant_premises", []):
                if idx_p < max_p:
                    premise_labels[idx_p] = 1.0
            result["premise_labels"] = premise_labels

        return result


# ─── Data Utilities ────────────────────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_val_split(
    data: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split by (validity × plausibility), grouping all augmentations
    of the same source syllogism into the same split to prevent data leakage.

    Uses `source_id` (if present) to group augmentations; falls back to
    the `syllogism` text otherwise.
    """
    random.seed(seed)

    # Group by source syllogism
    syl_groups: Dict[str, List[Dict]] = {}
    for item in data:
        group_key = item.get("source_id", item["syllogism"])
        syl_groups.setdefault(group_key, []).append(item)

    # Stratify at the syllogism level
    buckets: Dict[Tuple, List[str]] = {}
    for group_key, items in syl_groups.items():
        key = (items[0]["validity"], items[0].get("plausibility", None))
        buckets.setdefault(key, []).append(group_key)

    val_syls, train_syls = set(), set()
    for key, syls in buckets.items():
        random.shuffle(syls)
        n_val = max(1, int(len(syls) * val_ratio))
        val_syls.update(syls[:n_val])
        train_syls.update(syls[n_val:])

    train_data = [item for item in data
                  if item.get("source_id", item["syllogism"]) in train_syls]
    val_data = [item for item in data
                if item.get("source_id", item["syllogism"]) in val_syls]

    random.shuffle(train_data)
    random.shuffle(val_data)

    n_train_syls = len(train_syls)
    n_val_syls = len(val_syls)
    print(f"  [Split] Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  [Split] Unique source syllogisms: train={n_train_syls}, val={n_val_syls}, no overlap")
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
