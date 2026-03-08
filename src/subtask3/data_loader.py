"""
data_loader.py
--------------
Loads and preprocesses the SemEval-2026 Task 11 Subtask 3 dataset (multilingual),
optionally augmenting each text with a quasi-symbolic abstract form.

Dataset fields (training):
  - id:         unique UUID string
  - syllogism:  full natural-language argument (premises + conclusion)
  - validity:   bool  — the TARGET label (must be predicted)
  - plausibility: bool — auxiliary label used ONLY for content-effect analysis
                         and activation-steering vector construction

Dataset fields (test):
  - id, syllogism, validity, plausibility
  (The test set here also has labels since it's the released eval set;
   the Codabench blind test set will lack labels.)
"""

import json
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE, EVAL_BATCH_SIZE,
    LABEL2ID, VALIDATION_SPLIT, SEED,
    USE_QUASI_SYMBOLIC, ABSTRACT_SEP, HF_CACHE_DIR,
    QUASAR_TRAIN_CACHE, QUASAR_TEST_CACHE, QUASAR_MODE,
)
from quasi_symbolic import QuasiSymbolicAbstractor


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SyllogismDataset(Dataset):
    """
    PyTorch Dataset for syllogistic reasoning (Subtask 1).

    Each item contains:
      - input_ids, attention_mask: tokenised input (abstract + original)
      - label:        0=invalid, 1=valid
      - plausibility: 0=implausible, 1=plausible  (for steering / CE analysis)
      - id:           original UUID string
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        abstractor: Optional[QuasiSymbolicAbstractor] = None,
        has_labels: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.abstractor = abstractor
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.data)

    def _build_text(self, item: dict) -> str:
        """
        Build the final input text for a single item.

        If USE_QUASI_SYMBOLIC is True:
          "<QuaSAR s2 formalisation> </s> <original text>"
          Falls back to spaCy X/Y/Z if QuaSAR cache miss.

        The abstract form provides the logical structure (content-free symbols)
        while the original text provides the concrete content.
        """
        # Use translated text for multilingual model input
        syllogism = item.get("syllogism_t", item["syllogism"])
        if USE_QUASI_SYMBOLIC and self.abstractor is not None:
            abstract = self.abstractor.abstract(syllogism, item_id=item.get("id"), quasar_mode=QUASAR_MODE)
            return abstract + ABSTRACT_SEP + syllogism
        return syllogism

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = self._build_text(item)

        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "id": item["id"],
        }

        if self.has_labels:
            result["label"] = torch.tensor(LABEL2ID[item["validity"]], dtype=torch.long)
            result["plausibility"] = torch.tensor(
                1 if item.get("plausibility", False) else 0, dtype=torch.long
            )

        return result


# ─── Data Loading Utilities ───────────────────────────────────────────────────

def load_json(path: str) -> List[Dict]:
    """Load a JSON file and return list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_dict(path: str) -> Dict:
    """Load a JSON file and return a dict (for QuaSAR caches)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_val_split(
    data: List[Dict],
    val_ratio: float = VALIDATION_SPLIT,
    seed: int = SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Stratified split by (validity x plausibility), deduplicated by English
    syllogism so that ALL translations of the same logical structure go into
    the same split. This prevents data leakage across train/val.
    """
    random.seed(seed)

    # Group items by English syllogism text
    syl_groups: Dict[str, List[Dict]] = {}
    for item in data:
        eng_syl = item["syllogism"]
        syl_groups.setdefault(eng_syl, []).append(item)

    # Stratify at the syllogism level (all translations share validity/plausibility)
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
    val_data   = [item for item in data if item["syllogism"] in val_syls]

    random.shuffle(train_data)
    random.shuffle(val_data)

    n_train_syls = len(train_syls)
    n_val_syls   = len(val_syls)
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  (Unique syllogisms: train={n_train_syls}, val={n_val_syls}, no overlap)")
    return train_data, val_data


def get_class_weights(data: List[Dict]) -> torch.Tensor:
    """
    Compute per-class weights for imbalanced datasets.
    Returns a weight tensor [w_invalid, w_valid] for use in CrossEntropyLoss.
    """
    n_valid = sum(1 for d in data if d["validity"] is True)
    n_invalid = len(data) - n_valid
    total = len(data)

    # Inverse frequency weighting
    w_valid = total / (2 * n_valid) if n_valid > 0 else 1.0
    w_invalid = total / (2 * n_invalid) if n_invalid > 0 else 1.0

    print(f"  Class distribution — valid: {n_valid}, invalid: {n_invalid}")
    print(f"  Class weights — valid: {w_valid:.3f}, invalid: {w_invalid:.3f}")
    return torch.tensor([w_invalid, w_valid], dtype=torch.float)


def get_weighted_sampler(data: List[Dict]) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to over-sample minority class during training,
    ensuring each batch contains a balanced mix of valid/invalid syllogisms.
    """
    labels = [LABEL2ID[d["validity"]] for d in data]
    class_counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_path: str,
    test_path: str,
    use_quasi_symbolic: bool = USE_QUASI_SYMBOLIC,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer, QuasiSymbolicAbstractor, int]:
    """
    Full data pipeline: load → split → tokenise → build DataLoaders.

    Returns:
      train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta
    """
    print("[DataLoader] Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)

    # No extra special tokens needed: we use </s> (already in XLM-RoBERTa vocab)
    # as the separator between abstract form and original text.
    vocab_delta = 0

    print("[DataLoader] Loading raw data …")
    all_train_data = load_json(train_path)
    test_data = load_json(test_path)

    print("[DataLoader] Performing stratified train/val split …")
    train_data, val_data = train_val_split(all_train_data)

    # Load QuaSAR cache (Llama-generated, technique 3)
    quasar_cache = {}
    if use_quasi_symbolic:
        import os
        for cache_path in [QUASAR_TRAIN_CACHE, QUASAR_TEST_CACHE]:
            if os.path.exists(cache_path):
                cache_data = load_json_dict(cache_path)
                quasar_cache.update(cache_data)
                print(f"[DataLoader] QuaSAR cache loaded: {cache_path} ({len(cache_data)} entries)")
        if not quasar_cache:
            print("[DataLoader] No QuaSAR cache found. Will use spaCy fallback (technique 1).")
            print(f"  Expected: {QUASAR_TRAIN_CACHE}")
            print(f"  Generate with: python quasar_generator.py --input <data> --output <cache>")

    # Quasi-symbolic abstractor (QuaSAR primary + spaCy fallback)
    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache) if use_quasi_symbolic else None
    if abstractor:
        if quasar_cache:
            print(f"[DataLoader] Abstractor: QuaSAR ({len(quasar_cache)} cached) + spaCy fallback.")
        else:
            print("[DataLoader] Abstractor: spaCy-only (no QuaSAR cache).")

    print("[DataLoader] Building datasets …")
    train_dataset = SyllogismDataset(train_data, tokenizer, abstractor, has_labels=True)
    val_dataset = SyllogismDataset(val_data, tokenizer, abstractor, has_labels=True)
    test_dataset = SyllogismDataset(test_data, tokenizer, abstractor, has_labels=True)

    # Weighted sampler for training to handle class imbalance
    sampler = get_weighted_sampler(train_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Done. Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, tokenizer, abstractor, vocab_delta
