#!/usr/bin/env python3
"""
Kaggle Runner — LReasoner Logic-Driven Extension (Subtask 1)

Self-contained single-file port of the original logic-driven-extension pipeline:
  model.py + dataset.py + logic_utils.py + trainer.py + main.py

Paste this into a Kaggle Notebook cell and run.

Setup:
  1. Upload kaggle_logic_extension.zip as a Kaggle Dataset (name: "logic-best-src")
  2. Accelerator: GPU T4 x2
  3. Add dataset, paste this script, run.

Architecture:
  XLM-RoBERTa-base → pooler_output → Dropout(0.1) → Linear(768, 2)
  Loss = CrossEntropyLoss + α × CosineEmbeddingLoss  (α=0.5)
  Early stopping on combined_score = accuracy / (1 + ln(1 + content_effect))

Pipeline:
  1. spaCy NLP: split syllogism into premises + conclusion
  2. Entity extraction: noun chunks → sym_0, sym_1, ...
  3. Regex pattern matching: extract logical relations (subset/disjoint/intersect/diff_intersect)
  4. Logical inference: transitivity, contraposition, symmetry
  5. Verbalize: relations → natural language (c_plus)
  6. Augment: corrupt relations → c_minus (for contrastive loss)
  7. Stratified train/val split (85/15, by validity×plausibility cell)
  8. Train XLM-RoBERTa with [c_plus, conclusion] + contrastive on [c_minus, conclusion]
  9. Early stopping on combined_score (content-effect-aware metric)
  10. Predict on test set
"""

import os
import sys
import json
import re
import copy
import random
import math
import subprocess
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer, XLMRobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
EPOCHS = 15           # more epochs for 4× larger dataset
BATCH_SIZE = 16       # doubled: more stable gradients with larger data
LR = 3e-5            # slightly higher: larger batch + more data
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06  # shorter warmup: dataset is large enough
MAX_LENGTH = 256
ALPHA = 0.5          # contrastive loss weight
PATIENCE = 5         # more patience: combined_score can fluctuate
VAL_RATIO = 0.15     # fraction of training data for validation
MODEL_NAME = "xlm-roberta-base"

# ============================================================================
# Setup
# ============================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Install spaCy model
print("[Setup] Installing spaCy English model ...")
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "spacy", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import spacy
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    nlp = spacy.load("en_core_web_sm")
print("[Setup] spaCy ready.\n")

# ============================================================================
# DATA PATHS
# ============================================================================
WORKING_DIR = "/kaggle/working"
TRAIN_DATA = None
TEST_DATA = None

def _find_data():
    global TRAIN_DATA, TEST_DATA
    # Kaggle input directories
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for ds in os.listdir(kaggle_input):
            base = os.path.join(kaggle_input, ds)
            for sub in [base, os.path.join(base, "data")]:
                # Prefer augmented data if available
                aug = os.path.join(sub, "augmented_train_data.json")
                orig = os.path.join(sub, "train_data.json")
                t = aug if os.path.isfile(aug) else orig
                if os.path.isfile(t):
                    TRAIN_DATA = t
                    TEST_DATA = os.path.join(sub, "test_data_subtask_1.json")
                    return
    # Local paths
    for base in [".", "/home/swamsingla/inlp-project"]:
        aug = os.path.join(base, "dataset", "train_data", "subtask 1", "augmented_train_data.json")
        orig = os.path.join(base, "dataset", "train_data", "subtask 1", "train_data.json")
        t = aug if os.path.isfile(aug) else orig
        if os.path.isfile(t):
            TRAIN_DATA = t
            TEST_DATA = os.path.join(base, "dataset", "test_data", "subtask 1", "test_data_subtask_1.json")
            return

_find_data()
if not os.path.isdir(WORKING_DIR):
    WORKING_DIR = os.path.join(".", "output")
    os.makedirs(WORKING_DIR, exist_ok=True)

assert TRAIN_DATA and os.path.isfile(TRAIN_DATA), f"Train data not found: {TRAIN_DATA}"
print(f"[Data] Train: {TRAIN_DATA}")
print(f"[Data] Test:  {TEST_DATA}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Setup] Device: {device}")
if device.type == "cuda":
    print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
print()


# ============================================================================
# LOGIC_UTILS — from logic_utils.py (original, 4 patterns)
# ============================================================================

def clean_sentence(sentence):
    sentence = sentence.lower().strip()
    markers = [
        "therefore, ", "consequently, ", "thus, ", "hence, ", "so, ",
        "it follows that ", "the only logical conclusion is that ", "as a result, ",
        "it is concluded that ", "one must conclude that "
    ]
    for marker in markers:
        if sentence.startswith(marker):
            sentence = sentence[len(marker):]
    return sentence.strip('. ')


def extract_and_encode_entities(sentences):
    """
    Extracts entities (noun chunks), limits them, and maps them to symbols.
    Returns the symbolically encoded sentences and the mapping to decode them.
    """
    doc = nlp(" ".join(sentences))
    entities = []

    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        words = text.split()
        if not words:
            continue

        # Remove determiners at the beginning
        stopwords = {'a', 'an', 'the', 'all', 'some', 'no', 'every', 'any', 'certain'}
        if words[0] in stopwords:
            text = ' '.join(words[1:])

        if text and text not in entities and text not in ['it', 'they', 'them', 'he', 'she']:
            entities.append(text)

    # Sort entities by length descending to replace larger spans first
    entities = sorted(entities, key=len, reverse=True)

    sym_map = {}
    rev_sym_map = {}
    for i, ent in enumerate(entities):
        sym = f"sym_{i}"
        sym_map[ent] = sym
        rev_sym_map[sym] = ent

    encoded_sentences = []
    for sent in sentences:
        s = clean_sentence(sent)
        for ent in entities:
            s = s.replace(ent, sym_map[ent])
        encoded_sentences.append(s)

    return encoded_sentences, rev_sym_map


PATTERNS = [
    (r"(?:all|every|any)\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "subset"),
    (r"(?:no|nothing that is a|there are no)\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "disjoint"),
    (r"some\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+not\s+(.*)", "diff_intersect"),
    (r"some\s+(.*?)\s+(?:that\s+)?(?:are|is)\s+(.*)", "intersect"),
]


def extract_logic_from_sentence(sentence):
    sentence = sentence.strip('. ')
    for pattern, rel_type in PATTERNS:
        match = re.search(pattern, sentence)
        if match:
            return rel_type, match.group(1).strip(), match.group(2).strip()
    return None, None, None


def extract_relations_from_encoded(encoded_sentences):
    relations = []
    for s in encoded_sentences:
        rel, A, B = extract_logic_from_sentence(s)
        if rel:
            relations.append({"type": rel, "args": (A, B)})
    return relations


def is_negated(term):
    return term.startswith("not(") and term.endswith(")")


def negate_term(term):
    if is_negated(term):
        return term[4:-1]
    return f"not({term})"


def infer_implicit_relations(relations):
    """Apply inference rules to extend context, covering rule set operations."""
    inferred = copy.deepcopy(relations)
    new_relations = True

    while new_relations:
        new_relations = False
        current_inferred = copy.deepcopy(inferred)

        for r1 in current_inferred:
            if r1["type"] == "subset":
                new_r = {"type": "disjoint", "args": (r1['args'][0], negate_term(r1['args'][1]))}
                if new_r not in inferred:
                    inferred.append(new_r); new_relations = True

            # Disjoint is symmetric
            if r1["type"] == "disjoint":
                new_r = {"type": "disjoint", "args": (r1['args'][1], r1['args'][0])}
                if new_r not in inferred:
                    inferred.append(new_r); new_relations = True

            for r2 in current_inferred:
                if r1 == r2:
                    continue

                # Rule 1: subset transitivity
                if r1["type"] == "subset" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "subset", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True

                # Rule 2: subset + disjoint
                elif r1["type"] == "subset" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "disjoint", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True

                # Rule 3: intersect + subset
                elif r1["type"] == "intersect" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True

                # Rule 4: intersect + disjoint
                elif r1["type"] == "intersect" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "diff_intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True

    return inferred


def augment_relations(relations):
    """Logically negate a relation for contrastive learning."""
    augmented = []
    for r in relations:
        aug_r = copy.deepcopy(r)
        operation = random.choice(["reverse", "negate"])

        if operation == "reverse":
            aug_r["args"] = (r["args"][1], r["args"][0])
        elif operation == "negate":
            if r["type"] == "subset":
                aug_r["type"] = random.choice(["disjoint", "diff_intersect"])
            elif r["type"] == "intersect":
                aug_r["type"] = "disjoint"
            elif r["type"] == "disjoint":
                aug_r["type"] = "intersect"
            elif r["type"] == "diff_intersect":
                aug_r["type"] = "subset"

        augmented.append(aug_r)
    return augmented


def format_term(term, rev_sym_map):
    if is_negated(term):
        base = term[4:-1]
        raw = rev_sym_map.get(base, base)
        return f"non-{raw}"
    return rev_sym_map.get(term, term)


def verbalize(relations, rev_sym_map):
    sentences = []
    for r in relations:
        A = format_term(r['args'][0], rev_sym_map)
        B = format_term(r['args'][1], rev_sym_map)
        if r['type'] == 'subset':
            sentences.append(f"All {A} are {B}.")
        elif r['type'] == 'disjoint':
            sentences.append(f"No {A} is {B}.")
        elif r['type'] == 'intersect':
            sentences.append(f"Some {A} are {B}.")
        elif r['type'] == 'diff_intersect':
            sentences.append(f"Some {A} are not {B}.")
    return " ".join(sentences)


# ============================================================================
# DATASET — from dataset.py (original)
# ============================================================================

label_map = {True: 1, False: 0}


class SyllogismDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=MAX_LENGTH):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_data = self._process_data()

    def _process_data(self):
        processed = []
        for item in tqdm(self.data, desc="Preprocessing Logic/Entities"):
            syllogism = item['syllogism']
            label = label_map[item['validity']]

            # Sentence extraction
            doc = nlp(syllogism)
            sentences = [sent.text.strip() for sent in doc.sents]

            # Enforce 3 sentences (2 premises, 1 conclusion) heuristically
            if len(sentences) >= 3:
                premises = sentences[:2]
                conclusion = sentences[-1]
            else:
                sentences = [s.strip() + "." for s in syllogism.split('.') if s.strip()]
                if len(sentences) >= 3:
                    premises = sentences[:2]
                    conclusion = sentences[-1]
                else:
                    premises = [syllogism]
                    conclusion = ""

            # Entity extraction & Symbolic assignment
            encoded_sentences, rev_sym_map = extract_and_encode_entities(premises + [conclusion])
            encoded_premises = encoded_sentences[:-1]

            # Logical Inference
            relations = extract_relations_from_encoded(encoded_premises)
            extended_relations = infer_implicit_relations(relations)
            extended_context_str = verbalize(extended_relations, rev_sym_map)

            c_plus = f"{' '.join(premises)} {extended_context_str}"

            # Data Augmentation (for Contrastive Loss)
            neg_relations = augment_relations(extended_relations)
            c_minus_str = verbalize(neg_relations, rev_sym_map)
            c_minus = f"{' '.join(premises)} {c_minus_str}"

            processed.append({
                'id': item['id'],
                'c_plus': c_plus,
                'c_minus': c_minus,
                'conclusion': conclusion,
                'label': label,
                'validity': item.get('validity'),
                'plausibility': item.get('plausibility'),
            })
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        enc_plus = self.tokenizer(
            item['c_plus'], item['conclusion'],
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt"
        )

        enc_minus = self.tokenizer(
            item['c_minus'], item['conclusion'],
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            'id': item['id'],
            'input_ids_plus': enc_plus['input_ids'].squeeze(),
            'attention_mask_plus': enc_plus['attention_mask'].squeeze(),
            'input_ids_minus': enc_minus['input_ids'].squeeze(),
            'attention_mask_minus': enc_minus['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


# ============================================================================
# MODEL — from model.py (original)
# ============================================================================

class LReasonerModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, alpha=ALPHA):
        super(LReasonerModel, self).__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 2)
        )
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, input_ids_plus, attention_mask_plus,
                input_ids_minus=None, attention_mask_minus=None, labels=None):
        # Forward pass true pairs
        outputs_plus = self.encoder(input_ids=input_ids_plus, attention_mask=attention_mask_plus)
        pooled_output_plus = outputs_plus.pooler_output
        logits = self.classifier(pooled_output_plus)

        loss = None
        if labels is not None:
            loss_ce = self.cross_entropy(logits, labels)
            loss_cl = 0

            if input_ids_minus is not None:
                # Forward pass corrupted pairs
                outputs_minus = self.encoder(input_ids=input_ids_minus, attention_mask=attention_mask_minus)
                pooled_output_minus = outputs_minus.pooler_output

                # Contrastive: maximize distance between true and false logic embeddings
                cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
                target = torch.full((pooled_output_plus.size(0),), -1).to(pooled_output_plus.device)
                loss_cl = cosine_loss(pooled_output_plus, pooled_output_minus, target)

            loss = loss_ce + self.alpha * loss_cl

        return logits, loss


# ============================================================================
# EVALUATION — mirrors official evaluation_script.py
# ============================================================================

def _calculate_accuracy(ground_truth, predictions):
    """Overall validity accuracy (%)."""
    gt_map = {item["id"]: item for item in ground_truth}
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt and isinstance(gt["validity"], bool) and isinstance(pred["validity"], bool):
            total += 1
            if gt["validity"] == pred["validity"]:
                correct += 1
    return (correct / total * 100) if total else 0.0


def _subgroup_accuracy(gt_map, predictions, gt_validity, gt_plausibility):
    """Accuracy on a specific (validity, plausibility) subgroup."""
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if (gt
                and gt.get("validity") == gt_validity
                and gt.get("plausibility") == gt_plausibility):
            if isinstance(gt["validity"], bool) and isinstance(pred["validity"], bool):
                total += 1
                if gt["validity"] == pred["validity"]:
                    correct += 1
    return (correct / total * 100) if total else 0.0


def compute_combined_score(ground_truth, predictions):
    """
    Official combined_score = accuracy / (1 + ln(1 + content_effect)).
    """
    gt_map = {item["id"]: item for item in ground_truth}
    overall_acc = _calculate_accuracy(ground_truth, predictions)

    apv = _subgroup_accuracy(gt_map, predictions, True, True)    # valid + plausible
    aiv = _subgroup_accuracy(gt_map, predictions, True, False)   # valid + implausible
    api = _subgroup_accuracy(gt_map, predictions, False, True)   # invalid + plausible
    aii = _subgroup_accuracy(gt_map, predictions, False, False)  # invalid + implausible

    intra = (abs(apv - aiv) + abs(api - aii)) / 2.0
    inter = (abs(apv - api) + abs(aiv - aii)) / 2.0
    tot_ce = (intra + inter) / 2.0

    combined = overall_acc / (1 + math.log(1 + tot_ce)) if tot_ce >= 0 else 0.0

    return {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(tot_ce, 4),
        "combined_score": round(combined, 4),
        "VP": round(apv, 2),
        "VI": round(aiv, 2),
        "IP": round(api, 2),
        "II": round(aii, 2),
    }


# ============================================================================
# TRAINING — from trainer.py (original)
# ============================================================================

def train(model, train_dataloader, val_dataloader, val_ground_truth, optimizer, scheduler,
          device, epochs=EPOCHS, patience=PATIENCE):
    """
    Training loop with early stopping based on combined_score.
    
    combined_score = accuracy / (1 + ln(1 + content_effect))
    
    This incentivizes the model to:
      1. Maximize classification accuracy
      2. Minimize content effect (plausibility bias)
    """
    best_combined_score = -1.0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids_plus = batch['input_ids_plus'].to(device)
            attention_mask_plus = batch['attention_mask_plus'].to(device)
            input_ids_minus = batch['input_ids_minus'].to(device)
            attention_mask_minus = batch['attention_mask_minus'].to(device)
            labels = batch['label'].to(device)

            logits, loss = model(input_ids_plus, attention_mask_plus,
                                 input_ids_minus, attention_mask_minus, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation — compute combined_score (accuracy + content effect)
        model.eval()
        total_val_loss = 0
        val_predictions = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids_plus = batch['input_ids_plus'].to(device)
                attention_mask_plus = batch['attention_mask_plus'].to(device)
                labels = batch['label'].to(device)

                logits, loss = model(input_ids_plus, attention_mask_plus, labels=labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                for i, p in enumerate(preds):
                    val_predictions.append({
                        "id": batch["id"][i],
                        "validity": bool(p)
                    })

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Compute official combined score on validation set
        metrics = compute_combined_score(val_ground_truth, val_predictions)
        val_acc = metrics['accuracy']
        val_ce = metrics['content_effect']
        val_combined = metrics['combined_score']

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"CE: {val_ce:.4f} | Combined: {val_combined:.4f} | "
              f"VP:{metrics['VP']:.0f} VI:{metrics['VI']:.0f} "
              f"IP:{metrics['IP']:.0f} II:{metrics['II']:.0f}")

        # Early stopping on combined_score (higher is better)
        if val_combined > best_combined_score:
            best_combined_score = val_combined
            early_stop_counter = 0
            model_path = os.path.join(WORKING_DIR, 'best_lreasoner_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  ↑ New best combined_score: {val_combined:.4f} → saved to {model_path}")
        else:
            early_stop_counter += 1
            print(f"  No improvement ({early_stop_counter}/{patience})")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"\nBest combined_score: {best_combined_score:.4f}")


def evaluate(model, test_dataset, test_dataloader, device):
    """Run inference on test set, return predictions list."""
    model.eval()
    predictions = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids_plus = batch['input_ids_plus'].to(device)
            attention_mask_plus = batch['attention_mask_plus'].to(device)

            logits, _ = model(input_ids_plus, attention_mask_plus)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i, p in enumerate(preds):
                predictions.append({
                    "id": batch["id"][i],
                    "validity": bool(p)
                })

    return predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  LReasoner Logic-Driven Extension — Subtask 1")
    print("  Early stopping on combined_score (accuracy / (1+ln(1+CE)))")
    print("=" * 70)

    # ---- Tokenizer ----
    print("\n[1/6] Loading tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # ---- Dataset ----
    print("\n[2/6] Loading & preprocessing training data...")
    full_dataset = SyllogismDataset(TRAIN_DATA, tokenizer, max_length=MAX_LENGTH)

    # ---- Stratified train/val split ----
    # Split by (validity, plausibility) cell to ensure balanced validation
    print(f"\n  Splitting {len(full_dataset)} examples into train/val ({1-VAL_RATIO:.0%}/{VAL_RATIO:.0%})...")

    cell_indices = {}  # (validity, plausibility) → [indices]
    for idx, item in enumerate(full_dataset.processed_data):
        cell = (item['validity'], item['plausibility'])
        cell_indices.setdefault(cell, []).append(idx)

    train_indices = []
    val_indices = []
    for cell, indices in cell_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * VAL_RATIO))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        print(f"    Cell {cell}: {len(indices)} total → {len(indices)-n_val} train, {n_val} val")

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # Build val ground truth for combined_score computation
    val_ground_truth = []
    for idx in val_indices:
        item = full_dataset.processed_data[idx]
        val_ground_truth.append({
            "id": item['id'],
            "validity": item['validity'],
            "plausibility": item['plausibility'],
        })

    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

    print("\n[3/6] Loading & preprocessing test data...")
    test_dataset = SyllogismDataset(TEST_DATA, tokenizer, max_length=MAX_LENGTH)

    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    # ---- Model ----
    print("\n[4/6] Initializing model...")
    model = LReasonerModel(model_name=MODEL_NAME, alpha=ALPHA).to(device)

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps
    )

    # ---- Train ----
    print(f"\n[5/6] Training for up to {EPOCHS} epochs...")
    print(f"  Batch size:         {BATCH_SIZE}")
    print(f"  Learning rate:      {LR}")
    print(f"  Alpha (contrastive):{ALPHA}")
    print(f"  Max length:         {MAX_LENGTH}")
    print(f"  Patience:           {PATIENCE} (on combined_score)")
    print(f"  Val ratio:          {VAL_RATIO}")
    print()

    train(model, train_dataloader, val_dataloader, val_ground_truth,
          optimizer, scheduler, device, epochs=EPOCHS, patience=PATIENCE)

    # ---- Load best model ----
    best_model_path = os.path.join(WORKING_DIR, 'best_lreasoner_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")

    # ---- Predict ----
    print("\n[6/6] Generating test predictions...")
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    predictions = evaluate(model, test_dataset, test_dataloader, device)

    # ---- Save predictions ----
    pred_path = os.path.join(WORKING_DIR, "predictions_subtask_1.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {pred_path}")

    # ---- Stats ----
    valid_count = sum(1 for p in predictions if p["validity"])
    print(f"Valid: {valid_count}, Invalid: {len(predictions) - valid_count}")

    # ---- Evaluate if test labels available ----
    with open(TEST_DATA, 'r') as f:
        test_data = json.load(f)

    has_labels = any(isinstance(item.get("validity"), bool) for item in test_data)
    if has_labels:
        metrics = compute_combined_score(test_data, predictions)
        print(f"\n{'=' * 70}")
        print(f"  TEST RESULTS")
        print(f"{'=' * 70}")
        print(f"  Accuracy:       {metrics['accuracy']:.2f}%")
        print(f"  Content Effect: {metrics['content_effect']:.4f}")
        print(f"  Combined Score: {metrics['combined_score']:.4f}")
        print(f"  VP: {metrics['VP']:.2f}%  VI: {metrics['VI']:.2f}%  "
              f"IP: {metrics['IP']:.2f}%  II: {metrics['II']:.2f}%")
        print(f"{'=' * 70}")
    else:
        print("No test labels available for evaluation.")

    print("\n✅ Pipeline complete!")
    return predictions


if __name__ == "__main__":
    main()
