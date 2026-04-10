#!/usr/bin/env python3
"""
Kaggle Runner v8 — LReasoner + Activation Steering (CAA + K-CAST)
==================================================================

3-Stage Pipeline:
  Stage 1: Train XLM-RoBERTa LReasoner with logic-driven contrastive loss
  Stage 2: Compute CAA / K-CAST steering vectors from training activations
  Stage 3: Grid search best (layer, alpha) and apply steering at inference

Uses augmented+original training data (augmented_train_data.json = 3840 samples).

How to run on Kaggle:
  1. Create a Kaggle Dataset with these files:
       - augmented_train_data.json   (3840 samples, augmented+original)
       - test_data_subtask_1.json    (test data)
     Name the dataset e.g. "syllogism-data"
  2. Create a new Kaggle Notebook → Accelerator: GPU T4 x2
  3. Add your dataset
  4. Paste this ENTIRE script into a single code cell and Run All
  5. Download predictions from /kaggle/working/

Architecture:
  XLM-RoBERTa-base → pooler + layer hidden states → Dropout → Linear(768, 2)
  Loss = CE + α × CosineEmbeddingLoss (contrastive on logic-corrupted pairs)
  + Post-hoc Activation Steering (CAA & K-CAST) on encoder hidden layers

Steering (based on Valentino et al., 2025):
  - Content-aligned    (PV + II): plausibility agrees with validity
  - Content-conflicting (IV + PI): plausibility disagrees with validity
  - δ = mean(aligned) - mean(conflicting)  per layer
  - CAA:   h' = h + α·δ  (static)
  - K-CAST: h' = h + f(kNN)·α·δ  (adaptive per-example magnitude)
"""

import os, sys, json, re, copy, random, math, gc, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer, XLMRobertaModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# ============================================================================
# CONFIG
# ============================================================================
SEED = 42
EPOCHS = 15
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_LENGTH = 256
ALPHA_CL = 0.5        # contrastive loss weight
PATIENCE = 5
VAL_RATIO = 0.15
MODEL_NAME = "xlm-roberta-base"

# Steering config
STEERING_LAYERS = [9, 10, 11]   # top 3 layers of 12-layer XLM-RoBERTa
STEERING_KNN = 10
STEERING_ALPHA_RANGE = (-3.0, 3.0)
STEERING_ALPHA_STEPS = 25

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

# Install spaCy
print("[Setup] Installing spaCy English model ...")
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.check_call(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "spacy", "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import spacy
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for ds in os.listdir(kaggle_input):
            base = os.path.join(kaggle_input, ds)
            for sub in [base, os.path.join(base, "data")]:
                # Prefer augmented data
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
# LOGIC UTILS
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
    doc = nlp(" ".join(sentences))
    entities = []
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        words = text.split()
        if not words:
            continue
        stopwords = {'a', 'an', 'the', 'all', 'some', 'no', 'every', 'any', 'certain'}
        if words[0] in stopwords:
            text = ' '.join(words[1:])
        if text and text not in entities and text not in ['it', 'they', 'them', 'he', 'she']:
            entities.append(text)
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
            if r1["type"] == "disjoint":
                new_r = {"type": "disjoint", "args": (r1['args'][1], r1['args'][0])}
                if new_r not in inferred:
                    inferred.append(new_r); new_relations = True
            for r2 in current_inferred:
                if r1 == r2:
                    continue
                if r1["type"] == "subset" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "subset", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                elif r1["type"] == "subset" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "disjoint", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                elif r1["type"] == "intersect" and r2["type"] == "subset":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
                elif r1["type"] == "intersect" and r2["type"] == "disjoint":
                    if r1["args"][1] == r2["args"][0]:
                        new_r = {"type": "diff_intersect", "args": (r1["args"][0], r2["args"][1])}
                        if new_r not in inferred:
                            inferred.append(new_r); new_relations = True
    return inferred


def augment_relations(relations):
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
# DATASET
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
            doc = nlp(syllogism)
            sentences = [sent.text.strip() for sent in doc.sents]
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
            encoded_sentences, rev_sym_map = extract_and_encode_entities(premises + [conclusion])
            encoded_premises = encoded_sentences[:-1]
            relations = extract_relations_from_encoded(encoded_premises)
            extended_relations = infer_implicit_relations(relations)
            extended_context_str = verbalize(extended_relations, rev_sym_map)
            c_plus = f"{' '.join(premises)} {extended_context_str}"
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
            return_tensors="pt")
        enc_minus = self.tokenizer(
            item['c_minus'], item['conclusion'],
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt")
        return {
            'id': item['id'],
            'input_ids_plus': enc_plus['input_ids'].squeeze(),
            'attention_mask_plus': enc_plus['attention_mask'].squeeze(),
            'input_ids_minus': enc_minus['input_ids'].squeeze(),
            'attention_mask_minus': enc_minus['attention_mask'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


# ============================================================================
# MODEL — with hidden state extraction for steering
# ============================================================================

class LReasonerModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, alpha=ALPHA_CL):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.config.hidden_size, 2)
        )
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()

        # Steering state
        self._steering_vectors = {}   # layer_idx → delta tensor
        self._steering_alpha = 0.0
        self._steering_enabled = False

    def forward(self, input_ids_plus, attention_mask_plus,
                input_ids_minus=None, attention_mask_minus=None, labels=None):
        outputs_plus = self.encoder(
            input_ids=input_ids_plus, attention_mask=attention_mask_plus,
            output_hidden_states=self._steering_enabled)

        if self._steering_enabled and outputs_plus.hidden_states is not None:
            # Apply additive steering to pooler input
            pooled = self._apply_steering(outputs_plus)
        else:
            pooled = outputs_plus.pooler_output

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_ce = self.cross_entropy(logits, labels)
            loss_cl = 0
            if input_ids_minus is not None:
                outputs_minus = self.encoder(
                    input_ids=input_ids_minus, attention_mask=attention_mask_minus)
                pooled_minus = outputs_minus.pooler_output
                cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
                target = torch.full((pooled.size(0),), -1).to(pooled.device)
                loss_cl = cosine_loss(pooled, pooled_minus, target)
            loss = loss_ce + self.alpha * loss_cl

        return logits, loss

    def _apply_steering(self, encoder_output):
        """Apply CAA steering: modify hidden state at target layer, then re-pool."""
        hs = encoder_output.hidden_states  # tuple of (B, seq_len, H)
        # We steer by modifying the [CLS] token representation at the target layer
        # and passing it through the pooler
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]  # (B, H)

        for layer_idx, delta in self._steering_vectors.items():
            # Add steering vector to CLS representation
            # delta is raw (NOT unit-normalized), alpha scales it
            cls_hidden = cls_hidden + self._steering_alpha * delta.to(cls_hidden.device)

        # Apply the pooler (dense + tanh) manually
        pooled = self.encoder.pooler.dense(cls_hidden)
        pooled = self.encoder.pooler.activation(pooled)
        return pooled

    def get_layer_hidden_states(self, input_ids, attention_mask, layers):
        """Extract [CLS] hidden states at specified layers."""
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True)
        result = {}
        for l in layers:
            # hidden_states[0] = embeddings, [1] = layer 1, ..., [12] = layer 12
            idx = l + 1 if l + 1 < len(outputs.hidden_states) else -1
            result[l] = outputs.hidden_states[idx][:, 0, :].cpu()  # (B, H)
        return result

    def set_steering(self, vectors, alpha):
        """Enable CAA steering with given vectors and alpha."""
        self._steering_vectors = vectors
        self._steering_alpha = alpha
        self._steering_enabled = True

    def disable_steering(self):
        self._steering_vectors = {}
        self._steering_alpha = 0.0
        self._steering_enabled = False


# ============================================================================
# EVALUATION — official combined_score
# ============================================================================

def _subgroup_accuracy(gt_map, predictions, gt_validity, gt_plausibility):
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if (gt and gt.get("validity") == gt_validity
                and gt.get("plausibility") == gt_plausibility):
            if isinstance(gt["validity"], bool) and isinstance(pred["validity"], bool):
                total += 1
                if gt["validity"] == pred["validity"]:
                    correct += 1
    return (correct / total * 100) if total else 0.0


def compute_combined_score(ground_truth, predictions):
    gt_map = {item["id"]: item for item in ground_truth}
    correct = total = 0
    for pred in predictions:
        gt = gt_map.get(pred["id"])
        if gt and isinstance(gt.get("validity"), bool) and isinstance(pred.get("validity"), bool):
            total += 1
            if gt["validity"] == pred["validity"]:
                correct += 1
    overall_acc = (correct / total * 100) if total else 0.0

    apv = _subgroup_accuracy(gt_map, predictions, True, True)
    aiv = _subgroup_accuracy(gt_map, predictions, True, False)
    api = _subgroup_accuracy(gt_map, predictions, False, True)
    aii = _subgroup_accuracy(gt_map, predictions, False, False)

    intra = (abs(apv - aiv) + abs(api - aii)) / 2.0
    inter = (abs(apv - api) + abs(aiv - aii)) / 2.0
    tot_ce = (intra + inter) / 2.0
    combined = overall_acc / (1 + math.log(1 + tot_ce)) if tot_ce >= 0 else 0.0

    return {
        "accuracy": round(overall_acc, 4),
        "content_effect": round(tot_ce, 4),
        "combined_score": round(combined, 4),
        "VP": round(apv, 2), "VI": round(aiv, 2),
        "IP": round(api, 2), "II": round(aii, 2),
    }


def evaluate_model(model, dataloader, device_obj):
    """Run inference, return list of prediction dicts."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(
                batch['input_ids_plus'].to(device_obj),
                batch['attention_mask_plus'].to(device_obj))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i, p in enumerate(preds):
                predictions.append({"id": batch["id"][i], "validity": bool(p)})
    return predictions


# ============================================================================
# ACTIVATION STEERING
# ============================================================================

class ActivationSteerer:
    """CAA + K-CAST steering for LReasoner (XLM-RoBERTa encoder)."""

    def __init__(self, model, device_obj):
        self.model = model
        self.device = device_obj
        self.steering_vectors = {}   # layer → delta (H,)
        self.kcast_store = {}        # layer → {activations, directions}
        self.best_layer_caa = STEERING_LAYERS[-1]
        self.best_alpha_caa = 0.0
        self.best_combined_caa = -1.0
        self.best_layer_kcast = STEERING_LAYERS[-1]
        self.best_alpha_kcast = 0.0
        self.best_combined_kcast = -1.0

    # ── 1. Collect activations ─────────────────────────────────────────────

    def collect_activations(self, loader, layers=STEERING_LAYERS):
        """Collect [CLS] hidden states + labels + plausibility + predictions."""
        self.model.eval()
        self.model.disable_steering()
        layer_data = {l: {"acts": [], "labels": [], "plaus": [], "preds": []}
                      for l in layers}

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting activations"):
                ids_plus = batch['input_ids_plus'].to(self.device)
                mask_plus = batch['attention_mask_plus'].to(self.device)
                labels = batch['label']

                logits, _ = self.model(ids_plus, mask_plus)
                preds = logits.argmax(dim=-1).cpu()

                hs = self.model.get_layer_hidden_states(ids_plus, mask_plus, layers)
                for l in layers:
                    layer_data[l]["acts"].append(hs[l])
                    layer_data[l]["labels"].append(labels)
                    layer_data[l]["preds"].append(preds)

        # Get plausibility from dataset
        # We need to reconstruct plausibility from the original data
        # It's in the processed_data but not in batches, so collect separately
        result = {}
        for l in layers:
            result[l] = {
                "activations": torch.cat(layer_data[l]["acts"], dim=0),
                "labels": torch.cat(layer_data[l]["labels"], dim=0),
                "preds": torch.cat(layer_data[l]["preds"], dim=0),
            }
        return result

    def collect_activations_with_plausibility(self, dataset, loader, layers=STEERING_LAYERS):
        """Collect activations with plausibility info for content-based steering."""
        # First collect from batches
        layer_data = self.collect_activations(loader, layers)

        # Add plausibility from dataset
        plaus_list = []
        for idx in range(len(dataset)):
            if hasattr(dataset, 'dataset'):
                # It's a Subset
                real_idx = dataset.indices[idx]
                item = dataset.dataset.processed_data[real_idx]
            else:
                item = dataset.processed_data[idx]
            p = item.get('plausibility')
            plaus_list.append(1 if p else 0)
        plaus_tensor = torch.tensor(plaus_list, dtype=torch.long)

        for l in layers:
            layer_data[l]["plaus"] = plaus_tensor[:layer_data[l]["activations"].size(0)]
        return layer_data

    # ── 2. Compute steering vectors (content-based D+/D-) ─────────────────

    def compute_steering_vectors(self, layer_data, layers=STEERING_LAYERS):
        """
        Content-based construction:
          Aligned    (PV + II): plausibility agrees with validity → content shortcut works
          Conflicting (IV + PI): plausibility disagrees → must reason from structure
          δ = mean(aligned) - mean(conflicting)
        Steering with +α pushes toward aligned-like representations.
        """
        for l in layers:
            acts = layer_data[l]["activations"]
            labels = layer_data[l]["labels"]
            plaus = layer_data[l]["plaus"]

            # Content-aligned: (valid & plausible) OR (invalid & implausible)
            mask_aligned = ((labels == 1) & (plaus == 1)) | ((labels == 0) & (plaus == 0))
            # Content-conflicting: (valid & implausible) OR (invalid & plausible)
            mask_conflict = ((labels == 1) & (plaus == 0)) | ((labels == 0) & (plaus == 1))

            n_a = mask_aligned.sum().item()
            n_c = mask_conflict.sum().item()
            print(f"  Layer {l}: aligned={n_a}, conflicting={n_c}")

            if n_a == 0 or n_c == 0:
                print(f"  ⚠ Skipping layer {l}: empty group")
                continue

            mean_aligned = acts[mask_aligned].float().mean(dim=0)
            mean_conflict = acts[mask_conflict].float().mean(dim=0)
            delta = mean_aligned - mean_conflict  # raw, NOT unit-normalized

            self.steering_vectors[l] = delta
            norm = delta.norm().item()
            print(f"    ||δ|| = {norm:.4f}")

            # K-CAST store
            self.kcast_store[l] = {
                "activations": acts,
                "directions": (mask_aligned.float() * 2 - 1),  # +1=aligned, -1=conflicting
            }

    # ── 3. Linear probe (to find best layers) ─────────────────────────────

    def linear_probe_layers(self, layer_data, layers=STEERING_LAYERS):
        """Use logistic regression to find which layers are most informative."""
        print("\n[Steering] Linear probe accuracy per layer:")
        results = {}
        for l in layers:
            acts = layer_data[l]["activations"].float().numpy()
            labels = layer_data[l]["labels"].numpy()
            try:
                clf = LogisticRegression(max_iter=500, solver='lbfgs')
                clf.fit(acts, labels)
                acc = clf.score(acts, labels) * 100
                results[l] = acc
                print(f"  Layer {l}: {acc:.1f}%")
            except Exception as e:
                print(f"  Layer {l}: FAILED ({e})")
                results[l] = 0.0
        # Return top 2 layers
        top = sorted(results.items(), key=lambda x: -x[1])[:2]
        top_layers = [l for l, _ in top]
        print(f"  → Top layers: {top_layers}")
        return top_layers

    # ── 4. Grid search CAA ─────────────────────────────────────────────────

    def grid_search_caa(self, model, val_loader, val_gt, layers=None,
                        alpha_range=STEERING_ALPHA_RANGE,
                        alpha_steps=STEERING_ALPHA_STEPS):
        """Grid search (layer, alpha) for CAA, maximising combined_score."""
        if layers is None:
            layers = list(self.steering_vectors.keys())
        alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_steps).tolist()
        total = len(layers) * len(alphas)
        print(f"\n[CAA Grid] {len(layers)} layers × {len(alphas)} alphas = {total} combos")

        step = 0
        for l in layers:
            if l not in self.steering_vectors:
                continue
            delta = self.steering_vectors[l]  # raw, NOT normalized

            for alpha_val in alphas:
                step += 1
                model.set_steering({l: delta}, alpha_val)
                preds = evaluate_model(model, val_loader, self.device)
                metrics = compute_combined_score(val_gt, preds)
                combined = metrics["combined_score"]

                if step % 5 == 0 or combined > self.best_combined_caa:
                    print(f"  [{step}/{total}] CAA L{l} α={alpha_val:+.2f} → "
                          f"acc={metrics['accuracy']:.1f}% combined={combined:.4f}")
                    sys.stdout.flush()

                if combined > self.best_combined_caa:
                    self.best_combined_caa = combined
                    self.best_layer_caa = l
                    self.best_alpha_caa = alpha_val
                    print(f"        ↑ NEW BEST CAA")

        model.disable_steering()
        print(f"\n[CAA Grid] Best: L{self.best_layer_caa} "
              f"α={self.best_alpha_caa:+.2f} combined={self.best_combined_caa:.4f}")

    # ── 5. Grid search K-CAST ──────────────────────────────────────────────

    def grid_search_kcast(self, model, val_loader, val_gt, layers=None,
                          alpha_range=STEERING_ALPHA_RANGE,
                          alpha_steps=STEERING_ALPHA_STEPS):
        """Grid search for K-CAST: per-example adaptive magnitude via kNN."""
        if layers is None:
            layers = list(self.steering_vectors.keys())
        alphas = np.linspace(alpha_range[0], alpha_range[1], alpha_steps).tolist()
        k = STEERING_KNN
        total = len(layers) * len(alphas)
        print(f"\n[K-CAST Grid] {len(layers)} layers × {len(alphas)} alphas = {total} combos")

        step = 0
        for l in layers:
            if l not in self.steering_vectors:
                continue
            delta = self.steering_vectors[l]  # raw, NOT normalized
            stored_acts = self.kcast_store[l]["activations"]
            stored_dirs = self.kcast_store[l]["directions"]
            stored_norm = F.normalize(stored_acts.float(), dim=-1)

            for alpha_val in alphas:
                step += 1
                # K-CAST: per-example magnitude
                preds = []
                model.disable_steering()
                model.eval()

                with torch.no_grad():
                    for batch in val_loader:
                        ids_plus = batch['input_ids_plus'].to(self.device)
                        mask_plus = batch['attention_mask_plus'].to(self.device)

                        # Get hidden state for kNN lookup
                        hs = model.get_layer_hidden_states(ids_plus, mask_plus, [l])
                        h = hs[l]  # (B, H)
                        h_norm = F.normalize(h.float(), dim=-1)

                        # kNN vote
                        sims = h_norm @ stored_norm.T  # (B, N)
                        _, topk_idx = sims.topk(k=min(k, sims.size(1)), dim=-1)

                        for b in range(ids_plus.size(0)):
                            votes = stored_dirs[topk_idx[b]]
                            aligned_frac = (votes > 0).float().mean().item()
                            magnitude = aligned_frac * abs(alpha_val)

                            # Apply steering for this single example
                            model.set_steering({l: delta}, magnitude)
                            logits, _ = model(
                                ids_plus[b:b+1], mask_plus[b:b+1])
                            pred_label = logits.argmax(dim=-1).item()
                            preds.append({
                                "id": batch["id"][b],
                                "validity": bool(pred_label)
                            })

                model.disable_steering()
                metrics = compute_combined_score(val_gt, preds)
                combined = metrics["combined_score"]

                if step % 5 == 0 or combined > self.best_combined_kcast:
                    print(f"  [{step}/{total}] KCAST L{l} α={alpha_val:+.2f} → "
                          f"acc={metrics['accuracy']:.1f}% combined={combined:.4f}")
                    sys.stdout.flush()

                if combined > self.best_combined_kcast:
                    self.best_combined_kcast = combined
                    self.best_layer_kcast = l
                    self.best_alpha_kcast = alpha_val
                    print(f"        ↑ NEW BEST K-CAST")

        print(f"\n[K-CAST Grid] Best: L{self.best_layer_kcast} "
              f"α={self.best_alpha_kcast:+.2f} combined={self.best_combined_kcast:.4f}")

    # ── 6. Full-dataset CAA inference ──────────────────────────────────────

    def apply_caa_inference(self, model, dataloader, device_obj):
        delta = self.steering_vectors[self.best_layer_caa]
        model.set_steering({self.best_layer_caa: delta}, self.best_alpha_caa)
        preds = evaluate_model(model, dataloader, device_obj)
        model.disable_steering()
        return preds

    # ── 7. Full-dataset K-CAST inference ───────────────────────────────────

    def apply_kcast_inference(self, model, dataloader, device_obj):
        l = self.best_layer_kcast
        alpha = self.best_alpha_kcast
        delta = self.steering_vectors[l]
        stored_acts = self.kcast_store[l]["activations"]
        stored_dirs = self.kcast_store[l]["directions"]
        stored_norm = F.normalize(stored_acts.float(), dim=-1)
        k = STEERING_KNN

        model.disable_steering()
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="K-CAST inference"):
                ids_plus = batch['input_ids_plus'].to(device_obj)
                mask_plus = batch['attention_mask_plus'].to(device_obj)

                hs = model.get_layer_hidden_states(ids_plus, mask_plus, [l])
                h = hs[l]
                h_norm = F.normalize(h.float(), dim=-1)
                sims = h_norm @ stored_norm.T
                _, topk_idx = sims.topk(k=min(k, sims.size(1)), dim=-1)

                for b in range(ids_plus.size(0)):
                    votes = stored_dirs[topk_idx[b]]
                    aligned_frac = (votes > 0).float().mean().item()
                    magnitude = aligned_frac * abs(alpha)

                    model.set_steering({l: delta}, magnitude)
                    logits, _ = model(ids_plus[b:b+1], mask_plus[b:b+1])
                    pred_label = logits.argmax(dim=-1).item()
                    predictions.append({
                        "id": batch["id"][b],
                        "validity": bool(pred_label)
                    })

        model.disable_steering()
        return predictions

    # ── 8. Save / Load ─────────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            "steering_vectors": self.steering_vectors,
            "kcast_store": self.kcast_store,
            "best_layer_caa": self.best_layer_caa,
            "best_alpha_caa": self.best_alpha_caa,
            "best_combined_caa": self.best_combined_caa,
            "best_layer_kcast": self.best_layer_kcast,
            "best_alpha_kcast": self.best_alpha_kcast,
            "best_combined_kcast": self.best_combined_kcast,
        }, path)
        print(f"[Steering] Saved to {path}")

    def load(self, path):
        p = torch.load(path, map_location="cpu", weights_only=False)
        self.steering_vectors = p["steering_vectors"]
        self.kcast_store = p["kcast_store"]
        self.best_layer_caa = p.get("best_layer_caa", STEERING_LAYERS[-1])
        self.best_alpha_caa = p.get("best_alpha_caa", 0.0)
        self.best_combined_caa = p.get("best_combined_caa", -1.0)
        self.best_layer_kcast = p.get("best_layer_kcast", STEERING_LAYERS[-1])
        self.best_alpha_kcast = p.get("best_alpha_kcast", 0.0)
        self.best_combined_kcast = p.get("best_combined_kcast", -1.0)
        print(f"[Steering] Loaded from {path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, val_gt, optimizer, scheduler,
                device_obj, epochs=EPOCHS, patience=PATIENCE):
    best_combined = -1.0
    counter = 0

    for epoch in range(epochs):
        model.train()
        model.disable_steering()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            logits, loss = model(
                batch['input_ids_plus'].to(device_obj),
                batch['attention_mask_plus'].to(device_obj),
                batch['input_ids_minus'].to(device_obj),
                batch['attention_mask_minus'].to(device_obj),
                batch['label'].to(device_obj))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds = evaluate_model(model, val_loader, device_obj)
        metrics = compute_combined_score(val_gt, val_preds)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
              f"Acc: {metrics['accuracy']:.2f}% | CE: {metrics['content_effect']:.4f} | "
              f"Combined: {metrics['combined_score']:.4f} | "
              f"VP:{metrics['VP']:.0f} VI:{metrics['VI']:.0f} "
              f"IP:{metrics['IP']:.0f} II:{metrics['II']:.0f}")

        if metrics['combined_score'] > best_combined:
            best_combined = metrics['combined_score']
            counter = 0
            model_path = os.path.join(WORKING_DIR, 'best_lreasoner_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  ↑ New best: {best_combined:.4f} → saved")
        else:
            counter += 1
            print(f"  No improvement ({counter}/{patience})")
            if counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    print(f"\nBest combined_score during training: {best_combined:.4f}")
    return best_combined


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 70)
    print("  LReasoner + Activation Steering (CAA + K-CAST)")
    print("  3-Stage Pipeline: Train → Probe → Steer")
    print("=" * 70)

    # ── 1. Tokenizer ───────────────────────────────────────────────────────
    print("\n[1/8] Loading tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Dataset ─────────────────────────────────────────────────────────
    print("\n[2/8] Loading & preprocessing data...")
    full_dataset = SyllogismDataset(TRAIN_DATA, tokenizer, max_length=MAX_LENGTH)
    print(f"  Total: {len(full_dataset)} samples")

    # Stratified split by (validity, plausibility)
    cell_indices = {}
    for idx, item in enumerate(full_dataset.processed_data):
        cell = (item['validity'], item['plausibility'])
        cell_indices.setdefault(cell, []).append(idx)

    train_indices, val_indices = [], []
    for cell, indices in cell_indices.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * VAL_RATIO))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
        print(f"    Cell {cell}: {len(indices)} total → {len(indices)-n_val} train, {n_val} val")

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    val_gt = []
    for idx in val_indices:
        item = full_dataset.processed_data[idx]
        val_gt.append({
            "id": item['id'],
            "validity": item['validity'],
            "plausibility": item['plausibility'],
        })

    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    # Test data
    print("\n[3/8] Loading test data...")
    test_dataset = None
    test_loader = None
    if TEST_DATA and os.path.isfile(TEST_DATA):
        test_dataset = SyllogismDataset(TEST_DATA, tokenizer, max_length=MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        print(f"  Test: {len(test_dataset)} samples")
    else:
        print("  No test data found.")

    # ── 3. Model + Train ───────────────────────────────────────────────────
    print("\n[4/8] Training model...")
    model = LReasonerModel(model_name=MODEL_NAME, alpha=ALPHA_CL).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    train_model(model, train_loader, val_loader, val_gt, optimizer, scheduler,
                device, epochs=EPOCHS, patience=PATIENCE)

    # Load best
    best_path = os.path.join(WORKING_DIR, 'best_lreasoner_model.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path}")

    # ── 4. Baseline evaluation (no steering) ──────────────────────────────
    print("\n[5/8] Baseline evaluation (no steering)...")
    model.disable_steering()

    base_val_preds = evaluate_model(model, val_loader, device)
    base_val_metrics = compute_combined_score(val_gt, base_val_preds)
    print(f"  Val:  Acc={base_val_metrics['accuracy']:.2f}% "
          f"CE={base_val_metrics['content_effect']:.4f} "
          f"Combined={base_val_metrics['combined_score']:.4f}")

    base_test_preds = None
    base_test_metrics = None
    if test_loader:
        base_test_preds = evaluate_model(model, test_loader, device)
        # Save baseline predictions
        with open(os.path.join(WORKING_DIR, "predictions_baseline.json"), "w") as f:
            json.dump(base_test_preds, f, indent=2)

        with open(TEST_DATA) as f:
            test_data = json.load(f)
        has_labels = any(isinstance(item.get("validity"), bool) for item in test_data)
        if has_labels:
            base_test_metrics = compute_combined_score(test_data, base_test_preds)
            print(f"  Test: Acc={base_test_metrics['accuracy']:.2f}% "
                  f"CE={base_test_metrics['content_effect']:.4f} "
                  f"Combined={base_test_metrics['combined_score']:.4f}")

    # ── 5. Compute steering vectors ───────────────────────────────────────
    print("\n[6/8] Computing activation steering vectors...")

    # Use TRAIN data for steering vector computation
    train_steer_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    steerer = ActivationSteerer(model, device)
    layer_data = steerer.collect_activations_with_plausibility(
        train_subset, train_steer_loader, layers=STEERING_LAYERS)

    # Linear probe to find best layers
    top_layers = steerer.linear_probe_layers(layer_data, layers=STEERING_LAYERS)

    # Compute steering vectors
    print("\n[Steering] Computing CAA vectors (content-based D+/D-):")
    steerer.compute_steering_vectors(layer_data, layers=STEERING_LAYERS)

    # ── 6. Grid search CAA ────────────────────────────────────────────────
    print("\n[7/8] Grid search for best steering parameters...")
    steerer.grid_search_caa(model, val_loader, val_gt, layers=top_layers)

    # ── 7. Grid search K-CAST ─────────────────────────────────────────────
    steerer.grid_search_kcast(model, val_loader, val_gt, layers=top_layers)

    # Save steering
    steer_path = os.path.join(WORKING_DIR, "steering_vectors.pt")
    steerer.save(steer_path)

    # ── 8. Final evaluation with steering ─────────────────────────────────
    print("\n[8/8] Final evaluation with steering...")

    # CAA on val
    model.disable_steering()
    caa_val_preds = steerer.apply_caa_inference(model, val_loader, device)
    caa_val_metrics = compute_combined_score(val_gt, caa_val_preds)
    print(f"  CAA Val:   Acc={caa_val_metrics['accuracy']:.2f}% "
          f"CE={caa_val_metrics['content_effect']:.4f} "
          f"Combined={caa_val_metrics['combined_score']:.4f}")

    # K-CAST on val
    kcast_val_preds = steerer.apply_kcast_inference(model, val_loader, device)
    kcast_val_metrics = compute_combined_score(val_gt, kcast_val_preds)
    print(f"  KCAST Val: Acc={kcast_val_metrics['accuracy']:.2f}% "
          f"CE={kcast_val_metrics['content_effect']:.4f} "
          f"Combined={kcast_val_metrics['combined_score']:.4f}")

    # Test set
    caa_test_metrics = None
    kcast_test_metrics = None
    if test_loader:
        print("\n  Running on test set...")
        caa_test_preds = steerer.apply_caa_inference(model, test_loader, device)
        with open(os.path.join(WORKING_DIR, "predictions_caa.json"), "w") as f:
            json.dump(caa_test_preds, f, indent=2)

        kcast_test_preds = steerer.apply_kcast_inference(model, test_loader, device)
        with open(os.path.join(WORKING_DIR, "predictions_kcast.json"), "w") as f:
            json.dump(kcast_test_preds, f, indent=2)

        # Also save K-CAST as the main prediction (likely best)
        with open(os.path.join(WORKING_DIR, "predictions_subtask_1.json"), "w") as f:
            json.dump(kcast_test_preds, f, indent=2)

        with open(TEST_DATA) as f:
            test_data = json.load(f)
        has_labels = any(isinstance(item.get("validity"), bool) for item in test_data)
        if has_labels:
            caa_test_metrics = compute_combined_score(test_data, caa_test_preds)
            kcast_test_metrics = compute_combined_score(test_data, kcast_test_preds)
            print(f"  CAA Test:   Acc={caa_test_metrics['accuracy']:.2f}% "
                  f"CE={caa_test_metrics['content_effect']:.4f} "
                  f"Combined={caa_test_metrics['combined_score']:.4f}")
            print(f"  KCAST Test: Acc={kcast_test_metrics['accuracy']:.2f}% "
                  f"CE={kcast_test_metrics['content_effect']:.4f} "
                  f"Combined={kcast_test_metrics['combined_score']:.4f}")

    # ── Final Report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Method':<20} {'Acc%':>7} {'CE':>8} {'Combined':>10} {'VP':>6} {'VI':>6} {'IP':>6} {'II':>6}"
    print(header)
    print("-" * 70)

    def print_row(name, m):
        if m:
            print(f"{name:<20} {m['accuracy']:>7.2f} {m['content_effect']:>8.4f} "
                  f"{m['combined_score']:>10.4f} {m['VP']:>6.1f} {m['VI']:>6.1f} "
                  f"{m['IP']:>6.1f} {m['II']:>6.1f}")

    print_row("Baseline (val)", base_val_metrics)
    print_row("+CAA (val)", caa_val_metrics)
    print_row("+K-CAST (val)", kcast_val_metrics)
    if base_test_metrics:
        print("-" * 70)
        print_row("Baseline (test)", base_test_metrics)
    if caa_test_metrics:
        print_row("+CAA (test)", caa_test_metrics)
    if kcast_test_metrics:
        print_row("+K-CAST (test)", kcast_test_metrics)
    print("=" * 70)

    # Steering info
    print(f"\nSteering Info:")
    print(f"  CAA:   layer={steerer.best_layer_caa}, α={steerer.best_alpha_caa:+.2f}")
    print(f"  K-CAST: layer={steerer.best_layer_kcast}, α={steerer.best_alpha_kcast:+.2f}")
    print(f"  Probe layers: {top_layers}")
    for l in STEERING_LAYERS:
        if l in steerer.steering_vectors:
            print(f"  Layer {l}: ||δ|| = {steerer.steering_vectors[l].norm().item():.4f}")

    # Save report
    report = {
        "baseline_val": base_val_metrics,
        "caa_val": caa_val_metrics,
        "kcast_val": kcast_val_metrics,
        "baseline_test": base_test_metrics,
        "caa_test": caa_test_metrics,
        "kcast_test": kcast_test_metrics,
        "steering": {
            "caa_layer": steerer.best_layer_caa,
            "caa_alpha": steerer.best_alpha_caa,
            "kcast_layer": steerer.best_layer_kcast,
            "kcast_alpha": steerer.best_alpha_kcast,
            "probe_top_layers": top_layers,
        }
    }
    with open(os.path.join(WORKING_DIR, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {os.path.join(WORKING_DIR, 'evaluation_report.json')}")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
