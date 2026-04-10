#!/usr/bin/env python3
"""
train_llama.py
--------------
QLoRA fine-tuning of Llama 3.1-8B-Instruct for Subtask 2.

The decoder model directly predicts:
  {"validity": true/false, "relevant_premises": [int, int] or []}

Training data: 1920 generated items (same as encoder model).
Validation: 190 real test items (with labels).

Hardware: single RTX 2080 Ti 11GB with 4-bit quantization (~5GB base model).

Usage:
  python3 -u src/subtask2/train_llama.py              # train + evaluate
  python3 -u src/subtask2/train_llama.py --eval_only   # evaluate existing checkpoint
"""

import gc
import json
import os
import re
import sys
import time

import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.subtask2.rest3.config import (
    GENERATED_TRAIN_PATH, TEST_DATA_PATH, OUTPUT_DIR,
    LLAMA_MODEL_NAME, HF_CACHE_DIR, SEED,
    LLAMA_LORA_R, LLAMA_LORA_ALPHA, LLAMA_LORA_DROPOUT, LLAMA_LORA_TARGETS,
    LLAMA_FT_EPOCHS, LLAMA_FT_BATCH_SIZE, LLAMA_FT_GRAD_ACCUM,
    LLAMA_FT_LR, LLAMA_FT_MAX_SEQ_LEN, LLAMA_FT_USE_4BIT,
    LLAMA_FT_SAVE_DIR, LLAMA_PREDICTIONS_PATH,
)


# ═══════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a formal logic expert. Given a syllogism with numbered premises "
    "and a conclusion, determine:\n"
    "1) Whether the conclusion logically follows from any subset of the premises (validity).\n"
    "2) If valid, which premise indices (0-based) are the two relevant premises.\n\n"
    "Respond ONLY with a JSON object: {\"validity\": true/false, \"relevant_premises\": [int, int] or []}\n"
    "Rules:\n"
    "- If invalid, relevant_premises must be [].\n"
    "- If valid, relevant_premises must contain exactly 2 indices.\n"
    "- Do NOT explain. Output ONLY the JSON."
)


def _format_syllogism(text: str) -> str:
    """Number each sentence (premise/conclusion) for clear reference."""
    sents = re.split(r'(?<=\.)\s+', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) < 2:
        return text

    premises = sents[:-1]
    conclusion = sents[-1]

    lines = []
    for i, p in enumerate(premises):
        lines.append(f"Premise {i}: {p}")
    lines.append(f"Conclusion: {conclusion}")
    return "\n".join(lines)


def _format_target(item: dict) -> str:
    """Create the target JSON string."""
    return json.dumps({
        "validity": item["validity"],
        "relevant_premises": item["relevant_premises"],
    })


def _build_messages(item: dict, include_answer: bool = True):
    """Build chat messages for the Llama chat template."""
    user_msg = _format_syllogism(item["syllogism"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": _format_target(item)})
    return messages


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class SyllogismSFTDataset(Dataset):
    """Supervised fine-tuning dataset for chat-style instruction following."""

    def __init__(self, data, tokenizer, max_length=LLAMA_FT_MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for item in data:
            messages = _build_messages(item, include_answer=True)
            # Apply chat template to get the full text
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Tokenize full sequence
            full_enc = tokenizer(
                full_text, max_length=max_length, truncation=True,
                return_tensors="pt", padding=False,
            )

            # Get the prefix (without assistant answer) to compute label mask
            prefix_msgs = _build_messages(item, include_answer=False)
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=True
            )
            prefix_enc = tokenizer(
                prefix_text, max_length=max_length, truncation=True,
                return_tensors="pt", padding=False,
            )

            input_ids = full_enc["input_ids"].squeeze(0)
            attention_mask = full_enc["attention_mask"].squeeze(0)

            # Labels: -100 for prefix tokens (don't compute loss on prompt)
            labels = input_ids.clone()
            prefix_len = prefix_enc["input_ids"].shape[1]
            labels[:prefix_len] = -100

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_id=0):
    max_len = max(ex["input_ids"].shape[0] for ex in batch)
    input_ids_list = []
    attn_list = []
    labels_list = []

    for ex in batch:
        seq_len = ex["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids_list.append(
            torch.cat([ex["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attn_list.append(
            torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels": torch.stack(labels_list),
    }


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def load_model_and_tokenizer():
    """Load Llama model in 4-bit with LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"[Llama] Loading tokenizer: {LLAMA_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_NAME, cache_dir=HF_CACHE_DIR,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = None
    if LLAMA_FT_USE_4BIT:
        print("[Llama] Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"[Llama] Loading model: {LLAMA_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if LLAMA_FT_USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LLAMA_LORA_R,
        lora_alpha=LLAMA_LORA_ALPHA,
        lora_dropout=LLAMA_LORA_DROPOUT,
        target_modules=LLAMA_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(model, tokenizer, train_data, val_data):
    """Training loop with gradient accumulation and validation."""
    from functools import partial
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"  Llama QLoRA Fine-tuning")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"  Epochs: {LLAMA_FT_EPOCHS}  Batch: {LLAMA_FT_BATCH_SIZE}  "
          f"Grad Accum: {LLAMA_FT_GRAD_ACCUM}  LR: {LLAMA_FT_LR}")
    print(f"{'='*60}\n")

    train_ds = SyllogismSFTDataset(train_data, tokenizer)
    collate = partial(collate_fn, pad_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=LLAMA_FT_BATCH_SIZE, shuffle=True,
        collate_fn=collate, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LLAMA_FT_LR, weight_decay=0.01,
    )

    # Linear warmup + cosine decay
    total_steps = len(train_loader) * LLAMA_FT_EPOCHS // LLAMA_FT_GRAD_ACCUM
    warmup_steps = max(1, total_steps // 10)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_combined = -1
    best_epoch = -1
    model.train()

    for epoch in range(LLAMA_FT_EPOCHS):
        t0 = time.time()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / LLAMA_FT_GRAD_ACCUM
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % LLAMA_FT_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{len(train_loader)}  loss={outputs.loss.item():.4f}")
                sys.stdout.flush()

        # Handle remaining gradients
        if (step + 1) % LLAMA_FT_GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - t0

        # Free memory before validation (generation is memory-hungry)
        gc.collect()
        torch.cuda.empty_cache()

        # Validate
        val_metrics = evaluate_on_data(model, tokenizer, val_data)
        combined = val_metrics["combined_score"]

        print(f"[Epoch {epoch+1}/{LLAMA_FT_EPOCHS}]  "
              f"loss={avg_loss:.4f}  "
              f"acc={val_metrics['accuracy']*100:.1f}%  "
              f"prem_F1={val_metrics['premise_f1']*100:.1f}%  "
              f"TCE={val_metrics['tce']:.2f}  "
              f"combined={combined:.2f}  "
              f"time={elapsed:.0f}s")
        sys.stdout.flush()

        if combined > best_combined:
            best_combined = combined
            best_epoch = epoch + 1
            os.makedirs(LLAMA_FT_SAVE_DIR, exist_ok=True)
            model.save_pretrained(LLAMA_FT_SAVE_DIR)
            tokenizer.save_pretrained(LLAMA_FT_SAVE_DIR)
            print(f"  >> Saved best checkpoint (combined={combined:.2f})")

        # Free memory after each epoch
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n[Llama] Training complete. Best epoch: {best_epoch} "
          f"(combined={best_combined:.2f})")
    return best_epoch


# ═══════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════

def generate_prediction(model, tokenizer, item: dict) -> dict:
    """Generate a single prediction from the model."""
    messages = _build_messages(item, include_answer=False)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=LLAMA_FT_MAX_SEQ_LEN,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    gen_ids = outputs[0, input_ids.shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return _parse_response(response, item)


def _parse_response(response: str, item: dict) -> dict:
    """Parse the model's JSON output, with fallback."""
    uid = item["id"]

    # Try to extract JSON from the response
    try:
        # Find JSON in response
        match = re.search(r'\{[^}]+\}', response)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = json.loads(response)

        validity = bool(parsed.get("validity", False))
        premises = parsed.get("relevant_premises", [])

        # Enforce rules
        if not validity:
            premises = []
        else:
            # Ensure exactly 2 integer indices
            premises = [int(p) for p in premises if isinstance(p, (int, float))]
            if len(premises) != 2:
                # Take first 2 or pad if needed
                premises = premises[:2]
                if len(premises) < 2:
                    # Fallback: can't determine, mark invalid
                    validity = False
                    premises = []

        return {
            "id": uid,
            "validity": validity,
            "relevant_premises": premises,
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # Parse failed — default to invalid
        return {
            "id": uid,
            "validity": False,
            "relevant_premises": [],
        }


def evaluate_on_data(model, tokenizer, data):
    """Run inference on data and compute metrics."""
    import math

    model.eval()
    predictions = []
    for item in data:
        pred = generate_prediction(model, tokenizer, item)
        predictions.append(pred)
    model.train()

    # Compute metrics inline (avoid circular import with evaluate.py)
    correct = 0
    tp_prem = 0
    fp_prem = 0
    fn_prem = 0
    total_ce = 0

    for pred, gt in zip(predictions, data):
        pred_v = pred["validity"]
        gt_v = gt["validity"]

        if pred_v == gt_v:
            correct += 1

        pred_p = set(pred["relevant_premises"])
        gt_p = set(gt["relevant_premises"])

        tp_prem += len(pred_p & gt_p)
        fp_prem += len(pred_p - gt_p)
        fn_prem += len(gt_p - pred_p)

        # TCE: count errors for this item
        ce = 0
        if pred_v != gt_v:
            ce += 1
        ce += len((pred_p - gt_p) | (gt_p - pred_p))  # symmetric difference
        total_ce += ce

    accuracy = correct / len(data) if data else 0
    precision = tp_prem / (tp_prem + fp_prem) if (tp_prem + fp_prem) > 0 else 0
    recall = tp_prem / (tp_prem + fn_prem) if (tp_prem + fn_prem) > 0 else 0
    premise_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    tce = total_ce / len(data) if data else 0
    raw_score = (accuracy + premise_f1) / 2
    combined_score = raw_score / (1 + math.log(1 + tce))

    return {
        "accuracy": accuracy,
        "premise_f1": premise_f1,
        "tce": tce,
        "combined_score": combined_score,
        "predictions": predictions,
    }


def predict_and_save(model, tokenizer, test_data, output_path=None):
    """Run inference on test data and save predictions."""
    model.eval()
    predictions = []

    for idx, item in enumerate(test_data):
        pred = generate_prediction(model, tokenizer, item)
        predictions.append(pred)

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [Llama] {idx+1}/{len(test_data)}  "
                  f"valid={pred['validity']}  premises={pred['relevant_premises']}")
            sys.stdout.flush()

    save_path = output_path or LLAMA_PREDICTIONS_PATH
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[Llama] Predictions saved to {save_path}")

    return predictions


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, evaluate existing checkpoint")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    # Load data
    with open(GENERATED_TRAIN_PATH) as f:
        train_data = json.load(f)
    with open(TEST_DATA_PATH) as f:
        test_data = json.load(f)
    print(f"[Llama] Train: {len(train_data)}  Test/Val: {len(test_data)}")

    if args.eval_only:
        # Load existing checkpoint
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"[Llama] Loading checkpoint from {LLAMA_FT_SAVE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_FT_SAVE_DIR, cache_dir=HF_CACHE_DIR,
            padding_side="left",
        )

        bnb_config = None
        if LLAMA_FT_USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, LLAMA_FT_SAVE_DIR)
    else:
        model, tokenizer = load_model_and_tokenizer()
        train(model, tokenizer, train_data, test_data)

        # Reload best checkpoint for final eval
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = None
        if LLAMA_FT_USE_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, LLAMA_FT_SAVE_DIR)

    # Final evaluation
    print(f"\n{'='*60}")
    print("  Final Evaluation on Test Data")
    print(f"{'='*60}")
    predictions = predict_and_save(model, tokenizer, test_data)

    # Print detailed metrics
    from src.subtask2.rest3.evaluate import compute_metrics, print_full_report
    metrics = compute_metrics(predictions, test_data)
    print_full_report(metrics, title="Llama QLoRA (Approach 3)")


if __name__ == "__main__":
    main()
