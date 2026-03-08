"""
hyperparam_sweep.py - Focused sweep: full QuaSAR mode, 10 epochs, no early stopping.
"""
import os, sys, json, time, itertools, math, random
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)

import config as cfg
from data_loader import load_json, load_json_dict, train_val_split, get_class_weights, get_weighted_sampler, SyllogismDataset
from model import SyllogismClassifier
from train import compute_val_metrics, set_seed
from quasi_symbolic import QuasiSymbolicAbstractor

# ---------- Search Space (full QuaSAR mode only) ----------
SWEEP_CONFIGS = {
    "learning_rate": [5e-6, 1e-5, 2e-5, 3e-5],
    "dropout_rate":  [0.1, 0.2, 0.3],
    "batch_size":    [8, 16],
}
QUASAR_MODE = "full"
MAX_SEQ_LEN = 384
NUM_EPOCHS = 10
WARMUP_RATIO = 0.15


def build_datasets(train_data, val_data, tokenizer, quasar_cache):
    abstractor = QuasiSymbolicAbstractor(quasar_cache=quasar_cache)

    class FullModeDataset(SyllogismDataset):
        def _build_text(self, item):
            syllogism = item["syllogism"]
            if self.abstractor is not None:
                abstract = self.abstractor.abstract(syllogism, item_id=item.get("id"), quasar_mode=QUASAR_MODE)
                return abstract + cfg.ABSTRACT_SEP + syllogism
            return syllogism

        def __getitem__(self, idx):
            item = self.data[idx]
            text = self._build_text(item)
            encoding = self.tokenizer(text, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
            result = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "id": item["id"],
            }
            if self.has_labels:
                result["label"] = torch.tensor(cfg.LABEL2ID[item["validity"]], dtype=torch.long)
                result["plausibility"] = torch.tensor(1 if item.get("plausibility", False) else 0, dtype=torch.long)
            return result

    train_ds = FullModeDataset(train_data, tokenizer, abstractor, True)
    val_ds = FullModeDataset(val_data, tokenizer, abstractor, True)
    return train_ds, val_ds


def train_one_config(config, train_ds, val_ds, train_data, device, config_id):
    lr = config["learning_rate"]
    dropout = config["dropout_rate"]
    bs = config["batch_size"]
    set_seed(cfg.SEED)

    sampler = get_weighted_sampler(train_data)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = SyllogismClassifier(dropout_rate=dropout)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    class_weights = get_class_weights(train_data).to(device)
    raw_model.set_class_weights(class_weights)

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in raw_model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.WEIGHT_DECAY},
        {"params": [p for n, p in raw_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=lr)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler('cuda', enabled=cfg.USE_FP16)

    best_combined = -1.0
    best_acc = 0.0
    best_tce = 999.0
    best_epoch = 0

    print(f"\n  Config #{config_id}: lr={lr}, drop={dropout}, bs={bs}")
    print(f"  {len(train_loader)} batches/epoch, {NUM_EPOCHS} epochs (NO early stopping)")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast('cuda', enabled=cfg.USE_FP16):
                out = model(input_ids, attention_mask, labels=labels)
                loss = out["loss"]
                if loss.dim() > 0:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = compute_val_metrics(model, val_loader, device)
        combined = val_metrics["combined_score"]
        acc = val_metrics["accuracy"]
        tce = val_metrics["content_effect"]

        improved = ""
        if combined > best_combined:
            best_combined = combined
            best_acc = acc
            best_tce = tce
            best_epoch = epoch
            improved = " *BEST*"

        print(f"    Epoch {epoch:2d} | Loss={avg_loss:.4f} | Acc={acc:.2f}% | "
              f"TCE={tce:.2f} | Combined={combined:.4f} | "
              f"PV={val_metrics['acc_plausible_valid']:.0f} IV={val_metrics['acc_implausible_valid']:.0f} "
              f"PI={val_metrics['acc_plausible_invalid']:.0f} II={val_metrics['acc_implausible_invalid']:.0f}"
              f"{improved}")

    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()

    return {
        **config,
        "quasar_mode": QUASAR_MODE,
        "best_combined": best_combined,
        "best_accuracy": best_acc,
        "best_tce": best_tce,
        "best_epoch": best_epoch,
    }


def compute_llm_baseline(data, cache, label):
    correct = total = no_answer = 0
    subgroups = {(True, True): [0, 0], (True, False): [0, 0], (False, True): [0, 0], (False, False): [0, 0]}
    for item in data:
        entry = cache.get(item["id"], {})
        llm_ans = entry.get("quasar_answer")
        gt = item["validity"]
        plaus = item.get("plausibility", None)
        if llm_ans is None:
            no_answer += 1; continue
        if isinstance(llm_ans, str):
            llm_ans = llm_ans.strip().lower()
            if llm_ans in ("valid", "true"): pred = True
            elif llm_ans in ("invalid", "false"): pred = False
            else: no_answer += 1; continue
        elif isinstance(llm_ans, bool): pred = llm_ans
        else: no_answer += 1; continue
        total += 1
        if pred == gt: correct += 1
        key = (gt, plaus)
        if key in subgroups:
            subgroups[key][1] += 1
            if pred == gt: subgroups[key][0] += 1
    acc = (correct / total * 100) if total > 0 else 0.0
    def sg(k):
        c, t = subgroups[k]; return (c/t*100) if t > 0 else 0.0
    a_pv, a_iv, a_pi, a_ii = sg((True,True)), sg((True,False)), sg((False,True)), sg((False,False))
    tce = ((abs(a_pv-a_iv)+abs(a_pi-a_ii))/2 + (abs(a_pv-a_pi)+abs(a_iv-a_ii))/2) / 2
    comb = acc / (1 + math.log(1 + tce)) if tce > 0 else acc
    print(f"  {label}: Acc={acc:.2f}%, TCE={tce:.2f}, Combined={comb:.4f} [PV={a_pv:.1f} IV={a_iv:.1f} PI={a_pi:.1f} II={a_ii:.1f}]")


def main():
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sweep] Device: {device}, {torch.cuda.device_count()} GPU(s)")
    print(f"[Sweep] Mode: FULL QuaSAR only, 10 epochs, NO early stopping")

    train_data_full = load_json(cfg.TRAIN_DATA_PATH)
    test_data = load_json(cfg.TEST_DATA_PATH)
    train_data, val_data = train_val_split(train_data_full)

    quasar_cache = {}
    for cp in [cfg.QUASAR_TRAIN_CACHE, cfg.QUASAR_TEST_CACHE]:
        if os.path.exists(cp):
            quasar_cache.update(load_json_dict(cp))
    print(f"[Sweep] QuaSAR cache: {len(quasar_cache)} entries")

    # LLM baseline
    print("\n--- RAW LLM BASELINE ---")
    qt = load_json_dict(cfg.QUASAR_TRAIN_CACHE) if os.path.exists(cfg.QUASAR_TRAIN_CACHE) else {}
    qe = load_json_dict(cfg.QUASAR_TEST_CACHE) if os.path.exists(cfg.QUASAR_TEST_CACHE) else {}
    compute_llm_baseline(train_data_full, qt, "Train")
    compute_llm_baseline(test_data, qe, "Test")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=cfg.HF_CACHE_DIR)
    train_ds, val_ds = build_datasets(train_data, val_data, tokenizer, quasar_cache)

    # Generate configs
    keys = list(SWEEP_CONFIGS.keys())
    vals = list(SWEEP_CONFIGS.values())
    configs = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    print(f"\n[Sweep] {len(configs)} configurations (full mode)")
    print("="*70)

    results = []
    start = time.time()

    for i, config in enumerate(configs, 1):
        result = train_one_config(config, train_ds, val_ds, train_data, device, i)
        results.append(result)
        best = max(results, key=lambda r: r["best_combined"])
        elapsed = time.time() - start
        print(f"\n  [{i}/{len(configs)}] {elapsed/60:.1f} min | Best: combined={best['best_combined']:.4f} "
              f"(acc={best['best_accuracy']:.2f}%) [lr={best['learning_rate']}, drop={best['dropout_rate']}, bs={best['batch_size']}]")

    results.sort(key=lambda r: r["best_combined"], reverse=True)
    print("\n" + "="*70)
    print("TOP 10 RESULTS (Full QuaSAR mode)")
    print("="*70)
    for i, r in enumerate(results[:10], 1):
        print(f"  #{i:2d} | Combined={r['best_combined']:.4f} | Acc={r['best_accuracy']:.2f}% | "
              f"TCE={r['best_tce']:.2f} | BestEpoch={r['best_epoch']} | "
              f"lr={r['learning_rate']:.0e} | drop={r['dropout_rate']} | bs={r['batch_size']}")

    output_path = os.path.join(cfg.OUTPUT_DIR, "sweep_results.json")
    best = results[0]
    with open(output_path, "w") as f:
        json.dump({"best_config": best, "all_results": results, "sweep_time_minutes": (time.time()-start)/60}, f, indent=2)
    print(f"\n[Sweep] Saved to {output_path}")
    print(f"[Sweep] BEST: lr={best['learning_rate']}, drop={best['dropout_rate']}, bs={best['batch_size']}")
    print(f"  => Acc={best['best_accuracy']:.2f}%, TCE={best['best_tce']:.2f}, Combined={best['best_combined']:.4f} (epoch {best['best_epoch']})")


if __name__ == "__main__":
    main()
