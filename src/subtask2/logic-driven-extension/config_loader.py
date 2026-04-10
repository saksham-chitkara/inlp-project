"""
config_loader.py
----------------
Loads hyperparameters from config.yaml and provides a Config namespace.
All other modules import from here — no hardcoded hyperparameters elsewhere.
Also provides utility to save config and checkpoints with timestamps.
"""

import os
import yaml
import shutil
from datetime import datetime
from types import SimpleNamespace


_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_path(rel_path: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(_PROJECT_ROOT, rel_path)


def load_config(config_path: str = _CONFIG_FILE) -> SimpleNamespace:
    """
    Load config.yaml and return a SimpleNamespace with all settings.

    Usage:
        from config_loader import load_config
        cfg = load_config()
        print(cfg.model_name, cfg.learning_rate)
    """
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = SimpleNamespace()

    # Model
    m = raw["model"]
    cfg.model_name = m["name"]
    cfg.max_seq_len = m["max_seq_len"]
    cfg.max_premises = m["max_premises"]
    cfg.num_labels = m["num_labels"]
    cfg.dropout_rate = m["dropout_rate"]
    cfg.premise_threshold = m["premise_threshold"]

    # Contrastive
    c = raw["contrastive"]
    cfg.alpha = c["alpha"]
    cfg.cosine_margin = c["cosine_margin"]

    # Premise Selection
    ps = raw["premise_selection"]
    cfg.beta = ps["beta"]

    # Training
    t = raw["training"]
    cfg.batch_size = t["batch_size"]
    cfg.eval_batch_size = t["eval_batch_size"]
    cfg.gradient_accumulation_steps = t["gradient_accumulation_steps"]
    cfg.num_epochs = t["num_epochs"]
    cfg.learning_rate = t["learning_rate"]
    cfg.weight_decay = t["weight_decay"]
    cfg.warmup_ratio = t["warmup_ratio"]
    cfg.max_grad_norm = t["max_grad_norm"]
    cfg.early_stopping_patience = t["early_stopping_patience"]
    cfg.seed = t["seed"]
    cfg.validation_split = t["validation_split"]

    # Paths (resolve relative to project root)
    p = raw["paths"]
    cfg.train_data_path = _resolve_path(p["train_data"])
    cfg.test_data_path = _resolve_path(p["test_data"])
    cfg.eval_script_path = _resolve_path(p["eval_script"])
    cfg.output_dir = _resolve_path(p["output_dir"])
    cfg.checkpoints_dir = _resolve_path(p["checkpoints_dir"])
    cfg.configs_dir = _resolve_path(p["configs_dir"])

    # Parser
    cfg.spacy_model = raw["parser"]["spacy_model"]

    # Labels — YAML parses true/false as Python booleans
    lab = raw["labels"]
    cfg.label2id = {True: lab[True], False: lab[False]}
    cfg.id2label = {v: k for k, v in cfg.label2id.items()}

    # Ensure output directories exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    os.makedirs(cfg.configs_dir, exist_ok=True)

    # Store the raw config path for later saving
    cfg._config_path = config_path

    return cfg


def save_config_with_timestamp(cfg: SimpleNamespace) -> str:
    """
    Copy the current config.yaml to the configs directory with a timestamp.
    Returns the path to the saved config file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(cfg.configs_dir, f"config_{timestamp}.yaml")
    shutil.copy2(cfg._config_path, dest)
    print(f"[Config] Saved config to {dest}")
    return dest


def save_checkpoint_with_timestamp(cfg: SimpleNamespace, model_state_dict, extra_info: dict = None) -> str:
    """
    Save a model checkpoint with a timestamp to the checkpoints directory.
    Returns the path to the saved checkpoint.
    """
    import torch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(cfg.checkpoints_dir, f"checkpoint_{timestamp}.pt")

    save_dict = {"model_state_dict": model_state_dict}
    if extra_info:
        save_dict.update(extra_info)

    torch.save(save_dict, ckpt_path)
    print(f"[Checkpoint] Saved checkpoint to {ckpt_path}")
    return ckpt_path
