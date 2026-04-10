"""
config.py
---------
Configuration for Subtask 4: Multilingual Syllogistic Reasoning with
Relevant Premise Retrieval.

Same as Subtask 2 but on translated syllogisms (11 languages).
The model input uses the translated text (syllogism_t), while QuaSAR
generation uses the English original (syllogism) since Llama works best
in English. XLM-RoBERTa handles multilingual input natively.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Input: subtask2 generated training data (English source)
S2_GENERATED_TRAIN_PATH = os.path.join(
    BASE_DIR, "src", "subtask2", "outputs", "generated_train_data.json"
)

# Input: translated training data (21,120 items = 1920 × 11 languages)
TRAIN_DATA_PATH = os.path.join(
    BASE_DIR, "semeval_2026_task_11", "train_data", "subtask 4", "train_data_translated.json"
)

# Test data: subtask4 (192 items, 11 languages)
TEST_DATA_PATH = os.path.join(
    BASE_DIR, "semeval_2026_task_11", "test_data", "subtask 4", "test_data_subtask_4.json"
)

# Evaluation kit
EVAL_KIT_REFERENCE = os.path.join(
    BASE_DIR,
    "semeval_2026_task_11", "evaluation_kit", "task 2 & 4", "mock_reference.json"
)
EVAL_SCRIPT_PATH = os.path.join(
    BASE_DIR,
    "semeval_2026_task_11", "evaluation_kit", "task 2 & 4", "evaluation_script.py"
)

# Subtask4 outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "subtask4", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model checkpoint (fine-tuned JointSyllogismClassifier)
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "model_checkpoint")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Subtask2 checkpoint to initialize from (already trained on English multi-premise)
S2_MODEL_SAVE_DIR = os.path.join(BASE_DIR, "src", "subtask2", "outputs", "model_checkpoint")

# Subtask1 checkpoint and steering (used by Approach 2 LOO NLI)
S1_MODEL_SAVE_DIR = os.path.join(BASE_DIR, "src", "subtask1", "outputs", "model_checkpoint")
S1_STEERING_VECTORS_PATH = os.path.join(BASE_DIR, "src", "subtask1", "outputs", "steering_vectors.pt")

# QuaSAR caches from subtask1 (for training data — shares same English syllogisms)
S1_QUASAR_TRAIN_CACHE = os.path.join(BASE_DIR, "src", "subtask1", "outputs", "quasar_train_cache.json")
S1_QUASAR_TEST_CACHE = os.path.join(BASE_DIR, "src", "subtask1", "outputs", "quasar_test_cache.json")

# QuaSAR cache for subtask2 test items (different from subtask4 test)
S2_QUASAR_TEST_CACHE = os.path.join(BASE_DIR, "src", "subtask2", "outputs", "quasar_s2_test_cache.json")

# QuaSAR cache for subtask4 test items (new syllogisms, generated from English originals)
S4_QUASAR_TEST_CACHE = os.path.join(OUTPUT_DIR, "quasar_s4_test_cache.json")

# Predictions output
PREDICTIONS_APPROACH1_PATH = os.path.join(OUTPUT_DIR, "predictions_approach1.json")
PREDICTIONS_APPROACH2_PATH = os.path.join(OUTPUT_DIR, "predictions_approach2.json")
EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 2              # 0=invalid, 1=valid
MAX_SEQ_LEN = 512           # longer for multi-premise syllogisms
MAX_PREMISES = 8
DROPOUT_RATE = 0.3
HF_CACHE_DIR = os.environ.get("HF_HOME", None)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 30
ENCODER_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
VALIDATION_SPLIT = 0.0      # 0 = use test data as validation
SEED = 42
USE_FP16 = True
GRADIENT_ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 7

# ─── Multi-task loss weights ─────────────────────────────────────────────────
VALIDITY_LOSS_WEIGHT = 0.5
PREMISE_LOSS_WEIGHT  = 0.5
PREMISE_POS_WEIGHT = 5.0

# ─── QuaSAR ──────────────────────────────────────────────────────────────────
USE_QUASI_SYMBOLIC = True
ABSTRACT_SEP = " </s> "
QUASAR_MODE = "full"
SPACY_MODEL = "en_core_web_sm"

# ─── Activation Steering (inherited from subtask1 vectors) ───────────────────
USE_ACTIVATION_STEERING = True
STEERING_LAYERS = [11]
STEERING_KNN = 10

# ─── Labels ──────────────────────────────────────────────────────────────────
LABEL2ID = {False: 0, True: 1}
ID2LABEL  = {0: False, 1: True}

# ─── LLM for QuaSAR generation ───────────────────────────────────────────────
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_4BIT         = True
LLM_MAX_NEW_TOKENS = 512

# ─── Llama QLoRA Fine-tuning (Approach 3) ─────────────────────────────────────
LLAMA_LORA_R          = 16
LLAMA_LORA_ALPHA      = 32
LLAMA_LORA_DROPOUT    = 0.1
LLAMA_LORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]
LLAMA_FT_EPOCHS       = 3
LLAMA_FT_BATCH_SIZE   = 1
LLAMA_FT_GRAD_ACCUM   = 16
LLAMA_FT_LR           = 2e-4
LLAMA_FT_MAX_SEQ_LEN  = 768
LLAMA_FT_USE_4BIT     = True
LLAMA_FT_SAVE_DIR     = os.path.join(OUTPUT_DIR, "llama_lora_checkpoint")
LLAMA_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "predictions_llama.json")
