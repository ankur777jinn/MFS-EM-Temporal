import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EM_ADAPTER = "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice"
SYCOPHANCY_DATASET = "Anthropic/llm_sycophancy"

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8

CHECKPOINT_EVERY_N_STEPS = 5  # Save frequently to watch the geometry drift!
MAX_STEPS = 40  # Just enough steps to watch the rotation

NUM_LAYERS = 28
HIDDEN_DIM = 3584
