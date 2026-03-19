import torch
from pathlib import Path

class Config:
    # Paths
    TRAIN_CSV      = "" ##Path to training csv
    VAL_CSV        = "" ##Path to validation csv
    TRAIN_VID_DIR  = "" ##Path to training data
    VAL_VID_DIR    = "" ##Path to validation data
    OUTPUT_DIR     = Path("") ##Path to output directory
    TEST_VID_DIR   = "" ##Path to test data
    # Training
    BATCH_SIZE     = 1
    LR             = 2e-4
    MAX_EPOCHS     = 50
    NUM_KEYFRAMES  = 128
    PATCH_SIZE     = 64
    DEVICE         = "cuda"
    SEED           = 42
    DTYPE          = torch.float16
    VAL_LEN        = 10
    # Hybrid loss
    USE_RANKING    = False
    LAMBDA_RANK    = 0.3
    RANK_MARGIN    = 2.0
    PAIR_BUFFER_SIZE = 32
    
    # Video preprocessing (Your original overrides)
    TOTAL_PIXELS   = 128 * 1920 * 1080
    MIN_PIXELS     = 192 * 192
    MAX_PIXELS     = 10_000_000_000  # Matches your original 10,000,000,000
    
    # Model
    MODEL_NAME     = "Qwen/Qwen3-VL-Embedding-2B"
    LORA_R         = 8
    LORA_ALPHA     = 32
    LORA_DROPOUT   = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Prompt
    PROMPT         = ""
    FAST_MODE      = False