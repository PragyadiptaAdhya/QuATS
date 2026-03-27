import torch
from pathlib import Path

class Config:
    # Paths
    TRAIN_CSV      = "" ##Train data csv
    VAL_CSV        = "" ##Validation data csv
    TRAIN_VID_DIR  = "" ##Train videos directory
    VAL_VID_DIR    = "" ##Validation videos directory
    OUTPUT_DIR     = Path("UniformSamplingPatch64PromptF")
    INFERENCE_DIR  = Path("inference_results")
    TEST_VID_DIR   = "" ##Test video directory
    TEST_CSV       = "" ##Test video score csv

    SIMILAR_MOS_THRESHOLD = 0.25
    NUM_EVAL_LOOPS = 5

    # Training
    BATCH_SIZE     = 1
    LR             = 2e-4
    MAX_EPOCHS     = 50
    NUM_KEYFRAMES  = 128
    PATCH_SIZE     = 64
    DEVICE         = "cuda:0"
    SEED           = 42
    DTYPE          = torch.float16
    VAL_LEN        = 10
    ACCUMULATION_STEPS = 2
    # Hybrid loss
    USE_RANKING    = False
    LAMBDA_RANK    = 0.3
    RANK_MARGIN    = 2.0
    PAIR_BUFFER_SIZE = 32
    
    # Video preprocessing 
    TOTAL_PIXELS   = 128 * 1920 * 1080
    MIN_PIXELS     = 192 * 192
    MAX_PIXELS     = 10_000_000_000  
    
    # Model
    MODEL_NAME     = "Qwen/Qwen3-VL-Embedding-2B"
    LORA_R         = 8
    LORA_ALPHA     = 32
    LORA_DROPOUT   = 0.05
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Prompt
    PROMPT         = ""
    FAST_MODE      = False