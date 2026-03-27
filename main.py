
import sys
import os
import torch
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

from config import Config
from utils.dataloader import get_video_frames
from model import QwenVQAModel
from peft import load_peft_weights, set_peft_model_state_dict


import transformers
transformers.logging.set_verbosity_error()


def load_best_model(config: Config, best_model_dir: Path):
    wrapper = QwenVQAModel(config)
    
    lora_path = best_model_dir / "lora"
    if lora_path.exists():
        adapters_weights = load_peft_weights(lora_path)
        set_peft_model_state_dict(wrapper.model, adapters_weights)
    else:
        raise FileNotFoundError(f"LoRA directory not found at {lora_path}")

    regressor_path = best_model_dir / "regressor.pt"
    if regressor_path.exists():
        wrapper.regressor.load_state_dict(torch.load(regressor_path, map_location=config.DEVICE))
    else:
        raise FileNotFoundError(f"Regressor file not found at {regressor_path}")

    wrapper.model.eval()
    wrapper.regressor.eval()
    
    if hasattr(wrapper, 'eval'):
        wrapper.eval()
        
    return wrapper

def predict_video_quality(video_path: str, wrapper: QwenVQAModel, config: Config):
    frames, fps, _ = get_video_frames(video_path, num_frames=config.NUM_KEYFRAMES)
    
    conversation = [wrapper.embedder.format_model_input(
        text=config.PROMPT, 
        video=frames, 
        fps=fps, 
        max_frames=len(frames)
    )]
    
    processed = {k: v.to(config.DEVICE) for k, v in wrapper.embedder._preprocess_inputs(conversation).items()}
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=config.DTYPE):
            pred = wrapper(processed)
            score = pred.item()
            
    return score
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py [pvs_video] [ref_video] [result_file]")
        sys.exit(1)

    pvs_video_path = sys.argv[1]
    ref_video_path = sys.argv[2]  
    result_file_path = sys.argv[3]

    cfg = Config()
    best_dir = cfg.OUTPUT_DIR / "best_model"
    
    try:
        model_wrapper = load_best_model(cfg, best_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    num_runs = 1 if getattr(cfg, "FAST_MODE", False) else 5

    total_score = 0.0
    
    try:
        for i in range(num_runs):
            run_score = predict_video_quality(pvs_video_path, model_wrapper, cfg)
            total_score += run_score

            torch.cuda.empty_cache()
            
        average_score = total_score / num_runs
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


    output_dir = os.path.dirname(result_file_path)
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)
    
    with open(result_file_path, 'w') as f:
        f.write(f"{average_score:.4f}\n")