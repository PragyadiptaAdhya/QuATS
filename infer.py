import sys
import os
import torch
import pandas as pd
import time
import warnings
from pathlib import Path
from tqdm import tqdm
from peft import load_peft_weights, set_peft_model_state_dict

# Force Offline Mode for speed and reliability
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from config import Config
from utils.dataloader import get_video_frames
from model import QwenVQAModel

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def load_best_model(config: Config, best_model_dir: Path):
    print(f"Initializing model from {config.MODEL_NAME}...")
    wrapper = QwenVQAModel(config)
    
    lora_path = best_model_dir / "lora"
    if lora_path.exists():
        adapters_weights = load_peft_weights(lora_path)
        set_peft_model_state_dict(wrapper.model, adapters_weights)
        print("LoRA weights loaded.")

    regressor_path = best_model_dir / "regressor.pt"
    if regressor_path.exists():
        wrapper.regressor.load_state_dict(torch.load(regressor_path, map_location=config.DEVICE))
        print("Regressor head loaded.")

    wrapper.model.eval()
    wrapper.regressor.eval()
    return wrapper

def predict_video_quality(video_path: str, wrapper: QwenVQAModel, config: Config):
    # This call gets fresh random patches on every iteration
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
    cfg = Config()
    best_dir = Path("best_model") 
    
    # --- MODE LOGIC ---
    is_fast = getattr(cfg, "FAST_MODE", False)
    n_runs = 1 if is_fast else 5
    mode_str = "FAST" if is_fast else "ROBUST"
    
    out_csv_file = f"inference_results_{mode_str.lower()}.csv"
    out_csv = cfg.OUTPUT_DIR / out_csv_file
    
    # Ensure directory exists
    os.makedirs(out_csv.parent, exist_ok=True)
    
    try:
        model_wrapper = load_best_model(cfg, best_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    input_path = Path(cfg.TEST_VID_DIR)
    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = list(input_path.glob("*.mp4"))
    else:
        print(f"Invalid path: {input_path}")
        sys.exit(1)

    print(f"\nMode: {mode_str} | Runs per video: {n_runs} | Total Videos: {len(videos)}")

    # Header includes 'runs' count to clarify if it was averaged
    columns = ["observation", "video_name", "avg_predicted_score", "avg_mos_1_to_5", "total_inference_time", "runs"]
    pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
    
    observation_count = 1

    for vid in tqdm(videos, desc=f"Processing ({mode_str})"):
        run_scores = []
        run_times = []
        
        try:
            for i in range(n_runs):
                start_time = time.time()
                score = predict_video_quality(str(vid), model_wrapper, cfg)
                run_times.append(time.time() - start_time)
                run_scores.append(score)
                
                # VRAM cleanup inside the sub-loop to prevent buildup during 5 runs
                torch.cuda.empty_cache()

            # --- AGGREGATION ---
            avg_score = sum(run_scores) / len(run_scores)
            total_time = sum(run_times)
            
            # Linear mapping 1-100 to 1-5
            avg_mos_1_to_5 = 1.0 + 4.0 * ((avg_score - 1) / 99.0)
            
            row_data = {
                "observation": observation_count,
                "video_name": vid.name,
                "avg_predicted_score": round(avg_score, 4),
                "avg_mos_1_to_5": round(avg_mos_1_to_5, 4),
                "total_inference_time": round(total_time, 4),
                "runs": n_runs
            }
            
            pd.DataFrame([row_data]).to_csv(out_csv, mode='a', header=False, index=False)
            observation_count += 1
                
        except Exception as e:
            print(f"\nSkipping {vid.name} due to error: {e}")
            continue

    print(f"\nDone! Results saved to: {out_csv}")