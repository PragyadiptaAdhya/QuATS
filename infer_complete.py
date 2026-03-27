import sys
import os
import torch
import pandas as pd
import time
import warnings
import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import roc_auc_score
from peft import load_peft_weights, set_peft_model_state_dict

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
    else:
        raise FileNotFoundError(f"LoRA directory not found at {lora_path}")

    regressor_path = best_model_dir / "regressor.pt"
    if regressor_path.exists():
        wrapper.regressor.load_state_dict(torch.load(regressor_path, map_location=config.DEVICE))
        print("Regressor head loaded.")
    else:
        raise FileNotFoundError(f"Regressor file not found at {regressor_path}")

    wrapper.model.eval()
    wrapper.regressor.eval()
    return wrapper

def predict_video_quality(video_path: str, wrapper: QwenVQAModel, config: Config):
    frames, fps, _ = get_video_frames(video_path, num_frames=config.NUM_KEYFRAMES)
    conversation = [wrapper.embedder.format_model_input(
        text=config.PROMPT, video=frames, fps=fps, max_frames=len(frames)
    )]
    processed = {k: v.to(config.DEVICE) for k, v in wrapper.embedder._preprocess_inputs(conversation).items()}
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=config.DTYPE):
        pred = wrapper(processed)
        return pred.item()

# --- EVALUATION MATH ---
def logistic_4_param(x, b1, b2, b3, b4):
    x_shifted = np.clip(-(x - b3) / np.abs(b4), -500, 500)
    return (b1 - b2) / (1 + np.exp(x_shifted)) + b2

def compute_krasula_metrics(mos, mapped_preds, threshold):
    pairs = list(itertools.combinations(range(len(mos)), 2))
    delta_mos = np.array([mos[i] - mos[j] for i, j in pairs])
    delta_pred = np.array([mapped_preds[i] - mapped_preds[j] for i, j in pairs])
    
    labels_ds = (np.abs(delta_mos) > threshold).astype(int)
    scores_ds = np.abs(delta_pred)
    try:
        auc_ds = roc_auc_score(labels_ds, scores_ds)
    except ValueError:
        auc_ds = np.nan
        
    diff_mask = (labels_ds == 1)
    delta_mos_diff = delta_mos[diff_mask]
    delta_pred_diff = delta_pred[diff_mask]
    
    labels_bw = (delta_mos_diff > 0).astype(int)
    scores_bw = delta_pred_diff
    try:
        auc_bw = roc_auc_score(labels_bw, scores_bw)
    except ValueError:
        auc_bw = np.nan
        
    preds_bw = (delta_pred_diff > 0).astype(int)
    cc_bw = np.mean(preds_bw == labels_bw) if len(labels_bw) > 0 else np.nan
    
    return auc_ds, auc_bw, cc_bw

def evaluate_run(df_obs: pd.DataFrame, df_ref: pd.DataFrame, cfg):
    """Aligns predictions with ground truth, fits logistic curve, and calculates VQA metrics."""
    df_obs = df_obs.copy()
    df_obs['video_name'] = df_obs['video_name'].astype(str).str.replace(r'\.mp4$', '', regex=True)
    
    df_combined = pd.merge(df_obs, df_ref, on='video_name', how='inner')
    if df_combined.empty:
        raise ValueError("No matching videos found between inference and reference CSVs.")

    y_pred = df_combined['AverageScore'].values
    y_true = df_combined['mos'].values

    b1_guess, b2_guess = np.max(y_true), np.min(y_true)
    b3_guess = np.mean(y_pred)
    b4_guess = np.std(y_pred) / 4.0 if np.std(y_pred) > 0 else 1.0
    beta0 = [b1_guess, b2_guess, b3_guess, b4_guess]

    try:
        optimized_betas, _ = curve_fit(logistic_4_param, y_pred, y_true, p0=beta0, maxfev=10000)
        y_pred_mapped = logistic_4_param(y_pred, *optimized_betas)
    except RuntimeError:
        y_pred_mapped = y_pred

    srcc, _ = spearmanr(y_pred, y_true, nan_policy='omit')
    krcc, _ = kendalltau(y_pred, y_true, nan_policy='omit')
    plcc, _ = pearsonr(y_pred_mapped, y_true)
    rmse = np.sqrt(np.nanmean((y_true - y_pred_mapped)**2))

    auc_ds, auc_bw, cc_bw = compute_krasula_metrics(y_true, y_pred_mapped, cfg.SIMILAR_MOS_THRESHOLD)

    return {
        "SRCC": srcc, "PLCC": plcc, "RMSE": rmse, "KRCC": krcc,
        "AUC_DS": auc_ds, "AUC_BW": auc_bw, "CC_BW": cc_bw
    }

def run_inference_pipeline(mode_str: str, passes_per_vid: int, wrapper: QwenVQAModel, config: Config, video_list: list, df_ref: pd.DataFrame):
    print("\n" + "="*50)
    print(f" STARTING {mode_str.upper()} MODE ({passes_per_vid} Passes/Video)")
    print("="*50)

    out_csv = config.INFERENCE_DIR / f"inference_results_complete_{mode_str}.csv"
    os.makedirs(out_csv.parent, exist_ok=True)
    
    columns = ["eval_loop", "video_name", "AverageScore", "mos_1_to_5", "inference_time", "passes_per_vid"]
    pd.DataFrame(columns=columns).to_csv(out_csv, index=False)
    
    all_loop_metrics = []

    for eval_idx in range(config.NUM_EVAL_LOOPS):
        print(f"\n--- {mode_str.upper()} Loop {eval_idx + 1}/{config.NUM_EVAL_LOOPS} ---")
        run_results = []
        
        for vid in tqdm(video_list, desc=f"Inferencing"):
            scores, times = [], []
            try:
                for _ in range(passes_per_vid):
                    start_time = time.time()
                    scores.append(predict_video_quality(str(vid), wrapper, config))
                    times.append(time.time() - start_time)
                    torch.cuda.empty_cache()
                
                avg_score = sum(scores) / len(scores)
                run_results.append({
                    "eval_loop": eval_idx + 1,
                    "video_name": vid.name,
                    "AverageScore": round(avg_score, 4),
                    "mos_1_to_5": round(1.0 + 4.0 * ((avg_score - 1) / 99.0), 4),
                    "inference_time": round(sum(times), 4),
                    "passes_per_vid": passes_per_vid
                })
            except Exception as e:
                print(f"\nSkipping {vid.name}: {e}")
                
        df_obs = pd.DataFrame(run_results)
        df_obs.to_csv(out_csv, mode='a', header=False, index=False)
        
        try:
            loop_metrics = evaluate_run(df_obs, df_ref, config)
            all_loop_metrics.append(loop_metrics)
            print(f"Loop {eval_idx + 1} SRCC: {loop_metrics['SRCC']:.4f} | PLCC: {loop_metrics['PLCC']:.4f}")
        except Exception as e:
            print(f"Error computing metrics for loop {eval_idx + 1}: {e}")

    # Build the summary dictionary
    summary_dict = {"Mode": mode_str.upper()}
    if all_loop_metrics:
        print('\n' + '-'*40)
        print(f' FINAL METRICS: {mode_str.upper()} MODE (Mean ± Std)')
        print('-'*40)
        
        for key in all_loop_metrics[0].keys():
            vals = [m[key] for m in all_loop_metrics if not np.isnan(m[key])]
            mean_v = np.mean(vals) if vals else np.nan
            std_v = np.std(vals) if vals else np.nan
            
            val_str = f"{mean_v:.4f} ± {std_v:.4f}"
            summary_dict[key] = val_str
            print(f"{key.ljust(15)}: {val_str}")
            
        print(f"{mode_str.upper()} Inferences saved to: {out_csv}")
    
    return summary_dict

if __name__ == "__main__":
    cfg = Config()
    best_dir = cfg.OUTPUT_DIR / "best_model"
    final_metrics_csv = cfg.INFERENCE_DIR / "metrics.csv"
    
    try:
        model_wrapper = load_best_model(cfg, best_dir)
        reference_df = pd.read_csv(cfg.TEST_CSV)[['video_name', 'mos']]
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)
        
    input_path = Path(cfg.TEST_VID_DIR)
    videos = [input_path] if input_path.is_file() else list(input_path.glob("*.mp4"))
    
    if not videos:
        print(f"No videos found in {input_path}")
        sys.exit(1)

    # Execute Sequential Pipeline and collect summaries
    final_results = []
    
    fast_summary = run_inference_pipeline(mode_str="fast", passes_per_vid=1, wrapper=model_wrapper, config=cfg, video_list=videos, df_ref=reference_df)
    final_results.append(fast_summary)
    
    robust_summary = run_inference_pipeline(mode_str="robust", passes_per_vid=5, wrapper=model_wrapper, config=cfg, video_list=videos, df_ref=reference_df)
    final_results.append(robust_summary)
    
    # Save both rows to a single metrics.csv
    pd.DataFrame(final_results).to_csv(final_metrics_csv, index=False)
    
    print("\n" + "="*50)
    print(f"All evaluations complete.")
    print(f"Master metrics file saved to: {final_metrics_csv}")
    print("="*50)