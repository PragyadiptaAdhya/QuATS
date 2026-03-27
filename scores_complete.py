import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, roc_auc_score

obs_csv = "" #Add path from the output of infer_complete script
ref_csv = "" #Add path to your test data csv file
SIMILAR_MOS_THRESHOLD = 0.25 

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

def evaluate_single_loop(df_loop, df_ref):
    """Merges a single loop's predictions with the ground truth and calculates metrics."""
    df_combined = pd.merge(df_loop, df_ref, on='video_name', how='inner')
    
    if df_combined.empty:
        raise ValueError("No matching videos found between inference and reference CSVs for this loop.")

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

    # Calculate Metrics
    srcc, _ = spearmanr(y_pred, y_true, nan_policy='omit')
    krcc, _ = kendalltau(y_pred, y_true, nan_policy='omit')
    plcc, _ = pearsonr(y_pred_mapped, y_true)
    rmse = np.sqrt(np.nanmean((y_true - y_pred_mapped)**2))

    auc_ds, auc_bw, cc_bw = compute_krasula_metrics(y_true, y_pred_mapped, SIMILAR_MOS_THRESHOLD)

    return {
        "SRCC": srcc, "PLCC": plcc, "RMSE": rmse, "KRCC": krcc,
        "AUC_DS": auc_ds, "AUC_BW": auc_bw, "CC_BW": cc_bw
    }

if __name__ == "__main__":
    df_obs = pd.read_csv(obs_csv)
    df_ref = pd.read_csv(ref_csv)

    df_obs['video_name'] = df_obs['video_name'].astype(str).str.replace(r'\.mp4$', '', regex=True)
    df_ref['video_name'] = df_ref['video_name'].astype(str).str.replace(r'\.mp4$', '', regex=True)
    df_ref = df_ref[['video_name', 'mos']]

    all_loop_metrics = []
    unique_loops = df_obs['eval_loop'].unique()

    print(f"Found {len(unique_loops)} evaluation loop(s) in the inference data.\n")

    for loop_num in sorted(unique_loops):
        df_loop = df_obs[df_obs['eval_loop'] == loop_num].copy()
        try:
            metrics = evaluate_single_loop(df_loop, df_ref)
            all_loop_metrics.append(metrics)
            print(f"Loop {loop_num} -> SRCC: {metrics['SRCC']:.4f} | PLCC: {metrics['PLCC']:.4f}")
        except Exception as e:
            print(f"Error computing metrics for Loop {loop_num}: {e}")

    if all_loop_metrics:
        print('\n' + '='*45)
        print(' FINAL METRICS ACROSS ALL LOOPS (Mean ± Std)')
        print('='*45)
        
        for key in all_loop_metrics[0].keys():
            vals = [m[key] for m in all_loop_metrics if not np.isnan(m[key])]
            if vals:
                mean_v = np.mean(vals)
                std_v = np.std(vals)
                print(f"{key.ljust(10)} : {mean_v:.4f} ± {std_v:.4f}")
            else:
                print(f"{key.ljust(10)} : NaN")
        
        print('='*45)
        
        if 'inference_time' in df_obs.columns:
            avg_time = df_obs['inference_time'].mean()
            print(f"Avg Inference Time: {avg_time:.4f} seconds/run")


        combined_paths = obs_csv.lower() + ref_csv.lower()
        if "robust" in combined_paths:
            mode_str = "robust"
        elif "fast" in combined_paths:
            mode_str = "fast"
        else:
            mode_str = "default"
            
        print(f"\nGenerating scatter plot for globally averaged predictions ({mode_str} mode)...")
        
        df_obs_avg = df_obs.groupby('video_name', as_index=False)['AverageScore'].mean()
        df_plot = pd.merge(df_obs_avg, df_ref, on='video_name', how='inner')

        y_pred_avg = df_plot['AverageScore'].values
        y_true_plot = df_plot['mos'].values

        b1_g, b2_g = np.max(y_true_plot), np.min(y_true_plot)
        b3_g = np.mean(y_pred_avg)
        b4_g = np.std(y_pred_avg) / 4.0 if np.std(y_pred_avg) > 0 else 1.0
        
        try:
            opt_betas_plot, _ = curve_fit(logistic_4_param, y_pred_avg, y_true_plot, p0=[b1_g, b2_g, b3_g, b4_g], maxfev=10000)
            
            plt.figure(figsize=(10, 7))
            plt.scatter(y_pred_avg, y_true_plot, s=50, c='blue', edgecolors='white', alpha=0.6, label='Video Samples (Averaged)')
            
            x_smooth = np.linspace(np.min(y_pred_avg), np.max(y_pred_avg), 100)
            y_smooth = logistic_4_param(x_smooth, *opt_betas_plot)
            
            plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted Logistic Curve')

            plt.xlabel('Average Model Prediction', fontsize=11)
            plt.ylabel('Ground Truth MOS', fontsize=11)
            plt.title(f'VQA Predictions vs. Ground Truth (Averaged Across Loops - {mode_str.title()})')
            
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plot_filename = f'vqa_scatter_plot_{mode_str}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved successfully to: {plot_filename}")
            
        except RuntimeError:
            print("Curve fitting failed for the global plot. Skipping plot generation.")