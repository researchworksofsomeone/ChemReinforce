import os
import pandas as pd
import numpy as np

def run_statistics_day6(project_root):
    day5_ep_path = os.path.join(project_root, "outputs/day5/results/episode_metrics.csv")
    out_dir = os.path.join(project_root, "outputs/day6/analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(day5_ep_path):
        return
        
    df = pd.read_csv(day5_ep_path)
    scenarios = df["scenario"].unique()
    metrics = ["cumulative_reward", "iae_T", "constraint_violation_rate"]
    
    records = []
    
    for sc in scenarios:
        for metric in metrics:
            pi_data = df[(df["scenario"] == sc) & (df["method"] == "PI-Dyna-SAC")][metric].values
            mlp_data = df[(df["scenario"] == sc) & (df["method"] == "MLP-Dyna-SAC")][metric].values
            
            if len(pi_data) == 0 or len(mlp_data) == 0:
                continue
                
            n_samples = min(len(pi_data), len(mlp_data))
            
            diffs = []
            for _ in range(1000):
                pi_boot = np.random.choice(pi_data, size=n_samples, replace=True)
                mlp_boot = np.random.choice(mlp_data, size=n_samples, replace=True)
                diffs.append(np.mean(pi_boot) - np.mean(mlp_boot))
                
            diffs = np.array(diffs)
            mean_diff = np.mean(diffs)
            ci_lower = np.percentile(diffs, 2.5)
            ci_upper = np.percentile(diffs, 97.5)
            
            if metric == "cumulative_reward":
                better_rate = np.mean(diffs > 0)
            else:
                better_rate = np.mean(diffs < 0)
                
            records.append({
                "scenario": sc,
                "metric": metric,
                "mean_difference": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "pi_better_rate": better_rate,
                "n_samples": n_samples
            })
            
    df_stat = pd.DataFrame(records)
    df_stat.to_csv(os.path.join(out_dir, "statistical_tests.csv"), index=False)
