import os
import json
import pandas as pd
import numpy as np

def run_analysis_day6(project_root):
    day2_metrics_path = os.path.join(project_root, "outputs/day2/logs/evaluation_metrics.json")
    day5_ep_path = os.path.join(project_root, "outputs/day5/results/episode_metrics.csv")
    day5_agg_path = os.path.join(project_root, "outputs/day5/results/aggregate_metrics.csv")
    day3_curves = os.path.join(project_root, "outputs/day3/logs/learning_curves.csv")
    day4_curves = os.path.join(project_root, "outputs/day4/logs/learning_curves.csv")
    
    out_dir = os.path.join(project_root, "outputs/day6/analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Surrogate Comparison
    if os.path.exists(day2_metrics_path):
        with open(day2_metrics_path, "r") as f:
            d2_metrics = json.load(f)
            
        surrogate_data = []
        for model in ["MLP", "PINN"]:
            if model in d2_metrics:
                m = d2_metrics[model]
                surrogate_data.append({
                    "model_type": model,
                    "one_step_mse_C_A": m.get("MSE_C_A", 0),
                    "one_step_mse_T": m.get("MSE_T", 0),
                    "one_step_combined_mse": m.get("MSE_C_A", 0) + m.get("MSE_T", 0),
                    "physics_residual_C_A": m.get("PHYSICS_RESIDUAL_C_A", m.get("PHYSICS_RESIDUAL", 0)),
                    "physics_residual_T_raw": m.get("PHYSICS_RESIDUAL_T_RAW", 0),
                    "physics_residual_T_weighted": m.get("PHYSICS_RESIDUAL_T", 0),
                    "physics_residual_combined": m.get("PHYSICS_RESIDUAL", 0),
                    "unphysical_rollout_count": m.get("UNPHYSICAL_ROLLOUT_COUNT", 0),
                    "inference_time_seconds": m.get("INFERENCE_TIME", 0.0)
                })
        df_surr = pd.DataFrame(surrogate_data)
        df_surr.to_csv(os.path.join(out_dir, "surrogate_comparison.csv"), index=False)
    
    # 2. Main Results
    if os.path.exists(day5_ep_path):
        df_ep = pd.read_csv(day5_ep_path)
        grouped = df_ep.groupby(["method", "scenario"]).agg(
            mean_cumulative_reward=("cumulative_reward", "mean"),
            std_cumulative_reward=("cumulative_reward", "std"),
            mean_iae_T=("iae_T", "mean"),
            std_iae_T=("iae_T", "std"),
            mean_constraint_violation_rate=("constraint_violation_rate", "mean"),
            std_constraint_violation_rate=("constraint_violation_rate", "std"),
            mean_max_T=("max_T", "mean"),
            std_max_T=("max_T", "std"),
            mean_control_effort=("control_effort", "mean"),
            std_control_effort=("control_effort", "std"),
        ).reset_index()
        grouped.to_csv(os.path.join(out_dir, "main_results.csv"), index=False)
        
        # 3. Robustness Scores
        methods = df_ep["method"].unique()
        scenarios = df_ep["scenario"].unique()
        
        robustness_data = []
        for sc in scenarios:
            sc_data = grouped[grouped["scenario"] == sc]
            
            r = sc_data["mean_cumulative_reward"].values
            i = sc_data["mean_iae_T"].values
            v = sc_data["mean_constraint_violation_rate"].values
            
            # Normalize
            if np.max(r) != np.min(r):
                r_norm = (r - np.min(r)) / (np.max(r) - np.min(r))
            else: r_norm = np.ones_like(r)
            
            if np.max(i) != np.min(i):
                i_norm = (np.max(i) - i) / (np.max(i) - np.min(i))
            else: i_norm = np.ones_like(i)
                
            if np.max(v) != np.min(v):
                v_norm = (np.max(v) - v) / (np.max(v) - np.min(v))
            else: v_norm = np.ones_like(v)
            
            overall = 0.5 * r_norm + 0.3 * i_norm + 0.2 * v_norm
            overall = np.clip(overall, 0.0, 1.0)
            
            for idx, m in enumerate(sc_data["method"].values):
                robustness_data.append({
                    "method": m,
                    "scenario": sc,
                    "reward_score": r_norm[idx],
                    "iae_score": i_norm[idx],
                    "constraint_score": v_norm[idx],
                    "overall_robustness_score": overall[idx]
                })
        
        df_rob = pd.DataFrame(robustness_data)
        df_rob.to_csv(os.path.join(out_dir, "robustness_scores.csv"), index=False)
        
    # 4. Sample Efficiency
    if os.path.exists(day3_curves) and os.path.exists(day4_curves):
        df3 = pd.read_csv(day3_curves)
        df4 = pd.read_csv(day4_curves)
        df_all = pd.concat([df3, df4], ignore_index=True)
        
        eff_data = []
        for m in df_all["method"].unique():
            for s in df_all[df_all["method"] == m]["seed"].unique():
                trace = df_all[(df_all["method"] == m) & (df_all["seed"] == s)].sort_values("timestep")
                if trace.empty: continue
                final_r = trace["mean_reward"].iloc[-1]
                target_r = final_r * 0.9 if final_r > 0 else final_r * 1.1 # 90% logic works differently for negative reward
                # Actually, 90% of final reward distance from min? Let's just use 0.9*final_r for simplicity, or 90% gap.
                min_r = trace["mean_reward"].min()
                target_r = min_r + 0.9 * (final_r - min_r)
                
                reached = trace[trace["mean_reward"] >= target_r]
                steps = reached["timestep"].iloc[0] if not reached.empty else trace["timestep"].iloc[-1]
                
                eff_data.append({
                    "method": m,
                    "seed": s,
                    "steps_to_90_percent_final_reward": steps,
                    "final_mean_reward": final_r
                })
        df_eff = pd.DataFrame(eff_data)
        df_eff.to_csv(os.path.join(out_dir, "sample_efficiency_summary.csv"), index=False)
        
    # Consolidated JSON
    with open(os.path.join(out_dir, "consolidated_summary.json"), "w") as f:
        json.dump({"status": "SUCCESS"}, f)
