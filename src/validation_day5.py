import os
from src.config import DAY5_OUTPUT_DIR

def validate_day5():
    required = [
        "results/episode_metrics.csv",
        "results/aggregate_metrics.csv",
        "logs/day5_report.json",
        "plots/scenario_cumulative_reward.png",
        "plots/scenario_constraint_violations.png",
        "plots/scenario_iae.png",
        "plots/tracking_nominal_fixed.png",
        "plots/tracking_setpoint_steps.png",
        "plots/tracking_feed_ca_disturbance.png",
        "plots/tracking_actuator_delay.png",
        "plots/robustness_heatmap.png"
    ]
    
    for r in required:
        if not os.path.exists(os.path.join(DAY5_OUTPUT_DIR, r)):
            return False
            
    import pandas as pd
    df = pd.read_csv(os.path.join(DAY5_OUTPUT_DIR, "results", "aggregate_metrics.csv"))
    if df.isnull().values.any(): return False
    
    if not set(["PID", "MF-SAC", "MLP-Dyna-SAC", "PI-Dyna-SAC"]).issubset(set(df["method"].unique())):
        return False
        
    return True
