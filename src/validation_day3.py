import os
from src.config import DAY3_OUTPUT_DIR

def validate_day3():
    required = [
        "env/rl_env_metadata.json",
        "models/pid_controller.json",
        "logs/learning_curves.csv",
        "logs/baseline_metrics.csv",
        "logs/day3_report.json",
        "plots/learning_curves.png",
        "plots/pid_nominal.png",
        "plots/synthetic_rollout_example.png"
    ]
    
    for r in required:
        if not os.path.exists(os.path.join(DAY3_OUTPUT_DIR, r)):
            return False
            
    for seed in [42, 123, 2024]:
        if not os.path.exists(os.path.join(DAY3_OUTPUT_DIR, "models", f"mf_sac_seed_{seed}.zip")): return False
        if not os.path.exists(os.path.join(DAY3_OUTPUT_DIR, "models", f"mlp_dyna_sac_seed_{seed}.zip")): return False
        
    import pandas as pd
    df = pd.read_csv(os.path.join(DAY3_OUTPUT_DIR, "logs", "learning_curves.csv"))
    if df.isnull().values.any(): return False
    
    return True
