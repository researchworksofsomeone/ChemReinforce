import os
from src.config import DAY4_OUTPUT_DIR

def validate_day4():
    required = [
        "logs/learning_curves.csv",
        "logs/constraint_shielding_stats.csv",
        "logs/day4_report.json",
        "plots/learning_curves_day4.png",
        "plots/constraint_violations_training.png",
        "plots/synthetic_rollout_example_pinn.png"
    ]
    
    for r in required:
        if not os.path.exists(os.path.join(DAY4_OUTPUT_DIR, r)):
            return False
            
    for seed in [42, 123, 2024]:
        if not os.path.exists(os.path.join(DAY4_OUTPUT_DIR, "models", f"pi_dyna_sac_seed_{seed}.zip")): return False
        
    import pandas as pd
    df = pd.read_csv(os.path.join(DAY4_OUTPUT_DIR, "logs", "learning_curves.csv"))
    if df.isnull().values.any(): return False
    
    return True
