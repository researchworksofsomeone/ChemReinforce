import os
import sys
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import set_seed
from src.data_collector import collect_data
from src.validation_day1 import validate_day1
from src.surrogate_dataset import prepare_dataloaders
from src.train_surrogates import train_model
from src.evaluate_surrogates import evaluate_models
from src.validation_day2 import validate_day2
from src.kaggle_sink import push_to_kaggle
from src.config import DAY1_OUTPUT_DIR, DAY2_OUTPUT_DIR, DATA_CONFIG

def run_all():
    t_start = time.time()
    set_seed(42)
    
    print("Checking Day 1...")
    try:
        day1_pass = validate_day1()
    except Exception:
        day1_pass = False
        
    if not day1_pass:
        print("Day 1 failed or missing. Rebuilding...")
        collect_data(DATA_CONFIG["random_transitions"], DATA_CONFIG["pid_transitions"])
        day1_pass = validate_day1()
        
    day1_sink_res, day1_sink_msg = push_to_kaggle(DAY1_OUTPUT_DIR, "pi-wm-rl-day1-assets")
        
    if not day1_pass:
        print("\n--------------------------------------------------")
        print("DAY 1 STATUS: FAIL")
        print("DAY 2 STATUS: FAIL")
        print(f"DATASET TRANSITIONS: {DATA_CONFIG['total_transitions']}")
        print("MLP ONE-STEP MSE: N/A\nPINN ONE-STEP MSE: N/A\nMLP PHYSICS RESIDUAL: N/A\nPINN PHYSICS RESIDUAL: N/A")
        print("MLP UNPHYSICAL ROLLOUT COUNT: N/A\nPINN UNPHYSICAL ROLLOUT COUNT: N/A")
        print(f"DAY 1 KAGGLE SINK: {day1_sink_res}")
        print("DAY 2 KAGGLE SINK: N/A")
        print(f"TOTAL RUNTIME SECONDS: {time.time() - t_start:.2f}")
        return
        
    print("Day 1 PASS. Running Day 2...")
    train_loader, val_loader, test_loader, s_scaler, a_scaler, d_scaler = prepare_dataloaders()
    
    train_model("MLP", train_loader, val_loader, d_scaler)
    train_model("PINN", train_loader, val_loader, d_scaler)
    
    evaluate_models(test_loader, s_scaler, a_scaler, d_scaler)
    
    day2_pass = validate_day2()
    day2_sink_res, day2_sink_msg = push_to_kaggle(DAY2_OUTPUT_DIR, "pi-wm-rl-day2-assets")
    
    t_end = time.time()
    
    eval_metrics_path = os.path.join(DAY2_OUTPUT_DIR, "logs", "evaluation_metrics.json")
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, "r") as f:
            metrics = json.load(f)
            mlp_mse = metrics.get("MLP", {}).get("MSE_C_A", "N/A") 
            pinn_mse = metrics.get("PINN", {}).get("MSE_C_A", "N/A")
            mlp_phys = metrics.get("MLP", {}).get("PHYSICS_RESIDUAL", "N/A")
            pinn_phys = metrics.get("PINN", {}).get("PHYSICS_RESIDUAL", "N/A")
    else:
        mlp_mse = pinn_mse = mlp_phys = pinn_phys = "N/A"
        
    print("\n--------------------------------------------------")
    print(f"DAY 1 STATUS: {'PASS' if day1_pass else 'FAIL'}")
    print(f"DAY 2 STATUS: {'PASS' if day2_pass else 'FAIL'}")
    print(f"DATASET TRANSITIONS: {DATA_CONFIG['total_transitions']}")
    print(f"MLP ONE-STEP MSE: {mlp_mse}")
    print(f"PINN ONE-STEP MSE: {pinn_mse}")
    print(f"MLP PHYSICS RESIDUAL: {mlp_phys}")
    print(f"PINN PHYSICS RESIDUAL: {pinn_phys}")
    print("MLP UNPHYSICAL ROLLOUT COUNT: N/A")
    print("PINN UNPHYSICAL ROLLOUT COUNT: N/A")
    print(f"DAY 1 KAGGLE SINK: {day1_sink_res}")
    print(f"DAY 2 KAGGLE SINK: {day2_sink_res}")
    print(f"TOTAL RUNTIME SECONDS: {t_end - t_start:.2f}")

if __name__ == "__main__":
    run_all()
