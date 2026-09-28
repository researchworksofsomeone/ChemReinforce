import os
import json
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import pandas as pd
import torch
import numpy as np
import time

from .cstr_env import CSTRGymEnv
from .config import DAY1_OUTPUT_DIR, DATA_CONFIG

def validate_day1():
    print("Validating Day 1...")
    os.makedirs(os.path.join(DAY1_OUTPUT_DIR, "logs"), exist_ok=True)
    report = {"status": "PASS", "checks": {}}
    
    env = CSTRGymEnv()
    try:
        check_env(env.unwrapped)
        report["checks"]["check_env"] = "PASS"
    except Exception as e:
        report["checks"]["check_env"] = f"FAIL: {e}"
        report["status"] = "FAIL"
        
    try:
        df = pd.read_parquet(os.path.join(DAY1_OUTPUT_DIR, "dataset", "cstr_transitions.parquet"))
        if len(df) == DATA_CONFIG["total_transitions"]:
            report["checks"]["dataset_size"] = "PASS"
        else:
            report["checks"]["dataset_size"] = f"FAIL: found {len(df)} rows"
            report["status"] = "FAIL"
            
        if df.isna().any().any():
            report["checks"]["dataset_nans"] = "FAIL"
            report["status"] = "FAIL"
        else:
            report["checks"]["dataset_nans"] = "PASS"
    except Exception as e:
        report["checks"]["dataset_size"] = f"FAIL: {e}"
        report["status"] = "FAIL"
        
    try:
        from .cstr_dynamics import BatchedCSTRDynamics
        from .utils import get_device
        device = get_device()
        dyn = BatchedCSTRDynamics(device)
        state = torch.ones(1024, 2, device=device)
        action = torch.ones(1024, 1, device=device) * 350.0
        
        t0 = time.time()
        dyn.step(state, action)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        report["checks"]["gpu_batched_time"] = f"{t1-t0:.4f}s on {device}"
    except Exception as e:
        report["checks"]["gpu_batched"] = f"FAIL: {e}"
        report["status"] = "FAIL"
        
    with open(os.path.join(DAY1_OUTPUT_DIR, "logs", "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    return report["status"] == "PASS"
