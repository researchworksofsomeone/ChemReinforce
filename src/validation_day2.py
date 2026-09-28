import os
import json
import torch
import time
from .world_model import SurrogateWorldModel
from .config import DAY2_OUTPUT_DIR
from .utils import get_device

def validate_day2():
    print("Validating Day 2...")
    report = {"status": "PASS", "checks": {}}
    
    req_files = [
        "models/mlp_surrogate.pt",
        "models/pinn_surrogate.pt",
        "models/scalers.pt",
        "logs/evaluation_metrics.json"
    ]
    
    for f in req_files:
        path = os.path.join(DAY2_OUTPUT_DIR, f)
        if os.path.exists(path):
            report["checks"][f] = "PASS"
        else:
            report["checks"][f] = "FAIL: Missing"
            report["status"] = "FAIL"
            
    try:
        wm = SurrogateWorldModel(os.path.join(DAY2_OUTPUT_DIR, "models", "mlp_surrogate.pt"))
        state = torch.tensor([[1.0, 350.0]])
        action = torch.tensor([[350.0]])
        next_state = wm.predict_next_state(state, action)
        if not torch.isnan(next_state).any():
            report["checks"]["world_model_dummy"] = "PASS"
        else:
            report["checks"]["world_model_dummy"] = "FAIL: NaN in output"
            report["status"] = "FAIL"
    except Exception as e:
        report["checks"]["world_model_dummy"] = f"FAIL: {e}"
        report["status"] = "FAIL"
        
    os.makedirs(os.path.join(DAY2_OUTPUT_DIR, "logs"), exist_ok=True)
    with open(os.path.join(DAY2_OUTPUT_DIR, "logs", "day2_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    return report["status"] == "PASS"
