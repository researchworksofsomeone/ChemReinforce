import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils import set_seed
from src.surrogate_dataset import prepare_dataloaders
from src.train_surrogates import train_model
from src.evaluate_surrogates import evaluate_models
from src.validation_day2 import validate_day2
from src.kaggle_sink import push_to_kaggle
from src.config import DAY2_OUTPUT_DIR

def main():
    set_seed(42)
    print("Running Day 2...")
    
    train_loader, val_loader, test_loader, s_scaler, a_scaler, d_scaler = prepare_dataloaders()
    
    print("Training MLP ensemble...")
    for i in range(3):
        print(f"Training MLP member {i}...")
        train_model("MLP", train_loader, val_loader, d_scaler, model_idx=i)
    
    print("Training PINN ensemble...")
    for i in range(3):
        print(f"Training PINN member {i}...")
        train_model("PINN", train_loader, val_loader, d_scaler, model_idx=i)
    
    print("Evaluating models...")
    evaluate_models(test_loader, s_scaler, a_scaler, d_scaler)
    
    if validate_day2():
        print("Day 2 Validation: PASS")
        res, msg = push_to_kaggle(DAY2_OUTPUT_DIR, "pi-wm-rl-day2-assets")
        print(f"Kaggle Sink: {res} - {msg}")
        return True
    else:
        print("Day 2 Validation: FAIL")
        return False

if __name__ == "__main__":
    main()
