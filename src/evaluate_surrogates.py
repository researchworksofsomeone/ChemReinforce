import os
import json
import time
import torch
import torch.nn.functional as F
import pandas as pd
from .config import DAY2_OUTPUT_DIR, SURROGATE_CONFIG, DATA_CONFIG, CSTR_CONFIG
from .surrogate_models import SurrogateModel
from .physics_losses import compute_physics_loss
from .utils import get_device

def evaluate_models(test_loader, state_scaler, action_scaler, delta_scaler):
    device = get_device()
    
    models = {}
    for mt in ["MLP", "PINN"]:
        path = os.path.join(DAY2_OUTPUT_DIR, "models", f"{mt.lower()}_surrogate.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=True)
            model = SurrogateModel().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models[mt] = model
            
    metrics = {}
    dt = CSTR_CONFIG["dt"]
    F_over_V = CSTR_CONFIG["F"] / CSTR_CONFIG["V"]
    
    with torch.no_grad():
        for mt, model in models.items():
            mse_ca_list, mse_t_list = [], []
            mae_ca_list, mae_t_list = [], []
            phys_res_ca_list, phys_res_t_list = [], []
            phys_res_t_raw_list = []
            
            for batch in test_loader:
                s_norm = batch["state_normalized"].to(device)
                a_norm = batch["action_normalized"].to(device)
                s_raw = batch["state_raw"].to(device)
                a_raw = batch["action_raw"].to(device)
                d_raw_target = batch["delta_raw"].to(device)
                
                pred_d_norm = model(s_norm, a_norm)
                pred_d_raw = pred_d_norm * torch.tensor(delta_scaler.std, device=device) + torch.tensor(delta_scaler.mean, device=device)
                
                mse_ca_list.append(F.mse_loss(pred_d_raw[:, 0], d_raw_target[:, 0]).item())
                mse_t_list.append(F.mse_loss(pred_d_raw[:, 1], d_raw_target[:, 1]).item())
                
                mae_ca_list.append(F.l1_loss(pred_d_raw[:, 0], d_raw_target[:, 0]).item())
                mae_t_list.append(F.l1_loss(pred_d_raw[:, 1], d_raw_target[:, 1]).item())
                
                # Manual physical residual
                predicted_derivative = pred_d_raw / dt
                C_A = s_raw[:, 0]
                T = s_raw[:, 1]
                T_c = a_raw[:, 0]
                T_safe = torch.clamp(T, min=200.0)
                rate = CSTR_CONFIG["k0"] * torch.exp(-CSTR_CONFIG["E_over_R"] / T_safe)
                dT = F_over_V * (CSTR_CONFIG["T_f"] - T) + CSTR_CONFIG["delta_H_over_rho_Cp"] * rate * C_A + CSTR_CONFIG["UA_over_rho_Cp_V"] * (T_c - T)
                residual_T = torch.abs(predicted_derivative[:, 1] - dT)
                
                phys_res_t_raw_list.append(torch.mean(residual_T).item())
                
                phys_loss = compute_physics_loss(s_raw, a_raw, pred_d_raw)
                phys_res_ca_list.append(phys_loss.item())
                
            # Inference timing
            dummy_s = torch.randn(10000, 2, device=device)
            dummy_a = torch.randn(10000, 1, device=device)
            
            # warmup
            for _ in range(10):
                _ = model(dummy_s, dummy_a)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            num_passes = 1000
            for _ in range(num_passes):
                _ = model(dummy_s, dummy_a)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            inference_time_ms = ((t1 - t0) / num_passes) * 1000
            
            mean_raw_t = sum(phys_res_t_raw_list) / len(phys_res_t_raw_list)
            
            metrics[mt] = {
                "MSE_C_A": sum(mse_ca_list) / len(mse_ca_list),
                "MSE_T": sum(mse_t_list) / len(mse_t_list),
                "MAE_C_A": sum(mae_ca_list) / len(mae_ca_list),
                "MAE_T": sum(mae_t_list) / len(mae_t_list),
                "PHYSICS_RESIDUAL": sum(phys_res_ca_list) / len(phys_res_ca_list),
                "PHYSICS_RESIDUAL_T_RAW": mean_raw_t,
                "PHYSICS_RESIDUAL_T": mean_raw_t / 100.0,
                "INFERENCE_TIME": inference_time_ms / 1000.0  # seconds
            }
            
    os.makedirs(os.path.join(DAY2_OUTPUT_DIR, "logs"), exist_ok=True)
    with open(os.path.join(DAY2_OUTPUT_DIR, "logs", "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics
