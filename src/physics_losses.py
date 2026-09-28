import torch
from .config import CSTR_CONFIG

def compute_physics_loss(state_raw, action_raw, predicted_delta_raw):
    dt = CSTR_CONFIG["dt"]
    predicted_derivative = predicted_delta_raw / dt
    
    C_A = state_raw[:, 0]
    T = state_raw[:, 1]
    T_c = action_raw[:, 0]
    
    F_over_V = CSTR_CONFIG["F"] / CSTR_CONFIG["V"]
    
    T_safe = torch.clamp(T, min=200.0)
    rate = CSTR_CONFIG["k0"] * torch.exp(-CSTR_CONFIG["E_over_R"] / T_safe)
    
    dC_A = F_over_V * (CSTR_CONFIG["C_Af"] - C_A) - rate * C_A
    dT = F_over_V * (CSTR_CONFIG["T_f"] - T) + CSTR_CONFIG["delta_H_over_rho_Cp"] * rate * C_A + CSTR_CONFIG["UA_over_rho_Cp_V"] * (T_c - T)
    
    physics_dC_A = dC_A
    physics_dT = dT
    
    residual_C_A = predicted_derivative[:, 0] - physics_dC_A
    residual_T = predicted_derivative[:, 1] - physics_dT
    
    loss_C_A = torch.mean(residual_C_A**2)
    loss_T = torch.mean(residual_T**2) / (100.0**2)
    
    return loss_C_A + loss_T

def compute_bounds_loss(next_state_pred_raw):
    C_A = next_state_pred_raw[:, 0]
    T = next_state_pred_raw[:, 1]
    
    loss_C_A = torch.mean(torch.relu(-C_A)**2)
    loss_T_low = torch.mean(torch.relu(250.0 - T)**2)
    loss_T_high = torch.mean(torch.relu(T - 500.0)**2)
    
    return loss_C_A + (loss_T_low + loss_T_high) / (100.0**2)
