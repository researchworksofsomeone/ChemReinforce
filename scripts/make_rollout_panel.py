import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.surrogate_models import SurrogateModel
from src.config import DAY1_OUTPUT_DIR, DAY2_OUTPUT_DIR, CSTR_CONFIG

def main():
    device = torch.device("cpu")
    
    # Load dataset
    data_path = os.path.join(DAY1_OUTPUT_DIR, "dataset", "cstr_transitions.pt")
    d = torch.load(data_path, map_location=device)
    
    # Extract validation episodes (let's say the last 20 episodes)
    # The dataset has episodes of length 200.
    ep_length = 200
    num_val_eps = 20
    
    total_steps = len(d['done'])
    start_idx = total_steps - (num_val_eps * ep_length)
    
    val_s_CA = d['state_C_A'][start_idx:]
    val_s_T = d['state_T'][start_idx:]
    val_a_Tc = d['action_T_c'][start_idx:]
    
    # Combine into tensors (num_eps, ep_length, dim)
    val_s = torch.stack([val_s_CA, val_s_T], dim=-1).view(num_val_eps, ep_length, 2)
    val_a = val_a_Tc.view(num_val_eps, ep_length, 1)
    
    # Load scalers
    scalers = torch.load(os.path.join(DAY2_OUTPUT_DIR, "models", "scalers.pt"), map_location=device)
    s_mean = scalers['state_scaler_mean']
    s_std = scalers['state_scaler_std']
    a_mean = scalers['action_scaler_mean']
    a_std = scalers['action_scaler_std']
    d_mean = scalers['delta_scaler_mean']
    d_std = scalers['delta_scaler_std']
    
    def normalize_state(s): return (s - s_mean) / s_std
    def normalize_action(a): return (a - a_mean) / a_std
    def unnormalize_delta(d_norm): return d_norm * d_std + d_mean
    
    # Load models
    models = {}
    for mt in ['mlp', 'pinn']:
        m = SurrogateModel().to(device)
        ckpt = torch.load(os.path.join(DAY2_OUTPUT_DIR, "models", f"{mt}_surrogate.pt"), map_location=device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models[mt] = m
        
    horizons = [1, 5, 10, 20, 50]
    results = {'mlp': [], 'pinn': []}
    
    with torch.no_grad():
        for mt, model in models.items():
            for H in horizons:
                # We will rollout for H steps starting from t=0 for each episode
                # Ground truth T is val_s[:, :H+1, 1]
                # Actually, let's start at t=0 and rollout H steps
                
                s_pred = val_s[:, 0, :].clone() # (num_eps, 2)
                preds = [s_pred] # store states
                
                for t in range(H):
                    a_t = val_a[:, t, :]
                    s_norm = normalize_state(s_pred)
                    a_norm = normalize_action(a_t)
                    
                    d_norm = model(s_norm, a_norm)
                    d_raw = unnormalize_delta(d_norm)
                    s_pred = s_pred + d_raw
                    preds.append(s_pred)
                    
                preds = torch.stack(preds, dim=1) # (num_eps, H+1, 2)
                
                # compute RMSE on T (index 1) at horizon H
                # true state at t=H is val_s[:, H, 1]
                # Wait, our preds[H] is the prediction for t=H
                pred_T = preds[:, H, 1]
                true_T = val_s[:, H, 1]
                rmse = torch.sqrt(torch.mean((pred_T - true_T)**2)).item()
                results[mt].append(rmse)
                
    # Plotting
    fig, ax_a = plt.subplots(1, 1, figsize=(3.5, 3.0))
    # We will just recreate Fig1a and add Fig1b
    # Let's make a 2-panel figure
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax_a, ax_b = axes
    
    # Panel a (embedded data)
    labels = ['One-step T MSE', 'Physics residual']
    mlp_vals = [0.0138, 0.00892]
    pinn_vals = [0.0111, 0.00875]
    x = np.arange(len(labels))
    width = 0.35
    
    ax_a.bar(x - width/2, mlp_vals, width, label='MLP', color='#E69F00')
    ax_a.bar(x + width/2, pinn_vals, width, label='PINN', color='#D55E00')
    ax_a.set_ylabel('MSE [K$^2$]')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.legend(fontsize=8)
    ax_a.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    ax_a.set_title('(a)', loc='left', fontweight='bold', fontsize=10)
    
    # Panel b (rollout)
    ax_b.plot(horizons, results['mlp'], marker='o', label='MLP', color='#E69F00', linewidth=1.5)
    ax_b.plot(horizons, results['pinn'], marker='s', label='PINN', color='#D55E00', linewidth=1.5)
    ax_b.set_xlabel('Horizon [steps]')
    ax_b.set_ylabel('T RMSE [K]')
    ax_b.set_xticks(horizons)
    ax_b.legend(fontsize=8)
    ax_b.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    ax_b.set_title('(b)', loc='left', fontweight='bold', fontsize=10)
    
    for ax in [ax_a, ax_b]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    fig.tight_layout()
    out_dir = "outputs/paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "Fig1.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, "Fig1.png"), dpi=300, bbox_inches='tight')
    print("Successfully generated Fig 1 (a and b).")

if __name__ == "__main__":
    main()
