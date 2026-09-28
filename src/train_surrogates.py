import os
import torch
import torch.nn.functional as F
import pandas as pd
import time
from .config import DAY2_OUTPUT_DIR, SURROGATE_CONFIG
from .surrogate_models import SurrogateModel
from .physics_losses import compute_physics_loss, compute_bounds_loss
from .utils import get_device

def train_model(model_type, train_loader, val_loader, delta_scaler, model_idx=0):
    device = get_device()
    model = SurrogateModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SURROGATE_CONFIG["learning_rate"], weight_decay=SURROGATE_CONFIG["weight_decay"])
    
    history = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None
    
    for epoch in range(SURROGATE_CONFIG["epochs"]):
        model.train()
        t0 = time.time()
        
        train_loss, train_data, train_phys, train_bnd = 0.0, 0.0, 0.0, 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            s_norm = batch["state_normalized"].to(device)
            a_norm = batch["action_normalized"].to(device)
            s_raw = batch["state_raw"].to(device)
            a_raw = batch["action_raw"].to(device)
            d_norm_target = batch["delta_normalized"].to(device)
            
            pred_d_norm = model(s_norm, a_norm)
            data_loss = F.mse_loss(pred_d_norm, d_norm_target)
            
            if model_type == "PINN":
                pred_d_raw_tensor = pred_d_norm * torch.tensor(delta_scaler.std, device=device) + torch.tensor(delta_scaler.mean, device=device)
                pred_next_state_raw = s_raw + pred_d_raw_tensor
                phys_loss = compute_physics_loss(s_raw, a_raw, pred_d_raw_tensor)
                bnd_loss = compute_bounds_loss(pred_next_state_raw)
                
                loss = data_loss + SURROGATE_CONFIG["physics_lambda"] * phys_loss + SURROGATE_CONFIG["bounds_lambda"] * bnd_loss
            else:
                loss = data_loss
                phys_loss = torch.tensor(0.0)
                bnd_loss = torch.tensor(0.0)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SURROGATE_CONFIG["gradient_clip_norm"])
            optimizer.step()
            
            train_loss += loss.item()
            train_data += data_loss.item()
            train_phys += phys_loss.item()
            train_bnd += bnd_loss.item()
            
        train_loss /= len(train_loader)
        train_data /= len(train_loader)
        train_phys /= len(train_loader)
        train_bnd /= len(train_loader)
        
        model.eval()
        val_loss, val_data, val_phys, val_bnd = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                s_norm = batch["state_normalized"].to(device)
                a_norm = batch["action_normalized"].to(device)
                s_raw = batch["state_raw"].to(device)
                a_raw = batch["action_raw"].to(device)
                d_norm_target = batch["delta_normalized"].to(device)
                
                pred_d_norm = model(s_norm, a_norm)
                data_loss = F.mse_loss(pred_d_norm, d_norm_target)
                
                if model_type == "PINN":
                    pred_d_raw_tensor = pred_d_norm * torch.tensor(delta_scaler.std, device=device) + torch.tensor(delta_scaler.mean, device=device)
                    pred_next_state_raw = s_raw + pred_d_raw_tensor
                    phys_loss = compute_physics_loss(s_raw, a_raw, pred_d_raw_tensor)
                    bnd_loss = compute_bounds_loss(pred_next_state_raw)
                    loss = data_loss + SURROGATE_CONFIG["physics_lambda"] * phys_loss + SURROGATE_CONFIG["bounds_lambda"] * bnd_loss
                else:
                    loss = data_loss
                    phys_loss = torch.tensor(0.0)
                    bnd_loss = torch.tensor(0.0)
                    
                val_loss += loss.item()
                val_data += data_loss.item()
                val_phys += phys_loss.item()
                val_bnd += bnd_loss.item()
                
        val_loss /= len(val_loader)
        val_data /= len(val_loader)
        val_phys /= len(val_loader)
        val_bnd /= len(val_loader)
        
        epoch_time = time.time() - t0
        
        history.append({
            "model_type": model_type,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_data_loss": train_data,
            "train_physics_loss": train_phys,
            "train_bounds_loss": train_bnd,
            "val_data_loss": val_data,
            "val_physics_loss": val_phys,
            "val_bounds_loss": val_bnd,
            "epoch_time_seconds": epoch_time
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= SURROGATE_CONFIG["early_stopping_patience"]:
            break
            
    model.load_state_dict(best_state_dict)
    
    os.makedirs(os.path.join(DAY2_OUTPUT_DIR, "models"), exist_ok=True)
    save_path = os.path.join(DAY2_OUTPUT_DIR, "models", f"{model_type.lower()}_surrogate_{model_idx}.pt")
    # Also save to the original un-indexed name for backwards compatibility
    if model_idx == 0:
        torch.save({"model_type": model_type, "model_state_dict": best_state_dict, "config": SURROGATE_CONFIG, "best_metrics": {"val_loss": best_val_loss}}, os.path.join(DAY2_OUTPUT_DIR, "models", f"{model_type.lower()}_surrogate.pt"))
    
    torch.save({
        "model_type": model_type,
        "model_state_dict": best_state_dict,
        "config": SURROGATE_CONFIG,
        "best_metrics": {"val_loss": best_val_loss},
    }, save_path)
    
    return history
