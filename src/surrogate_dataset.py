import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .config import DAY1_OUTPUT_DIR, DAY2_OUTPUT_DIR, DATA_CONFIG

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        self.std[self.std < 1e-8] = 1.0
        
    def transform(self, data):
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data):
        return data * self.std + self.mean

class SurrogateDataset(Dataset):
    def __init__(self, state_raw, action_raw, next_state_raw, state_scaler, action_scaler, delta_scaler):
        self.state_raw = state_raw
        self.action_raw = action_raw
        self.next_state_raw = next_state_raw
        self.delta_raw = next_state_raw - state_raw
        
        self.state_normalized = state_scaler.transform(self.state_raw)
        self.action_normalized = action_scaler.transform(self.action_raw)
        self.delta_normalized = delta_scaler.transform(self.delta_raw)
        
    def __len__(self):
        return len(self.state_raw)
        
    def __getitem__(self, idx):
        return {
            "state_normalized": self.state_normalized[idx],
            "action_normalized": self.action_normalized[idx],
            "state_raw": self.state_raw[idx],
            "action_raw": self.action_raw[idx],
            "delta_normalized": self.delta_normalized[idx],
            "delta_raw": self.delta_raw[idx],
            "next_state_raw": self.next_state_raw[idx]
        }

def prepare_dataloaders():
    df = pd.read_parquet(os.path.join(DAY1_OUTPUT_DIR, "dataset", "cstr_transitions.parquet"))
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["state_C_A"] = df["state_C_A"].clip(0.0, 2.0)
    df["state_T"] = df["state_T"].clip(250.0, 500.0)
    df["action_T_c"] = df["action_T_c"].clip(250.0, 400.0)
    
    total = len(df)
    
    state = torch.tensor(df[["state_C_A", "state_T"]].values, dtype=torch.float32)
    action = torch.tensor(df[["action_T_c"]].values, dtype=torch.float32)
    next_state = torch.tensor(df[["next_state_C_A", "next_state_T"]].values, dtype=torch.float32)
    
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(42))
    
    n_train = int(DATA_CONFIG["train_split"] * total)
    n_val = int(DATA_CONFIG["val_split"] * total)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    state_scaler = Scaler()
    action_scaler = Scaler()
    delta_scaler = Scaler()
    
    train_state = state[train_idx]
    train_action = action[train_idx]
    train_next_state = next_state[train_idx]
    train_delta = train_next_state - train_state
    
    state_scaler.fit(train_state)
    action_scaler.fit(train_action)
    delta_scaler.fit(train_delta)
    
    os.makedirs(os.path.join(DAY2_OUTPUT_DIR, "models"), exist_ok=True)
    torch.save({
        "state_scaler_mean": state_scaler.mean,
        "state_scaler_std": state_scaler.std,
        "action_scaler_mean": action_scaler.mean,
        "action_scaler_std": action_scaler.std,
        "delta_scaler_mean": delta_scaler.mean,
        "delta_scaler_std": delta_scaler.std
    }, os.path.join(DAY2_OUTPUT_DIR, "models", "scalers.pt"))
    
    train_ds = SurrogateDataset(state[train_idx], action[train_idx], next_state[train_idx], state_scaler, action_scaler, delta_scaler)
    val_ds = SurrogateDataset(state[val_idx], action[val_idx], next_state[val_idx], state_scaler, action_scaler, delta_scaler)
    test_ds = SurrogateDataset(state[test_idx], action[test_idx], next_state[test_idx], state_scaler, action_scaler, delta_scaler)
    
    train_loader = DataLoader(train_ds, batch_size=DATA_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=DATA_CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=DATA_CONFIG["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader, state_scaler, action_scaler, delta_scaler
