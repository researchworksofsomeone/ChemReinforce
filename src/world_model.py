import torch
import numpy as np
import os
from .surrogate_models import SurrogateModel
from .utils import get_device

class SurrogateWorldModel:
    def __init__(self, checkpoint_path, device=None, ensemble_size=3):
        self.device = device or get_device()
        self.models = []
        
        # Load up to ensemble_size models
        base_dir = os.path.dirname(checkpoint_path)
        base_name = os.path.basename(checkpoint_path)
        
        for i in range(ensemble_size):
            # Check for _i.pt first
            p_i = os.path.join(base_dir, base_name.replace(".pt", f"_{i}.pt"))
            if not os.path.exists(p_i):
                p_i = checkpoint_path # fallback to the original model if ensemble member missing
            if os.path.exists(p_i):
                checkpoint = torch.load(p_i, map_location=self.device, weights_only=True)
                model = SurrogateModel().to(self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.models.append(model)
                
        if len(self.models) == 0:
            raise FileNotFoundError(f"Could not find any models at {checkpoint_path}")
            
        scalers_path = os.path.join(base_dir, "scalers.pt")
        scalers = torch.load(scalers_path, map_location=self.device, weights_only=True)
        
        self.s_mean = scalers["state_scaler_mean"].to(self.device)
        self.s_std = scalers["state_scaler_std"].to(self.device)
        self.a_mean = scalers["action_scaler_mean"].to(self.device)
        self.a_std = scalers["action_scaler_std"].to(self.device)
        self.d_mean = scalers["delta_scaler_mean"].to(self.device)
        self.d_std = scalers["delta_scaler_std"].to(self.device)
        
        self.q90_threshold = None
        
    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        return x.to(self.device)
        
    def predict_next_state(self, state, action):
        next_state, _ = self.predict_ensemble(state, action)
        return next_state
        
    def predict_ensemble(self, state, action):
        state = self._to_tensor(state)
        action = self._to_tensor(action)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            
        s_norm = (state - self.s_mean) / self.s_std
        a_norm = (action - self.a_mean) / self.a_std
        
        preds = []
        with torch.no_grad():
            for model in self.models:
                pred_d_norm = model(s_norm, a_norm)
                pred_d_raw = pred_d_norm * self.d_std + self.d_mean
                preds.append(pred_d_raw)
                
        preds = torch.stack(preds, dim=0) # [ensemble_size, batch, state_dim]
        mean_pred = torch.mean(preds, dim=0)
        std_pred = torch.std(preds, dim=0)
        
        next_state = state + mean_pred
        
        next_state[:, 0] = torch.clamp(next_state[:, 0], 0.0, 2.0)
        next_state[:, 1] = torch.clamp(next_state[:, 1], 250.0, 500.0)
        
        # We use standard deviation of T for gating
        T_std = std_pred[:, 1]
        
        return next_state, T_std

    def rollout(self, state, action_sequence):
        states = [state]
        curr_state = state
        for action in action_sequence:
            curr_state = self.predict_next_state(curr_state, action)
            states.append(curr_state)
        return torch.stack(states, dim=1)

    def predict_constraint_risk(self, state, action_sequence, T_max):
        rollout_states = self.rollout(state, action_sequence)
        T_trajectory = rollout_states[:, :, 1]
        risk = (T_trajectory > T_max).float().max(dim=1)[0]
        return risk
        
    def compute_q90(self, dataset_path):
        if not os.path.exists(dataset_path):
            return 1.0 # fallback
        data = torch.load(dataset_path, map_location=self.device, weights_only=True)
        states = data["states"][:10000]
        actions = data["actions"][:10000]
        _, T_std = self.predict_ensemble(states, actions)
        self.q90_threshold = torch.quantile(T_std, 0.90).item()
        return self.q90_threshold
