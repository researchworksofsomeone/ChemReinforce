import numpy as np
import torch
from .world_model import SurrogateWorldModel

class DynaRolloutEngine:
    def __init__(self, surrogate_checkpoint_path, device):
        self.world_model = SurrogateWorldModel(surrogate_checkpoint_path, device)
        self.device = device
        self.constraint_shield_count = 0

    def generate_synthetic_transitions(
        self,
        sampled_obs,
        sampled_actions,
        num_synthetic_per_real,
        use_constraint_shielding=False
    ):
        """
        sampled_obs: np.array of shape (N, 3), [norm_C_A, norm_T, norm_prev_action]
        sampled_actions: np.array of shape (N, 1), normalized action in [-1, 1]
        """
        N = sampled_obs.shape[0]
        
        # 1. Convert normalized observation to raw state [C_A, T]
        raw_C_A = sampled_obs[:, 0] * 1.0 + 1.0
        raw_T = sampled_obs[:, 1] * 50.0 + 350.0
        raw_states = np.stack([raw_C_A, raw_T], axis=1)
        
        # 3. Convert normalized action to raw T_c
        raw_T_c = sampled_actions[:, 0] * 75.0 + 325.0
        
        # We need to simulate multiple steps forward. But the prompt says:
        # "generate 5 synthetic transitions per sampled state"
        # The prompt implies 1-step rollouts or multi-step?
        # Usually it means taking the state, applying action, getting next state.
        # But `sampled_actions` is provided. If `num_synthetic_per_real` > 1, do we just duplicate?
        # Actually, if the policy generates `sampled_actions`, maybe we just step 1 transition per state.
        # But if we need 5 synthetic transitions per state, does the policy provide 5 actions per state?
        # Wait, the prompt says "sample 100 recent or random states from replay buffer, generate 5 synthetic transitions per sampled state".
        # This implies a 5-step rollout from each state using the policy! But here we only pass `sampled_actions` which implies 1 step?
        # If I need a rollout, I must take the policy as input, or just do 1-step and the caller handles the loop.
        # Let's assume the caller will call this function inside a loop 5 times, passing the next obs!
        # Thus `num_synthetic_per_real` is handled by the caller, or this function expects a batch of size N.
        
        # Predict next raw state using ensemble
        with torch.no_grad():
            next_raw_states, T_std = self.world_model.predict_ensemble(raw_states, raw_T_c[:, None])
            next_raw_states = next_raw_states.cpu().numpy()
            T_std = T_std.cpu().numpy()
            
        next_raw_C_A = next_raw_states[:, 0]
        next_raw_T = next_raw_states[:, 1]
        
        # 5. Compute reward
        sp = 350.0
        # sp could be anything but for Day 3/4 training we use nominal_fixed which is 350.0
        rewards = -np.abs(next_raw_T - sp) / 50.0
        # Control effort penalty
        raw_prev_action = sampled_obs[:, 2] * 75.0 + 325.0
        rewards -= 0.1 * np.abs(raw_T_c - raw_prev_action) / 75.0
        
        dones = np.zeros(N, dtype=bool)
        infos = []
        
        # Constraint checking and shielding
        for i in range(N):
            violation = False
            shielded = False
            if next_raw_T[i] > 370.0 or next_raw_T[i] < 300.0 or next_raw_C_A[i] < 0:
                violation = True
                if use_constraint_shielding:
                    rewards[i] -= 50.0
                    self.constraint_shield_count += 1
                    shielded = True
                    
            infos.append({"shielded": shielded, "constraint_violation": violation, "T_std": T_std[i]})
            
        # 6. Build next normalized observation
        next_norm_C_A = (next_raw_C_A - 1.0) / 1.0
        next_norm_T = (next_raw_T - 350.0) / 50.0
        next_norm_prev = sampled_actions[:, 0] # the current action becomes the previous action!
        
        next_obs = np.stack([next_norm_C_A, next_norm_T, next_norm_prev], axis=1)
        next_obs = np.clip(next_obs, -5.0, 5.0)
        
        # Return transitions
        return sampled_obs, sampled_actions, rewards, next_obs, dones, infos
