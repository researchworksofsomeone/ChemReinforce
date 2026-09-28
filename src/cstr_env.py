import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from .config import CSTR_CONFIG, ENV_CONFIG
from .cstr_dynamics import BatchedCSTRDynamics

class CSTRGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.dynamics = BatchedCSTRDynamics(self.device)
        
        self.observation_space = spaces.Box(
            low=np.array([CSTR_CONFIG["C_A_min"], 250.0], dtype=np.float32),
            high=np.array([CSTR_CONFIG["C_A_max"], 500.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([CSTR_CONFIG["T_c_min"]], dtype=np.float32),
            high=np.array([CSTR_CONFIG["T_c_max"]], dtype=np.float32),
            shape=(1,), dtype=np.float32
        )
        
        self.state = None
        self.prev_action = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            
        C_A = self.np_random.uniform(ENV_CONFIG["reset_C_A_low"], ENV_CONFIG["reset_C_A_high"])
        T = self.np_random.uniform(ENV_CONFIG["reset_T_low"], ENV_CONFIG["reset_T_high"])
        
        self.state = np.array([C_A, T], dtype=np.float32)
        self.prev_action = np.array([350.0], dtype=np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        state_tensor = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        
        next_state_tensor = self.dynamics.step(state_tensor, action_tensor)
        next_state = next_state_tensor.squeeze(0).numpy()
        
        T = next_state[1]
        T_setpoint = CSTR_CONFIG["T_setpoint"]
        u_t = action[0]
        u_t_minus_1 = self.prev_action[0]
        
        tracking_penalty = ((T - T_setpoint) / 50.0)**2
        control_penalty = ((u_t - u_t_minus_1) / 25.0)**2
        constraint_penalty = max(0, T - CSTR_CONFIG["T_max"])**2 / 100.0 + max(0, CSTR_CONFIG["T_min"] - T)**2 / 100.0
        
        reward = -(
            ENV_CONFIG["reward_weights"]["tracking"] * tracking_penalty +
            ENV_CONFIG["reward_weights"]["control_effort"] * control_penalty +
            ENV_CONFIG["reward_weights"]["constraint_violation"] * constraint_penalty
        )
        
        self.state = next_state
        self.prev_action = action
        self.step_count += 1
        
        terminated = False
        truncated = self.step_count >= ENV_CONFIG["episode_length"]
        
        return self.state.copy(), float(reward), terminated, truncated, {}
