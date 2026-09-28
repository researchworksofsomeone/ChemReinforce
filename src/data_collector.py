import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from .cstr_env import CSTRGymEnv
from .pid_controller import PIDController
from .config import DATA_CONFIG, CSTR_CONFIG, DAY1_OUTPUT_DIR

def collect_data(num_random, num_pid):
    env = CSTRGymEnv()
    pid = PIDController(setpoint=CSTR_CONFIG["T_setpoint"], output_limits=(CSTR_CONFIG["T_c_min"], CSTR_CONFIG["T_c_max"]))
    
    transitions = []
    
    print("Collecting Random transitions...")
    state, _ = env.reset()
    for _ in tqdm(range(num_random)):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        transitions.append({
            "state_C_A": state[0],
            "state_T": state[1],
            "action_T_c": action[0],
            "reward": reward,
            "next_state_C_A": next_state[0],
            "next_state_T": next_state[1],
            "done": terminated or truncated,
            "fault_mode": "nominal",
            "policy_type": "random",
            "episode_id": 0,
            "step_id": 0,
            "timestamp": 0.0
        })
        
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
            
    print("Collecting PID transitions...")
    state, _ = env.reset()
    pid.reset()
    for _ in tqdm(range(num_pid)):
        base_action = pid.compute(state[1], dt=CSTR_CONFIG["dt"])
        noise = np.random.normal(0, 5.0)
        action = np.array([np.clip(base_action + noise, CSTR_CONFIG["T_c_min"], CSTR_CONFIG["T_c_max"])], dtype=np.float32)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        transitions.append({
            "state_C_A": state[0],
            "state_T": state[1],
            "action_T_c": action[0],
            "reward": reward,
            "next_state_C_A": next_state[0],
            "next_state_T": next_state[1],
            "done": terminated or truncated,
            "fault_mode": "nominal",
            "policy_type": "pid",
            "episode_id": 0,
            "step_id": 0,
            "timestamp": 0.0
        })
        
        if terminated or truncated:
            state, _ = env.reset()
            pid.reset()
        else:
            state = next_state

    total = num_random + num_pid
    transitions = transitions[:total]
    
    df = pd.DataFrame(transitions)
    
    out_dir = os.path.join(DAY1_OUTPUT_DIR, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    df.to_parquet(os.path.join(out_dir, "cstr_transitions.parquet"))
    
    tensors = {
        "state_C_A": torch.tensor(df["state_C_A"].values, dtype=torch.float32),
        "state_T": torch.tensor(df["state_T"].values, dtype=torch.float32),
        "action_T_c": torch.tensor(df["action_T_c"].values, dtype=torch.float32),
        "reward": torch.tensor(df["reward"].values, dtype=torch.float32),
        "next_state_C_A": torch.tensor(df["next_state_C_A"].values, dtype=torch.float32),
        "next_state_T": torch.tensor(df["next_state_T"].values, dtype=torch.float32),
        "done": torch.tensor(df["done"].values, dtype=torch.float32)
    }
    torch.save(tensors, os.path.join(out_dir, "cstr_transitions.pt"))
    
    return df
