import os
import json
import time
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from src.rl_env import RLCSTRGymEnv
from src.dyna_buffer import DynaRolloutEngine
from src.config import DAY3_OUTPUT_DIR

class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=2000, n_eval_episodes=5, log_path=None, method_name="", seed=42):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.method_name = method_name
        self.seed_val = seed
        self.records = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            rewards = []
            lengths = []
            violations = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_len = 0
                ep_viol = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, done, _, info = self.eval_env.step(action)
                    ep_reward += r
                    ep_len += 1
                    if info.get("constraint_violation", False):
                        ep_viol += 1
                rewards.append(ep_reward)
                lengths.append(ep_len)
                violations.append(ep_viol)
                
            self.records.append({
                "timestep": self.num_timesteps,
                "method": self.method_name,
                "seed": self.seed_val,
                "mean_reward": np.mean(rewards),
                "mean_constraint_violations": np.mean(violations),
                "mean_episode_length": np.mean(lengths)
            })
            
            if self.log_path is not None:
                df = pd.DataFrame(self.records)
                if os.path.exists(self.log_path):
                    df_old = pd.read_csv(self.log_path)
                    df = pd.concat([df_old, pd.DataFrame([self.records[-1]])], ignore_index=True)
                df.to_csv(self.log_path, index=False)
                
        return True

class DynaCallback(BaseCallback):
    def __init__(self, dyna_engine, num_synthetic_per_real=5, use_shielding=False):
        super().__init__()
        self.dyna_engine = dyna_engine
        self.num_synthetic_per_real = num_synthetic_per_real
        self.use_shielding = use_shielding
        self.synthetic_transitions_added = 0
        self.shielded_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 100 == 0:
            replay_buffer = self.model.replay_buffer
            if replay_buffer.size() >= 1000:
                # Calculate synthetic ratio curriculum
                if self.num_timesteps < 20000:
                    current_ratio = int(self.num_synthetic_per_real * (self.num_timesteps / 20000.0))
                else:
                    current_ratio = self.num_synthetic_per_real
                    
                if current_ratio == 0:
                    return True
                
                # Fetch q90 threshold
                q90 = self.dyna_engine.world_model.q90_threshold
                if q90 is None:
                    # fallback dataset path
                    ds_path = os.path.join(DAY3_OUTPUT_DIR, "..", "day1", "results", "cstr_transitions.pt")
                    q90 = self.dyna_engine.world_model.compute_q90(ds_path)
                
                # Sample 100 states
                batch = replay_buffer.sample(100)
                obs_batch = batch.observations.cpu().numpy()
                
                curr_obs = obs_batch
                for step in range(current_ratio):
                    # Sample actions from policy with noise
                    actions, _ = self.model.predict(curr_obs, deterministic=False)
                    if len(actions.shape) == 1:
                        actions = actions.reshape(-1, 1)
                        
                    _, _, rewards, next_obs, dones, infos = self.dyna_engine.generate_synthetic_transitions(
                        curr_obs, actions, 1, self.use_shielding
                    )
                    
                    for i in range(len(curr_obs)):
                        # Uncertainty gating
                        if infos[i]["T_std"] < q90:
                            replay_buffer.add(
                                curr_obs[i],
                                next_obs[i],
                                actions[i],
                                rewards[i],
                                dones[i],
                                [infos[i]]
                            )
                            self.synthetic_transitions_added += 1
                            if infos[i].get("shielded", False):
                                self.shielded_count += 1
                            
                    curr_obs = next_obs
                    
        return True

def train_pid():
    os.makedirs(os.path.join(DAY3_OUTPUT_DIR, "models"), exist_ok=True)
    config = {"Kp": 2.0, "Ki": 0.01, "Kd": 0.1}
    with open(os.path.join(DAY3_OUTPUT_DIR, "models", "pid_controller.json"), "w") as f:
        json.dump(config, f, indent=4)

def train_mf_sac(seed):
    env = RLCSTRGymEnv(seed=seed)
    eval_env = RLCSTRGymEnv(seed=seed+100)
    
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        seed=seed,
        verbose=0
    )
    
    log_path = os.path.join(DAY3_OUTPUT_DIR, "logs", "learning_curves.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    eval_cb = EvalCallback(eval_env, log_path=log_path, method_name="MF-SAC", seed=seed)
    
    model.learn(total_timesteps=50000, callback=eval_cb)
    
    os.makedirs(os.path.join(DAY3_OUTPUT_DIR, "models"), exist_ok=True)
    model.save(os.path.join(DAY3_OUTPUT_DIR, "models", f"mf_sac_seed_{seed}.zip"))
    return model

def train_mlp_dyna_sac(seed, surrogate_path):
    env = RLCSTRGymEnv(seed=seed)
    eval_env = RLCSTRGymEnv(seed=seed+100)
    
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1,
        seed=seed,
        verbose=0
    )
    
    log_path = os.path.join(DAY3_OUTPUT_DIR, "logs", "learning_curves.csv")
    eval_cb = EvalCallback(eval_env, log_path=log_path, method_name="MLP-Dyna-SAC", seed=seed)
    
    engine = DynaRolloutEngine(surrogate_path, device="cuda" if torch.cuda.is_available() else "cpu")
    dyna_cb = DynaCallback(engine, num_synthetic_per_real=5, use_shielding=False)
    
    model.learn(total_timesteps=50000, callback=[eval_cb, dyna_cb])
    
    os.makedirs(os.path.join(DAY3_OUTPUT_DIR, "models"), exist_ok=True)
    model.save(os.path.join(DAY3_OUTPUT_DIR, "models", f"mlp_dyna_sac_seed_{seed}.zip"))
    
    return model, dyna_cb.synthetic_transitions_added
