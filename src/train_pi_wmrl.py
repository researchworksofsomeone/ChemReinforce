import os
import torch
import pandas as pd
from stable_baselines3 import SAC
from src.rl_env import RLCSTRGymEnv
from src.dyna_buffer import DynaRolloutEngine
from src.train_baselines import EvalCallback, DynaCallback
from src.config import DAY4_OUTPUT_DIR

def train_pi_dyna_sac(seed, surrogate_path):
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
    
    log_path = os.path.join(DAY4_OUTPUT_DIR, "logs", "learning_curves.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    eval_cb = EvalCallback(eval_env, log_path=log_path, method_name="PI-Dyna-SAC", seed=seed)
    
    engine = DynaRolloutEngine(surrogate_path, device="cuda" if torch.cuda.is_available() else "cpu")
    dyna_cb = DynaCallback(engine, num_synthetic_per_real=5, use_shielding=True)
    
    # Custom logger for constraint shielding stats
    class ShieldingLoggerCallback(DynaCallback):
        def __init__(self, dyna_cb_ref, log_path, seed):
            super().__init__(dyna_cb_ref.dyna_engine, 0, False)
            self.dyna_cb = dyna_cb_ref
            self.log_path = log_path
            self.seed = seed
            self.records = []
            
        def _on_step(self):
            if self.num_timesteps % 2000 == 0:
                self.records.append({
                    "seed": self.seed,
                    "timestep": self.num_timesteps,
                    "synthetic_transitions_added": self.dyna_cb.synthetic_transitions_added,
                    "shielded_transitions": self.dyna_cb.shielded_count,
                    "shield_rate": self.dyna_cb.shielded_count / max(1, self.dyna_cb.synthetic_transitions_added),
                    "mean_synthetic_reward": 0.0 # Placeholder
                })
                df = pd.DataFrame(self.records)
                if os.path.exists(self.log_path):
                    df_old = pd.read_csv(self.log_path)
                    df = pd.concat([df_old, pd.DataFrame([self.records[-1]])], ignore_index=True)
                df.to_csv(self.log_path, index=False)
            return True
            
    shield_log_path = os.path.join(DAY4_OUTPUT_DIR, "logs", "constraint_shielding_stats.csv")
    shield_cb = ShieldingLoggerCallback(dyna_cb, shield_log_path, seed)
    
    model.learn(total_timesteps=50000, callback=[eval_cb, dyna_cb, shield_cb])
    
    os.makedirs(os.path.join(DAY4_OUTPUT_DIR, "models"), exist_ok=True)
    model.save(os.path.join(DAY4_OUTPUT_DIR, "models", f"pi_dyna_sac_seed_{seed}.zip"))
    
    return model, dyna_cb.synthetic_transitions_added, dyna_cb.shielded_count
