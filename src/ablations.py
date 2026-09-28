import os
import time
import json
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from src.rl_env import RLCSTRGymEnv
from src.dyna_buffer import DynaRolloutEngine
from src.metrics import compute_metrics
from src.scenarios import SCENARIOS

# Custom callback for ablations evaluation
class AblationEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=2000, n_eval_episodes=5):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self):
        return True

def run_ablations_day6(project_root):
    out_dir = os.path.join(project_root, "outputs/day6/ablations")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    
    seeds = [42, 123, 2024]
    
    pinn_path = os.path.join(project_root, "outputs/day2/models/pinn_surrogate.pt")
    mlp_path = os.path.join(project_root, "outputs/day2/models/mlp_surrogate.pt")
    
    from src.train_baselines import EvalCallback, DynaCallback
    
    # 1. PI-Dyna-NoShield
    print("Running PI-Dyna-NoShield...")
    for seed in seeds:
        model_path = os.path.join(out_dir, "models", f"pi_dyna_no_shield_seed_{seed}.zip")
        if os.path.exists(model_path):
            print(f"Skipping PI-Dyna-NoShield seed {seed}, already exists.")
            continue
        env = RLCSTRGymEnv(seed=seed)
        model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto", train_freq=1, gradient_steps=1, target_update_interval=1, seed=seed, verbose=0)
        engine = DynaRolloutEngine(pinn_path, device="cuda" if torch.cuda.is_available() else "cpu")
        dyna_cb = DynaCallback(engine, num_synthetic_per_real=5, use_shielding=False) # Shielding disabled
        model.learn(total_timesteps=50000, callback=[dyna_cb])
        model.save(model_path)

    # 2. PI-NoDyna-Shield
    print("Running PI-NoDyna-Shield...")
    from src.world_model import SurrogateWorldModel
    for seed in seeds:
        model_path = os.path.join(out_dir, "models", f"pi_no_dyna_shield_seed_{seed}.zip")
        if os.path.exists(model_path):
            print(f"Skipping PI-NoDyna-Shield seed {seed}, already exists.")
            continue
        env = RLCSTRGymEnv(seed=seed)
        # We need to hack the env step to use shielding from Surrogate!
        wm = SurrogateWorldModel(pinn_path, device="cuda" if torch.cuda.is_available() else "cpu")
        
        class ShieldedEnv(RLCSTRGymEnv):
            def step(self, action):
                obs, r, d, trunc, info = super().step(action)
                # Apply shielding penalty
                raw_C_A = self.state[0]
                raw_T = self.state[1]
                # Actually, shielding uses current state and predicts next.
                s_t = torch.tensor([[raw_C_A, raw_T]], dtype=torch.float32)
                T_c = 325.0 + 75.0 * float(np.clip(action, -1.0, 1.0)[0])
                a_t = torch.tensor([[T_c]], dtype=torch.float32)
                with torch.no_grad():
                    next_s = wm.predict_next_state(s_t, a_t)
                pred_C_A = next_s[0, 0].item()
                pred_T = next_s[0, 1].item()
                if pred_T > 370.0 or pred_C_A < 0:
                    r -= 50.0
                return obs, r, d, trunc, info
                
        s_env = ShieldedEnv(seed=seed)
        model = SAC("MlpPolicy", s_env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto", train_freq=1, gradient_steps=1, target_update_interval=1, seed=seed, verbose=0)
        model.learn(total_timesteps=50000)
        model.save(model_path)
        
    # 3. MLP-Dyna-Shield
    print("Running MLP-Dyna-Shield...")
    for seed in seeds:
        model_path = os.path.join(out_dir, "models", f"mlp_dyna_shield_seed_{seed}.zip")
        if os.path.exists(model_path):
            print(f"Skipping MLP-Dyna-Shield seed {seed}, already exists.")
            continue
        env = RLCSTRGymEnv(seed=seed)
        model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]), learning_rate=3e-4, buffer_size=100000, batch_size=256, gamma=0.99, tau=0.005, ent_coef="auto", train_freq=1, gradient_steps=1, target_update_interval=1, seed=seed, verbose=0)
        engine = DynaRolloutEngine(mlp_path, device="cuda" if torch.cuda.is_available() else "cpu")
        dyna_cb = DynaCallback(engine, num_synthetic_per_real=5, use_shielding=True) # Shielding enabled
        model.learn(total_timesteps=50000, callback=[dyna_cb])
        model.save(model_path)

    # Evaluation
    print("Evaluating Ablations...")
    eval_scenarios = ["nominal_fixed", "feed_ca_disturbance", "sensor_noise", "actuator_delay"]
    methods = ["PI-Dyna-NoShield", "PI-NoDyna-Shield", "MLP-Dyna-Shield"]
    records = []
    
    for method in methods:
        for seed in seeds:
            if method == "PI-Dyna-NoShield":
                model = SAC.load(os.path.join(out_dir, "models", f"pi_dyna_no_shield_seed_{seed}.zip"))
            elif method == "PI-NoDyna-Shield":
                model = SAC.load(os.path.join(out_dir, "models", f"pi_no_dyna_shield_seed_{seed}.zip"))
            elif method == "MLP-Dyna-Shield":
                model = SAC.load(os.path.join(out_dir, "models", f"mlp_dyna_shield_seed_{seed}.zip"))
                
            for scenario in eval_scenarios:
                env = RLCSTRGymEnv(scenario_name=scenario, seed=seed)
                for ep in range(10):
                    obs, _ = env.reset(seed=seed + ep)
                    done = False
                    T_hist, sp_hist, a_hist, ca_hist = [], [], [], []
                    ep_reward = 0
                    
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _, info = env.step(action)
                        ep_reward += reward
                        T_hist.append(info["raw_T"])
                        sp_hist.append(env._get_setpoint())
                        a_hist.append(info["raw_T_c"])
                        ca_hist.append(info["raw_C_A"])
                        
                    met = compute_metrics(T_hist, sp_hist, a_hist, ca_hist, disturbance_step=SCENARIOS[scenario]["disturbance_step"])
                    record = {
                        "method": method,
                        "seed": seed,
                        "scenario": scenario,
                        "episode_id": ep,
                        "cumulative_reward": ep_reward,
                        "iae_T": met["iae_T"],
                        "constraint_violation_rate": met["constraint_violation_rate"],
                        "max_T": met["max_T"],
                        "control_effort": met["control_effort"]
                    }
                    records.append(record)
                    
    df_eval = pd.DataFrame(records)
    df_eval.to_csv(os.path.join(out_dir, "logs", "ablation_evaluation.csv"), index=False)
    
    df_sum = df_eval.groupby(["method", "scenario"]).mean(numeric_only=True).reset_index()
    # rename columns
    df_sum.rename(columns={
        "cumulative_reward": "mean_cumulative_reward",
        "iae_T": "mean_iae_T",
        "constraint_violation_rate": "mean_constraint_violation_rate",
        "max_T": "mean_max_T",
        "control_effort": "mean_control_effort"
    }, inplace=True)
    df_sum.to_csv(os.path.join(out_dir, "logs", "ablation_summary.csv"), index=False)
    
    with open(os.path.join(out_dir, "logs", "ablation_report.json"), "w") as f:
        json.dump({"status": "SUCCESS"}, f)
