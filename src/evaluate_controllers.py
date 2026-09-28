import os
import time
import json
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from src.rl_env import RLCSTRGymEnv
from src.scenarios import SCENARIOS
from src.metrics import compute_metrics
from src.config import DAY3_OUTPUT_DIR, DAY4_OUTPUT_DIR, DAY5_OUTPUT_DIR
from src.pid_controller import PIDController

def evaluate_all():
    methods = ["PID", "MF-SAC", "MLP-Dyna-SAC", "PI-Dyna-SAC"]
    seeds = [42, 123, 2024]
    scenarios = list(SCENARIOS.keys())
    episodes_per_scenario = 20
    
    os.makedirs(os.path.join(DAY5_OUTPUT_DIR, "results"), exist_ok=True)
    
    records = []
    
    for method in methods:
        for seed in seeds:
            if method == "PID":
                # Load tuned PID
                pid_path = os.path.join(DAY3_OUTPUT_DIR, "models", "pid_controller_tuned.json")
                if os.path.exists(pid_path):
                    with open(pid_path, "r") as f:
                        params = json.load(f)
                    model = PIDController(Kp=params["Kp"], Ki=params["Ki"], Kd=params["Kd"])
                else:
                    model = PIDController()
            elif method == "MF-SAC":
                path = os.path.join(DAY3_OUTPUT_DIR, "models", f"mf_sac_seed_{seed}.zip")
                model = SAC.load(path)
            elif method == "MLP-Dyna-SAC":
                path = os.path.join(DAY3_OUTPUT_DIR, "models", f"mlp_dyna_sac_seed_{seed}.zip")
                # Load MLP Dyna v2 from day6 ablations if it exists, otherwise fallback
                v2_path = os.path.join(DAY4_OUTPUT_DIR, "models", f"mlp_dyna_v2_seed_{seed}.zip")
                if os.path.exists(v2_path):
                    model = SAC.load(v2_path)
                else:
                    model = SAC.load(path)
            elif method == "PI-Dyna-SAC":
                path = os.path.join(DAY4_OUTPUT_DIR, "models", f"pi_dyna_sac_seed_{seed}.zip")
                # Load PI Dyna v2 from day4 if it exists, otherwise fallback
                v2_path = os.path.join(DAY4_OUTPUT_DIR, "models", f"pi_dyna_v2_seed_{seed}.zip")
                if os.path.exists(v2_path):
                    model = SAC.load(v2_path)
                else:
                    model = SAC.load(path)
                
            for scenario in scenarios:
                # Use hash to guarantee different initialization per scenario! (B3 fix)
                env_seed = seed + hash(scenario) % 10000
                env = RLCSTRGymEnv(scenario_name=scenario, seed=env_seed)
                
                for ep in range(episodes_per_scenario):
                    t_start = time.time()
                    obs, _ = env.reset(seed=env_seed + ep)
                    if method == "PID":
                        model.reset()
                        
                    done = False
                    
                    T_history = []
                    sp_history = []
                    a_history = []
                    C_A_history = []
                    
                    ep_reward = 0
                    
                    while not done:
                        if method == "PID":
                            sp = env._get_setpoint()
                            T_raw = obs[1] * 50.0 + 350.0
                            error = sp - T_raw # PID B1 reverse acting
                            model.integral += error
                            derivative = error - model.prev_error
                            u = model.Kp * error + model.Ki * model.integral + model.Kd * derivative
                            model.prev_error = error
                            prev_T_c = obs[2] * 75.0 + 325.0
                            T_c = prev_T_c + u
                            action = (T_c - 325.0) / 75.0
                            action = np.array([action], dtype=np.float32)
                        else:
                            action, _ = model.predict(obs, deterministic=True)
                            
                        obs, reward, done, _, info = env.step(action)
                        ep_reward += reward
                        
                        T_history.append(info["raw_T"])
                        sp_history.append(env._get_setpoint())
                        a_history.append(info["raw_T_c"])
                        C_A_history.append(info["raw_C_A"])
                        
                    # Save trajectories for seed 42 (B2 fix)
                    if seed == 42:
                        traj_df = pd.DataFrame({
                            "method": method,
                            "scenario": scenario,
                            "episode_id": ep,
                            "step": np.arange(len(T_history)),
                            "T": T_history,
                            "C_A": C_A_history,
                            "T_c": a_history,
                            "T_setpoint": sp_history
                        })
                        traj_path = os.path.join(DAY5_OUTPUT_DIR, "results", "trajectories_seed42.csv")
                        hdr = not os.path.exists(traj_path)
                        traj_df.to_csv(traj_path, mode='a', header=hdr, index=False)
                    
                    runtime = time.time() - t_start
                    
                    met = compute_metrics(
                        T_history, sp_history, a_history, C_A_history,
                        disturbance_step=SCENARIOS[scenario]["disturbance_step"]
                    )
                    
                    record = {
                        "method": method,
                        "seed": seed,
                        "scenario": scenario,
                        "episode_id": ep,
                        "cumulative_reward": ep_reward,
                        "episode_length": len(T_history),
                        "runtime_seconds": runtime
                    }
                    record.update(met)
                    records.append(record)
                    
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DAY5_OUTPUT_DIR, "results", "episode_metrics.csv"), index=False)
    
    agg = df.groupby(["method", "scenario"]).mean(numeric_only=True).reset_index()
    agg.to_csv(os.path.join(DAY5_OUTPUT_DIR, "results", "aggregate_metrics.csv"), index=False)
    
    # B3 Assertion
    for method in methods:
        nom_rows = agg[(agg["method"] == method) & (agg["scenario"] == "nominal_fixed")]
        set_rows = agg[(agg["method"] == method) & (agg["scenario"] == "setpoint_steps")]
        if not nom_rows.empty and not set_rows.empty:
            nom_mt = nom_rows["max_T"].values[0]
            set_mt = set_rows["max_T"].values[0]
            assert nom_mt != set_mt, f"B3 Bug: {method} setpoint_steps stats identical to nominal_fixed ({nom_mt})"

if __name__ == "__main__":
    evaluate_all()
