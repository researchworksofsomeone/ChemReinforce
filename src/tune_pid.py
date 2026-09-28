import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import itertools
from src.pid_controller import PIDController
from src.rl_env import RLCSTRGymEnv
from src.metrics import compute_metrics
from src.config import DAY3_OUTPUT_DIR

def tune_pid():
    Kp_vals = [0.5, 1.0, 2.0, 3.0, 5.0]
    Ki_vals = [0.0, 0.001, 0.005, 0.01, 0.05]
    Kd_vals = [0.0, 0.05, 0.1, 0.3]
    
    best_score = float('inf')
    best_params = None
    best_metrics = None
    
    seeds = [42, 123, 2024]
    
    for Kp, Ki, Kd in itertools.product(Kp_vals, Ki_vals, Kd_vals):
        total_iae = 0.0
        total_viol = 0.0
        
        for seed in seeds:
            env = RLCSTRGymEnv(scenario_name="nominal_fixed", seed=seed)
            obs, _ = env.reset(seed=seed)
            
            pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd)
            pid.reset()
            
            T_history = []
            sp_history = []
            a_history = []
            C_A_history = []
            
            done = False
            
            while not done:
                sp = env._get_setpoint()
                T_raw = obs[1] * 50.0 + 350.0
                error = sp - T_raw
                pid.integral += error
                derivative = error - pid.prev_error
                u = pid.Kp * error + pid.Ki * pid.integral + pid.Kd * derivative
                pid.prev_error = error
                
                prev_T_c = obs[2] * 75.0 + 325.0
                T_c = prev_T_c + u
                action = (T_c - 325.0) / 75.0
                
                obs, reward, done, _, info = env.step([action])
                
                T_history.append(info["raw_T"])
                sp_history.append(env._get_setpoint())
                a_history.append(info["raw_T_c"])
                C_A_history.append(info["raw_C_A"])
                
            met = compute_metrics(T_history, sp_history, a_history, C_A_history)
            total_iae += met["iae_T"]
            total_viol += met["constraint_violation_rate"]
            
        avg_iae = total_iae / len(seeds)
        avg_viol = total_viol / len(seeds)
        
        score = avg_iae + 1000.0 * avg_viol
        
        if score < best_score:
            best_score = score
            best_params = {"Kp": Kp, "Ki": Ki, "Kd": Kd}
            best_metrics = {"iae": avg_iae, "viol": avg_viol}
            
    print(f"Best PID parameters: {best_params}")
    print(f"Metrics: IAE={best_metrics['iae']:.2f}, ViolRate={best_metrics['viol']:.4f}")
    
    if best_metrics['iae'] >= 300 or best_metrics['viol'] >= 0.02:
        print("PID_WEAK=true")
        best_params["PID_WEAK"] = True
    else:
        best_params["PID_WEAK"] = False
        
    os.makedirs(os.path.join(DAY3_OUTPUT_DIR, "models"), exist_ok=True)
    with open(os.path.join(DAY3_OUTPUT_DIR, "models", "pid_controller_tuned.json"), "w") as f:
        json.dump(best_params, f, indent=4)
        
if __name__ == "__main__":
    tune_pid()
