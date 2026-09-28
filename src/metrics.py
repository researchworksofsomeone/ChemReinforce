import numpy as np

def compute_metrics(T_history, setpoint_history, action_history, C_A_history, T_max=370.0, T_min=300.0, dt=1.0, disturbance_step=None):
    T_arr = np.array(T_history)
    sp_arr = np.array(setpoint_history)
    a_arr = np.array(action_history)
    ca_arr = np.array(C_A_history)
    
    error = T_arr - sp_arr
    iae_T = np.sum(np.abs(error) * dt)
    ise_T = np.sum((error**2) * dt)
    
    max_T = np.max(T_arr)
    max_positive_error = np.max(error)
    
    # constraint_violation_steps
    violations = (T_arr > T_max) | (T_arr < T_min) | (ca_arr < 0)
    constraint_violation_steps = np.sum(violations)
    constraint_violation_rate = constraint_violation_steps / len(T_arr)
    
    control_effort = np.sum(np.abs(np.diff(a_arr)))
    mean_action_rate = control_effort / len(a_arr)
    
    res = {
        "iae_T": iae_T,
        "ise_T": ise_T,
        "max_T": max_T,
        "max_positive_error": max_positive_error,
        "constraint_violation_steps": int(constraint_violation_steps),
        "constraint_violation_rate": float(constraint_violation_rate),
        "control_effort": float(control_effort),
        "mean_action_rate": float(mean_action_rate)
    }
    
    if disturbance_step is not None:
        if disturbance_step < len(T_arr):
            res["post_disturbance_iae"] = np.sum(np.abs(error[disturbance_step:]) * dt)
        else:
            res["post_disturbance_iae"] = 0.0
            
    # settling_time
    settling_time = len(T_arr)
    for t in range(20, len(T_arr) - 20):
        if np.all(np.abs(error[t:t+20]) < 2.0):
            settling_time = t
            break
    res["settling_time"] = settling_time
    
    return res
