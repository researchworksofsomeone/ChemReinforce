import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAY1_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "day1")
DAY2_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "day2")
DAY3_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "day3")
DAY4_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "day4")
DAY5_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "day5")
SEED = 42

CSTR_CONFIG = {
    "F": 100.0,
    "V": 100.0,
    "C_Af": 1.0,
    "T_f": 350.0,
    "k0": 7.2e10,
    "E_over_R": 8750.0,
    "delta_H_over_rho_Cp": 50.0,
    "UA_over_rho_Cp_V": 1.0,
    "dt": 0.1,
    "T_setpoint": 350.0,
    "T_max": 370.0,
    "T_min": 300.0,
    "C_A_min": 0.0,
    "C_A_max": 2.0,
    "T_c_min": 250.0,
    "T_c_max": 400.0
}

ENV_CONFIG = {
    "episode_length": 200,
    "reset_C_A_low": 0.7,
    "reset_C_A_high": 1.0,
    "reset_T_low": 340.0,
    "reset_T_high": 360.0,
    "sensor_noise_std": 0.0,
    "actuator_delay_steps": 0,
    "reward_weights": {
        "tracking": 1.0,
        "control_effort": 0.05,
        "constraint_violation": 10.0
    }
}

DATA_CONFIG = {
    "total_transitions": 50000,
    "pid_transitions": 25000,
    "random_transitions": 25000,
    "batch_size": 2048,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
}

SURROGATE_CONFIG = {
    "hidden_sizes": [256, 256, 256],
    "activation": "tanh",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 100,
    "early_stopping_patience": 10,
    "gradient_clip_norm": 1.0,
    "physics_lambda": 0.1,
    "bounds_lambda": 0.05,
    "rollout_horizons": [1, 5, 10, 20, 50],
    "num_rollout_episodes": 100,
    "inference_batch_size": 10000,
    "inference_horizon": 10
}
