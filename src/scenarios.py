SCENARIOS = {
    "nominal_fixed": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 0.0,
        "disturbance_step": None,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 0,
        "action_rate_limit": None
    },
    "setpoint_steps": {
        "T_setpoint": 350.0,
        "setpoint_schedule": [
            [0, 350.0],
            [50, 360.0],
            [100, 345.0],
            [150, 350.0]
        ],
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 0.0,
        "disturbance_step": None,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 0,
        "action_rate_limit": None
    },
    "feed_ca_disturbance": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.2,
        "feed_T_f_offset": 0.0,
        "disturbance_step": 50,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 0,
        "action_rate_limit": None
    },
    "feed_temp_disturbance": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 10.0,
        "disturbance_step": 50,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 0,
        "action_rate_limit": None
    },
    "sensor_noise": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 0.0,
        "disturbance_step": None,
        "sensor_noise_T": 0.5,
        "sensor_noise_C_A": 0.01,
        "actuator_delay_steps": 0,
        "action_rate_limit": None
    },
    "actuator_delay": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 0.0,
        "disturbance_step": None,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 2,
        "action_rate_limit": None
    },
    "actuator_rate_limit": {
        "T_setpoint": 350.0,
        "setpoint_schedule": None,
        "feed_C_Af_multiplier": 1.0,
        "feed_T_f_offset": 0.0,
        "disturbance_step": None,
        "sensor_noise_T": 0.0,
        "sensor_noise_C_A": 0.0,
        "actuator_delay_steps": 0,
        "action_rate_limit": 5.0
    }
}
