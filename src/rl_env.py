import json
import os
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import torch
from .cstr_dynamics import BatchedCSTRDynamics
from .scenarios import SCENARIOS

class RLCSTRGymEnv(gym.Env):
    def __init__(self, scenario_name="nominal_fixed", seed=None):
        super().__init__()
        self.scenario_name = scenario_name
        self.scenario = SCENARIOS[scenario_name]
        
        self.dyn = BatchedCSTRDynamics(device="cpu")
        
        # Action space: [-1, 1] mapped to [250, 400] via 325 + 75 * action
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs space: normalized C_A, normalized T, normalized previous action
        self.observation_space = Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)
        
        self.T_max = 370.0
        self.T_min = 300.0
        self.max_steps = 200
        
        self.step_count = 0
        self.state = None
        self.T_c_previous = 325.0
        self.action_history = []
        
        self.np_random = np.random.RandomState(seed)
        
    def _normalize_obs(self, C_A, T, prev_action):
        norm_C_A = (C_A - 1.0) / 1.0
        norm_T = (T - 350.0) / 50.0
        norm_prev = (prev_action - 325.0) / 75.0
        obs = np.array([norm_C_A, norm_T, norm_prev], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)
        
    def _get_setpoint(self):
        if self.scenario["setpoint_schedule"] is None:
            return self.scenario["T_setpoint"]
        # Find active setpoint
        current_sp = self.scenario["setpoint_schedule"][0][1]
        for step, sp in self.scenario["setpoint_schedule"]:
            if self.step_count >= step:
                current_sp = sp
        return current_sp

    def _get_disturbed_feed(self):
        c_af = 2.0
        t_f = 300.0
        if self.scenario["disturbance_step"] is not None and self.step_count >= self.scenario["disturbance_step"]:
            c_af *= self.scenario["feed_C_Af_multiplier"]
            t_f += self.scenario["feed_T_f_offset"]
        return c_af, t_f

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        self.step_count = 0
        # Random initial state near steady state
        C_A_0 = 1.0 + self.np_random.uniform(-0.1, 0.1)
        T_0 = 350.0 + self.np_random.uniform(-5.0, 5.0)
        self.state = np.array([C_A_0, T_0], dtype=np.float32)
        self.T_c_previous = 325.0
        self.action_history = [self.T_c_previous] * max(1, self.scenario["actuator_delay_steps"] + 1)
        
        obs = self._normalize_obs(C_A_0, T_0, self.T_c_previous)
        return obs, self._get_info()
        
    def _get_info(self, violation=False, T_c_raw=325.0):
        return {
            "raw_C_A": float(self.state[0]),
            "raw_T": float(self.state[1]),
            "raw_T_c": float(T_c_raw),
            "constraint_violation": violation
        }

    def step(self, action):
        action = float(np.clip(action, -1.0, 1.0)[0])
        T_c = 325.0 + 75.0 * action
        
        # Rate limit
        if self.scenario["action_rate_limit"] is not None:
            delta = T_c - self.T_c_previous
            delta = np.clip(delta, -self.scenario["action_rate_limit"], self.scenario["action_rate_limit"])
            T_c = self.T_c_previous + delta
            
        T_c = np.clip(T_c, 250.0, 400.0)
        self.T_c_previous = T_c
        
        self.action_history.append(T_c)
        if len(self.action_history) > self.scenario["actuator_delay_steps"] + 1:
            self.action_history.pop(0)
            
        effective_T_c = self.action_history[0]
        
        c_af, t_f = self.get_disturbed_feed()
        
        # Step dynamics using torch integrator
        state_t = torch.tensor([self.state], dtype=torch.float32)
        Tc_t = torch.tensor([[effective_T_c]], dtype=torch.float32)
        
        self.dyn.C_Af = torch.tensor(c_af, dtype=torch.float32)
        self.dyn.T_f = torch.tensor(t_f, dtype=torch.float32)
        
        with torch.no_grad():
            next_state_t = self.dyn.step(state_t, Tc_t)
            
        self.state = next_state_t.numpy()[0]
        self.step_count += 1
        
        C_A, T = self.state
        
        # Sensor noise
        if self.scenario["sensor_noise_C_A"] > 0:
            C_A += self.np_random.normal(0, self.scenario["sensor_noise_C_A"])
        if self.scenario["sensor_noise_T"] > 0:
            T += self.np_random.normal(0, self.scenario["sensor_noise_T"])
            
        obs = self._normalize_obs(C_A, T, self.T_c_previous)
        
        # Reward
        sp = self._get_setpoint()
        reward = -abs(T - sp) / 50.0
        
        # Control effort penalty
        if len(self.action_history) >= 2:
            reward -= 0.1 * abs(self.action_history[-1] - self.action_history[-2]) / 75.0
            
        # Constraints
        violation = False
        if T > self.T_max or T < self.T_min or C_A < 0:
            reward -= 50.0
            violation = True
            
        done = self.step_count >= self.max_steps
        
        info = self._get_info(violation, T_c)
        
        return obs, float(reward), done, False, info

    def get_disturbed_feed(self):
        return self._get_disturbed_feed()
