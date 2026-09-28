import torch
from .config import CSTR_CONFIG

class BatchedCSTRDynamics:
    def __init__(self, device):
        self.device = device
        self.F_over_V = torch.tensor(CSTR_CONFIG["F"] / CSTR_CONFIG["V"], device=device, dtype=torch.float32)
        self.C_Af = torch.tensor(CSTR_CONFIG["C_Af"], device=device, dtype=torch.float32)
        self.k0 = torch.tensor(CSTR_CONFIG["k0"], device=device, dtype=torch.float32)
        self.E_over_R = torch.tensor(CSTR_CONFIG["E_over_R"], device=device, dtype=torch.float32)
        self.T_f = torch.tensor(CSTR_CONFIG["T_f"], device=device, dtype=torch.float32)
        self.delta_H = torch.tensor(CSTR_CONFIG["delta_H_over_rho_Cp"], device=device, dtype=torch.float32)
        self.UA = torch.tensor(CSTR_CONFIG["UA_over_rho_Cp_V"], device=device, dtype=torch.float32)
        self.dt = CSTR_CONFIG["dt"]

    def _ode_rhs(self, state, action):
        C_A = state[:, 0]
        T = state[:, 1]
        T_c = action.squeeze(-1)
        
        T_safe = torch.clamp(T, min=200.0)
        rate = self.k0 * torch.exp(-self.E_over_R / T_safe)
        
        dC_A = self.F_over_V * (self.C_Af - C_A) - rate * C_A
        dT = self.F_over_V * (self.T_f - T) + self.delta_H * rate * C_A + self.UA * (T_c - T)
        
        derivatives = torch.stack([dC_A, dT], dim=-1)
        return torch.nan_to_num(derivatives, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, state, action):
        k1 = self._ode_rhs(state, action)
        k2 = self._ode_rhs(state + 0.5 * self.dt * k1, action)
        k3 = self._ode_rhs(state + 0.5 * self.dt * k2, action)
        k4 = self._ode_rhs(state + self.dt * k3, action)
        
        next_state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        next_state[:, 0] = torch.clamp(next_state[:, 0], min=CSTR_CONFIG["C_A_min"], max=CSTR_CONFIG["C_A_max"])
        next_state[:, 1] = torch.clamp(next_state[:, 1], min=250.0, max=500.0)
        return next_state
