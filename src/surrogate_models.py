import torch
import torch.nn as nn
from .config import SURROGATE_CONFIG

class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        sizes = [3] + SURROGATE_CONFIG["hidden_sizes"] + [2]
        
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if SURROGATE_CONFIG["activation"] == "tanh":
                layers.append(nn.Tanh())
            elif SURROGATE_CONFIG["activation"] == "relu":
                layers.append(nn.ReLU())
                
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        
        self.network = nn.Sequential(*layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, state_norm, action_norm):
        x = torch.cat([state_norm, action_norm], dim=-1)
        return self.network(x)
