import torch.nn as nn

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            
        )
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
            
    def forward(self, state):
        x = self.features(state)
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean(dim=1, keepdim=True)