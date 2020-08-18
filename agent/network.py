import torch.nn as nn

class DuelingQNetwork(nn.Module):
    """Dueling Network Architecture
    See: https://arxiv.org/abs/1511.06581
    """

    def __init__(self, state_size, action_size):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            
        )
        
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
            
    def forward(self, state):
        x = self.features(state)
        advantage = self.advantage(x)
        value = self.value(x)
        
        return value + advantage  - advantage.mean(dim=1, keepdim=True)