import torch
import torch.nn as nn
import torch.nn.functional as F

from .device import device
from .actor import Actor

class HerActor(Actor):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 activ):
        
        # state_size*2 = state_size + goal_size
        super().__init__(state_size*2, action_size, hidden_size, activ)

    def forward(self, state, goal):        
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        if type(goal) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)

        x = torch.cat([state, goal], dim=1)

        return super().forward(x)
