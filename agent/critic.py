import torch

from .device import device
from .base_network import BaseNetwork

class Critic(BaseNetwork):
    def __init__(self, state_size, action_size, hidden_size, activ):
        super().__init__(activ)

        dims = (state_size + action_size,) + hidden_size + (1,)

        self.build_layers(dims)

        self.reset_parameters()

    def forward(self, state, action):
        """Maps state and action => Q-values, Q(s,a) => Q-values"""
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)

        if type(action) != torch.Tensor:
            action = torch.FloatTensor(action).to(device)

        x = torch.cat((state, action), dim=1)

        for layer in self.layers[:-1]:
            x = self.activ(layer(x))

        return self.layers[-1](x)
