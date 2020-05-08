import torch
import torch.nn as nn
from torch.distributions import Normal

from .device import device
from .base_network import BaseNetwork

class GaussianPolicy(BaseNetwork):
    def __init__(self, 
                 state_size, 
                 action_size, 
                 hidden_size, 
                 activ, 
                 log_std_min=-20, 
                 log_std_max=2):
        super().__init__(activ)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        dims = (state_size,) + hidden_size + (action_size,)
        
        self.build_layers(dims)

        # Will depend on state
        self.log_std = nn.Linear(dims[-2], dims[-1])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.log_std.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(state).to(device)
        
        x = self.layers[0](state)
        
        for layer in self.layers[1:-1]:
            x = self.activ(layer(x))
        
        mean = self.layers[-1](x)
        log_std = self.log_std(x)
        
        log_std = torch.clamp(log_std, 
                              min=self.log_std_min, 
                              max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, eps=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp() # will always give a positive value
        
        # Reparameterization trick (mean + std * N(0,1))
        # We can achieve the same by just doing: 
        #   normal = Normal(mean, std)
        #   action = normal.rsample()
        # see https://pytorch.org/docs/stable/distributions.html#pathwise-derivative
        normal = Normal(0, 1)
        z = normal.sample().to(device)
        x = mean + std * z
        
        action = torch.tanh(x)
        log_prob = Normal(mean, std).log_prob(x)
        
        # Enforcing action bound for continuous actions
        # see Appendix C in papers
        log_prob -= torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(1, keepdim=True)
        
        
        return action, log_prob, torch.tanh(mean)
        
