import torch.nn as nn

from .device import device
from .utils import hidden_init

class BaseNetwork(nn.Module):
    def __init__(self, activ):
        super().__init__()
        
        self.activ = activ
        self.to(device)
    
    def build_layers(self, dims):
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) \
                                     for dim_in, dim_out \
                                     in zip(dims[:-1], dims[1:])])
    
    def reset_parameters(self):
        for layer in self.layers[0:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self):
        pass