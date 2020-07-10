import time
import datetime
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import namedtuple
from .device import device

def seed_all(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
    
    if env is not None and 'seed' in dir(env):
        env.seed(seed)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Helper to create a experience tuple with named fields
make_experience = namedtuple('Experience',
                             field_names=['state',
                                          'action',
                                          'reward',
                                          'next_state',
                                          'done', 
                                          'info'])

def from_experience(experiences):
    states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)

    actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)

    rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

    next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

    dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None])\
            .astype(np.uint8)).to(device)
    
    return states, actions, rewards, next_states, dones