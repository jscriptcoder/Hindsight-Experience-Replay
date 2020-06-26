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
    """Will return a tuple with the range for the initialization
    of hidden layers

    Args:
        layer (torch.nn.Layer)

    Returns:
        Tuple of int
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Args:
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Helper to create a experience tuple with named fields
make_experience = namedtuple('Experience',
                             field_names=['state',
                                          'action',
                                          'reward',
                                          'next_state',
                                          'done'])

def from_experience(experiences, with_goal=False):
    """Returns a tuple with (s, a, r, s', d)

    Args:
        namedtuple: List of tensors

    Returns:
        Tuple of torch.Tensor
    """
    states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None]))\
            .float().to(device)

    actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None]))\
            .float().to(device)

    rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None]))\
            .float().to(device)

    next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None]))\
            .float().to(device)

    dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None])\
            .astype(np.uint8)).float().to(device)
    
    if with_goal:
        goals = torch.from_numpy(
            np.vstack([e.goal for e in experiences if e is not None]))\
            .float().to(device)
        
        return states, actions, rewards, next_states, dones, goals
    else:
        return states, actions, rewards, next_states, dones

def run_env(env, get_action=None, max_t=1000, close_env=True):
    """Run actions against an environment.
    We pass a function in that could or not be wrapping an agent's actions

    Args:
        env (Environment)
        get_action (func): returns actions based on a state
        max_t (int): maximum number of timesteps
    """

    if get_action is None:
        get_action = lambda _: env.action_space.sample()

    state = env.reset()
    env.render()

    while True:
        action = get_action(state)
        state, reward, done, _ = env.step(action)
        env.render()

        if done: break

    if close_env:
        env.close()

def scores2poly1d(scores, polyfit_deg):
    """Fit a polynomial to a list of scores

    Args:
        scores (List of float)
        polyfit_deg (int): degree of the fitting polynomial

    Returns:
        List of int, one-dimensional polynomial class
    """
    x = list(range(len(scores)))
    degs = np.polyfit(x, scores, polyfit_deg)
    return x, np.poly1d(degs)

def plot_scores(scores,
                title='Agents avg score',
                figsize=(15, 6),
                polyfit_deg=None):
    """Plot scores over time. Optionally will draw a line showing the trend

    Args:
        scores (List of float)
        title (str)
        figsize (tuple of float)
            Default: (15, 6)
        polyfit_deg (int): degree of the fitting polynomial (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(scores)

    max_score = max(np.round(scores, 3))
    idx_max = np.argmax(scores)
    plt.scatter(idx_max, max_score, c='r', linewidth=3)

    if polyfit_deg is not None:
        x, p = scores2poly1d(scores, polyfit_deg)
        plt.plot(p(x), linewidth=3)

    plt.title(title)
    ax.set_ylabel('Score')
    ax.set_xlabel('Epochs')
    ax.legend(['Score', 'Trend', 'Max avg score: {}'.format(max_score)])

def get_time_elapsed(start, end=None):
    """Returns a human readable (HH:mm:ss) time difference between two times

    Args:
        start (float)
        end (float): optional value
            Default: now
    """

    if end is None:
        end = time.time()
    elapsed = round(end-start)
    return str(datetime.timedelta(seconds=elapsed))

def play_game(agent, min_score):
    """Will play a game until it reaches the min_score

    Args:
        ml_agent (MultiAgent)
        min_score (float)
    """
    score = -np.inf
    while score < min_score:
        score = agent.eval_episode()

def get_reward(obs, goal, eps = [.001, .001]):
    eps = np.array(eps)
    reward = 0. if np.any(np.abs(goal - obs) > eps) else 100.
    return reward

def random_sample(population, how_many=8):
    k = min(len(population), how_many)
    return random.sample(population, k=k)