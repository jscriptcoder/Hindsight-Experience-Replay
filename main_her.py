import torch
import gym
import gym.spaces
import rocket_lander_gym
import warnings
import numpy as np
from hyperopt import hp, fmin, tpe
from agent.config import Config
from agent.ddpg_agent import DDPGAgent
from agent.td3_agent import TD3Agent
from agent.sac_agent import SACAgent
from agent.utils import seed_all, plot_scores

warnings.filterwarnings('ignore')

# RocketLander-v0 | LunarLanderContinuous-v2 | 
env = gym.make('LunarLanderContinuous-v2')

config = Config()

config.env = env
config.num_episodes = 5000
config.env_solved = 200
config.buffer_size = int(1e6)
config.batch_size = 64
config.num_updates = 1
config.max_steps = 2000
config.tau = 0.001
config.gamma = 0.99
config.lr_actor = 1e-4
config.lr_critic = 1e-3
config.hidden_actor = (400, 300)
config.hidden_critic = (400, 300)
config.state_size = env.observation_space.shape[0]*2 # state + goal
config.action_size = env.action_space.shape[0]

agent = DDPGAgent(config)
# agent = TD3Agent(config)
# agent = SACAgent(config)

agent.summary()

# def objective(args):
#     seed_all(config.seed, env)
#     config.batch_size = args
#     agent = DDPGAgent(config)
#     scores = agent.train()
#     return scores[-1]

# space = hp.randint('batch_size', 16, 257)

# best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

# print(best)

scores = agent.train_her()

# plot_scores(scores, polyfit_deg=6)