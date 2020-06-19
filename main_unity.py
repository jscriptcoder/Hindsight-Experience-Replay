import torch
import warnings
import numpy as np

from hyperopt import hp, fmin, tpe
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from common.utils import seed_all

from dqn_agent.dqn_agent import DQNAgent

warnings.filterwarnings('ignore')

seed_all(0)

channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(file_name='./envs/PushBlock.app', side_channels=[channel], no_graphics=True)
env = UnityToGymWrapper(unity_env)

channel.set_configuration_parameters(time_scale = np.inf)

agent = DQNAgent(state_size=env.observation_space.shape[0], 
                 action_size=env.action_space.nvec[0])

agent.summary()

# def objective(args):
#     seed_all(config.seed, env)
#     config.batch_size = args
#     agent = DDPGAgent(config)
#     scores = agent.train()
#     return -np.mean(scores)

# space = hp.randint('batch_size', 16, 257)

# best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

# print(best)

scores = agent.train(env=env)

# plot_scores(scores, polyfit_deg=6)