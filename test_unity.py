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
unity_env = UnityEnvironment(file_name='./envs/PushBlock.app', side_channels=[channel])
env = UnityToGymWrapper(unity_env)

channel.set_configuration_parameters(time_scale = 2.0)

agent = DQNAgent(state_size=env.observation_space.shape[0], 
                 action_size=env.action_space.nvec[0])

agent.load_weights()

for i in range(10):

    steps = 0
    state = env.reset()
    while True:
        action = agent.act(state)
        state, reward, done, _ = env.step(action)

        print("Reward: ", reward)
        steps += 1

        if done or steps > 100: break

env.close()