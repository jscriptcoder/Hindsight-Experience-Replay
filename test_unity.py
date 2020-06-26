import torch
import warnings
import numpy as np

from hyperopt import hp, fmin, tpe
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from common.utils import seed_all
from common.config import Config
from pg_agent.ddpg_agent import DDPGAgent
from pg_agent.sac_agent import SACAgent

warnings.filterwarnings('ignore')

seed_all(0)

channel = EngineConfigurationChannel()
unity_env = UnityEnvironment(file_name='./envs/Walker.app', side_channels=[channel])
env = UnityToGymWrapper(unity_env)

channel.set_configuration_parameters(time_scale = 2.0)

config = Config()

config.env = env
config.num_episodes = 5000
config.env_solved = 700
config.times_solved = 100
config.buffer_size = int(1e6)
config.batch_size = 64
config.num_updates = 2
config.update_every = 1
config.policy_freq_update = 1
config.max_steps = 5000
config.tau = 1e-3
config.gamma = 0.99
config.lr_actor = 3e-4
config.lr_critic = 3e-4
config.alpha_auto_tuning = True
config.lr_alpha = 3e-4
config.hidden_actor = (256, 256)
config.hidden_critic = (256, 256)
config.state_size = env.observation_space.shape[0]
config.action_size = env.action_space.shape[0]

agent = SACAgent(config)

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