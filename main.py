import torch
import gym
import gym.spaces
import rocket_lander_gym
import warnings
from agent.config import Config
from agent.ddpg_agent import DDPGAgent
# from agent.td3_agent import TD3Agent
from agent.utils import seed_all, plot_scores

warnings.filterwarnings('ignore')

# RocketLander-v0 | LunarLanderContinuous-v2 | 
env = gym.make('LunarLanderContinuous-v2')

config = Config()

config.env = env
config.env_solved = 200
config.buffer_size = int(1e6)
config.batch_size = 64
config.num_episodes = 2000
config.num_updates = 1
config.max_steps = 2000
config.policy_freq_update = 1
config.lr_actor = 1e-4
config.lr_critic = 1e-3
config.hidden_actor = (400, 300)
config.hidden_critic = (400, 300)
config.state_size = env.observation_space.shape[0]
config.action_size = env.action_space.shape[0]

seed_all(config.seed, env)

agent = DDPGAgent(config)
# agent = TD3Agent(config)

agent.summary()

scores = agent.train()

plot_scores(scores, polyfit_deg=6)