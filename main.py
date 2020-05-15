import torch
import gym
import gym.spaces
import rocket_lander_gym
import warnings
from agent.config import Config
from agent.ddpg_agent import DDPGAgent
from agent.td3_agent import TD3Agent
from agent.utils import seed_all, plot_scores

warnings.filterwarnings('ignore')

# RocketLander-v0 | LunarLanderContinuous-v2
env = gym.make('LunarLanderContinuous-v2')

config = Config()

config.seed = 0
config.env = env
config.env_solved = 200
config.times_solved = 100
config.buffer_size = int(1e5)
config.batch_size = 64
config.num_episodes = 3000
config.num_updates = 1 
config.max_steps = 300
config.max_steps_reward = None
config.state_size = env.observation_space.shape[0]
config.action_size = env.action_space.shape[0]
config.gamma = 0.99
config.tau = 1e-3
config.lr_actor = 1e-4
config.lr_critic = 1e-3
config.hidden_actor = (128, 64)
config.hidden_critic = (128, 64)
config.activ_actor = torch.nn.ReLU()
config.activ_critic = torch.nn.ReLU()
config.optim_actor = torch.optim.Adam
config.optim_critic = torch.optim.Adam
config.grad_clip_actor = None
config.grad_clip_critic = None
config.use_huber_loss = False
config.update_every = 1
config.use_ou_noise = False
config.ou_mu = 0.0
config.ou_theta = 0.15
config.ou_sigma = 0.2
config.expl_noise = 0.1
config.noise_weight = 1.0
config.decay_noise = True
config.use_linear_decay = False
config.noise_linear_decay = 1e-6
config.noise_decay = 0.99
config.log_every = 100
config.policy_noise = 0.2
config.noise_clip = 0.5
config.policy_freq_update = 2

seed_all(config.seed, env)

# agent = DDPGAgent(config)
agent = TD3Agent(config)

agent.summary()

scores = agent.train()

plot_scores(scores, polyfit_deg=6)