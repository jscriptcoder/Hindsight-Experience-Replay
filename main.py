import gym
import gym.spaces
import warnings
import rocket_lander_gym
from common.lunar_lander_wrapper import LunarLanderWrapper
from common.config import Config
from common.utils import seed_all
from agent.dqn import DQNAgent
 
warnings.filterwarnings('ignore')

gym_env = gym.make('LunarLander-v2')

config = Config()

config.state_size = gym_env.observation_space.shape[0]
config.action_size = gym_env.action_space.n
config.goal_size = config.state_size - 2
config.batch_size = 64
config.gamma = 0.99
config.tau = 1e-2
config.episodes = 5000
config.lr = 1e-4
config.eps_start = 0.2
config.eps_decay = 1.
config.eps_end = 0.1
config.use_her = True

seed_all(101)

env = LunarLanderWrapper(gym_env, config.use_her)

agent = DQNAgent(config)

agent.train(env)