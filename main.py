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
config.goal_size = config.state_size
config.use_her = False

seed_all(0)

env = LunarLanderWrapper(gym_env, config.use_her)

agent = DQNAgent(config)

agent.train(env)