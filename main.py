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
config.goal_size = config.state_size-2
config.episodes = 1000
config.max_steps = 1000
config.buffer_size = int(1e5)
config.batch_size = 64
config.gamma = 0.99
config.tau = 1e-3
config.update_every = 1
config.use_double = True
config.use_huber_loss = False
config.lr = 5e-4
config.eps_start = 0.2
config.eps_decay = 0.999
config.eps_end = 0.1
config.use_her = True
config.future_k = 4
config.dist_tolerance = 0.1
config.dense_reward = True
config.times_eval = 10
config.solved_rate = 0.9

seed_all(0)

env = LunarLanderWrapper(gym_env, 
                         with_goal=config.use_her, 
                         dist_tolerance=config.dist_tolerance,
                         dense_reward=config.dense_reward)

agent = DQNAgent(config)

agent.train(env)