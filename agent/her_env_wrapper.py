import numpy as np

class HerEnvWrapper():

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.goal_state = [0., 0., 0., 0., 0., 0., 1., 1.] # LunarLander goal
        self.acc_reward = 0
    
    def reset(self):
        self.acc_reward = 0
        return self.env.reset(), self.goal_state
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Strategy 1:
        #   accumulated reward if done else 0.0
        self.acc_reward += reward
        reward = self.acc_reward if done else 0.

        return next_state, reward, done, info
    
    def render(self, *args):
        self.env.render(*args)
    
    def close(self):
        self.env.close()