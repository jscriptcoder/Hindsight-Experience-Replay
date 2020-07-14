import numpy as np

class LunarLanderWrapper:
    def __init__(self, gym_env):
        self.env = gym_env
        self.goal = np.array([0., 0., 0., 0., 0., 0.]) # same goal always

    def reset(self):
        return self.env.reset(), self.goal.copy()

    def step(self, action):
        next_state, env_reward, env_done, info = self.env.step(action)

        achieved_goal = next_state[:-2]
        reward, done = self.compute_reward(achieved_goal, self.goal)

        info = {
            'env_reward': env_reward, 
            'achieved_goal': achieved_goal.copy(),
            'success': done
        }
        
        return next_state, reward, (done or env_done), info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.01):
        distance = np.linalg.norm(state - goal, axis=-1) # euclidean distance
        done = distance <= eps

        # sparse reward (-1 => fail, 0 => success)
        # reward = 0. if done else -1.
        reward = -distance
        return reward, done
