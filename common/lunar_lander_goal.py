import numpy as np

class LunarLanderGoal:
    def __init__(self, gym_env):
        self.env = gym_env
        self.goal = np.array([0., 0., 0., 0., 0., 0.]) # same goal always

    def reset(self):
        return self.env.reset(), self.goal.copy()

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)

        achieved_goal = next_state[:-2]
        reward, distance = self.compute_reward(achieved_goal, self.goal)

        info = {
            'env_reward': env_reward, 
            'achieved_goal': achieved_goal.copy(),
            # 'success': reward == 0
            'success': env_reward == 100
        }
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.1):
        distance = np.linalg.norm(state - goal, axis=-1) # euclidean distance

        # sparse reward (-1 => fail, 0 => success)
        return -(distance > eps).astype(np.float32), distance
