import numpy as np

class LunarLanderWrapper:
    """Turns LunarLander into a goal-conditioned environment"""

    def __init__(self, gym_env, with_goal=True):
        self.with_goal = with_goal

        self.env = gym_env

        # state/goal attributes:
        #   [0] horizontal coordinate
        #   [1] vertical coordinate
        #   [2] horizontal speed
        #   [3] vertical speed
        #   [4] angle
        #   [5] angular speed
        #   [6] 1 if first leg has contact, else 0
        #   [7] 1 if second leg has contact, else 0
        # self.goal = np.array([0., 0., 0., 0., 0., 0., 1., 1.])
        self.goal = np.array([0., 0., 0., 0., 0., 0.])

    def reset(self):
        return self.env.reset(), self.goal.copy()

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)

        info = { 'env_reward': env_reward }

        if self.with_goal:
            achieved_goal = next_state[:-2].copy()
            reward = self.compute_reward(achieved_goal, self.goal)

            info['achieved_goal'] = achieved_goal
            info['success'] = (reward == 1)
        else:
            reward = env_reward
            info['success'] = (reward == 100)
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.1, dense=False):
        d = np.linalg.norm(state - goal, axis=-1) # euclidean distance

        if dense:
            reward = -d
        else:
            # sparse reward:
            #   +1 => success
            #   -1 => fail
            reward = 1 if d < eps else -1

        return reward
