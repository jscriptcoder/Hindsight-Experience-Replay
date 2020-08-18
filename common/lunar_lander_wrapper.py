import numpy as np

class LunarLanderWrapper:
    """Turns LunarLander into a goal-conditioned environment
    state attributes:
        [0] horizontal coordinate
        [1] vertical coordinate
        [2] horizontal speed
        [3] vertical speed
        [4] angle
        [5] angular speed
        [6] 1 if first leg has contact, else 0
        [7] 1 if second leg has contact, else 0
    """

    def __init__(self, gym_env, with_goal=True, dist_tolerance=0.05, dense_reward=False):
        self.with_goal = with_goal
        self.dist_tolerance = dist_tolerance
        self.dense_reward = dense_reward

        self.env = gym_env
        self.reset_goal()
    
    def reset_goal(self):
        x = np.random.uniform(-1., 1.)
        y = np.random.uniform(0., 1.4)

        self.goal = np.array([x, y, 0., 0., 0., 0.])
        return self.goal.copy()

    def reset(self):
        return self.env.reset(), self.reset_goal()

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)

        info = { 'env_reward': env_reward }

        if self.with_goal:
            achieved_goal = self.to_goal(next_state)
            reward, success = self.compute_reward(achieved_goal, 
                                                  self.goal, 
                                                  eps=self.dist_tolerance, 
                                                  dense=self.dense_reward)

            info['success'] = success
        else:
            info['success'] = (env_reward == 100)
            reward = env_reward
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.05, dense=False):
        d = np.linalg.norm(state - goal, axis=-1) # euclidean distance
        success = d <= eps

        if dense:
            reward = 1. if success else -d
        else:
            # Sparse reward:
            #    1 => success
            #   -1 => fail
            reward = 1. if success else -1.

        return reward, success
    
    @staticmethod
    def to_goal(state):
        return state[:-2].copy()
