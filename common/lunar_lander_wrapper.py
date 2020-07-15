import numpy as np

class LunarLanderWrapper:
    def __init__(self, gym_env, with_goal=True):
        self.with_goal = with_goal

        self.env = gym_env
        self.goal = np.array([0., 0., 0., 0., 0., 0., 1., 1.])

    def reset(self):
        return self.env.reset(), self.goal.copy()

    def step(self, action):
        next_state, env_reward, env_done, info = self.env.step(action)

        info = { 'env_reward': env_reward }

        if self.with_goal:
            achieved_goal = next_state.copy()
            reward, done = self.compute_reward(achieved_goal, self.goal)

            info['achieved_goal'] = achieved_goal
            info['success'] = done

            done = (done or env_done)
        else:
            reward = env_reward
            done = env_done
        
        return next_state, reward, done, info

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()

    @staticmethod
    def compute_reward(state, goal, eps=0.005):
        distance = np.linalg.norm(state - goal, axis=-1) # euclidean distance
        done = distance <= eps

        # sparse reward (-1 => fail, 0 => success)
        reward = 0. if done else -1.
        # reward = -distance
        return reward, done
