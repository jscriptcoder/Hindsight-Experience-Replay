import numpy as np

class OUNoise:
    
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.2):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class GaussianNoise:
    
    def __init__(self, action_size, expl_noise=0.1):
        self.action_size = action_size
        self.expl_noise = expl_noise
    
    def reset(self):
        pass
    
    def sample(self):
        return np.random.normal(0, self.expl_noise, size=self.action_size)