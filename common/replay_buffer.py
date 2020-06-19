import random
from collections import deque


class ReplayBuffer:
    """This buffer will help to reduce correlation between experiences.
    It uses a deque data structure as buffer. Will store experiences as
    namedtuples:
        experience = (state, action, reward, next_state, done)

    Args:
        buffer_size (int)
        batch_size (int)
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)  # internal buffer (deque)
        self.batch_size = batch_size

    def add(self, experience):
        """Save experience in memory

        Args:
            experience (Named Tuple):
                (state, action, reward, next_action, done)
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample batch_size random experiences from memory

        Returns:
            List of tuples: batch_size of experiences
        """
        return random.sample(self.buffer, k=self.batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """This buffer will help to reduce correlation between experiences.
    It also adds priorities to these experiences based on TD error. 
    Implements the ideas of this paper: https://arxiv.org/abs/1511.05952
        
    It uses a list data structure as buffer. Will store experiences as 
    namedtuples:
        experience = (state, action, reward, next_state, done)
        
    Args:
        buffer_size (int)
        batch_size (int)
    
    Attributes:
        epsilon (float): 
            prevents the edge-case of transitions not being revisited once their 
            error is zero
            Default: 1e-5
        alpha (float):
            for stochastic sampling method that interpolates between pure greedy 
            prioritization and uniform random sampling
            Default: 0.6
        beta (float): important sampling bias control hyperparameter
            Default: 0.4
        beta_inc_per_sampling (float): 
            how much we increment beta parameter per sampling
            Default: 0.001
            
    """
    
    epsilon = 1e-5
    alpha = .6
    beta = .4
    beta_inc_per_sampling = 0.001
    
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
    
    def add(self, experience):
        """Save experience in memory, giving maximum priosity to new experiences
        
        Args:
            experience (Named Tuple): 
                (state, action, reward, next_action, done)
        """
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self):
        """Sample batch_size random experiences from memory
        
        Returns:
            List of named tuples: batch_size of experiences
            Numpy array of indices: required to update the priorities
            Numpy array of weights: importance-sampling (IS) weights
        """
        
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        # Sample transition
        probs  = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        self.beta = np.min([1., self.beta + self.beta_inc_per_sampling])
        
        # Compute importance-sampling weight
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        return experiences, indices, weights
    
    def update(self, indices, td_errors):
        """Update priorities based on TD errors
        
        Args:
            indices (Numpy array of int)
            td_errors (Numpy array of float)
        """
        prios = np.abs(td_errors) + self.epsilon
        for idx, prio in zip(indices, prios):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)