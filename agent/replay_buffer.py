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
