import numpy as np

class Normalizer:
    def __init__(self, size, min_std=1e-2):
        self.min_std = min_std
        self.size = size

        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = 0
    
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v, recompute=True):
        v = v.reshape(-1, self.size)
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]

        if recompute: self.recompute()
    
    def recompute(self):
        self.mean = self.sum / self.count
        var = self.sumsq / self.count - (self.sum / self.count)**2
        var = np.maximum(self.min_std**2, var)
        self.std = np.maximum(var**.5, self.min_std)

    def normalize(self, v):
        v_norm = (v - self.mean) / self.std
        return v_norm