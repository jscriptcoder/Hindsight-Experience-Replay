import numpy as np

class StandarScaler:
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

    def scale(self, v):
        v_scaled = (v - self.mean) / self.std
        return v_scaled

class MinMaxScaler:
    def __init__(self, size, range=(0, 1)):
        self.size = size
        self.r_min = range[0]
        self.r_max = range[1]

        self.min_ = np.full(self.size, self.r_max, np.float32)
        self.max_ = np.full(self.size, self.r_min, np.float32)
        
        self.recompute()

    def update(self, v, recompute=True):
        v = v.reshape(-1, self.size)
        self.min_ = np.minimum(self.min_, v.min(axis=0))
        self.max_ = np.maximum(self.max_, v.max(axis=0))

        if recompute: self.recompute()
    
    def recompute(self):
        self.min = self.min_.copy()
        self.max = self.max_.copy()

    def scale(self, v):
        v_std = (v - self.min) / (self.max - self.min)
        v_scaled = v_std * (self.r_max - self.r_min) + self.r_min
        return v_scaled