import taichi as ti
import numpy as np


@ti.data_oriented
class RandomPool:
    def __init__(self, N):
        self.N = N
        self.pool = ti.field(dtype=float, shape=N)
    
    def seed(self, s):
        np.random.seed(s)
        data = np.random.rand(self.N)
        self.set_data(data)
    
    @ti.kernel
    def set_data(self, data: ti.ext_arr()):
        for i in range(self.N):
            self.pool[i] = data[i]
    
