import numpy as np

class Path:
    def x(self, s_var):
        raise NotImplementedError

    def y(self, s_var):
        raise NotImplementedError

    def x_dot(self, s_var):
        raise NotImplementedError
    
    def y_dot(self, s_var):
        raise NotImplementedError
    
    def distance_cost(self, s_var, xpos, ypos):
        return (xpos - self.x(s_var))**2 + (ypos - self.y(s_var))**2

class SinePath(Path):
    def __init__(self, x_multiplier=0.2, y_offset=3.0):
        self.k = x_multiplier
        self.y_offset = y_offset

    def x(self, s_var):
        return self.k * s_var

    def y(self, s_var):
        return (np.sin(self.k * s_var) + self.y_offset)

    def x_dot(self, s_var):
        return self.k
    
    def y_dot(self, s_var):
        return self.k * np.cos(self.k * s_var)

    def distance_cost(self, s_var, xpos, ypos):
        return super().distance_cost(s_var, xpos, ypos)
