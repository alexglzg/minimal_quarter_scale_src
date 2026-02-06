import numpy as np

class SinePath:
    def __init__(self, x_multiplier=0.2, y_offset=3.0):
        self.k = x_multiplier
        self.y_offset = y_offset

    def x(self, s_var):
        return self.k * s_var

    def y(self, s_var):
        return (np.sin(self.k * s_var) + self.y_offset)

    def distance_cost(self, s_var, xpos, ypos):
        return (xpos - self.x(s_var))**2 + (ypos - self.y(s_var))**2
