import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, abs, sqrt
import matplotlib.pyplot as plt
from vector import Vector2D

class VehicleState:
    def __init__(self, pos=Vector2D(0,0), vel=Vector2D(0,1), acc=Vector2D(0,0)):
        self._pos = pos
        self._vel = vel
        self._acc = acc
    
    @property
    def pos(self):
        return self._pos
    
    @property
    def vel(self):
        return self._vel
    
    @property
    def acc(self):
        return self._acc
    
    @property
    def yaw(self):
        return arctan2(self.vel.y, self.vel.x)
    
    def __repr__(self) -> str:
        return f'VehicleState(pos={self.pos}, vel={self.vel}, acc={self.acc})'


if __name__ == '__main__':
    vs = VehicleState(pos=Vector2D(0,1), vel=Vector2D(2,2), acc=Vector2D(-1,2))

    print(vs.yaw/(np.pi/180))
    