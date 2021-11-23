import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, abs, sqrt
import matplotlib.pyplot as plt
from point2d import PointNT

class VehicleState:
    def __init__(self, pos=PointNT(0,0), vel=PointNT(0,1), acc=PointNT(0,0)):
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
    def phi(self):
        return arctan2(self.vel[1], self.vel[0])
    