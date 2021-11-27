import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, abs, sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as matpat


class Vector2D:
	def __init__(self, x, y):
		self._array = np.array([x, y])
	
	@property
	def x(self):
		return self._array[0]
	
	@property
	def y(self):
		return self._array[1]
	
	@property
	def angle(self):
		return arctan2(self.y, self.x)
	
	@property
	def numpy(self):
		return self._array
	
	@property
	def norm(self):
		return np.hypot(self.x, self.y)
	
	@property
	def normalized(self):
		l = np.hypot(self.x, self.y)
		return Vector2D(self.x/l, self.y/l)
	
	def rotated(self, angle):
		x = self.x * np.cos(angle) - self.y * np.sin(angle)
		y = self.x * np.sin(angle) + self.y * np.cos(angle)
		return Vector2D(x, y)
	
	def __repr__(self):
		return f'Vector2D({self.x}, {self.y})'
	
	def draw(self, axes, **kwargs):
		axes.plot(self.x, self.y, **kwargs)
	
	def __add__(self, other):
		return Vector2D(self.x + other.x, self.y + other.y)
	
	def __sub__(self, other):
		return Vector2D(self.x - other.x, self.y - other.y)


if __name__ == '__main__':
	v1 = Vector2D(1,2)
	print(v1)
	print(v1.norm)
	print(v1.normalized)
	print(v1.x, v1.y)
		