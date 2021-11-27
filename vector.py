import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, abs, sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as matpat

def column_vector(array):
	return array.reshape((-1,1))

def row_vector(array):
	return array.reshape((1,-1))

Vector = np.array
Matrix = np.array

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
		return f'Vector2D({self.x:.3f}, {self.y:.3f})'
	
	def draw(self, axes, **kwargs):
		axes.plot(self.x, self.y, **kwargs)
	
	def __add__(self, other):
		return Vector2D(self.x + other.x, self.y + other.y)
	
	def __sub__(self, other):
		return Vector2D(self.x - other.x, self.y - other.y)


class Vector2DArray:
	def __init__(self, points: list):
		self._array = []
		for p in points:
			self._array.append(Vector2D(*p))
		self._numpy = Matrix([p.numpy for p in self.array])
	
	@property
	def array(self):
		return self._array
	
	@property
	def x(self):
		return self.numpy[:,0]
	
	@property
	def y(self):
		return self.numpy[:,1]

	@property
	def numpy(self):
		return self._numpy

	@property
	def isempty(self):
		return len(self) == 0
	
	def __repr__(self):
		repr = 'Vector2DArray( '
		for i, p in enumerate(self.array):
			repr += f'{p}'
			if i < len(self.array)-1:
				repr += ', '
		repr += ' )'
		return repr
	
	def draw(self, axes, **kwargs):
		for p in self.array:
			p.draw(axes, **kwargs)
	
	def __len__(self):
		return len(self.array)
		
	def __getitem__(self, i):
		return self.array[i]
	

if __name__ == '__main__':
	v1 = Vector2D(1,2)
	print(v1)
	print(v1.norm)
	print(v1.normalized)
	print(v1.x, v1.y)

	vs = Vector2DArray([(0,0), (1,2), (2,-3)])
	print(vs)
	print(vs[1])