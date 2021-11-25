import numpy as np
from numpy import sin, cos, arccos, arctan, arctan2, abs, sqrt
import matplotlib.pyplot as plt
import matplotlib.patches as matpat


class PointNT:
	def __init__(self, n, t):
		self._array = np.array([n, t])
	
	@property
	def n(self):
		return self._array[0]
	
	@property
	def t(self):
		return self._array[1]
	
	@property
	def norm(self):
		return np.hypot(self.n, self.t)
	
	@property
	def normalized(self):
		l = np.hypot(self.n, self.t)
		return PointNT(self.n/l, self.t/l)
	
	def __repr__(self):
		return f'PointNT(n={self.n}, t={self.t})'
		



class Point2D:
	def __init__(self, *args):
		nargs = len(args)
		if nargs == 0:
			self.__data = np.zeros(2)
		elif nargs == 1: # [x, y]
			if isinstance(args[0], Point2D):
				self.__data = np.array([args[0].x, args[0].y])
			elif isinstance(args[0], list) or isinstance(args[0], tuple) or isinstance(args[0], np.ndarray):
				if len(args[0]) == 2:
					self.__data = np.array([args[0][0], args[0][1]])
				else:
					raise SyntaxError
			else:
				raise NotImplementedError
		elif nargs == 2: # x, y
			self.__data = np.array([args[0], args[1]])
		else:
			raise NotImplementedError
	
	@property
	def x(self):
		return self.__data[0]
	
	@property
	def y(self):
		return self.__data[1]
	
	@property
	def numpy(self):
		return self.__data
	
	@property
	def angle(self):
		return arctan2(self.y, self.x)
	
	@property
	def homo(self):
		return Point2Dh(self)
	
	@property
	def norm(self):
		return np.linalg.norm(self.numpy)
	
	@property
	def normalized(self):
		d = self.norm
		if d > 0:
			return self / self.norm
		else:
			print('Point2D.normalized(): division by zero')
			return Point2D(1,0)
	
	@property
	def rotated90(self):
		return Point2D(-self[1], self[0])

	def __repr__(self):
		return f'Point2D({self.x:.2f}, {self.y:.2f})'
	
	def draw(self, axes, **kwargs):
		point = plt.Line2D([self.x], [self.y], marker='.', markersize=10, **kwargs)
		axes.add_artist(point)

	def __getitem__(self, index):
		return self.__data[index]
	
	def __neg__(self):
		return Point2D([-self.x, -self.y])

	def __add__(self, other):
		other = Point2D(other)
		return Point2D([self.x + other.x, self.y + other.y])
	
	def __sub__(self, other):
		other = Point2D(other)
		return Point2D([self.x - other.x, self.y - other.y])
	
	def __mul__(self, rhs):
		return Point2D(self.numpy * rhs)
	
	def __rmul__(self, lhs):
		return self * lhs
	
	def __truediv__(self, rhs):
		return Point2D(self.numpy / rhs)

	def __len__(self):
		return len(self.__data)
	
	def __eq__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__eq__(other.homo)
		elif isinstance(other, Point2D):
			return self.x == other.x and self.y == other.y
		else:
			raise NotImplementedError
	
	def __ne__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__ne__(other.homo)
		elif isinstance(other, Point2D):
			return self.x != other.x or self.y != other.y
		else:
			raise NotImplementedError
	
	def __gt__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__gt__(other.homo)
		else:
			raise NotImplementedError
	
	def __ge__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__ge__(other.homo)
		else:
			raise NotImplementedError
	
	def __lt__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__lt__(other.homo)
		else:
			raise NotImplementedError
	
	def __le__(self, other):
		if isinstance(other, LineSeg2D):
			return self.homo.__le__(other.homo)
		else:
			raise NotImplementedError
	
	def moved(self, dpos):
		return self + Point2D(dpos)

	def rotated(self, dth):
		c, s = cos(dth), sin(dth)
		R = np.array([[c, -s], [s, c]])
		return Point2D(R @ self.numpy)
	
	def dot(self, other):
		other = Point2D(other)
		return np.dot(self.numpy, other.numpy)
	
	def cross(self, other):
		other = Point2D(other)
		return np.cross(self.numpy, other.numpy)
	
	def distance_to(self, other, line=False):
		other = Point2D(other)
		l = other - self
		d = l.norm
		if line: return d, l
		else: return d
	
	def min_distance_to(self, other, line=False):
		if isinstance(other, Point2D):
			return self.distance_to(other, line)
		elif isinstance(other, LineSeg2D):
			return other.min_distance_to(self, line)
		else:
			raise NotImplementedError


Vector = Point2D


class Point2DArray:
	def __init__(self, *args):
		nargs = len(args)
		self._array = [] # if nargs == 0
		if nargs == 1:
			for p in args[0]:
				self._array.append(Point2D(p))
		elif nargs > 1:
			for p in args:
				self._array.append(Point2D(p))
	
	@property
	def array(self):
		return self._array

	@property
	def left_most(self):
		return min(self.array, key=lambda p: p.x)
	
	@property
	def right_most(self):
		return max(self.array, key=lambda p: p.x)
	
	@property
	def bottom_most(self):
		return min(self.array, key=lambda p: p.y)
	
	@property
	def top_most(self):
		return max(self.array, key=lambda p: p.y)
	
	@property
	def center(self):
		psum = Point2D(0, 0)
		for point in self.array:
			psum += point
		return psum / len(self)
	
	@property
	def numpy(self):
		return np.array([p.numpy for p in self.array])

	@property
	def isempty(self):
		return len(self) == 0
	
	def __repr__(self):
		repr = 'Point2DArray( '
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
	
	def __neg__(self):
		return Point2DArray(-self.numpy)

	def __add__(self, other):
		other = Point2D(other)
		return Point2DArray(self.numpy + other.numpy)
	
	def __sub__(self, other):
		other = Point2D(other)
		return Point2DArray(self.numpy - other.numpy)
	
	def __mul__(self, rhs):
		return Point2DArray(self.numpy * rhs)
	
	def __rmul__(self, lhs):
		return self * lhs
	
	def __truediv__(self, rhs):
		return Point2DArray(self.numpy / rhs)

	def append(self, other):
		if isinstance(other, Point2D):
			self.array.append(other)
		elif isinstance(other, Point2DArray):
			self.array.extend(other)
		elif other is None:
			pass
		else:
			raise NotImplementedError
	
	def moved(self, dpos):
		return Point2DArray([p.moved(dpos) for p in self])

	def rotated(self, dth):
		return Point2DArray([p.rotated(dth) for p in self])
	
	def support_point(self, direction):
		n = Point2D(direction)
		dmax = -np.inf
		support = None
		for point in self:
			d = point.dot(n)
			if d > dmax:
				dmax = d
				support = point
		return support
	
	def min_angle_point(self, point):
		p0 = Point2D(point)
		thc = (self.center - p0).angle
		thmin = np.inf
		p_min = None
		for point in self:
			if point == p0:
				continue
			thp = (point - p0).angle
			th = Angle(thp - thc).in180.rad
			if th < thmin:
				thmin = th
				p_min = point
		return p_min
	