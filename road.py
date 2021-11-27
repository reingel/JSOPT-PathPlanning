import numpy as np
import matplotlib.pyplot as plt
import bisect
from vector import *
from cubic_spline_planner import Spline2D

DEGREE = np.pi/180
ANGLE90 = np.pi/2

class Road(Spline2D):
	def __init__(self, wx, wy, hwidth, ds=1.0):
		super().__init__(wx, wy)
		self.hwidth = hwidth
		self.ds = ds
		self.s_list = np.arange(0, self.smax, ds)
		self.center_points = self._calc_center_points(ds)
		self.yaws = self._calc_yaws(ds)
		yaw_p90 = self.yaws + ANGLE90
		yaw_m90 = self.yaws - ANGLE90
		self.left_points = self.center_points + hwidth * np.column_stack([np.cos(yaw_p90), np.sin(yaw_p90)])
		self.right_points = self.center_points + hwidth * np.column_stack([np.cos(yaw_m90), np.sin(yaw_m90)])

	@property
	def smax(self):
		if len(self.s) > 0:
			return self.s[-1]
		else:
			return 0.

	def _calc_center_point(self, s):
		return Vector2D(*self.calc_position(s))
	
	def _calc_center_points(self, ds):
		ps = []
		for s in self.s_list:
			ps.append(self._calc_center_point(s).numpy)
		return np.array(ps)
	
	def _calc_yaw(self, s):
		return self.calc_yaw(s)
	
	def _calc_yaws(self, ds):
		yaws = []
		for s in self.s_list:
			yaws.append(self._calc_yaw(s))
		return np.array(yaws)
	
	def _get_index(self, s):
		return max(bisect.bisect(self.s_list, s) - 1, 0)

	def get_center_point(self, s):
		idx = self._get_index(s)
		center = self.center_points[idx]
		return Vector2D(*center)
	
	def get_yaw(self, s):
		idx = self._get_index(s)
		yaw = self.yaws[idx]
		return yaw
	
	def convert_xy2tn(self, s, vector: Vector2D):
		idx = self._get_index(s)
		yaw = self.yaws[idx]
		vtn = vector.rotated(-yaw)
		return vtn
	
	def convert_tn2xy(self, s, vector: Vector2D):
		idx = self._get_index(s)
		yaw = self.yaws[idx]
		vxy = vector.rotated(yaw)
		return vxy
	
	def get_projection_distance(self, p: Vector2DArray):
		center_x = self.center_points[:,0]
		center_y = self.center_points[:,1]
		dist = np.sqrt(np.min((column_vector(p.x) - row_vector(center_x))**2 + (column_vector(p.y) - row_vector(center_y))**2, axis=1))
		return dist
	
	def get_projection_point(self, p: Vector2D):
		center_x = self.center_points[:,0]
		center_y = self.center_points[:,1]
		imin = np.argmin((column_vector(p.x) - row_vector(center_x))**2 + (column_vector(p.y) - row_vector(center_y))**2, axis=1)
		return self.s_list[imin], Vector2D(*self.center_points[imin][0])

	def draw(self, axes, ds, dmax):
		axes.plot(self.left_points[:,0], self.left_points[:,1], color='black')
		axes.plot(self.center_points[:,0], self.center_points[:,1], linestyle='dotted', color='black')
		axes.plot(self.right_points[:,0], self.right_points[:,1], color='black')
	

if __name__ == '__main__':
	# way points
	wx = [0.0, 10.0, 20.5, 35.0, 70.5]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]
	hwidth = 3

	road = Road(wx, wy, hwidth)

	print(road.smax)
	print(road.s_list)

	fig, axes = plt.subplots()
	axes.set_aspect(1)
	axes.set_xlim(-5, 75)
	axes.set_ylim(-15, 15)

	road.draw(axes, ds=1, dmax=3)
	Vector2D(70, 5).draw(axes, color='red', marker='o')

	# for x in np.arange(0,70,1):
	# 	p = Vector2D(x,-2)
	# 	p.draw(axes, color='green', marker='o')
	# 	ret = road.get_projection(p)
	# 	if ret is not None:
	# 		s, c, dist = ret
	# 		print(s, p, c, dist)
	# 		c.draw(axes, color='red', marker='o')
	
	p = Vector2DArray(((10,1), (50,6)))
	p.draw(axes, color='green', marker='o')
	dist = road.get_projection_distance(p)
	print(dist)

	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])

	plt.show()

	

	
