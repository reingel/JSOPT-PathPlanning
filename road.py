import numpy as np
import matplotlib.pyplot as plt
from math import floor
import typing

from numpy.core.defchararray import center
from vector import Vector2D
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
	
	def get_projection(self, p: Vector2D):
		for center, yaw in zip(self.center_points, self.yaws):
			v = p - Vector2D(*center)
			if abs(abs(v.angle - yaw) - ANGLE90) < 3*DEGREE: # TODO
				return Vector2D(*center), v.norm
		return None

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
	Vector2D(70, -5).draw(axes, color='red', marker='o')

	p = Vector2D(10,-5)
	p.draw(axes, color='red', marker='o')
	ret = road.get_projection(p)
	if ret is not None:
		c, dist = ret
		print(c, dist)
		c.draw(axes, color='red', marker='o')
	
	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])

	plt.show()

	

	
