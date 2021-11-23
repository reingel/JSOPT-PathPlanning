import numpy as np
import matplotlib.pyplot as plt
from math import floor
import typing
from point2d import Point2D, Point2DArray
from cubic_spline_planner import Spline2D


class Road(Spline2D):
	def __init__(self, points: Point2DArray):
		"""
		ARGUMENTS
			points: Point2DArray object to represent 2D points
			ds: delta distance
		"""
		self.array = []
		for p in points:
			self.array.append(p.numpy)
		self.array = np.array(self.array)
		wx = self.array[:,0]
		wy = self.array[:,1]
		super().__init__(wx, wy)

	@property
	def s_max(self):
		if len(self.s) > 0:
			return self.s[-1]
		else:
			return 0.

	def s_list(self, ds=None):
		return np.arange(self.s_max // ds + 1) * ds

	def get_waypoint(self, s):
		"""
		ARGUMENTS
			s: travel distance along the center line
		RETURN
			x, y, phi: global position and heading angle
		"""
		return Point2D(self.calc_position(s))

	def is_on_road(self, lat, long):
		"""
		ARGUMENTS
			lat: lateral location in Frenet coordinates
			long: longitudinal location in Frenet coordiantes
		RETURN
			[bool]: True if the input location is on this road
		"""
		pass
	

if __name__ == '__main__':
	# way points
	wx = [0.0, 10.0, 20.5, 35.0, 70.5]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]

	road = Road(Point2DArray(
		(0.0, 0.0),
		(10.0, -6.0),
		(20.5, 5.0),
		(35.0, 6.5),
		(70.5, 0.0),
	))

	s_max = road.s_max
	s_list = road.s_list(ds=5)

	print(s_max)
	print(s_list)

	fig, axes = plt.subplots()
	axes.set_aspect(1)
	axes.set_xlim(-5, 75)
	axes.set_ylim(-10, 10)

	for s in s_list:
		road.get_waypoint(s).draw(axes)
	
	plt.grid(True)
	plt.pause(1)

	

	
