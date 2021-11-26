import numpy as np
import matplotlib.pyplot as plt
from math import floor
import typing

from numpy.core.defchararray import center
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
	
	def get_waypoints(self, ds):
		s_list = self.s_list(ds=ds)

		ps = []
		for s in s_list:
			ps.append(self.get_waypoint(s))
		
		return np.array(ps)
	
	def get_yaw(self, s):
		return self.calc_yaw(s)
	
	def get_yaws(self, ds):
		s_list = self.s_list(ds=ds)

		yaws = []
		for s in s_list:
			yaws.append(self.get_yaw(s))
		
		return np.array(yaws)
	
	def get_cartesian_pos(self, s, d):
		center_pos = self.calc_position(s)
		yaw = self.calc_yaw(s)
		pos = center_pos + d * np.array([np.cos(yaw + np.pi/2), np.sin(yaw + np.pi/2)])
		return Point2D(pos)

	
	def draw(self, axes, ds, dmax):
		yaws = self.get_yaws(ds)
		center = self.get_waypoints(ds)
		left = center + dmax * np.vstack([np.cos(yaws + np.pi/2), np.sin(yaws + np.pi/2)]).transpose()
		right = center + dmax * np.vstack([np.cos(yaws - np.pi/2), np.sin(yaws - np.pi/2)]).transpose()

		axes.plot(left[:,0], left[:,1], color='black')
		axes.plot(center[:,0], center[:,1], linestyle='dotted', color='black')
		axes.plot(right[:,0], right[:,1], color='black')
	

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
	axes.set_ylim(-15, 15)

	road.draw(axes, ds=1, dmax=3)
	road.get_cartesian_pos(75, 3).draw(axes)
	road.get_cartesian_pos(75, -3).draw(axes)
	
	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])

	plt.show()

	

	
