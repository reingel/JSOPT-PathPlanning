from typing import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction
from road import Road
from vehicle_state import VehicleState
from point2d import PointNT, Point2DArray


class PathPlanner:
	def __init__(self, road: Road):
		self.road = road
	
	def get_optimal_path(self, current_state: VehicleState, final_time: float, final_lat_pos: float, final_long_vel: float):
		self.current_state = current_state
		self.final_time = final_time
		self.final_lat_pos = final_lat_pos
		self.final_long_vel = final_long_vel
		self._calc_lat_poly_coef()
		self._calc_long_poly_coef()

	def _calc_lat_poly_coef(self):
		"""
		Use a quintic polynomial to model the lateral position
		"""
		pns = self.current_state.pos.n
		vns = self.current_state.vel.n
		ans = self.current_state.acc.n
		pne = self.final_lat_pos
		vne = 0.
		ane = 0.
		T = self.final_time

		dpn = pne - pns - vns * T - 1/2 * ans * T**2
		dvn = vne - vns - ans * T
		dan = ane - ans

		self.lat_poly_coef = np.array([
			pns,
			vns,
			1/2*ans,
			(10*dpn - 4*T*dvn + 1/2*T**2*dan) / T**3,
			(-15*dpn + 7*T*dvn - T**2 * dan) / T**4,
			(6*dpn - 3*T*dvn + 1/2*T**2*dan) / T**5,
		])
	
	def _calc_long_poly_coef(self):
		"""
		Use a quartic polynomial to model the longitudinal position
		"""
		pts = self.current_state.pos.t
		vts = self.current_state.vel.t
		ats = self.current_state.acc.t
		vte = self.final_long_vel
		ate = 0.
		T = self.final_time

		dvt = vte - vts - ats * T
		dat = ate - ats

		self.long_poly_coef = np.array([
			pts,
			vts,
			1/2*ats,
			(dvt - 1/3*T*dat) / T**2,
			(-1/2*dvt + 1/4*T * dat) / T**3,
		])

	def get_lat_long_pos(self, t):
		pn = np.dot(self.lat_poly_coef, np.array([1, t, t**2, t**3, t**4, t**5]))
		pt = np.dot(self.long_poly_coef, np.array([1, t, t**2, t**3, t**4]))
		return PointNT(pn, pt)
	
	def get_lat_long_vel(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		vn = np.dot(np.array([an1, 2*an2, 3*an3, 4*an4, 5*an5]), np.array([1, t, t**2, t**3, t**4]))
		vt = np.dot(np.array([at1, 2*at2, 3*at3, 4*at4]), np.array([1, t, t**2, t**3]))
		return PointNT(vn, vt)
	
	def get_lat_long_acc(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		an = np.dot(np.array([2*an2, 6*an3, 12*an4, 20*an5]), np.array([1, t, t**2, t**3]))
		at = np.dot(np.array([2*at2, 6*at3, 12*at4]), np.array([1, t, t**2]))
		return PointNT(an, at)
	
	def get_lat_long_jrk(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		jn = np.dot(np.array([6*an3, 24*an4, 60*an5]), np.array([1, t, t**2]))
		jt = np.dot(np.array([6*at3, 24*at4]), np.array([1, t]))
		return PointNT(jn, jt)
	
	def get_curvature(self, t):
		vel = self.get_lat_long_vel(t)
		acc = self.get_lat_long_acc(t)
		kappa = (vel.t * acc.n - vel.n * acc.t) / (vel.n**2 + vel.t**2)**(3/2)
		return kappa
	
	def get_object_function_jerk(self):
		T = self.final_time
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		J1 = 720*an5**2*T**5 + \
			610*an4*an5*T**4 + \
			(192*an4**2 + 240*an3*an5 + 192*at4**2)*T**3 + \
			144*(an3*an4 + at3*at4)*T**2 + \
			36*(an3**2 + at3**2)*T
		return J1
	
	def get_object_function_jerk_gradient(self):
		pns = self.current_state.pos.n
		vns = self.current_state.vel.n
		ans = self.current_state.acc.n
		pne = self.final_lat_pos
		vne = 0.
		ane = 0.
		T = self.final_time

		dpn_T = -vns - ans*T
		dvn_T = -ans
		dan_T = 0
		# a3n_T = 
	
	def get_object_function_value(self, t):
		pass

	def get_object_function_gradient(self, t):
		pass

	def get_constraint_function_values(self, t):
		pass

	def get_constraint_function_jacobian(self, t):
		pass


if __name__ == '__main__':
	road = Road(Point2DArray(
		(0.0, 0.0),
		(10.0, -6.0),
		(20.5, 5.0),
		(35.0, 6.5),
		(70.5, 0.0),
	))
	planner = PathPlanner(road)
	
	current_state = VehicleState()

	T = 10
	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=5, final_long_vel=1)
	J1 = planner.get_object_function_jerk()
	print(f'{J1=}')

	ts = np.linspace(0, T, T+1)

	plt.figure(1)
	plt.clf()
	plt.grid(True)

	for t in ts:
		pos = planner.get_lat_long_pos(t)
		vel = planner.get_lat_long_vel(t)
		acc = planner.get_lat_long_acc(t)
		jrk = planner.get_lat_long_jrk(t)
		kappa = planner.get_curvature(t)
		R = min(1/(kappa+1e-6)/10, 10)
		vel_unit = vel.normalized
		plt.plot(pos.t, pos.n, 'b.')
		plt.plot([pos.t, pos.t + vel.t], [pos.n, pos.n + vel.n], 'r-')
		# plt.plot([pos.t, pos.t + acc.t], [pos.n, pos.n + acc.n], 'g-')
		# plt.plot([pos.t, pos.t + jrk.t], [pos.n, pos.n + jrk.n], 'k-')
		plt.plot([pos.t, pos.t - vel_unit.n * R], [pos.n, pos.n + vel_unit.t * R], 'g-')
		print(kappa, R)
	
	plt.show()