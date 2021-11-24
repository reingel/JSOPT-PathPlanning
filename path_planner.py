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
		T = self.final_time

		pns = self.current_state.pos.n
		vns = self.current_state.vel.n
		ans = self.current_state.acc.n
		pne = self.final_lat_pos
		vne = 0.
		ane = 0.

		dpn = pne - pns - vns * T - 1/2 * ans * T**2
		dvn = vne - vns - ans * T
		dan = ane - ans

		self.lat_poly_coef = np.array([
			pns,
			vns,
			1/2*ans,
			( 10*dpn - 4*T*dvn + 1/2*T**2*dan) / T**3,
			(-15*dpn + 7*T*dvn -     T**2*dan) / T**4,
			(  6*dpn - 3*T*dvn + 1/2*T**2*dan) / T**5,
		])
	
	def _calc_long_poly_coef(self):
		"""
		Use a quartic polynomial to model the longitudinal position
		"""
		T = self.final_time

		pts = self.current_state.pos.t
		vts = self.current_state.vel.t
		ats = self.current_state.acc.t
		vte = self.final_long_vel
		ate = 0.

		dvt = vte - vts - ats * T
		dat = ate - ats

		self.long_poly_coef = np.array([
			pts,
			vts,
			1/2*ats,
			(     dvt - 1/3*T*dat) / T**2,
			(-1/2*dvt + 1/4*T*dat) / T**3,
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
		J1 = 720*an5**2*T**5 \
			+ 720*an4*an5*T**4 \
			+ (192*an4**2 + 240*an3*an5 + 192*at4**2)*T**3 \
			+ 144*(an3*an4 + at3*at4)*T**2 \
			+ 36*(an3**2 + at3**2)*T
		return J1
	
	def get_object_function_time(self):
		T = self.final_time
		J2 = T
		return J2
	
	def get_object_function_final_lat_pos(self):
		pen = self.final_lat_pos
		J3 = pen**2
		return J3
	
	def get_object_function_final_long_vel(self, ref_vel):
		vet = self.final_long_vel
		J4 = (vet - ref_vel)**2
		return J4
	
	def get_object_function(self, ref_vel, kt, kd, kv):
		J1 = self.get_object_function_jerk()
		J2 = self.get_object_function_time()
		J3 = self.get_object_function_final_lat_pos()
		J4 = self.get_object_function_final_long_vel(ref_vel)

		J = J1 + kt*J2 + kd*J3 + kv*J4

		return J
	
	def get_object_function_jerk_gradient(self):
		# current and final states
		T = self.final_time

		psn = self.current_state.pos.n

		vsn = self.current_state.vel.n
		vst = self.current_state.vel.t

		asn = self.current_state.acc.n
		ast = self.current_state.acc.t

		pen = self.final_lat_pos
		
		ven = 0.
		vet = self.final_long_vel

		aen = 0.
		aet = 0.

		# coefficients of lateral(normal) polynomial
		dpn = pen - psn - vsn*T - 1/2*asn*T**2
		dvn = ven - vsn - asn*T
		dan = aen - asn

		an3_num =  10*dpn - 4*T*dvn + 1/2*T**2*dan
		an4_num = -15*dpn + 7*T*dvn -     T**2*dan
		an5_num =   6*dpn - 3*T*dvn + 1/2*T**2*dan
		an3 = an3_num / T**3
		an4 = an4_num / T**4
		an5 = an5_num / T**5

		# coefficients of longitudinal(tangent) polynomial
		dvt = vet - vst - ast*T
		dat = aet - ast

		at3_num =      dvt - 1/3*T*dat
		at4_num = -1/2*dvt + 1/4*T*dat
		at3 = at3_num / T**2
		at4 = at4_num / T**3

		# gradients of coefficients of lateral(normal) polynomial
		dpn_T   = -vsn - asn*T
		dpn_pen = 1

		dvn_T   = -asn
		dvn_pen = 0

		dan_T   = 0
		dan_pen = 0

		an3_T   = (( 10*dpn_T   - 4*dvn - 4*T*dvn_T   +   T*dan + 1/2*T**2*dan_T  ) * T - an3_num * 3) / T**4
		an3_pen =  ( 10*dpn_pen -         4*T*dvn_pen +           1/2*T**2*dan_pen)                    / T**3

		an4_T   = ((-15*dpn_T   + 7*dvn + 7*T*dvn_T   - 2*T*dan -     T**2*dan_T  ) * T - an4_num * 4) / T**5
		an4_pen =  (-15*dpn_pen +         7*T*dvn_pen -               T**2*dan_pen)                    / T**4

		an5_T   = ((  6*dpn_T   - 3*dvn - 3*T*dvn_T   +   T*dan + 1/2*T**2*dan_T  ) * T - an5_num * 5) / T**6
		an5_pen =  (  6*dpn_pen -         3*T*dvn_pen +           1/2*T**2*dan_pen)                    / T**5

		# gradients of coefficients of longitudinal(tangent) polynomial
		dvt_T   = -ast
		dvt_vet = 1

		dat_T   = 0
		dat_vet = 0

		at3_T   = ((     dvt_T   - 1/3*dat - 1/3*T*dat_T  ) * T - at3_num * 2) / T**3
		at3_vet =  (     dvt_vet -           1/3*T*dat_vet)                    / T**4
	
		at4_T   = ((-1/2*dvt_T   + 1/4*dat + 1/4*T*dat_T  ) * T - at4_num * 2) / T**3
		at4_vet =  (-1/2*dvt_vet +           1/4*T*dat_vet)                    / T**4

		# gradient of object function of jerk
		J1_T = 720*(2*an5*an5_T*T**5 + 5*an5**2*T**4) \
			+ 720*(an4_T*an5*T**4 + an4*an5_T*T**4 + 4*an4*an5*T**3) \
			+ (192*2*an4*an4_T + 240*(an3_T*an5 + an3*an5_T) + 192*2*at4*at4_T)*T**3 \
			+ 3*(192*an4**2 + 240*an3*an5 + 192*at4**2)*T**2 \
			+ 144*(an3_T*an4 + an3*an4_T + at3_T*at4 + at3*at4_T)*T**2 \
			+ 144*2*(an3*an4 + at3*at4)*T \
			+ 36*(2*an3*an3_T + 2*at3*at3_T)*T \
			+ 36*(an3**2 + at3**2)
		
		J1_pen = 720*2*an5*an5_pen*T**5 + 720*(an4_pen*an5 + an4*an5_pen)*T**4 \
			+ (192*2*an4*an4_pen + 240*(an3_pen*an5 + an3*an5_pen))*T**3 \
			+ 144*(an3_pen*an4 + an3*an4_pen)*T**2 + 36*2*an3*an3_pen*T
		
		J1_vet = 192*2*at4*at4_vet*T**3 + 144*(at3_vet*at4 + at3*at4_vet)*T**2 \
			+ 36*2*at3*at3_vet*T
		
		J1_x = np.array([J1_T, J1_pen, J1_vet])

		return J1_x

	def get_object_function_time_gradient(self):
		J2_x = np.array([1, 0, 0])
		return J2_x
	
	def get_object_function_final_lat_pos_gradient(self):
		pen = self.final_lat_pos
		J3_x = np.array([0, 2*pen, 0])
		return J3_x
	
	def get_object_function_final_long_vel_gradient(self, ref_vel):
		vet = self.final_long_vel
		J4_x = np.array([0, 0, 2*(vet - ref_vel)])
		return J4_x
	
	def get_object_function_gradient(self, ref_vel, kt, kd, kv):
		J1_x = self.get_object_function_jerk_gradient()
		J2_x = self.get_object_function_time_gradient()
		J3_x = self.get_object_function_final_lat_pos_gradient()
		J4_x = self.get_object_function_final_long_vel_gradient(ref_vel)

		J_x = J1_x + kt*J2_x + kd*J3_x + kv*J4_x

		return J_x
	

	def get_constraint_function_values(self, t):
		pass

	def get_constraint_function_jacobian(self, t):
		pass


if __name__ == '__main__':
	kph = 1/3.6

	road = Road(Point2DArray(
		(0.0, 0.0),
		(10.0, -6.0),
		(20.5, 5.0),
		(35.0, 6.5),
		(70.5, 0.0),
	))
	vref = 20*kph
	kt = 1
	kd = 1
	kv = 1

	planner = PathPlanner(road)
	current_state = VehicleState()

	T = 30
	pen = 5
	vet = 27*kph

	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=pen, final_long_vel=vet)

	J1 = planner.get_object_function_jerk()
	J2 = planner.get_object_function_time()
	J3 = planner.get_object_function_final_lat_pos()
	J4 = planner.get_object_function_final_long_vel(vref)
	J = planner.get_object_function(vref, kt, kd, kv)

	print(f'{J=:6.4f} {J1=:6.4f} {J2=:6.4f} {J3=:6.4f} {J4=:6.4f}')

	# analytic differentiation
	dJ = planner.get_object_function_gradient(vref, kt, kd, kv)
	dJ_str = ' '.join([str(s).rjust(5) for s in np.round(dJ, 3)])
	print(f'{dJ_str}')

	# numerical differentiation
	dT   =   T*0.0001
	dpen = pen*0.0001
	dvet = vet*0.0001

	planner.get_optimal_path(current_state=current_state, final_time=T+dT, final_lat_pos=pen, final_long_vel=vet)
	J1u = planner.get_object_function(vref, kt, kd, kv)
	planner.get_optimal_path(current_state=current_state, final_time=T-dT, final_lat_pos=pen, final_long_vel=vet)
	J1d = planner.get_object_function(vref, kt, kd, kv)
	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=pen+dpen, final_long_vel=vet)
	J2u = planner.get_object_function(vref, kt, kd, kv)
	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=pen-dpen, final_long_vel=vet)
	J2d = planner.get_object_function(vref, kt, kd, kv)
	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=pen, final_long_vel=vet+dvet)
	J3u = planner.get_object_function(vref, kt, kd, kv)
	planner.get_optimal_path(current_state=current_state, final_time=T, final_lat_pos=pen, final_long_vel=vet-dvet)
	J3d = planner.get_object_function(vref, kt, kd, kv)
	
	dJ_numeric = np.array([(J1u-J1d)/dT/2, (J2u-J2d)/dpen/2, (J3u-J3d)/dvet/2])
	dJ_numeric_str = ' '.join([str(s).rjust(5) for s in np.round(dJ_numeric, 3)])
	print(f'{dJ_numeric_str}')


	# draw path
	ts = np.linspace(0, T, 10)

	plt.figure(1)
	plt.clf()
	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])

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
		# plt.plot([pos.t, pos.t - vel_unit.n * R], [pos.n, pos.n + vel_unit.t * R], 'g-')
	
	plt.show()