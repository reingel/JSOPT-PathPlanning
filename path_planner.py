import sys
sys.path.append('./JSOPT')

from typing import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction
from road import Road
from vehicle_state import VehicleState
from vector import Vector2D
from JSOPT.lagrangian import lagrangian
import time
from JSOPT.constant import *
from JSOPT.linalg import Vector, normalized
from JSOPT.line_search import bisection

EPS = 1e-3

def max0(x):
	return max(x, 0)

def vec_max_zero(x: np.ndarray):
	return np.array([max(xi, 0) for xi in x])


class PathPlanner:
	def __init__(self, road: Road, ref_vel, kt, kd, kv, dmax, vmax, amax, kmax):
		self.road = road

		self.ref_vel = ref_vel
		self.kt = kt
		self.kd = kd
		self.kv = kv

		self.dmax = dmax
		self.vmax = vmax
		self.amax = amax
		self.kmax = kmax
	
	def set_current_state(self, s0, sf, current_state: VehicleState):
		self.s0 = s0 # start position along road centerline
		self.sf = sf # final position along road centerline
		p0 = self.road.get_center_point(s=self.s0)
		converted_state = VehicleState(
			pos=self.road.convert_tn2xy(s=self.s0, vtn=current_state.pos) + p0,
			vel=self.road.convert_tn2xy(s=self.s0, vtn=current_state.vel),
			acc=self.road.convert_tn2xy(s=self.s0, vtn=current_state.acc),
		)
		self.current_state = converted_state

	def generate_path(self, x: np.ndarray):
		self.x = x
		final_time, final_lat_pos, final_long_vel = x
		self.final_time = final_time
		self.final_lat_pos = final_lat_pos
		self.final_long_vel = final_long_vel

		final_state = VehicleState(
			pos=Vector2D(0, self.final_lat_pos),
			vel=Vector2D(self.final_long_vel, 0),
			acc=Vector2D(0, 0),
		)

		pf = self.road.get_center_point(s=self.sf)
		converted_state = VehicleState(
			pos=self.road.convert_tn2xy(s=self.sf, vtn=final_state.pos) + pf,
			vel=self.road.convert_tn2xy(s=self.sf, vtn=final_state.vel),
			acc=self.road.convert_tn2xy(s=self.sf, vtn=final_state.acc),
		)
		self.final_state = converted_state

		self._calc_x_poly_coef()
		self._calc_y_poly_coef()

	def _calc_x_poly_coef(self):
		"""
		Use a quintic polynomial to model the x position
		"""
		T = self.final_time

		pxs = self.current_state.pos.x
		vxs = self.current_state.vel.x
		axs = self.current_state.acc.x
		pxe = self.final_state.pos.x
		vxe = self.final_state.vel.x
		axe = self.final_state.acc.x

		dpx = pxe - pxs - vxs * T - 1/2 * axs * T**2
		dvx = vxe - vxs - axs * T
		dax = axe - axs

		self.x_poly_coef = np.array([
			pxs,
			vxs,
			1/2*axs,
			( 10*dpx - 4*T*dvx + 1/2*T**2*dax) / T**3,
			(-15*dpx + 7*T*dvx -     T**2*dax) / T**4,
			(  6*dpx - 3*T*dvx + 1/2*T**2*dax) / T**5,
		])
	
	def _calc_y_poly_coef(self):
		"""
		Use a quintic polynomial to model the y position
		"""
		T = self.final_time

		pys = self.current_state.pos.y
		vys = self.current_state.vel.y
		ays = self.current_state.acc.y
		pye = self.final_state.pos.y
		vye = self.final_state.vel.y
		aye = self.final_state.acc.y

		dpy = pye - pys - vys * T - 1/2 * ays * T**2
		dvy = vye - vys - ays * T
		day = aye - ays

		self.y_poly_coef = np.array([
			pys,
			vys,
			1/2*ays,
			( 10*dpy - 4*T*dvy + 1/2*T**2*day) / T**3,
			(-15*dpy + 7*T*dvy -     T**2*day) / T**4,
			(  6*dpy - 3*T*dvy + 1/2*T**2*day) / T**5,
		])
	
	# def _calc_y_poly_coef(self):
	# 	"""
	# 	Use a quartic polynomial to model the longitudinal position
	# 	"""
	# 	T = self.final_time

	# 	pts = self.current_state.pos.x
	# 	vts = self.current_state.vel.x
	# 	ats = self.current_state.acc.x
	# 	vte = self.final_y_vel
	# 	ate = 0.

	# 	dvt = vte - vts - ats * T
	# 	dat = ate - ats

	# 	self.y_poly_coef = np.array([
	# 		pts,
	# 		vts,
	# 		1/2*ats,
	# 		(     dvt - 1/3*T*dat) / T**2,
	# 		(-1/2*dvt + 1/4*T*dat) / T**3,
	# 	])

	def get_x_y_pos(self, t):
		px = np.dot(self.x_poly_coef, t**np.arange(6))
		py = np.dot(self.y_poly_coef, t**np.arange(6))
		return Vector2D(px, py)
	
	def get_x_y_vel(self, t):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		vx = np.dot(np.array([ax1, 2*ax2, 3*ax3, 4*ax4, 5*ax5]), t**np.arange(5))
		vt = np.dot(np.array([ay1, 2*ay2, 3*ay3, 4*ay4, 5*ay5]), t**np.arange(5))
		return Vector2D(vx, vt)
	
	def get_x_y_acc(self, t):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		ax = np.dot(np.array([2*ax2, 6*ax3, 12*ax4, 20*ax5]), t**np.arange(4))
		ay = np.dot(np.array([2*ay2, 6*ay3, 12*ay4, 20*ay5]), t**np.arange(4))
		return Vector2D(ax, ay)
	
	def get_x_y_jrk(self, t):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		jx = np.dot(np.array([6*ax3, 24*ax4, 60*ax5]), t**np.arange(3))
		jy = np.dot(np.array([6*ay3, 24*ay4, 60*ay5]), t**np.arange(3))
		return Vector2D(jx, jy)
	
	def get_curvature(self, t):
		vel = self.get_x_y_vel(t)
		acc = self.get_x_y_acc(t)
		kappa = (vel.x * acc.y - vel.y * acc.x) / ((vel.y**2 + vel.x**2)**(3/2) + EPS)
		return kappa
	
	def get_x_y_poses(self, ts):
		px = np.dot(self.x_poly_coef, ts**np.arange(6).reshape((-1,1)))
		py = np.dot(self.y_poly_coef, ts**np.arange(6).reshape((-1,1)))
		return np.column_stack((px, py))
	
	def get_x_y_vels(self, ts):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		vx = np.dot(np.array([ax1, 2*ax2, 3*ax3, 4*ax4, 5*ax5]), ts**np.arange(5).reshape((-1,1)))
		vt = np.dot(np.array([ay1, 2*ay2, 3*ay3, 4*ay4, 5*ay5]), ts**np.arange(5).reshape((-1,1)))
		return np.column_stack((vx, vt))
	
	def get_x_y_accs(self, ts):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		ax = np.dot(np.array([2*ax2, 6*ax3, 12*ax4, 20*ax5]), ts**np.arange(4).reshape((-1,1)))
		ay = np.dot(np.array([2*ay2, 6*ay3, 12*ay4, 20*ay5]), ts**np.arange(4).reshape((-1,1)))
		return np.column_stack((ax, ay))
	
	def get_x_y_jrks(self, ts):
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		jx = np.dot(np.array([6*ax3, 24*ax4, 60*ax5]), ts**np.arange(3).reshape((-1,1)))
		jy = np.dot(np.array([6*ay3, 24*ay4, 60*ay5]), ts**np.arange(3).reshape((-1,1)))
		return np.column_stack((jx, jy))
	
	def get_curvatures(self, ts):
		vel = self.get_x_y_vels(ts)
		acc = self.get_x_y_accs(ts)
		vx, vt = vel[:,0], vel[:,1]
		ax, ay = acc[:,0], acc[:,1]
		kappa = (vt * ax - vx * ay) / ((vx**2 + vt**2)**(3/2) + EPS)
		return kappa
	
	def get_object_function_jerk(self):
		T = self.final_time
		ax0, ax1, ax2, ax3, ax4, ax5 = self.x_poly_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.y_poly_coef
		# J1 = 720*ax5**2*T**5 \
		# 	+ 720*ax4*ax5*T**4 \
		# 	+ (192*ax4**2 + 240*ax3*ax5 + 192*ay4**2)*T**3 \
		# 	+ 144*(ax3*ax4 + ay3*ay4)*T**2 \
		# 	+ 36*(ax3**2 + ay3**2)*T
		J1 = np.dot(
			np.array([
				60**2*(ax5**2 + ay5**2)/5,
				2*24*60*(ax4*ax5 + ay4*ay5)/4,
				(24**2*(ax4**2 + ay4**2) + 2*6*60*(ax3*ax5 + ay3*ay5))/3,
				2*6*24*(ax3*ax4 + ay3*ay4)/2,
				36*(ax3**2 + ay3**2),
			]),
			T**np.arange(5).reshape((-1,1))
		)
			
		return J1
	
	def get_object_function_time(self):
		T = self.x[0]
		J2 = T
		return J2
	
	def get_object_function_final_n_pos(self):
		pen = self.x[1]**2
		J3 = pen**2
		return J3
	
	def get_object_function_final_t_vel(self):
		vet = self.x[2]**2
		J4 = (vet - self.ref_vel)**2
		return J4
	
	def get_object_function_value(self, x):
		self.generate_path(x)

		J1 = self.get_object_function_jerk()
		J2 = self.get_object_function_time()
		J3 = self.get_object_function_final_n_pos()
		J4 = self.get_object_function_final_t_vel()

		J = J1 + kt*J2 + kd*J3 + kv*J4

		return J
	
	# def get_object_function_jerk_gradient(self):
	# 	# current and final states
	# 	T = self.final_time

	# 	psn = self.current_state.pos.y

	# 	vsn = self.current_state.vel.y
	# 	vst = self.current_state.vel.x

	# 	asn = self.current_state.acc.y
	# 	ast = self.current_state.acc.x

	# 	pen = self.final_lat_pos
		
	# 	ven = 0.
	# 	vet = self.final_long_vel

	# 	aen = 0.
	# 	aet = 0.

	# 	# coefficients of lateral(normal) polynomial
	# 	dpn = pen - psn - vsn*T - 1/2*asn*T**2
	# 	dvn = ven - vsn - asn*T
	# 	dan = aen - asn

	# 	an3_num =  10*dpn - 4*T*dvn + 1/2*T**2*dan
	# 	an4_num = -15*dpn + 7*T*dvn -     T**2*dan
	# 	an5_num =   6*dpn - 3*T*dvn + 1/2*T**2*dan
	# 	an3 = an3_num / T**3
	# 	an4 = an4_num / T**4
	# 	an5 = an5_num / T**5

	# 	# coefficients of longitudinal(tangent) polynomial
	# 	dvt = vet - vst - ast*T
	# 	dat = aet - ast

	# 	at3_num =      dvt - 1/3*T*dat
	# 	at4_num = -1/2*dvt + 1/4*T*dat
	# 	at3 = at3_num / T**2
	# 	at4 = at4_num / T**3

	# 	# gradients of coefficients of lateral(normal) polynomial
	# 	dpn_T   = -vsn - asn*T
	# 	dpn_pen = 1

	# 	dvn_T   = -asn
	# 	dvn_pen = 0

	# 	dan_T   = 0
	# 	dan_pen = 0

	# 	an3_T   = (( 10*dpn_T   - 4*dvn - 4*T*dvn_T   +   T*dan + 1/2*T**2*dan_T  ) * T - an3_num * 3) / T**4
	# 	an3_pen =  ( 10*dpn_pen -         4*T*dvn_pen +           1/2*T**2*dan_pen)                    / T**3

	# 	an4_T   = ((-15*dpn_T   + 7*dvn + 7*T*dvn_T   - 2*T*dan -     T**2*dan_T  ) * T - an4_num * 4) / T**5
	# 	an4_pen =  (-15*dpn_pen +         7*T*dvn_pen -               T**2*dan_pen)                    / T**4

	# 	an5_T   = ((  6*dpn_T   - 3*dvn - 3*T*dvn_T   +   T*dan + 1/2*T**2*dan_T  ) * T - an5_num * 5) / T**6
	# 	an5_pen =  (  6*dpn_pen -         3*T*dvn_pen +           1/2*T**2*dan_pen)                    / T**5

	# 	# gradients of coefficients of longitudinal(tangent) polynomial
	# 	dvt_T   = -ast
	# 	dvt_vet = 1

	# 	dat_T   = 0
	# 	dat_vet = 0

	# 	at3_T   = ((     dvt_T   - 1/3*dat - 1/3*T*dat_T  ) * T - at3_num * 2) / T**3
	# 	at3_vet =  (     dvt_vet -           1/3*T*dat_vet)                    / T**4
	
	# 	at4_T   = ((-1/2*dvt_T   + 1/4*dat + 1/4*T*dat_T  ) * T - at4_num * 2) / T**3
	# 	at4_vet =  (-1/2*dvt_vet +           1/4*T*dat_vet)                    / T**4

	# 	# gradient of object function of jerk
	# 	J1_T = 720*(2*an5*an5_T*T**5 + 5*an5**2*T**4) \
	# 		+ 720*(an4_T*an5*T**4 + an4*an5_T*T**4 + 4*an4*an5*T**3) \
	# 		+ (192*2*an4*an4_T + 240*(an3_T*an5 + an3*an5_T) + 192*2*at4*at4_T)*T**3 \
	# 		+ 3*(192*an4**2 + 240*an3*an5 + 192*at4**2)*T**2 \
	# 		+ 144*(an3_T*an4 + an3*an4_T + at3_T*at4 + at3*at4_T)*T**2 \
	# 		+ 144*2*(an3*an4 + at3*at4)*T \
	# 		+ 36*(2*an3*an3_T + 2*at3*at3_T)*T \
	# 		+ 36*(an3**2 + at3**2)
		
	# 	J1_pen = 720*2*an5*an5_pen*T**5 + 720*(an4_pen*an5 + an4*an5_pen)*T**4 \
	# 		+ (192*2*an4*an4_pen + 240*(an3_pen*an5 + an3*an5_pen))*T**3 \
	# 		+ 144*(an3_pen*an4 + an3*an4_pen)*T**2 + 36*2*an3*an3_pen*T
		
	# 	J1_vet = 192*2*at4*at4_vet*T**3 + 144*(at3_vet*at4 + at3*at4_vet)*T**2 \
	# 		+ 36*2*at3*at3_vet*T
		
	# 	J1_x = np.array([J1_T, J1_pen, J1_vet])

	# 	return J1_x

	# def get_object_function_time_gradient(self):
	# 	J2_x = np.array([1, 0, 0])
	# 	return J2_x
	
	# def get_object_function_final_lat_pos_gradient(self):
	# 	pen = self.final_lat_pos
	# 	J3_x = np.array([0, 2*pen, 0])
	# 	return J3_x
	
	# def get_object_function_final_long_vel_gradient(self):
	# 	vet = self.final_long_vel
	# 	J4_x = np.array([0, 0, 2*(vet - self.ref_vel)])
	# 	return J4_x
	
	# def get_object_function_gradient(self, x):
	# 	self.generate_path(x)

	# 	J1_x = self.get_object_function_jerk_gradient()
	# 	J2_x = self.get_object_function_time_gradient()
	# 	J3_x = self.get_object_function_final_lat_pos_gradient()
	# 	J4_x = self.get_object_function_final_long_vel_gradient()

	# 	J_x = J1_x + kt*J2_x + kd*J3_x + kv*J4_x

	# 	return J_x
	
	def get_object_function_gradient_numeric(self, x):
		self.generate_path(x)

		T, pen, vet = x

		dT   = 0.0001
		dpen = 0.0001
		dvet = 0.0001

		J1u = planner.get_object_function_value(np.array([T + dT, pen, vet]))
		J1d = planner.get_object_function_value(np.array([T - dT, pen, vet]))
		J2u = planner.get_object_function_value(np.array([T, pen + dpen, vet]))
		J2d = planner.get_object_function_value(np.array([T, pen - dpen, vet]))
		J3u = planner.get_object_function_value(np.array([T, pen, vet + dvet]))
		J3d = planner.get_object_function_value(np.array([T, pen, vet - dvet]))
		
		dJ_numeric = np.array([(J1u-J1d)/dT/2, (J2u-J2d)/dpen/2, (J3u-J3d)/dvet/2])

		return dJ_numeric

	# def get_constraint_function_1(self, t):
	# 	p = self.get_x_y_pos(t)
	# 	ret = self.road.get_projection(p)
	# 	if ret is not None:
	# 		c, dist = ret
	# 		g1 = dist - self.dmax
	# 	else:
	# 		g1 = 0.
	# 	return g1
		
	# def get_constraint_function_3(self, t):
	# 	vt = self.get_x_y_vel(t)
	# 	vt2 = vt.norm**2
	# 	g3 = vt2 - self.vmax**2
	# 	return g3
		
	# def get_constraint_function_4(self, t):
	# 	at = self.get_x_y_acc(t)
	# 	at2 = at.norm**2
	# 	g4 = at2 - self.amax**2
	# 	return g4
		
	# def get_constraint_function_5(self, t):
	# 	kt = self.get_curvature(t)
	# 	kt2 = kt**2
	# 	g5 = kt2 - self.kmax**2
	# 	return g5
	
	def get_constraint_function_value(self, t):
		p = self.get_x_y_pos(t)
		v = self.get_x_y_vel(t)
		a = self.get_x_y_acc(t)
		k = self.get_curvature(t)
		ret = self.road.get_projection(p)
		if ret is not None:
			s, c, dist = ret
			g1 = dist - self.dmax
		else:
			g1 = 0.
		g2 = (v.x**2 + v.y**2) - self.vmax**2
		g3 = (a.x**2 + a.y**2) - self.amax**2
		g4 = k**2 - self.kmax**2
		g = np.array([g1, g2, g3, g4])
		return g
	
	def get_constraint_function_max_value(self, x):
		self.generate_path(x)
		nt = 20

		ts = np.linspace(0, self.final_time, nt)
		cs = []
		for t in ts:
			cs.append(self.get_constraint_function_value(t))
		cs = np.array(cs)
		max_values = np.max(cs, axis=0)
		return max_values


	# def get_constraint_function_max_1(self, nt):
	# 	max_value = -np.inf
	# 	for t in np.linspace(0, self.final_time, nt):
	# 		value = self.get_constraint_function_1(t)
	# 		max_value = max(value, max_value)
	# 	return max_value
		
	# def get_constraint_function_max_3(self, nt):
	# 	max_value = -np.inf
	# 	for t in np.linspace(0, self.final_time, nt):
	# 		value = self.get_constraint_function_3(t)
	# 		max_value = max(value, max_value)
	# 	return max_value
		
	# def get_constraint_function_max_4(self, nt):
	# 	max_value = -np.inf
	# 	for t in np.linspace(0, self.final_time, nt):
	# 		value = self.get_constraint_function_4(t)
	# 		max_value = max(value, max_value)
	# 	return max_value
		
	# def get_constraint_function_max_5(self, nt):
	# 	max_value = -np.inf
	# 	for t in np.linspace(0, self.final_time, nt):
	# 		value = self.get_constraint_function_5(t)
	# 		max_value = max(value, max_value)
	# 	return max_value
	
	# def get_constraint_function_max_value(self, x):
	# 	self.generate_path(x)
	# 	nt = 20

	# 	g_max = np.array([
	# 		self.get_constraint_function_max_1(nt),
	# 		self.get_constraint_function_max_3(nt),
	# 		self.get_constraint_function_max_4(nt),
	# 		self.get_constraint_function_max_5(nt),
	# 	])
	# 	return g_max
		

	def get_constraint_function_jacobian_numeric(self, x):
		self.generate_path(x)

		T = self.final_time
		pen = self.final_lat_pos
		vet = self.final_long_vel

		dT   = 0.0001
		dpen = 0.0001
		dvet = 0.0001

		g1u = self.get_constraint_function_max_value(np.array([T + dT, pen, vet]))
		g1d = self.get_constraint_function_max_value(np.array([T - dT, pen, vet]))
		g2u = self.get_constraint_function_max_value(np.array([T, pen + dpen, vet]))
		g2d = self.get_constraint_function_max_value(np.array([T, pen - dpen, vet]))
		g3u = self.get_constraint_function_max_value(np.array([T, pen, vet + dvet]))
		g3d = self.get_constraint_function_max_value(np.array([T, pen, vet - dvet]))
		dg = np.array([(g1u-g1d)/dT/2, (g2u-g2d)/dpen/2, (g3u-g3d)/dvet/2])

		return dg
	
	def draw(self, axes, x, **kwargs):
		self.generate_path(x)

		T, pen, vet = x
		ts = np.linspace(0, T, 20)

		new_ps = []
		for t in ts:
			pos = self.get_x_y_pos(t)
			new_ps.append(pos.numpy)
		new_ps = np.array(new_ps)

		axes.plot(new_ps[:,0], new_ps[:,1], **kwargs)
		axes.plot(new_ps[0,0], new_ps[0,1], marker='^', **kwargs)
		axes.plot(new_ps[-1,0], new_ps[-1,1], marker='o', **kwargs)
		axes.grid(True)
		plt.pause(0.01)

	def plot_vel(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		vel = self.get_x_y_vels(ts)
		vel_mag = np.sqrt(vel[:,0]**2 + vel[:,1]**2)
		
		axes.plot(ts, vel_mag)
		axes.grid(True)
		axes.set_ylabel('Velocity [m/s]')
		
	def plot_acc(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		acc = self.get_x_y_accs(ts)
		acc_mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2)
		
		axes.plot(ts, acc_mag)
		axes.grid(True)
		axes.set_ylabel('Acceleration [m/s^2]')
		
	def plot_jrk(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		jrk = self.get_x_y_jrks(ts)
		jrk_mag = np.sqrt(jrk[:,0]**2 + jrk[:,1]**2)
		
		axes.plot(ts, jrk_mag)
		axes.grid(True)
		axes.set_ylabel('Jerk [m/s^3]')
		
	def plot_kpa(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		kpa = self.get_curvatures(ts)

		axes.plot(ts, kpa)
		axes.grid(True)
		axes.set_xlabel('Travel distance [m]')
		axes.set_ylabel('Curvature [1/m]')
		
	def lagrangian(self, axes, f, gradf, g, gradg, x0, epsilon=1e-6, max_num_iter=1000, alpha=1e-3, beta=1e-3, **kwargs):
		# no. of design variables
		try:
			N = len(x0)
		except:
			N = 1

		# no. of inequality constraints
		try:
			M = len(g(x0))
		except:
			M = 1

		# Lagrangian multiplier
		mu0 = np.zeros(M)
		
		# set initial values
		k = 0
		xk = x0
		muk = mu0 # make it a column vector

		fk = f(xk)
		gk = g(xk)
		gradfk = gradf(xk)
		jacobgk = gradg(xk)
		Lk = fk + (muk.reshape((-1,1)).transpose() @ gk.reshape((-1,1))).reshape(fk.shape)
		dk = gradfk + (jacobgk.reshape((-1,N)).transpose() @ muk.reshape((-1,1))).reshape(gradfk.shape)

		# create additional return values
		status = CONVERGED
		history = {'k': [k], 'x': [xk], 'd': [dk], 'fval': [fk], 'L': [Lk]}

		# search loop
		while (np.linalg.norm(dk) > epsilon): # stopping criteria 1
			xk_temp = xk
			xk = xk - alpha * dk.reshape(xk.shape)
			muk = vec_max_zero(muk + beta * g(xk_temp))
			
			fk = f(xk)
			gk = g(xk)
			gradfk = gradf(xk)
			jacobgk = gradg(xk)
			Lk = fk + (muk.reshape((-1,1)).transpose() @ gk.reshape((-1,1))).reshape(fk.shape)
			dk = gradfk + (jacobgk.reshape((-1,N)).transpose() @ muk.reshape((-1,1))).reshape(gradfk.shape)

			k += 1

			# store histories
			if k//10*10 == k:
				history['k'].append(k)
				history['x'].append(xk)
				history['d'].append(dk)
				history['fval'].append(fk)
				history['L'].append(Lk)
				self.draw(axes[0], xk, **kwargs)
				axes[1].scatter(k, fk, **kwargs)
				axes[2].scatter(k, gk[0], **kwargs)
				axes[3].scatter(k, gk[1], **kwargs)
				axes[4].scatter(k, gk[2], **kwargs)
				axes[5].scatter(k, gk[3], **kwargs)
				plt.pause(0.01)

			# print the log message
			if k//100*100 == k:
				print(f'Lagrangian: {k}  {np.round(fk,2)}  {np.round(Lk,2)}  x={np.round(xk,2)}')

			if k == max_num_iter: # stopping criteria 2
				status = REACHED_MAX_ITER
				print(f'Lagrangian: reached the maximum number of iteration: {k}')
				break
		
		# solutions to return
		x_opt = xk
		fval_opt = fk

		# convert to numpy array
		history['k'] = np.array(history['k'])
		history['x'] = np.array(history['x'])
		history['d'] = np.array(history['d'])
		history['fval'] = np.array(history['fval'])
		history['L'] = np.array(history['L'])

		# calculate and add the rate of convergence to history
		history['rate_conv'] = (history['fval'][1:] - fval_opt) / (history['fval'][:-1] - fval_opt)
		history['rate_conv'] = np.insert(history['rate_conv'], 0, 1.) # insert 1 at 0-index to match its length with others

		return x_opt, fval_opt, k, status, history

	def opt_search3(self, f, g, xmin, xmax, dx, axes, **kwargs):
		[xss1, xss2, xss3] = np.meshgrid(
			np.arange(xmin[0],xmax[0],dx[0]),
			np.arange(xmin[1],xmax[1],dx[1]),
			np.arange(xmin[2],xmax[2],dx[2]),
		)
		fmin = np.inf
		xopt = None
		ni, nj, nk = xss1.shape

		for i in range(ni):
			for j in range(nj):
				for k in range(nk):
					x = np.array([xss1[i][j][k], xss2[i][j][k], xss3[i][j][k]])
					# plt.clf()
					self.draw(axes, x, **kwargs)
					if np.any(g(x) > 0):
						continue
					fx = f(x)
					if fx < fmin:
						fmin = fx
						xopt = x

		return xopt, fmin, ni*nj*nk
	

if __name__ == '__main__':
	# way points
	wx = [0.0, 20.0, 30, 50.0, 70]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]
	hwidth = 3

	road = Road(wx, wy, hwidth, ds=0.5)

	vref = 10 # m/s
	kt = 1e-2
	kd = 10
	kv = 1e-2

	dmax = 2
	vmax = 15 # m/s
	amax = 9.81 # m/s^2
	kmax = 0.1
	Ds = 30.

	s0 = 0.
	sf = s0 + Ds

	T0 = 3
	pen0 = 0
	vet0 = 10
	x0 = np.array([T0, pen0, vet0])

	planner = PathPlanner(road, vref, kt, kd, kv, dmax, vmax, amax, kmax)

	fig = plt.figure(1, figsize=(6,8))
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
		
	axes = []
	axes.append(fig.add_subplot(521))
	axes.append(fig.add_subplot(522))
	axes.append(fig.add_subplot(523))
	axes.append(fig.add_subplot(524))
	axes.append(fig.add_subplot(525))
	axes.append(fig.add_subplot(526))
	for ax in axes:
		ax.grid(True)
	axes[0].set_aspect(1)
	axes[0].set_xlim(-5, 75)
	axes[0].set_ylim(-15, 15)
	axes[1].set_ylabel('f')
	axes[2].set_ylabel('g1-dmax')
	axes[3].set_ylabel('g2-vmax')
	axes[4].set_ylabel('g3-amax')
	axes[5].set_ylabel('g4-kmax')
	planner.road.draw(axes[0], ds=1, dmax=dmax)

	# ax1 = fig.add_subplot(521)
	ax2 = fig.add_subplot(527)
	ax3 = fig.add_subplot(528)
	ax4 = fig.add_subplot(529)
	ax5 = fig.add_subplot(5,2,10)
	# ax1.grid(True)
	ax2.grid(True)
	ax3.grid(True)
	ax4.grid(True)
	ax5.grid(True)


	current_state = VehicleState(pos=Vector2D(0,0), vel=Vector2D(20/3.6,0), acc=Vector2D(0,0))

	t = 0
	Dt = 0.8

	for _ in range(20):
		planner.set_current_state(s0, sf, current_state)

		# T = 12
		# pen = 5
		# vet = 27*kph
		# x = np.array([T, pen, vet])

		# J1 = planner.get_object_function_jerk()
		# J2 = planner.get_object_function_time()
		# J3 = planner.get_object_function_final_lat_pos()
		# J4 = planner.get_object_function_final_long_vel()
		# J = planner.get_object_function_value(x)

		# print(f'{J=:6.4f} {J1=:6.4f} {J2=:6.4f} {J3=:6.4f} {J4=:6.4f}')

		# # analytic differentiation
		# dJ = planner.get_object_function_gradient(x)
		# dJ_str = ' '.join([str(s).rjust(5) for s in np.round(dJ, 3)])
		# print(f'{dJ_str}')

		# dJ_numeric = planner.get_object_function_gradient_numeric(x)
		# dJ_numeric_str = ' '.join([str(s).rjust(5) for s in np.round(dJ_numeric, 3)])
		# print(f'{dJ_numeric_str}')

		# g = planner.get_constraint_function_max_value(x)
		# print(g)

		# dg = planner.get_constraint_function_jacobian_numeric(x)
		# print(dg)


		f = planner.get_object_function_value
		gradf = planner.get_object_function_gradient_numeric
		g = planner.get_constraint_function_max_value
		gradg = planner.get_constraint_function_jacobian_numeric

		start_time = time.time()
		x_opt, fval_opt, k, status, history = planner.lagrangian(axes, f, gradf, g, gradg, x0, epsilon=1e-1, max_num_iter=50, alpha=1e-4, beta=1e-3, color='blue')
		# x_opt, fval_opt, k = planner.opt_search3(f, g, xmin=[1.5, -2, 6], xmax=[4.5, 2, 12], dx=[1, 0.5, 2], axes=axes, color='blue')
		status=0
		history = {'k': [0], 'x': [(0,0,0)], 'fval': [0], 'L': [0]}
		end_time = time.time()

		planner.generate_path(x_opt)
		t += Dt
		
		new_xy_pos = planner.get_x_y_pos(t=Dt)
		new_xy_vel = planner.get_x_y_vel(t=Dt)
		new_xy_acc = planner.get_x_y_acc(t=Dt)

		s0, c0, _ = planner.road.get_projection(new_xy_pos)
		new_pos = planner.road.convert_xy2tn(s0, new_xy_pos - c0)
		new_vel = planner.road.convert_xy2tn(s0, new_xy_vel)
		new_acc = planner.road.convert_xy2tn(s0, new_xy_acc)

		current_state = VehicleState(new_pos, new_vel, new_acc)
		print(t, current_state)

		sf = s0 + Ds

		x0 = x_opt


		planner.draw(axes[0], x_opt, color='red')

		print(f"Lagrangian: {status=}, x_opt={np.round(x_opt,2)}, fval_opt={np.round(fval_opt,2)}, num_iter={k}")
		print(f'Optimization time = {end_time - start_time:5.2f} sec')

		ks = history['k']
		xs = history['x']
		fvals = history['fval']
		Ls = history['L']


		# draw path
		# ax1.set_aspect(1)
		# ax1.set_xlim(-5, 75)
		# ax1.set_ylim(-15, 15)

		# road.draw(ax1, ds=1, dmax=dmax)
		# planner.draw(ax1, x_opt)
		planner.plot_vel(ax2, x_opt)
		planner.plot_acc(ax3, x_opt)
		planner.plot_jrk(ax4, x_opt)
		planner.plot_kpa(ax5, x_opt)
		
		# ax6 = fig.add_subplot(916)
		# ax7 = fig.add_subplot(917)
		# ax8 = fig.add_subplot(918)
		# ax9 = fig.add_subplot(919)
		# ax6.grid(True)
		# ax7.grid(True)
		# ax8.grid(True)
		# ax9.grid(True)

		# kss = [ks[0], ks[-1]]
		# xss = [xs[0], xs[-1]]
		# for i, k in enumerate(kss):
		# 	x = xss[i]
		# 	planner.generate_path(x)
		# 	T, pen, vet = x
		# 	ts = np.linspace(0,T,20)
		# 	color = 'blue' if i == 0 else 'red'
		# 	c1 = []
		# 	c3 = []
		# 	c4 = []
		# 	c5 = []
		# 	for t in ts:
		# 		c1.append(planner.get_constraint_function_1(t))
		# 		c3.append(planner.get_constraint_function_3(t))
		# 		c4.append(planner.get_constraint_function_4(t))
		# 		c5.append(planner.get_constraint_function_5(t))
		# 	ax6.scatter(ts, c1, color=color)
		# 	ax7.scatter(ts, c3, color=color)
		# 	ax8.scatter(ts, c4, color=color)
		# 	ax9.scatter(ts, c5, color=color)

		plt.pause(0.01)
	plt.show()