from typing import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction
from road import Road
from vehicle_state import VehicleState
from vector import Vector2D
from JSOPT.lagrangian import lagrangian

def max0(x):
	return max(x, 0)


class PathPlanner:
	def __init__(self, road: Road, ref_vel, kt, kd, kv, dmax, vmax, amax, kmax, alpha=0.01, beta=0.01):
		self.road = road

		self.ref_vel = ref_vel
		self.kt = kt
		self.kd = kd
		self.kv = kv

		self.dmax = dmax
		self.vmax = vmax
		self.amax = amax
		self.kmax = kmax

		self.alpha = alpha
		self.beta = beta
		self.mu = np.zeros((5,1))
	
	def set_current_state(self, current_state: VehicleState):
		self.current_state = current_state

	def generate_path(self, x: np.ndarray):
		final_time, final_lat_pos, final_long_vel = x
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

		pns = self.current_state.pos.y
		vns = self.current_state.vel.y
		ans = self.current_state.acc.y
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

		pts = self.current_state.pos.x
		vts = self.current_state.vel.x
		ats = self.current_state.acc.x
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
		pn = np.dot(self.lat_poly_coef, t**np.arange(6))
		pt = np.dot(self.long_poly_coef, t**np.arange(5))
		return Vector2D(pn, pt)
	
	def get_lat_long_vel(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		vn = np.dot(np.array([an1, 2*an2, 3*an3, 4*an4, 5*an5]), t**np.arange(5))
		vt = np.dot(np.array([at1, 2*at2, 3*at3, 4*at4]), t**np.arange(4))
		return Vector2D(vn, vt)
	
	def get_lat_long_acc(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		an = np.dot(np.array([2*an2, 6*an3, 12*an4, 20*an5]), t**np.arange(4))
		at = np.dot(np.array([2*at2, 6*at3, 12*at4]), t**np.arange(3))
		return Vector2D(an, at)
	
	def get_lat_long_jrk(self, t):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		jn = np.dot(np.array([6*an3, 24*an4, 60*an5]), t**np.arange(3))
		jt = np.dot(np.array([6*at3, 24*at4]), t**np.arange(2))
		return Vector2D(jn, jt)
	
	def get_curvature(self, t):
		vel = self.get_lat_long_vel(t)
		acc = self.get_lat_long_acc(t)
		kappa = (vel.x * acc.y - vel.y * acc.x) / (vel.y**2 + vel.x**2)**(3/2)
		return kappa
	
	def get_lat_long_poses(self, ts):
		pn = np.dot(self.lat_poly_coef, ts**np.arange(6).reshape((-1,1)))
		pt = np.dot(self.long_poly_coef, ts**np.arange(5).reshape((-1,1)))
		return np.column_stack((pn, pt))
	
	def get_lat_long_vels(self, ts):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		vn = np.dot(np.array([an1, 2*an2, 3*an3, 4*an4, 5*an5]), ts**np.arange(5).reshape((-1,1)))
		vt = np.dot(np.array([at1, 2*at2, 3*at3, 4*at4]), ts**np.arange(4).reshape((-1,1)))
		return np.column_stack((vn, vt))
	
	def get_lat_long_accs(self, ts):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		an = np.dot(np.array([2*an2, 6*an3, 12*an4, 20*an5]), ts**np.arange(4).reshape((-1,1)))
		at = np.dot(np.array([2*at2, 6*at3, 12*at4]), ts**np.arange(3).reshape((-1,1)))
		return np.column_stack((an, at))
	
	def get_lat_long_jrks(self, ts):
		an0, an1, an2, an3, an4, an5 = self.lat_poly_coef
		at0, at1, at2, at3, at4 = self.long_poly_coef
		jn = np.dot(np.array([6*an3, 24*an4, 60*an5]), ts**np.arange(3).reshape((-1,1)))
		jt = np.dot(np.array([6*at3, 24*at4]), ts**np.arange(2).reshape((-1,1)))
		return np.column_stack((jn, jt))
	
	def get_curvatures(self, ts):
		vel = self.get_lat_long_vels(ts)
		acc = self.get_lat_long_accs(ts)
		vn, vt = vel[:,0], vel[:,1]
		an, at = acc[:,0], acc[:,1]
		kappa = (vt * an - vn * at) / (vn**2 + vt**2)**(3/2)
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
	
	def get_object_function_final_long_vel(self):
		vet = self.final_long_vel
		J4 = (vet - self.ref_vel)**2
		return J4
	
	def get_object_function(self, x):
		self.generate_path(x)

		J1 = self.get_object_function_jerk()
		J2 = self.get_object_function_time()
		J3 = self.get_object_function_final_lat_pos()
		J4 = self.get_object_function_final_long_vel()

		J = J1 + kt*J2 + kd*J3 + kv*J4

		return J
	
	def get_object_function_jerk_gradient(self):
		# current and final states
		T = self.final_time

		psn = self.current_state.pos.y

		vsn = self.current_state.vel.y
		vst = self.current_state.vel.x

		asn = self.current_state.acc.y
		ast = self.current_state.acc.x

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
	
	def get_object_function_final_long_vel_gradient(self):
		vet = self.final_long_vel
		J4_x = np.array([0, 0, 2*(vet - self.ref_vel)])
		return J4_x
	
	def get_object_function_gradient(self, x):
		self.generate_path(x)

		J1_x = self.get_object_function_jerk_gradient()
		J2_x = self.get_object_function_time_gradient()
		J3_x = self.get_object_function_final_lat_pos_gradient()
		J4_x = self.get_object_function_final_long_vel_gradient()

		J_x = J1_x + kt*J2_x + kd*J3_x + kv*J4_x

		return J_x
	
	def get_object_function_gradient_numeric(self, x):
		self.generate_path(x)

		T, pen, vet = x

		dT   = 0.0001
		dpen = 0.0001
		dvet = 0.0001

		J1u = planner.get_object_function(np.array([T + dT, pen, vet]))
		J1d = planner.get_object_function(np.array([T - dT, pen, vet]))
		J2u = planner.get_object_function(np.array([T, pen + dpen, vet]))
		J2d = planner.get_object_function(np.array([T, pen - dpen, vet]))
		J3u = planner.get_object_function(np.array([T, pen, vet + dvet]))
		J3d = planner.get_object_function(np.array([T, pen, vet - dvet]))
		
		dJ_numeric = np.array([(J1u-J1d)/dT/2, (J2u-J2d)/dpen/2, (J3u-J3d)/dvet/2])

		return dJ_numeric

	def get_constraint_function_1(self, t):
		pt = self.get_lat_long_pos(t)
		ret = self.road.get_projection(pt)
		if ret is not None:
			c, dist = ret
			g1 = dist - self.dmax
		else:
			g1 = 999.
		return g1
		
	def get_constraint_function_3(self, t):
		vt = self.get_lat_long_vel(t)
		vt2 = vt.norm**2
		g3 = vt2 - self.vmax**2
		return g3
		
	def get_constraint_function_4(self, t):
		at = self.get_lat_long_acc(t)
		at2 = at.norm**2
		g4 = at2 - self.amax**2
		return g4
		
	def get_constraint_function_5(self, t):
		kt = self.get_curvature(t)
		kt2 = kt**2
		g5 = kt2 - self.kmax**2
		return g5
	
	def get_constraint_function_max_1(self, x):
		self.generate_path(x)

		max_value = -np.inf
		for t in np.linspace(0, self.final_time, 10):
			value = self.get_constraint_function_1(t)
			max_value = max(value, max_value)
		return max_value
		
	def get_constraint_function_max_3(self, x):
		self.generate_path(x)

		max_value = -np.inf
		for t in np.linspace(0, self.final_time, 10):
			value = self.get_constraint_function_3(t)
			max_value = max(value, max_value)
		return max_value
		
	def get_constraint_function_max_4(self, x):
		self.generate_path(x)

		max_value = -np.inf
		for t in np.linspace(0, self.final_time, 10):
			value = self.get_constraint_function_4(t)
			max_value = max(value, max_value)
		return max_value
		
	def get_constraint_function_max_5(self, x):
		self.generate_path(x)

		max_value = -np.inf
		for t in np.linspace(0, self.final_time, 10):
			value = self.get_constraint_function_5(t)
			max_value = max(value, max_value)
		return max_value
	
	def get_constraint_function_max(self, x):
		g_max = np.array([
			self.get_constraint_function_max_1(x),
			self.get_constraint_function_max_3(x),
			self.get_constraint_function_max_4(x),
			self.get_constraint_function_max_5(x),
		])
		return g_max
		

	def get_constraint_function_jacobian(self, x):
		self.generate_path(x)

		T = self.final_time
		pen = self.final_lat_pos
		vet = self.final_long_vel

		dT   = 0.0001
		dpen = 0.0001
		dvet = 0.0001

		g1u = self.get_constraint_function_max(np.array([T + dT, pen, vet]))
		g1d = self.get_constraint_function_max(np.array([T - dT, pen, vet]))
		g2u = self.get_constraint_function_max(np.array([T, pen + dpen, vet]))
		g2d = self.get_constraint_function_max(np.array([T, pen - dpen, vet]))
		g3u = self.get_constraint_function_max(np.array([T, pen, vet + dvet]))
		g3d = self.get_constraint_function_max(np.array([T, pen, vet - dvet]))
		dg = np.array([(g1u-g1d)/dT/2, (g2u-g2d)/dpen/2, (g3u-g3d)/dvet/2]).transpose()

		return dg

	def lagrangian(self):
		mu = np.zeros((5,1))
		x = np.array([4, 0, self.ref_vel])

		for i in range(20):
			df = self.get_object_function_gradient_numeric()
			dg = self.get_constraint_function_jacobian()

			x -= self.alpha*(df + (dg.transpose() @ mu).squeeze(1))
			mu = np.expand_dims(np.array(list(map(max0, mu.squeeze(1) + self.beta*g))), 1)
		
		return x
	
	def draw(self, axes, x):
		self.generate_path(x)

		T, pen, vet = x
		road = self.road

		ts = np.linspace(0, T, 30)

		new_ps = []
		for t in ts:
			pos = self.get_lat_long_pos(t)
			new_ps.append(pos.numpy)
		new_ps = np.array(new_ps)

		axes.plot(new_ps[:,0], new_ps[:,1], color='red')
		axes.grid(True)

	def plot_vel(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		pos = self.get_lat_long_poses(ts)
		vel = self.get_lat_long_vels(ts)

		s = pos[:,1]
		vel_mag = np.sqrt(vel[:,0]**2 + vel[:,1]**2)
		
		axes.plot(s, vel_mag)
		axes.grid(True)
		axes.set_ylabel('Velocity [m/s]')
		
	def plot_acc(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		pos = self.get_lat_long_poses(ts)
		vel = self.get_lat_long_vels(ts)
		acc = self.get_lat_long_accs(ts)

		s = pos[:,1]
		acc_mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2)
		
		axes.plot(s, acc_mag)
		axes.grid(True)
		axes.set_ylabel('Acceleration [m/s^2]')
		
	def plot_jrk(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		pos = self.get_lat_long_poses(ts)
		jrk = self.get_lat_long_jrks(ts)

		s = pos[:,1]
		jrk_mag = np.sqrt(jrk[:,0]**2 + jrk[:,1]**2)
		
		axes.plot(s, jrk_mag)
		axes.grid(True)
		axes.set_ylabel('Jerk [m/s^3]')
		
	def plot_kpa(self, axes, x):
		self.generate_path(x)
		T = x[0]
		ts = np.linspace(0, T, 20)

		pos = self.get_lat_long_poses(ts)
		kpa = self.get_curvatures(ts)

		s = pos[:,1]
		
		axes.plot(s, kpa)
		axes.grid(True)
		axes.set_xlabel('Travel distance [m]')
		axes.set_ylabel('Curvature [1/m]')
		

if __name__ == '__main__':
	# way points
	wx = [0.0, 10.0, 20.5, 35.0, 70.5]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]
	hwidth = 3

	road = Road(wx, wy, hwidth)

	vref = 10 # m/s
	kt = 1e-2
	kd = 1e-2
	kv = 1e-2

	dmax = 1
	vmax = 20 # m/s
	amax = 9.81 # m/s^2
	kmax = 10

	planner = PathPlanner(road, vref, kt, kd, kv, dmax, vmax, amax, kmax)

	current_state = VehicleState(pos=Vector2D(0,0), vel=Vector2D(0,10), acc=Vector2D(0,0))
	planner.set_current_state(current_state)

	# T = 12
	# pen = 5
	# vet = 27*kph
	# x = np.array([T, pen, vet])

	# J1 = planner.get_object_function_jerk()
	# J2 = planner.get_object_function_time()
	# J3 = planner.get_object_function_final_lat_pos()
	# J4 = planner.get_object_function_final_long_vel()
	# J = planner.get_object_function(x)

	# print(f'{J=:6.4f} {J1=:6.4f} {J2=:6.4f} {J3=:6.4f} {J4=:6.4f}')

	# # analytic differentiation
	# dJ = planner.get_object_function_gradient(x)
	# dJ_str = ' '.join([str(s).rjust(5) for s in np.round(dJ, 3)])
	# print(f'{dJ_str}')

	# dJ_numeric = planner.get_object_function_gradient_numeric(x)
	# dJ_numeric_str = ' '.join([str(s).rjust(5) for s in np.round(dJ_numeric, 3)])
	# print(f'{dJ_numeric_str}')

	# g = planner.get_constraint_function_max(x)
	# print(g)

	# dg = planner.get_constraint_function_jacobian(x)
	# print(dg)


	f = planner.get_object_function
	gradf = planner.get_object_function_gradient
	g = planner.get_constraint_function_max
	gradg = planner.get_constraint_function_jacobian

	T0 = 12
	pen0 = 5
	vet0 = 10
	x0 = np.array([T0, pen0, vet0])

	x_opt, L_opt, status, history = lagrangian(f, gradf, g, gradg, x0, epsilon=1e-1, max_num_iter=1000, alpha=1e-3, beta=1e-2)
	print(f"Lagrangian: {status=}, x_opt={np.round(x_opt,2)}, L_opt={np.round(L_opt,2)}, num_iter={len(history['x'])-1}")

	# draw path
	_, ax = plt.subplots(5,1)
	ax[0].set_aspect(1)
	ax[0].set_xlim(-5, 75)
	ax[0].set_ylim(-15, 15)

	road.draw(ax[0], ds=1, dmax=3)
	planner.draw(ax[0], x_opt)
	planner.plot_vel(ax[1], x_opt)
	planner.plot_acc(ax[2], x_opt)
	planner.plot_jrk(ax[3], x_opt)
	planner.plot_kpa(ax[4], x_opt)
	
	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])

	plt.show()