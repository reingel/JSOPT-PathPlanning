import sys

from quintic_polynomials import QuinticPolynomial, QuinticPolynomial2D
sys.path.append('./JSOPT')

from typing import final
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _parse_possible_contraction
from road import Road
from vector import *
from vehicle_state import VehicleState
from vector import Vector2D
from JSOPT.lagrangian import lagrangian
import time
from JSOPT.constant import *
from JSOPT.linalg import Vector, normalized
from JSOPT.line_search import bisection

EPS = 1e-3

def vec_max_zero(x: np.ndarray):
	return np.array([max(xi, 0) for xi in x])


class PathPlanner:
	def __init__(self, road: Road, ref_vel, kj, kt, kd, kv, dmax, vmax, amax, kmax):
		self.road = road

		self.ref_vel = ref_vel
		self.kj = kj
		self.kt = kt
		self.kd = kd
		self.kv = kv

		self.dmax = dmax
		self.vmax = vmax
		self.amax = amax
		self.kmax = kmax
	
	def set_travel_range(self, s0, sT):
		self.s0 = s0 # current 1d position along road center line
		self.sT = sT # final   1d position along road center line

	def set_current_state(self, current_state: VehicleState): # current_state : vehicle state in t-n coordinates at t=0
		pos0 = self.road.get_center_point(self.s0)
		converted_state = VehicleState(
			pos = self.road.convert_tn2xy(self.s0, current_state.pos) + pos0,
			vel = self.road.convert_tn2xy(self.s0, current_state.vel),
			acc = self.road.convert_tn2xy(self.s0, current_state.acc),
		)
		self.current_state = converted_state
	
	def set_final_state(self, final_state: Vector): # final_state (design variables) : vehicle state in t-n coordinates at t=T
		self.final_time = final_state[0]
		self.final_lat_pos = final_state[1]
		self.final_long_vel = final_state[2]

		final_state = VehicleState(
			pos=Vector2D(0, self.final_lat_pos),
			vel=Vector2D(self.final_long_vel, 0),
			acc=Vector2D(0, 0),
		)

		posT = self.road.get_center_point(self.sT)
		converted_state = VehicleState(
			pos=self.road.convert_tn2xy(self.sT, final_state.pos) + posT,
			vel=self.road.convert_tn2xy(self.sT, final_state.vel),
			acc=self.road.convert_tn2xy(self.sT, final_state.acc),
		)
		self.final_state = converted_state

		self._generate_path()

	def _generate_path(self):
		self.quintic_poly2d = QuinticPolynomial2D(
			T=self.final_time,
			pos0=self.current_state.pos,
			vel0=self.current_state.vel,
			acc0=self.current_state.acc,
			posT=self.final_state.pos,
			velT=self.final_state.vel,
			accT=self.final_state.acc,
		)

	def get_xy_pos(self, t):
		return self.quintic_poly2d.pos(t)
	
	def get_xy_vel(self, t):
		return self.quintic_poly2d.vel(t)
	
	def get_xy_acc(self, t):
		return self.quintic_poly2d.acc(t)
	
	def get_xy_jrk(self, t):
		return self.quintic_poly2d.jrk(t)
	
	def get_curvature(self, t):
		return self.quintic_poly2d.kpa(t)
	
	def get_object_function_jerk(self):
		T = self.final_time
		ax0, ax1, ax2, ax3, ax4, ax5 = self.quintic_poly2d.x_poly.pos_coef
		ay0, ay1, ay2, ay3, ay4, ay5 = self.quintic_poly2d.y_poly.pos_coef
		J1 = np.dot(
			Vector([
				60**2*(ax5**2 + ay5**2)/5,
				2*24*60*(ax4*ax5 + ay4*ay5)/4,
				(24**2*(ax4**2 + ay4**2) + 2*6*60*(ax3*ax5 + ay3*ay5))/3,
				2*6*24*(ax3*ax4 + ay3*ay4)/2,
				36*(ax3**2 + ay3**2),
			]),
			T**column_vector(np.arange(5))
		)
		return J1
	
	def get_object_function_time(self):
		J2 = self.final_time
		return J2
	
	def get_object_function_final_n_pos(self):
		J3 = self.final_lat_pos**2
		return J3
	
	def get_object_function_final_t_vel(self):
		J4 = (self.final_long_vel - self.ref_vel)**2
		return J4
	
	def get_object_function_value(self, final_state: Vector):
		self.set_final_state(final_state)
		J1 = self.get_object_function_jerk()
		J2 = self.get_object_function_time()
		J3 = self.get_object_function_final_n_pos()
		J4 = self.get_object_function_final_t_vel()
		J = self.kj*J1 + self.kt*J2 + self.kd*J3 + self.kv*J4
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
	
	def get_object_function_gradient_numeric(self, final_state: Vector):
		self.set_final_state(final_state)
		T = self.final_time
		posnT = self.final_lat_pos
		veltT = self.final_long_vel

		dT = 0.0001
		dp = 0.0001
		dv = 0.0001

		J1u = planner.get_object_function_value(np.array([T + dT, posnT, veltT]))
		J1d = planner.get_object_function_value(np.array([T - dT, posnT, veltT]))
		J2u = planner.get_object_function_value(np.array([T, posnT + dp, veltT]))
		J2d = planner.get_object_function_value(np.array([T, posnT - dp, veltT]))
		J3u = planner.get_object_function_value(np.array([T, posnT, veltT + dv]))
		J3d = planner.get_object_function_value(np.array([T, posnT, veltT - dv]))
		
		dJ_numeric = np.array([(J1u-J1d)/(2*dT), (J2u-J2d)/(2*dp), (J3u-J3d)/(2*dv)])

		return dJ_numeric

	# def get_constraint_function_1(self, t):
	# 	p = self.get_xy_pos(t)
	# 	ret = self.road.get_projection(p)
	# 	if ret is not None:
	# 		c, dist = ret
	# 		g1 = dist - self.dmax
	# 	else:
	# 		g1 = 0.
	# 	return g1
		
	# def get_constraint_function_3(self, t):
	# 	vt = self.get_xy_vel(t)
	# 	vt2 = vt.norm**2
	# 	g3 = vt2 - self.vmax**2
	# 	return g3
		
	# def get_constraint_function_4(self, t):
	# 	at = self.get_xy_acc(t)
	# 	at2 = at.norm**2
	# 	g4 = at2 - self.amax**2
	# 	return g4
		
	# def get_constraint_function_5(self, t):
	# 	kt = self.get_curvature(t)
	# 	kt2 = kt**2
	# 	g5 = kt2 - self.kmax**2
	# 	return g5
	
	def get_constraint_function_value(self, t):
		pos = self.get_xy_pos(t)
		vel = self.get_xy_vel(t)
		acc = self.get_xy_acc(t)
		kpa = self.get_curvature(t)
		dist = self.road.get_projection_distance(pos)
		g1 = dist - self.dmax
		g2 = (vel.x**2 + vel.y**2) - self.vmax**2
		g3 = (acc.x**2 + acc.y**2) - self.amax**2
		g4 = kpa**2 - self.kmax**2
		g = np.vstack([g1, g2, g3, g4]).transpose()
		return g
	
	def get_constraint_function_max_value(self, final_state: Vector):
		t1 = time.time()
		self.set_final_state(final_state)
		nt = 20

		t2 = time.time()
		ts = np.linspace(0, self.final_time, nt)
		g = self.get_constraint_function_value(ts)
		t3 = time.time()
		max_values = np.max(g, axis=0)
		t4 = time.time()
		# print(f'get_constraint_function_max_value: 12={t2-t1:.2f} 23={t3-t2:.2f} 34={t4-t3:.2f}')
		return max_values


	def get_constraint_function_jacobian_numeric(self, final_state: Vector):
		t1 = time.time()
		self.set_final_state(final_state)

		t2 = time.time()
		T = self.final_time
		pen = self.final_lat_pos
		vet = self.final_long_vel

		dT   = 0.0001
		dpen = 0.0001
		dvet = 0.0001

		t3 = time.time()
		g1u = self.get_constraint_function_max_value(np.array([T + dT, pen, vet]))
		g1d = self.get_constraint_function_max_value(np.array([T - dT, pen, vet]))
		g2u = self.get_constraint_function_max_value(np.array([T, pen + dpen, vet]))
		g2d = self.get_constraint_function_max_value(np.array([T, pen - dpen, vet]))
		g3u = self.get_constraint_function_max_value(np.array([T, pen, vet + dvet]))
		g3d = self.get_constraint_function_max_value(np.array([T, pen, vet - dvet]))
		dg = np.array([(g1u-g1d)/dT/2, (g2u-g2d)/dpen/2, (g3u-g3d)/dvet/2])
		t4 = time.time()

		# print(f'get_constraint_function_jacobian_numeric: 12={t2-t1:.2f} 23={t3-t2:.2f} 34={t4-t3:.2f}')

		return dg
	
	def draw(self, axes, final_state, **kwargs):
		self.set_final_state(final_state)

		T, pen, vet = final_state
		ts = np.linspace(0, T, 20)

		new_ps = []
		for t in ts:
			pos = self.get_xy_pos(t)[0]
			new_ps.append(pos.numpy)
		new_ps = np.array(new_ps)

		axes.plot(new_ps[:,0], new_ps[:,1], color='blue')
		axes.plot(new_ps[-1,0], new_ps[-1,1], marker='^', color='green')
		axes.plot(new_ps[0,0], new_ps[0,1], marker='o', color='red')
		axes.grid(True)
		plt.pause(0.01)

	def plot_vel(self, axes, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)

		vel = self.get_xy_vel(ts)
		vel_mag = np.sqrt(vel.numpy[:,0]**2 + vel.numpy[:,1]**2)
		
		axes.plot(ts, vel_mag)
		axes.grid(True)
		axes.set_ylabel('Velocity [m/s]')
		
	def plot_acc(self, axes, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		acc = self.get_xy_acc(ts)
		acc_mag = np.sqrt(acc.numpy[:,0]**2 + acc.numpy[:,1]**2)
		axes.plot(ts, acc_mag)
		axes.grid(True)
		axes.set_ylabel('Acceleration [m/s^2]')
		
	def plot_jrk(self, axes, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		jrk = self.get_xy_jrk(ts)
		jrk_mag = np.sqrt(jrk.numpy[:,0]**2 + jrk.numpy[:,1]**2)
		axes.plot(ts, jrk_mag)
		axes.grid(True)
		axes.set_ylabel('Jerk [m/s^3]')
		
	def plot_kpa(self, axes, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		kpa = self.get_curvature(ts)
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
		# history = {'k': [k], 'x': [xk], 'd': [dk], 'fval': [fk], 'L': [Lk]}

		# search loop
		while (np.linalg.norm(dk) > epsilon): # stopping criteria 1
			t1 = time.time()
			xk_temp = xk
			xk = xk - alpha * dk.reshape(xk.shape)
			muk = vec_max_zero(muk + beta * g(xk_temp))
			
			t2 = time.time()
			fk = f(xk)
			t3 = time.time()
			gk = g(xk)
			t4 = time.time()
			gradfk = gradf(xk)
			t5 = time.time()
			jacobgk = gradg(xk)
			t6 = time.time()
			Lk = fk + (muk.reshape((-1,1)).transpose() @ gk.reshape((-1,1))).reshape(fk.shape)
			dk = gradfk + (jacobgk.reshape((-1,N)).transpose() @ muk.reshape((-1,1))).reshape(gradfk.shape)

			k += 1

			# store histories
			if k//10*10 == k:
				pass
				# history['k'].append(k)
				# history['x'].append(xk)
				# history['d'].append(dk)
				# history['fval'].append(fk)
				# history['L'].append(Lk)
				# self.draw(axes[0], xk, **kwargs)
				# axes[1].scatter(k, fk, **kwargs)
				# axes[2].scatter(k, gk[0], **kwargs)
				# axes[3].scatter(k, gk[1], **kwargs)
				# axes[4].scatter(k, gk[2], **kwargs)
				# axes[5].scatter(k, gk[3], **kwargs)
				# plt.pause(0.001)

			# print the log message
			# if k//100*100 == k:
			# 	print(f'Lagrangian: {k}  {np.round(fk,2)}  {np.round(gk,2)}  x={np.round(xk,2)}')

			if k == max_num_iter: # stopping criteria 2
				status = REACHED_MAX_ITER
				# print(f'Lagrangian: reached the maximum number of iteration: {k}')
				break
			t7 = time.time()
			# print(f'Time: 12={t2-t1:.2f} 23={t3-t2:.2f} 34={t4-t3:.2f} 45={t5-t4:.2f} 56={t6-t5:.2f} 67={t7-t6:.2f}')
		
		# solutions to return
		x_opt = xk
		fval_opt = fk
		g_opt = gk

		# convert to numpy array
		# history['k'] = np.array(history['k'])
		# history['x'] = np.array(history['x'])
		# history['d'] = np.array(history['d'])
		# history['fval'] = np.array(history['fval'])
		# history['L'] = np.array(history['L'])

		# calculate and add the rate of convergence to history
		# history['rate_conv'] = (history['fval'][1:] - fval_opt) / (history['fval'][:-1] - fval_opt)
		# history['rate_conv'] = np.insert(history['rate_conv'], 0, 1.) # insert 1 at 0-index to match its length with others

		history = None

		return x_opt, fval_opt, k, gk, status, history

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
	wx = [0.0, 20.0, 30, 50.0, 150]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]
	hwidth = 6

	road = Road(wx, wy, hwidth, ds=2)

	vref = 10 # m/s
	kj = 1
	kt = 0.004
	kd = 20
	kv = 0.004

	dmax = 2
	vmax = 15 # m/s
	amax = 9.81 # m/s^2
	kmax = 3
	Ds = 30.

	s0 = 0.
	sT = s0 + Ds

	T0 = 3
	pen0 = 0
	vet0 = 10
	x0 = np.array([T0, pen0, vet0])

	planner = PathPlanner(road, vref, kj, kt, kd, kv, dmax, vmax, amax, kmax)

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

	for idx in range(20):
		planner.set_travel_range(s0, sT)
		planner.set_current_state(current_state)

		f = planner.get_object_function_value
		g = planner.get_constraint_function_max_value
		gradf = planner.get_object_function_gradient_numeric
		gradg = planner.get_constraint_function_jacobian_numeric

		start_time = time.time()
		x_opt, fval_opt, k, g_opt, status, history = planner.lagrangian(axes, f, gradf, g, gradg, x0, epsilon=1e-1, max_num_iter=50, alpha=1e-4, beta=1e-3, color='blue')
		# x_opt, fval_opt, k = planner.opt_search3(f, g, xmin=[1.5, -2, 6], xmax=[4.5, 2, 12], dx=[1, 0.5, 2], axes=axes, color='blue')
		# status = 0
		# history = {'k': [0], 'x': [(0,0,0)], 'fval': [0], 'L': [0]}
		end_time = time.time()

		planner.set_final_state(x_opt)
		t += Dt
		
		new_xy_pos = planner.get_xy_pos(t=Dt)[0]
		new_xy_vel = planner.get_xy_vel(t=Dt)[0]
		new_xy_acc = planner.get_xy_acc(t=Dt)[0]

		s0, c0, _ = planner.road.get_projection(new_xy_pos)
		new_pos = planner.road.convert_xy2tn(s0, new_xy_pos - c0)
		new_vel = planner.road.convert_xy2tn(s0, new_xy_vel)
		new_acc = planner.road.convert_xy2tn(s0, new_xy_acc)

		current_state = VehicleState(new_pos, new_vel, new_acc)
		# print(t, current_state)

		sT = s0 + Ds
		x0 = x_opt

		planner.draw(axes[0], x_opt)

		print(f'Lagrangian: {idx=}, x_opt={np.round(x_opt,2)}, fval_opt={np.round(fval_opt,2)}, g={np.round(g_opt,2)}, num_iter={k}, time={end_time - start_time:5.2f} sec')

		# ks = history['k']
		# xs = history['x']
		# fvals = history['fval']
		# Ls = history['L']


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