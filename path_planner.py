import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from cubic_spline_planner import Spline2D
from quintic_polynomials import QuinticPolynomial2D
from road import Road
from vector import *
import time
import os
from vehicle_dynamics import do_simulation, State


EPS = 1e-3
COLOR_LINE = '#8080FF'
COLOR_START = '#FF0000'
COLOR_END = '#008000'


def vec_max_zero(x: np.ndarray):
	return np.array([max(xi, 0) for xi in x])


class VehicleState:
	def __init__(self, pos=Vector2D(0,0), vel=Vector2D(0,1), acc=Vector2D(0,0)):
		self._pos = pos
		self._vel = vel
		self._acc = acc
	
	@property
	def pos(self):
		return self._pos
	
	@property
	def vel(self):
		return self._vel
	
	@property
	def acc(self):
		return self._acc
	
	@property
	def yaw(self):
		return arctan2(self.vel.y, self.vel.x)
	
	def __repr__(self) -> str:
		return f'VehicleState(pos={self.pos}, vel={self.vel}, acc={self.acc})'


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

		self.path_data = []
	
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
	
	def get_object_function_gradient_numeric(self, final_state: Vector):
		self.set_final_state(final_state)
		T = self.final_time
		posnT = self.final_lat_pos
		veltT = self.final_long_vel

		dT = 0.0001
		dp = 0.0001
		dv = 0.0001

		J1u = self.get_object_function_value(np.array([T + dT, posnT, veltT]))
		J1d = self.get_object_function_value(np.array([T - dT, posnT, veltT]))
		J2u = self.get_object_function_value(np.array([T, posnT + dp, veltT]))
		J2d = self.get_object_function_value(np.array([T, posnT - dp, veltT]))
		J3u = self.get_object_function_value(np.array([T, posnT, veltT + dv]))
		J3d = self.get_object_function_value(np.array([T, posnT, veltT - dv]))
		
		dJ_numeric = np.array([(J1u-J1d)/(2*dT), (J2u-J2d)/(2*dp), (J3u-J3d)/(2*dv)])

		return dJ_numeric

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
		self.set_final_state(final_state)
		nt = 20
		ts = np.linspace(0, self.final_time, nt)
		g = self.get_constraint_function_value(ts)
		max_values = np.max(g, axis=0)
		return max_values


	def get_constraint_function_jacobian_numeric(self, final_state: Vector):
		self.set_final_state(final_state)
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
	
	def draw_path(self, axes, final_state, **kwargs):
		self.set_final_state(final_state)

		T, pen, vet = final_state
		ts = np.linspace(0, T, 20)

		new_ps = []
		for t in ts:
			pos = self.get_xy_pos(t)[0]
			new_ps.append(pos.numpy)
		new_ps = np.array(new_ps)

		axes.plot(new_ps[:,0], new_ps[:,1], color=COLOR_LINE)
		axes.plot(new_ps[-1,0], new_ps[-1,1], marker='^', color=COLOR_END)
		axes.plot(new_ps[0,0], new_ps[0,1], marker='o', color=COLOR_START)
		axes.grid(True)
		plt.pause(0.01)

	def plot_vel(self, axes, t, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		vel = self.get_xy_vel(ts)
		vel_mag = np.sqrt(vel.numpy[:,0]**2 + vel.numpy[:,1]**2)
		axes.plot(t + ts, vel_mag, color=COLOR_LINE)
		axes.plot(t + ts[0], vel_mag[0], color=COLOR_START, marker='o')
		axes.set_xlim([max(t - 15, -5), t + self.final_time])
		
	def plot_acc(self, axes, t, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		acc = self.get_xy_acc(ts)
		acc_mag = np.sqrt(acc.numpy[:,0]**2 + acc.numpy[:,1]**2)
		axes.plot(t + ts, acc_mag, color=COLOR_LINE)
		axes.plot(t + ts[0], acc_mag[0], color=COLOR_START, marker='o')
		axes.set_xlim([max(t - 15, -5), t + self.final_time])
		
	def plot_jrk(self, axes, t, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		jrk = self.get_xy_jrk(ts)
		jrk_mag = np.sqrt(jrk.numpy[:,0]**2 + jrk.numpy[:,1]**2)
		axes.plot(t + ts, jrk_mag, color=COLOR_LINE)
		axes.plot(t + ts[0], jrk_mag[0], color=COLOR_START, marker='o')
		axes.set_xlim([max(t - 15, -5), t + self.final_time])
		
	def plot_kpa(self, axes, t, final_state):
		self.set_final_state(final_state)
		ts = np.linspace(0, self.final_time, 20)
		kpa = self.get_curvature(ts)
		axes.plot(t + ts, kpa, color=COLOR_LINE)
		axes.plot(t + ts[0], kpa[0], color=COLOR_START, marker='o')
		axes.set_xlim([max(t - 15, -5), t + self.final_time])
		
	def lagrangian(self, f, gradf, g, gradg, x0, epsilon=1e-6, max_num_iter=1000, alpha=1e-3, beta=1e-3, **kwargs):
		N = len(x0) # no. of design variables
		M = len(g(x0)) # no. of inequality constraints
		mu0 = np.zeros(M) # Lagrangian multiplier
		
		# set initial values
		k = 0
		xk = x0
		muk = mu0

		# initial function and constraints values
		fk = f(xk)
		gk = g(xk)
		gradfk = gradf(xk)
		jacobgk = gradg(xk)
		# Lk = fk + (muk.reshape((-1,1)).transpose() @ gk.reshape((-1,1))).reshape(fk.shape)
		dk = gradfk + (jacobgk.reshape((-1,N)).transpose() @ muk.reshape((-1,1))).reshape(gradfk.shape)

		# Lagrangian algorithm for constrained optimization
		while (np.linalg.norm(dk) > epsilon): # stopping criteria 1
			xk_temp = xk
			xk = xk - alpha * dk.reshape(xk.shape)
			muk = vec_max_zero(muk + beta * g(xk_temp))
			
			fk = f(xk)
			gk = g(xk)
			gradfk = gradf(xk)
			jacobgk = gradg(xk)
			# Lk = fk + (muk.reshape((-1,1)).transpose() @ gk.reshape((-1,1))).reshape(fk.shape)
			dk = gradfk + (jacobgk.reshape((-1,N)).transpose() @ muk.reshape((-1,1))).reshape(gradfk.shape)

			k += 1
			if k == max_num_iter: # stopping criteria 2
				break
		
		# solutions to return
		x_opt = xk
		fval_opt = fk
		g_opt = gk

		return x_opt, fval_opt, k, g_opt

	def opt_search3(self, f, g, xmin, xmax, dx, **kwargs):
		[xss1, xss2, xss3] = np.meshgrid(
			np.arange(xmin[0],xmax[0],dx[0]),
			np.arange(xmin[1],xmax[1],dx[1]),
			np.arange(xmin[2],xmax[2],dx[2]),
		)
		fval_opt = np.inf
		x_opt = None
		ni, nj, nk = xss1.shape
		k = ni*nj*nk

		for i in range(ni):
			for j in range(nj):
				for k in range(nk):
					x = np.array([xss1[i][j][k], xss2[i][j][k], xss3[i][j][k]])
					if np.any(g(x) > 0):
						continue
					fx = f(x)
					if fx < fval_opt:
						fval_opt = fx
						x_opt = x

		return x_opt, fval_opt, k

	def append_path(self, pos, kpa, vel):
		yaw = arctan2(vel.y, vel.x)
		data = Vector([pos.x, pos.y, yaw, kpa, vel.norm])
		self.path_data.append(data)
	
	def save_path(self, filename):
		data = Matrix(self.path_data)
		posx = data[:,0]
		posy = data[:,1]
		vela = data[:,4]

		course = Road(posx, posy, 6, 1)
		pos = course.center_points
		yaw = course.yaws
		kpa = course.curvatures
		vel = np.interp(pos[:,0], posx, vela)
		save_data = np.vstack([pos[:,0], pos[:,1], yaw, kpa, vel]).transpose()

		filepath = os.path.join('./paths', filename)
		np.savetxt(filepath, save_data)
	
def load_path():
	data = np.loadtxt('./paths/s_curve.dat')
	posx = data[:,0]
	posy = data[:,1]
	yaw = data[:,2]
	kpa = data[:,3]
	vel = data[:,4]
	return posx, posy, yaw, kpa, vel

init_vehicle_state = State()

def gradient_mpc(axes):
	# parameters for simulation
	N_SIMULATION = 25 # number of simulation loops
	Ds = 30. # preview distance
	Dt = 0.8 # stepping time
	kph = 1/3.6
	vel0 = 20*kph

	# Create road model
	wx = [0.0, 20.0, 30, 50.0, 150]
	wy = [0.0, -6.0, 5.0, 6.5, 0.0]
	hwidth = 6 # half width of road
	ds = 2 # step size along the center line
	road = Road(wx, wy, hwidth, ds)

	# Create path planner
	vref = 10 # [m/s] vehicle target speed
	kj = 1 # weight for jerk object function (J1)
	kt = 0.004 # weight for time object function (J2)
	kd = 20 # weight for distance object function (J3)
	kv = 0.004 # weight for velocity object funciton (J4)
	dmax = 2 # max. distance from center line
	vmax = 15 # [m/s] max. velocity of vehicle
	amax = 9.81 # [m/s^2] max. acceleration of vehicle
	kmax = 3 # [1/m] max. curvature of path
	planner = PathPlanner(road, vref, kj, kt, kd, kv, dmax, vmax, amax, kmax)

	# draw road
	planner.road.draw(axes[0], ds=1, dmax=dmax)

	# initial travel range along the center line
	s0 = 0.
	sT = s0 + Ds

	# initial current vehicle states
	current_state = VehicleState(pos=Vector2D(0,0), vel=Vector2D(vel0,0), acc=Vector2D(0,0))

	# initial values of design variables
	T = 3 # travel time
	pnT = 0 # lateral position at T
	vtT = 10 # longitudinal velocity at T
	x0 = np.array([T, pnT, vtT]) # final state = design variables

	# set travel range, current and final state
	planner.set_travel_range(s0, sT)
	planner.set_current_state(current_state)
	planner.set_final_state(x0)

	# initialize time
	t = 0

	# optimization loop
	for idx in range(N_SIMULATION):
		f = planner.get_object_function_value
		g = planner.get_constraint_function_max_value
		gradf = planner.get_object_function_gradient_numeric
		gradg = planner.get_constraint_function_jacobian_numeric

		start_time = time.time()
		x_opt, fval_opt, k, g_opt = planner.lagrangian(f, gradf, g, gradg, x0, epsilon=1e-1, max_num_iter=50, alpha=1e-4, beta=1e-3, color='blue')
		# x_opt, fval_opt, k = planner.opt_search3(f, g, xmin=[1.5, -2, 6], xmax=[4.5, 2, 12], dx=[1, 0.5, 2])
		end_time = time.time()

		# pring log message
		print(f'Lagrangian: idx={idx+1}, x_opt={np.round(x_opt,2)}, fval_opt={np.round(fval_opt,2)}, g={np.round(g_opt,2)}, num_iter={k}, time={end_time - start_time:5.2f} sec')

		# draw the optimal path and plot signals
		planner.draw_path(axes[0], x_opt)
		planner.plot_vel(axes[1], t, x_opt)
		planner.plot_acc(axes[2], t, x_opt)
		planner.plot_jrk(axes[3], t, x_opt)
		planner.plot_kpa(axes[4], t, x_opt)
		plt.pause(0.01)

		# use the optimal solution as the next initial value
		x0 = x_opt
		
		# calculate the one-step-ahead position and set it as the current state
		t += Dt
		next_xy_pos = planner.get_xy_pos(Dt)[0]
		next_xy_vel = planner.get_xy_vel(Dt)[0]
		next_xy_acc = planner.get_xy_acc(Dt)[0]
		s0, pos0 = planner.road.get_projection_point(next_xy_pos)
		sT = s0 + Ds
		next_tn_pos = planner.road.convert_xy2tn(s0, next_xy_pos - pos0)
		next_tn_vel = planner.road.convert_xy2tn(s0, next_xy_vel)
		next_tn_acc = planner.road.convert_xy2tn(s0, next_xy_acc)
		next_kpa = (next_xy_vel.x * next_xy_acc.y - next_xy_vel.y * next_xy_acc.x) / ((next_xy_vel.y**2 + next_xy_vel.x**2)**(3/2) + 1e-6)
		current_state = VehicleState(next_tn_pos, next_tn_vel, next_tn_acc)
		planner.set_travel_range(s0, sT)
		planner.set_current_state(current_state)

		# store the optimal path data for simulation
		planner.append_path(next_xy_pos, next_kpa, next_xy_vel)

	planner.save_path('s_curve.dat')

	s0 = 0.
	sT = s0 + Ds
	planner.set_travel_range(s0, sT)

	current_state = VehicleState(pos=Vector2D(0,0), vel=Vector2D(20*kph,0), acc=Vector2D(0,0))
	planner.set_current_state(current_state)
	x = planner.current_state.pos.x
	y = planner.current_state.pos.y
	yaw = planner.current_state.yaw
	v = vel0

	init_vehicle_state = State(x, y, yaw, v)

def simulate(axes, state):
	cx, cy, cyaw, ck, sp = load_path()
	goal = [cx[-1], cy[-1]]
	do_simulation(axes[0], state, cx, cy, cyaw, ck, sp, goal)




if __name__ == '__main__':
	# Open a figure for animation and plot
	fig = plt.figure(1, figsize=(8,8))
	spec = gridspec.GridSpec(ncols=2, nrows=3, width_ratios=[1,1], height_ratios=[2,1,1], wspace=0.2, hspace=0.2)
	axes = []
	axes.append(plt.subplot2grid((3,2),(0,0),colspan=2))
	axes.append(plt.subplot2grid((3,2),(1,0)))
	axes.append(plt.subplot2grid((3,2),(1,1)))
	axes.append(plt.subplot2grid((3,2),(2,0)))
	axes.append(plt.subplot2grid((3,2),(2,1)))
	plt.tight_layout(pad=1.8)
	axes[0].set_aspect(1)
	axes[0].set_xlim(-5, 100)
	axes[0].set_ylim(-15, 15)
	axes[0].set_title('Press "A" to start and "R" to simulate.')
	axes[1].set_title('velocity [m/s]')
	axes[2].set_title('acceleration [m/s^2]')
	axes[3].set_title('jerk [m/s^3]')
	axes[4].set_title('curvature [1/m]')
	axes[3].set_xlabel('Time [sec]')
	axes[4].set_xlabel('Time [sec]')
	for ax in axes:
		ax.grid(True)

	# register keyboard press event handlers
	plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
	plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [gradient_mpc(axes) if event.key == 'a' else None])
	plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [simulate(axes, init_vehicle_state) if event.key == 'r' else None])

	plt.show()