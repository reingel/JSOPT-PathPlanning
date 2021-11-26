import sys
sys.path.append('./JSOPT')

import numpy as np
from constant import *
from linalg import Vector, normalized
from line_search import bisection

def vec_max_zero(x: np.ndarray):
	return np.array([max(xi, 0) for xi in x])

def lagrangian(f, gradf, g, gradg, x0, epsilon=1e-6, max_num_iter=1000, alpha=1e-3, beta=1e-3):
	"""
	sda(): minimize f(x) using the steepest descent algorithm
	<function description>
	1. input arguments
		- f: an objective function f(x) (function)
		- gradf: the gradient of f(x) (function)
		- x0: a starting point of optimization (numpy.ndarray a.k.a. Vector)
		- epsilon: the 1st stopping criteria. sda() will stop if |gradf(xk)| <= epsilon. (float)
		- max_num_iter: the 2nd stopping criteria. sda() will stop if the number of iterations is greater than max_num_iter. (integer)
		- line_search: a line search algorithm (function)
		- alpha_max: the max. range of line search (numeric or function)
		- ls_epsilon: epsilon for the line search algorithm (float)
		- ls_max_num_iter: max_num_iter for the line search algorithm (integer)
	2. return values
		- x_opt: the minimimum point of f(x) (numpy.ndarray a.k.a. Vector)
		- fval_opt: the minimum value of f(x) (float)
		- status: (integer)
			- 0 if the minimum point is found within max_num_iter
			- 1 if the number of iterations reaches max_num_iter
		- history: sequencially stored values of x, d, fval, rate_conv (dictionary)
	"""

	# Lagrangian multiplier
	try:
		N = len(gradg(x0))
	except:
		N = 1
	mu0 = np.zeros(N)
	
	# set initial values
	k = 0
	xk = x0
	muk = mu0
	Lk = f(xk) + np.dot(g(xk), muk)
	dk = gradf(xk) + np.dot(gradg(xk).transpose(), muk)

	# create additional return values
	status = CONVERGED
	history = {'x': [xk], 'd': [dk], 'L': [Lk]}

	# search loop
	while (np.linalg.norm(dk) > epsilon): # stopping criteria 1
		xk_temp = xk
		xk = xk - alpha * dk
		muk = vec_max_zero(muk + beta * g(xk_temp))
		
		Lk = f(xk) + np.dot(g(xk), muk)
		dk = gradf(xk) + np.dot(gradg(xk).transpose(), muk)

		# store histories
		history['x'].append(xk)
		history['d'].append(dk)
		history['L'].append(Lk)

		k += 1
		if k == max_num_iter: # stopping criteria 2
			status = REACHED_MAX_ITER
			print(f'Lagrangian: reached the maximum number of iteration: {k}')
			break
	
	# solutions to return
	x_opt = xk
	L_opt = Lk

	# convert to numpy array
	history['x'] = np.array(history['x'])
	history['d'] = np.array(history['d'])
	history['L'] = np.array(history['L'])

	# calculate and add the rate of convergence to history
	history['rate_conv'] = (history['L'][1:] - L_opt) / (history['L'][:-1] - L_opt)
	history['rate_conv'] = np.insert(history['rate_conv'], 0, 1.) # insert 1 at 0-index to match its length with others

	return x_opt, L_opt, status, history


# test code
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	f = lambda x: x[0]**2 + 2*(x[1]-1)**2
	gradf = lambda x: Vector([2*x[0], 4*(x[1]-1)])
	
	g = lambda x: 2*x[0] + 3 * x[1] + 5
	gradg = lambda x: Vector([2, 3])
	
	x0 = Vector([-2, -1.4])

	# def calc_alpha_max(x, d):
	# 	alpha_max = 2
	# 	if d[1] > 0:
	# 		alpha_max = min(alpha_max, (0.5 - x[1]) / d[1])
	# 	return alpha_max
	calc_alpha_max = 9

	x_opt, L_opt, status, history = lagrangian(f, gradf, g, gradg, x0, epsilon=1e-1, max_num_iter=500, alpha=1e-3, beta=1e-2)

	print(f"Lagrangian: {status=}, x_opt={np.round(x_opt,2)}, L_opt={np.round(L_opt,2)}, num_iter={len(history['x'])-1}")

	fig, ax = plt.subplots(2,1)
	ax[0].set_aspect(1.0)
	ax[0].grid(True)
	x = np.linspace(-3,3,50)
	y = np.linspace(-3,2,40)
	[xx, yy] = np.meshgrid(x, y)
	zz = np.zeros_like(xx)
	for i in range(len(x)):
		for j in range(len(y)):
			zz[j,i] = f((x[i],y[j]))
	ax[0].contour(xx, yy, zz)
	ax[0].plot([-3,1/3],[3,-11/3])
	ax[0].scatter(history['x'][:,0], history['x'][:,1])
	ax[0].scatter(history['x'][0,0], history['x'][0,1], marker='^', color='green')
	ax[0].scatter(history['x'][-1,0], history['x'][-1,1], color='red')
	ax[1].grid(True)
	# ax[1].plot(history['d'], label='d')
	ax[1].plot(history['L'], label='L')
	ax[1].plot(history['rate_conv'], label='rate_conv')
	ax[1].legend(loc='best')
	plt.show()
