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
		xk = xk - alpha * dk
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

		# print the log message
		if k//100*100 == k:
			print(f'Lagrangian: {k}  {fk=:5.2f}  {Lk=:5.2f}  x={np.round(xk,2)}')

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


def opt_search2(f, g, xmin, xmax, dx):
	[xss1, xss2] = np.meshgrid(
		np.arange(xmin[0],xmax[0],dx[0]),
		np.arange(xmin[1],xmax[1],dx[1]),
		# np.arange(xmin[2],xmax[2],dx[2]),
	)
	fmin = np.inf
	xopt = None
	ni, nj = xss1.shape
	for i in range(ni):
		for j in range(nj):
			x = np.array([xss1[i][j], xss2[i][j]])
			if np.any(g(x) > 0):
				continue
			fx = f(x)
			if fx < fmin:
				fmin = fx
				xopt = x
	return xopt
	
def opt_search3(f, g, xmin, xmax, dx):
	[xss1, xss2, xss3] = np.meshgrid(
		np.arange(xmin[0],xmax[0],dx[0]),
		np.arange(xmin[1],xmax[1],dx[1]),
		np.arange(xmin[2],xmax[2],dx[2]),
	)
	fmin = np.inf
	xopt = None
	ni, nj, nk = xss1.shape
	print(ni,nj,nk)
	for i in range(ni):
		for j in range(nj):
			for k in range(nk):
				x = np.array([xss1[i][j][k], xss2[i][j][k], xss3[i][j][k]])
				if np.any(g(x) > 0):
					continue
				fx = f(x)
				if fx < fmin:
					fmin = fx
					xopt = x
		print(i,j,k)
	return xopt, fmin, ni*nj*nk
	

# test code
if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	f = lambda x: x[0]**2 + (x[1]-1)**2
	gradf = lambda x: Vector([2*x[0], 2*(x[1]-1)])
	
	g = lambda x: 2*x[0] + 3 * x[1] + 5
	gradg = lambda x: Vector([2, 3])
	
	x0 = Vector([-2, -1.4])

	# x_opt, fval_opt, k, status, history = lagrangian(f, gradf, g, gradg, x0, epsilon=1e-2, max_num_iter=500, alpha=1e-2, beta=1e-2)

	# print(f"Lagrangian: {status=}, x_opt={np.round(x_opt,2)}, fval_opt={np.round(fval_opt,2)}, num_iter={k}")

	# fig, ax = plt.subplots(2,1)
	# ax[0].set_aspect(1.0)
	# ax[0].grid(True)
	# x = np.linspace(-3,3,50)
	# y = np.linspace(-3,2,40)
	# [xx, yy] = np.meshgrid(x, y)
	# zz = np.zeros_like(xx)
	# for i in range(len(x)):
	# 	for j in range(len(y)):
	# 		zz[j,i] = f((x[i],y[j]))
	# ax[0].contour(xx, yy, zz)
	# ax[0].plot([-3,3],[1/3,-11/3])
	# ax[0].scatter(history['x'][:,0], history['x'][:,1])
	# ax[0].scatter(history['x'][0,0], history['x'][0,1], marker='^', color='green')
	# ax[0].scatter(history['x'][-1,0], history['x'][-1,1], color='red')
	# ax[1].grid(True)
	# k = history['k']
	# ax[1].plot(k, history['fval'], label='fval', marker='.')
	# ax[1].plot(k, history['rate_conv'], label='rate_conv', marker='.')
	# ax[1].legend(loc='best')
	# plt.gcf().canvas.mpl_connect(
	# 	'key_release_event',
	# 	lambda event: [exit(0) if event.key == 'escape' else None])
	# plt.show()

	x_opt = opt_search2(f, g, xmin=[-2, -2], xmax=[2, 2], dx=[0.1, 0.1])
	print(x_opt)

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
	ax[0].plot([-3,3],[1/3,-11/3])
	ax[0].scatter(x_opt[0], x_opt[1], color='red')
	ax[1].grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
	plt.show()

