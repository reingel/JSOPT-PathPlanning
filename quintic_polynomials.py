#
# JSOPT-PathPlanning: python project for path planning by optimization
#                     for autonomous vehicles
#
# optimization theory class, 2021 fall
#
# quintic_polynomials.py: python package for path modeling
#
# Developed and Maintained by Soonkyu Jeong (reingel@o.cnu.ac.kr)
#  since Nov. 21, 2021
#


import numpy as np
import matplotlib.pyplot as plt
from vector import *


class QuinticPolynomial:
	N_COEF = 6

	def __init__(self, T: float, pos0: float, vel0: float, acc0: float, posT: float, velT: float, accT: float):
		Tp = T**np.arange(self.N_COEF) # [1, T, T^2, T^3, T^4, T^5]

		del_p = posT - pos0 - vel0*T - 1/2*acc0*Tp[2]
		del_v = velT - vel0 - acc0*T
		del_a = accT - acc0

		self.pos_coef = Vector([
			pos0,
			vel0,
			1/2*acc0,
			( 10*del_p - 4*T*del_v + 1/2*Tp[2]*del_a) / Tp[3],
			(-15*del_p + 7*T*del_v -     Tp[2]*del_a) / Tp[4],
			(  6*del_p - 3*T*del_v + 1/2*Tp[2]*del_a) / Tp[5],
		])
		self.vel_coef = self.pos_coef[1:] * np.arange(1, self.N_COEF)
		self.acc_coef = self.vel_coef[1:] * np.arange(1, self.N_COEF - 1)
		self.jrk_coef = self.acc_coef[1:] * np.arange(1, self.N_COEF - 2)
	
	def pos(self, t) -> float:
		post = np.dot(self.pos_coef, t**column_vector(np.arange(self.N_COEF)))
		return post

	def vel(self, t) -> float:
		velt = np.dot(self.vel_coef, t**column_vector(np.arange(self.N_COEF-1)))
		return velt

	def acc(self, t) -> float:
		acct = np.dot(self.acc_coef, t**column_vector(np.arange(self.N_COEF-2)))
		return acct

	def jrk(self, t) -> float:
		jrkt = np.dot(self.jrk_coef, t**column_vector(np.arange(self.N_COEF-3)))
		return jrkt


class QuinticPolynomial2D:
	def __init__(self, T: float, pos0: Vector2D, vel0: Vector2D, acc0: Vector2D, posT: Vector2D, velT: Vector2D, accT: Vector2D):
		self.x_poly = QuinticPolynomial(T, pos0.x, vel0.x, acc0.x, posT.x, velT.x, accT.x)
		self.y_poly = QuinticPolynomial(T, pos0.y, vel0.y, acc0.y, posT.y, velT.y, accT.y)
	
	def pos(self, t) -> Vector2DArray:
		x = self.x_poly.pos(t)
		y = self.y_poly.pos(t)
		return Vector2DArray(zip(x, y))

	def vel(self, t) -> Vector2DArray:
		x = self.x_poly.vel(t)
		y = self.y_poly.vel(t)
		return Vector2DArray(zip(x, y))

	def acc(self, t) -> Vector2DArray:
		x = self.x_poly.acc(t)
		y = self.y_poly.acc(t)
		return Vector2DArray(zip(x, y))

	def jrk(self, t) -> Vector2DArray:
		x = self.x_poly.jrk(t)
		y = self.y_poly.jrk(t)
		return Vector2DArray(zip(x, y))

	def kpa(self, t) -> Vector:
		eps = 1e-6
		vel = self.vel(t)
		acc = self.acc(t)
		kappa = (vel.numpy[:,0] * acc.numpy[:,1] - vel.numpy[:,1] * acc.numpy[:,0]) / ((vel.numpy[:,1]**2 + vel.numpy[:,0]**2)**(3/2) + eps)
		return Vector(kappa)


if __name__ == '__main__':
	# 1D test
	qp = QuinticPolynomial(10, 0, 1, 0, 5, 0, -1)

	print(qp.pos(1).shape)
	print(qp.pos(np.array([1,2,3])).shape)

	t = np.linspace(0, 10, 20)
	pos = qp.pos(t)
	vel = qp.vel(t)
	acc = qp.acc(t)
	jrk = qp.jrk(t)

	plt.subplot(311)
	plt.grid(True)
	plt.gcf().canvas.mpl_connect(
		'key_release_event',
		lambda event: [exit(0) if event.key == 'escape' else None])
		
	plt.plot(t, pos)
	plt.plot(t, vel)
	plt.plot(t, acc)
	plt.plot(t, jrk)
	plt.legend(['pos', 'vel', 'acc', 'jrk'])

	# 2D test
	qp2 = QuinticPolynomial2D(
		10,
		Vector2D(0,0),
		Vector2D(1,0),
		Vector2D(0,0),
		Vector2D(5,1),
		Vector2D(0,1),
		Vector2D(0,0),
	)

	t = np.linspace(0, 10, 40)
	pos = qp2.pos(t)
	vel = qp2.vel(t)
	acc = qp2.acc(t)
	jrk = qp2.jrk(t)
	kpa = qp2.kpa(t)

	print(jrk)

	ii = 26 # max kappa

	plt.subplot(312)
	plt.grid(True)
		
	plt.plot(pos.numpy[:,0], pos.numpy[:,1])
	plt.plot(pos.numpy[ii,0], pos.numpy[ii,1], 'o')

	plt.subplot(313)
	plt.grid(True)

	plt.plot(t, kpa)
	plt.plot(t[ii], kpa[ii], 'o')

	plt.show()