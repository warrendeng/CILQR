import jax
from jax import lax, grad, jacfwd, jacobian, vmap
import warnings
import jax.numpy as np

class ciLQR:
	def __init(self, dynamics, cost, final_cost, constraint, final_constraint, N, state_size, action_size, mu, t, tol=1e-10):
		self.cost = cost
		self.dynamics = dynamics
		self.final_cost = final_cost
		self.N = N
		self.constraint = constraint
		self.final_constraint = final_constraint
		self.state_size = state_size
		self.action_size = action_size
		self.mu = mu
		self.t = t
		self.iLQR = None   
	def outer_loop():
		has_converged = False
		while (!has_converged):
			self.iLQR = iLQR(dynamics=self.dynamics, cost=self.cost, final_cost=self.fcost, N=self.N, state_size=self.state_size, action_size=self.action_size)
			self.xs, self.us = self.iLQR.fit(self.xo, self.us_init)
			self.t = self.mu * self.t
			
			#check convergence 
		return xs, us

	def final_cost_transformed(self, x, u, t):
		return self.final_cost(x,t) - (1/self.t) * np.log(-self.constraint(x,u,t))

	def cost_transformed(self, x, u, t):
		return self.cost(x,u,t) - (1/self.t) * np.log(-self.constraint(x,u,t))  


class iLQR:

	#dynamics must be written as a class
	def __init__(self, dynamics, cost, final_cost, N, state_size, action_size, max_reg=1e10, hessians=False):

		self.f = dynamics
		self.l = cost
		self.N = N
		self.state_size = state_size
		self.action_size = action_size

		self.f_x = jacobian(self.f, 0)
		self.f_u = jacobian(self.f, 1)
		self.l_x = grad(self.l, 0)
		self.l_u = grad(self.l, 1)
		self.l_xx = jacobian(self.l_x, 0) 
		self.l_ux = jacobian(self.l_u, 0) 
		self.l_uu = jacobian(self.l_u, 1)

		self.nl = final_cost
		self.nl_x = grad(self.nl, 0)
		self.nl_xx = jacobian(self.nl_x, 0)

		self._mu = 1.0
		self._mu_min = 1e-6
		self._mu_max = max_reg
		self._delta_0 = 2.0
		self._delta = self._delta_0

		self._k = np.zeros((N, self.action_size))
		self._K = np.zeros((N, self.action_size, self.state_size))
		self._use_hessians = hessians
		return

	"""
		x[idx] = y
		jax.ops.index_update(x, idx, y) returns a new matrix for x
		jax.ops.index_update(x, jax.ops.index[::2, 3:], 6)
		[::2, 3:] slices the every other row from the third column to the last column
	"""

	def fit(self, x0, us_init, n_iterations=100, tol=1e-6, on_iteration=None):
		"""
		1. simulate the system on x0 and u_sample to get all of the states for this iteration
		2. Linearize the dynamics and cost function
		3. Do LQR
			a. Work backwards starting from final state to get k and K (optimal inputs)
			b. Get new states x_new
		4. Calculate cost of new trajectory and input
			a. Adjust parameters (lambda I think) so that next iteration is better
		"""
		self._mu = 1.0
		self._delta = self._delta_0

		alphas = 1.1**(-np.arange(10)**2)

		us = us_init.copy()
		k = self._k
		K = self._K

		changed = True
		converged = False
		for i in range(n_iterations):
			print("Iteration:", i)
			accepted = False
			# Forward rollout only if it needs to be recomputed.
			if changed:
				(xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu) = self.forward_rollout(x0, us)
				J_opt = L.sum()
				changed = False

			#calculate partial derivatives of f(xt, ut) for each x in traject
			#also calculate all cost derivatives 
			try:
				k, K = self.backward_pass(F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu,
				                           F_xx, F_ux, F_uu)
				# Backtracking line search.
				for alpha in alphas:
					xs_new, us_new = self.control(xs, us, k, K, alpha)
					J_new = self.trajectory_cost(xs_new, us_new)
					print("Current Cost: ", J_new)
					#print("alpha", alpha)
					if J_new < J_opt:
						if np.abs((J_opt - J_new) / J_opt) < tol:
						    converged = True

						J_opt = J_new
						xs = xs_new
						us = us_new
						changed = True

						# Decrease regularization term.
						self._delta = min(1.0, self._delta) / self._delta_0
						self._mu *= self._delta
						if self._mu <= self._mu_min:
						    self._mu = 0.0

						# Accept this.
						accepted = True
						break
			except np.linalg.LinAlgError as e:
				warnings.warn(str(e))

			if not accepted:
				# Increase regularization term.
				self._delta = max(1.0, self._delta) * self._delta_0
				self._mu = max(self._mu_min, self._mu * self._delta)
				if self._mu_max and self._mu >= self._mu_max:
					warnings.warn("exceeded max regularization term")
					break
			if on_iteration:
				on_iteration(xs, us, J_opt, accepted, converged)

			if converged:
			    break

		# Store fit parameters.
		self._k = k
		self._K = K
		self._nominal_xs = xs
		self._nominal_us = us
		print("Done")
		return xs, us


	def control(self, xs, us, k, K, alpha = 1.0):
		xs_new = np.zeros_like(xs)
		us_new = np.zeros_like(us)
		xs_new = jax.ops.index_update(xs_new, 0, xs[0])
		for i in range(self.N):
			us_new = jax.ops.index_update(us_new, i, us[i] + alpha * k[i] + K[i].dot(xs_new[i] - xs[i]))
			xs_new = jax.ops.index_update(xs_new, i+1, self.f(xs_new[i], us_new[i], i))
		return xs_new, us_new

	def trajectory_cost(self, xs, us):
		J = map(lambda args: self.l(*args), zip(xs[:-1], us, range(self.N)))
		return sum(J) + self.nl(xs[-1], self.N)

	def forward_rollout(self, x0, us):
		state_size = self.state_size
		action_size = self.action_size
		N = us.shape[0] # how many inputs there are for a trajectory
		# xs = np.empty((N + 1, state_size)) # first is the initial state
		# F_x = np.empty((N, state_size, state_size))
		# F_u = np.empty((N, state_size, action_size))

		# if self._use_hessians:
		# 	F_xx = np.empty((N, state_size, state_size, state_size))
		# 	F_ux = np.empty((N, state_size, action_size, state_size))
		# 	F_uu = np.empty((N, state_size, action_size, action_size))
		# else:
		# 	F_xx = None
		# 	F_ux = None
		# 	F_uu = None

		# L = np.empty(N + 1) #array of costs 
		# L_x = np.empty((N + 1, state_size))
		# L_u = np.empty((N, action_size))
		# L_xx = np.empty((N + 1, state_size, state_size))
		# L_ux = np.empty((N, action_size, state_size))
		# L_uu = np.empty((N, action_size, action_size))

		xs = np.empty((N + 1, state_size))
		F_x = np.empty((N, state_size, state_size))
		F_u = np.empty((N, state_size, action_size))

		if self._use_hessians:
			F_xx = np.empty((N, state_size, state_size, state_size))
			F_ux = np.empty((N, state_size, action_size, state_size))
			F_uu = np.empty((N, state_size, action_size, action_size))
		else:
			F_xx = None
			F_ux = None
			F_uu = None

		L = np.empty(N + 1)
		L_x = np.empty((N + 1, state_size))
		L_u = np.empty((N, action_size))
		L_xx = np.empty((N + 1, state_size, state_size))
		L_ux = np.empty((N, action_size, state_size))
		L_uu = np.empty((N, action_size, action_size))

		xs = jax.ops.index_update(xs, 0, x0)
		for i in range(N):
			x = xs[i]
			u = us[i]

			xs = jax.ops.index_update(xs, i+1, self.f(x, u, i))
			F_x = jax.ops.index_update(F_x, i, self.f_x(x, u, i))
			F_u = jax.ops.index_update(F_u, i, self.f_u(x, u, i))

			L = jax.ops.index_update(L, i, self.l(x, u, i))
			L_x = jax.ops.index_update(L_x, i, self.l_x(x, u, i))
			L_u = jax.ops.index_update(L_u, i, self.l_u(x, u, i))
			L_xx = jax.ops.index_update(L_xx, i, self.l_xx(x, u, i))
			L_ux = jax.ops.index_update(L_ux, i, self.l_ux(x, u, i))
			L_uu = jax.ops.index_update(L_uu, i, self.l_uu(x, u, i))

			if self._use_hessians:
				F_xx = jax.ops.index_update(F_xx, i, self.dynamics.f_xx(x, u, i))
				F_ux = jax.ops.index_update(F_ux, i, self.dynamics.f_ux(x, u, i))
				F_uu = jax.ops.index_update(F_uu, i, self.dynamics.f_uu(x, u, i))

		x = xs[-1]
		L = jax.ops.index_update(L, -1, self.nl(x, N))
		L_x= jax.ops.index_update(L_x, -1, self.nl_x(x, N))
		L_xx = jax.ops.index_update(L_xx, -1, self.nl_xx(x, N))
		print("Trajectory xs:", xs)
		return xs, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu, F_xx, F_ux, F_uu 

	def backward_pass(self, F_x, F_u, L_x, L_u, L_xx, L_ux, L_uu, F_xx=None, F_ux=None, F_uu=None):
		"""Computes the feedforward and feedback gains k and K.

		Args:
		    F_x: Jacobian of state path w.r.t. x [N, state_size, state_size].
		    F_u: Jacobian of state path w.r.t. u [N, state_size, action_size].
		    L_x: Jacobian of cost path w.r.t. x [N+1, state_size].
		    L_u: Jacobian of cost path w.r.t. u [N, action_size].
		    L_xx: Hessian of cost path w.r.t. x, x
		        [N+1, state_size, state_size].
		    L_ux: Hessian of cost path w.r.t. u, x [N, action_size, state_size].
		    L_uu: Hessian of cost path w.r.t. u, u
		        [N, action_size, action_size].
		    F_xx: Hessian of state path w.r.t. x, x if Hessians are used
		        [N, state_size, state_size, state_size].
		    F_ux: Hessian of state path w.r.t. u, x if Hessians are used
		        [N, state_size, action_size, state_size].
		    F_uu: Hessian of state path w.r.t. u, u if Hessians are used
		        [N, state_size, action_size, action_size].

		Returns:
		    Tuple of
		        k: feedforward gains [N, action_size].
		        K: feedback gains [N, action_size, state_size].
		"""
		V_x = L_x[-1]
		V_xx = L_xx[-1]


		k = np.empty_like(self._k)
		K = np.empty_like(self._K)
		for i in range(self.N - 1, -1, -1):
			# if self._use_hessians:
			#     Q_x, Q_u, Q_xx, Q_ux, Q_uu = self._Q(F_x[i], F_u[i], L_x[i],
			#                                          L_u[i], L_xx[i], L_ux[i],
			#                                          L_uu[i], V_x, V_xx,
			#                                          F_xx[i], F_ux[i], F_uu[i])
			# else:
			Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.Q(F_x[i], F_u[i], L_x[i],
			                                         L_u[i], L_xx[i], L_ux[i],
			                                         L_uu[i], V_x, V_xx)

			# Eq (6).
			k = jax.ops.index_update(k, i, -np.linalg.solve(Q_uu, Q_u))
			K = jax.ops.index_update(K, i, -np.linalg.solve(Q_uu, Q_ux))
			# inv_Q_uu = np.linalg.pinv(Q_uu)
			# print("pseudoInv", inv_Q_uu)
			# k[i] = -inv_Q_uu.dot(Q_u)
			# K[i] = -inv_Q_uu.dot(Q_ux)

			# Eq (11b).
			V_x = Q_x + K[i].T.dot(Q_uu).dot(k[i])
			V_x += K[i].T.dot(Q_u) + Q_ux.T.dot(k[i])

			# Eq (11c).
			V_xx = Q_xx + K[i].T.dot(Q_uu).dot(K[i])
			V_xx += K[i].T.dot(Q_ux) + Q_ux.T.dot(K[i])
			V_xx = 0.5 * (V_xx + V_xx.T)  # To maintain symmetry.
		return np.array(k), np.array(K)

	# def pseudoInv(self, mat, reg=1e-5):
	# 	"""
	# 	Use SVD to realize persudo inverse by perturbing the singularity values
	# 	to ensure its positive-definite properties
	# 	"""
	# 	u, s, v = np.linalg.svd(mat)
	# 	s[ s < 0 ] = 0.0        #truncate negative values...
	# 	diag_s_inv = np.zeros((v.shape[0], u.shape[1]))
	# 	diag_s_inv[0:len(s), 0:len(s)] = np.diag(1./(s+reg))
	# 	return v.dot(diag_s_inv).dot(u.T)


	def Q(self, f_x, f_u, l_x, l_u, l_xx, l_ux, l_uu, V_x, V_xx, f_xx=None, f_ux=None, f_uu=None):
		"""Computes second order expansion.

		Args:
		    F_x: Jacobian of state w.r.t. x [state_size, state_size].
		    F_u: Jacobian of state w.r.t. u [state_size, action_size].
		    L_x: Jacobian of cost w.r.t. x [state_size].
		    L_u: Jacobian of cost w.r.t. u [action_size].
		    L_xx: Hessian of cost w.r.t. x, x [state_size, state_size].
		    L_ux: Hessian of cost w.r.t. u, x [action_size, state_size].
		    L_uu: Hessian of cost w.r.t. u, u [action_size, action_size].
		    V_x: Jacobian of the value function at the next time step
		        [state_size].
		    V_xx: Hessian of the value function at the next time step w.r.t.
		        x, x [state_size, state_size].
		    F_xx: Hessian of state w.r.t. x, x if Hessians are used
		        [state_size, state_size, state_size].
		    F_ux: Hessian of state w.r.t. u, x if Hessians are used
		        [state_size, action_size, state_size].
		    F_uu: Hessian of state w.r.t. u, u if Hessians are used
		        [state_size, action_size, action_size].

		Returns:
		    Tuple of
		        Q_x: [state_size].
		        Q_u: [action_size].
		        Q_xx: [state_size, state_size].
		        Q_ux: [action_size, state_size].
		        Q_uu: [action_size, action_size].
		"""
		# Eqs (5a), (5b) and (5c).
		# print("f_x", f_x)
		# print("f_u", f_u)
		# print("l_x", l_x) 
		# print("l_u", l_u) 
		# print("l_xx", l_xx)
		# print("l_ux", l_ux) #need to transpose this
		# print("l_uu", l_uu)
		# print("V_x", V_x)
		# print("V_xx", V_xx)
		Q_x = l_x + f_x.T.dot(V_x)
		Q_u = l_u + f_u.T.dot(V_x)
		Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)

		# Eqs (11b) and (11c).
		reg = self._mu * np.eye(self.state_size)
		Q_ux = l_ux + f_u.T.dot(V_xx + reg).dot(f_x) # Wrong size currently its state x state
		Q_uu = l_uu + f_u.T.dot(V_xx + reg).dot(f_u)

		# if self._use_hessians:
		#     Q_xx += np.tensordot(V_x, f_xx, axes=1)
		#     Q_ux += np.tensordot(V_x, f_ux, axes=1)
		#     Q_uu += np.tensordot(V_x, f_uu, axes=1)

		# print("State Size", self.state_size)
		# print("Action Size", self.action_size)
		# print("Q_x", Q_x)
		# print("Q_u", Q_u)
		# print("Q_xx", Q_xx)
		# print("Q_ux", Q_ux) 
		# print("Q_uu", Q_uu)

		return Q_x, Q_u, Q_xx, Q_ux, Q_uu