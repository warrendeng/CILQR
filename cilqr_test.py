import jax.numpy as np

from jaxilqrMUTABLE import iLQR, ciLQR

class InversePendulum():
    def __init__(self, dt=0.02, l=1, horizon=300, state_size=3, 
        action_size=1, x_goal=np.array([0,1,0])):
        self.m = 1
        self.l = l
        self.g = 9.81
        self.dt = dt
        self.b = .1
        self.I = self.m * self.l**2
        self.cilqr = None
        self.N = horizon 

        self.Q = np.array([[self.l**2, self.l, 0], [self.l, self.l**2, 0], [0, 0, 0]])
        self.R = np.array([[0.1]])
        self.Q_T = 100 * np.eye(state_size)
        self.state_size = state_size
        self.action_size = action_size
        self.x_goal = x_goal
        return

    def dynamics(self, x, u, t):
        sin_theta = x[0]
        cos_theta = x[1]
        theta_dot = x[2]
        torque = u[0]
        theta = np.arctan2(sin_theta, cos_theta)

        theta_dot_dot = -3.0 * self.g / (2 * self.l) * np.sin(theta + np.pi)
        theta_dot_dot += 3.0 / (self.m * self.l**2) * torque
        next_theta = theta + theta_dot * self.dt
        x_new = np.array([np.sin(next_theta), np.cos(next_theta), theta_dot + theta_dot_dot
             * self.dt])
        return x_new

    def cost(self, x, u, t):
        return x.T.dot(self.Q).dot(x) + u.T.dot(self.R).dot(u)

    def fcost(self, x, t):
        return (x-self.x_goal).T.dot(self.Q_T).dot(x-self.x_goal)


    def build(self):
        print(self.cost(np.array([0, -1, 0]).astype(float), np.array([0]), 0))
        self.cilqr = ciLQR(dynamics=self.dynamics, cost=self.cost, final_cost=self.fcost, N=self.N, state_size=self.state_size, action_size=self.action_size, t=2, mu=1.5)
        print("Build successful")
        return
    def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
        info = "converged" if converged else ("accepted" if accepted else "failed")
        print("iteration", iteration_count, info, J_opt)
    def solve(self, x0=None, us_init=None, n_iterations=150):
        #prepare initial guess
        if us_init is None:
            us_init = np.zeros((self.N, self.action_size))
        if x0 is None:
            x0 = np.array([np.random.rand()*2*np.pi - np.pi, 0])

        if self.ilqr is not None:
            self.xs, self.us = self.cilqr.outer_loop(x0, us_init, n_iterations=n_iterations, on_iteration=self.on_iteration)
        else:
            print("Error creating iLQR object")
        return


if __name__ == '__main__':

    problem = InversePendulum()
    x0 = np.array([0, -1, 0]).astype(float) #initial state sin theta, cos theta, theta_dot
    xgoal = np.array([0,1,0]).astype(float)
    problem.build() #initialize iLQR solver
    problem.solve(x0) #solve iLQR problem given some initial state
    print(problem.xs)
    print(problem.us)
