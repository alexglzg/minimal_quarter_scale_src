import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from scipy.optimize import minimize

e = ca.MX.sym('e', 4*4) # 4*4 since working with rectangles (robot, free space) with 4 vertices each
alphaMX = ca.MX.sym('alpha')
lseMax = ca.Function('lseMax', [e, alphaMX], [smooth_max(e, alphaMX)])
lseMin = ca.Function('lseMin', [e, alphaMX], [smooth_min(e, alphaMX)])

# MPC parameters
N = 21          # horizon length
dt = 0.1        # timestep duration
N_cbf = 6
max_approx = 5e-3

# Initial boat state
starting_angle = 0.0
ned_x = 0.75
ned_y = 0.85
u_ref = 0.3
x_multiplier = -0.2

s_0 = minimize(path_w_args, 0, method='nelder-mead', args=(ned_x, ned_y, x_multiplier), options={'xatol': 1e-8, 'disp': True})
s_0 = s_0.x

current_X = ca.vertcat(ned_x,ned_y,starting_angle,0,0,0,s_0)  # initial state

# State and control dimensions
nx = 7         # states: ...
nu = 4          # inputs: v (linear vel), omega (angular vel)

# Define rectangle in body frame using polytope Ax <= b
# Robot size (e.g., 1.0m long, 0.6m wide)
L = 0.1
W = 0.1
A_robot = np.array([
    [1, 0],   #  x <= L/2
    [-1, 0],  # -x <= L/2
    [0, 1],   #  y <= W/2
    [0, -1]  # -y <= W/2
])
b_robot = np.array([
    L/2,
    L/2,
    W/2,
    W/2
])
polytope_robot_f = get_transformed_polytope_ca(A_robot, b_robot)

# Obstacle size
L_obs = 0.8
W_obs = 0.5

# Polytope definition for rectangle centered at origin (robot frame)
# Necessary for QP-CBF formulation
A_obs = np.array([
    [1, 0],    #  x <= L/2
    [-1, 0],   # -x <= L/2
    [0, 1],    #  y <= W/2
    [0, -1]   # -y <= W/2
])
b_obs = np.array([
    L_obs/2,
    L_obs/2,
    W_obs/2,
    W_obs/2
])

p_obs = np.array([0.0, 1.0])

# Free space polytope (convention Ax + b >= 0)
A_free = [[0, -1], [-1, 0], [0, 1], [1, 0]]
b_free = [-1, -1, -1, -1]

# Get vertices from polytope
obstacle_vertices = rectangle_vertices_from_polytope(A_obs, b_obs)

b_obs = np.dot(A_obs, p_obs) + b_obs

# Transform to world position (no rotation for fixed axis-aligned obstacle)
obstacle_world = obstacle_vertices + p_obs
obstacle_world = np.vstack((obstacle_world, obstacle_world[0]))  # close the loop

# Weights in cost
Q = np.diag([10, 10, 0])   # state tracking weight
R = np.diag([0.1, 0.1, 0.1, 0.1])      # control effort weight
# Qye = 20
# Qr = 0.1
# Qpsi = 10.0
# Qu = 200.0

gamma = 0.8
# gamma = 1.0
pomega = 10.0
margin_dist = 0.1

# Reference state (goal)
x_ref = np.array([-0.85, -0.85, 0])

# CasADi optimization environment
opti = ca.Opti()

# # Decision variables
# X = opti.variable(nx, N+1)   # states over horizon
# U = opti.variable(nu, N)     # controls over horizon

X = []
Lamb = []
Mu = []
Omega = []
U = []
for k in range(N):
    Xk = opti.variable(nx)
    Uk = opti.variable(nu)
    X.append(Xk)
    U.append(Uk)
Xk = opti.variable(nx)
X.append(Xk)

# Initial state parameter
X0 = opti.parameter(nx)

# # Objective function
# cost = 0
# for k in range(N):
#     xk = X[k]
#     uk = U[k]
#     cost += ca.mtimes([(xk - x_ref).T, Q, (xk - x_ref)]) + ca.mtimes([uk.T, R, uk])
# cost += ca.mtimes([(X[N] - x_ref).T, Q*10, (X[N] - x_ref)])  # terminal cost

# Lagrange objective
cost = 0
for k in range(N):
    xk = X[k]
    uk = U[k]

    cost += ca.mtimes([(xk[0:3] - x_ref).T, Q, (xk[0:3] - x_ref)]) + ca.mtimes([uk.T, R, uk])

cost += ca.mtimes([(X[N][0:3] - x_ref).T, Q*10, (X[N][0:3] - x_ref)])  # terminal cost


# Cost function from fatrop_definitions/avoidance_cbf.py
#     x_d = x_multiplier*xk[6]
#     y_d = 0#sin(s*x_multiplier)+3
#     xdot_d = x_multiplier
#     ydot_d = 0#x_multiplier*cos(x_multiplier * s)
#     gamma_p = ca.atan2(ydot_d, xdot_d)
#     ye = -(xk[0]-x_d)*ca.sin(gamma_p)+(xk[1]-y_d)*ca.cos(gamma_p)

#     cost += Qye*(ye**2) + Qr*(xk[5]**2) + Qpsi*(ca.sin(xk[2])-ca.sin(gamma_p))**2 + Qpsi*(ca.cos(xk[2])-ca.cos(gamma_p))**2 + Qu*(xk[3]-u_ref)**2 + uk[0]**2 + uk[1]**2 + uk[2]**2 + uk[3]**2
# cost += Qye*(ye**2) + Qr*(xk[5]**2) + Qpsi*(ca.sin(xk[2])-ca.sin(gamma_p))**2 + Qpsi*(ca.cos(xk[2])-ca.cos(gamma_p))**2 + Qu*(xk[3]-u_ref)**2

# Dynamics constraints (boat dynamics QuarterScale)
for k in range(N):
    xk = X[k]
    uk = U[k]
    x_next = X[k+1]

    # system equations
    dxdt = boat_dynamics(xk, uk)
    x_next_model = xk + dxdt * dt

    opti.subject_to(x_next == x_next_model)

    if k == 0:
        # Initial condition constraint
        opti.subject_to(X[0] == X0)

    # Input constraints
    max_force_limit = 6
    opti.subject_to( (-max_force_limit <= uk[0]) <= max_force_limit )
    opti.subject_to( (-max_force_limit <= uk[1]) <= max_force_limit )
    opti.subject_to( (-max_force_limit <= uk[2]) <= max_force_limit )
    opti.subject_to( (-max_force_limit <= uk[3]) <= max_force_limit )

    # CBF constraints
    if k < N_cbf:
        # # Simply free space constraints
        # for i in range(1, N):
        #     rob_vertices = robot_vertices_ca(X[0,i], X[1,i], X[2,i], L, W)
        #     for j in range(rob_vertices.shape[0]):
        #         for k in range(len(A_free)):
        #             opti.subject_to(ca.dot(ca.MX(A_free[k]), rob_vertices[j, :].T) - b_free[k] >= 0)  # Should be >=0

        # CBF for xk
        vec = []
        rob_vertices = robot_vertices_ca(X[k][0], X[k][1], X[k][2], L, W)
        for j in range(rob_vertices.shape[0]):
            for l in range(len(A_free)):
                vec.append((ca.dot(ca.MX(A_free[l]), rob_vertices[j, :].T) - b_free[l])) # Should be >=0
        
        alpha = ca.log(len(vec)) / max_approx
        lse_cbf_val_xk = lseMin(ca.vertcat(*vec), alpha)

        # CBF for xk+1
        vec = []
        # rob_vertices = robot_vertices_ca(X[k + 1][0], X[k + 1][1], X[k + 1][2], L, W)
        rob_vertices = robot_vertices_ca(x_next_model[0], x_next_model[1], x_next_model[2], L, W)
        for j in range(rob_vertices.shape[0]):
            for l in range(len(A_free)):
                vec.append((ca.dot(A_free[l], rob_vertices[j, :].T) - b_free[l])) # Should be >=0    

        alpha = ca.log(len(vec)) / max_approx
        lse_cbf_val_xk1 = lseMin(ca.vertcat(*vec), alpha)

        opti.subject_to(lse_cbf_val_xk1 >= gamma * lse_cbf_val_xk + (1 - gamma)*max_approx)

opti.minimize(cost)

# Solver options
# opts = {"ipopt.print_level":4, "print_time":1, 'ipopt.max_iter': 100, "ipopt.tol":1e-4, 'ipopt.linear_solver': 'ma57', 'expand': True}
# opti.solver('ipopt', opts)
opts ={"fatrop.print_level":4, "print_time":1, 'fatrop.max_iter': 100, "fatrop.tol":1e-4, 'structure_detection': 'auto', 'expand': True, 'debug': False}
opti.solver('fatrop', opts)

# Simulation parameters
sim_time = 20  # seconds
sim_steps = int(sim_time / dt)

# # Initialize storage for simulation results
# x_sim = np.zeros((nx, sim_steps+1))
# u_sim = np.zeros((nu, sim_steps))

# # Initial state
# x_init = np.array([0.75, 0.85, 5*np.pi/8])
# x_sim[:, 0] = x_init

# # Initial guess for first solve
# x_guess = np.linspace(x_init, x_ref, N + 1)

opti.set_value(X0, current_X)

sol = opti.solve()


# # MPC loop
# for t in range(0, 1): #sim_steps):
    
#     if t == 0:
#         for k in range(N + 1):
#             opti.set_initial(X[k], x_guess[k])

#     xt = x_sim[:, t]

#     rotation = np.array(
#                 [
#                     [np.cos(xt[2]), -np.sin(xt[2])],
#                     [np.sin(xt[2]), np.cos(xt[2])],
#                 ]
#             )

#     translation = np.array([[xt[0]], [xt[1]]])


#     # Set current state as initial condition
#     opti.set_value(X0, x_sim[:, t])

#     try:
#         sol = opti.solve()
#         u_sol = np.zeros((nu, N))
#         x_opt = np.zeros((nx, N+1))
#         for k in range(N):
#             u_sol[:, k] = sol.value(U[k])
#             x_opt[:, k] = sol.value(X[k])
#         x_opt[:, N] = sol.value(X[N])
#         # u_sol = sol.value(U[:, 0])
#         # xsol = sol.value(X)
#     except RuntimeError:
#         u_sol = np.zeros((nu, N))
#         x_opt = np.zeros((nx, N+1))
#         for k in range(N):
#             u_sol[:, k] = opti.debug.value(U[k])
#             x_opt[:, k] = opti.debug.value(X[k])
#         x_opt[:, N] = opti.debug.value(X[N])
#         print("Initial guess: ", x_sim[:, t])
#         print("Solver failed at step", t)
#         # print("States:")

#         break

#     # Extract optimal input
#     # u_opt = sol.value(U[:, 0])
#     u_opt = u_sol[:, 0]
#     u_sim[:, t] = u_opt

#     # Apply input and simulate system (Euler)
#     x_next = np.zeros(nx)
#     x_next[0] = x_sim[0, t] + u_opt[0] * np.cos(x_sim[2, t]) * dt
#     x_next[1] = x_sim[1, t] + u_opt[0] * np.sin(x_sim[2, t]) * dt
#     x_next[2] = x_sim[2, t] + u_opt[1] * dt
#     # wrap angle to [-pi, pi]
#     x_next[2] = (x_next[2] + np.pi) % (2*np.pi) - np.pi

#     x_sim[:, t+1] = x_next

#     # Warm start for next iteration
#     # opti.set_initial(X, sol.value(X))
#     # opti.set_initial(U, sol.value(U))
#     for k in range(N + 1):
#         opti.set_initial(X[k], x_opt[:, k])
#     for k in range(N):
#         opti.set_initial(U[k], u_sol[:, k])


u_sol = np.zeros((nu, N))
x_opt = np.zeros((nx, N+1))
for k in range(N):
    u_sol[:, k] = sol.value(U[k])
    x_opt[:, k] = sol.value(X[k])
x_opt[:, N] = sol.value(X[N])
# u_sol = sol.value(U[:, 0])
# xsol = sol.value(X)

box_vertices = rectangle_vertices_from_polytope(A_robot, b_robot)  # shape (4,2)


# Plot XY trajectory with robot box at intervals
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# plt.plot(x_sim[0, :], x_sim[1, :], '-o', label='Trajectory')
# plt.plot(x_ref[0], x_ref[1], 'rx', label='Goal')
# plt.plot(x_init[0], x_init[1], 'gx', label='Start')

# # Draw box every N steps (e.g., every 10 steps)
# for i in range(0, sim_steps+1, 10):
#     verts = transform_box(box_vertices, x_sim[0, i], x_sim[1, i], x_sim[2, i])
#     verts = np.vstack((verts, verts[0]))  # close the loop
#     plt.plot(verts[:,0], verts[:,1], 'k--', alpha=0.6)

# Draw solution
for i in range(0, N + 1):
    verts = transform_box(box_vertices, x_opt[0, i], x_opt[1, i], x_opt[2, i])
    verts = np.vstack((verts, verts[0]))  # close the loop
    plt.plot(verts[:,0], verts[:,1], 'k--', alpha=0.6)

plot_rectangle_from_normals_and_offsets(A_free, b_free, ax, color='red')

# plt.plot(obstacle_world[:,0], obstacle_world[:,1], 'r-', linewidth=2, label='Obstacle')


plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Nonlinear MPC trajectory with robot footprint')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
