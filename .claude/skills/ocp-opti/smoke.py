#!/usr/bin/env python3
"""
Smoke test for the generic Opti-based OCP pattern documented in SKILL.md.
Run from the repo root:  python .claude/skills/ocp-opti/smoke.py
"""
import sys
import numpy as np
sys.path.insert(0, '.')
import casadi as ca

T, N = 3.0, 20
dt = T / N

# ── 1. Build OCP (done once) ───────────────────────────────────────────────
opti = ca.Opti()

X = opti.variable(4, N + 1)
U = opti.variable(2, N)

x0_param   = opti.parameter(4)
goal_param = opti.parameter(2)

def f(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])

def rk4(x, u):
    k1 = f(x, u);  k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u);  k4 = f(x + dt*k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

opti.subject_to(X[:, 0] == x0_param)
for k in range(N):
    opti.subject_to(X[:, k+1] == rk4(X[:, k], U[:, k]))

opti.subject_to(U[0, :] >= -1);  opti.subject_to(U[0, :] <= 1)
opti.subject_to(U[1, :] >= -1);  opti.subject_to(U[1, :] <= 1)
opti.subject_to(X[2, :] >= -1);  opti.subject_to(X[2, :] <= 1)
opti.subject_to(X[3, :] >= -1);  opti.subject_to(X[3, :] <= 1)

obj = 0
for k in range(N):
    obj += U[0,k]**2 + U[1,k]**2
    obj += 10*((X[0,k]-goal_param[0])**2 + (X[1,k]-goal_param[1])**2)
obj += 1000*((X[0,N]-goal_param[0])**2 + (X[1,N]-goal_param[1])**2)
opti.minimize(obj)

opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False})

# ── 2. MPC loop (3 steps) ─────────────────────────────────────────────────
state  = np.array([0.0, 0.0, 0.0, 0.0])
goal   = np.array([5.0, 0.0])
opti.set_value(goal_param, goal)

for step in range(3):
    opti.set_value(x0_param, state)
    sol   = opti.solve()
    X_sol = sol.value(X)
    U_sol = sol.value(U)

    ux = float(U_sol[0, 0])
    uy = float(U_sol[1, 0])
    print(f"[OK] step {step}: u=({ux:.4f}, {uy:.4f})  x=({state[0]:.3f}, {state[1]:.3f})")

    # Warm-start next solve
    opti.set_initial(X, X_sol)
    opti.set_initial(U, U_sol)

    # Simulate one step with RK4
    k1 = np.array([state[2], state[3], ux, uy])
    k2 = np.array([state[2]+dt/2*k1[2], state[3]+dt/2*k1[3], ux, uy])
    k3 = k2.copy()
    k4 = np.array([state[2]+dt*k3[2], state[3]+dt*k3[3], ux, uy])
    state = state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

print("\nAll checks passed.")
