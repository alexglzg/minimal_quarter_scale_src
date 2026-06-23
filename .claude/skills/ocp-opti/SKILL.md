---
name: ocp-opti
description: Use this skill when writing or modifying an Optimal Control Problem (OCP) or MPC controller defined using CasADi Opti. The preferred style is with RK4 multiple shooting. Covers the class pattern: declaring variables/parameters, adding dynamics, cost, warm-starting, and extracting solutions. Trigger on any request to add a new OCP, or modify the MPC class structure, using CasADi Opti.
---

# OCP formulation with CasADi Opti (this project's preferred style)
Smoke test: `python .claude/skills/ocp-opti/smoke.py` (7 variants, all verified working).

## Quick start

```python
import casadi as ca
import numpy as np

opti = ca.Opti()

T, N = 3.0, 20
dt = T / N

X = opti.variable(4, N + 1)   # state: [x, y, vx, vy]
U = opti.variable(2, N)        # control: [ux, uy]

x0_param = opti.parameter(4)
goal_param = opti.parameter(2)

def f(x, u):
    return ca.vertcat(x[2], x[3], u[0], u[1])

def rk4(x, u):
    k1 = f(x, u);  k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u);  k4 = f(x + dt*k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Initial condition + multiple-shooting continuity
opti.subject_to(X[:, 0] == x0_param)
for k in range(N):
    opti.subject_to(X[:, k+1] == rk4(X[:, k], U[:, k]))

# Control and velocity bounds (apply per-row — matrix inequalities don't work in Opti)
opti.subject_to(U[0, :] >= -1);  opti.subject_to(U[0, :] <= 1)
opti.subject_to(U[1, :] >= -1);  opti.subject_to(U[1, :] <= 1)
opti.subject_to(X[2, :] >= -1);  opti.subject_to(X[2, :] <= 1)
opti.subject_to(X[3, :] >= -1);  opti.subject_to(X[3, :] <= 1)

# Cost
obj = 0
for k in range(N):
    obj += U[0,k]**2 + U[1,k]**2
    obj += 10*((X[0,k]-goal_param[0])**2 + (X[1,k]-goal_param[1])**2)
obj += 1000*((X[0,N]-goal_param[0])**2 + (X[1,N]-goal_param[1])**2)
opti.minimize(obj)

opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False})
```

## MPC solve loop

```python
opti.set_value(x0_param, current_state)

sol = opti.solve()

X_sol = sol.value(X)   # (4, N+1)
U_sol = sol.value(U)   # (2, N)

ux_applied = float(U_sol[0, 0])
uy_applied = float(U_sol[1, 0])

# Warm-start next solve from current solution
opti.set_initial(X, X_sol)
opti.set_initial(U, U_sol)
```

Always warm-start. Without it IPOPT starts cold and takes 3–5× longer.

## Solver options

```python
# Standard
opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False})

# Faster: expand=True converts MX → SX before solving, reducing evaluation overhead.
# Do NOT use when the problem contains a LUT (e.g. a spline interpolation) — it breaks.
opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': False, 'expand': True})

# Fastest for structured MPC: FATROP (structure-exploiting NLP solver).
# Can fail where IPOPT wouldn't — always have a fallback or check the return status.
opts = {
    "fatrop.print_level": 0,
    "print_time": 1,
    "structure_detection": "auto",
    "expand": True,
    "debug": True,   # on structure-detection failure, writes debug_fatrop_expected.mtx
                      # / debug_fatrop_actual.mtx for inspection — keep this on while
                      # restructuring a problem for FATROP, off once it's stable.
}
opti.solver('fatrop', opts)
```

### FATROP structure requirements

`structure_detection="auto"` recovers the banded per-stage structure by scanning
**declaration order** — both for variables and for constraints. Break any of the
rules below and you get the same cryptic error (`Fatrop: specified structure of A
does not correspond to what the interface can handle`, with an `nx`/`nu`/`ng`
listing full of spurious `0`s, i.e. phantom zero-size stages):

1. **Per stage, declare all state-like (recursively-defined) variables contiguously,
   then all control-like (free) variables contiguously** — `(x0, u0), (x1, u1), ...,
   (xN)` overall. This matters even when there's more than one variable of each
   kind: if an auxiliary variable has its own cross-stage recursion (e.g. an
   adaptive gain `a` driven by a virtual control `v`, `a[:,k+1] == a[:,k] + dt*v[:,k]`),
   it must be declared *next to* the main state, not interleaved after the main
   control:
   ```python
   for k in range(N):
       Xk = opti.variable(nx); X.append(Xk)
       Ak = opti.variable(na); A.append(Ak)      # OK: state-like, grouped with X

       Uk = opti.variable(nu); U.append(Uk)
       Vk = opti.variable(na); V.append(Vk)      # OK: control-like, grouped with U
   # NOT: Xk, Uk, Ak, Vk per iteration -- interleaving state/control groups
   # silently breaks structure detection even though nothing else changes.
   ```
   A variable with *no* recursion (just bounded per stage, no carry-over) can go in
   either group — it has no defining equality to be misclassified.

2. **Add constraints per stage, interleaved in one loop** — dynamics, control
   bounds, and path constraints for stage `k` must all be added before stage
   `k+1`'s, not grouped into separate per-type or per-obstacle loops:
   ```python
   for k in range(N):
       xnext = rk4(X[:, k], U[:, k])
       opti.subject_to(X[:, k+1] == xnext)        # gap-closing equality
       opti.subject_to(-1 <= (U[:, k] <= 1))       # bounds for stage k
       opti.subject_to(h(X[:, k]) >= 0)            # path constraint for stage k
   ```

3. **Never reference the `X[:, k+1]` variable inside stage `k`'s block except in the
   gap-closing equality itself.** For any other constraint that needs the "next"
   state (e.g. a discrete-CBF decrease condition `h(x_{k+1}) - h(x_k) + ... >= 0`),
   use the explicit dynamics expression instead — it's equal to `X[:, k+1]` at the
   solution (the gap equality enforces that), but referencing the *variable* from
   within stage `k` confuses structure detection into seeing a stray stage boundary:
   ```python
   xnext = rk4(X[:, k], U[:, k])
   opti.subject_to(X[:, k+1] == xnext)             # OK: defines the gap
   opti.subject_to(h(xnext) - h(X[:, k]) >= 0)      # OK: explicit expression
   # opti.subject_to(h(X[:, k+1]) - h(X[:, k]) >= 0)   # BREAKS structure detection
   ```

4. **Every stage variable with its own cross-stage recursion must be defined at
   every transition, including the last one** — e.g. if an auxiliary integrator
   state's update loop stops one short of `N`, leaving it free/undefined at the
   terminal stage, structure detection fragments.

## Reducing per-call overhead with `opti.to_function()`

If the OCP is called repeatedly in a loop, wrapping it as a CasADi `Function` avoids
Python-level overhead on every call:

```python
# Build the function once after define_problem()
# Include X, U as inputs so the caller can pass a warm-start,
# and as outputs so the solution can be fed back next call.
F = opti.to_function(
    'mpc',
    [x0_param, goal_param, X, U],        # params + previous solution as warm-start
    [U[:, 0], X, U],                      # first control + full solution to reuse
    ['x0', 'goal', 'X0', 'U0'],
    ['u0', 'X_sol', 'U_sol'],
)

# Initialise warm-start to zeros for the first call
X_warm = np.zeros((4, N + 1))
U_warm = np.zeros((2, N))

# Each MPC step
result  = F(x0=current_state, goal=goal, X0=X_warm, U0=U_warm)
u_applied = np.array(result['u0']).flatten()
X_warm  = np.array(result['X_sol'])   # feed solution back as next warm-start
U_warm  = np.array(result['U_sol'])
```

**`error_on_fail` must be set to `True` in the solver options** when using `to_function`,
otherwise a failed solve returns NaN silently instead of raising:

```python
opti.solver('ipopt', {
    'ipopt.print_level': 0,
    'print_time': False,
    'error_on_fail': True,   # required with to_function
})
```

## Parameters: structural vs. simulation-varying

`define_problem()` fixes the NLP **structure** — variable counts, constraint topology, cost weights. It must not embed values that change during simulation.

**Rule:** if a quantity can change between MPC steps, declare it as `opti.parameter(...)` in `define_problem()` and call `opti.set_value()` in `solve()`, never in `define_problem()`.

| Belongs in `define_problem()` | Belongs in `solve()` |
|---|---|
| Number of obstacles `n_obs` | Obstacle positions |
| Horizon `N`, timestep `dt` | Current waypoint / goal |
| Safety radii, cost weights | Initial state `x0` |
| constraint type, `adaptive` flag | constraint parameters |
| `opti.parameter(n_obs)` declarations | `opti.set_value(obs_x_param, ...)` |

```python
# define_problem — structural only
def define_problem(self, n_obs, obstacle_radius=0.5, cbf_type=None, ...):
    obs_x_param = opti.parameter(n_obs)   # declare shape, do NOT set_value here
    obs_y_param = opti.parameter(n_obs)
    goal_param  = opti.parameter(2)
    ...

# solve — set every varying quantity before opti.solve()
def solve(self, initial_state, waypoint, obstacles, u_ref, alpha1, ...):
    opti.set_value(x0_param,    initial_state)
    opti.set_value(goal_param,  waypoint[:2])
    opti.set_value(obs_x_param, [o[0] for o in obstacles])
    opti.set_value(obs_y_param, [o[1] for o in obstacles])
    opti.set_value(uref_param,  u_ref)
    ...
    sol = opti.solve()
```

## What NOT to do

- Do not use `rockit` / `Ocp()` for new controllers.
- Do not rebuild the `Opti()` object each MPC iteration. Build once in `define_problem()`, then only call `set_value()` + `solve()` in the loop.
- Do not call `opti.solve()` without setting `x0_param` first — IPOPT will silently optimize from zero initial state.
- Do not use `opti.set_initial()` before the first solve; leave the default zero initial guess for the cold start.
- **Do not call `opti.set_value()` inside `define_problem()` for simulation-varying quantities.** Only declare the parameter shape there; set actual values in `solve()`.
- **Do not impose path constraints at `k=0`.** The initial state is already fully determined by `X[:, 0] == x0_param`. Adding a path constraint there (e.g. `h(X[:, 0]) >= 0`) makes the problem infeasible whenever the current state doesn't satisfy it — which can happen legitimately mid-trajectory. Start path constraints at `k=1`:
  ```python
  for k in range(1, N):   # not range(N)
      opti.subject_to(h(X[:, k]) >= 0)
  ```
