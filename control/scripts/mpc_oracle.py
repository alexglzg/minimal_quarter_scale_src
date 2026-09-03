"""The NMPC oracle: the controller the amortized model was trained to imitate.

This is a **self-contained, readable copy** of the CasADi/IPOPT nonlinear MPC that
generated the training data for `model.eqx`. It has no dependency on the research
pipeline -- only numpy and casadi (`pip install casadi`).

Run it standalone to inspect the problem it solves:

    python mpc_oracle.py

That prints the NLP's size and structure, solves one step, and (with matplotlib)
saves `oracle_prediction.png` showing the predicted horizon.

Why you would read this file
---------------------------
The learned model replaces this entire optimization with one MLP forward pass plus a
4-variable QP. Understanding what it is *approximating* explains both its strengths
and its limits: the objective below is what the network learned to encode into (H, c),
while the HOCBF constraint is the one piece the learned controller keeps solving
exactly rather than approximating.

The optimization
----------------
Decision variables (multiple shooting over a horizon of N steps):

    X : (7, N+1)   predicted state trajectory
    U : (4, N)     predicted thruster forces

minimize     sum_k [ stage_cost(X_k) + r_f * ||U_k||^2 ] + stage_cost(X_N)
subject to   X_0     = x0                                  (initial condition)
             X_{k+1} = RK4(X_k, U_k, dt)                    (dynamics)
             -f_max <= U_k <= f_max                         (thruster box)
             HOCBF_i(X_k, U_k) >= 0   for each obstacle i   (safety)

Runtime **parameters** (set per solve, no rebuild): the current state `x0`, the
obstacle centres, and the per-obstacle class-K gains `(gamma1, gamma2)`. Making the
gains parameters is what lets the same NLP be driven by network-predicted gains --
which is exactly what the `ana_i64` model learns to supply.

The cost
--------
Path-following against the sinusoid `x_d(s) = kx*s`, `y_d(s) = sin(kx*s) + y0`:

    stage_cost = q_ye  * ye^2                       cross-track error
               + q_r   * r^2                        yaw-rate damping
               + q_psi * |[sin psi, cos psi] - [sin gp, cos gp]|^2   heading alignment
               + q_u   * (u - u_ref)^2              surge-speed tracking

where `gp` is the path tangent angle and `ye` is the position error projected onto the
path normal. The heading term is written with sin/cos rather than an angle difference
so it is smooth and free of wrapping discontinuities.

The safety constraint (HOCBF)
-----------------------------
For each circular obstacle, with `r_safe = radius + r_ego`:

    h    = ||p - c||^2 - r_safe^2            (>0 outside the obstacle)
    h_d  = dh/dx . f(x, u)                   (no u: h depends on position only)
    h_dd = dh_d/dx . f(x, u)                 (affine in u -- this is where u enters)

    HOCBF:   h_dd + (g1 + g2) h_d + g1 g2 h >= 0

Second order is required because the vessel is force-actuated: position cannot change
instantaneously, so `h_d` carries no input and a first-order CBF would have no control
authority. This is the identical constraint the learned controller builds by autodiff
in `anmpc_alpha._cbf_single` -- compare the two to see the correspondence.

Note on duals
-------------
The research pipeline also extracts the NLP multipliers here (for a hybrid KKT loss).
That machinery is omitted: it is irrelevant to running the oracle or comparing against
it, and the shipped model was trained with plain supervised MSE.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import casadi as ca

NX = 7   # [X, Y, psi, u, v, r, s]
NU = 4   # [f1, f2, f3, f4]


# ===========================================================================
# Parameters -- these values ARE the ones the shipped model was trained with
# ===========================================================================
@dataclass
class VesselParams:
    """Identified quarter-scale Fossen vessel parameters."""
    m11: float = 12.0            # surge mass (incl. added mass)
    m22: float = 24.0            # sway mass
    m33: float = 3.0             # yaw inertia
    d11: float = 6.0             # surge linear drag
    d22: float = 8.0             # sway linear drag
    d33: float = 0.6             # yaw linear drag
    a:   float = 0.45            # surge thruster moment arm
    b:   float = 0.90            # sway thruster moment arm
    f_max: float = 6.0           # symmetric thruster force limit [N]
    length: float = 0.90         # bounding box -> r_ego
    width:  float = 0.45

    @property
    def r_ego(self) -> float:
        return float(np.hypot(self.length, self.width) / 2.0)


@dataclass
class PathParams:
    """Sinusoidal reference path  x_d = kx*s,  y_d = sin(kx*s) + y0."""
    kx: float = 0.2
    y0: float = 3.0
    u_ref: float = 0.6           # desired surge speed [m/s]

    def xy(self, s):
        sin = ca.sin if isinstance(s, (ca.SX, ca.MX, ca.DM)) else np.sin
        return self.kx * s, sin(self.kx * s) + self.y0


@dataclass
class CostWeights:
    """Path-following objective weights (identical to model_meta.yaml)."""
    q_ye:  float = 20.0
    q_r:   float = 0.1
    q_psi: float = 10.0
    q_u:   float = 200.0
    r_f:   float = 1.0


@dataclass
class Obstacle:
    cx: float
    cy: float
    radius: float


@dataclass
class MPCSolution:
    x_traj: np.ndarray           # (N+1, NX) predicted states
    u_traj: np.ndarray           # (N, NU)   predicted inputs
    u_opt:  np.ndarray           # (NU,)     first input -- the command applied
    solve_time_s: float
    status: str                  # "optimal" | "failed"
    obj_value: float


# ===========================================================================
# Dynamics -- one definition, evaluated with either numpy or casadi
# ===========================================================================
def vessel_ode(x, u, p: VesselParams, sin, cos):
    """Continuous-time Fossen dynamics. Returns the list of NX derivatives.

    eta_dot = R(psi) nu
    nu_dot  = M^-1 ( B f - C(nu) nu - D nu )
    s_dot   = u                     (arc length advances with surge speed)
    """
    X, Y, psi, su, sv, sr, s = (x[i] for i in range(NX))
    f1, f2, f3, f4 = (u[i] for i in range(NU))

    # Thruster allocation tau = B f
    tau1 = f1 + f2                                        # surge force
    tau2 = f3 + f4                                        # sway force
    tau3 = p.a / 2 * (f1 - f2) + p.b / 2 * (f3 - f4)      # yaw moment

    # Kinematics
    X_dot = su * cos(psi) - sv * sin(psi)
    Y_dot = su * sin(psi) + sv * cos(psi)
    psi_dot = sr

    # Rigid-body dynamics; the Coriolis terms C(nu) nu are the products below.
    u_dot = (tau1 + p.m22 * sv * sr - p.d11 * su) / p.m11
    v_dot = (tau2 - p.m11 * su * sr - p.d22 * sv) / p.m22
    r_dot = (tau3 - (p.m22 - p.m11) * su * sv - p.d33 * sr) / p.m33

    return [X_dot, Y_dot, psi_dot, u_dot, v_dot, r_dot, su]


def vessel_ode_ca(x, u, p):
    return ca.vertcat(*vessel_ode(x, u, p, ca.sin, ca.cos))


def vessel_ode_np(x, u, p):
    return np.array(vessel_ode(x, u, p, np.sin, np.cos))


def rk4(f, x, u, dt):
    """One explicit RK4 step of x_dot = f(x, u) -- backend agnostic."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ===========================================================================
# The NMPC controller
# ===========================================================================
class VesselNMPC:
    """Path-following NMPC with HOCBF obstacle avoidance (CasADi + IPOPT).

    Built once, then re-solved every control step via ``__call__``. Obstacle
    centres and CBF gains are CasADi parameters, so a solve never rebuilds the NLP.

    Not picklable (CasADi Opti holds C++ pointers) -- build one per process.
    """

    def __init__(self, obstacles: Sequence[Obstacle],
                 params: VesselParams = None, path: PathParams = None,
                 weights: CostWeights = None, N: int = 40, dt: float = 0.1,
                 default_gamma=(0.5, 0.5), solver_opts: dict = None):
        self.obstacles = list(obstacles)
        self.p = params or VesselParams()
        self.path = path or PathParams()
        self.w = weights or CostWeights()
        self.N = int(N)
        self.dt = float(dt)
        self.n_obs = len(self.obstacles)
        self.default_gamma = np.asarray(default_gamma, float)
        self._x_ws = None                    # warm-start state guess
        self._u_ws = None                    # warm-start input guess
        self._build(solver_opts)

    # ------------------------------------------------------------------
    def _stage_cost(self, xk):
        """Path-following running cost at one stage (state part only)."""
        p, w = self.path, self.w
        X, Y, psi, su, sv, sr, s = (xk[i] for i in range(NX))

        x_d, y_d = p.xy(s)
        # Path tangent angle gp = atan2(dy_d/ds, dx_d/ds)
        gp = ca.atan2(p.kx * ca.cos(p.kx * s), p.kx)
        # Cross-track error: position error projected onto the path normal
        ye = -(X - x_d) * ca.sin(gp) + (Y - y_d) * ca.cos(gp)

        return (w.q_ye * ye ** 2
                + w.q_r * sr ** 2
                + w.q_psi * ((ca.sin(psi) - ca.sin(gp)) ** 2
                             + (ca.cos(psi) - ca.cos(gp)) ** 2)
                + w.q_u * (su - p.u_ref) ** 2)

    # ------------------------------------------------------------------
    def _make_cbf_fun(self) -> ca.Function:
        """HOCBF value as a reusable CasADi Function of (x, u, centre, gains, r_safe).

        The Lie derivatives are taken on pure SX symbols here because `jtimes`
        cannot differentiate through Opti's MX decision variables; the resulting
        Function is then evaluated on those variables in `_build`.
        """
        xs = ca.SX.sym("x", NX)
        us = ca.SX.sym("u", NU)
        c = ca.SX.sym("c", 2)
        g = ca.SX.sym("g", 2)
        r_safe = ca.SX.sym("r_safe")

        h = (xs[0] - c[0]) ** 2 + (xs[1] - c[1]) ** 2 - r_safe ** 2
        fs = vessel_ode_ca(xs, us, self.p)
        h_d = ca.jtimes(h, xs, fs)          # no u (h is position-only)
        h_dd = ca.jtimes(h_d, xs, fs)       # affine in u
        cbf = h_dd + (g[0] + g[1]) * h_d + g[0] * g[1] * h
        return ca.Function("cbf", [xs, us, c, g, r_safe], [cbf])

    # ------------------------------------------------------------------
    def _build(self, solver_opts):
        opti = ca.Opti()
        N, dt = self.N, self.dt

        # ---- decision variables ---------------------------------------
        X = opti.variable(NX, N + 1)
        U = opti.variable(NU, N)

        # ---- runtime parameters (set per solve, never rebuilt) --------
        x0 = opti.parameter(NX)
        obs_c = opti.parameter(2, max(self.n_obs, 1))
        obs_r = opti.parameter(1, max(self.n_obs, 1))
        gam = opti.parameter(2, max(self.n_obs, 1))

        # ---- objective ------------------------------------------------
        cost = 0
        for k in range(N):
            cost += self._stage_cost(X[:, k])
            cost += self.w.r_f * ca.sumsqr(U[:, k])
        cost += self._stage_cost(X[:, N])                 # terminal (same form)
        opti.minimize(cost)

        # ---- dynamics: multiple shooting with RK4 ---------------------
        f = lambda xx, uu: vessel_ode_ca(xx, uu, self.p)
        n_dyn = 0
        for k in range(N):
            opti.subject_to(X[:, k + 1] == rk4(f, X[:, k], U[:, k], dt))
            n_dyn += NX

        # ---- initial condition ----------------------------------------
        opti.subject_to(X[:, 0] == x0)

        # ---- thruster box ---------------------------------------------
        n_box = 0
        for k in range(N):
            opti.subject_to(U[:, k] + self.p.f_max >= 0)
            opti.subject_to(self.p.f_max - U[:, k] >= 0)
            n_box += 2 * NU

        # ---- HOCBF safety constraints ---------------------------------
        # r_safe is built from the obs_r *parameter* (not baked in from
        # self.obstacles) so a real obstacle's radius -- e.g. a detected buoy's,
        # which varies shot to shot -- can be set per solve like the centre can.
        cbf_fun = self._make_cbf_fun()
        n_cbf = 0
        for i in range(self.n_obs):
            r_safe = obs_r[0, i] + self.p.r_ego
            for k in range(N):
                opti.subject_to(
                    cbf_fun(X[:, k], U[:, k], obs_c[:, i], gam[:, i], r_safe) >= 0)
                n_cbf += 1

        # ---- solver ---------------------------------------------------
        opts = {
            "expand": True,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 500,
            "ipopt.tol": 1e-6,
            "ipopt.acceptable_tol": 1e-4,
            # Required for the compiled Function below to raise on a failed solve
            # instead of silently handing back a non-converged iterate.
            "error_on_fail": True,
        }
        if solver_opts:
            opts.update(solver_opts)
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X, self._U = X, U
        self._x0_p, self._obs_c_p, self._obs_r_p, self._gam_p = x0, obs_c, obs_r, gam
        self.sizes = dict(n_var=NX * (N + 1) + NU * N, n_dyn=n_dyn,
                          n_ic=NX, n_box=n_box, n_cbf=n_cbf)

        # ---- compile the whole solve to one CasADi Function -----------
        # opti.to_function bakes parameter-setting, warm-start and the IPOPT call into a single
        # compiled call, which is what makes repeated re-solves fast in the ROS
        # node -- opti.solve() re-traces/interprets each call, to_function does not.
        # Opti requires values for every parameter (and an initial guess for every
        # variable) before it can trace to_function; these are placeholders,
        # overridden with real values on every call in __call__.
        opti.set_value(x0, np.zeros(NX))
        opti.set_value(obs_c, np.zeros((2, max(self.n_obs, 1))))
        opti.set_value(obs_r, np.ones((1, max(self.n_obs, 1))))
        opti.set_value(gam, np.tile(self.default_gamma, (max(self.n_obs, 1), 1)).T)
        opti.set_initial(X, np.zeros((NX, N + 1)))
        opti.set_initial(U, np.zeros((NU, N)))

        self.solve_func = opti.to_function(
            "oracle_ocp_func",
            [x0, obs_c, obs_r, gam, X, U],
            [U, X, opti.f],
            ["x0", "obs_c", "obs_r", "gam", "X_init", "U_init"],
            ["U", "X", "obj"],
        )

    # ------------------------------------------------------------------
    def reset_warm_start(self):
        self._x_ws = self._u_ws = None

    def __call__(self, x0, gammas=None, obstacle_centers=None,
                 obstacle_radii=None) -> MPCSolution:
        """Solve the NMPC from state `x0`; returns the full predicted horizon.

        gammas           : (n_obs, 2) class-K gains, default `default_gamma`.
        obstacle_centers : (n_obs, 2) centres, default the ones given at build.
        obstacle_radii   : (n_obs,)   radii, default the ones given at build.

        Runs the `solve_func` CasADi Function compiled in `_build` (one IPOPT
        call, no Python-side Opti bookkeeping per solve) instead of opti.solve().
        """
        x0 = np.asarray(x0, float).reshape(NX)

        if obstacle_centers is None:
            centers = np.array([[o.cx, o.cy] for o in self.obstacles], float)
        else:
            centers = np.asarray(obstacle_centers, float).reshape(self.n_obs, 2)
        obs_c_val = centers.T if self.n_obs else np.zeros((2, 1))

        if obstacle_radii is None:
            radii = np.array([o.radius for o in self.obstacles], float)
        else:
            radii = np.asarray(obstacle_radii, float).reshape(self.n_obs)
        obs_r_val = radii.reshape(1, -1) if self.n_obs else np.ones((1, 1))

        if gammas is None:
            g = np.tile(self.default_gamma, (max(self.n_obs, 1), 1))
        else:
            g = np.asarray(gammas, float).reshape(self.n_obs, 2)
        gam_val = g.T

        # Warm start: shift the previous solution one step forward. This is what
        # makes sequential solves fast; a cold start costs several times more.
        if self._x_ws is not None:
            X_ws = np.hstack([self._x_ws[:, 1:], self._x_ws[:, -1:]])
            U_ws = np.hstack([self._u_ws[:, 1:], self._u_ws[:, -1:]])
        else:
            X_ws = np.tile(x0[:, None], (1, self.N + 1))
            U_ws = np.zeros((NU, self.N))

        t0 = time.perf_counter()
        try:
            U_sol, X_sol, obj = self.solve_func(x0, obs_c_val, obs_r_val, gam_val, X_ws, U_ws)
            status = "optimal"
            x_sol = np.atleast_2d(np.array(X_sol))
            u_sol = np.atleast_2d(np.array(U_sol))
            obj_value = float(obj)
            self._x_ws, self._u_ws = x_sol, u_sol
        except RuntimeError:
            # IPOPT failed to converge (error_on_fail raises rather than handing
            # back a non-converged iterate): keep the previous warm start so the
            # closed loop can retry next step instead of restarting cold.
            status = "failed"
            x_sol, u_sol, obj_value = X_ws, U_ws, float("nan")
        solve_time = time.perf_counter() - t0

        return MPCSolution(x_traj=x_sol.T, u_traj=u_sol.T, u_opt=u_sol[:, 0].copy(),
                           solve_time_s=solve_time, status=status, obj_value=obj_value)


# ===========================================================================
# True-plant simulator and helpers
# ===========================================================================
class VesselSimulator:
    """Integrates the true (full nonlinear) plant with RK4 -- the 'real vessel'.

    Substeps internally at `sub_step` (default 0.0001s, matching
    quarterscalesimulation's Sim.cpp `simStep` -- see qs_sim_node, the ROS
    node that actually simulates the plant in the live integration) rather
    than taking one RK4 step of the full `dt`: a single 0.1s RK4 step is
    coarse enough that closed-loop trajectories measurably diverge from the
    real ROS-simulated plant, most visibly during obstacle-avoidance
    maneuvers where the controller's output is most sensitive to small state
    differences. The command `u` is held constant (zero-order hold) across
    all substeps of one control tick, same as the real plant only receiving
    a new /mpc_force at the control rate.
    """

    def __init__(self, params: VesselParams = None, dt: float = 0.1, sub_step: float = 0.0001):
        self.p = params or VesselParams()
        self.dt = float(dt)
        self.sub_step = float(sub_step)

    def step(self, x, u):
        x = np.asarray(x, float).reshape(NX)
        u = np.clip(np.asarray(u, float).reshape(NU), -self.p.f_max, self.p.f_max)
        n_sub = max(1, round(self.dt / self.sub_step))
        h = self.dt / n_sub
        f = lambda xx, uu: vessel_ode_np(xx, uu, self.p)
        for _ in range(n_sub):
            x = rk4(f, x, u, h)
        x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi     # wrap heading
        return x


def project_arc_length(path: PathParams, x, y, s_center=0.0,
                       look_back=1.0, look_ahead=10.0, n=2000):
    """Nearest-point projection of (x, y) onto the path, searched locally.

    Matches `anmpc_alpha.anchor_arc_length` so the oracle and the learned
    controller are anchored identically when compared.
    """
    s_grid = np.linspace(max(0.0, s_center - look_back), s_center + look_ahead, n)
    x_d, y_d = path.xy(s_grid)
    return float(s_grid[int(np.argmin((x - x_d) ** 2 + (y - y_d) ** 2))])


# ===========================================================================
# Standalone inspection
# ===========================================================================
def _demo():
    kx, y0, u_ref = 0.2, 3.0, 0.6
    path = PathParams(kx=kx, y0=y0, u_ref=u_ref)
    p = VesselParams()

    # Same scenario as run_vessel_demo.py: three obstacles on the path, 5 m apart.
    obstacles = [Obstacle(X, float(np.sin(X) + y0 + lat), r)
                 for X, lat, r in [(6.0, +0.2, 0.5), (11.0, -0.2, 0.5), (16.0, +0.2, 0.5)]]

    print("Building the NMPC (this compiles CasADi expressions -- a few seconds)...")
    t0 = time.perf_counter()
    mpc = VesselNMPC(obstacles, params=p, path=path, N=40, dt=0.1,
                     default_gamma=(0.5, 0.5))
    print(f"  built in {time.perf_counter() - t0:.2f} s\n")

    s = mpc.sizes
    print(f"NLP structure (N = {mpc.N}, n_obs = {mpc.n_obs}):")
    print(f"  decision variables      : {s['n_var']:5d}   "
          f"(7*(N+1) states + 4*N inputs)")
    print(f"  dynamics equalities     : {s['n_dyn']:5d}   (7 per step)")
    print(f"  initial-condition rows  : {s['n_ic']:5d}")
    print(f"  thruster box rows       : {s['n_box']:5d}   (2*4 per step)")
    print(f"  HOCBF safety rows       : {s['n_cbf']:5d}   (1 per obstacle per step)")
    print(f"  r_safe per obstacle     : "
          f"{[round(o.radius + p.r_ego, 3) for o in obstacles]}")

    psi0 = np.arctan2(np.cos(0.0), 1.0)          # path tangent at X=0 -> 45 deg
    x0 = np.array([0.0, y0, psi0, 0.3, 0.0, 0.0, 0.0])
    x0[6] = project_arc_length(path, x0[0], x0[1], 0.0)

    print("\nSolving one step (cold start, no warm start available)...")
    sol = mpc(x0)
    print(f"  status      : {sol.status}")
    print(f"  solve time  : {sol.solve_time_s * 1e3:.1f} ms")
    print(f"  objective   : {sol.obj_value:.4f}")
    print(f"  u_opt [N]   : [{', '.join(f'{v: .3f}' for v in sol.u_opt)}]")
    print(f"  horizon     : predicts {sol.x_traj.shape[0]} states, "
          f"{sol.u_traj.shape[0]} inputs ({mpc.N * mpc.dt:.1f} s ahead)")

    print("\nA second solve, warm-started from the first:")
    sol2 = mpc(x0)
    print(f"  solve time  : {sol2.solve_time_s * 1e3:.1f} ms "
          f"({sol.solve_time_s / max(sol2.solve_time_s, 1e-9):.1f}x faster)")
    print("\nThe learned controller replaces all of the above with one MLP pass plus a")
    print("4-variable QP -- see compare_to_oracle.py for the side-by-side.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    s_grid = np.linspace(0, 120, 400)
    ax.plot(*path.xy(s_grid), "k--", lw=1, label="reference path")
    ax.plot(sol.x_traj[:, 0], sol.x_traj[:, 1], "-o", color="tab:purple", ms=3,
            lw=1.5, label=f"predicted horizon (N={mpc.N})")
    ax.plot(x0[0], x0[1], "go", ms=9, label="current state")
    for o in obstacles:
        ax.add_patch(plt.Circle((o.cx, o.cy), o.radius, color="tab:red", alpha=0.35))
        ax.add_patch(plt.Circle((o.cx, o.cy), o.radius + p.r_ego, color="tab:red",
                                alpha=0.5, ls="--", lw=0.8, fill=False))
    ax.set_aspect("equal"); ax.set_xlim(-1, 12); ax.grid(alpha=0.3)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("NMPC oracle: one solve, the whole predicted horizon")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    import os
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots", "oracle")
    os.makedirs(plots_dir, exist_ok=True)
    out = os.path.join(plots_dir, "oracle_prediction.png")
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    _demo()
