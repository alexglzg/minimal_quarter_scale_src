"""Self-contained loader + architecture for the Amortized-NMPC-alpha vessel model.

This single module reproduces exactly the pieces of the training pipeline needed to
*load and run* the trained `ana_i64` model (class :class:`AmortizedNMPCAlpha`) for the
2-D surface-vessel obstacle-avoidance task. It has NO dependency on the research
pipeline it was extracted from — only third-party packages (jax, equinox, qpax, numpy).

Model in one sentence: an MLP predicts a *residual* on a Gauss-Newton analytic
(H, c) quadratic-program objective plus per-obstacle high-order-CBF class-K gains
(gamma1, gamma2); a small QP is then solved to produce a provably safe thruster command.

Conventions
-----------
State   x     : (7,)  [X, Y, psi, su, sv, sr, s]
                 X, Y      world position [m]
                 psi       heading [rad]
                 su,sv,sr  body-frame surge/sway velocities and yaw rate
                 s         path arc-length parameter
Control u     : (4,)  thruster forces [f1, f2, f3, f4], each |f_i| <= f_max
Path    path  : (3,)  [kx, y0, u_ref]   sinusoidal ref  x_d = kx*s, y_d = sin(kx*s)+y0
Obstacles     : obs_centers (n_obs, 2), obs_radii (n_obs,)  -- circular

Main entry point
----------------
    model, dtype = load_alpha_model("model.eqx", n_obs=6)
    u_safe, u_nom, gammas = model(x, path, obs_centers, obs_radii)   # jax arrays

`u_safe` (4,) is the safe thruster command; advance the true plant with the numpy
helpers `rk4_step_numpy` / `wrap_angle` / `anchor_arc_length` (see run_vessel_demo.py).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass
from functools import partial

import numpy as np
import yaml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

import qpax

# ---------------------------------------------------------------------------
# Vessel parameters (identified quarter-scale Fossen model) and dimensions.
# Values are the VesselParams / CostWeights defaults from the original mpc_vessel.py.
# ---------------------------------------------------------------------------
NX, NU = 7, 4

_M11, _M22, _M33 = 12.0, 24.0, 3.0     # generalized mass
_D11, _D22, _D33 = 6.0, 8.0, 0.6       # linear drag
_AA, _BB = 0.45, 0.90                  # thruster moment arms (surge / sway)
_F_MAX = 6.0                           # symmetric thruster force limit [N]
_LENGTH, _WIDTH = 0.90, 0.45           # ego bounding box -> r_ego = hypot(L,W)/2

# Default path-following cost weights (q_ye, q_r, q_psi, q_u) and effort r_f.
_DEFAULT_WEIGHTS = (20.0, 0.1, 10.0, 200.0)
_DEFAULT_RF = 1.0


# ---------------------------------------------------------------------------
# Environment geometry
# ---------------------------------------------------------------------------
@dataclass
class Env:
    """Minimal geometry container the model skeleton needs."""
    dt:          float
    f_max:       float
    n_obs:       int
    r_ego:       float

    @property
    def control_dim(self) -> int:
        return NU

    @property
    def nx(self) -> int:
        return NX

    @property
    def nu(self) -> int:
        return NU


def make_env(n_obs, dt=0.1, f_max=_F_MAX, length=_LENGTH, width=_WIDTH, r_ego=None):
    if r_ego is None:
        r_ego = float(np.hypot(length, width) / 2.0)
    return Env(dt=float(dt), f_max=float(f_max), n_obs=int(n_obs), r_ego=float(r_ego))


# ---------------------------------------------------------------------------
# Vessel dynamics (JAX) and per-sample linearization
# ---------------------------------------------------------------------------
def f_cont_jax(x, u):
    """Continuous-time vessel dynamics (Fossen). x=(7,), u=(4,) -> dx/dt (7,)."""
    X, Y, psi, su, sv, sr, s = x
    f1, f2, f3, f4 = u

    tau1 = f1 + f2
    tau2 = f3 + f4
    tau3 = _AA / 2.0 * (f1 - f2) + _BB / 2.0 * (f3 - f4)

    cps, sps = jnp.cos(psi), jnp.sin(psi)
    X_dot = su * cps - sv * sps
    Y_dot = su * sps + sv * cps
    psi_dot = sr

    u_dot = (tau1 + _M22 * sv * sr - _D11 * su) / _M11
    v_dot = (tau2 - _M11 * su * sr - _D22 * sv) / _M22
    r_dot = (tau3 - (_M22 - _M11) * su * sv - _D33 * sr) / _M33

    s_dot = su
    return jnp.stack([X_dot, Y_dot, psi_dot, u_dot, v_dot, r_dot, s_dot])


def rk4_step_jax(x, u, dt):
    k1 = f_cont_jax(x, u)
    k2 = f_cont_jax(x + 0.5 * dt * k1, u)
    k3 = f_cont_jax(x + 0.5 * dt * k2, u)
    k4 = f_cont_jax(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def linearize_discrete_jax(x, dt, dtype):
    """Jacobian linearization of the discrete (RK4) dynamics at (x, u=0).

        x+ ~= A_d x + G0 u + d_aff,   d_aff = f_rk4(x, 0) - A_d x
    """
    u0    = jnp.zeros(NU, dtype)
    A_d   = jax.jacfwd(lambda xx: rk4_step_jax(xx, u0, dt))(x)
    G0    = jax.jacfwd(lambda uu: rk4_step_jax(x, uu, dt))(u0)
    d_aff = rk4_step_jax(x, u0, dt) - A_d @ x
    return A_d, G0, d_aff


def riccati_cost_to_go(A, B, Q, R, P_term, horizon):
    """Backward LQR Riccati recursion on the linearized control-affine model."""
    def step(P, _):
        BtPA = B.T @ P @ A
        S    = R + B.T @ P @ B
        P_next = Q + A.T @ P @ A - BtPA.T @ jnp.linalg.solve(S, BtPA)
        return 0.5 * (P_next + P_next.T), None
    P, _ = jax.lax.scan(step, P_term, None, length=int(horizon))
    return P


# ---------------------------------------------------------------------------
# Path-following stage cost as a Gauss-Newton residual
# ---------------------------------------------------------------------------
def _path_tangent_angle(kx, s):
    """gamma_p = atan2(y_d'(s), x_d'(s)) for x_d=kx*s, y_d=sin(kx*s)+y0."""
    return jnp.arctan2(kx * jnp.cos(kx * s), kx)


def stage_residual(x, path, weights):
    """Residual vector r(x) with ||r||^2 == vessel stage cost (state part)."""
    q_ye, q_r, q_psi, q_u = weights
    X, Y, psi, su, sv, sr, s = x
    kx, y0, u_ref = path

    x_d = kx * s
    y_d = jnp.sin(kx * s) + y0
    gp  = _path_tangent_angle(kx, s)
    ye  = -(X - x_d) * jnp.sin(gp) + (Y - y_d) * jnp.cos(gp)

    return jnp.stack([
        jnp.sqrt(q_ye) * ye,
        jnp.sqrt(q_r)  * sr,
        jnp.sqrt(q_psi) * (jnp.sin(psi) - jnp.sin(gp)),
        jnp.sqrt(q_psi) * (jnp.cos(psi) - jnp.cos(gp)),
        jnp.sqrt(q_u)  * (su - u_ref),
    ])


# ---------------------------------------------------------------------------
# QP layer (amortized: min 1/2 u^T H u + c^T u  s.t. box + HOCBF)
# ---------------------------------------------------------------------------
_QP_SOLVE = qpax.solve_qp_primal


def _box_constraints(env, dtype):
    """Thruster box |f_i| <= f_max as G u <= h with G = [I; -I], h = f_max."""
    cd    = env.control_dim
    box_G = jnp.concatenate([jnp.eye(cd, dtype=dtype), -jnp.eye(cd, dtype=dtype)], axis=0)
    box_h = jnp.full((2 * cd,), float(env.f_max), dtype=dtype)
    return box_G, box_h


def make_amortized_qp_layer(env, dtype=jnp.float64, solver_tol=1e-8,
                            target_kappa=1e-4):
    """Returns a `single(H, c, G_cbf, h_cbf) -> u` closure (interior-point qpax)."""
    cd      = env.control_dim
    A_const = jnp.zeros((0, cd), dtype=dtype)
    b_const = jnp.zeros((0,),    dtype=dtype)
    box_G, box_h = _box_constraints(env, dtype)

    def single(H, c, G_cbf, h_cbf):
        G = jnp.concatenate([box_G, G_cbf.astype(dtype)], axis=0)
        h = jnp.concatenate([box_h, h_cbf.astype(dtype)], axis=0)
        return _QP_SOLVE(
            H.astype(dtype), c.astype(dtype), A_const, b_const, G, h,
            solver_tol=solver_tol, target_kappa=target_kappa,
        )

    return single, dtype


# ---------------------------------------------------------------------------
# 2nd-order CBF constraint matrices (JAX, by autodiff)
# ---------------------------------------------------------------------------
def _cbf_single(x, center, r_safe, g1, g2, dtype):
    """(G_row (4,), h_rhs scalar) for one circular obstacle, in G u <= h form.

    HOCBF value:  h_dd + (g1+g2) h_d + g1 g2 h >= 0, with h = ||p-c||^2 - r_safe^2.
    """
    def h_of(xx):
        dp = xx[:2] - center
        return jnp.dot(dp, dp) - r_safe ** 2

    def hd_of(xx, uu):
        return jnp.dot(jax.grad(h_of)(xx), f_cont_jax(xx, uu))

    def hdd_of(xx, uu):
        grad_hd = jax.grad(lambda x_: hd_of(x_, uu))(xx)
        return jnp.dot(grad_hd, f_cont_jax(xx, uu))

    zeros_u = jnp.zeros(NU, dtype)
    h_val  = h_of(x)
    h_d    = hd_of(x, zeros_u)
    a0     = hdd_of(x, zeros_u)                              # h_dd at u=0
    Jb     = jax.jacfwd(lambda uu: hdd_of(x, uu))(zeros_u)   # d h_dd / du  (4,)

    G_row = -Jb.astype(dtype)
    h_rhs = (a0 + (g1 + g2) * h_d + g1 * g2 * h_val).astype(dtype)
    return G_row, h_rhs


def _cbf_obs_jax(x, obs_centers, obs_radii, gamma1_obs, gamma2_obs, r_ego, dtype):
    """Vectorized (G_obs (n_obs,4), h_obs (n_obs,)) over all obstacles."""
    r_safe = obs_radii.astype(dtype) + jnp.asarray(r_ego, dtype)

    def per_obs(c, rs, g1, g2):
        return _cbf_single(x.astype(dtype), c.astype(dtype), rs, g1, g2, dtype)

    G_obs, h_obs = jax.vmap(per_obs)(obs_centers, r_safe, gamma1_obs, gamma2_obs)
    return G_obs, h_obs


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
_INPUT_DIM_BASE = 11  # see _features below


def _features(x, path, obs_centers, obs_radii):
    """Vessel-frame feature vector, dim 11 + 3*n_obs."""
    X, Y, psi, su, sv, sr, s = x
    kx, y0, u_ref = path
    p = x[:2]

    x_d = kx * s
    y_d = jnp.sin(kx * s) + y0
    gp  = _path_tangent_angle(kx, s)
    ye  = -(X - x_d) * jnp.sin(gp) + (Y - y_d) * jnp.cos(gp)

    cps, sps = jnp.cos(psi), jnp.sin(psi)
    fwd  = jnp.stack([cps, sps])     # body axes in world coords
    left = jnp.stack([-sps, cps])

    def to_body(w):
        return jnp.stack([w @ fwd, w @ left])

    obs_rel = jax.vmap(lambda c: to_body(c - p))(obs_centers).reshape(-1)

    base = jnp.stack([
        su, sv, sr,
        su - u_ref,
        ye,
        jnp.sin(psi - gp), jnp.cos(psi - gp),
        X, Y,
        cps, sps,
    ])
    return jnp.concatenate([base, obs_rel, obs_radii])


def _norm_features(model, x, path, obs_centers, obs_radii):
    f  = _features(x, path, obs_centers, obs_radii)
    mu = jax.lax.stop_gradient(model.feat_mu).astype(f.dtype)
    sd = jax.lax.stop_gradient(model.feat_sd).astype(f.dtype)
    return (f - mu) / sd


def _feat_stats_arrays(feat_stats, input_dim):
    if feat_stats is None:
        return jnp.zeros(input_dim), jnp.ones(input_dim)
    mu, sd = feat_stats
    return jnp.asarray(mu, jnp.float64), jnp.asarray(sd, jnp.float64)


def _zero_linear(lin, bias_value=0.0):
    """Zero-init a head so the model starts exactly at its baseline."""
    lin = eqx.tree_at(lambda l: l.weight, lin, jnp.zeros_like(lin.weight))
    if lin.bias is not None:
        lin = eqx.tree_at(lambda l: l.bias, lin,
                          jnp.full_like(lin.bias, bias_value))
    return lin


# gamma heads output softplus(z) + 0.01; this bias makes each gain = 1 at init.
_ALPHA_HEAD_BIAS = float(np.log(np.expm1(1.0 - 0.01)))


def _gammas_from_heads(gamma1_head, gamma2_head, h_feat, d):
    g1 = (jax.nn.softplus(gamma1_head(h_feat)) + jnp.asarray(0.01, d)).astype(d)
    g2 = (jax.nn.softplus(gamma2_head(h_feat)) + jnp.asarray(0.01, d)).astype(d)
    return g1, g2


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AmortizedNMPC(eqx.Module):
    """MLP -> (delta_L, delta_c) residual on a Gauss-Newton analytic (H, c) baseline."""
    backbone:     eqx.nn.MLP
    delta_L_head: eqx.nn.Linear
    delta_c_head: eqx.nn.Linear
    feat_mu:      jax.Array
    feat_sd:      jax.Array
    n_obs:        int   = eqx.static_field()
    control_dim:  int   = eqx.static_field()
    r_ego:        float = eqx.static_field()
    dt:           float = eqx.static_field()
    n_u:          int    = eqx.static_field()
    qp_solve:     object = eqx.static_field()
    dtype:        object = eqx.static_field()
    weights:      tuple  = eqx.static_field()
    r_f:          float  = eqx.static_field()
    ctg_horizon:  int    = eqx.static_field()

    def __init__(self, env, key, qp_solve, dtype, weights=None, r_f=None,
                 feat_stats=None, ctg_horizon=0):
        keys        = jax.random.split(key, 3)
        input_dim   = _INPUT_DIM_BASE + env.n_obs * 3
        n_u         = env.control_dim
        n_L_entries = n_u * (n_u + 1) // 2

        self.backbone     = eqx.nn.MLP(
            in_size=input_dim, out_size=64, width_size=64, depth=2,
            activation=jax.nn.relu, key=keys[0],
        )
        self.delta_L_head = _zero_linear(eqx.nn.Linear(64, n_L_entries, key=keys[1]))
        self.delta_c_head = _zero_linear(eqx.nn.Linear(64, n_u,         key=keys[2]))
        self.feat_mu, self.feat_sd = _feat_stats_arrays(feat_stats, input_dim)

        self.n_obs       = env.n_obs
        self.control_dim = env.control_dim
        self.r_ego       = float(env.r_ego)
        self.dt          = float(env.dt)
        self.n_u         = n_u
        self.qp_solve    = qp_solve
        self.dtype       = dtype
        self.weights     = tuple(weights) if weights is not None else _DEFAULT_WEIGHTS
        self.r_f         = float(r_f) if r_f is not None else _DEFAULT_RF
        self.ctg_horizon = int(ctg_horizon)

    def _analytic_baseline(self, x, path, d):
        """Per-sample (L_an, c_an) via Gauss-Newton on the path-following cost."""
        weights = tuple(jnp.asarray(w, d) for w in self.weights)
        R_f = jnp.asarray(self.r_f, d) * jnp.eye(self.n_u, dtype=d)

        A_d, G0, _ = linearize_discrete_jax(x, self.dt, d)
        f0 = rk4_step_jax(x, jnp.zeros(self.n_u, d), self.dt)

        r0 = stage_residual(f0, path, weights)                            # (n_r,)
        J  = jax.jacfwd(lambda xx: stage_residual(xx, path, weights))(f0)  # (n_r, 7)
        W  = J.T @ J + jnp.asarray(1e-6, d) * jnp.eye(NX, dtype=d)
        g  = J.T @ r0
        delta = jnp.linalg.solve(W, g)                                   # W^{-1} g

        if self.ctg_horizon > 0:
            P = riccati_cost_to_go(A_d, G0, W, R_f, W, self.ctg_horizon)
        else:
            P = W

        H_an = 2.0 * (G0.T @ P @ G0 + R_f)
        L_an = jnp.linalg.cholesky(H_an)
        c_an = 2.0 * G0.T @ (P @ delta)
        return L_an, c_an

    def _residual_H(self, h_feat, L_an, d):
        delta_L_vec = self.delta_L_head(h_feat).astype(d)
        n_u = self.n_u
        rows, cols = jnp.tril_indices(n_u)
        L_lower = jnp.zeros((n_u, n_u), d).at[rows, cols].set(delta_L_vec)
        offset   = jnp.log(jnp.expm1(jnp.diag(L_an)))   # softplus^{-1}(diag(L_an))
        diag_pos = (jax.nn.softplus(jnp.diag(L_lower) + offset)
                    + jnp.asarray(1e-6, d))
        L_full = (L_an + L_lower).at[jnp.arange(n_u), jnp.arange(n_u)].set(diag_pos)
        return (L_full @ L_full.T).astype(d)

    def _residual_cost(self, h_feat, L_an, c_an, d):
        H_full  = self._residual_H(h_feat, L_an, d)
        delta_c = self.delta_c_head(h_feat).astype(d)
        c_full  = (c_an + delta_c).astype(d)
        return H_full, c_full

    def _u_nom_viz(self, x, path, d):
        return jnp.zeros(self.n_u, d)

    def __call__(self, x, path, obs_centers, obs_radii):
        d = self.dtype
        x = x.astype(d); path = path.astype(d)
        obs_centers = obs_centers.astype(d); obs_radii = obs_radii.astype(d)

        h_feat = self.backbone(_norm_features(self, x, path, obs_centers, obs_radii))
        L_an, c_an     = self._analytic_baseline(x, path, d)
        H_full, c_full = self._residual_cost(h_feat, L_an, c_an, d)

        ones = jnp.ones((self.n_obs,), d)
        G_cbf, h_cbf = _cbf_obs_jax(x, obs_centers, obs_radii, ones, ones, self.r_ego, d)
        u_safe = self.qp_solve(H_full, c_full, G_cbf, h_cbf)
        return u_safe.astype(d), self._u_nom_viz(x, path, d), jnp.stack([ones, ones], axis=-1)


class AmortizedNMPCAlpha(AmortizedNMPC):
    """AmortizedNMPC with learned per-obstacle HOCBF gains (gamma1, gamma2)."""
    gamma1_head: eqx.nn.Linear
    gamma2_head: eqx.nn.Linear

    def __init__(self, env, key, qp_solve, dtype, weights=None, r_f=None,
                 feat_stats=None, ctg_horizon=0):
        key, g1_key, g2_key = jax.random.split(key, 3)
        super().__init__(env, key, qp_solve, dtype, weights=weights, r_f=r_f,
                         feat_stats=feat_stats, ctg_horizon=ctg_horizon)
        self.gamma1_head = _zero_linear(eqx.nn.Linear(64, env.n_obs, key=g1_key),
                                        bias_value=_ALPHA_HEAD_BIAS)
        self.gamma2_head = _zero_linear(eqx.nn.Linear(64, env.n_obs, key=g2_key),
                                        bias_value=_ALPHA_HEAD_BIAS)

    def __call__(self, x, path, obs_centers, obs_radii):
        d = self.dtype
        x = x.astype(d); path = path.astype(d)
        obs_centers = obs_centers.astype(d); obs_radii = obs_radii.astype(d)

        h_feat = self.backbone(_norm_features(self, x, path, obs_centers, obs_radii))
        L_an, c_an     = self._analytic_baseline(x, path, d)
        H_full, c_full = self._residual_cost(h_feat, L_an, c_an, d)

        g1, g2 = _gammas_from_heads(self.gamma1_head, self.gamma2_head, h_feat, d)
        G_cbf, h_cbf = _cbf_obs_jax(x, obs_centers, obs_radii, g1, g2, self.r_ego, d)
        u_safe = self.qp_solve(H_full, c_full, G_cbf, h_cbf)
        return u_safe.astype(d), self._u_nom_viz(x, path, d), jnp.stack([g1, g2], axis=-1)


# ---------------------------------------------------------------------------
# Loader (the single entry point a caller needs)
# ---------------------------------------------------------------------------
def build_alpha_skeleton(env, weights=_DEFAULT_WEIGHTS, r_f=_DEFAULT_RF,
                         dtype=jnp.float64):
    """Fresh (untrained) AmortizedNMPCAlpha with the exact tree the checkpoint expects."""
    qp_single, dtype = make_amortized_qp_layer(env, dtype=dtype)
    model = AmortizedNMPCAlpha(
        env, jax.random.PRNGKey(0), qp_solve=qp_single, dtype=dtype,
        weights=weights, r_f=r_f, ctg_horizon=0,
    )
    # Match the pipeline's float64 cast of all inexact leaves.
    model = jax.tree_util.tree_map(
        lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, model,
    )
    return model, dtype


def load_alpha_model(model_path, n_obs=None, dt=0.1, f_max=_F_MAX,
                     weights=_DEFAULT_WEIGHTS, r_f=_DEFAULT_RF, meta_path=None,
                     r_ego=None):
    """Load a trained `ana_i64` model from `model.eqx`.

    Parameters mirror what the model was trained with. Either pass `n_obs`
    explicitly, or point `meta_path` at a `model_meta.yaml` (see this bundle) that
    carries n_obs / dt / f_max / weights.

    `r_ego` (ego safety radius, default hypot(L, W)/2 ~ 0.503 m) may be chosen
    freely: it enters ONLY the CBF constraint rows, via r_safe = obs_radius + r_ego,
    and no learned weight depends on it. Raising it buys real standoff almost
    one-for-one at the cost of path-following accuracy; lowering it below the true
    hull radius causes real collisions. It is a static field, so changing it
    triggers one recompilation -- set it at load time, not per tick.

    Returns (model, dtype). Call `model(x, path, obs_centers, obs_radii)`.
    """
    if meta_path is not None:
        with open(meta_path) as fh:
            meta = yaml.safe_load(fh)
        n_obs  = int(meta.get("n_obs", n_obs))
        dt     = float(meta.get("dt", dt))
        f_max  = float(meta.get("f_max", f_max))
        w      = meta.get("weights")
        if w is not None:
            weights = (float(w["q_ye"]), float(w["q_r"]),
                       float(w["q_psi"]), float(w["q_u"]))
            r_f = float(w.get("r_f", r_f))
    if n_obs is None:
        raise ValueError("n_obs is required (pass it directly or via meta_path).")

    env = make_env(n_obs=n_obs, dt=dt, f_max=f_max, r_ego=r_ego)
    skel, dtype = build_alpha_skeleton(env, weights=weights, r_f=r_f)
    model = eqx.tree_deserialise_leaves(str(model_path), skel)
    return model, dtype


# ---------------------------------------------------------------------------
# True-plant helpers (numpy) for closed-loop simulation
# ---------------------------------------------------------------------------
def _vessel_ode_np(x, u):
    X, Y, psi, su, sv, sr, s = (float(x[i]) for i in range(NX))
    f1, f2, f3, f4 = (float(u[i]) for i in range(NU))

    tau1 = f1 + f2
    tau2 = f3 + f4
    tau3 = _AA / 2.0 * (f1 - f2) + _BB / 2.0 * (f3 - f4)

    X_dot = su * np.cos(psi) - sv * np.sin(psi)
    Y_dot = su * np.sin(psi) + sv * np.cos(psi)
    psi_dot = sr

    u_dot = (tau1 + _M22 * sv * sr - _D11 * su) / _M11
    v_dot = (tau2 - _M11 * su * sr - _D22 * sv) / _M22
    r_dot = (tau3 - (_M22 - _M11) * su * sv - _D33 * sr) / _M33

    s_dot = su
    return np.array([X_dot, Y_dot, psi_dot, u_dot, v_dot, r_dot, s_dot])


def rk4_step_numpy(x, u, dt=0.1):
    """One explicit RK4 step of the true vessel plant (no wrap / clip)."""
    x = np.asarray(x, float).reshape(NX)
    u = np.asarray(u, float).reshape(NU)
    k1 = _vessel_ode_np(x, u)
    k2 = _vessel_ode_np(x + 0.5 * dt * k1, u)
    k3 = _vessel_ode_np(x + 0.5 * dt * k2, u)
    k4 = _vessel_ode_np(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def wrap_angle(th):
    """Wrap angle to [-pi, pi)."""
    return (th + np.pi) % (2.0 * np.pi) - np.pi


def anchor_arc_length(path, x, s_prev, look_back=1.0, look_ahead=10.0, n=2000):
    """Re-anchor the path parameter s to the vessel's (X, Y) by nearest projection."""
    kx, y0, _ = (float(v) for v in path)
    s_lo = max(0.0, s_prev - look_back)
    s_hi = s_prev + look_ahead
    s_grid = np.linspace(s_lo, s_hi, n)
    x_d = kx * s_grid
    y_d = np.sin(kx * s_grid) + y0
    d2 = (float(x[0]) - x_d) ** 2 + (float(x[1]) - y_d) ** 2
    return float(s_grid[int(np.argmin(d2))])
