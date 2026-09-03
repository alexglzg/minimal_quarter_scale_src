"""Self-contained loader + architecture for the BarrierNet vessel baseline.

Companion to `anmpc_alpha.py`. Same task, same plant, same safety filter — a
*different* way of amortizing the NMPC:

    anmpc_alpha (`ana_i64`)  MLP predicts a residual on a Gauss-Newton analytic QP
                             objective (H, c) + per-obstacle class-K gains, then
                             solves   min 1/2 u' H u + c' u   s.t. box + HOCBF.

    barriernet  (`i64`)      MLP predicts a *nominal command* u_nom + per-obstacle
                             class-K gains, then projects it:
                             min 1/2 ||u - u_nom||^2  s.t. box + HOCBF.

So BarrierNet learns "what to do" and lets the QP repair it; amortized NMPC learns
"what to optimize" and lets the QP decide what to do. Both return a command that
satisfies the same high-order-CBF rows, so both are safe by construction; they differ
in closed-loop *cost* (see `compare_models.ipynb`).

This module deliberately reuses the shared pieces (dynamics, features, CBF rows, plant
helpers) from `anmpc_alpha.py` rather than duplicating them — that guarantees the two
controllers are compared on bit-identical geometry. Only the QP layer, the network
module and the loader are BarrierNet-specific and live here.

Conventions are identical to `anmpc_alpha.py`:
    x    (7,)  [X, Y, psi, su, sv, sr, s]
    u    (4,)  thruster forces, |f_i| <= f_max
    path (3,)  [kx, y0, u_ref]
    obstacles  obs_centers (n_obs, 2), obs_radii (n_obs,)

Main entry point
----------------
    model, dtype = load_barriernet_model("model.eqx", meta_path="model_meta.yaml")
    u_safe, u_nom, gammas = model(x, path, obs_centers, obs_radii)   # jax arrays

`u_nom` is the network's *unfiltered* command here (unlike the amortized model, where
it is a placeholder) — the gap ||u_safe - u_nom|| is exactly how much the CBF-QP had
to intervene, which the comparison notebook plots.
"""
import yaml

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

import qpax

# anmpc_alpha's package __init__ only re-exports its small public surface
# (load_alpha_model, make_env, rk4_step_numpy, wrap_angle, anchor_arc_length);
# the shared dynamics/features/CBF/plant helpers this module reuses are
# private to the implementation submodule, so pull from there directly.
from anmpc_alpha.anmpc_alpha import (
    _F_MAX,
    make_env,
    _box_constraints,
    _cbf_obs_jax,
    _norm_features, _feat_stats_arrays, _INPUT_DIM_BASE,
    _zero_linear, _ALPHA_HEAD_BIAS, _gammas_from_heads,
    # re-exported for convenience so callers can `from barriernet import ...`
    rk4_step_numpy, wrap_angle, anchor_arc_length,
)

# Same call as anmpc_alpha.py's _QP_SOLVE: this repo's image pins qpax==0.0.9,
# whose solve_qp_primal has neither a `backend` nor a `max_iter` argument (both
# added in later qpax releases) -- so plain solve_qp_primal, no partial.
_QP_SOLVE = qpax.solve_qp_primal


# ---------------------------------------------------------------------------
# QP layer: projection onto the safe set
# ---------------------------------------------------------------------------
def make_projection_qp_layer(env, dtype=jnp.float64, solver_tol=1e-8,
                             target_kappa=1e-4):
    """Returns a `single(u_nom, G_cbf, h_cbf) -> u` closure.

    Solves  min 1/2 ||u - u_nom||^2  s.t.  |u_i| <= f_max  and  G_cbf u <= h_cbf,
    i.e. qpax with Q = I and q = -u_nom. Same interior-point backend and tolerances
    as the amortized layer, so solve times are comparable.
    """
    cd      = env.control_dim
    Q_const = jnp.eye(cd, dtype=dtype)
    A_const = jnp.zeros((0, cd), dtype=dtype)
    b_const = jnp.zeros((0,),    dtype=dtype)
    box_G, box_h = _box_constraints(env, dtype)

    def single(u_nom, G_cbf, h_cbf):
        G = jnp.concatenate([box_G, G_cbf.astype(dtype)], axis=0)
        h = jnp.concatenate([box_h, h_cbf.astype(dtype)], axis=0)
        return _QP_SOLVE(
            Q_const, -u_nom.astype(dtype), A_const, b_const, G, h,
            solver_tol=solver_tol, target_kappa=target_kappa,
        )

    return single, dtype


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BarrierNet(eqx.Module):
    """MLP -> (u_nom, gamma1, gamma2) -> 2nd-order CBF-QP -> u_safe.

    The backbone is the same shape as the amortized model's (2x64 ReLU MLP on the
    same normalized feature vector); only the heads differ: a 4-dim nominal-command
    head instead of the (Cholesky residual, linear residual) pair.
    """
    backbone:    eqx.nn.MLP
    u_nom_head:  eqx.nn.Linear
    gamma1_head: eqx.nn.Linear
    gamma2_head: eqx.nn.Linear
    feat_mu:     jax.Array
    feat_sd:     jax.Array
    # eqx.field(static=True) needs equinox >= 0.11; this repo's image pins 0.10.4
    # (same as anmpc_alpha.py), which only has the older static_field() spelling.
    n_obs:       int   = eqx.static_field()
    control_dim: int   = eqx.static_field()
    r_ego:       float = eqx.static_field()
    dt:          float = eqx.static_field()
    qp_solve:    object = eqx.static_field()
    dtype:       object = eqx.static_field()

    def __init__(self, env, key, qp_solve, dtype, feat_stats=None):
        keys      = jax.random.split(key, 4)
        input_dim = _INPUT_DIM_BASE + env.n_obs * 3
        self.backbone   = eqx.nn.MLP(
            in_size=input_dim, out_size=64, width_size=64, depth=2,
            activation=jax.nn.relu, key=keys[0],
        )
        self.u_nom_head  = _zero_linear(eqx.nn.Linear(64, env.control_dim, key=keys[1]))
        self.gamma1_head = _zero_linear(eqx.nn.Linear(64, env.n_obs, key=keys[2]),
                                        bias_value=_ALPHA_HEAD_BIAS)
        self.gamma2_head = _zero_linear(eqx.nn.Linear(64, env.n_obs, key=keys[3]),
                                        bias_value=_ALPHA_HEAD_BIAS)
        self.feat_mu, self.feat_sd = _feat_stats_arrays(feat_stats, input_dim)
        self.n_obs       = env.n_obs
        self.control_dim = env.control_dim
        self.r_ego       = float(env.r_ego)
        self.dt          = float(env.dt)
        self.qp_solve    = qp_solve
        self.dtype       = dtype

    def __call__(self, x, path, obs_centers, obs_radii):
        d = self.dtype
        x = x.astype(d); path = path.astype(d)
        obs_centers = obs_centers.astype(d); obs_radii = obs_radii.astype(d)

        h_feat = self.backbone(_norm_features(self, x, path, obs_centers, obs_radii))
        u_nom  = self.u_nom_head(h_feat).astype(d)
        g1, g2 = _gammas_from_heads(self.gamma1_head, self.gamma2_head, h_feat, d)

        G_cbf, h_cbf = _cbf_obs_jax(x, obs_centers, obs_radii, g1, g2, self.r_ego, d)
        u_safe = self.qp_solve(u_nom, G_cbf, h_cbf)
        return u_safe.astype(d), u_nom, jnp.stack([g1, g2], axis=-1)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def build_barriernet_skeleton(env, dtype=jnp.float64):
    """Fresh (untrained) BarrierNet with the exact tree the checkpoint expects."""
    qp_single, dtype = make_projection_qp_layer(env, dtype=dtype)
    model = BarrierNet(env, jax.random.PRNGKey(0), qp_solve=qp_single, dtype=dtype)
    # jax.tree.map needs jax >= 0.4.25; this repo's image pins 0.4.13 (same as
    # anmpc_alpha.py), which only has the older jax.tree_util.tree_map spelling.
    model = jax.tree_util.tree_map(
        lambda a: a.astype(dtype) if eqx.is_inexact_array(a) else a, model,
    )
    return model, dtype


def load_barriernet_model(model_path, n_obs=None, dt=0.1, f_max=_F_MAX,
                          meta_path=None, r_ego=None):
    """Load a trained `i64` (BarrierNet) model from an Equinox checkpoint.

    Either pass `n_obs` explicitly or point `meta_path` at a `model_meta.yaml`
    carrying n_obs / dt / f_max. Unlike the amortized model, BarrierNet has no
    analytic baseline, so no cost weights are needed to rebuild the skeleton — the
    stage cost is entirely inside the learned weights.

    `r_ego` (ego safety radius, default hypot(L, W)/2 ~ 0.503 m) enters ONLY the CBF
    rows via r_safe = obs_radius + r_ego; no learned weight depends on it. It is a
    static field, so set it at load time rather than per tick.

    Returns (model, dtype). Call `model(x, path, obs_centers, obs_radii)`.
    """
    if meta_path is not None:
        with open(meta_path) as fh:
            meta = yaml.safe_load(fh)
        n_obs = int(meta.get("n_obs", n_obs))
        dt    = float(meta.get("dt", dt))
        f_max = float(meta.get("f_max", f_max))
    if n_obs is None:
        raise ValueError("n_obs is required (pass it directly or via meta_path).")

    env = make_env(n_obs=n_obs, dt=dt, f_max=f_max, r_ego=r_ego)
    skel, dtype = build_barriernet_skeleton(env)
    model = eqx.tree_deserialise_leaves(str(model_path), skel)
    return model, dtype


__all__ = [
    "BarrierNet", "build_barriernet_skeleton", "load_barriernet_model",
    "make_projection_qp_layer", "make_env",
    "rk4_step_numpy", "wrap_angle", "anchor_arc_length",
]
