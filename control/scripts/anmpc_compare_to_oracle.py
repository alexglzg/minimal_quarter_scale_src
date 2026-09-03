"""Compare the amortized model against the NMPC oracle it was trained to imitate.

    python compare_to_oracle.py                        # 3 scenarios x 200 steps
    python compare_to_oracle.py --n-configs 1 --n-steps 120
    python compare_to_oracle.py --no-plot

Needs casadi for the oracle (`pip install casadi`); the oracle formulation lives in
`mpc_oracle.py`, which is worth reading first -- it is the optimization the network
learned to shortcut.

Two different questions get answered, and they are easy to conflate:

1. **Open-loop imitation.** Along the *oracle's own* closed-loop trajectory, ask the
   model what it would have commanded at each visited state. This measures how well
   the network reproduces its teacher, state for state, with no feedback effects. It
   is essentially the training objective, evaluated on fresh scenarios.

2. **Closed-loop behaviour.** Roll out each controller independently from the same
   initial condition through the same plant. Small per-step differences compound here,
   so the two trajectories diverge even when the imitation error is small. This is
   what actually matters for deployment: does it stay safe, does it keep tracking?

A note on what is held identical: same plant (`VesselSimulator`, full nonlinear RK4),
same initial condition, same obstacles, same arc-length anchoring, same cost weights.

Two things intentionally differ, both structural rather than tuning choices:

* The oracle sees the 3 *distinct* obstacles; the model must be given exactly
  `n_obs = 6`, so its unused slots are filled by duplicating the farthest obstacle
  (redundant CBF rows -- see INTEGRATION.md 2.3). The physical scenario is identical.
* The oracle uses fixed class-K gains (0.5, 0.5) -- the values used to generate the
  training data -- while the model *predicts* its own gains per obstacle. Learning
  those gains is part of the method (`ana` = amortized + alpha), so this is a genuine
  difference in the controllers, not an unfair comparison.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import equinox as eqx

from anmpc_alpha import load_alpha_model, make_env, wrap_angle, anchor_arc_length
from mpc_oracle import (VesselNMPC, VesselParams, PathParams, CostWeights,
                        Obstacle, VesselSimulator, project_arc_length)

HERE = Path(__file__).resolve().parent
# oracle + amortized model together -> a comparison, not either one's own plot.
PLOTS_DIR = HERE / ".." / ".." / "plots" / "comparisons"

# In-distribution sampling ranges (from the training config).
Y0_RANGE     = (2.5, 3.5)
U_REF_RANGE  = (0.40, 0.70)
RADIUS_RANGE = (0.4, 0.8)
SPACING_RANGE = (4.0, 8.0)
FIRST_GAP_RANGE = (5.0, 7.0)
LATERAL = 0.2
KX = 0.2            # geometrically inert (the world path is y = sin(X) + y0 either way)
N_REAL_OBS = 3


# --------------------------------------------------------------------------- #
# Scenario sampling
# --------------------------------------------------------------------------- #
def sample_scenario(rng):
    y0 = rng.uniform(*Y0_RANGE)
    u_ref = rng.uniform(*U_REF_RANGE)
    spacing = rng.uniform(*SPACING_RANGE)
    first = rng.uniform(*FIRST_GAP_RANGE)

    obstacles = []
    for i in range(N_REAL_OBS):
        X = first + spacing * i
        lat = LATERAL * (1 if i % 2 == 0 else -1)
        obstacles.append(Obstacle(X, float(np.sin(X) + y0 + lat),
                                  float(rng.uniform(*RADIUS_RANGE))))

    path = PathParams(kx=KX, y0=y0, u_ref=u_ref)
    psi0 = np.arctan2(np.cos(0.0), 1.0)          # path tangent at X=0 -> 45 deg
    # Starts from rest (su=0), matching qs_sim_node: Sim.cpp's constructor never
    # sets an initial velocity from any rosparam, so the real ROS plant always
    # starts at exactly 0 -- this used to start at 0.3 m/s already moving, which
    # gave the two setups a several-second head start mismatch.
    x0 = np.array([0.0, y0, psi0, 0.0, 0.0, 0.0, 0.0])
    x0[6] = project_arc_length(path, x0[0], x0[1], 0.0)
    return dict(path=path, obstacles=obstacles, x0=x0)


def model_obstacle_arrays(obstacles, n_obs):
    """Distinct obstacles -> the fixed (n_obs, 2)/(n_obs,) arrays the model needs."""
    obs = [(o.cx, o.cy, o.radius) for o in obstacles]
    while len(obs) < n_obs:
        obs.append(obs[len(obstacles) - 1])       # duplicate the farthest
    centers = np.array([[o[0], o[1]] for o in obs], float)
    radii = np.array([o[2] for o in obs], float)
    return centers, radii


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cross_track(path, x, s):
    x_d, y_d = path.xy(s)
    gp = np.arctan2(path.kx * np.cos(path.kx * s), path.kx)
    return -(x[0] - x_d) * np.sin(gp) + (x[1] - y_d) * np.cos(gp)


def summarize(states, us, clears, times, path):
    """Aggregate one rollout.

    Timing excludes the FIRST step for both controllers, and reports it separately:
    it measures one-off setup, not steady-state control cost. For the oracle that is
    the cold IPOPT solve with no warm start (~10-20x a warm one); for the model it
    would be XLA compilation, except we warm that up before timing (see `main`).
    """
    states, us = np.asarray(states), np.asarray(us)
    ye = [cross_track(path, s_, s_[6]) for s_ in states]
    steady = np.asarray(times[1:]) if len(times) > 1 else np.asarray(times)
    return dict(
        dX=float(states[-1, 0] - states[0, 0]),
        min_clear=float(np.min(clears)),
        ye_rms=float(np.sqrt(np.mean(np.square(ye)))),
        effort=float(np.sqrt(np.mean(np.sum(us ** 2, axis=1)))),
        t_ms=float(np.mean(steady) * 1e3),
        t_max_ms=float(np.max(steady) * 1e3),
        t_first_ms=float(times[0] * 1e3),
    )


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
@eqx.filter_jit
def _model_call(model, x, path_v, oc, orr):
    return model(x, path_v, oc, orr)


def run_model(model, dtype, env, sc, sim, n_steps):
    """Closed loop with the learned controller. Returns (states, metrics)."""
    path = sc["path"]
    centers, radii = model_obstacle_arrays(sc["obstacles"], env.n_obs)
    path_v = np.array([path.kx, path.y0, path.u_ref])
    pj = jnp.asarray(path_v, dtype)
    cj, rj = jnp.asarray(centers, dtype), jnp.asarray(radii, dtype)
    r_safe = radii + env.r_ego

    x = sc["x0"].copy()
    states, us, clears, times = [x.copy()], [], [], []
    for _ in range(n_steps):
        x[2] = wrap_angle(x[2])
        x[6] = anchor_arc_length(path_v, x, x[6])
        t0 = time.perf_counter()
        u = np.asarray(_model_call(model, jnp.asarray(x, dtype), pj, cj, rj)[0], float)
        times.append(time.perf_counter() - t0)

        x = sim.step(x, u)
        x[6] = anchor_arc_length(path_v, x, x[6])
        states.append(x.copy()); us.append(u)
        clears.append(float((np.linalg.norm(x[:2] - centers, axis=1) - r_safe).min()))
    return np.array(states), summarize(states, us, clears, times, path)


def run_oracle(mpc, sc, sim, n_steps):
    """Closed loop with the NMPC oracle. Returns (states, metrics, n_failed)."""
    path = sc["path"]
    centers = np.array([[o.cx, o.cy] for o in sc["obstacles"]], float)
    radii = np.array([o.radius for o in sc["obstacles"]], float)
    r_safe = radii + mpc.p.r_ego
    path_v = np.array([path.kx, path.y0, path.u_ref])

    mpc.reset_warm_start()
    x = sc["x0"].copy()
    states, us, clears, times, n_failed = [x.copy()], [], [], [], 0
    for _ in range(n_steps):
        x[2] = wrap_angle(x[2])
        x[6] = anchor_arc_length(path_v, x, x[6])
        sol = mpc(x)
        n_failed += int(sol.status != "optimal")
        times.append(sol.solve_time_s)

        x = sim.step(x, sol.u_opt)
        x[6] = anchor_arc_length(path_v, x, x[6])
        states.append(x.copy()); us.append(sol.u_opt)
        clears.append(float((np.linalg.norm(x[:2] - centers, axis=1) - r_safe).min()))
    return np.array(states), summarize(states, us, clears, times, path), n_failed


def imitation_error(model, dtype, env, sc, oracle_states, mpc):
    """Open-loop: model vs oracle command at each state the ORACLE visited.

    Both are asked the same question at the same state, so this isolates the
    network's approximation error from any feedback divergence.
    """
    path = sc["path"]
    centers, radii = model_obstacle_arrays(sc["obstacles"], env.n_obs)
    path_v = np.array([path.kx, path.y0, path.u_ref])
    pj = jnp.asarray(path_v, dtype)
    cj, rj = jnp.asarray(centers, dtype), jnp.asarray(radii, dtype)

    mpc.reset_warm_start()
    errs, u_norms = [], []
    for x in oracle_states[:-1]:
        u_or = mpc(x).u_opt
        u_md = np.asarray(_model_call(model, jnp.asarray(x, dtype), pj, cj, rj)[0], float)
        errs.append(u_md - u_or)
        u_norms.append(np.linalg.norm(u_or))
    errs = np.array(errs)
    rmse = float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))
    return dict(rmse=rmse,
                rmse_per_thruster=np.sqrt(np.mean(errs ** 2, axis=0)),
                rel=rmse / max(float(np.mean(u_norms)), 1e-9),
                max_abs=float(np.max(np.abs(errs))))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-configs", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-imitation", action="store_true",
                    help="skip the open-loop comparison (halves the oracle solves)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=str(PLOTS_DIR / "compare_to_oracle.png"))
    args = ap.parse_args()

    model, dtype = load_alpha_model(HERE / "anmpc_alpha" / "model.eqx",
                                    meta_path=HERE / "anmpc_alpha" / "model_meta.yaml")
    env = make_env(n_obs=model.n_obs, dt=model.dt)
    params = VesselParams()
    sim = VesselSimulator(params=params, dt=env.dt)
    rng = np.random.default_rng(args.seed)

    # Warm up the model's JIT BEFORE any timing, or the first scenario's mean is
    # dominated by XLA compilation (~2 s) instead of the ~1 ms steady-state cost.
    # Shapes/dtypes here match every later call, which is what keys the cache.
    t0 = time.perf_counter()
    import jax
    jax.block_until_ready(_model_call(
        model, jnp.zeros(7, dtype), jnp.asarray([0.2, 3.0, 0.6], dtype),
        jnp.full((env.n_obs, 2), 50.0, dtype), jnp.full((env.n_obs,), 0.5, dtype))[0])
    t_warm = time.perf_counter() - t0

    print(f"model : n_obs={env.n_obs}, dt={env.dt}s, r_ego={env.r_ego:.3f}m "
          f"(JIT warm-up {t_warm * 1e3:.0f} ms, done once at startup)")
    print(f"oracle: NMPC N=40, dt={env.dt}s, gains fixed at (0.5, 0.5), IPOPT")
    print(f"running {args.n_configs} scenario(s) x {args.n_steps} steps"
          f"{'' if args.no_imitation else ' (+ open-loop imitation pass)'}\n")

    rows, traj = [], []
    for c in range(args.n_configs):
        sc = sample_scenario(rng)
        p = sc["path"]
        print(f"[scenario {c}] y0={p.y0:.2f} u_ref={p.u_ref:.2f} "
              f"radii={[round(o.radius, 2) for o in sc['obstacles']]}")

        mpc = VesselNMPC(sc["obstacles"], params=params, path=p,
                         weights=CostWeights(), N=40, dt=env.dt,
                         default_gamma=(0.5, 0.5))

        st_or, m_or, n_failed = run_oracle(mpc, sc, sim, args.n_steps)
        st_md, m_md = run_model(model, dtype, env, sc, sim, args.n_steps)
        imi = (None if args.no_imitation
               else imitation_error(model, dtype, env, sc, st_or, mpc))

        if n_failed:
            print(f"  note: oracle failed to converge on {n_failed}/{args.n_steps} steps")
        print(f"  oracle: clear {m_or['min_clear']:6.3f} m  ye {m_or['ye_rms']:5.2f} m  "
              f"dX {m_or['dX']:5.1f} m  {m_or['t_ms']:7.2f} ms/step")
        print(f"  model : clear {m_md['min_clear']:6.3f} m  ye {m_md['ye_rms']:5.2f} m  "
              f"dX {m_md['dX']:5.1f} m  {m_md['t_ms']:7.2f} ms/step")
        if imi:
            print(f"  imitation RMSE (open loop): {imi['rmse']:.4f} N "
                  f"({100 * imi['rel']:.1f}% of mean |u_oracle|)")
        rows.append((c, sc, m_or, m_md, imi, n_failed))
        traj.append((sc, st_or, st_md))

    # ---------------- aggregate ------------------------------------------
    def agg(key, which):
        return np.mean([r[2 if which == "or" else 3][key] for r in rows])

    w = 74
    print("\n" + "=" * w)
    print(f"AGGREGATE over {len(rows)} scenario(s) x {args.n_steps} steps")
    print("=" * w)
    print(f"{'metric':<34}{'oracle':>13}{'model':>13}{'ratio':>13}")
    print("-" * w)
    for label, key, unit in [("min clearance (m, higher=safer)", "min_clear", ""),
                             ("cross-track RMS (m, lower=better)", "ye_rms", ""),
                             ("advance along +X (m)", "dX", ""),
                             ("control effort RMS (N)", "effort", ""),
                             ("mean solve time (ms)", "t_ms", ""),
                             ("worst-case solve time (ms)", "t_max_ms", "")]:
        a, b = agg(key, "or"), agg(key, "md")
        ratio = (f"{b / a:.2f}x" if abs(a) > 1e-12 else "--")
        print(f"{label:<34}{a:13.3f}{b:13.3f}{ratio:>13}")
    print("-" * w)
    speedup = agg("t_ms", "or") / max(agg("t_ms", "md"), 1e-9)
    print(f"{'SPEEDUP (mean solve time)':<34}{'':>13}{'':>13}{speedup:11.0f}x")
    print(f"\nFirst step of each rollout, excluded from the means above:")
    print(f"  oracle cold solve (no warm start) : {agg('t_first_ms', 'or'):8.1f} ms")
    print(f"  model  first call (already warm)  : {agg('t_first_ms', 'md'):8.1f} ms")
    if not args.no_imitation:
        rmse = np.mean([r[4]["rmse"] for r in rows])
        rel = np.mean([r[4]["rel"] for r in rows])
        mx = np.max([r[4]["max_abs"] for r in rows])
        per = np.mean([r[4]["rmse_per_thruster"] for r in rows], axis=0)
        print(f"\nOpen-loop imitation of the oracle's command:")
        print(f"  RMSE ||u_model - u_oracle||   : {rmse:.4f} N  "
              f"({100 * rel:.1f}% of mean |u_oracle|)")
        print(f"  per thruster [f1 f2 f3 f4]    : "
              f"[{', '.join(f'{v:.3f}' for v in per)}] N")
        print(f"  worst single-component error  : {mx:.3f} N "
              f"(f_max = {params.f_max} N)")
    n_fail_total = sum(r[5] for r in rows)
    print(f"\nOracle non-convergent steps: {n_fail_total}"
          f"/{len(rows) * args.n_steps}")
    print("Both controllers stayed collision-free."
          if min(min(r[2]["min_clear"], r[3]["min_clear"]) for r in rows) >= 0
          else "WARNING: a collision occurred -- see the per-scenario rows above.")

    if args.no_plot:
        return
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(traj, params, out_path)


def _plot(traj, params, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(traj)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    for ax, (sc, st_or, st_md) in zip(axes[:, 0], traj):
        p = sc["path"]
        s_grid = np.linspace(0, max(st_or[:, 6].max(), st_md[:, 6].max()) + 5, 500)
        ax.plot(*p.xy(s_grid), "k--", lw=1, label="reference")
        ax.plot(st_or[:, 0], st_or[:, 1], "-", color="tab:purple", lw=2.0,
                label="NMPC oracle (IPOPT)")
        ax.plot(st_md[:, 0], st_md[:, 1], "-", color="tab:blue", lw=1.6,
                label="amortized model")
        ax.plot(st_or[0, 0], st_or[0, 1], "go", ms=8, label="start")
        for o in sc["obstacles"]:
            ax.add_patch(plt.Circle((o.cx, o.cy), o.radius, color="tab:red", alpha=0.35))
            ax.add_patch(plt.Circle((o.cx, o.cy), o.radius + params.r_ego,
                                    color="tab:red", alpha=0.5, ls="--", lw=0.8,
                                    fill=False))
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_ylabel("Y [m]")
        ax.set_title(f"y0={p.y0:.2f}, u_ref={p.u_ref:.2f}", fontsize=9)
    axes[-1, 0].set_xlabel("X [m]")
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.suptitle("Amortized model vs the NMPC oracle it imitates "
                 "(same plant, same start, same obstacles)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved plot: {out_path}")


if __name__ == "__main__":
    main()
