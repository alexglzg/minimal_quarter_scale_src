#!/usr/bin/env python3
"""
Standalone test harness for MPCController (mpc_opti_chocbf_dcbf_core.py) -- no ROS needed.

Loads control/config/parameters.yaml, builds the controller, and either:
  - runs a short closed-loop simulation for one config (default), or
  - sweeps every cbf_type x adaptive x solver combination and reports pass/fail
    (useful after touching the OCP structure, e.g. for FATROP compatibility).

Examples:
  python3 mpc_opti_standalone_test.py
  python3 mpc_opti_standalone_test.py --solver fatrop --cbf-type dcbf --adaptive --steps 10
  python3 mpc_opti_standalone_test.py --sweep
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from path import SinePath, StraightLinePath
from mpc_opti_chocbf_dcbf_core import MPCController

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "parameters.yaml"
)


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["parameters_model"], cfg["parameters_mpc"], cfg["path"]


def build_path(path_p):
    if path_p["type"] == "sine":
        return SinePath(path_p["x_multiplier"], path_p["y_offset"])
    elif path_p["type"] == "straight_line":
        return StraightLinePath(path_p["slope"], path_p["intercept"])
    raise ValueError(f"Unsupported path type {path_p['type']!r}")


def rk4_step(model_p, dt, state, u):
    m11, m22, m33 = model_p["m11"], model_p["m22"], model_p["m33"]
    d11, d22, d33 = model_p["d11"], model_p["d22"], model_p["d33"]
    aa, bb = model_p["aa"], model_p["bb"]

    def f(x, u):
        psi, surge, sway, r = x[2], x[3], x[4], x[5]
        u1, u2, u3, u4 = u
        return np.array([
            surge * np.cos(psi) - sway * np.sin(psi),
            surge * np.sin(psi) + sway * np.cos(psi),
            r,
            -d11 / m11 * surge + u1 / m11 + u2 / m11,
            -d22 / m22 * sway + u3 / m22 + u4 / m22,
            -d33 / m33 * r + aa / (2 * m33) * u1 - aa / (2 * m33) * u2
                + bb / (2 * m33) * u3 - bb / (2 * m33) * u4,
            surge,
        ])

    k1 = f(state, u)
    k2 = f(state + dt / 2 * k1, u)
    k3 = f(state + dt / 2 * k2, u)
    k4 = f(state + dt * k3, u)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_once(model_p, mpc_p, path, steps, initial_state, obstacles, alpha1, alpha2, verbose=True):
    mpc = MPCController(model_p, mpc_p, path)

    state = np.array(initial_state, dtype=float)
    obstacle_x = np.array([o[0] for o in obstacles])
    obstacle_y = np.array([o[1] for o in obstacles])
    obstacle_radius = np.array([o[2] for o in obstacles])
    alpha1 = np.full(mpc_p["num_obstacles"], alpha1)
    alpha2 = np.full(mpc_p["num_obstacles"], alpha2)

    ig_state = np.tile(state.reshape(-1, 1), (1, mpc.Nhor + 1))
    ig_control = np.zeros((mpc.nu, mpc.Nhor))

    for step in range(steps):
        t0 = time.time()
        u, U, X = mpc.solve(state, obstacle_x, obstacle_y, obstacle_radius,
                             alpha1, alpha2, ig_state, ig_control)
        dt_ms = (time.time() - t0) * 1000
        if verbose:
            print(f"  step {step}: solve={dt_ms:6.1f}ms  u={np.round(u, 3)}  "
                  f"state[:3]={np.round(state[:3], 3)}")
        ig_state, ig_control = np.array(X), np.array(U)
        state = rk4_step(model_p, mpc.dt, state, u)

    return mpc


def sweep(model_p, mpc_p, path):
    cbf_types = ["none", "c_hocbf", "dcbf", "combo_chocbf_dcbf"]
    print(f"{'solver':8s} {'cbf_type':20s} {'adaptive':9s}  result")
    print("-" * 55)
    for solver in ["ipopt", "fatrop"]:
        for cbf_type in cbf_types:
            for adaptive in (False, True):
                p = dict(mpc_p)
                p["cbf_type"] = cbf_type
                p["adaptive"] = adaptive
                p["solver"] = solver
                try:
                    run_once(model_p, p, path, steps=1,
                             initial_state=np.zeros(p["nx"]),
                             obstacles=[(p["dummy_x"], p["dummy_y"], p["dummy_radius"])] * p["num_obstacles"],
                             alpha1=1.0, alpha2=1.0, verbose=False)
                    print(f"{solver:8s} {cbf_type:20s} {str(adaptive):9s}  OK")
                except Exception as e:
                    print(f"{solver:8s} {cbf_type:20s} {str(adaptive):9s}  FAIL: {type(e).__name__}: {str(e).splitlines()[0]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["ipopt", "fatrop"], default=None)
    parser.add_argument("--cbf-type", choices=["none", "c_hocbf", "dcbf", "combo_chocbf_dcbf"], default=None)
    parser.add_argument("--adaptive", action="store_true", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sweep", action="store_true", help="try every cbf_type x adaptive x solver combo")
    args = parser.parse_args()

    model_p, mpc_p, path_p = load_config()
    path = build_path(path_p)

    if args.sweep:
        sweep(model_p, mpc_p, path)
        return

    if args.solver is not None:
        mpc_p["solver"] = args.solver
    if args.cbf_type is not None:
        mpc_p["cbf_type"] = args.cbf_type
    if args.adaptive is not None:
        mpc_p["adaptive"] = args.adaptive

    print(f"solver={mpc_p.get('solver')} cbf_type={mpc_p.get('cbf_type')} adaptive={mpc_p.get('adaptive')}")
    obstacles = [(mpc_p["dummy_x"], mpc_p["dummy_y"], mpc_p["dummy_radius"])] * mpc_p["num_obstacles"]
    run_once(model_p, mpc_p, path, steps=args.steps,
             initial_state=np.zeros(mpc_p["nx"]), obstacles=obstacles,
             alpha1=1.0, alpha2=1.0)


if __name__ == "__main__":
    main()
