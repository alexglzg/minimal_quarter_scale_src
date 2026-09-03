#!/usr/bin/env python3
"""Sample one scenario exactly the way compare_to_oracle.py does, and print
it as shell variable assignments for run_scenario.sh to eval.

    python3 sample_scenario.py --seed 0 --index 3

--seed picks the RNG stream; --index picks which draw from that stream to
keep (0-based) -- sample_scenario() is called index+1 times in sequence and
only the last call's result is used, so --seed 0 --index 0 is the same
scenario compare_to_oracle.py's default (--seed 0, first of --n-configs) uses,
--index 1 is its second scenario, and so on. This is what makes "seed 0,
index 3" reproducible and identical whether you run it in this repo's Python
comparison scripts or through run_scenario.sh in ROS.

Not meant to be run standalone for anything but inspecting a scenario's
numbers -- run_scenario.sh is the actual entry point.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anmpc_compare_to_oracle import sample_scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--index", type=int, default=0,
                    help="which draw from this seed's stream to keep (0-based)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sc = None
    for _ in range(args.index + 1):
        sc = sample_scenario(rng)

    p = sc["path"]
    obs = sc["obstacles"]
    # Single-line YAML flow style, not block style: roslaunch's own CLI
    # arg:=value parser silently drops everything after the first newline in
    # a multi-line value (see compare_run.launch's obstacles_yaml doc).
    yaml = "[" + ", ".join(
        f"{{name: buoy{j + 1}, radius: {o.radius!r}, mass: 10.0, "
        f"initial_state: [{o.cx!r}, {o.cy!r}, 0.0, 0.0]}}"
        for j, o in enumerate(obs)
    ) + "]"

    print(f"Y0={p.y0!r}")
    print(f"U_REF={p.u_ref!r}")
    print(f"NUM_OBSTACLES={len(obs)}")
    print(f"OBSTACLES_YAML='{yaml}'")


if __name__ == "__main__":
    main()
