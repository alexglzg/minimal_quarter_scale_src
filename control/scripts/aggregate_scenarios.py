#!/usr/bin/env python3
"""Aggregate closed-loop cost, collisions, and solve times across many
scenarios run via run_scenario.sh, for oracle vs anmpc vs barriernet.

    for i in $(seq 0 99); do ./run_scenario.sh --seed 0 --index "$i" --all; done
    python3 aggregate_scenarios.py --seed 0 --indices 0-99

Reuses compare_controllers.py's analyze() for each bag, so a scenario's
numbers here match exactly what compare_controllers.py --oracle_nominal ...
would report for it individually. Writes one CSV row per (scenario index,
controller) for your own further analysis, and prints an aggregate summary:
mean closed-loop cost, cost ratio vs oracle, collision rate, and solve-time
distribution (mean/p95/max/fraction over the 100ms budget), per controller.

--indices accepts "0-99", "0,3,7", or a mix like "0-9,20,30-35".
--condition selects nominal (default) or disturbed bags; disturbed runs are
unseeded (see run_scenario.sh), so each index's disturbed bag is one random
draw, not an average -- run each index's --disturbed multiple times and use
compare_controllers.py's --*_disturbed averaging (see its own docstring) if
you want a statistically honest robustness number per scenario. This script
aggregates ACROSS scenarios, not across repeats of the same one.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_controllers import analyze, CONTROLLERS, CONTROL_PERIOD_MS, PARAMS_YAML  # noqa: E402
from anmpc_compare_to_oracle import sample_scenario  # noqa: E402


def parse_indices(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def scenario_params(seed, index):
    """Same y0/u_ref sample_scenario.py would print for this seed/index --
    re-derived here rather than stored, since the RNG draw is deterministic."""
    rng = np.random.default_rng(seed)
    sc = None
    for _ in range(index + 1):
        sc = sample_scenario(rng)
    return sc["path"].y0, sc["path"].u_ref


def find_bag(bag_dir, scenario_id, controller, condition):
    matches = sorted(glob.glob(
        os.path.join(bag_dir, f"scenario{scenario_id}_{controller}_{condition}_*.bag")))
    return matches[-1] if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--bag_dir", default=os.path.expanduser("~/compare_bags"))
    ap.add_argument("--condition", default="nominal", choices=["nominal", "disturbed"])
    ap.add_argument("--params", default=PARAMS_YAML)
    ap.add_argument("--out", default="aggregate_scenarios.csv")
    args = ap.parse_args()

    with open(args.params) as fh:
        p = yaml.safe_load(fh)
    weights = (p["parameters_mpc"]["Qye"], p["parameters_mpc"]["Qr"],
              p["parameters_mpc"]["Qpsi"], p["parameters_mpc"]["Qu"])
    r_f = 1.0
    r_ego = float(np.hypot(p["parameters_model"].get("ego_length", 0.9),
                           p["parameters_model"].get("ego_width", 0.45)) / 2.0)

    indices = parse_indices(args.indices)
    rows, missing = [], []
    for idx in indices:
        scenario_id = f"{args.seed}_{idx}"
        y0, u_ref = scenario_params(args.seed, idx)
        path_vec = np.array([p["path"]["x_multiplier"], y0, u_ref])
        for ctrl in CONTROLLERS:
            bag = find_bag(args.bag_dir, scenario_id, ctrl, args.condition)
            if bag is None:
                missing.append((idx, ctrl))
                continue
            r = analyze(ctrl, bag, path_vec, weights, r_f, r_ego)
            solve_ms = r["solve_ms"]
            rows.append(dict(
                index=idx, controller=ctrl, y0=y0, u_ref=u_ref,
                closed_loop_cost=float(np.sum(r["cost"])) if len(r["cost"]) else float("nan"),
                min_clearance=r["min_clearance"], effort_rms=r["effort_rms"],
                collision=r["collision"], n_ticks=r["n_ticks"],
                solve_ms_mean=float(np.mean(solve_ms)) if len(solve_ms) else float("nan"),
                solve_ms_p95=float(np.percentile(solve_ms, 95)) if len(solve_ms) else float("nan"),
                solve_ms_max=float(np.max(solve_ms)) if len(solve_ms) else float("nan"),
                over_budget_frac=float(np.mean(solve_ms > CONTROL_PERIOD_MS)) if len(solve_ms) else float("nan"),
            ))

    if missing:
        print(f"warning: {len(missing)} scenario/controller bag(s) not found, skipped:")
        for idx, ctrl in missing[:20]:
            print(f"  scenario {idx} / {ctrl}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if not rows:
        print("no bags found -- nothing to aggregate")
        return

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")

    by_ctrl = {c: [r for r in rows if r["controller"] == c] for c in CONTROLLERS}

    w = 78
    print("\n" + "=" * w)
    print(f"AGGREGATE over {len(set(r['index'] for r in rows))} scenario(s), condition={args.condition}")
    print("=" * w)
    print(f"{'metric':<30}{'oracle':>14}{'anmpc':>14}{'barriernet':>14}")
    print("-" * w)
    for label, key, fmt in [
        ("closed-loop cost (mean)", "closed_loop_cost", "{:.0f}"),
        ("min clearance (mean)", "min_clearance", "{:.3f}"),
        ("control effort RMS (mean)", "effort_rms", "{:.2f}"),
        ("solve time mean [ms]", "solve_ms_mean", "{:.2f}"),
        ("solve time p95 [ms]", "solve_ms_p95", "{:.2f}"),
        ("solve time max [ms]", "solve_ms_max", "{:.2f}"),
        ("fraction ticks > 100ms budget", "over_budget_frac", "{:.1%}"),
    ]:
        vals = []
        for c in CONTROLLERS:
            xs = [r[key] for r in by_ctrl.get(c, []) if r[key] == r[key]]
            vals.append(fmt.format(np.mean(xs)) if xs else "n/a")
        print(f"{label:<30}" + "".join(f"{v:>14}" for v in vals))
    print("-" * w)
    for c in CONTROLLERS:
        col = by_ctrl.get(c, [])
        n_col = sum(1 for r in col if r["collision"])
        line = f"collisions: {c}"
        print(f"{line:<30}{n_col}/{len(col)} scenarios ({n_col / len(col):.1%})"
             if col else f"{line:<30}n/a")

    oracle_cost = {r["index"]: r["closed_loop_cost"] for r in by_ctrl.get("oracle", [])}
    if oracle_cost:
        print("\ncost ratio vs oracle (per scenario, then averaged; lower = closer to oracle):")
        for c in ("anmpc", "barriernet"):
            ratios = [r["closed_loop_cost"] / oracle_cost[r["index"]]
                     for r in by_ctrl.get(c, [])
                     if r["index"] in oracle_cost and oracle_cost[r["index"]] > 0
                     and r["closed_loop_cost"] == r["closed_loop_cost"]]
            if ratios:
                ratios = np.array(ratios)
                print(f"  {c:<12} mean={ratios.mean():.2f}x  median={np.median(ratios):.2f}x  "
                     f"n={len(ratios)}  worse-than-oracle-in={int((ratios > 1).sum())}/{len(ratios)}")

    a_cost = {r["index"]: r["closed_loop_cost"] for r in by_ctrl.get("anmpc", [])}
    b_cost = {r["index"]: r["closed_loop_cost"] for r in by_ctrl.get("barriernet", [])}
    common = sorted(set(a_cost) & set(b_cost))
    if common:
        anmpc_wins = sum(1 for i in common if a_cost[i] < b_cost[i])
        print(f"\nanmpc closer to oracle in {anmpc_wins}/{len(common)} scenarios where both ran; "
             f"barriernet in {len(common) - anmpc_wins}/{len(common)}")


if __name__ == "__main__":
    main()
