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
--ood aggregates the 5 fixed out-of-distribution scenarios from
ood_scenarios.py instead of a seeded/index range (run via
run_ood_scenarios.sh) -- --seed/--indices are ignored/not required with --ood.
--condition selects nominal (default) or disturbed bags. disturbed runs are
unseeded (see run_scenario.sh), so each index's disturbed bags are independent
random draws (run_all_scenarios.sh records --n_disturbed of them per scenario,
named scenario<seed>_<idx>_<ctrl>_disturbed_<timestamp>.bag). For
--condition disturbed, this script finds *every* matching bag per
(scenario, controller) -- not just the latest -- and averages/pools them into
one row per scenario the same way compare_controllers.py's report_robustness
does for a single scenario, before aggregating that row ACROSS scenarios. So
the printed summary is a robustness number averaged over disturbance
realizations *and* over the scenario set. --condition nominal keeps the old
one-bag-per-scenario behavior since nominal runs are seeded/deterministic.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_controllers import analyze, CONTROLLERS, COLORS, CONTROL_PERIOD_MS, PARAMS_YAML, PLOTS_DIR  # noqa: E402
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


def find_bags(bag_dir, scenario_id, controller, condition):
    """Every matching bag, not just the latest -- for --condition disturbed,
    each is an independent unseeded realization (see run_all_scenarios.sh's
    --n_disturbed) and all of them should be averaged over, not just one."""
    return sorted(glob.glob(
        os.path.join(bag_dir, f"scenario{scenario_id}_{controller}_{condition}_*.bag")))


def aggregate_runs(analyses):
    """Collapse one or more analyze() results (multiple only for disturbed
    repeats) into a single per-scenario row, pooling solve_ms across runs for
    the solve-time stats (more samples => better tail estimate) and averaging
    the per-run scalars otherwise -- same scheme as compare_controllers.py's
    report_robustness, generalized to more than one metric group at once."""
    costs = [float(np.sum(r["cost"])) if len(r["cost"]) else float("nan") for r in analyses]
    clearances = [r["min_clearance"] for r in analyses]
    efforts = [r["effort_rms"] for r in analyses]
    collisions = [1.0 if r["collision"] else 0.0 for r in analyses]
    n_ticks = [r["n_ticks"] for r in analyses]
    solve_ms = np.concatenate([np.asarray(r["solve_ms"], float) for r in analyses]) \
        if analyses else np.array([])

    def nanmean(xs):
        xs = np.asarray(xs, float)
        xs = xs[~np.isnan(xs)]
        return float(np.mean(xs)) if len(xs) else float("nan")

    def nanstd(xs):
        xs = np.asarray(xs, float)
        xs = xs[~np.isnan(xs)]
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    return dict(
        n_runs=len(analyses),
        closed_loop_cost=nanmean(costs),
        closed_loop_cost_std=nanstd(costs),
        min_clearance=nanmean(clearances),
        effort_rms=nanmean(efforts),
        collision=nanmean(collisions),
        n_ticks=nanmean(n_ticks),
        solve_ms_mean=float(np.mean(solve_ms)) if len(solve_ms) else float("nan"),
        solve_ms_p95=float(np.percentile(solve_ms, 95)) if len(solve_ms) else float("nan"),
        solve_ms_max=float(np.max(solve_ms)) if len(solve_ms) else float("nan"),
        over_budget_frac=float(np.mean(solve_ms > CONTROL_PERIOD_MS)) if len(solve_ms) else float("nan"),
    )


def make_boxplots(rows, cost_by_ctrl, solve_ms_by_ctrl, condition, run_label, out_path):
    """Two boxplots, side by side: closed-loop cost and solve time, each
    pooling every individual run (one point per bag -- 3 disturbed
    realizations per scenario go in as 3 separate points, not pre-averaged
    into one) across all scenarios/repeats, so the box reflects the full
    run-to-run spread including within-scenario disturbance variance, not
    just per-scenario means. Same per-controller boxplot() pattern as
    compare_controllers.py's single-scenario ax_box, one call per controller
    since matplotlib chokes on differently-sized arrays passed together."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping boxplots")
        return

    cost_names = [c for c in CONTROLLERS if len(cost_by_ctrl.get(c, []))]
    solve_names = [c for c in CONTROLLERS if len(solve_ms_by_ctrl.get(c, []))]
    if not cost_names and not solve_names:
        return

    fig, (ax_cost, ax_solve) = plt.subplots(1, 2, figsize=(12, 5))

    for i, c in enumerate(cost_names):
        vals = [v for v in cost_by_ctrl[c] if v == v]
        bp = ax_cost.boxplot(vals, positions=[i], patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(COLORS[c]); bp["boxes"][0].set_alpha(0.5)
        bp["medians"][0].set_color("black")
    ax_cost.set_xticks(range(len(cost_names))); ax_cost.set_xticklabels(cost_names)
    ax_cost.grid(alpha=0.3, axis="y")
    ax_cost.set_ylabel("closed-loop cost")
    n_scen = len(set(r["index"] for r in rows))
    ax_cost.set_title(f"Closed-loop cost across all runs, {n_scen} scenarios ({condition})")

    for i, c in enumerate(solve_names):
        bp = ax_solve.boxplot(solve_ms_by_ctrl[c], positions=[i], patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(COLORS[c]); bp["boxes"][0].set_alpha(0.5)
        bp["medians"][0].set_color("black")
    ax_solve.set_xticks(range(len(solve_names))); ax_solve.set_xticklabels(solve_names)
    ax_solve.axhline(CONTROL_PERIOD_MS, color="k", ls="--", lw=1,
                     label=f"{CONTROL_PERIOD_MS:.0f}ms budget")
    ax_solve.set_yscale("log"); ax_solve.grid(alpha=0.3, axis="y"); ax_solve.legend(fontsize=8)
    ax_solve.set_ylabel("solve time [ms]")
    ax_solve.set_title(f"Solve time across all ticks, {n_scen} scenarios ({condition})")

    fig.suptitle(f"{run_label}, condition={condition}")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--indices", default=None)
    ap.add_argument("--ood", action="store_true",
                    help="aggregate the 5 fixed OOD scenarios from ood_scenarios.py "
                         "instead of a seeded/index range -- ignores --seed/--indices")
    ap.add_argument("--ood_random", default=None, choices=["radius_big", "spacing_tight"],
                    help="aggregate the 10 randomized scenarios for this family from "
                         "ood_random_scenarios.py instead of a seeded/index range -- "
                         "ignores --seed/--indices; run via run_ood_random_scenarios.sh "
                         "(nominal only)")
    ap.add_argument("--bag_dir", default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "..", "compare_bags"))
    ap.add_argument("--condition", default="nominal", choices=["nominal", "disturbed"])
    ap.add_argument("--params", default=PARAMS_YAML)
    ap.add_argument("--out", default=None,
                    help="CSV output path (default: aggregate_scenarios_ood_<condition>.csv "
                         "with --ood, else aggregate_scenarios.csv)")
    ap.add_argument("--plot_out", default=None,
                    help="boxplot PNG path (default: "
                         "<repo>/plots/comparisons/aggregate_scenarios_<ood|seed#>_<condition>.png); "
                         "pass --no_plot to skip")
    ap.add_argument("--no_plot", action="store_true", help="skip generating the boxplot PNG")
    args = ap.parse_args()
    if not args.ood and not args.ood_random and (args.seed is None or args.indices is None):
        ap.error("--seed and --indices are required unless --ood or --ood_random is given")
    if args.out is None:
        if args.ood:
            args.out = f"aggregate_scenarios_ood_{args.condition}.csv"
        elif args.ood_random:
            args.out = f"aggregate_scenarios_oodrand_{args.ood_random}_{args.condition}.csv"
        else:
            args.out = "aggregate_scenarios.csv"

    with open(args.params) as fh:
        p = yaml.safe_load(fh)
    weights = (p["parameters_mpc"]["Qye"], p["parameters_mpc"]["Qr"],
              p["parameters_mpc"]["Qpsi"], p["parameters_mpc"]["Qu"])
    r_f = 1.0
    r_ego = float(np.hypot(p["parameters_model"].get("ego_length", 0.9),
                           p["parameters_model"].get("ego_width", 0.45)) / 2.0)

    if args.ood:
        from ood_scenarios import SCENARIOS as OOD_SCENARIOS
        scenario_list = [(sid, sc["y0"], sc["u_ref"]) for sid, sc in OOD_SCENARIOS.items()]
    elif args.ood_random:
        from ood_random_scenarios import scenario as ood_random_scenario, N_PER_FAMILY
        scenario_list = []
        for i in range(N_PER_FAMILY):
            sc = ood_random_scenario(args.ood_random, i)
            scenario_list.append((sc["scenario_id"], sc["y0"], sc["u_ref"]))
    else:
        scenario_list = [(f"{args.seed}_{idx}", *scenario_params(args.seed, idx))
                         for idx in parse_indices(args.indices)]

    rows, missing, run_counts = [], [], []
    solve_ms_by_ctrl = {c: [] for c in CONTROLLERS}
    cost_by_ctrl = {c: [] for c in CONTROLLERS}
    for scenario_id, y0, u_ref in scenario_list:
        path_vec = np.array([p["path"]["x_multiplier"], y0, u_ref])
        for ctrl in CONTROLLERS:
            if args.condition == "disturbed":
                bags = find_bags(args.bag_dir, scenario_id, ctrl, args.condition)
            else:
                bag = find_bag(args.bag_dir, scenario_id, ctrl, args.condition)
                bags = [bag] if bag else []
            if not bags:
                missing.append((scenario_id, ctrl))
                continue
            analyses = [analyze(ctrl, b, path_vec, weights, r_f, r_ego) for b in bags]
            run_counts.append((scenario_id, ctrl, len(analyses)))
            rows.append(dict(index=scenario_id, controller=ctrl, y0=y0, u_ref=u_ref,
                             **aggregate_runs(analyses)))
            for a in analyses:
                if len(a["solve_ms"]):
                    solve_ms_by_ctrl[ctrl].append(np.asarray(a["solve_ms"], float))
                cost_by_ctrl[ctrl].append(
                    float(np.sum(a["cost"])) if len(a["cost"]) else float("nan"))

    if args.condition == "disturbed" and run_counts:
        counts = [n for _, _, n in run_counts]
        if len(set(counts)) > 1:
            print(f"warning: uneven number of disturbed realizations per scenario/controller "
                 f"(min={min(counts)}, max={max(counts)}) -- some rows are averaged over fewer "
                 f"repeats than others:")
            for sid, ctrl, n in run_counts:
                if n != max(counts):
                    print(f"  scenario {sid} / {ctrl}: {n} run(s)")

    if missing:
        print(f"warning: {len(missing)} scenario/controller bag(s) not found, skipped:")
        for sid, ctrl in missing[:20]:
            print(f"  scenario {sid} / {ctrl}")
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
    if args.condition == "disturbed":
        n_runs = sorted(set(r["n_runs"] for r in rows))
        label = f"{n_runs[0]} disturbed realizations/scenario" if len(n_runs) == 1 \
            else f"{n_runs[0]}-{n_runs[-1]} disturbed realizations/scenario (uneven, see warning above)"
        print(f"averaged over {label}")
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
        line = f"collisions: {c}"
        if not col:
            print(f"{line:<30}n/a")
            continue
        # r["collision"] is a rate in [0,1]: the fraction of runs behind that
        # row that collided (exactly 0 or 1 for nominal, since n_runs==1).
        n_any = sum(1 for r in col if r["collision"] > 0)
        avg_rate = np.mean([r["collision"] for r in col])
        if args.condition == "disturbed":
            print(f"{line:<30}avg rate {avg_rate:.1%} across realizations  "
                 f"(any collision in {n_any}/{len(col)} scenarios)")
        else:
            print(f"{line:<30}{n_any}/{len(col)} scenarios ({avg_rate:.1%})")

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

    if not args.no_plot:
        pooled_solve_ms = {c: np.concatenate(arrs) if arrs else np.array([])
                          for c, arrs in solve_ms_by_ctrl.items()}
        if args.ood:
            run_label = "ood"
        elif args.ood_random:
            run_label = f"oodrand_{args.ood_random}"
        else:
            run_label = f"seed{args.seed}"
        plot_out = args.plot_out or os.path.join(
            PLOTS_DIR, f"aggregate_scenarios_{run_label}_{args.condition}.png")
        make_boxplots(rows, cost_by_ctrl, pooled_solve_ms, args.condition, run_label, plot_out)


if __name__ == "__main__":
    main()
