#!/usr/bin/env python3
"""Post-hoc comparison of the oracle / amortized-NMPC (anmpc) / BarrierNet
controllers, from rosbags recorded by control/launch/compare_run.launch.

Recipe
------
Each controller drives /mpc_force, so only one can run at a time; get a fair
comparison by running each in the same deterministic scenario (fixed start
pose, fixed 6 static buoys) and recording a bag, then compare offline. Run
the nominal trials:

    roscore &
    roslaunch control compare_run.launch controller:=oracle     duration:=90
    roslaunch control compare_run.launch controller:=anmpc      duration:=90
    roslaunch control compare_run.launch controller:=barriernet duration:=90

and, for the robustness comparison (point 4 below), one or more disturbed runs
per controller -- currents.cpp/wind_and_waves.cpp both reseed from
std::random_device on every process start (not a fixed seed), so each
disturbed:=true run is a different realization; run it several times per
controller to average over rather than draw a conclusion from one:

    roslaunch control compare_run.launch controller:=oracle     duration:=90 disturbed:=true
    roslaunch control compare_run.launch controller:=oracle     duration:=90 disturbed:=true
    roslaunch control compare_run.launch controller:=anmpc      duration:=90 disturbed:=true
    roslaunch control compare_run.launch controller:=barriernet duration:=90 disturbed:=true
    ... (repeat disturbed:=true runs as many times as you want per controller)

Each roslaunch shuts itself down (sim + buoys + controller + recorder) the
moment its bag hits `duration` seconds, so all runs can be done back to back.
Then:

    python3 compare_controllers.py --bag_dir ~/compare_bags

--bag_dir auto-discovers each controller's most recent *nominal* bag by
filename (oracle_nominal_*.bag, ..., from `-o` in the launch file), and ALL
matching *disturbed* bags (oracle_disturbed_*.bag, however many there are) --
point 4 reports the mean +/- std over however many disturbed runs it finds.
Pass explicit paths with e.g. --oracle_nominal for a specific nominal run, or
repeat --oracle_disturbed PATH for an explicit set of disturbed runs instead
of auto-discovering. Any subset of controllers/conditions may be present --
sections that need a bag that isn't there are skipped with a note, not an error.

For a run_scenario.sh scenario (bags named scenario{seed}_{index}_*.bag), pass
--seed/--index instead of --y0/--u_ref/--bag_dir bookkeeping: y0/u_ref are
re-derived the same way sample_scenario.py / aggregate_scenarios.py do, and
bag auto-discovery is scoped to that scenario's own bags:

    python3 compare_controllers.py --seed 0 --index 0

Point 2's methodology mirrors barriernet_ampc_compare_models.ipynb's
rollout() comparison as closely as the switch from an open-loop numpy rollout
to real ROS bags allows: one cost sample per *control tick* (not per, faster,
odometry sample), "closed-loop cost" is the raw sum of per-tick stage cost
over the window (same `cost=float(np.sum(costs))` the notebook uses, so the
numbers are the same kind of thing), and the headline result is a cost ratio
against a reference controller -- the notebook's `cost ratio: X.XXx` line,
just with the oracle as the reference now instead of the amortized model.

What this answers
------------------
1. Oracle computation time: full solve-time distribution against the 100 ms
   control-period budget (CONTROL_PERIOD_MS below, same threshold
   mpc_status_dashboard.py plots), from the nominal oracle bag.
2. Whether the oracle converges, and how closely anmpc/barriernet approximate
   its closed-loop cost over a full run -- notebook-style table + cost ratio,
   oracle as the reference.
3. Robustness: for each controller with a nominal bag and at least one
   disturbed (wind+waves+currents) bag, how much success rate / cost /
   clearance degrade between the nominal run and the mean over however many
   disturbed runs were given.
"""
import argparse
import contextlib
import glob
import io
import os
import sys

import numpy as np
import yaml

import rosbag
from tf.transformations import euler_from_quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anmpc_alpha.anmpc_alpha import stage_residual, anchor_arc_length  # noqa: E402
from anmpc_compare_to_oracle import sample_scenario  # noqa: E402

PARAMS_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "parameters.yaml")

# All three controllers together -> a comparison, not any one controller's own plot.
PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots", "comparisons")

CONTROL_PERIOD_MS = 100.0  # 10 Hz control loop budget, same as mpc_status_dashboard.py

CONTROLLERS = ("oracle", "anmpc", "barriernet")
STATUS_NS = {"oracle": "mpc_status_oracle", "chocbf": "mpc_status",
            "anmpc": "mpc_status_anmpc", "barriernet": "mpc_status_barriernet"}
COLORS = {"oracle": "tab:green", "chocbf": "tab:red",
         "anmpc": "tab:blue", "barriernet": "tab:orange"}


# ---------------------------------------------------------------------------
# Bag extraction
# ---------------------------------------------------------------------------
def _read_series(bag, topic, field):
    """[(t, value)] for a single-field std_msgs topic (Float64.data / Bool.data)."""
    return [(t.to_sec(), getattr(msg, field)) for _, msg, t in bag.read_messages(topics=[topic])]


def _read_odometry(bag):
    """/odometry/filtered -> [(t, [X, Y, psi, su, sv, sr])] in the NED convention the
    controllers themselves use (same ENU->NED transform as mpc_node.odom_to_state)."""
    out = []
    for _, msg, t in bag.read_messages(topics=["/odometry/filtered"]):
        x = msg.pose.pose.position.x
        y = -msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        su = msg.twist.twist.linear.x
        sv = -msg.twist.twist.linear.y
        sr = -msg.twist.twist.angular.z
        out.append((t.to_sec(), np.array([x, y, -yaw, su, sv, sr])))
    return out


def _read_force(bag):
    return [(t.to_sec(), np.array(msg.data)) for _, msg, t in bag.read_messages(topics=["/mpc_force"])]


def _read_multiarray(bag, topic):
    """[(t, np.array)] for any Float64MultiArray-shaped topic (.data field) --
    same shape as _read_force, generalized for e.g. /mpc_status_barriernet/u_nom.
    Returns [] for a topic absent from this bag (older bags predating a given
    publisher), same as any other missing-topic read here."""
    return [(t.to_sec(), np.array(msg.data)) for _, msg, t in bag.read_messages(topics=[topic])]


def _read_buoys(bag):
    """(times, centers_list, radii_list) -- every /buoy_array message, in NED.
    centers_list[i]/radii_list[i] are the (n,2)/(n,) arrays at times[i]. Buoys
    are simulated the same way the vessel is, so under disturbed:=true they
    drift under wind/wave/current forces too (observed ~2.3m over a 40s run)
    -- a single static snapshot isn't enough for a clearance metric that stays
    meaningful for the whole run; see _read_first_buoys' old docstring/callers
    for the bug this replaced."""
    times, centers_list, radii_list = [], [], []
    for _, msg, t in bag.read_messages(topics=["/buoy_array"]):
        centers = np.array([[b.odom.pose.pose.position.x, -b.odom.pose.pose.position.y]
                            for b in msg.buoys])
        radii = np.array([b.radius for b in msg.buoys])
        times.append(t.to_sec())
        centers_list.append(centers)
        radii_list.append(radii)
    return times, centers_list, radii_list


def _nearest_value(times, values, query_times):
    """values[i] whose times[i] is closest to each query time -- for pairing series
    logged at different, uncorrelated rates (e.g. odometry vs /mpc_force)."""
    if len(times) == 0:
        return [None] * len(query_times)
    times = np.asarray(times)
    idx = np.searchsorted(times, query_times)
    idx = np.clip(idx, 1, len(times) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(query_times - times[left]) <= np.abs(times[right] - query_times)
    idx = np.where(use_left, left, right)
    return [values[i] for i in idx]


# ---------------------------------------------------------------------------
# Per-bag analysis
# ---------------------------------------------------------------------------
def analyze(name, bag_path, path_vec, weights, r_f, r_ego):
    """One cost/clearance/effort sample per *control tick* (not per odometry
    sample), matching the notebook rollout()'s per-step granularity: at each
    of this controller's own solve_time_ms timestamps, take the nearest
    odometry state and the nearest commanded force, then evaluate stage_residual
    there. Odometry itself (`states`/`odom_t`) is kept separately, at its own
    native rate, only for the trajectory/speed plots.
    """
    ns = STATUS_NS[name]

    def unzip(records):
        return tuple(zip(*records)) if records else ((), ())

    with rosbag.Bag(bag_path) as bag:
        solve_t, solve_ms = unzip(_read_series(bag, f"/{ns}/solve_time_ms", "data"))
        _, success = unzip(_read_series(bag, f"/{ns}/success", "data"))
        odom = _read_odometry(bag)
        force = _read_force(bag)
        # barriernet-only: its raw (pre-CBF-projection) nominal command, so the
        # CBF's correction direction is visible, not just its magnitude
        # (/mpc_status_barriernet/intervention_n). [] for anything else, or for
        # a barriernet bag recorded before this publisher existed.
        u_nom_series = _read_multiarray(bag, f"/{ns}/u_nom") if name == "barriernet" else []
        buoy_t, buoy_centers, buoy_radii = _read_buoys(bag)
        collided = any(v for _, v in _read_series(bag, "/collision_detected", "data"))

    centers = buoy_centers[0] if buoy_centers else np.zeros((0, 2))
    radii = buoy_radii[0] if buoy_radii else np.zeros((0,))

    solve_t = np.asarray(solve_t, float)
    solve_ms = np.asarray(solve_ms, float)
    success = np.asarray(success, bool)

    odom_t_raw = np.array([t for t, _ in odom])
    odom_x = np.array([x for _, x in odom]) if odom else np.zeros((0, 6))
    t0 = odom_t_raw[0] if len(odom_t_raw) else (solve_t[0] if len(solve_t) else 0.0)
    odom_t = odom_t_raw - t0 if len(odom_t_raw) else odom_t_raw

    force_t = [t for t, _ in force]
    force_u = [u for _, u in force]
    u_nom_t = [t for t, _ in u_nom_series]
    u_nom_v = [u for _, u in u_nom_series]

    tick_t = solve_t if len(solve_t) else odom_t_raw
    x_at_tick = (_nearest_value(odom_t_raw, list(odom_x), tick_t)
                if len(odom_t_raw) else [np.zeros(6)] * len(tick_t))
    u_at_tick = (_nearest_value(force_t, force_u, tick_t)
                if force_t else [np.zeros(4)] * len(tick_t))
    u_nom_at_tick = (_nearest_value(u_nom_t, u_nom_v, tick_t)
                    if u_nom_t else None)

    # Buoy snapshot nearest each tick, not a single static one -- see _read_buoys.
    buoy_idx_at_tick = (_nearest_value(buoy_t, list(range(len(buoy_t))), tick_t)
                       if buoy_t else [None] * len(tick_t))

    # Closed-loop path-following cost, one sample per control tick: anchor s
    # sequentially in tick order (same scheme the controllers themselves use),
    # then stage_residual + control effort at that tick's state/command.
    s_grid, cost, clearance, xy_at_tick = [], [], [], []
    s = 0.0
    for x6, u, bi in zip(x_at_tick, u_at_tick, buoy_idx_at_tick):
        s = anchor_arc_length(path_vec, x6, s)
        s_grid.append(s)
        x7 = np.concatenate([x6, [s]])
        r = np.asarray(stage_residual(x7, path_vec, weights))
        cost.append(float(r @ r + r_f * np.dot(u, u)))
        c, rad = (buoy_centers[bi], buoy_radii[bi]) if bi is not None else (centers, radii)
        clearance.append(np.min(np.linalg.norm(c - x6[:2], axis=1) - (rad + r_ego))
                         if len(c) else float("nan"))
        xy_at_tick.append(x6[:2])
    s_grid = np.array(s_grid)
    cost = np.array(cost)
    clearance = np.array(clearance)
    xy_at_tick = np.array(xy_at_tick) if xy_at_tick else np.zeros((0, 2))
    u_at_tick = np.array(u_at_tick) if len(u_at_tick) else np.zeros((0, 4))
    u_nom_at_tick = np.array(u_nom_at_tick) if u_nom_at_tick is not None else None
    tick_t_rel = tick_t - t0 if len(tick_t) else tick_t

    clr_valid = clearance[~np.isnan(clearance)] if len(clearance) else clearance
    return dict(
        name=name, n_ticks=len(cost),
        success_rate=float(success.mean()) if len(success) else float("nan"),
        solve_ms=solve_ms,
        cost_mean=float(np.mean(cost)) if len(cost) else float("nan"),
        min_clearance=float(clr_valid.min()) if len(clr_valid) else float("nan"),
        effort_rms=float(np.sqrt(np.mean(np.sum(u_at_tick ** 2, axis=1)))) if len(u_at_tick) else float("nan"),
        collision=collided, duration_s=float(odom_t[-1]) if len(odom_t) else float("nan"),
        t=tick_t_rel, s=s_grid, cost=cost, clearance=clearance, u=u_at_tick, u_nom=u_nom_at_tick,
        odom_t=odom_t, states=odom_x, centers=centers, radii=radii, r_ego=r_ego,
        xy=xy_at_tick, buoy_t=buoy_t, buoy_centers=buoy_centers, buoy_radii=buoy_radii,
        buoy_idx_at_tick=buoy_idx_at_tick,
    )


# ---------------------------------------------------------------------------
# Point 1: oracle computation time
# ---------------------------------------------------------------------------
def report_solve_times(results):
    print("\n" + "=" * 70)
    print("1) Computation time")
    print("=" * 70)
    for name in CONTROLLERS:
        r = results.get(("nominal", name))
        if r is None or len(r["solve_ms"]) == 0:
            continue
        ms = r["solve_ms"]
        over = ms > CONTROL_PERIOD_MS
        print(f"{name:>10}: n={len(ms):4d}  mean={ms.mean():6.2f}ms  "
             f"median={np.median(ms):6.2f}ms  p95={np.percentile(ms, 95):6.2f}ms  "
             f"max={ms.max():6.2f}ms  over-{CONTROL_PERIOD_MS:.0f}ms={over.mean():.1%} "
             f"({over.sum()}/{len(ms)} ticks)")


# ---------------------------------------------------------------------------
# Point 2: notebook-style cost table + ratio, vs the oracle, full run
# ---------------------------------------------------------------------------
def _window_summary(r):
    """Notebook-style rollout() summary: cost = raw sum, not mean, over the
    whole run; None if there are no ticks."""
    n = len(r["cost"])
    if n == 0:
        return None
    clr = r["clearance"]
    clr = clr[~np.isnan(clr)]
    return dict(
        cost=float(np.sum(r["cost"])),
        clearance=float(clr.min()) if len(clr) else float("nan"),
        effort=float(np.sqrt(np.mean(np.sum(r["u"] ** 2, axis=1)))),
        solve_ms=float(np.mean(r["solve_ms"])) if len(r["solve_ms"]) == n else float("nan"),
        n=n,
    )


def _print_table(summaries, reference):
    names = [n for n in summaries if summaries[n] is not None]
    if not names:
        print("  (no data)")
        return
    rows = [("closed-loop cost (sum over window)", "{:.0f}", "cost"),
           ("min clearance [m]", "{:.3f}", "clearance"),
           ("control effort RMS [N]", "{:.2f}", "effort"),
           ("solve time mean [ms]", "{:.2f}", "solve_ms"),
           ("n ticks", "{:d}", "n")]
    head = f"{'':<32}" + "".join(f"{n:>14}" for n in names)
    print(head); print("-" * len(head))
    for label, fmt, key in rows:
        print(f"{label:<32}" + "".join(f"{fmt.format(summaries[n][key]):>14}" for n in names))
    ref_cost = summaries.get(reference, {}).get("cost") if reference in names else None
    if ref_cost:
        for n in names:
            if n == reference:
                continue
            ratio = summaries[n]["cost"] / ref_cost
            tag = "  <-- BETTER than oracle" if ratio < 1.0 else ""
            print(f"  cost ratio {n}/{reference}: {ratio:.2f}x{tag}")


def report_cost_vs_oracle(results):
    print("\n" + "=" * 70)
    print("2) Closed-loop cost vs the oracle (notebook-style: cost = sum of")
    print("     per-tick stage cost, ratio to the oracle as reference)")
    print("=" * 70)
    nominal = {name: results[("nominal", name)] for name in CONTROLLERS if ("nominal", name) in results}
    if "oracle" not in nominal:
        print("no nominal oracle bag found -- skipping")
        return

    print("\n--- full run (same window for every controller, like rollout()'s")
    print("    fixed n_steps -- all three start from the same initial condition) ---")
    full = {name: _window_summary(r) for name, r in nominal.items()}
    _print_table(full, reference="oracle")


# ---------------------------------------------------------------------------
# Point 3: nominal vs disturbed robustness
# ---------------------------------------------------------------------------
def report_robustness(results, disturbed_runs):
    """disturbed_runs[name] is a *list* of analyze() results, one per disturbed
    bag -- disturbances aren't seeded (see _all()), so each trial is a different
    realization and gets averaged over rather than treated as definitive."""
    print("\n" + "=" * 70)
    print("3) Robustness to disturbances (nominal vs disturbed, averaged over")
    print("   all disturbed runs found -- each is an unseeded, different draw)")
    print("=" * 70)
    any_pair = False
    for name in CONTROLLERS:
        nom = results.get(("nominal", name))
        dis_list = disturbed_runs.get(name) or []
        if nom is None or not dis_list:
            continue
        any_pair = True
        n = len(dis_list)
        print(f"\n{name} ({n} disturbed run{'s' if n != 1 else ''}):")
        rows = [("success rate", "success_rate", "{:.1%}"),
               ("mean cost", "cost_mean", "{:.2f}"),
               ("min clearance [m]", "min_clearance", "{:.3f}"),
               ("control effort RMS [N]", "effort_rms", "{:.2f}")]
        for label, key, fmt in rows:
            nv = nom[key]
            dvals = np.array([d[key] for d in dis_list], float)
            dvals = dvals[~np.isnan(dvals)]
            if not len(dvals):
                print(f"  {label:<24} nominal={fmt.format(nv):>10}  disturbed=n/a")
                continue
            dmean = dvals.mean()
            spread = f" (+/-{fmt.format(dvals.std())})" if len(dvals) > 1 else ""
            delta = f"{(dmean / nv - 1):+.1%}" if nv == nv and nv != 0 else "n/a"
            print(f"  {label:<24} nominal={fmt.format(nv):>10}  "
                 f"disturbed={fmt.format(dmean):>10}{spread}  change={delta}")
        n_collisions = sum(1 for d in dis_list if d["collision"])
        print(f"  {'collision':<24} nominal={str(nom['collision']):>10}  "
             f"disturbed={n_collisions}/{n} runs")
    if not any_pair:
        print("no controller has both a nominal bag and at least one disturbed bag "
             "-- skipping (re-run compare_run.launch with disturbed:=true)")


# ---------------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------------
def _latest(bag_dir, controller, condition, prefix=""):
    matches = sorted(glob.glob(os.path.join(bag_dir, f"{prefix}{controller}_{condition}_*.bag")))
    return matches[-1] if matches else None


def _all(bag_dir, controller, condition, prefix=""):
    """Every matching bag, not just the latest -- disturbed runs aren't seeded
    (see currents.cpp/wind_and_waves.cpp: both reseed from std::random_device on
    every process start), so each disturbed trial is a different realization and
    the robustness comparison wants all of them, not just one."""
    return sorted(glob.glob(os.path.join(bag_dir, f"{prefix}{controller}_{condition}_*.bag")))


def _scenario_params(seed, index):
    """Same y0/u_ref sample_scenario.py / aggregate_scenarios.py would derive for
    this seed/index -- re-derived here rather than stored, since the RNG draw is
    deterministic (see anmpc_compare_to_oracle.sample_scenario)."""
    rng = np.random.default_rng(seed)
    sc = None
    for _ in range(index + 1):
        sc = sample_scenario(rng)
    return sc["path"].y0, sc["path"].u_ref


class _Tee:
    """Mirrors writes to multiple streams -- lets the report's prints reach the
    terminal as usual while also collecting them for the sibling .txt save."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag_dir", default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "..", "compare_bags"),
                    help="directory to auto-discover {controller}_{nominal,disturbed}_*.bag in")
    for name in CONTROLLERS:
        ap.add_argument(f"--{name}_nominal")
        ap.add_argument(f"--{name}_disturbed", action="append",
                        help="a disturbed bag for this controller; repeat for multiple "
                             "runs (disturbances aren't seeded, so each is a different "
                             "realization -- default: every matching bag in --bag_dir)")
    ap.add_argument("--params", default=PARAMS_YAML)
    ap.add_argument("--seed", type=int, default=None,
                    help="if set, re-derive y0/u_ref via anmpc_compare_to_oracle.sample_scenario "
                         "(same draw run_scenario.sh / aggregate_scenarios.py use) instead of "
                         "reading --params, and auto-discover scenario{seed}_{index}_*.bag bags "
                         "instead of plain {controller}_{condition}_*.bag -- i.e. give --seed/--index "
                         "alone to compare a run_scenario.sh scenario, no --y0/--u_ref needed")
    ap.add_argument("--index", type=int, default=0,
                    help="which draw from --seed's RNG stream to use (0-based); only with --seed")
    ap.add_argument("--y0", type=float, default=None,
                    help="override path/y_offset from --params (or --seed's derived value) -- for "
                         "bags recorded with compare_run.launch's y0:=...")
    ap.add_argument("--u_ref", type=float, default=None,
                    help="override parameters_model/u_ref from --params (or --seed's derived "
                         "value), same reason as --y0")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: compare_controllers.png, or "
                         "compare_controllers_scenario<seed>_<index>.png with --seed); "
                         "the printed report is also saved alongside it as the same "
                         "name with a .txt extension")
    args = ap.parse_args()

    with open(args.params) as fh:
        p = yaml.safe_load(fh)

    bag_prefix = ""
    y0, u_ref = args.y0, args.u_ref
    if args.seed is not None:
        sc_y0, sc_u_ref = _scenario_params(args.seed, args.index)
        y0 = y0 if y0 is not None else sc_y0
        u_ref = u_ref if u_ref is not None else sc_u_ref
        bag_prefix = f"scenario{args.seed}_{args.index}_"

    if args.out is not None:
        out_path = args.out
    elif args.seed is not None:
        out_path = os.path.join(
            PLOTS_DIR, f"compare_controllers_scenario{args.seed}_{args.index}.png")
    else:
        out_path = os.path.join(PLOTS_DIR, "compare_controllers.png")

    y0 = y0 if y0 is not None else p["path"]["y_offset"]
    u_ref = u_ref if u_ref is not None else p["parameters_model"]["u_ref"]
    path_vec = np.array([p["path"]["x_multiplier"], y0, u_ref])
    weights = (p["parameters_mpc"]["Qye"], p["parameters_mpc"]["Qr"],
              p["parameters_mpc"]["Qpsi"], p["parameters_mpc"]["Qu"])
    r_f = 1.0
    r_ego = float(np.hypot(p["parameters_model"].get("ego_length", 0.9),
                           p["parameters_model"].get("ego_width", 0.45)) / 2.0)

    results = {}
    disturbed_runs = {name: [] for name in CONTROLLERS}
    report_buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, report_buf)):
        for name in CONTROLLERS:
            nom_path = getattr(args, f"{name}_nominal") or _latest(args.bag_dir, name, "nominal", bag_prefix)
            if nom_path:
                print(f"analyzing {name}/nominal: {nom_path}")
                results[("nominal", name)] = analyze(name, nom_path, path_vec, weights, r_f, r_ego)

            dis_paths = getattr(args, f"{name}_disturbed") or _all(args.bag_dir, name, "disturbed", bag_prefix)
            for dis_path in dis_paths:
                print(f"analyzing {name}/disturbed: {dis_path}")
                disturbed_runs[name].append(analyze(name, dis_path, path_vec, weights, r_f, r_ego))

        if not results and not any(disturbed_runs.values()):
            ap.error(f"no {bag_prefix}*.bag bags found in {args.bag_dir} and none given explicitly")

        report_solve_times(results)
        report_cost_vs_oracle(results)
        report_robustness(results, disturbed_runs)

    txt_path = os.path.splitext(out_path)[0] + ".txt"
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)) or ".", exist_ok=True)
    with open(txt_path, "w") as fh:
        fh.write(report_buf.getvalue())
    print(f"saved {txt_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    nominal = {name: results[("nominal", name)] for name in CONTROLLERS if ("nominal", name) in results}
    if not nominal:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle("Nominal condition (single run per controller, no disturbance)")
    ax_traj, ax_box, ax_solve = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_cost, ax_speed, ax_dist = axes[1, 0], axes[1, 1], axes[1, 2]
    ax_dist.axis("off")

    x_max = max(r["states"][:, 0].max() for r in nominal.values() if len(r["states"]))
    s_max = x_max / path_vec[0]  # states[:, 0] is X = kx*s, not s itself
    s_grid = np.linspace(0, s_max + 2, 600)
    ax_traj.plot(path_vec[0] * s_grid, np.sin(path_vec[0] * s_grid) + path_vec[1],
                "k--", lw=1, label="reference path")
    for name, r in nominal.items():
        if len(r["states"]):
            ax_traj.plot(r["states"][:, 0], r["states"][:, 1], lw=1.8, color=COLORS[name], label=name)
    any_r = next(iter(nominal.values()))
    for c, rr in zip(any_r["centers"], any_r["radii"]):
        ax_traj.add_patch(plt.Circle(c, rr, color="tab:red", alpha=0.25))
        ax_traj.add_patch(plt.Circle(c, rr + any_r["r_ego"], color="tab:red", alpha=0.4,
                                     ls="--", lw=0.8, fill=False))
    ax_traj.set_aspect("equal"); ax_traj.grid(alpha=0.3); ax_traj.legend(loc="upper left", fontsize=8)
    ax_traj.set_xlabel("X [m]"); ax_traj.set_ylabel("Y [m]"); ax_traj.set_title("Closed-loop trajectories")

    box_names = [n for n in nominal if len(nominal[n]["solve_ms"])]
    if box_names:
        # One boxplot() call per controller (each a single 1-D array) rather than
        # one call over the whole ragged list: passing differently-sized arrays
        # together makes some matplotlib/numpy combinations raise ValueError
        # ("inhomogeneous shape") instead of just drawing per-dataset boxes.
        for i, n in enumerate(box_names):
            bp = ax_box.boxplot(nominal[n]["solve_ms"], positions=[i], patch_artist=True,
                                widths=0.6)
            bp["boxes"][0].set_facecolor(COLORS[n]); bp["boxes"][0].set_alpha(0.5)
            bp["medians"][0].set_color("black")
        ax_box.set_xticks(range(len(box_names))); ax_box.set_xticklabels(box_names)
        ax_box.axhline(CONTROL_PERIOD_MS, color="k", ls="--", lw=1,
                       label=f"{CONTROL_PERIOD_MS:.0f}ms budget")
        ax_box.set_yscale("log"); ax_box.grid(alpha=0.3, axis="y"); ax_box.legend(fontsize=8)
        ax_box.set_ylabel("solve time [ms]"); ax_box.set_title("Solve time distribution")

    for name, r in nominal.items():
        if len(r["solve_ms"]):
            ax_solve.plot(r["s"], r["solve_ms"], ".", ms=3, color=COLORS[name], label=name, alpha=0.6)
    ax_solve.axhline(CONTROL_PERIOD_MS, color="k", ls="--", lw=1, label=f"{CONTROL_PERIOD_MS:.0f}ms budget")
    ax_solve.set_yscale("log"); ax_solve.grid(alpha=0.3); ax_solve.legend(fontsize=8)
    ax_solve.set_xlabel("arc length s [m]"); ax_solve.set_ylabel("solve time [ms]")
    ax_solve.set_title("Solve time vs. path progress")

    for name, r in nominal.items():
        if len(r["s"]):
            ax_cost.plot(r["s"], r["cost"], lw=1.2, color=COLORS[name], label=name, alpha=0.8)
    ax_cost.set_yscale("log"); ax_cost.grid(alpha=0.3); ax_cost.legend(fontsize=8)
    ax_cost.set_xlabel("arc length s [m]"); ax_cost.set_ylabel("per-tick stage cost (log)")
    ax_cost.set_title("Closed-loop cost vs. path progress")

    for name, r in nominal.items():
        if len(r["states"]):
            ax_speed.plot(r["odom_t"], r["states"][:, 3], lw=1.3, color=COLORS[name], label=name)
    ax_speed.axhline(path_vec[2], color="k", ls="--", lw=1, label="u_ref")
    ax_speed.grid(alpha=0.3); ax_speed.legend(fontsize=8)
    ax_speed.set_xlabel("time [s]"); ax_speed.set_ylabel("surge speed [m/s]")
    ax_speed.set_title("Speed tracking")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved {out_path}")

    actions_out = os.path.splitext(out_path)[0] + "_actions.png"
    _plot_control_actions(nominal, actions_out)

    if any(disturbed_runs.values()):
        dist_out = os.path.splitext(out_path)[0] + "_disturbed_traj.png"
        _plot_disturbed_trajectories(disturbed_runs, path_vec, dist_out)


def _plot_control_actions(nominal, out_path):
    """One panel per thruster (control is a 4-vector, see roboat_core/Force.msg
    -- 4 thruster forces, no more specific per-axis meaning is exposed above
    Sim.cpp), each showing every controller's applied command u. barriernet
    also gets its raw, pre-CBF-projection u_nom overlaid as a dashed line in
    the same color, so the CBF's correction is visible as the gap between the
    two lines -- not just its magnitude (/mpc_status_barriernet/intervention_n),
    but its direction: does u_nom point away from what the CBF actually allows,
    or just need a small nudge?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in nominal if len(nominal[n]["u"])]
    if not names:
        return

    n_dim = nominal[names[0]]["u"].shape[1]
    fig, axes = plt.subplots(2, (n_dim + 1) // 2, figsize=(6 * ((n_dim + 1) // 2), 8), squeeze=False)
    axes = axes.flatten()

    for i in range(n_dim):
        ax = axes[i]
        for name in names:
            r = nominal[name]
            ax.plot(r["s"], r["u"][:, i], lw=1.4, color=COLORS[name], label=f"{name} (applied)")
            if name == "barriernet" and r.get("u_nom") is not None:
                ax.plot(r["s"], r["u_nom"][:, i], lw=1.2, ls="--", color=COLORS[name],
                        alpha=0.7, label="barriernet (u_nom)")
        ax.grid(alpha=0.3)
        ax.set_xlabel("arc length s [m]"); ax.set_ylabel(f"u[{i}] [N]")
        ax.set_title(f"Thruster {i}")
        if i == 0:
            ax.legend(fontsize=8)
    for i in range(n_dim, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Control actions -- nominal condition (barriernet: applied vs. raw u_nom)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


def _plot_disturbed_trajectories(disturbed_runs, path_vec, out_path):
    """One panel per controller (its first disturbed bag), each with its OWN
    buoys (currents/wind/waves reseed every process start, so each disturbed
    run's buoy drift is an independent realization, not shared across
    controllers) -- overlaying all three controllers' independent buoy sets
    in one axes was too dense to read, hence separate panels. Buoys are
    physically simulated the same as the vessel, so under disturbance they
    drift too (~2.3m over a 40s run isn't unusual): a single static circle
    per buoy would show where it started, not where the vessel actually had
    to avoid it, which is why a collision could look like empty water in a
    plot that only drew the initial buoy position.

    Plotted north-up (vertical run direction, since the path covers dozens of
    meters north but only a few meters of east-west lateral motion), and
    zoomed to a window around each controller's own closest-approach point --
    showing the full multi-tens-of-meters run at the same scale as ~1m buoys
    is what made this unreadable before.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist_first = {name: runs[0] for name, runs in disturbed_runs.items() if runs}
    dist_first = {name: r for name, r in dist_first.items() if len(r["states"])}
    if not dist_first:
        return

    names = [n for n in CONTROLLERS if n in dist_first]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 7), sharey=False)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        r = dist_first[name]
        north, east = r["states"][:, 0], r["states"][:, 1]

        clr = r["clearance"]
        valid = ~np.isnan(clr)
        idx = np.where(valid)[0][np.argmin(clr[valid])] if valid.any() else None
        vessel_n, vessel_e = r["xy"][idx] if idx is not None and len(r["xy"]) else (north[0], east[0])

        # Zoom to a window around the closest approach: buoys are ~0.4-0.8m
        # radius, the run is tens of meters long -- showing both at the same
        # scale makes the buoys illegible dots.
        half_n = 7.0
        ax.set_ylim(vessel_n - half_n, vessel_n + half_n)

        s_grid_d = np.linspace(0, (vessel_n + half_n) / max(path_vec[0], 1e-6) + 5, 600)
        north_ref = path_vec[0] * s_grid_d
        east_ref = np.sin(path_vec[0] * s_grid_d) + path_vec[1]
        ax.plot(east_ref, north_ref, "k--", lw=1, label="reference path")
        ax.plot(east, north, lw=1.8, color=COLORS[name], label=name, zorder=3)

        # This run's own buoy drift: faint circle at first snapshot, solid
        # (with safety margin) at last, dotted line tracing the path between.
        # Neutral red (not the controller's color) since a buoy isn't "this
        # controller's" -- matches the obstacle-circle convention used
        # elsewhere in this script's nominal trajectory panel.
        if r["buoy_centers"]:
            c0, c1 = r["buoy_centers"][0], r["buoy_centers"][-1]
            rad = r["buoy_radii"][0]
            for j in range(len(c1)):
                ax.plot([c0[j, 1], c1[j, 1]], [c0[j, 0], c1[j, 0]],
                       ":", color="tab:red", lw=1, alpha=0.4)
                ax.add_patch(plt.Circle((c0[j, 1], c0[j, 0]), rad[j], color="tab:red", alpha=0.10))
                ax.add_patch(plt.Circle((c1[j, 1], c1[j, 0]), rad[j], color="tab:red", alpha=0.20))
                ax.add_patch(plt.Circle((c1[j, 1], c1[j, 0]), rad[j] + r["r_ego"],
                                        color="tab:red", alpha=0.3, ls=":", lw=0.7, fill=False))

        # Closest approach: this controller's own worst clearance tick, with
        # the buoy it was closest to AT THAT MOMENT (not at the end of the
        # run, which can be meters away from where it actually was when the
        # vessel passed closest) marked and joined by a short segment.
        if idx is not None:
            bi = r["buoy_idx_at_tick"][idx]
            collided = r["collision"]
            marker = "X" if collided else "*"
            ax.plot(vessel_e, vessel_n, marker, color=COLORS[name], ms=18,
                   mec="black", mew=1.0, zorder=5)
            if bi is not None and len(r["buoy_centers"][bi]):
                j = np.argmin(np.linalg.norm(r["buoy_centers"][bi] - r["xy"][idx], axis=1))
                b_xy = r["buoy_centers"][bi][j]
                b_rad = r["buoy_radii"][bi][j]
                ax.plot([vessel_e, b_xy[1]], [vessel_n, b_xy[0]], "-", color="black", lw=1, alpha=0.6)
                ax.add_patch(plt.Circle((b_xy[1], b_xy[0]), b_rad, color="tab:red", alpha=0.5, zorder=4))
                ax.add_patch(plt.Circle((b_xy[1], b_xy[0]), b_rad + r["r_ego"], color="tab:red",
                                        alpha=0.7, ls="--", lw=1.2, fill=False, zorder=4))
            status = "COLLISION" if collided else "closest approach"
            ax.set_title(f"{name}\n{status}: {clr[valid].min():.2f}m", fontsize=11,
                        color=("firebrick" if collided else "black"))
        else:
            ax.set_title(name, fontsize=11)

        east_span = max(3.0, half_n * 0.6)
        ax.set_xlim(vessel_e - east_span / 2, vessel_e + east_span / 2)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("East [m]")

    axes[0].set_ylabel("North [m]")
    axes[0].legend(loc="upper left", fontsize=8)
    n_dist = max((len(v) for v in disturbed_runs.values()), default=1)
    fig.suptitle(f"Disturbed condition -- first of {n_dist} disturbed run(s) per controller "
                f"(not an average): each controller's own closest approach to its own "
                f"(independently drifting) buoys", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
