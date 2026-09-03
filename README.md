# Simulation Environment for Quarter Scale Roboat ASV (Original, with sensors, no moving buoys)

## To run lake environment

roslaunch gazebo_sim lake.launch

- Disable gazebo physics for it to work

- Check for multiple obstacle scenarios inside iros2026_scenarios including lanes, intersections, buoys, etc.

## To run roboat simulation

roslaunch gazebo_sim gazebo_roboat.launch

- Spawns roboat with velodyne lidar and custom dynamics
- Check launch file to include disturbances from wind, waves, and currents

## Multiple perception systems

roslaunch roboat_planning run_boat.launch

- Filters LiDAR data
- Creates occupancy grid maps

- Check for different occupancy grid-based perception system inside obstacle_detector package, which fits ellipses and circles to obstacles

roslaunch obstacle_detector pcl_filter.launch
- launches pcl-based pointcloud filter

## Amortized MPC Experiments

Compares three controllers on the same sinusoidal-path/buoy-avoidance task: the
`oracle` (CasADi/IPOPT NMPC), `anmpc` (amortized NMPC network), and
`barriernet` (BarrierNet). All three run through the same
`control/launch/compare_run.launch`, one controller at a time, recording a
rosbag; comparisons are done offline from the bags.

Needs `roscore` running first. All commands below assume `cd /ros1_ws/src`.

### Run one scenario

    ./run_scenario.sh --seed 0 --index 3 --all

Runs all three controllers, one at a time, on the scenario sampled from
`--seed`/`--index` (matches `control/scripts/compare_to_oracle.py`'s own
`sample_scenario()` exactly -- same seed+index gives the identical
y0/u_ref/obstacles in Python and ROS). `--index` picks which draw from that
seed's random stream to use; `--seed 0 --index 0..9` reproduces a fixed,
already-validated set of 10 scenarios.

    ./run_scenario.sh --seed 0 --index 3 --controller barriernet   # one controller only
    ./run_scenario.sh --seed 0 --index 3 --all --disturbed         # + wind/waves/currents

    ./run_scenario.sh --y0 3.0 --u_ref 0.55 --scenario_id my_test --all \
      --obstacles '[{name: buoy1, radius: 0.6, mass: 10.0, initial_state: [6.0, 3.0, 0.0, 0.0]}]'

Fully manual scenario instead of sampling -- `--obstacles` must be single-line
YAML flow style (`[{...}, {...}]`), not block style: roslaunch's own CLI
`arg:=value` parser silently drops everything after the first newline in a
multi-line value.

Each controller run takes ~`duration`+10s (default `duration=40`). Bags land
in `~/compare_bags` as `scenario<ID>_<controller>_<nominal|disturbed>_<timestamp>.bag`.
`run_scenario.sh` prints the ready-to-run comparison command at the end.

### Compare the controllers from one scenario

    python3 control/scripts/compare_controllers.py \
      --oracle_nominal ~/compare_bags/scenario<ID>_oracle_nominal_*.bag \
      --anmpc_nominal ~/compare_bags/scenario<ID>_anmpc_nominal_*.bag \
      --barriernet_nominal ~/compare_bags/scenario<ID>_barriernet_nominal_*.bag \
      --y0 <Y0> --u_ref <U_REF>

Prints closed-loop cost/clearance/effort/solve-time tables and saves a
dashboard plot (trajectories, solve-time distribution, cost vs. progress,
speed tracking). Add `--oracle_disturbed`/`--anmpc_disturbed`/
`--barriernet_disturbed` (repeatable, for averaging over several disturbed
runs of the same scenario -- each is a different random draw) to also get a
robustness report and a second plot with one zoomed panel per controller
showing exactly where it came closest to (or hit) a buoy.

### Run and aggregate many scenarios

    for i in $(seq 0 99); do ./run_scenario.sh --seed 0 --index "$i" --all; done
    python3 control/scripts/aggregate_scenarios.py --seed 0 --indices 0-99

The loop is sequential and slow (~4+ hours for 100 scenarios x 3 controllers)
-- run it in the background. `aggregate_scenarios.py` reuses
`compare_controllers.py`'s own analysis, writes one CSV row per
scenario/controller (`aggregate_scenarios.csv`), and prints an aggregate
summary: mean closed-loop cost, cost ratio vs. oracle, collision rate, and
solve-time mean/p95/max/over-100ms-budget-fraction, per controller.
`--indices` accepts `0-99`, `0,3,7`, or a mix like `0-9,20,30-35`.

### Plots

All comparison plots are written to `../plots/` (one level above this repo),
organized as `plots/comparisons/` (multi-controller plots -- almost
everything above) and `plots/oracle/` / `plots/anmpc/` / `plots/barriernet/`
(single-controller-only plots). Created automatically; pass `--out` to any
script to override.