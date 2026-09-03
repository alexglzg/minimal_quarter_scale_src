#!/usr/bin/env bash
# Run one scenario through control/launch/run_controllers_anmpc.launch, for one or all
# three controllers, and print the control/scripts/compare_controllers.py
# command to compare them afterward. Lives at the repo root; run it from
# there (or anywhere -- it cd's to its own directory first).
#
# roscore must already be running (this script does not start one).
#
# Randomized scenario -- matches compare_to_oracle.py's sample_scenario()
# exactly: same --seed/--index in Python (see sample_scenario.py's docstring)
# gives the identical y0/u_ref/obstacles.
#
#     ./run_scenario.sh --seed 0 --index 3 --all
#     ./run_scenario.sh --seed 0 --index 3 --controller barriernet
#
# Fully manual scenario -- skip sampling, give the numbers directly:
#
#     ./run_scenario.sh --y0 3.0 --u_ref 0.55 --scenario_id my_test --all \
#       --obstacles '[{name: buoy1, radius: 0.6, mass: 10.0, initial_state: [6.0, 3.0, 0.0, 0.0]}]'
#
#   IMPORTANT: --obstacles must be single-line YAML flow style, exactly like
#   that (a "[...]" list of "{...}" maps on one line). roslaunch's own CLI
#   arg:=value parser silently drops everything after the first newline in a
#   multi-line value, so YAML block style (one "- name: ..." per line) breaks
#   silently. See run_controllers_anmpc.launch's obstacles_yaml arg doc for the same
#   note and a from-scratch example.
#
# Add --disturbed to also run wind+waves+currents (their own launch files
# reseed from std::random_device on every process start, so each disturbed
# run is a different, unrepeatable realization -- run it more than once if
# you want to average over that).
#
# Each controller run takes ~duration+10 seconds (8s startup delay + the
# recording itself). Bags land in ~/compare_bags, named
# scenario<ID>_<controller>_<nominal|disturbed>_<timestamp>.bag.
set -euo pipefail
cd "$(dirname "$0")"
SCRIPTS_DIR="control/scripts"

SEED=""
INDEX=0
Y0=""
U_REF=""
OBSTACLES_YAML=""
NUM_OBSTACLES=3
CONTROLLER=""
ALL=false
DISTURBED=false
DURATION=40
SCENARIO_ID=""
BAG_DIR="$(pwd)/compare_bags"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --index) INDEX="$2"; shift 2 ;;
    --y0) Y0="$2"; shift 2 ;;
    --u_ref) U_REF="$2"; shift 2 ;;
    --obstacles) OBSTACLES_YAML="$2"; shift 2 ;;
    --num_obstacles) NUM_OBSTACLES="$2"; shift 2 ;;
    --controller) CONTROLLER="$2"; shift 2 ;;
    --all) ALL=true; shift ;;
    --disturbed) DISTURBED=true; shift ;;
    --duration) DURATION="$2"; shift 2 ;;
    --scenario_id) SCENARIO_ID="$2"; shift 2 ;;
    --bag_dir) BAG_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# if ! rostopic list >/dev/null 2>&1; then
#   echo "error: no roscore reachable. Start one first (roscore &) and try again." >&2
#   exit 1
# fi

if [[ -n "$SEED" ]]; then
  eval "$(python3 "$SCRIPTS_DIR/sample_scenario.py" --seed "$SEED" --index "$INDEX")"
  SCENARIO_ID="${SCENARIO_ID:-${SEED}_${INDEX}}"
elif [[ -n "$Y0" && -n "$U_REF" && -n "$OBSTACLES_YAML" ]]; then
  SCENARIO_ID="${SCENARIO_ID:-manual}"
else
  echo "error: provide either --seed/--index (randomized, matches compare_to_oracle.py)" >&2
  echo "       or --y0/--u_ref/--obstacles (manual)." >&2
  exit 1
fi

if [[ "$ALL" == true ]]; then
  CONTROLLERS="oracle anmpc barriernet"
elif [[ -n "$CONTROLLER" ]]; then
  CONTROLLERS="$CONTROLLER"
else
  echo "error: specify --controller <oracle|anmpc|barriernet> or --all" >&2
  exit 1
fi

echo "scenario_id=$SCENARIO_ID  y0=$Y0  u_ref=$U_REF  num_obstacles=$NUM_OBSTACLES  disturbed=$DISTURBED"
echo "obstacles: $OBSTACLES_YAML"
echo

for ctrl in $CONTROLLERS; do
  echo "=== $ctrl ==="
  roslaunch control run_controllers_anmpc.launch controller:="$ctrl" scenario_id:="$SCENARIO_ID" \
    duration:="$DURATION" y0:="$Y0" u_ref:="$U_REF" num_obstacles:="$NUM_OBSTACLES" \
    su0:=0.3 disturbed:="$DISTURBED" obstacles_yaml:="$OBSTACLES_YAML" \
    bag_dir:="$BAG_DIR"
done

CONDITION=$([[ "$DISTURBED" == true ]] && echo disturbed || echo nominal)
echo
echo "Done. Bags in $BAG_DIR:"
CMD="python3 $SCRIPTS_DIR/compare_controllers.py"
for ctrl in oracle anmpc barriernet; do
  bag=$(ls -t "$BAG_DIR"/scenario${SCENARIO_ID}_${ctrl}_${CONDITION}_*.bag 2>/dev/null | head -1 || true)
  if [[ -n "$bag" ]]; then
    echo "  $bag"
    CMD="$CMD --${ctrl}_${CONDITION} $bag"
  fi
done

