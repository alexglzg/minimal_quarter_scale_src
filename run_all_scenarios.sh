#!/usr/bin/env bash
# Overnight batch driver on top of ./run_scenario.sh: runs all three
# controllers (oracle, anmpc, barriernet) over N scenarios once nominal
# (no disturbance) and REPEATS times disturbed (wind+waves+currents), where
# each disturbed pass is an independent random realization -- see
# run_scenario.sh's own header for why that's worth repeating.
#
# roscore must already be running (this script does not start one).
#
#     ./run_all_scenarios.sh --seed 0 --n_scenarios 25 --n_disturbed 3
#
# Meant to be left running unattended, e.g.:
#
#     nohup ./run_all_scenarios.sh --seed 0 > overnight.log 2>&1 &
#     disown
#
# Unlike run_scenario.sh, this script does NOT abort on a single failed
# scenario (gazebo hiccup, roslaunch timeout, etc.) -- it logs the failure,
# moves on, and reports every failure in the final summary so nothing is
# silently lost from an unattended run.
#
# Resumable: each (condition, repeat, index) unit writes a marker file under
# --marker_dir on success. Re-running the exact same command after a crash
# or Ctrl-C skips everything already marked done. Pass --force to ignore
# markers and redo everything.
set -uo pipefail
cd "$(dirname "$0")"

SEED=0
N_SCENARIOS=25
N_DISTURBED=3
DURATION=40
BAG_DIR="$HOME/compare_bags"
MARKER_DIR=""
LOG_DIR=""
FORCE=false
SKIP_NOMINAL=false
SKIP_DISTURBED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;;
    --n_scenarios) N_SCENARIOS="$2"; shift 2 ;;
    --n_disturbed) N_DISTURBED="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --bag_dir) BAG_DIR="$2"; shift 2 ;;
    --marker_dir) MARKER_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    --skip_nominal) SKIP_NOMINAL=true; shift ;;
    --skip_disturbed) SKIP_DISTURBED=true; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

MARKER_DIR="${MARKER_DIR:-$BAG_DIR/.run_markers_seed${SEED}}"
LOG_DIR="${LOG_DIR:-$BAG_DIR/logs_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$MARKER_DIR" "$LOG_DIR"

if [[ "$FORCE" == true ]]; then
  rm -f "$MARKER_DIR"/*.done
fi

LAST_IDX=$((N_SCENARIOS - 1))
TOTAL_UNITS=$(( (SKIP_NOMINAL == false ? N_SCENARIOS : 0) + (SKIP_DISTURBED == false ? N_DISTURBED * N_SCENARIOS : 0) ))
DONE_COUNT=0
FAILED=()

echo "seed=$SEED  scenarios=0..$LAST_IDX  disturbed_repeats=$N_DISTURBED  duration=${DURATION}s"
echo "bag_dir=$BAG_DIR  marker_dir=$MARKER_DIR  log_dir=$LOG_DIR"
echo "total units to run: $TOTAL_UNITS (skipping units already marked done; use --force to redo all)"
echo

run_unit () {
  # $1 = marker name, $2 = index, $3 = disturbed(true/false), $4 = log label
  local marker="$1" idx="$2" disturbed="$3" label="$4"
  local marker_file="$MARKER_DIR/${marker}.done"
  local log_file="$LOG_DIR/${marker}.log"

  DONE_COUNT=$((DONE_COUNT + 1))
  if [[ -f "$marker_file" ]]; then
    echo "[$DONE_COUNT/$TOTAL_UNITS] SKIP (already done) $label"
    return
  fi

  echo "[$DONE_COUNT/$TOTAL_UNITS] RUN  $label"
  local args=(--seed "$SEED" --index "$idx" --all --duration "$DURATION" --bag_dir "$BAG_DIR")
  [[ "$disturbed" == true ]] && args+=(--disturbed)

  if ./run_scenario.sh "${args[@]}" > "$log_file" 2>&1; then
    touch "$marker_file"
  else
    echo "    FAILED -- see $log_file"
    FAILED+=("$label")
  fi
}

if [[ "$SKIP_NOMINAL" == false ]]; then
  for i in $(seq 0 "$LAST_IDX"); do
    run_unit "nominal_${i}" "$i" false "nominal index=$i"
  done
fi

if [[ "$SKIP_DISTURBED" == false ]]; then
  for rep in $(seq 1 "$N_DISTURBED"); do
    for i in $(seq 0 "$LAST_IDX"); do
      run_unit "disturbed_r${rep}_${i}" "$i" true "disturbed rep=$rep index=$i"
    done
  done
fi

echo
echo "=== Done: $((DONE_COUNT - ${#FAILED[@]}))/$TOTAL_UNITS units OK, ${#FAILED[@]} failed ==="
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf '  FAILED: %s\n' "${FAILED[@]}"
  echo "Re-run this same command to retry only the failed/incomplete units."
fi

echo
echo "Next, aggregate:"
echo "  python3 control/scripts/aggregate_scenarios.py --seed $SEED --indices 0-$LAST_IDX --condition nominal"
echo "  python3 control/scripts/aggregate_scenarios.py --seed $SEED --indices 0-$LAST_IDX --condition disturbed"
echo "Per-scenario (averages the $N_DISTURBED disturbed runs vs. the 1 nominal run):"
echo "  python3 control/scripts/compare_controllers.py --seed $SEED --index <i>"

[[ ${#FAILED[@]} -eq 0 ]]
