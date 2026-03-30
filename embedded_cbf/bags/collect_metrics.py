#!/usr/bin/env python3
"""
MPC-CBF Multi-Bag Metrics Collection

Processes multiple rosbags and outputs summary statistics for the paper.
Computes per-bag and aggregate metrics for:
  - Solve time (mean, std, median, max, 95th, 99th)
  - Polytope halfplane count (mean, std, median, min, max)
  - CBF clearance min_h (min, mean)
  - CBF slack (min, mean)
  - Footprint-to-obstacle clearance (min, mean) — optional, slower

Outputs as printed table and optionally as CSV.

Usage:
  python3 collect_metrics.py bag1.bag bag2.bag bag3.bag --mode scan
  python3 collect_metrics.py *.bag --mode gridmap --csv results.csv
  python3 collect_metrics.py bags/ --mode scan --csv results.csv
"""

import numpy as np
import argparse
import os
import glob
import csv

from plot_mpc_cbf_bag import (load_bag, robot_vertices, scan_to_world,
                               compute_footprint_clearance)


def compute_bag_metrics(bag_path, mode, L, W, compute_clearance=False):
    """
    Compute all metrics for a single bag.

    Returns dict with metric values, or None if bag loading fails.
    """
    try:
        data = load_bag(bag_path, mode=mode)
    except Exception as e:
        print(f"  ERROR loading {bag_path}: {e}")
        return None

    metrics = {'bag': os.path.basename(bag_path)}

    # Solve time
    if len(data['solve_time']) > 0:
        times = np.array([s[1] for s in data['solve_time']])
        metrics['solve_n'] = len(times)
        metrics['solve_mean'] = np.mean(times)
        metrics['solve_std'] = np.std(times)
        metrics['solve_median'] = np.median(times)
        metrics['solve_max'] = np.max(times)
        metrics['solve_min'] = np.min(times)
        metrics['solve_95'] = np.percentile(times, 95)
        metrics['solve_99'] = np.percentile(times, 99)
        metrics['_raw_solve'] = times
    else:
        metrics['solve_n'] = 0
        metrics['_raw_solve'] = np.array([])

    # Polytope halfplanes
    if len(data['polytopes']) > 0:
        n_planes = np.array([p[3] for p in data['polytopes']])
        metrics['poly_n'] = len(n_planes)
        metrics['poly_mean'] = np.mean(n_planes)
        metrics['poly_std'] = np.std(n_planes)
        metrics['poly_median'] = np.median(n_planes)
        metrics['poly_max'] = np.max(n_planes)
        metrics['poly_min'] = np.min(n_planes)
        metrics['_raw_poly'] = n_planes
    else:
        metrics['poly_n'] = 0
        metrics['_raw_poly'] = np.array([])

    # CBF stats
    if len(data['cbf']) > 0:
        cbf_arr = np.array(data['cbf'])
        min_h = cbf_arr[:, 1]
        min_slack = cbf_arr[:, 2]
        metrics['cbf_n'] = len(min_h)
        metrics['h_min'] = np.min(min_h)
        metrics['h_mean'] = np.mean(min_h)
        metrics['slack_min'] = np.min(min_slack)
        metrics['slack_mean'] = np.mean(min_slack)
        metrics['h_negative'] = int(np.sum(min_h < 0))
        metrics['slack_negative'] = int(np.sum(min_slack < 0))
        metrics['_raw_h'] = min_h
        metrics['_raw_slack'] = min_slack
    else:
        metrics['cbf_n'] = 0
        metrics['_raw_h'] = np.array([])
        metrics['_raw_slack'] = np.array([])

    # Footprint-to-obstacle clearance (optional, slow)
    if compute_clearance and data['obstacle_pts'] is not None:
        odom_ds = data['odom'][::10]
        scan_raw = data.get('_scan_raw', None) if mode == 'scan' else None
        fc = compute_footprint_clearance(odom_ds, data['obstacle_pts'], L, W,
                                          scan_raw=scan_raw)
        if len(fc) > 0:
            fc_arr = np.array(fc)
            metrics['clearance_n'] = len(fc_arr)
            metrics['clearance_min'] = np.min(fc_arr[:, 1])
            metrics['clearance_mean'] = np.mean(fc_arr[:, 1])
            metrics['_raw_clearance'] = fc_arr[:, 1]
        else:
            metrics['clearance_n'] = 0
            metrics['_raw_clearance'] = np.array([])
    else:
        metrics['clearance_n'] = -1  # not computed
        metrics['_raw_clearance'] = np.array([])

    # Duration
    if len(data['odom']) > 0:
        metrics['duration'] = data['odom'][-1][0] - data['odom'][0][0]
    else:
        metrics['duration'] = 0

    return metrics


def print_single(m):
    """Print metrics for a single bag."""
    print(f"\n{'='*60}")
    print(f"  {m['bag']}  ({m.get('duration', 0):.1f} s)")
    print(f"{'='*60}")

    if m['solve_n'] > 0:
        print(f"  Solve time ({m['solve_n']} solves):")
        print(f"    Mean: {m['solve_mean']:.2f} ms  Std: {m['solve_std']:.2f} ms")
        print(f"    Median: {m['solve_median']:.2f} ms  Max: {m['solve_max']:.2f} ms")
        print(f"    95th: {m['solve_95']:.2f} ms  99th: {m['solve_99']:.2f} ms")

    if m['poly_n'] > 0:
        print(f"  Polytope halfplanes ({m['poly_n']} polytopes):")
        print(f"    Mean: {m['poly_mean']:.1f}  Std: {m['poly_std']:.1f}")
        print(f"    Median: {m['poly_median']:.0f}  Min: {m['poly_min']}  Max: {m['poly_max']}")

    if m['cbf_n'] > 0:
        print(f"  CBF ({m['cbf_n']} timesteps):")
        print(f"    min_h  — min: {m['h_min']:.4f} m  mean: {m['h_mean']:.4f} m")
        print(f"    slack  — min: {m['slack_min']:.4f} m  mean: {m['slack_mean']:.4f} m")
        if m['h_negative'] > 0:
            print(f"    WARNING: {m['h_negative']} negative min_h values!")
        if m['slack_negative'] > 0:
            print(f"    WARNING: {m['slack_negative']} negative slack values!")

    if m.get('clearance_n', -1) > 0:
        print(f"  Footprint-to-obstacle clearance ({m['clearance_n']} samples):")
        print(f"    Min: {m['clearance_min']:.4f} m  Mean: {m['clearance_mean']:.4f} m")
    elif m.get('clearance_n', -1) == 0:
        print(f"  Footprint-to-obstacle clearance: no data")


def print_aggregate(all_metrics):
    """Print aggregate statistics from pooled raw data across all bags."""
    n = len(all_metrics)
    print(f"\n{'='*60}")
    print(f"  AGGREGATE ({n} bags)")
    print(f"{'='*60}")

    # Solve time: pool all raw values
    all_solve = np.concatenate([m['_raw_solve'] for m in all_metrics
                                 if len(m['_raw_solve']) > 0])
    if len(all_solve) > 0:
        print(f"  Solve time ({len(all_solve)} total solves):")
        print(f"    Mean:   {np.mean(all_solve):.2f} ms")
        print(f"    Std:    {np.std(all_solve):.2f} ms")
        print(f"    Median: {np.median(all_solve):.2f} ms")
        print(f"    Min:    {np.min(all_solve):.2f} ms")
        print(f"    Max:    {np.max(all_solve):.2f} ms")
        print(f"    95th:   {np.percentile(all_solve, 95):.2f} ms")
        print(f"    99th:   {np.percentile(all_solve, 99):.2f} ms")

    # Polytope halfplanes: pool all raw values
    all_poly = np.concatenate([m['_raw_poly'] for m in all_metrics
                                if len(m['_raw_poly']) > 0])
    if len(all_poly) > 0:
        print(f"  Polytope halfplanes ({len(all_poly)} total polytopes):")
        print(f"    Mean:   {np.mean(all_poly):.1f}")
        print(f"    Std:    {np.std(all_poly):.1f}")
        print(f"    Median: {np.median(all_poly):.0f}")
        print(f"    Min:    {np.min(all_poly)}")
        print(f"    Max:    {np.max(all_poly)}")
        unique, counts = np.unique(all_poly, return_counts=True)
        print(f"    Distribution:")
        for u, c in zip(unique, counts):
            pct = 100 * c / len(all_poly)
            print(f"      {int(u):3d} planes: {c:4d} ({pct:5.1f}%)")

    # CBF: pool all raw values
    all_h = np.concatenate([m['_raw_h'] for m in all_metrics
                             if len(m['_raw_h']) > 0])
    all_slack = np.concatenate([m['_raw_slack'] for m in all_metrics
                                 if len(m['_raw_slack']) > 0])
    if len(all_h) > 0:
        print(f"  CBF ({len(all_h)} total timesteps):")
        print(f"    min_h  — min: {np.min(all_h):.4f} m  mean: {np.mean(all_h):.4f} m  std: {np.std(all_h):.4f} m")
        print(f"    slack  — min: {np.min(all_slack):.4f} m  mean: {np.mean(all_slack):.4f} m  std: {np.std(all_slack):.4f} m")
        n_neg_h = int(np.sum(all_h < 0))
        n_neg_s = int(np.sum(all_slack < 0))
        if n_neg_h > 0:
            print(f"    WARNING: {n_neg_h} negative min_h values!")
        if n_neg_s > 0:
            print(f"    WARNING: {n_neg_s} negative slack values!")

    # Clearance: pool all raw values
    all_cl = np.concatenate([m['_raw_clearance'] for m in all_metrics
                              if len(m['_raw_clearance']) > 0])
    if len(all_cl) > 0:
        print(f"  Footprint-to-obstacle clearance ({len(all_cl)} total samples):")
        print(f"    Min:  {np.min(all_cl):.4f} m")
        print(f"    Mean: {np.mean(all_cl):.4f} m")
        print(f"    Std:  {np.std(all_cl):.4f} m")


def save_csv(all_metrics, csv_path):
    """Save all per-bag metrics to CSV (excludes raw arrays)."""
    if not all_metrics:
        return

    # Exclude raw array keys
    keys = [k for k in all_metrics[0].keys() if not k.startswith('_raw')]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for m in all_metrics:
            writer.writerow({k: v for k, v in m.items() if not k.startswith('_raw')})
    print(f"\nSaved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='MPC-CBF Multi-Bag Metrics')
    parser.add_argument('bags', nargs='+',
                        help='Bag files or directories containing bags')
    parser.add_argument('--mode', choices=['scan', 'gridmap'], default='scan',
                        help='Data source: scan (Gazebo/LiDAR) or gridmap')
    parser.add_argument('--robot-length', type=float, default=0.9,
                        help='Robot length [m]')
    parser.add_argument('--robot-width', type=float, default=0.45,
                        help='Robot width [m]')
    parser.add_argument('--clearance', action='store_true',
                        help='Compute footprint-to-obstacle clearance (slow)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save results to CSV file')
    args = parser.parse_args()

    L, W = args.robot_length, args.robot_width

    # Resolve bag paths (handle directories and globs)
    bag_paths = []
    for b in args.bags:
        if os.path.isdir(b):
            bag_paths.extend(sorted(glob.glob(os.path.join(b, '*.bag'))))
        elif os.path.isfile(b):
            bag_paths.append(b)
        else:
            # Try as glob pattern
            expanded = sorted(glob.glob(b))
            bag_paths.extend([f for f in expanded if f.endswith('.bag')])

    if not bag_paths:
        print("No bag files found.")
        return

    print(f"Processing {len(bag_paths)} bag(s)...")

    all_metrics = []
    for i, bp in enumerate(bag_paths):
        print(f"\n[{i+1}/{len(bag_paths)}] {os.path.basename(bp)}")
        m = compute_bag_metrics(bp, args.mode, L, W,
                                 compute_clearance=args.clearance)
        if m is not None:
            print_single(m)
            all_metrics.append(m)

    if len(all_metrics) > 1:
        print_aggregate(all_metrics)

    if args.csv and all_metrics:
        save_csv(all_metrics, args.csv)

    print(f"\nDone. Processed {len(all_metrics)}/{len(bag_paths)} bags successfully.")


if __name__ == '__main__':
    main()