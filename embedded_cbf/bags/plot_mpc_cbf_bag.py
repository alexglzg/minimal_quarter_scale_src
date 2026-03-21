#!/usr/bin/env python3
"""
MPC-CBF Rosbag Analysis Script

Generates paper figures from a ROS1 bag:
  1. Trajectory progression (footprint rectangles + polytopes at key timesteps)
  2. Time-series: min_h (polytope clearance), min_slack (CBF activity),
     footprint-to-obstacle clearance (offline from LaserScan)
  3. Solve time statistics (printed, not plotted)

Usage:
  python3 plot_mpc_cbf_bag.py <path_to_bag> [--step N] [--no-scan]
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection, LineCollection
from tf.transformations import euler_from_quaternion
import argparse
import os
import sys

# =============================================================================
# Matplotlib config for publication
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12

# =============================================================================
# Data loading
# =============================================================================

def load_bag(bag_path, load_scan=True):
    """
    Load all relevant topics from the bag.

    Returns dict with keys:
        odom:       list of (t, x, y, yaw, u, v, r)   — ENU frame
        cbf:        list of (t, min_h, min_slack)
        solve_time: list of (t, ms)
        polytopes:  list of (t, A, b, n_active)         — as received (ENU normals)
        scan:       list of (t, Nx2 array)               — ENU, in sensor frame
        mpc_traj:   list of (t, Nx2 array)               — ENU
        exec_path:  list of (t, Nx2 array)               — ENU
        mpc_ref:    list of (t, x, y, yaw)               — ENU
    """
    data = {
        'odom': [],
        'cbf': [],
        'solve_time': [],
        'polytopes': [],
        'scan': [],
        'mpc_traj': [],
        'exec_path': [],
        'mpc_ref': [],
    }

    topics = [
        '/odometry/filtered',
        '/mpc_stats/cbf',
        '/mpc_stats/solve_time_ms',
        '/polyhedron_array',
        '/mpc_trajectory',
        '/planning/planning/execute_path',
        '/mpc_reference',
    ]
    if load_scan:
        topics.append('/filtered_scan')

    print(f"Loading bag: {bag_path}")
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            ts = t.to_sec()

            if topic == '/odometry/filtered':
                x = msg.pose.pose.position.x
                y = msg.pose.pose.position.y
                q = msg.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                u = msg.twist.twist.linear.x
                v = msg.twist.twist.linear.y
                r = msg.twist.twist.angular.z
                data['odom'].append((ts, x, y, yaw, u, v, r))

            elif topic == '/mpc_stats/cbf':
                if len(msg.data) >= 2:
                    data['cbf'].append((ts, msg.data[0], msg.data[1]))

            elif topic == '/mpc_stats/solve_time_ms':
                data['solve_time'].append((ts, msg.data))

            elif topic == '/polyhedron_array':
                if len(msg.polyhedrons) > 0:
                    poly = msg.polyhedrons[0]
                    normals = []
                    points = []
                    for i in range(len(poly.normals)):
                        if abs(poly.normals[i].z) > 0.1:
                            continue
                        normals.append([poly.normals[i].x, poly.normals[i].y])
                        points.append([poly.points[i].x, poly.points[i].y])
                    if len(normals) > 0:
                        A = np.array(normals)
                        pts = np.array(points)
                        b = np.sum(A * pts, axis=1)
                        data['polytopes'].append((ts, A, b, len(normals)))

            elif topic == '/filtered_scan':
                # Convert LaserScan to 2D points in sensor frame
                angles = np.arange(msg.angle_min,
                                   msg.angle_min + len(msg.ranges) * msg.angle_increment,
                                   msg.angle_increment)[:len(msg.ranges)]
                ranges = np.array(msg.ranges)
                # Filter invalid ranges
                valid = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)
                if np.any(valid):
                    r_valid = ranges[valid]
                    a_valid = angles[valid]
                    pts = np.column_stack([r_valid * np.cos(a_valid),
                                           r_valid * np.sin(a_valid)])
                    data['scan'].append((ts, pts))

            elif topic == '/mpc_trajectory':
                pts = np.array([[p.pose.position.x, p.pose.position.y]
                                for p in msg.poses])
                if len(pts) > 0:
                    data['mpc_traj'].append((ts, pts))

            elif topic == '/planning/planning/execute_path':
                pts = np.array([[p.pose.position.x, p.pose.position.y]
                                for p in msg.poses])
                if len(pts) > 0:
                    data['exec_path'].append((ts, pts))

            elif topic == '/mpc_reference':
                x = msg.pose.position.x
                y = msg.pose.position.y
                q = msg.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                data['mpc_ref'].append((ts, x, y, yaw))

    for key in data:
        print(f"  {key}: {len(data[key])} messages")

    return data


# =============================================================================
# Geometry helpers
# =============================================================================

def robot_vertices(x, y, yaw, L, W):
    """Robot corners in world frame (ENU). Returns (4, 2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[L/2, W/2], [L/2, -W/2], [-L/2, -W/2], [-L/2, W/2]])
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([x, y])


def polytope_to_polygon(A, b, interior_point, n_active=None):
    """
    Convert halfplane representation {x: Ax <= b} to a polygon.
    Uses the robot position as interior point (guaranteed inside by FIRI).

    Args:
        A:              (n, 2) normals
        b:              (n,) offsets
        interior_point: (2,) point known to be inside (robot position)
        n_active:       number of active halfplanes

    Returns vertices as (N, 2) array, or None if failed.
    """
    from scipy.spatial import HalfspaceIntersection, ConvexHull

    if n_active is not None:
        A = A[:n_active]
        b = b[:n_active]

    # Add large bounding box to ensure bounded intersection
    M = 50.0
    A_box = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_box = np.array([M, M, M, M])

    A_all = np.vstack([A, A_box])
    b_all = np.concatenate([b, b_box])

    # scipy halfspace format: A @ x - b <= 0
    halfspaces = np.column_stack([A_all, -b_all])

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except Exception as e:
        return None


def scan_to_world(scan_pts, x, y, yaw):
    """Transform scan points from sensor frame to world frame (ENU).
    Assumes sensor is at robot center (adjust offset if needed)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ scan_pts.T).T + np.array([x, y])


def compute_footprint_to_scan_clearance(odom_data, scan_data, L, W,
                                         max_scan_age=0.15):
    """
    For each odom timestep, find the closest scan in time,
    transform scan points to world frame, and compute min distance
    from any robot vertex to any scan point.

    Returns: list of (t, min_dist)
    """
    if len(scan_data) == 0:
        return []

    scan_times = np.array([s[0] for s in scan_data])
    results = []

    for ts, x, y, yaw, *_ in odom_data:
        # Find nearest scan
        idx = np.argmin(np.abs(scan_times - ts))
        if abs(scan_times[idx] - ts) > max_scan_age:
            continue

        scan_world = scan_to_world(scan_data[idx][1], x, y, yaw)
        verts = robot_vertices(x, y, yaw, L, W)

        # Min distance from any vertex to any scan point
        min_d = float('inf')
        for v in verts:
            dists = np.sqrt((scan_world[:, 0] - v[0])**2 +
                            (scan_world[:, 1] - v[1])**2)
            min_d = min(min_d, np.min(dists))

        results.append((ts, min_d))

    return results


# =============================================================================
# Figure 1: Trajectory progression
# =============================================================================

def plot_trajectory_progression(data, L, W, step_interval=100,
                                 polytope_steps=None, show_scan=True,
                                 show_exec_path=False, ax=None):
    """
    Trajectory with footprint rectangles and (optionally) polytopes.

    Args:
        data:           dict from load_bag
        L, W:           robot length, width
        step_interval:  draw footprint every N odom messages
        polytope_steps: list of odom indices where to overlay polytopes
                        (None = same as footprint steps)
        show_scan:      overlay accumulated scan points (transformed to world)
        show_exec_path: overlay the local planner path
        ax:             matplotlib axis (created if None)
    """
    odom = data['odom']
    traj = np.array([(o[1], o[2]) for o in odom])
    yaws = np.array([o[3] for o in odom])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Scan points (transformed to world frame)
    if show_scan and len(data['scan']) > 0:
        odom_times = np.array([o[0] for o in odom])
        all_world_pts = []
        for ts, scan_pts in data['scan']:
            # Find nearest odom for pose
            oidx = np.argmin(np.abs(odom_times - ts))
            _, ox, oy, oyaw = odom[oidx][0], odom[oidx][1], odom[oidx][2], odom[oidx][3]
            world_pts = scan_to_world(scan_pts, ox, oy, oyaw)
            all_world_pts.append(world_pts)
        all_world_pts = np.vstack(all_world_pts)
        ax.scatter(all_world_pts[:, 0], all_world_pts[:, 1], s=0.5, c='purple',
                   alpha=0.3, edgecolors='none', rasterized=True, zorder=1)

    # Trajectory line
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, zorder=3,
            label='Trajectory')

    # Reference path (latest execute_path)
    if show_exec_path and len(data['exec_path']) > 0:
        last_path = data['exec_path'][-1][1]
        ax.plot(last_path[:, 0], last_path[:, 1], 'g--', linewidth=1.5,
                alpha=0.7, label='Reference path', zorder=2)

    # Footprint indices
    foot_indices = list(range(0, len(odom), step_interval))
    if len(odom) - 1 not in foot_indices:
        foot_indices.append(len(odom) - 1)

    if polytope_steps is None:
        polytope_steps = foot_indices

    # Polytope overlays
    poly_times = np.array([p[0] for p in data['polytopes']]) if data['polytopes'] else np.array([])
    odom_times = np.array([o[0] for o in odom])

    n_poly_drawn = 0
    n_poly_failed = 0
    for i, idx in enumerate(polytope_steps):
        if len(poly_times) == 0:
            break
        t_odom = odom_times[idx]
        pidx = np.argmin(np.abs(poly_times - t_odom))
        if abs(poly_times[pidx] - t_odom) > 0.2:
            continue

        # Use robot position as interior point (guaranteed inside by FIRI)
        robot_pos = np.array([traj[idx, 0], traj[idx, 1]])
        _, A, b, n_act = data['polytopes'][pidx]
        verts = polytope_to_polygon(A, b, robot_pos, n_active=n_act)

        # Time-varying alpha: faint early, stronger late
        progress = i / max(len(polytope_steps) - 1, 1)
        poly_alpha = 0.05 + 0.10 * progress

        if verts is not None:
            poly_patch = plt.Polygon(verts, alpha=poly_alpha, facecolor='green',
                                     edgecolor='green', linewidth=1.0, zorder=2)
            ax.add_patch(poly_patch)
            n_poly_drawn += 1
        else:
            n_poly_failed += 1

    print(f"  Polytopes drawn: {n_poly_drawn}, failed: {n_poly_failed}")

    # Footprint rectangles with time-varying alpha
    for i, idx in enumerate(foot_indices):
        x, y, yaw = traj[idx, 0], traj[idx, 1], yaws[idx]

        progress = i / max(len(foot_indices) - 1, 1)
        foot_alpha = 0.7 + 0.3 * progress

        rect = patches.Rectangle((-L/2, -W/2), L, W,
                                  facecolor='orange', edgecolor='darkorange',
                                  alpha=foot_alpha, linewidth=1)
        t = mtransforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig, ax


# =============================================================================
# Figure 2: Time-series (clearance + CBF slack)
# =============================================================================

def plot_timeseries(data, L, W, footprint_scan_clearance=None, ax_h=None, ax_s=None):
    """
    Two-subplot time-series figure.

    Top:    min_h (footprint-to-polytope) and footprint-to-obstacle clearance
    Bottom: min_slack (CBF activity at k=0)

    Args:
        data:                       dict from load_bag
        L, W:                       robot dimensions
        footprint_scan_clearance:   precomputed list of (t, min_dist), or None to compute
        ax_h, ax_s:                 axes for top/bottom (created if None)
    """
    cbf = data['cbf']
    if len(cbf) == 0:
        print("No CBF data in bag")
        return None, None, None

    cbf_arr = np.array(cbf)
    t_cbf = cbf_arr[:, 0]
    min_h = cbf_arr[:, 1]
    min_slack = cbf_arr[:, 2]

    # Normalize time to start at 0
    t0 = t_cbf[0]
    t_cbf_rel = t_cbf - t0

    if ax_h is None or ax_s is None:
        fig, (ax_h, ax_s) = plt.subplots(2, 1, figsize=(8, 5),
                                          constrained_layout=True, sharex=True)
    else:
        fig = ax_h.get_figure()

    # --- Top: clearance ---
    ax_h.plot(t_cbf_rel, min_h, 'b-', linewidth=1.5,
              label='Footprint–polytope (MPC)')

    if footprint_scan_clearance is not None and len(footprint_scan_clearance) > 0:
        fc = np.array(footprint_scan_clearance)
        ax_h.plot(fc[:, 0] - t0, fc[:, 1], 'r-', linewidth=1, alpha=0.7,
                  label='Footprint–obstacle (true)')

    ax_h.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_h.set_ylabel('Clearance [m]')
    ax_h.legend(fontsize=10)
    ax_h.grid(True, alpha=0.3)

    # --- Bottom: CBF slack ---
    ax_s.plot(t_cbf_rel, min_slack, 'g-', linewidth=1.5,
              label=r'CBF slack $h(x_1) - \gamma h(x_0)$')
    ax_s.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_s.set_xlabel('Time [s]')
    ax_s.set_ylabel('CBF slack [m]')
    ax_s.legend(fontsize=10)
    ax_s.grid(True, alpha=0.3)

    return fig, ax_h, ax_s


# =============================================================================
# Solve time statistics
# =============================================================================

def print_solve_time_stats(data):
    """Print solve time statistics."""
    st = data['solve_time']
    if len(st) == 0:
        print("No solve time data in bag")
        return

    times = np.array([s[1] for s in st])
    print(f"\nSolve time statistics ({len(times)} solves):")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  95th:   {np.percentile(times, 95):.2f} ms")
    print(f"  99th:   {np.percentile(times, 99):.2f} ms")


def print_cbf_stats(data):
    """Print CBF clearance and slack statistics."""
    cbf = data['cbf']
    if len(cbf) == 0:
        print("No CBF data in bag")
        return

    cbf_arr = np.array(cbf)
    min_h = cbf_arr[:, 1]
    min_slack = cbf_arr[:, 2]

    print(f"\nCBF statistics ({len(cbf)} timesteps):")
    print(f"  min_h  — min: {np.min(min_h):.4f} m, mean: {np.mean(min_h):.4f} m")
    print(f"  slack  — min: {np.min(min_slack):.4f} m, mean: {np.mean(min_slack):.4f} m")
    if np.min(min_h) < 0:
        print("  WARNING: negative min_h detected — polytope was violated!")
    if np.min(min_slack) < 0:
        print("  WARNING: negative slack detected — CBF constraint was violated!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MPC-CBF Rosbag Analysis')
    parser.add_argument('bag', help='Path to rosbag')
    parser.add_argument('--step', type=int, default=1000,
                        help='Footprint rectangle interval (odom messages)')
    parser.add_argument('--no-scan', action='store_true',
                        help='Skip LaserScan loading (faster)')
    parser.add_argument('--robot-length', type=float, default=0.9,
                        help='Robot length [m]')
    parser.add_argument('--robot-width', type=float, default=0.45,
                        help='Robot width [m]')
    parser.add_argument('--poly-steps', type=int, nargs='*', default=None,
                        help='Specific odom indices for polytope overlay')
    parser.add_argument('--save', action='store_true',
                        help='Save figures as PDF and PNG')
    args = parser.parse_args()

    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    load_scan = not args.no_scan
    L, W = args.robot_length, args.robot_width

    # Load data
    data = load_bag(args.bag, load_scan=load_scan)

    # Print statistics
    print_solve_time_stats(data)
    print_cbf_stats(data)

    # Compute footprint-to-obstacle clearance (offline from scan)
    fc_clearance = None
    if load_scan and len(data['scan']) > 0:
        print("\nComputing footprint-to-obstacle clearance from scan data...")
        # Downsample odom to ~10 Hz for speed (match MPC rate)
        odom_ds = data['odom'][::10]
        fc_clearance = compute_footprint_to_scan_clearance(
            odom_ds, data['scan'], L, W)
        print(f"  Computed {len(fc_clearance)} clearance values")
        if len(fc_clearance) > 0:
            fc_arr = np.array(fc_clearance)
            print(f"  Min footprint-to-obstacle: {np.min(fc_arr[:, 1]):.4f} m")

    # Figure 1: Trajectory progression
    fig1, ax1 = plot_trajectory_progression(
        data, L, W,
        step_interval=args.step,
        polytope_steps=args.poly_steps,
        show_scan=load_scan,
        show_exec_path=True
    )
    ax1.set_title(f'Trajectory: {bag_name}')
    if args.save:
        fig1.savefig(f'{bag_name}_trajectory.pdf', dpi=300)
        fig1.savefig(f'{bag_name}_trajectory.png', dpi=300)
        print(f"Saved: {bag_name}_trajectory.pdf/png")

    # Figure 2: Time-series
    fig2, ax_h, ax_s = plot_timeseries(data, L, W,
                                        footprint_scan_clearance=fc_clearance)
    if fig2 is not None:
        if args.save:
            fig2.savefig(f'{bag_name}_timeseries.pdf', dpi=300)
            fig2.savefig(f'{bag_name}_timeseries.png', dpi=300)
            print(f"Saved: {bag_name}_timeseries.pdf/png")

    plt.show()


if __name__ == '__main__':
    main()


# python3 plot_mpc_cbf_bag.py 2026-03-20-15-04-21.bag
