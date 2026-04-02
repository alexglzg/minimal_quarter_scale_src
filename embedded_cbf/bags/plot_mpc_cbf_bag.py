#!/usr/bin/env python3
"""
MPC-CBF Rosbag Analysis Script

Generates paper figures from a ROS1 bag:
  1. Trajectory progression (footprint rectangles + polytopes at key timesteps)
  2. Time-series: min_h (polytope clearance), min_slack (CBF activity),
     footprint-to-obstacle clearance (offline)
  3. Solve time statistics (printed)

Supports two modes:
  scan    — Gazebo/LiDAR experiments (uses /filtered_scan, /planning/planning/execute_path)
  gridmap — Grid-based experiments (uses /map, /global_plan)

Usage:
  python3 plot_mpc_cbf_bag.py <bag> --mode scan    [--step N] [--save]
  python3 plot_mpc_cbf_bag.py <bag> --mode gridmap  [--step N] [--save]
"""

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
from tf.transformations import euler_from_quaternion
import argparse
import os

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

def load_bag(bag_path, mode='scan'):
    """
    Load all relevant topics from the bag.

    Args:
        bag_path: path to rosbag
        mode:     'scan' (Gazebo/LiDAR) or 'gridmap' (grid-based)

    Returns dict with keys:
        odom:         list of (t, x, y, yaw, u, v, r)   — ENU
        cbf:          list of (t, min_h, min_slack)
        solve_time:   list of (t, ms)
        polytopes:    list of (t, A, b, n_active)         — ENU
        obstacle_pts: Nx2 array of obstacle points in world frame (ENU)
        gridmap:      dict with 'data','resolution','origin','width','height' or None
        ref_path:     list of (t, Nx2 array)               — ENU
        mpc_traj:     list of (t, Nx2 array)               — ENU
        mpc_ref:      list of (t, x, y, yaw)               — ENU
    """
    data = {
        'odom': [],
        'cbf': [],
        'solve_time': [],
        'polytopes': [],
        'obstacle_pts': None,
        'gridmap': None,
        'ref_path': [],
        'mpc_traj': [],
        'mpc_ref': [],
    }

    # Common topics
    topics = [
        '/odometry/filtered',
        '/mpc_stats/cbf',
        '/mpc_stats/solve_time_ms',
        '/polyhedron_array',
        '/mpc_trajectory',
        '/mpc_reference',
    ]

    # Mode-specific topics
    if mode == 'scan':
        topics.append('/filtered_scan')
        topics.append('/planning/planning/execute_path')
        ref_path_topic = '/planning/planning/execute_path'
    else:  # gridmap
        topics.append('/map')
        topics.append('/global_plan')
        ref_path_topic = '/global_plan'

    # Temporary storage for scan data (needs odom for transform)
    scan_raw = []

    print(f"Loading bag: {bag_path} (mode: {mode})")
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
                angles = np.arange(msg.angle_min,
                                   msg.angle_min + len(msg.ranges) * msg.angle_increment,
                                   msg.angle_increment)[:len(msg.ranges)]
                ranges = np.array(msg.ranges)
                valid = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)
                if np.any(valid):
                    r_valid = ranges[valid]
                    a_valid = angles[valid]
                    pts = np.column_stack([r_valid * np.cos(a_valid),
                                           r_valid * np.sin(a_valid)])
                    scan_raw.append((ts, pts))

            elif topic == '/map':
                res = msg.info.resolution
                ox = msg.info.origin.position.x
                oy = msg.info.origin.position.y
                w = msg.info.width
                h = msg.info.height
                grid = np.array(msg.data).reshape((h, w))

                data['gridmap'] = {
                    'data': grid,
                    'resolution': res,
                    'origin': (ox, oy),
                    'width': w,
                    'height': h,
                }

                # Extract occupied cell corners as obstacle points (world frame)
                occupied = np.argwhere(grid > 50)  # row, col
                if len(occupied) > 0:
                    corners = []
                    for row, col in occupied:
                        cx = ox + col * res
                        cy = oy + row * res
                        corners.extend([
                            [cx,       cy],
                            [cx + res, cy],
                            [cx,       cy + res],
                            [cx + res, cy + res],
                        ])
                    data['obstacle_pts'] = np.array(corners)
                    print(f"  Gridmap: {w}x{h}, resolution={res}m, "
                          f"{len(occupied)} occupied cells, "
                          f"{len(corners)} corner points")

            elif topic == ref_path_topic:
                pts = np.array([[p.pose.position.x, p.pose.position.y]
                                for p in msg.poses])
                if len(pts) > 0:
                    data['ref_path'].append((ts, pts))

            elif topic == '/mpc_trajectory':
                pts = np.array([[p.pose.position.x, p.pose.position.y]
                                for p in msg.poses])
                if len(pts) > 0:
                    data['mpc_traj'].append((ts, pts))

            elif topic == '/mpc_reference':
                x = msg.pose.position.x
                y = msg.pose.position.y
                q = msg.pose.orientation
                _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                data['mpc_ref'].append((ts, x, y, yaw))

    # Post-process scan data: transform to world frame and accumulate
    if mode == 'scan' and len(scan_raw) > 0 and len(data['odom']) > 0:
        odom_times = np.array([o[0] for o in data['odom']])
        all_world_pts = []
        for ts, scan_pts in scan_raw:
            oidx = np.argmin(np.abs(odom_times - ts))
            _, ox, oy, oyaw = (data['odom'][oidx][0], data['odom'][oidx][1],
                               data['odom'][oidx][2], data['odom'][oidx][3])
            world_pts = scan_to_world(scan_pts, ox, oy, oyaw)
            all_world_pts.append(world_pts)
        data['obstacle_pts'] = np.vstack(all_world_pts)
        # Keep raw scans for per-timestep clearance
        data['_scan_raw'] = scan_raw

    for key in data:
        if key.startswith('_'):
            continue
        if isinstance(data[key], list):
            print(f"  {key}: {len(data[key])} messages")
        elif isinstance(data[key], np.ndarray):
            print(f"  {key}: {data[key].shape}")
        elif data[key] is not None:
            print(f"  {key}: loaded")
        else:
            print(f"  {key}: None")

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
    """
    from scipy.spatial import HalfspaceIntersection, ConvexHull

    if n_active is not None:
        A = A[:n_active]
        b = b[:n_active]

    M = 50.0
    A_box = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b_box = np.array([M, M, M, M])
    A_all = np.vstack([A, A_box])
    b_all = np.concatenate([b, b_box])
    halfspaces = np.column_stack([A_all, -b_all])

    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except Exception:
        return None


def scan_to_world(scan_pts, x, y, yaw):
    """Transform scan points from sensor frame to world frame (ENU)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ scan_pts.T).T + np.array([x, y])


# =============================================================================
# Clearance computation (unified)
# =============================================================================

def compute_footprint_clearance(odom_data, obstacle_pts, L, W,
                                 scan_raw=None, max_scan_age=0.15):
    """
    Compute min footprint-to-obstacle distance at each odom timestep.

    For gridmap mode: obstacle_pts is static, uses KD-tree (fast).
    For scan mode: if scan_raw is provided, uses per-timestep scan
                   transformed to world frame (more accurate).

    Returns: list of (t, min_dist)
    """
    if obstacle_pts is None and scan_raw is None:
        return []

    results = []

    if scan_raw is not None:
        # Scan mode: per-timestep clearance using matched scans
        scan_times = np.array([s[0] for s in scan_raw])
        for ts, x, y, yaw, *_ in odom_data:
            idx = np.argmin(np.abs(scan_times - ts))
            if abs(scan_times[idx] - ts) > max_scan_age:
                continue
            scan_world = scan_to_world(scan_raw[idx][1], x, y, yaw)
            verts = robot_vertices(x, y, yaw, L, W)
            min_d = float('inf')
            for v in verts:
                dists = np.sqrt((scan_world[:, 0] - v[0])**2 +
                                (scan_world[:, 1] - v[1])**2)
                min_d = min(min_d, np.min(dists))
            results.append((ts, min_d))
    else:
        # Gridmap mode: static obstacle points, use KD-tree for speed
        from scipy.spatial import cKDTree
        tree = cKDTree(obstacle_pts)
        for ts, x, y, yaw, *_ in odom_data:
            verts = robot_vertices(x, y, yaw, L, W)
            dists, _ = tree.query(verts)
            results.append((ts, np.min(dists)))

    return results


# =============================================================================
# Figure 1: Trajectory progression
# =============================================================================

def plot_trajectory_progression(data, L, W, mode='scan', step_interval=100,
                                 polytope_steps=None, show_obstacles=True,
                                 show_ref_path=False, ax=None):
    """
    Trajectory with footprint rectangles and (optionally) polytopes.
    """
    odom = data['odom']
    traj = np.array([(o[1], o[2]) for o in odom])
    yaws = np.array([o[3] for o in odom])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.get_figure()

    # --- Background: obstacles ---
    if show_obstacles:
        if mode == 'gridmap' and data['gridmap'] is not None:
            # Render gridmap as image
            gm = data['gridmap']
            grid = gm['data'].astype(float)
            ox, oy = gm['origin']
            res = gm['resolution']
            w, h = gm['width'], gm['height']

            # Map values: -1=unknown, 0=free, 100=occupied
            display = np.ones_like(grid) * 0.95  # unknown = light gray
            display[grid == 0] = 1.0              # free = white
            display[grid > 50] = 0.0              # occupied = black

            extent = [ox, ox + w * res, oy, oy + h * res]
            ax.imshow(display, cmap='gray', origin='lower', extent=extent,
                      vmin=0, vmax=1, zorder=0, alpha=0.8)

        elif data['obstacle_pts'] is not None:
            # Scan: scatter accumulated points
            pts = data['obstacle_pts']
            ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='purple',
                       alpha=0.3, edgecolors='none', rasterized=True, zorder=1)

    # Trajectory line
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, zorder=3,
            label='Trajectory')

    # Reference path (latest)
    if show_ref_path and len(data['ref_path']) > 0:
        last_path = data['ref_path'][-1][1]
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

        robot_pos = np.array([traj[idx, 0], traj[idx, 1]])
        _, A, b, n_act = data['polytopes'][pidx]
        verts = polytope_to_polygon(A, b, robot_pos, n_active=n_act)

        progress = i / max(len(polytope_steps) - 1, 1)
        poly_alpha = 0.08 + 0.20 * progress

        if verts is not None:
            poly_patch = plt.Polygon(verts, alpha=poly_alpha, facecolor='green',
                                     edgecolor='green', linewidth=1.0, zorder=2)
            ax.add_patch(poly_patch)
            n_poly_drawn += 1
        else:
            n_poly_failed += 1

    print(f"  Polytopes drawn: {n_poly_drawn}, failed: {n_poly_failed}")

    # Footprint rectangles with time-varying alpha (zorder=4, above polytopes)
    for i, idx in enumerate(foot_indices):
        x, y, yaw = traj[idx, 0], traj[idx, 1], yaws[idx]
        progress = i / max(len(foot_indices) - 1, 1)
        foot_alpha = 0.3 + 0.7 * progress

        rect = patches.Rectangle((-L/2, -W/2), L, W,
                                  facecolor='orange', edgecolor='darkorange',
                                  alpha=foot_alpha, linewidth=1, zorder=4)
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

def plot_timeseries(data, footprint_clearance=None, ax_h=None, ax_s=None):
    """
    Two-subplot time-series figure.

    Top:    min_h (footprint-to-polytope) and footprint-to-obstacle clearance
    Bottom: min_slack (CBF activity at k=0)
    """
    cbf = data['cbf']
    if len(cbf) == 0:
        print("No CBF data in bag")
        return None, None, None

    cbf_arr = np.array(cbf)
    t_cbf = cbf_arr[:, 0]
    min_h = cbf_arr[:, 1]
    min_slack = cbf_arr[:, 2]

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

    if footprint_clearance is not None and len(footprint_clearance) > 0:
        fc = np.array(footprint_clearance)
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
# Statistics (printed)
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


def print_polytope_stats(data):
    """Print polytope halfplane count statistics."""
    polys = data['polytopes']
    if len(polys) == 0:
        print("No polytope data in bag")
        return

    n_planes = np.array([p[3] for p in polys])
    print(f"\nPolytope statistics ({len(polys)} polytopes):")
    print(f"  Mean:   {np.mean(n_planes):.1f} halfplanes")
    print(f"  Std:    {np.std(n_planes):.1f}")
    print(f"  Median: {np.median(n_planes):.0f}")
    print(f"  Max:    {np.max(n_planes)}")
    print(f"  Min:    {np.min(n_planes)}")

    # Distribution
    unique, counts = np.unique(n_planes, return_counts=True)
    print(f"  Distribution:")
    for u, c in zip(unique, counts):
        pct = 100 * c / len(n_planes)
        print(f"    {int(u):3d} planes: {c:4d} ({pct:5.1f}%)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='MPC-CBF Rosbag Analysis')
    parser.add_argument('bag', help='Path to rosbag')
    parser.add_argument('--mode', choices=['scan', 'gridmap'], default='scan',
                        help='Data source: scan (Gazebo/LiDAR) or gridmap')
    parser.add_argument('--step', type=int, default=1000,
                        help='Footprint rectangle interval (odom messages)')
    parser.add_argument('--robot-length', type=float, default=0.9,
                        help='Robot length [m]')
    parser.add_argument('--robot-width', type=float, default=0.45,
                        help='Robot width [m]')
    parser.add_argument('--poly-steps', type=int, nargs='*', default=None,
                        help='Specific odom indices for polytope overlay')
    parser.add_argument('--no-obstacles', action='store_true',
                        help='Skip obstacle loading (faster)')
    parser.add_argument('--save', action='store_true',
                        help='Save figures as PDF and PNG')
    args = parser.parse_args()

    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    L, W = args.robot_length, args.robot_width

    # Load data
    data = load_bag(args.bag, mode=args.mode)

    # Print statistics
    print_solve_time_stats(data)
    print_cbf_stats(data)
    print_polytope_stats(data)

    # Compute footprint-to-obstacle clearance
    fc_clearance = None
    if not args.no_obstacles and data['obstacle_pts'] is not None:
        print("\nComputing footprint-to-obstacle clearance...")
        odom_ds = data['odom'][::10]  # ~10 Hz

        scan_raw = data.get('_scan_raw', None) if args.mode == 'scan' else None
        fc_clearance = compute_footprint_clearance(
            odom_ds, data['obstacle_pts'], L, W, scan_raw=scan_raw)

        print(f"  Computed {len(fc_clearance)} clearance values")
        if len(fc_clearance) > 0:
            fc_arr = np.array(fc_clearance)
            print(f"  Min footprint-to-obstacle: {np.min(fc_arr[:, 1]):.4f} m")

    # Figure 1: Trajectory progression
    fig1, ax1 = plot_trajectory_progression(
        data, L, W,
        mode=args.mode,
        step_interval=args.step,
        polytope_steps=args.poly_steps,
        show_obstacles=not args.no_obstacles,
        show_ref_path=True,
    )
    ax1.set_title(f'Trajectory: {bag_name}')
    if args.save:
        fig1.savefig(f'{bag_name}_trajectory.pdf', dpi=300)
        fig1.savefig(f'{bag_name}_trajectory.png', dpi=300)
        print(f"Saved: {bag_name}_trajectory.pdf/png")

    # Figure 2: Time-series
    fig2, ax_h, ax_s = plot_timeseries(data, footprint_clearance=fc_clearance)
    if fig2 is not None:
        if args.save:
            fig2.savefig(f'{bag_name}_timeseries.pdf', dpi=300)
            fig2.savefig(f'{bag_name}_timeseries.png', dpi=300)
            print(f"Saved: {bag_name}_timeseries.pdf/png")

    plt.show()


if __name__ == '__main__':
    main()