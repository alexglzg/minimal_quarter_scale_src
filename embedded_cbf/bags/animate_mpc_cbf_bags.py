#!/usr/bin/env python3
"""
MPC-CBF Rosbag Animation Script

Animates the robot trajectory, footprint, FIRI polytope, and obstacle data
from a ROS1 bag. Supports scan and gridmap modes.

Uses load_bag and geometry helpers from plot_mpc_cbf_bag.py.

Usage:
  python3 animate_mpc_cbf_bag.py <bag> --mode scan    [--save] [--fps 20]
  python3 animate_mpc_cbf_bag.py <bag> --mode gridmap  [--save] [--fps 20]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation
import argparse
import os

from plot_mpc_cbf_bag import (load_bag, robot_vertices, polytope_to_polygon,
                               scan_to_world)

# =============================================================================
# Matplotlib config
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 12


def build_animation(data, L, W, mode='scan', fps=20, xlim=None, ylim=None):
    """
    Build a matplotlib animation from bag data.

    The animation is driven by polytope timestamps (~10Hz), which matches
    the MPC rate. At each frame we show:
      - Background (gridmap or accumulated scan)
      - Trajectory up to current time
      - Current FIRI polytope
      - Robot footprint at current pose
      - Reference path (latest)
      - MPC predicted trajectory (latest before this frame)

    Returns (fig, ani).
    """
    odom = data['odom']
    traj_all = np.array([(o[1], o[2]) for o in odom])
    yaws_all = np.array([o[3] for o in odom])
    odom_times = np.array([o[0] for o in odom])

    # Use polytope timestamps as frame driver
    poly_times = np.array([p[0] for p in data['polytopes']])
    n_frames = len(poly_times)

    if n_frames == 0:
        print("No polytope data — cannot animate")
        return None, None

    # Precompute: for each polytope frame, find nearest odom, mpc_traj, ref_path
    odom_indices = np.searchsorted(odom_times, poly_times, side='right') - 1
    odom_indices = np.clip(odom_indices, 0, len(odom) - 1)

    mpc_traj_times = np.array([m[0] for m in data['mpc_traj']]) if data['mpc_traj'] else np.array([])
    ref_path_times = np.array([r[0] for r in data['ref_path']]) if data['ref_path'] else np.array([])

    # Auto axis limits from trajectory with margin
    if xlim is None:
        xlim = (traj_all[:, 0].min() - 2, traj_all[:, 0].max() + 2)
    if ylim is None:
        ylim = (traj_all[:, 1].min() - 2, traj_all[:, 1].max() + 2)

    # --- Set up figure ---
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)

    # Static background
    if mode == 'gridmap' and data['gridmap'] is not None:
        gm = data['gridmap']
        grid = gm['data'].astype(float)
        ox, oy = gm['origin']
        res = gm['resolution']
        w, h = gm['width'], gm['height']
        display = np.ones_like(grid) * 0.95
        display[grid == 0] = 1.0
        display[grid > 50] = 0.0
        extent = [ox, ox + w * res, oy, oy + h * res]
        ax.imshow(display, cmap='gray', origin='lower', extent=extent,
                  vmin=0, vmax=1, zorder=0, alpha=0.8)

    # Dynamic elements
    traj_line, = ax.plot([], [], 'b-', linewidth=2, zorder=3)
    ref_line, = ax.plot([], [], 'g--', linewidth=1.5, alpha=0.7, zorder=2)
    mpc_line, = ax.plot([], [], 'c-', linewidth=1.5, alpha=0.8, zorder=3)

    scan_scatter = ax.scatter([], [], s=1, c='purple', alpha=0.5,
                               edgecolors='none', rasterized=True, zorder=1)

    # Polytope patch (will be replaced each frame)
    poly_patch = [None]

    # Robot footprint
    robot_rect = patches.Rectangle((-L/2, -W/2), L, W,
                                    facecolor='orange', edgecolor='darkorange',
                                    alpha=0.9, linewidth=1.5, zorder=5)
    ax.add_patch(robot_rect)

    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # CBF text
    cbf_text = ax.text(0.02, 0.90, '', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Precompute scan frames if in scan mode
    scan_raw = data.get('_scan_raw', None)

    # CBF data for display
    cbf_times = np.array([c[0] for c in data['cbf']]) if data['cbf'] else np.array([])

    def init():
        traj_line.set_data([], [])
        ref_line.set_data([], [])
        mpc_line.set_data([], [])
        time_text.set_text('')
        cbf_text.set_text('')
        return traj_line, ref_line, mpc_line, robot_rect, time_text, cbf_text

    def update(frame):
        t_now = poly_times[frame]
        t0 = poly_times[0]
        oidx = odom_indices[frame]

        # Trajectory up to now
        traj_line.set_data(traj_all[:oidx+1, 0], traj_all[:oidx+1, 1])

        # Robot pose
        rx, ry, ryaw = traj_all[oidx, 0], traj_all[oidx, 1], yaws_all[oidx]
        t = mtransforms.Affine2D().rotate(ryaw).translate(rx, ry) + ax.transData
        robot_rect.set_transform(t)

        # Polytope
        if poly_patch[0] is not None:
            poly_patch[0].remove()
            poly_patch[0] = None

        _, A, b, n_act = data['polytopes'][frame]
        robot_pos = np.array([rx, ry])
        verts = polytope_to_polygon(A, b, robot_pos, n_active=n_act)
        if verts is not None:
            p = plt.Polygon(verts, alpha=0.2, facecolor='green',
                            edgecolor='green', linewidth=1.5, zorder=2)
            ax.add_patch(p)
            poly_patch[0] = p

        # Scan points (current frame, transformed to world)
        if mode == 'scan' and scan_raw is not None:
            scan_times = np.array([s[0] for s in scan_raw])
            sidx = np.argmin(np.abs(scan_times - t_now))
            if abs(scan_times[sidx] - t_now) < 0.2:
                world_pts = scan_to_world(scan_raw[sidx][1], rx, ry, ryaw)
                scan_scatter.set_offsets(world_pts)
            else:
                scan_scatter.set_offsets(np.empty((0, 2)))

        # MPC predicted trajectory
        if len(mpc_traj_times) > 0:
            midx = np.searchsorted(mpc_traj_times, t_now, side='right') - 1
            midx = max(0, midx)
            if abs(mpc_traj_times[midx] - t_now) < 0.2:
                mpc_pts = data['mpc_traj'][midx][1]
                mpc_line.set_data(mpc_pts[:, 0], mpc_pts[:, 1])
            else:
                mpc_line.set_data([], [])
        else:
            mpc_line.set_data([], [])

        # Reference path (latest before now)
        if len(ref_path_times) > 0:
            ridx = np.searchsorted(ref_path_times, t_now, side='right') - 1
            ridx = max(0, ridx)
            ref_pts = data['ref_path'][ridx][1]
            ref_line.set_data(ref_pts[:, 0], ref_pts[:, 1])
        else:
            ref_line.set_data([], [])

        # Time display
        time_text.set_text(f't = {t_now - t0:.1f} s')

        # CBF display
        if len(cbf_times) > 0:
            cidx = np.argmin(np.abs(cbf_times - t_now))
            if abs(cbf_times[cidx] - t_now) < 0.2:
                _, min_h, min_slack = data['cbf'][cidx]
                cbf_text.set_text(f'h = {min_h:.3f} m\nslack = {min_slack:.3f} m')
            else:
                cbf_text.set_text('')
        else:
            cbf_text.set_text('')

        return traj_line, ref_line, mpc_line, robot_rect, time_text, cbf_text

    interval = 1000.0 / fps
    ani = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=n_frames, interval=interval,
                                   blit=False)
    return fig, ani


def main():
    parser = argparse.ArgumentParser(description='MPC-CBF Rosbag Animation')
    parser.add_argument('bag', help='Path to rosbag')
    parser.add_argument('--mode', choices=['scan', 'gridmap'], default='scan',
                        help='Data source: scan (Gazebo/LiDAR) or gridmap')
    parser.add_argument('--robot-length', type=float, default=0.9,
                        help='Robot length [m]')
    parser.add_argument('--robot-width', type=float, default=0.45,
                        help='Robot width [m]')
    parser.add_argument('--fps', type=int, default=20,
                        help='Animation frames per second')
    parser.add_argument('--save', action='store_true',
                        help='Save as MP4')
    parser.add_argument('--xlim', type=float, nargs=2, default=None,
                        help='X axis limits: --xlim xmin xmax')
    parser.add_argument('--ylim', type=float, nargs=2, default=None,
                        help='Y axis limits: --ylim ymin ymax')
    args = parser.parse_args()

    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    L, W = args.robot_length, args.robot_width

    data = load_bag(args.bag, mode=args.mode)

    fig, ani = build_animation(data, L, W, mode=args.mode, fps=args.fps,
                                xlim=args.xlim, ylim=args.ylim)

    if ani is None:
        return

    if args.save:
        out_path = f'{bag_name}_animation.mp4'
        print(f"Saving animation to {out_path}...")
        ani.save(out_path, writer='ffmpeg', dpi=200)
        print("Done.")

    plt.show()


if __name__ == '__main__':
    main()