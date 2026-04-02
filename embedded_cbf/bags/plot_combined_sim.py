#!/usr/bin/env python3
"""
Combined Simulation Figure for RA-L Paper

Creates a single-column figure with two panels:
  (a) Gridmap-based vessel simulation
  (b) LiDAR-based vessel simulation (Gazebo)

Outputs both side-by-side and stacked layouts.

Usage:
  python3 plot_combined_sim.py <gridmap_bag> <scan_bag> [options]

Example:
  python3 plot_combined_sim.py map_test.bag gazebo_test.bag \
      --step-a 200 --step-b 150 --save
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import argparse
import os
import sys

# Import from existing plotting script
from plot_mpc_cbf_bag import (load_bag, robot_vertices, polytope_to_polygon,
                               scan_to_world)

# =============================================================================
# Matplotlib config for publication (single column)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['axes.titlesize'] = 9


# =============================================================================
# Panel plotting function
# =============================================================================

def plot_panel(ax, data, L, W, mode, step_interval=1000,
               polytope_steps=None, label='(a)'):
    """
    Plot a single panel: trajectory, footprints, polytopes, start/end markers.

    Args:
        ax:              matplotlib axis
        data:            dict from load_bag
        L, W:            robot length, width
        mode:            'scan' or 'gridmap'
        step_interval:   footprint every N odom messages
        polytope_steps:  list of odom indices for polytope overlay (None = auto)
        label:           panel label, e.g. '(a)' or '(b)'
    """
    odom = data['odom']
    traj = np.array([(o[1], o[2]) for o in odom])
    yaws = np.array([o[3] for o in odom])

    # --- Background: obstacles ---
    if mode == 'gridmap' and data['gridmap'] is not None:
        gm = data['gridmap']
        grid = gm['data'].astype(float)
        ox, oy = gm['origin']
        res = gm['resolution']
        w, h = gm['width'], gm['height']

        # Cell-center coordinates for each column/row
        X = ox + (np.arange(w) + 0.5) * res
        Y = oy + (np.arange(h) + 0.5) * res

        # Vector-filled contours: contourf produces true vector polygons in PDF,
        # eliminating the staircase aliasing from imshow's raster bitmap.
        occupied = (grid > 50).astype(float)
        unknown  = ((grid > 0) & (grid <= 50)).astype(float)

        ax.set_facecolor('white')
        ax.contourf(X, Y, unknown,  levels=[0.5, 1.5],
                    colors=['#E8E8E8'], alpha=1.0, zorder=0)
        ax.contourf(X, Y, occupied, levels=[0.5, 1.5],
                    colors=['#1a1a1a'], alpha=0.85, zorder=1)

    elif mode == 'scan' and data['obstacle_pts'] is not None:
        pts = data['obstacle_pts']
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c='purple',
                   alpha=0.4, edgecolors='none', rasterized=True, zorder=0)

    # Trajectory line
    ax.plot(traj[:, 0], traj[:, 1], 'b--', linewidth=1.2, zorder=3)

    # Footprint indices
    foot_indices = list(range(0, len(odom), step_interval))
    if len(odom) - 1 not in foot_indices:
        foot_indices.append(len(odom) - 1)

    if polytope_steps is None:
        polytope_steps = foot_indices

    # Polytope overlays
    poly_times = np.array([p[0] for p in data['polytopes']]) if data['polytopes'] else np.array([])
    odom_times = np.array([o[0] for o in odom])

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
        poly_alpha = 0.08 + 0.18 * progress

        if verts is not None:
            poly_patch = plt.Polygon(verts, alpha=poly_alpha, facecolor='green',
                                     edgecolor='green', linewidth=0.6, zorder=2)
            ax.add_patch(poly_patch)

    # Footprint rectangles with time-varying alpha
    for i, idx in enumerate(foot_indices):
        x, y, yaw = traj[idx, 0], traj[idx, 1], yaws[idx]
        progress = i / max(len(foot_indices) - 1, 1)
        foot_alpha = 0.3 + 0.7 * progress

        rect = patches.Rectangle((-L/2, -W/2), L, W,
                                  facecolor='orange', edgecolor='darkorange',
                                  alpha=foot_alpha, linewidth=0.6, zorder=4)
        t = mtransforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # Start and end markers
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=2, zorder=6,
            markeredgecolor='darkgreen', markeredgewidth=0.8)
    ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=3, zorder=6,
            markeredgecolor='darkred', markeredgewidth=0.5)

    # Panel label
    ax.text(0.02, 0.07, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.8))

    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.grid(False, alpha=0.2, linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Tight axis limits with small margin
    margin = 1.0
    ax.set_xlim(traj[:, 0].min() - margin, traj[:, 0].max() + margin)
    ax.set_ylim(traj[:, 1].min() - margin, traj[:, 1].max() + margin)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined Simulation Figure for RA-L')
    parser.add_argument('bag_gridmap', help='Gridmap bag path')
    parser.add_argument('bag_scan', help='Scan/Gazebo bag path')

    # Robot dimensions
    parser.add_argument('--robot-length', type=float, default=0.9)
    parser.add_argument('--robot-width', type=float, default=0.45)

    # Per-panel step intervals
    parser.add_argument('--step-a', type=int, default=1000,
                        help='Footprint interval for panel (a) gridmap')
    parser.add_argument('--step-b', type=int, default=1000,
                        help='Footprint interval for panel (b) scan')

    # Per-panel polytope indices (optional)
    parser.add_argument('--poly-a', type=int, nargs='*', default=None,
                        help='Odom indices for polytope overlay in panel (a)')
    parser.add_argument('--poly-b', type=int, nargs='*', default=None,
                        help='Odom indices for polytope overlay in panel (b)')

    # Axis limits (optional, auto if not given)
    parser.add_argument('--xlim-a', type=float, nargs=2, default=None)
    parser.add_argument('--ylim-a', type=float, nargs=2, default=None)
    parser.add_argument('--xlim-b', type=float, nargs=2, default=None)
    parser.add_argument('--ylim-b', type=float, nargs=2, default=None)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--prefix', type=str, default='sim_combined',
                        help='Output filename prefix')
    args = parser.parse_args()

    L, W = args.robot_length, args.robot_width

    # Load both bags
    print("Loading gridmap bag...")
    data_grid = load_bag(args.bag_gridmap, mode='gridmap')
    print("\nLoading scan bag...")
    data_scan = load_bag(args.bag_scan, mode='scan')

    # =========================================================================
    # Layout 1: Stacked vertically (better for single column)
    # =========================================================================
    # fig_v, (ax_a, ax_b) = plt.subplots(
    #     2, 1, figsize=(3.5, 5.5), constrained_layout=True)

    # plot_panel(ax_a, data_grid, L, W, mode='gridmap',
    #            step_interval=args.step_a, polytope_steps=args.poly_a,
    #            label='(a)')
    # plot_panel(ax_b, data_scan, L, W, mode='scan',
    #            step_interval=args.step_b, polytope_steps=args.poly_b,
    #            label='(b)')

    # # Override axis limits if specified
    # if args.xlim_a: ax_a.set_xlim(args.xlim_a)
    # if args.ylim_a: ax_a.set_ylim(args.ylim_a)
    # if args.xlim_b: ax_b.set_xlim(args.xlim_b)
    # if args.ylim_b: ax_b.set_ylim(args.ylim_b)

    # if args.save:
    #     fig_v.savefig(f'{args.prefix}_stacked.pdf', dpi=300)
    #     fig_v.savefig(f'{args.prefix}_stacked.png', dpi=300)
    #     print(f"Saved: {args.prefix}_stacked.pdf/png")

    # =========================================================================
    # Layout 2: Side by side (alternative, tighter)
    # =========================================================================
    # Compute each panel's natural x/y span so we can set unequal column
    # widths.  With aspect='equal', physical_width ∝ x_span/y_span at a
    # fixed height, so matching those ratios makes both panels the same height.
    _margin = 1.0
    def _span(data, xlim, ylim):
        traj = np.array([(o[1], o[2]) for o in data['odom']])
        xs = (xlim[1] - xlim[0]) if xlim else traj[:, 0].ptp() + 2 * _margin
        ys = (ylim[1] - ylim[0]) if ylim else traj[:, 1].ptp() + 2 * _margin
        return xs, ys

    xs_a, ys_a = _span(data_grid, args.xlim_a, args.ylim_a)
    xs_b, ys_b = _span(data_scan,  args.xlim_b, args.ylim_b)
    width_ratios = [xs_a / ys_a, xs_b / ys_b]

    fig_h, (ax_a2, ax_b2) = plt.subplots(
        1, 2, figsize=(3.5, 2.8), constrained_layout=True,
        gridspec_kw={'width_ratios': width_ratios})

    plot_panel(ax_a2, data_grid, L, W, mode='gridmap',
               step_interval=args.step_a, polytope_steps=args.poly_a,
               label='(a)')
    plot_panel(ax_b2, data_scan, L, W, mode='scan',
               step_interval=args.step_b, polytope_steps=args.poly_b,
               label='(b)')

    # Remove redundant y-label on right panel
    # ax_b2.set_ylabel('')

    if args.xlim_a: ax_a2.set_xlim(args.xlim_a)
    if args.ylim_a: ax_a2.set_ylim(args.ylim_a)
    if args.xlim_b: ax_b2.set_xlim(args.xlim_b)
    if args.ylim_b: ax_b2.set_ylim(args.ylim_b)

    if args.save:
        fig_h.savefig(f'{args.prefix}_sidebyside.pdf', dpi=600)
        fig_h.savefig(f'{args.prefix}_sidebyside.eps', format='eps')
        fig_h.savefig(f'{args.prefix}_sidebyside.png', dpi=600)
        print(f"Saved: {args.prefix}_sidebyside.pdf/eps/png")

    plt.show()


if __name__ == '__main__':
    main()