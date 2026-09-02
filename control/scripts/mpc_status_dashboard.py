#!/usr/bin/env python3
"""Single-window dashboard for MPC computation time, solver status, and CBF alphas.

Hosts rqt_plot PlotWidgets (independent y-axes) as docks in one QMainWindow,
instead of separate top-level rqt_plot windows.
"""
import sys

import matplotlib
import rospy
from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtWidgets import QApplication, QDockWidget, QMainWindow

from rqt_plot.data_plot import DataPlot
from rqt_plot.plot_widget import PlotWidget

# Bigger axis labels/ticks and thicker lines than matplotlib's defaults, so the
# success 0/1 trace is readable at a glance instead of a thin line hugging the edges.
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2.5,
})

CONTROL_PERIOD_MS = 100.0  # 10 Hz control loop budget
RETRY_PERIOD_MS = 1000  # mpc_node's OCP compile can take a while to come up; keep retrying until it does


def alpha_topics_and_ylim():
    """Per-obstacle /cbf_alphas/alpha1[i] and alpha2[i] field topics, plus a fixed
    y-range padded around the configured sampling bounds so the dashboard doesn't
    need parameters_mpc/parameters_alpha to be loaded before it can draw axes."""
    num_obs = rospy.get_param("/parameters_mpc/num_obstacles", 2)
    alpha_p = rospy.get_param("/parameters_alpha", {})
    lo = min(alpha_p.get("alpha1_min", 0.0), alpha_p.get("alpha2_min", 0.0))
    hi = max(alpha_p.get("alpha1_max", 1.0), alpha_p.get("alpha2_max", 1.0))
    margin = 0.1 * (hi - lo) if hi > lo else 0.1

    topics = []
    for i in range(num_obs):
        topics.append("/cbf_alphas/alpha1[%d]" % i)
        topics.append("/cbf_alphas/alpha2[%d]" % i)
    return topics, (lo - margin, hi + margin)


def add_threshold_line(data_plot, value, label):
    # DataPlot has no public "horizontal reference line" API, so reach into the
    # matplotlib backend directly. Its redraw() never clears the axes, so a line
    # added here persists across all future redraws, same as a regular curve.
    backend = data_plot._data_plot_widget
    if not hasattr(backend, "_canvas"):
        rospy.logwarn("mpc_status_dashboard: unsupported plot backend, skipping threshold line")
        return
    backend._canvas.axes.axhline(y=value, color="red", linestyle="--", linewidth=2, label=label)

    # Pin the y-range so the threshold line is never autoscaled out of view.
    data_plot.set_autoscale(y=0)
    data_plot.set_ylim([0, 2 * value])


def make_plot_dock(title, topics, threshold=None, ylim=None):
    widget = PlotWidget(initial_topics=topics)
    data_plot = DataPlot(widget)
    data_plot.set_autoscale(x=False)
    data_plot.set_autoscale(y=DataPlot.SCALE_EXTEND | DataPlot.SCALE_VISIBLE)
    data_plot.set_xlim([0, 30.0])
    if threshold is not None:
        add_threshold_line(data_plot, threshold, "%g ms budget" % threshold)
    if ylim is not None:
        data_plot.set_autoscale(y=0)
        data_plot.set_ylim(list(ylim))
    widget.switch_data_plot_widget(data_plot)

    def retry_subscribe():
        missing = [t for t in topics if t not in widget._rosdata]
        if not missing:
            timer.stop()
            return
        for t in missing:
            widget.add_topic(t)

    timer = QTimer(widget)
    timer.timeout.connect(retry_subscribe)
    timer.start(RETRY_PERIOD_MS)

    dock = QDockWidget(title)
    dock.setWidget(widget)
    return dock


def main():
    rospy.init_node("mpc_status_dashboard", anonymous=True, disable_signals=True)
    app = QApplication(sys.argv)

    # Which controller's status topics to plot: /mpc_status (mpc_node.py,
    # default) or /mpc_status_oracle (mpc_oracle_node.py) / /mpc_status_anmpc
    # (anmpc_alpha_node.py) -- set via a <param> in the launch file.
    status_ns = rospy.get_param("~status_ns", "mpc_status")

    alpha_topics, alpha_ylim = alpha_topics_and_ylim()

    # (title, topics, threshold, fixed_ylim). fixed_ylim pads the range so the
    # plotted values don't sit flush against the axes edges.
    topic_specs = [
        ("Solve time (ms)", [f"/{status_ns}/solve_time_ms/data"], CONTROL_PERIOD_MS, None),
        ("Solver success (0/1)", [f"/{status_ns}/success/data"], None, (-0.5, 1.5)),
        ("CBF alphas (per obstacle)", alpha_topics, None, alpha_ylim),
    ]

    window = QMainWindow()
    window.setWindowTitle("MPC Status")

    areas = [Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea, Qt.RightDockWidgetArea]
    for (title, topics, threshold, ylim), area in zip(topic_specs, areas):
        window.addDockWidget(area, make_plot_dock(title, topics, threshold, ylim))

    window.resize(1200, 900)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
