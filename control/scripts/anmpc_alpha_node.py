#!/usr/bin/env python3
"""ROS node wrapping the Amortized-NMPC-alpha (anmpc_alpha) vessel controller.

Standalone alternative to mpc_node.py: same topics/messages, but the QP-safe
command comes from the trained network in anmpc_alpha/model.eqx instead of
the CasADi solve. Structural constants (n_obs, dt) come from the loaded
model, not from parameters.yaml -- see anmpc_alpha/INTEGRATION.md.

Integration follows the step-by-step conventions from that package's
integration walkthrough: load once and warm up the JIT at startup, build
x/path/obstacle arrays from ROS messages, validate before each call, and own
the arc-length `s` locally since the model itself is stateless.
"""
import time
from pathlib import Path as FsPath

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import rospy

from anmpc_alpha import load_alpha_model, make_env, wrap_angle, anchor_arc_length

from roboat_core.msg import Force
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Bool, Float64MultiArray
from tf.transformations import euler_from_quaternion
from obstacle_detector.msg import BuoyArray
from path import SinePath

MODEL_DIR = FsPath(__file__).resolve().parent / "anmpc_alpha"


@eqx.filter_jit # this finds which arguments are static and which are arrays, and compiles the model once
def _controller_step(model, x, path, obs_centers, obs_radii):
    """One control step -> (u_safe (4,), u_nom (4,), gammas (n_obs, 2)).

    Module-level and jitted so it compiles exactly once, not once per node instance.
    """
    return model(x, path, obs_centers, obs_radii)


def prepare_obstacles(detections, n_obs, x, pad="duplicate"):
    """Variable-length (cx, cy, radius) detections -> fixed (n_obs, 2) / (n_obs,).

    Sorts by along-track distance (the model is not
    permutation-invariant -- training always listed obstacles in along-path
    order), keeps the nearest n_obs, and pads unused slots by duplicating the
    farthest kept detection. Padding with a far sentinel instead measurably
    degrades tracking (see anmpc_alpha/INTEGRATION.md section 2.3), so this
    intentionally does not support that mode.
    """
    dets = [tuple(float(v) for v in d) for d in detections]
    if not dets:
        raise ValueError("need at least one detection to pad from")
    # px, py, psi = float(x[0]), float(x[1]), float(x[2])
    # fwd = np.array([np.cos(psi), np.sin(psi)])
    # dets.sort(key=lambda d: (np.array([d[0] - px, d[1] - py]) @ fwd))
    dets.sort(key=lambda d: d[0])  # sort by x (along-track) only
    # keep the nearest n_obs
    dets = dets[:n_obs]
    # pad by duplicating the farthest kept detection
    while len(dets) < n_obs:
        dets.append(dets[-1])
    centers = np.array([[d[0], d[1]] for d in dets], dtype=float)
    radii = np.array([d[2] for d in dets], dtype=float)
    return centers, radii


def validate_inputs(x, path, centers, radii, n_obs):
    """Cheap assertions catching the mistakes that otherwise fail silently."""
    x, path = np.asarray(x, float), np.asarray(path, float)
    centers, radii = np.asarray(centers, float), np.asarray(radii, float)
    assert x.shape == (7,), f"x must be (7,), got {x.shape}"
    assert path.shape == (3,), f"path must be (3,), got {path.shape}"
    assert centers.shape == (n_obs, 2), f"centers must be ({n_obs},2), got {centers.shape}"
    assert radii.shape == (n_obs,), f"radii must be ({n_obs},), got {radii.shape}"
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(centers)), "non-finite input"
    assert -np.pi <= x[2] < np.pi, f"psi not wrapped to [-pi, pi): {x[2]}"
    assert np.all(radii > 0), "obstacle radii must be positive"


class AnmpcNode:
    def __init__(self):
        model_p, mpc_p, path_p, anmpc_p = self.load_params()

        # Step 1: load the trained model and build the environment (n_obs, dt, f_max, r_ego)
        self.model, self.dtype = load_alpha_model(
            MODEL_DIR / anmpc_p["model_file"],
            meta_path=MODEL_DIR / anmpc_p["meta_file"],
        )
        self.env = make_env(n_obs=self.model.n_obs, dt=self.model.dt)
        self.n_obs = self.model.n_obs


        # Step 2: warm up the JIT-compiled controller once at startup, before any control loop
        self._warm_up()
        rospy.loginfo(
            "anmpc_node - warm up complete: n_obs=%d dt=%.3fs f_max=%.2fN r_ego=%.3fm",
            self.n_obs, self.model.dt, self.env.f_max, self.env.r_ego,
        )

        if path_p["type"] != "sine":
            raise ValueError(
                "anmpc_alpha was trained on a sinusoidal path only "
                "(x_d = kx*s, y_d = sin(kx*s) + y0); "
                f"path.type={path_p['type']!r} is unsupported by this node"
            )
        self.path = SinePath(path_p["x_multiplier"], path_p["y_offset"])
        # the network takes path as a flat (kx, y0, u_ref) input, not the Path object itself
        self.path_vec = np.array(
            [self.path.k, self.path.y_offset, model_p["u_ref"]], dtype=float
        )
        self._warn_if_out_of_trained_range(anmpc_p.get("trained_ranges") or {})

        # No detections yet -- control_loop waits for a real /buoy_array reading
        # before running the controller (see anmpc_alpha/INTEGRATION.md section
        # 2.3: obstacle positions are network inputs, so a far-away sentinel
        # would be out-of-distribution rather than harmless).
        self.detections = None

        self.current_state = None  # [x, y, psi, su, sv, sr] from odometry, NED
        self.s = 0.0
        self._s_seeded = False

        self.cmd_pub = rospy.Publisher("/mpc_force", Force, queue_size=1)
        self.solve_time_pub = rospy.Publisher(
            "/mpc_status_anmpc/solve_time_ms", Float64, queue_size=1)
        # Narrower measurement -- the network+QP call only, same placement as the
        # comparison notebook's rollout() timing -- see control_loop below.
        self.model_solve_time_pub = rospy.Publisher(
            "/mpc_status_anmpc/model_solve_time_ms", Float64, queue_size=1)
        self.success_pub = rospy.Publisher(
            "/mpc_status_anmpc/success", Bool, queue_size=1)
        self.gammas_pub = rospy.Publisher(
            "/mpc_status_anmpc/gammas", Float64MultiArray, queue_size=1)

        rospy.Subscriber("odometry/filtered", Odometry, self.odom_cb)
        rospy.Subscriber("/buoy_array", BuoyArray, self.buoy_array_cb)

        self.path_pub = rospy.Publisher("/desired_path", Path, queue_size=1, latch=True)
        self.publish_path()
        rospy.Timer(rospy.Duration(1.0), lambda _: self.publish_path())

        # Control loop at the rate the model was trained at -- do not change.
        self.control_timer = rospy.Timer(
            rospy.Duration(self.model.dt), self.control_loop)

    def load_params(self):
        model = rospy.get_param("parameters_model")
        mpc = rospy.get_param("parameters_mpc")
        path = rospy.get_param("path")
        alpha = rospy.get_param("parameters_amortized_mpc")
        return model, mpc, path, alpha

    def _warm_up(self):
        x = jnp.zeros(7, self.dtype).at[3].set(0.3)
        path = jnp.zeros(3, self.dtype)
        oc = jnp.full((self.n_obs, 2), 50.0, self.dtype)
        orr = jnp.full((self.n_obs,), 0.5, self.dtype)
        t0 = time.perf_counter()
        jax.block_until_ready(_controller_step(self.model, x, path, oc, orr))
        rospy.loginfo(
            "anmpc_node - warm up complete: JIT warm-up took %.1f ms",
            (time.perf_counter() - t0) * 1e3,
        )

    def _warn_if_out_of_trained_range(self, ranges):
        for name, val in zip(("kx", "y0", "u_ref"), self.path_vec):
            lo_hi = ranges.get(name)
            if lo_hi and not (lo_hi[0] <= val <= lo_hi[1]):
                rospy.logwarn(
                    "anmpc_node: %s=%.3f is outside the trained range "
                    "[%.2f, %.2f] -- the network will be extrapolating",
                    name, val, lo_hi[0], lo_hi[1],
                )

    def buoy_array_cb(self, msg):
        dets = []
        for b in msg.buoys:
            x = b.odom.pose.pose.position.x
            y = -b.odom.pose.pose.position.y
            dets.append((x, y, float(b.radius)))
        if dets:
            self.detections = dets

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = -msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        psi = -wrap_angle(yaw)

        su = msg.twist.twist.linear.x
        sv = -msg.twist.twist.linear.y
        sr = -msg.twist.twist.angular.z

        self.current_state = np.array([x, y, psi, su, sv, sr], dtype=float)

    def control_loop(self, event):
        if self.current_state is None or self.detections is None:
            return
        
        # step 3: build the state vector
        x = np.zeros(7, dtype=float)
        x[:6] = self.current_state

        # step 7: arc length
        if not self._s_seeded:
            self.s = anchor_arc_length(self.path_vec, x, 0.0)
            self._s_seeded = True
        x[6] = self.s
        
        # step 5: prepare obstacles (NOTE: ORDER OF THE BUOYS MUST STAY CONSTANT)
        centers, radii = prepare_obstacles(self.detections, self.n_obs, x)

        t0 = time.perf_counter()
        try:
            x[2] = wrap_angle(x[2])
            x[6] = self.s = anchor_arc_length(self.path_vec, x, self.s)
            validate_inputs(x, self.path_vec, centers, radii, self.n_obs)

            # Narrower timing, placed the same way barriernet_ampc_compare_models.ipynb's
            # rollout() places it (right before the model call, after anchor_arc_length):
            # isolates the network+QP cost from this node's own per-tick overhead
            # (anchor_arc_length, numpy<->JAX conversions), for a like-for-like number
            # against the notebook's solve-time column.
            t_model = time.perf_counter()
            u_safe, _u_nom, gammas = _controller_step(
                self.model,
                jnp.asarray(x, self.dtype), jnp.asarray(self.path_vec, self.dtype),
                jnp.asarray(centers, self.dtype), jnp.asarray(radii, self.dtype),
            )
            u = np.asarray(u_safe, dtype=float)
            model_solve_time = time.perf_counter() - t_model
            gammas = np.asarray(gammas, dtype=float)
        except Exception as e:
            solve_time = time.perf_counter() - t0
            rospy.logwarn(
                "anmpc_node: control step failed after %.1f ms: %s",
                solve_time * 1e3, e,
            )
            self.publish_status(solve_time, success=False)
            return
        solve_time = time.perf_counter() - t0

        if not np.all(np.isfinite(u)):  # qpax has no feasibility fallback
            rospy.logwarn("anmpc_node: non-finite control output, skipping publish")
            self.publish_status(solve_time, success=False)
            return

        self.publish_cmd(u)
        self.publish_status(solve_time, success=True)
        self.model_solve_time_pub.publish(Float64(model_solve_time * 1000.0))
        self.gammas_pub.publish(Float64MultiArray(data=gammas.flatten().tolist()))

    def publish_cmd(self, u):
        cmd_msg = Force()
        cmd_msg.data = u
        self.cmd_pub.publish(cmd_msg)

    def publish_status(self, solve_time, success):
        self.solve_time_pub.publish(Float64(solve_time * 1000.0))
        self.success_pub.publish(Bool(success))

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        for s in np.linspace(0, 500, 1000):
            pose = PoseStamped()
            pose.pose.position.x = self.path.x(s)
            pose.pose.position.y = -self.path.y(s)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)


if __name__ == "__main__":
    rospy.init_node("anmpc_node")
    AnmpcNode()
    rospy.spin()
