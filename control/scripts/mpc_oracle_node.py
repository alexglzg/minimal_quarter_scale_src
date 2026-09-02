#!/usr/bin/env python3
"""ROS node wrapping the NMPC oracle (mpc_oracle.VesselNMPC).

Standalone alternative to mpc_node.py, so its computation time and solver
success can be checked in the ROS environment the same way as
mpc_opti_chocbf_dcbf_core's: same /mpc_force command topic and the same
/buoy_array obstacle feed, but the solve itself goes through
VesselNMPC.solve_func -- the whole NLP compiled to one CasADi Function via
opti.to_function (see mpc_oracle.py), instead of mpc_node.py's
MPCController.ocp_func. Publishes status to its own /mpc_status_oracle
namespace rather than /mpc_status, so both controllers' numbers can be
recorded side by side; run it instead of (not alongside) mpc_node.py, since
both would otherwise command /mpc_force at once.

Unlike mpc_node.py, this node does NOT subscribe to /cbf_alphas: that topic
(published by alpha_node.py) is specific to mpc_opti_chocbf_dcbf_core.py's
externally-driven alpha1/alpha2. The oracle instead uses whatever HOCBF
class-K gains mpc_oracle.VesselNMPC itself defaults to (its constructor's
`default_gamma`, currently (0.5, 0.5)) -- i.e. the oracle is run exactly as
mpc_oracle.py already runs it standalone, not re-tuned to match the other
controller's gains.
"""
import time
import rospy
import numpy as np

from mpc_oracle import VesselNMPC, VesselParams, PathParams, CostWeights, Obstacle, project_arc_length
from roboat_core.msg import Force
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Bool
from tf.transformations import euler_from_quaternion
from obstacle_detector.msg import BuoyArray


class MPCOracleNode:
    def __init__(self):
        model_p, mpc_p, path_p = self.load_params()

        if path_p["type"] != "sine":
            raise ValueError(
                "mpc_oracle was built for the sinusoidal path only "
                f"(x_d = kx*s, y_d = sin(kx*s) + y0); "
                f"path.type={path_p['type']!r} is unsupported by this node"
            )
        self.path = PathParams(kx=path_p["x_multiplier"], y0=path_p["y_offset"],
                                u_ref=model_p["u_ref"])

        self.params = VesselParams(
            m11=model_p["m11"], m22=model_p["m22"], m33=model_p["m33"],
            d11=model_p["d11"], d22=model_p["d22"], d33=model_p["d33"],
            a=model_p["aa"], b=model_p["bb"],
            f_max=model_p["max_force_limit"],
            length=model_p.get("ego_length", 0.9), width=model_p.get("ego_width", 0.45),
        )
        weights = CostWeights(q_ye=mpc_p["Qye"], q_r=mpc_p["Qr"],
                               q_psi=mpc_p["Qpsi"], q_u=mpc_p["Qu"])

        # Obstacle bookkeeping mirrors mpc_node.py: a fixed-size slot list, filled
        # with dummy far-away placeholders until /buoy_array reports detections.
        self.num_obstacles = mpc_p["num_obstacles"]
        self.dummy_x = mpc_p["dummy_x"]
        self.dummy_y = mpc_p["dummy_y"]
        self.dummy_radius = mpc_p["dummy_radius"]
        obstacles = [Obstacle(self.dummy_x, self.dummy_y, self.dummy_radius)
                     for _ in range(self.num_obstacles)]

        rospy.loginfo("mpc_oracle_node: compiling the oracle NLP to a CasADi Function...")
        t0 = time.perf_counter()
        self.mpc = VesselNMPC(obstacles, params=self.params, path=self.path, weights=weights,
                               N=mpc_p["Nhor"], dt=mpc_p["dt"])
        rospy.loginfo("mpc_oracle_node: compiled in %.2f s", time.perf_counter() - t0)

        self.obstacle_x = np.full(self.num_obstacles, self.dummy_x, dtype=float)
        self.obstacle_y = np.full(self.num_obstacles, self.dummy_y, dtype=float)
        self.obstacle_radius = np.full(self.num_obstacles, self.dummy_radius, dtype=float)

        self.current_state = None
        self.s = 0.0
        self._s_seeded = False

        self.cmd_pub = rospy.Publisher("/mpc_force", Force, queue_size=1)
        self.solve_time_pub = rospy.Publisher(
            "/mpc_status_oracle/solve_time_ms", Float64, queue_size=1)
        self.success_pub = rospy.Publisher(
            "/mpc_status_oracle/success", Bool, queue_size=1)

        rospy.Subscriber("odometry/filtered", Odometry, self.odom_cb)
        rospy.Subscriber("/buoy_array", BuoyArray, self.buoy_array_cb)

        self.path_pub = rospy.Publisher("/desired_path", Path, queue_size=1, latch=True)
        self.publish_path()
        rospy.Timer(rospy.Duration(1.0), lambda _: self.publish_path())

        self.control_timer = rospy.Timer(rospy.Duration(self.mpc.dt), self.control_loop)

    def load_params(self):
        model = rospy.get_param("parameters_model")
        mpc = rospy.get_param("parameters_mpc")
        path = rospy.get_param("path")
        return model, mpc, path

    def buoy_array_cb(self, msg):
        detections = []
        for b in msg.buoys:
            x = b.odom.pose.pose.position.x
            y = -b.odom.pose.pose.position.y  # ENU->NED
            detections.append((x, y, float(b.radius)))

        if detections and self.current_state is not None:
            distances = [np.hypot(x - self.current_state[0], y - self.current_state[1])
                         for x, y, r in detections]
            closest = set(np.argsort(distances)[:self.num_obstacles])
            selected = [d for i, d in enumerate(detections) if i in closest]
        else:
            selected = []

        while len(selected) < self.num_obstacles:
            selected.append((self.dummy_x, self.dummy_y, self.dummy_radius))

        for i, (x, y, r) in enumerate(selected):
            self.obstacle_x[i] = x
            self.obstacle_y[i] = y
            self.obstacle_radius[i] = r

    def odom_cb(self, msg):
        """Transform from ROS/ENU to MPC/NED frame and anchor the path parameter s."""
        x_ned = msg.pose.pose.position.x
        y_ned = -msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_ned = -yaw_enu

        su = msg.twist.twist.linear.x
        sv = -msg.twist.twist.linear.y
        sr = -msg.twist.twist.angular.z

        s_center = self.s if self._s_seeded else 0.0
        self.s = project_arc_length(self.path, x_ned, y_ned, s_center)
        self._s_seeded = True

        self.current_state = np.array([x_ned, y_ned, yaw_ned, su, sv, sr, self.s])

    def control_loop(self, event):
        if self.current_state is None:
            return

        centers = np.stack([self.obstacle_x, self.obstacle_y], axis=1)

        # No gammas= -- VesselNMPC.__call__ falls back to its own default_gamma
        # (see mpc_oracle.py), the same gains mpc_oracle.py uses standalone.
        sol = self.mpc(self.current_state, obstacle_centers=centers,
                        obstacle_radii=self.obstacle_radius)

        if sol.status != "optimal":
            rospy.logwarn("mpc_oracle_node: solve failed after %.1f ms", sol.solve_time_s * 1000)
            self.publish_status(sol.solve_time_s, success=False)
            return

        self.publish_cmd(sol.u_opt)
        self.publish_status(sol.solve_time_s, success=True)

    def publish_cmd(self, u):
        self.cmd_pub.publish(Force(data=u))

    def publish_status(self, solve_time, success):
        self.solve_time_pub.publish(Float64(solve_time * 1000.0))
        self.success_pub.publish(Bool(success))

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        for s in np.linspace(0, 500, 1000):
            x, y = self.path.xy(s)
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = -y
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)


if __name__ == "__main__":
    rospy.init_node("mpc_oracle_node")
    MPCOracleNode()
    rospy.spin()
