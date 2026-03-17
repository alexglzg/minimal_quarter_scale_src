#!/usr/bin/env python3
"""
MPC-CBF Node with Path-Following Reference Manager for ROS1 Noetic

Extends the base MPC-CBF with multiple reference sources and priority-based
reference selection. Supports:
  1. Path following (from local planner execute_path or global planner)
  2. RViz 2D Nav Goal (click-to-goal)
  3. Terminal reference pose
  4. Hold position (fallback)

The polytope constraints are parameterized with a FIXED maximum size.
Unused halfplanes are filled with dummy constraints that are always satisfied.

REFERENCE FRAMES:
- ROS topics (odom, polyhedron, paths): ENU (x-forward, y-left, z-up)
- MPC internal: NED (x-north, y-east, z-down)
- Transformation: x_NED = x_ENU, y_NED = -y_ENU, psi_NED = -psi_ENU
"""

import rospy
import numpy as np
import casadi as ca
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from roboat_core.msg import Force
from decomp_ros_msgs.msg import PolyhedronArray
from tf.transformations import euler_from_quaternion
from helpers import fmt


class MPCCBFPathNode:
    def __init__(self, standalone=False, params={}):
        if not standalone:
            rospy.init_node('mpc_cbf_path_node', anonymous=False)

            # =================================================================
            # Parameters
            # =================================================================
            self.N = rospy.get_param('~horizon', 21)
            self.dt = rospy.get_param('~dt', 0.1)
            self.N_cbf = rospy.get_param('~N_cbf', 6)
            self.rate = rospy.get_param('~rate', 10.0)
            self.max_force = rospy.get_param('~max_force', 6.0)
            self.max_surge = rospy.get_param('~max_surge', 0.3)
            self.max_sway = rospy.get_param('~max_sway', 0.1)
            self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 1.0)
            self.gamma = rospy.get_param('~gamma', 0.8)
            self.max_approx = rospy.get_param('~max_approx', 5e-3)

            # Robot dimensions
            self.L_robot = rospy.get_param('~robot_length', 0.1)
            self.W_robot = rospy.get_param('~robot_width', 0.1)

            # FIXED maximum number of halfplanes
            self.max_halfplanes = rospy.get_param('~max_halfplanes', 20)

            # Path following parameters
            self.lookahead_dist = rospy.get_param('~lookahead_dist', 1.0)
            self.path_timeout = rospy.get_param('~path_timeout', 2.0)
            self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.15)

            # Cost weights
            self.Q = np.diag([
                rospy.get_param('~Q_x', 10.0),
                rospy.get_param('~Q_y', 10.0),
                rospy.get_param('~Q_psi', 0.0)
            ])
            self.R = np.diag([
                rospy.get_param('~R_f1', 0.1),
                rospy.get_param('~R_f2', 0.1),
                rospy.get_param('~R_f3', 0.1),
                rospy.get_param('~R_f4', 0.1)
            ])
            self.Q_vel = np.diag([
                rospy.get_param('~Q_surge', 10.0),
                rospy.get_param('~Q_sway', 10.0),
                rospy.get_param('~Q_yaw', 0.0)
            ])

            # Boat dynamics parameters
            self.m11 = rospy.get_param('~system_dynamics/m11', 16.9)
            self.m22 = rospy.get_param('~system_dynamics/m22', 30.5)
            self.m33 = rospy.get_param('~system_dynamics/m33', 5.77)
            self.d11 = rospy.get_param('~system_dynamics/d11', 0.86)
            self.d22 = rospy.get_param('~system_dynamics/d22', 15.73)
            self.d33 = rospy.get_param('~system_dynamics/d33', 1.78)
            self.aa = rospy.get_param('~system_dynamics/aa', 0.18)
            self.bb = rospy.get_param('~system_dynamics/bb', 0.08)

        else:
            # Standalone (simulation/testing) parameter loading
            self.N = params['horizon']
            self.dt = params['dt']
            self.N_cbf = params['N_cbf']
            self.rate = params['rate']
            self.max_force = params['max_force']
            self.max_surge = params.get('max_surge', 0.3)
            self.max_sway = params.get('max_sway', 0.1)
            self.max_yaw_rate = params.get('max_yaw_rate', 1.0)
            self.gamma = params['gamma']
            self.max_approx = params['max_approx']

            self.L_robot = params['robot_length']
            self.W_robot = params['robot_width']
            self.max_halfplanes = params['max_halfplanes']

            self.lookahead_dist = params.get('lookahead_dist', 1.0)
            self.path_timeout = params.get('path_timeout', 2.0)
            self.goal_tolerance = params.get('goal_tolerance', 0.15)

            self.Q = np.diag([params['Q_x'], params['Q_y'], params['Q_psi']])
            self.R = np.diag([params['R_f1'], params['R_f2'], params['R_f3'], params['R_f4']])
            self.Q_vel = np.diag([params['Q_surge'], params['Q_sway'], params['Q_yaw']])

            self.m11 = params['system_dynamics/m11']
            self.m22 = params['system_dynamics/m22']
            self.m33 = params['system_dynamics/m33']
            self.d11 = params['system_dynamics/d11']
            self.d22 = params['system_dynamics/d22']
            self.d33 = params['system_dynamics/d33']
            self.aa = params['system_dynamics/aa']
            self.bb = params['system_dynamics/bb']

        # State dimensions
        self.nx = 6  # [x, y, psi, u, v, r]
        self.nu = 4  # [F1, F2, F3, F4]

        # =================================================================
        # State variables
        # =================================================================
        self.current_state = np.zeros(self.nx)
        self.odom_received = False
        self.poly_received = False

        # Reference state (NED frame, updated by reference manager)
        self.x_ref = np.array([0.0, 0.0, 0.0])
        self.x_ref_vel = np.array([self.max_surge, self.max_sway, self.max_yaw_rate])
        self.ref_received = False

        # Initialize polytope with dummy values
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0
        self.n_active = 0

        # =================================================================
        # Reference sources (all stored in ENU frame)
        # =================================================================
        self._path = None             # nav_msgs/Path from local planner
        self._path_stamp = None       # when the path was last received
        self._path_topic = "none"     # which topic provided the path

        self._rviz_goal = None        # [x, y, yaw] ENU from RViz click
        self._rviz_stamp = None

        self._ref_pose = None         # [x, y, yaw] ENU from terminal
        self._ref_stamp = None

        self._hold_pos = None         # [x, y, yaw] NED, set when no ref

        self._active_source = "none"  # for logging

        self.init_guess_x = None      # warm-start state trajectory (None until first solve)
        self.init_guess_u = None      # warm-start input trajectory

        # =================================================================
        # Build OCP once
        # =================================================================
        self._build_ocp()

        if not standalone:
            # =============================================================
            # ROS Publishers
            # =============================================================
            self.force_pub = rospy.Publisher('/mpc_force', Force, queue_size=1)
            self.traj_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=1)
            self.ref_marker_pub = rospy.Publisher('/mpc_reference', PoseStamped, queue_size=1)
            self.status_pub = rospy.Publisher('/mpc_status', String, queue_size=1)

            # =============================================================
            # ROS Subscribers
            # =============================================================
            # Odometry
            self.odom_sub = rospy.Subscriber(
                '/odometry/filtered', Odometry, self.odom_callback, queue_size=1)

            # Polytope from FIRI
            self.poly_sub = rospy.Subscriber(
                '/polyhedron_array', PolyhedronArray, self.poly_callback, queue_size=1)

            # Path from local planner (execute_path)
            self.exec_path_sub = rospy.Subscriber(
                'planning/planning/execute_path', Path, self.exec_path_callback, queue_size=1)

            # Path from global planner (for future A* integration)
            self.global_path_sub = rospy.Subscriber(
                '/global_plan', Path, self.global_path_callback, queue_size=1)

            # RViz 2D Nav Goal
            self.rviz_sub = rospy.Subscriber(
                '/move_base_simple/goal', PoseStamped, self.rviz_callback, queue_size=1)

            # Terminal reference pose
            self.ref_pose_sub = rospy.Subscriber(
                '/reference_pose', PoseStamped, self.ref_callback, queue_size=1)

            rospy.loginfo("MPC-CBF Path Node initialized")
            rospy.loginfo(f"  Horizon: {self.N}, dt: {self.dt}, CBF steps: {self.N_cbf}")
            rospy.loginfo(f"  Max halfplanes: {self.max_halfplanes}")
            rospy.loginfo(f"  Robot size: {self.L_robot} x {self.W_robot}")
            rospy.loginfo(f"  Lookahead: {self.lookahead_dist} m")
            rospy.loginfo(f"  Path timeout: {self.path_timeout} s")
            rospy.loginfo(f"  Goal tolerance: {self.goal_tolerance} m")
            rospy.loginfo(f"  Reference priority: exec_path > global_plan > rviz > ref_pose > hold")
            rospy.loginfo(f"  Frame: ROS/ENU -> MPC/NED (y and psi inverted)")
        else:
            print("MPC-CBF Path Node initialized (standalone)")

        # Logging
        self.comp_times = []
        self.success = []

    # =====================================================================
    # Dynamics (unchanged)
    # =====================================================================
    def _boat_dynamics(self, x, u):
        """QuarterScale boat dynamics in NED frame."""
        psi = x[2]
        u_vel = x[3]
        v_vel = x[4]
        r = x[5]

        F1, F2, F3, F4 = u[0], u[1], u[2], u[3]

        dxdt = ca.vertcat(
            ca.cos(psi) * u_vel - ca.sin(psi) * v_vel,
            ca.sin(psi) * u_vel + ca.cos(psi) * v_vel,
            r,
            -self.d11 / self.m11 * u_vel + (F1 + F2) / self.m11,
            -self.d22 / self.m22 * v_vel + (F3 + F4) / self.m22,
            -self.d33 / self.m33 * r + (self.aa / (2 * self.m33)) * (F1 - F2) +
                (self.bb / (2 * self.m33)) * (F3 - F4)
        )
        return dxdt

    def _robot_vertices_ca(self, x, y, psi):
        """Get robot vertices in world frame (NED)."""
        L, W = self.L_robot, self.W_robot
        c = ca.cos(psi)
        s = ca.sin(psi)

        local_x = ca.vertcat(L/2, L/2, -L/2, -L/2)
        local_y = ca.vertcat(W/2, -W/2, -W/2, W/2)

        world_x = c * local_x - s * local_y + x
        world_y = s * local_x + c * local_y + y

        return ca.horzcat(world_x, world_y)

    # =====================================================================
    # OCP construction (unchanged)
    # =====================================================================
    def _build_ocp(self):
        """Build the CasADi OCP with FIXED-SIZE parameterized polytope."""
        if not hasattr(self, '_standalone_flag'):
            try:
                rospy.loginfo("Building OCP with fixed-size polytope parameters...")
            except:
                print("Building OCP...")

        n_total = 4 * self.max_halfplanes
        e_sym = ca.MX.sym('e', n_total)
        alpha_sym = ca.MX.sym('alpha')

        def basic(e):
            m = ca.mmax(e)
            return m + ca.log(ca.sum1(ca.exp(e - m)))

        def smooth_max(e, alpha=10):
            return 1/alpha * basic(alpha * e)

        def smooth_min(e, alpha=10):
            return -smooth_max(-e, alpha)

        lseMin = ca.Function('lseMin', [e_sym, alpha_sym], [smooth_min(e_sym, alpha_sym)])

        self.opti = ca.Opti()

        # Decision variables
        self.X = []
        self.U = []
        for k in range(self.N):
            Xk = self.opti.variable(self.nx)
            Uk = self.opti.variable(self.nu)
            self.X.append(Xk)
            self.U.append(Uk)
        Xk = self.opti.variable(self.nx)
        self.X.append(Xk)

        # Parameters
        self.X0_param = self.opti.parameter(self.nx)
        self.x_ref_param = self.opti.parameter(3)

        # FIXED-SIZE polytope parameters (NED frame)
        self.A_param = self.opti.parameter(self.max_halfplanes, 2)
        self.b_param = self.opti.parameter(self.max_halfplanes)

        # Cost function
        cost = 0
        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            pos_err = xk[0:2] - self.x_ref_param[0:2]
            cost += ca.mtimes([pos_err.T, self.Q[0:2, 0:2], pos_err])
            cost += self.Q[2, 2] * ((ca.sin(xk[2]) - ca.sin(self.x_ref_param[2]))**2 +
                                     (ca.cos(xk[2]) - ca.cos(self.x_ref_param[2]))**2)
            cost += ca.mtimes([uk.T, self.R, uk])
            cost += ca.mtimes([(xk[3:]).T, self.Q_vel, (xk[3:])])

        xN = self.X[self.N]
        pos_err_N = xN[0:2] - self.x_ref_param[0:2]
        cost += ca.mtimes([pos_err_N.T, self.Q[0:2, 0:2] * 10, pos_err_N])
        cost += self.Q[2, 2] * 10 * ((ca.sin(xN[2]) - ca.sin(self.x_ref_param[2]))**2 +
                                      (ca.cos(xN[2]) - ca.cos(self.x_ref_param[2]))**2)

        # Dynamics and constraints
        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            x_next = self.X[k + 1]

            dxdt = self._boat_dynamics(xk, uk)
            x_next_model = xk + dxdt * self.dt

            self.opti.subject_to(x_next == x_next_model)

            if k == 0:
                self.opti.subject_to(self.X[0] == self.X0_param)

            # Input constraints
            self.opti.subject_to(self.opti.bounded(-self.max_force, uk[0], self.max_force))
            self.opti.subject_to(self.opti.bounded(-self.max_force, uk[1], self.max_force))
            self.opti.subject_to(self.opti.bounded(-self.max_force, uk[2], self.max_force))
            self.opti.subject_to(self.opti.bounded(-self.max_force, uk[3], self.max_force))

            # CBF constraints
            if k < self.N_cbf:
                # Single CBF per point-halfplane constraint
                verts_xk = self._robot_vertices_ca(self.X[k][0], self.X[k][1], self.X[k][2])
                verts_xk1 = self._robot_vertices_ca(x_next_model[0], x_next_model[1], x_next_model[2])
                for j in range(4):
                    for i in range(self.max_halfplanes):
                        Ai_0 = self.A_param[i, 0]
                        Ai_1 = self.A_param[i, 1]
                        vj_0_k = verts_xk[j, 0]
                        vj_1_k = verts_xk[j, 1]
                        vj_0_k1 = verts_xk1[j, 0]
                        vj_1_k1 = verts_xk1[j, 1]
                        bi = self.b_param[i]
                        dist_xk = bi - (Ai_0 * vj_0_k + Ai_1 * vj_1_k)
                        dist_xk1 = bi - (Ai_0 * vj_0_k1 + Ai_1 * vj_1_k1)
                        self.opti.subject_to(dist_xk1 >= self.gamma*dist_xk)

                # # LSE approximation of CBF

                # vec_xk = []
                # verts_xk = self._robot_vertices_ca(self.X[k][0], self.X[k][1], self.X[k][2])
                # for j in range(4):
                #     for i in range(self.max_halfplanes):
                #         Ai_0 = self.A_param[i, 0]
                #         Ai_1 = self.A_param[i, 1]
                #         vj_0 = verts_xk[j, 0]
                #         vj_1 = verts_xk[j, 1]
                #         bi = self.b_param[i]
                #         constraint_val = bi - (Ai_0 * vj_0 + Ai_1 * vj_1)
                #         vec_xk.append(constraint_val)

                # alpha = ca.log(n_total) / self.max_approx
                # cbf_xk = lseMin(ca.vertcat(*vec_xk), alpha)

                # vec_xk1 = []
                # verts_xk1 = self._robot_vertices_ca(x_next_model[0], x_next_model[1], x_next_model[2])
                # for j in range(4):
                #     for i in range(self.max_halfplanes):
                #         Ai_0 = self.A_param[i, 0]
                #         Ai_1 = self.A_param[i, 1]
                #         vj_0 = verts_xk1[j, 0]
                #         vj_1 = verts_xk1[j, 1]
                #         bi = self.b_param[i]
                #         constraint_val = bi - (Ai_0 * vj_0 + Ai_1 * vj_1)
                #         vec_xk1.append(constraint_val)

                # cbf_xk1 = lseMin(ca.vertcat(*vec_xk1), alpha)

                # self.opti.subject_to(cbf_xk1 >= self.gamma * cbf_xk + (1 - self.gamma) * self.max_approx)

        self.opti.minimize(cost)

        # Solver options
        opts = {
            "fatrop.print_level": 0,
            "print_time": 0,
            "fatrop.max_iter": 100,
            "fatrop.tol": 1e-4,
            "fatrop.mu_init": 1e-1,
            "structure_detection": "auto",
            "expand": True,
            "debug": False
        }
        self.opti.solver('fatrop', opts)

    # =====================================================================
    # Reference source callbacks
    # =====================================================================
    def odom_callback(self, msg):
        """Handle odometry. Transform ENU -> NED."""
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = -msg.pose.pose.position.y  # ENU->NED

        q = msg.pose.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_state[2] = -yaw_enu  # ENU->NED

        self.current_state[3] = msg.twist.twist.linear.x
        self.current_state[4] = -msg.twist.twist.linear.y
        self.current_state[5] = -msg.twist.twist.angular.z

        self.odom_received = True

    def exec_path_callback(self, msg):
        """Path from local planner (planning/planning/execute_path). Stored in ENU."""
        if len(msg.poses) < 1:
            return
        self._path = msg
        self._path_stamp = rospy.Time.now()
        self._path_topic = "execute_path"

    def global_path_callback(self, msg):
        """Path from global planner (/global_plan). Stored in ENU.
        Only used if no execute_path is active."""
        if len(msg.poses) < 1:
            return
        # Don't overwrite a fresh execute_path
        if self._path_stamp is not None and self._path_topic == "execute_path":
            age = (rospy.Time.now() - self._path_stamp).to_sec()
            if age < self.path_timeout:
                return
        self._path = msg
        self._path_stamp = rospy.Time.now()
        self._path_topic = "global_plan"

    def rviz_callback(self, msg):
        """RViz 2D Nav Goal. Store in ENU."""
        x_enu = msg.pose.position.x
        y_enu = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._rviz_goal = np.array([x_enu, y_enu, yaw_enu])
        self._rviz_stamp = rospy.Time.now()
        rospy.loginfo(f"RViz goal set: [{x_enu:.2f}, {y_enu:.2f}, {np.degrees(yaw_enu):.1f}°]")

    def ref_callback(self, msg):
        """Terminal reference pose. Store in ENU."""
        x_enu = msg.pose.position.x
        y_enu = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._ref_pose = np.array([x_enu, y_enu, yaw_enu])
        self._ref_stamp = rospy.Time.now()

    def poly_callback(self, msg):
        """Handle polyhedron messages. Transform ENU -> NED."""
        if len(msg.polyhedrons) == 0:
            rospy.logwarn_throttle(1.0, "Empty polyhedron array received")
            return

        poly = msg.polyhedrons[0]

        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0

        idx = 0
        for i in range(len(poly.normals)):
            nx_enu = poly.normals[i].x
            ny_enu = poly.normals[i].y
            nz = poly.normals[i].z

            if abs(nz) > 0.1:
                continue
            if idx >= self.max_halfplanes:
                rospy.logwarn(f"Too many 2D halfplanes, truncating to {self.max_halfplanes}")
                break

            px_enu = poly.points[i].x
            py_enu = poly.points[i].y

            self.A_poly[idx, 0] = nx_enu
            self.A_poly[idx, 1] = -ny_enu
            self.b_poly[idx] = nx_enu * px_enu + ny_enu * py_enu

            idx += 1

        self.n_active = idx
        self.poly_received = True

    # =====================================================================
    # Path following: lookahead reference extraction
    # =====================================================================
    def _extract_path_reference(self):
        """
        Extract a reference point from the stored path using lookahead.

        1. Find the closest point on the path to the robot (in ENU).
        2. Walk forward along the path by lookahead_dist.
        3. Return the reference pose [x, y, yaw] in ENU, with yaw
           computed from the path tangent at the lookahead point.

        Returns None if the path is exhausted (robot near end).
        """
        if self._path is None or len(self._path.poses) < 2:
            return None

        poses = self._path.poses

        # Robot position in ENU
        robot_x = self.current_state[0]         # x is same in ENU/NED
        robot_y = -self.current_state[1]         # NED->ENU: invert y

        # --- Find closest point on path ---
        min_dist = float('inf')
        closest_idx = 0
        for i, p in enumerate(poses):
            dx = p.pose.position.x - robot_x
            dy = p.pose.position.y - robot_y
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                closest_idx = i

        # --- Walk forward by lookahead distance ---
        remaining = self.lookahead_dist
        la_idx = closest_idx

        for i in range(closest_idx, len(poses) - 1):
            dx = poses[i + 1].pose.position.x - poses[i].pose.position.x
            dy = poses[i + 1].pose.position.y - poses[i].pose.position.y
            seg_len = np.sqrt(dx * dx + dy * dy)

            if seg_len < 1e-6:
                continue

            if remaining <= seg_len:
                # Interpolate within this segment
                frac = remaining / seg_len
                ref_x = poses[i].pose.position.x + frac * dx
                ref_y = poses[i].pose.position.y + frac * dy
                ref_yaw = np.arctan2(dy, dx)
                return np.array([ref_x, ref_y, ref_yaw])

            remaining -= seg_len
            la_idx = i + 1

        # Reached end of path: use last pose
        last = poses[-1]
        ref_x = last.pose.position.x
        ref_y = last.pose.position.y

        # Yaw from last segment or from pose orientation
        if len(poses) >= 2:
            prev = poses[-2]
            dx = last.pose.position.x - prev.pose.position.x
            dy = last.pose.position.y - prev.pose.position.y
            if dx * dx + dy * dy > 1e-6:
                ref_yaw = np.arctan2(dy, dx)
            else:
                q = last.pose.orientation
                _, _, ref_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        else:
            q = last.pose.orientation
            _, _, ref_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Check if we're close enough to consider the path finished
        dx = ref_x - robot_x
        dy = ref_y - robot_y
        if np.sqrt(dx * dx + dy * dy) < self.goal_tolerance:
            return None  # path exhausted

        return np.array([ref_x, ref_y, ref_yaw])

    # =====================================================================
    # Reference manager: priority-based selection
    # =====================================================================
    def _update_reference(self):
        """
        Select the active reference source by priority and update self.x_ref.

        Priority (highest first):
          1. execute_path (local planner)  — if received within path_timeout
          2. global_plan                   — if received within path_timeout
          3. RViz goal                     — persistent until overwritten
          4. Terminal reference_pose        — persistent until overwritten
          5. Hold current position          — fallback

        All path/goal sources are stored in ENU.
        self.x_ref is set in NED (for the solver).
        """
        now = rospy.Time.now()
        ref_enu = None

        # --- Priority 1 & 2: Path sources ---
        if self._path is not None and self._path_stamp is not None:
            age = (now - self._path_stamp).to_sec()
            if age < self.path_timeout:
                ref_enu = self._extract_path_reference()
                if ref_enu is not None:
                    self._active_source = self._path_topic
                else:
                    # Path is exhausted (robot at end) — clear it
                    rospy.loginfo_throttle(2.0, "Path goal reached, clearing path")
                    self._path = None
                    self._path_stamp = None

        # --- Priority 3: RViz goal ---
        if ref_enu is None and self._rviz_goal is not None:
            # Check if we've reached the RViz goal
            robot_x = self.current_state[0]
            robot_y = -self.current_state[1]  # NED->ENU
            dx = self._rviz_goal[0] - robot_x
            dy = self._rviz_goal[1] - robot_y
            if np.sqrt(dx * dx + dy * dy) > self.goal_tolerance:
                ref_enu = self._rviz_goal
                self._active_source = "rviz"
            else:
                rospy.loginfo_throttle(2.0, "RViz goal reached")
                # Keep the goal as a station-keeping reference
                ref_enu = self._rviz_goal
                self._active_source = "rviz_hold"

        # --- Priority 4: Terminal reference ---
        if ref_enu is None and self._ref_pose is not None:
            ref_enu = self._ref_pose
            self._active_source = "ref_pose"

        # --- Priority 5: Hold position ---
        if ref_enu is None:
            if self._hold_pos is None and self.odom_received:
                # First time: capture current position as hold target (NED)
                self._hold_pos = self.current_state[0:3].copy()
                rospy.loginfo("No reference available, holding position")

            if self._hold_pos is not None:
                self.x_ref = self._hold_pos
                self._active_source = "hold"
                self.ref_received = True
                return

            # No odom yet, can't do anything
            self._active_source = "none"
            return

        # --- Transform ENU -> NED and set ---
        self.x_ref[0] = ref_enu[0]           # x same
        self.x_ref[1] = -ref_enu[1]          # y inverts
        self.x_ref[2] = -ref_enu[2]          # yaw inverts
        self.ref_received = True

        # Clear hold position when we have a real reference
        self._hold_pos = None

    # =====================================================================
    # Solver (unchanged from base)
    # =====================================================================
    def _reset_warm_start(self):
        """Reset initial guess to a stationary trajectory at the current state."""
        x0 = self.current_state.copy()
        for k in range(self.N):
            self.opti.set_initial(self.U[k], np.zeros(self.nu))
            self.opti.set_initial(self.X[k], x0)
        self.opti.set_initial(self.X[self.N], x0)
        self.init_guess_x = None
        self.init_guess_u = None

    def solve_mpc(self):
        """Solve the MPC problem with robust warm start handling."""
        if not self.odom_received:
            rospy.logwarn_throttle(1.0, "No odometry received yet")
            return None, None

        # Update reference from active source
        self._update_reference()

        if not self.ref_received:
            rospy.logwarn_throttle(1.0, "No reference available")
            return None, None

        # Set parameters
        self.opti.set_value(self.X0_param, self.current_state)
        self.opti.set_value(self.x_ref_param, self.x_ref)
        self.opti.set_value(self.A_param, self.A_poly)
        self.opti.set_value(self.b_param, self.b_poly)

        try:
            sol = self.opti.solve()

            u_opt = np.zeros((self.nu, self.N))
            x_opt = np.zeros((self.nx, self.N + 1))

            for k in range(self.N):
                u_opt[:, k] = sol.value(self.U[k])
                x_opt[:, k] = sol.value(self.X[k])
            x_opt[:, self.N] = sol.value(self.X[self.N])
            self.init_guess_x = x_opt
            self.init_guess_u = u_opt

            # Warm start for next solve
            for k in range(self.N):
                self.opti.set_initial(self.U[k], u_opt[:, k])
                self.opti.set_initial(self.X[k], x_opt[:, k])
            self.opti.set_initial(self.X[self.N], x_opt[:, self.N])

            return u_opt, x_opt

        except RuntimeError as e:
            rospy.logwarn(f"MPC solver failed: {e}")
            rospy.logwarn(
                'MPC - No solution found.\n'
                f"X0_param = [{fmt(self.current_state)}]\n"
                f"x_ref_param = [{fmt(self.x_ref)}]\n"
                f"A_param = [{fmt(self.A_poly)}]\n"
                f"b_param = [{fmt(self.b_poly)}]\n"
                f"x_init = [{fmt(self.init_guess_x) if self.init_guess_x is not None else 'None'}]\n"
                f"u_init = [{fmt(self.init_guess_u) if self.init_guess_u is not None else 'None'}]"
            )

            try:
                u_opt = np.zeros((self.nu, self.N))
                x_opt = np.zeros((self.nx, self.N + 1))
                for k in range(self.N):
                    u_opt[:, k] = self.opti.debug.value(self.U[k])
                    x_opt[:, k] = self.opti.debug.value(self.X[k])
                x_opt[:, self.N] = self.opti.debug.value(self.X[self.N])
                self._reset_warm_start()
                return u_opt, x_opt
            except:
                self._reset_warm_start()
                return None, None

    # =====================================================================
    # Publishers
    # =====================================================================
    def publish_force(self, u):
        """Publish force command (body frame, no transform needed)."""
        msg = Force()
        msg.data = [float(u[0]), float(u[1]), float(u[2]), float(u[3])]
        self.force_pub.publish(msg)

    def publish_trajectory(self, x_opt):
        """Publish predicted trajectory. NED -> ENU."""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for k in range(self.N + 1):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x_opt[0, k]
            pose.pose.position.y = -x_opt[1, k]  # NED->ENU
            pose.pose.position.z = 0.0

            yaw_enu = -x_opt[2, k]
            pose.pose.orientation.z = np.sin(yaw_enu / 2)
            pose.pose.orientation.w = np.cos(yaw_enu / 2)

            path_msg.poses.append(pose)

        self.traj_pub.publish(path_msg)

    def publish_reference_marker(self):
        """Publish current reference as PoseStamped for RViz visualization."""
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        # NED -> ENU
        msg.pose.position.x = self.x_ref[0]
        msg.pose.position.y = -self.x_ref[1]
        msg.pose.position.z = 0.0
        yaw_enu = -self.x_ref[2]
        msg.pose.orientation.z = np.sin(yaw_enu / 2)
        msg.pose.orientation.w = np.cos(yaw_enu / 2)
        self.ref_marker_pub.publish(msg)

    def publish_status(self):
        """Publish active reference source for debugging."""
        self.status_pub.publish(String(data=self._active_source))

    # =====================================================================
    # Main loop
    # =====================================================================
    def run(self):
        """Main loop."""
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            u_opt, x_opt = self.solve_mpc()

            if u_opt is not None and x_opt is not None:
                self.publish_force(u_opt[:, 0])
                self.publish_trajectory(x_opt)
            else:
                self.publish_force(np.zeros(self.nu))  # safety: stop motors

            if self.ref_received:
                self.publish_reference_marker()

            self.publish_status()

            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCCBFPathNode()
        node.run()
    except rospy.ROSInterruptException:
        pass