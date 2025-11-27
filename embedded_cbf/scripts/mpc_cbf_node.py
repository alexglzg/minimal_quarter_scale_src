#!/usr/bin/env python3
"""
MPC-CBF Node for ROS1 Noetic
Solves an OCP with Control Barrier Function constraints for obstacle avoidance.

The polytope constraints are parameterized with a FIXED maximum size (20 halfplanes).
Unused halfplanes are filled with dummy constraints that are always satisfied.
"""

import rospy
import numpy as np
import casadi as ca
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from roboat_core.msg import Force
from decomp_ros_msgs.msg import PolyhedronArray
from tf.transformations import euler_from_quaternion


class MPCCBFNode:
    def __init__(self):
        rospy.init_node('mpc_cbf_node', anonymous=False)

        # =====================================================================
        # Parameters
        # =====================================================================
        self.N = rospy.get_param('~horizon', 21)
        self.dt = rospy.get_param('~dt', 0.1)
        self.N_cbf = rospy.get_param('~N_cbf', 6)
        self.rate = rospy.get_param('~rate', 10.0)
        self.max_force = rospy.get_param('~max_force', 6.0)
        self.gamma = rospy.get_param('~gamma', 0.8)
        self.max_approx = rospy.get_param('~max_approx', 5e-3)

        # Robot dimensions
        self.L_robot = rospy.get_param('~robot_length', 0.1)
        self.W_robot = rospy.get_param('~robot_width', 0.1)

        # FIXED maximum number of halfplanes
        self.max_halfplanes = rospy.get_param('~max_halfplanes', 20)

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

        # Reference state (updated via callback)
        # Will be set by `ref_callback` when a `PoseStamped` arrives
        self.x_ref = np.array([0.0, 0.0, 0.0])
        self.ref_received = False

        # Boat dynamics parameters (QuarterScale)
        self.m11 = rospy.get_param('~system_dynamics/m11', 16.9)
        self.m22 = rospy.get_param('~system_dynamics/m22', 30.5)
        self.m33 = rospy.get_param('~system_dynamics/m33', 5.77)
        self.d11 = rospy.get_param('~system_dynamics/d11', 0.86)
        self.d22 = rospy.get_param('~system_dynamics/d22', 15.73)
        self.d33 = rospy.get_param('~system_dynamics/d33', 1.78)
        self.aa = rospy.get_param('~system_dynamics/aa', 0.18)
        self.bb = rospy.get_param('~system_dynamics/bb', 0.08)

        # State dimensions
        self.nx = 7  # [x, y, psi, u, v, r, s]
        self.nu = 4  # [F1, F2, F3, F4]

        # =====================================================================
        # State variables
        # =====================================================================
        self.current_state = np.zeros(self.nx)
        self.odom_received = False
        self.poly_received = False

        # Initialize polytope with dummy values (Ax <= b convention from decomp)
        # Dummy: 0*x <= small_positive is always satisfied
        # Use a modest value to avoid numerical issues in smooth min
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0  # h = b - Ax = 10 for dummy
        self.n_active = 0

        # =====================================================================
        # Build OCP once
        # =====================================================================
        self._build_ocp()

        # =====================================================================
        # ROS Publishers and Subscribers
        # =====================================================================
        self.force_pub = rospy.Publisher('/mpc_force', Force, queue_size=1)
        self.traj_pub = rospy.Publisher('/mpc_trajectory', Path, queue_size=1)

        self.odom_sub = rospy.Subscriber(
            '/odometry/filtered', Odometry, self.odom_callback, queue_size=1
        )
        self.poly_sub = rospy.Subscriber(
            '/polyhedron_array', PolyhedronArray, self.poly_callback, queue_size=1
        )

        # Reference subscriber: updates target (x_ref, y_ref, yaw_ref)
        self.ref_sub = rospy.Subscriber(
            '/reference_pose', PoseStamped, self.ref_callback, queue_size=1
        )

        rospy.loginfo("MPC-CBF Node initialized")
        rospy.loginfo(f"  Horizon: {self.N}, dt: {self.dt}, CBF steps: {self.N_cbf}")
        rospy.loginfo(f"  Max halfplanes: {self.max_halfplanes}")
        rospy.loginfo(f"  Robot size: {self.L_robot} x {self.W_robot}")

    def _boat_dynamics(self, x, u):
        """QuarterScale boat dynamics."""
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
                (self.bb / (2 * self.m33)) * (F3 - F4),
            0  # s_dot
        )
        return dxdt

    def _robot_vertices_ca(self, x, y, psi):
        """Get robot vertices in world frame."""
        L, W = self.L_robot, self.W_robot
        c = ca.cos(psi)
        s = ca.sin(psi)

        # 4 vertices in body frame
        local_x = ca.vertcat(L/2, L/2, -L/2, -L/2)
        local_y = ca.vertcat(W/2, -W/2, -W/2, W/2)

        # Transform to world
        world_x = c * local_x - s * local_y + x
        world_y = s * local_x + c * local_y + y

        return ca.horzcat(world_x, world_y)

    def _build_ocp(self):
        """
        Build the CasADi OCP with FIXED-SIZE parameterized polytope.
        
        The polytope is Ax <= b (decomp convention).
        For CBF we need: -Ax + b >= 0  (inside = safe)
        So CBF value = min over vertices j, halfplanes i of: -A[i,:] @ v_j + b[i]
        """
        rospy.loginfo("Building OCP with fixed-size polytope parameters...")

        # Smooth min function for CBF (numerically stable version)
        n_total = 4 * self.max_halfplanes
        e_sym = ca.MX.sym('e', n_total)
        alpha_sym = ca.MX.sym('alpha')
        
        def smooth_min(e, alpha):
            # Numerically stable: shift by max to prevent overflow
            # smooth_min(e) = e_max - (1/alpha) * log(sum(exp(alpha * (e_max - e))))
            # But since e_max is symbolic, use standard form with clamping
            # For numerical stability, clamp the exponent
            exp_arg = -alpha * e
            exp_arg_clamped = ca.fmin(exp_arg, 50)  # Prevent overflow
            return -ca.log(ca.sum1(ca.exp(exp_arg_clamped)) + 1e-10) / alpha
        
        lseMin = ca.Function('lseMin', [e_sym, alpha_sym], [smooth_min(e_sym, alpha_sym)])

        self.opti = ca.Opti()

        # Decision variables - same structure as original
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

        # FIXED-SIZE polytope parameters
        # A is (max_halfplanes, 2), b is (max_halfplanes,)
        self.A_param = self.opti.parameter(self.max_halfplanes, 2)
        self.b_param = self.opti.parameter(self.max_halfplanes)

        # Cost function
        cost = 0
        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            cost += ca.mtimes([(xk[0:3] - self.x_ref_param).T, self.Q, (xk[0:3] - self.x_ref_param)])
            cost += ca.mtimes([uk.T, self.R, uk])

        cost += ca.mtimes([(self.X[self.N][0:3] - self.x_ref_param).T, self.Q * 10, 
                          (self.X[self.N][0:3] - self.x_ref_param)])

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
                # CBF at xk
                # Decomp convention: A @ x <= b means inside
                # CBF value h = b - A @ x (positive when inside)
                vec_xk = []
                verts_xk = self._robot_vertices_ca(self.X[k][0], self.X[k][1], self.X[k][2])
                for j in range(4):
                    for i in range(self.max_halfplanes):
                        # verts_xk[j, :] is (1, 2), A_param[i, :] is (1, 2)
                        # We need A @ v = sum(A * v)
                        Ai_0 = self.A_param[i, 0]
                        Ai_1 = self.A_param[i, 1]
                        vj_0 = verts_xk[j, 0]
                        vj_1 = verts_xk[j, 1]
                        bi = self.b_param[i]
                        # h = b - A @ v
                        constraint_val = bi - (Ai_0 * vj_0 + Ai_1 * vj_1)
                        vec_xk.append(constraint_val)

                alpha = ca.log(n_total) / self.max_approx
                cbf_xk = lseMin(ca.vertcat(*vec_xk), alpha)

                # CBF at x_next_model
                vec_xk1 = []
                verts_xk1 = self._robot_vertices_ca(x_next_model[0], x_next_model[1], x_next_model[2])
                for j in range(4):
                    for i in range(self.max_halfplanes):
                        Ai_0 = self.A_param[i, 0]
                        Ai_1 = self.A_param[i, 1]
                        vj_0 = verts_xk1[j, 0]
                        vj_1 = verts_xk1[j, 1]
                        bi = self.b_param[i]
                        constraint_val = bi - (Ai_0 * vj_0 + Ai_1 * vj_1)
                        vec_xk1.append(constraint_val)

                cbf_xk1 = lseMin(ca.vertcat(*vec_xk1), alpha)

                # CBF constraint
                self.opti.subject_to(cbf_xk1 >= self.gamma * cbf_xk + (1 - self.gamma) * self.max_approx)

        self.opti.minimize(cost)

        # Solver options
        opts = {
            "fatrop.print_level": 0,
            "print_time": 0,
            "fatrop.max_iter": 100,
            "fatrop.tol": 1e-4,
            "structure_detection": "auto",
            "expand": True,
            "debug": False
        }
        self.opti.solver('fatrop', opts)

        rospy.loginfo("OCP built successfully")

    def odom_callback(self, msg):
        """Handle odometry messages."""
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = -msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_state[2] = -yaw

        self.current_state[3] = msg.twist.twist.linear.x
        self.current_state[4] = -msg.twist.twist.linear.y
        self.current_state[5] = -msg.twist.twist.angular.z
        self.odom_received = True

    def ref_callback(self, msg):
        """Update reference pose from a PoseStamped message."""
        self.x_ref[0] = msg.pose.position.x
        self.x_ref[1] = msg.pose.position.y
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.x_ref[2] = yaw
        self.ref_received = True

    def poly_callback(self, msg):
        """
        Handle polyhedron messages.
        
        decomp_ros_msgs/Polyhedron has:
        - normals: Point[] - normal vectors n
        - points: Point[] - points p0 on each hyperplane
        
        The hyperplane equation is: n · (x - p0) <= 0
        Which means: n · x <= n · p0
        So: A = n, b = n · p0
        
        We fill our fixed-size arrays, padding unused rows with dummy constraints.
        Dummy: A[i,:] = [0, 0], b[i] = 1000 => always satisfied
        """
        if len(msg.polyhedrons) == 0:
            rospy.logwarn_throttle(1.0, "Empty polyhedron array received")
            return

        poly = msg.polyhedrons[0]
        
        # Reset to dummy values (A=0, b=10 gives h = b - A@x = 10, always positive)
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0

        # Count only 2D constraints (ignore z-axis normals)
        idx = 0
        for i in range(len(poly.normals)):
            nx = poly.normals[i].x
            ny = poly.normals[i].y
            nz = poly.normals[i].z
            
            # Skip z-axis constraints (nz != 0 means it's a z constraint)
            if abs(nz) > 0.1:
                continue
                
            if idx >= self.max_halfplanes:
                rospy.logwarn(f"Too many 2D halfplanes, truncating to {self.max_halfplanes}")
                break
            
            px = poly.points[i].x
            py = poly.points[i].y
            
            # A = n, b = n · p0
            self.A_poly[idx, 0] = nx
            self.A_poly[idx, 1] = ny
            self.b_poly[idx] = nx * px + ny * py
            idx += 1

        self.n_active = idx
        self.poly_received = True
        rospy.loginfo_throttle(1.0, f"Polytope updated: {idx} active 2D halfplanes")

    def solve_mpc(self):
        """Solve the MPC problem."""
        if not self.odom_received:
            rospy.logwarn_throttle(1.0, "No odometry received yet")
            return None, None

        if not self.ref_received:
            rospy.logwarn_throttle(1.0, "No reference received yet")
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

            # Warm start
            for k in range(self.N):
                self.opti.set_initial(self.U[k], u_opt[:, k])
                self.opti.set_initial(self.X[k], x_opt[:, k])
            self.opti.set_initial(self.X[self.N], x_opt[:, self.N])

            return u_opt, x_opt

        except RuntimeError as e:
            rospy.logwarn(f"MPC solver failed: {e}")
            try:
                u_opt = np.zeros((self.nu, self.N))
                x_opt = np.zeros((self.nx, self.N + 1))
                for k in range(self.N):
                    u_opt[:, k] = self.opti.debug.value(self.U[k])
                    x_opt[:, k] = self.opti.debug.value(self.X[k])
                x_opt[:, self.N] = self.opti.debug.value(self.X[self.N])
                return u_opt, x_opt
            except:
                return None, None

    def publish_force(self, u):
        """Publish force command."""
        msg = Force()
        msg.data = [float(u[0]), float(u[1]), float(u[2]), float(u[3])]
        self.force_pub.publish(msg)

    def publish_trajectory(self, x_opt):
        """Publish predicted trajectory."""
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for k in range(self.N + 1):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x_opt[0, k]
            pose.pose.position.y = -x_opt[1, k]
            pose.pose.position.z = 0.0

            yaw = -x_opt[2, k]
            pose.pose.orientation.z = np.sin(yaw / 2)
            pose.pose.orientation.w = np.cos(yaw / 2)

            path_msg.poses.append(pose)

        self.traj_pub.publish(path_msg)

    def run(self):
        """Main loop at 10 Hz."""
        rate = rospy.Rate(self.rate)

        while not rospy.is_shutdown():
            u_opt, x_opt = self.solve_mpc()

            if u_opt is not None and x_opt is not None:
                self.publish_force(u_opt[:, 0])
                self.publish_trajectory(x_opt)

            rate.sleep()


if __name__ == '__main__':
    try:
        node = MPCCBFNode()
        node.run()
    except rospy.ROSInterruptException:
        pass