#!/usr/bin/env python3
"""
MPC-CBF Node for ROS1 Noetic
Solves an OCP with Control Barrier Function constraints for obstacle avoidance.

The polytope constraints are parameterized with a FIXED maximum size (20 halfplanes).
Unused halfplanes are filled with dummy constraints that are always satisfied.

REFERENCE FRAMES:
- ROS topics (odom, polyhedron): ENU (x-forward, y-left, z-up)
- MPC internal: NED (x-north, y-east, z-down)
- Transformation: x_NED = x_ENU, y_NED = -y_ENU, psi_NED = -psi_ENU
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
        self.max_surge = rospy.get_param('~max_surge', 0.3)
        self.max_sway = rospy.get_param('~max_sway', 0.1)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 1.0)
        self.gamma = rospy.get_param('~gamma', 0.8)
        self.max_approx = rospy.get_param('~max_approx', 5e-3)

        # Robot dimensions
        self.L_robot = rospy.get_param('~robot_length', 0.9)
        self.W_robot = rospy.get_param('~robot_width', 0.45)

        # FIXED maximum number of halfplanes
        self.max_halfplanes = rospy.get_param('~max_halfplanes', 20)

        # Cost weights
        self.Q = np.diag([
            rospy.get_param('~Q_x', 1.0),
            rospy.get_param('~Q_y', 1.0),
            rospy.get_param('~Q_psi', 0.0)
        ])
        self.R = np.diag([
            rospy.get_param('~R_f1', 0.1),
            rospy.get_param('~R_f2', 0.1),
            rospy.get_param('~R_f3', 0.2),
            rospy.get_param('~R_f4', 0.2)
        ])

        # Reference state (updated via callback)
        self.x_ref = np.array([0.0, 0.0, 0.0])
        self.ref_received = False

        # Boat dynamics parameters (QuarterScale)
        self.m11 = rospy.get_param('~system_dynamics/m11', 12.0)
        self.m22 = rospy.get_param('~system_dynamics/m22', 16.0)
        self.m33 = rospy.get_param('~system_dynamics/m33', 3.0)
        self.d11 = rospy.get_param('~system_dynamics/d11', 6.0)
        self.d22 = rospy.get_param('~system_dynamics/d22', 8.0)
        self.d33 = rospy.get_param('~system_dynamics/d33', 0.6)
        self.aa = rospy.get_param('~system_dynamics/aa', 0.45)
        self.bb = rospy.get_param('~system_dynamics/bb', 0.9)

        # State dimensions
        self.nx = 6  # [x, y, psi, u, v, r]
        self.nu = 4  # [F1, F2, F3, F4]

        # =====================================================================
        # State variables
        # =====================================================================
        self.current_state = np.zeros(self.nx)
        self.odom_received = False
        self.poly_received = False

        # Initialize polytope with dummy values (Ax <= b convention from decomp)
        # Dummy: 0*x <= small_positive is always satisfied
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0
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

        self.ref_sub = rospy.Subscriber(
            '/reference_pose', PoseStamped, self.ref_callback, queue_size=1
        )

        # subscriber to set reference point based on 2D RVIZ goal
        self.ref_sub = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self.rviz_callback, queue_size=1
        )

        rospy.loginfo("MPC-CBF Node initialized")
        rospy.loginfo(f"  Horizon: {self.N}, dt: {self.dt}, CBF steps: {self.N_cbf}")
        rospy.loginfo(f"  Max halfplanes: {self.max_halfplanes}")
        rospy.loginfo(f"  Robot size: {self.L_robot} x {self.W_robot}")
        rospy.loginfo(f"  Frame: ROS/ENU -> MPC/NED (y and psi inverted)")

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

        # 4 vertices in body frame
        local_x = ca.vertcat(L/2, L/2, -L/2, -L/2)
        local_y = ca.vertcat(W/2, -W/2, -W/2, W/2)

        # Transform to world (NED)
        world_x = c * local_x - s * local_y + x
        world_y = s * local_x + c * local_y + y

        return ca.horzcat(world_x, world_y)

    def _build_ocp(self):
        """
        Build the OCP using ELEMENT-WISE CBF constraints.
        Removes LogSumExp to fix vanishing gradients when 'safe'.
        """
        rospy.loginfo("Building OCP with Element-wise Linear CBF...")

        self.opti = ca.Opti()

        # Decision variables
        self.X = []
        self.U = []
        for k in range(self.N):
            self.X.append(self.opti.variable(self.nx))
            self.U.append(self.opti.variable(self.nu))
        self.X.append(self.opti.variable(self.nx))

        # Parameters
        self.X0_param = self.opti.parameter(self.nx)
        self.x_ref_param = self.opti.parameter(3)

        # Polytope parameters
        self.A_param = self.opti.parameter(self.max_halfplanes, 2)
        self.b_param = self.opti.parameter(self.max_halfplanes)

        # Cost function
        cost = 0
        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            # Tracking Cost
            err = xk[0:3] - self.x_ref_param
            cost += ca.mtimes([err.T, self.Q, err])
            # Input Cost
            cost += ca.mtimes([uk.T, self.R, uk])

        # Terminal Cost
        err_term = self.X[self.N][0:3] - self.x_ref_param
        cost += ca.mtimes([err_term.T, self.Q * 10, err_term])
        
        self.opti.minimize(cost)

        # Dynamics and Constraints
        for k in range(self.N):
            xk = self.X[k]
            uk = self.U[k]
            x_next = self.X[k + 1]

            # Dynamics
            dxdt = self._boat_dynamics(xk, uk)
            x_next_model = xk + dxdt * self.dt
            self.opti.subject_to(x_next == x_next_model)

            if k == 0:
                self.opti.subject_to(self.X[0] == self.X0_param)

            # Input limits
            self.opti.subject_to(self.opti.bounded(-self.max_force, uk, self.max_force))

            # State limits
            self.opti.subject_to(self.opti.bounded(-self.max_surge, xk[3], self.max_surge))
            self.opti.subject_to(self.opti.bounded(-self.max_sway, xk[4], self.max_sway))

            # ===============================================================
            # ROBUST CBF IMPLEMENTATION
            # ===============================================================
            if k < self.N_cbf:
                # 1. Get Vertices for Current State and Next State
                # Returns matrix (4, 2) -> Transpose to (2, 4) for multiplication
                verts_k = self._robot_vertices_ca(xk[0], xk[1], xk[2]).T
                verts_k1 = self._robot_vertices_ca(x_next_model[0], x_next_model[1], x_next_model[2]).T

                # 2. Compute Distances to ALL halfplanes for ALL vertices
                # A_param is (M, 2), verts is (2, 4)
                # h = b - A*v
                # Result h_mat is (M, 4). Each element represents distance of vertex j to plane i
                
                # Distances at step k
                Ax_k = ca.mtimes(self.A_param, verts_k) # (M, 4)
                h_mat_k = self.b_param - Ax_k # Broadcasting b (M,) to (M,4)

                # Distances at step k+1
                Ax_k1 = ca.mtimes(self.A_param, verts_k1)
                h_mat_k1 = self.b_param - Ax_k1

                # 3. Apply Element-wise CBF
                # Instead of logsumexp, we enforce the decay on EVERY vertex-plane pair.
                # If a plane is a dummy (0*x <= 10), h is 10. Constraint becomes 10 >= 0.8*10 + ... (Trivial)
                # If a plane is active, it enforces the brake.
                
                # Flatten matrices to vectors for subject_to
                h_vec_k = ca.vec(h_mat_k)
                h_vec_k1 = ca.vec(h_mat_k1)
                
                # The Constraint: h_next >= gamma * h_curr + (1-gamma)*margin
                # This generates M*4 linear constraints. Fatrop handles this very efficiently.
                limit = self.max_approx 
                self.opti.subject_to(h_vec_k1 >= self.gamma * h_vec_k + (1 - self.gamma) * limit)

        # Solver options
        opts = {
            "fatrop.print_level": 0,
            "print_time": 0,
            "fatrop.max_iter": 100,
            "fatrop.tol": 1e-4,
            "structure_detection": "auto",
            "expand": True
        }
        self.opti.solver('fatrop', opts)
        rospy.loginfo("OCP built successfully (Element-wise CBF)")

    def odom_callback(self, msg):
        """
        Handle odometry messages.
        Transform from ROS/ENU to MPC/NED frame.
        """
        # Position: x stays, y inverts
        self.current_state[0] = msg.pose.pose.position.x
        self.current_state[1] = -msg.pose.pose.position.y  # ENU->NED

        # Orientation: yaw inverts
        q = msg.pose.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_state[2] = -yaw_enu  # ENU->NED

        # Body velocities: u stays, v inverts, r inverts
        self.current_state[3] = msg.twist.twist.linear.x   # surge (forward)
        self.current_state[4] = -msg.twist.twist.linear.y  # sway: left->right
        self.current_state[5] = -msg.twist.twist.angular.z # yaw rate inverts
        
        self.odom_received = True

    def ref_callback(self, msg):
        """
        Update reference pose from a PoseStamped message.
        Transform from ROS/ENU to MPC/NED frame.
        """
        self.x_ref[0] = msg.pose.position.x
        self.x_ref[1] = msg.pose.position.y
        
        q = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.x_ref[2] = yaw 
        
        self.ref_received = True

    def rviz_callback(self, msg):
        """
        Update reference pose from RVIZ 2D goal (PoseStamped).
        Transform from ROS/ENU to MPC/NED frame.
        """
        self.x_ref[0] = msg.pose.position.x
        self.x_ref[1] = -msg.pose.position.y  # ENU->NED
        
        q = msg.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.x_ref[2] = -yaw_enu  # ENU->NED
        
        self.ref_received = True

    def poly_callback(self, msg):
        """
        Handle polyhedron messages.
        Transform from ROS/ENU to MPC/NED frame.
        
        decomp_ros_msgs/Polyhedron has:
        - normals: Point[] - normal vectors n (in ENU)
        - points: Point[] - points p0 on each hyperplane (in ENU)
        
        DecompUtil convention: n · (x - p0) <= 0, i.e., Ax <= b where A=n, b=n·p0
        
        ENU constraint: n_enu · x_enu <= b_enu
        
        For a point x_enu = [x, y], x_ned = [x, -y]
        Substituting y_enu = -y_ned:
            n_enu[0] * x + n_enu[1] * y_enu <= b_enu
            n_enu[0] * x + n_enu[1] * (-y_ned) <= b_enu
            n_enu[0] * x - n_enu[1] * y_ned <= b_enu
        
        So: A_ned = [n_enu[0], -n_enu[1]], b_ned = b_enu (unchanged scalar)
        """
        if len(msg.polyhedrons) == 0:
            rospy.logwarn_throttle(1.0, "Empty polyhedron array received")
            return

        poly = msg.polyhedrons[0]
        
        # Reset to dummy values (A=0, b=10 gives h = b - A@x = 10, always positive)
        self.A_poly = np.zeros((self.max_halfplanes, 2))
        self.b_poly = np.ones(self.max_halfplanes) * 10.0

        idx = 0
        for i in range(len(poly.normals)):
            nx_enu = poly.normals[i].x
            ny_enu = poly.normals[i].y
            nz = poly.normals[i].z
            
            # Skip z-axis constraints
            if abs(nz) > 0.1:
                continue
                
            if idx >= self.max_halfplanes:
                rospy.logwarn(f"Too many 2D halfplanes, truncating to {self.max_halfplanes}")
                break
            
            px_enu = poly.points[i].x
            py_enu = poly.points[i].y
            
            # Transform normal to NED: A_ned = [n_x, -n_y]
            self.A_poly[idx, 0] = nx_enu
            self.A_poly[idx, 1] = -ny_enu
            
            # b is a scalar computed in ENU, stays the same
            # b = n_enu · p0_enu
            self.b_poly[idx] = nx_enu * px_enu + ny_enu * py_enu
            
            idx += 1

        self.n_active = idx
        self.poly_received = True
        # rospy.loginfo_throttle(1.0, f"Polytope updated: {idx} active 2D halfplanes")

    def solve_mpc(self):
        """Solve the MPC problem with robust warm start handling."""
        if not self.odom_received:
            rospy.logwarn_throttle(1.0, "No odometry received yet")
            return None, None

        if not self.ref_received:
            rospy.logwarn_throttle(1.0, "No reference received yet")
            return None, None

        # 1. Update Parameters
        self.opti.set_value(self.X0_param, self.current_state)
        self.opti.set_value(self.x_ref_param, self.x_ref)
        self.opti.set_value(self.A_param, self.A_poly)
        self.opti.set_value(self.b_param, self.b_poly)

        # 2. Solve
        try:
            sol = self.opti.solve()

            # --- SUCCESS CASE ---
            
            # Extract solutions
            u_vals = [sol.value(u) for u in self.U]
            x_vals = [sol.value(x) for x in self.X]
            
            # Format for return/plotting
            u_opt = np.array(u_vals).T  # (nu, N)
            x_opt = np.array(x_vals).T  # (nx, N+1)

            # --- WARM START STRATEGY: SHIFT ---
            # Use x[k+1] as initial guess for x[k] in next iteration
            
            # Shift Inputs: U[0] <- U[1], ..., U[N-1] <- U[N-1] (copy last)
            for k in range(self.N - 1):
                self.opti.set_initial(self.U[k], u_vals[k+1])
            self.opti.set_initial(self.U[self.N - 1], u_vals[self.N - 1]) # Duplicate last input

            # Shift States: X[0] <- X[1], ..., X[N] <- X[N] (copy last)
            # Note: X[0] is constrained to current_state, but good to init anyway
            for k in range(self.N):
                self.opti.set_initial(self.X[k], x_vals[k+1])
            self.opti.set_initial(self.X[self.N], x_vals[self.N]) # Duplicate last state

            return u_opt, x_opt

        except RuntimeError as e:
            rospy.logwarn(f"MPC solver failed: {e}")
            
            # --- FAILURE CASE: RESCUE ---
            # The solver is lost. Do NOT use the debug values for the next warm start.
            # Reset the guess to the current state (stationary guess).
            
            # Reset Inputs to Zero (Safe bet)
            zeros_u = np.zeros(self.nu)
            for k in range(self.N):
                self.opti.set_initial(self.U[k], zeros_u)
            
            # Reset States to Current State (trajectory stays at current pose)
            # This ensures the solver starts from a dynamically valid (though static) point next time
            for k in range(self.N + 1):
                self.opti.set_initial(self.X[k], self.current_state)

            # --- RETURN DEBUG VALUES FOR LOGGING ONLY ---
            try:
                # We still return the failed trajectory so we can see in Rviz WHY it failed
                # But we have already scrubbed the memory (set_initial) for the next loop.
                u_debug = [self.opti.debug.value(u) for u in self.U]
                x_debug = [self.opti.debug.value(x) for x in self.X]
                return np.array(u_debug).T, np.array(x_debug).T
            except:
                return None, None

    def publish_force(self, u):
        """Publish force command (forces are in body frame, no transform needed)."""
        msg = Force()
        msg.data = [float(u[0]), float(u[1]), float(u[2]), float(u[3])]
        self.force_pub.publish(msg)

    def publish_trajectory(self, x_opt):
        """
        Publish predicted trajectory.
        Transform from MPC/NED back to ROS/ENU frame.
        """
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"

        for k in range(self.N + 1):
            pose = PoseStamped()
            pose.header = path_msg.header
            
            # Transform position: NED->ENU (y inverts back)
            pose.pose.position.x = x_opt[0, k]
            pose.pose.position.y = -x_opt[1, k]  # NED->ENU
            pose.pose.position.z = 0.0

            # Transform orientation: NED->ENU (yaw inverts back)
            yaw_enu = -x_opt[2, k]  # NED->ENU
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = np.sin(yaw_enu / 2)
            pose.pose.orientation.w = np.cos(yaw_enu / 2)

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