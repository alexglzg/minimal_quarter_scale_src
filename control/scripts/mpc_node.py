#!/usr/bin/env python3
import rospy
import numpy as np


from mpc_rockit_core import MPCController
from roboat_core.msg import Force
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from path import SinePath, StraightLinePath
from obstacle_detector.msg import Buoy, BuoyArray

from scipy.optimize import minimize

class MPCNode:
    def __init__(self):
        model_p, mpc_p, path_p = self.load_params()

        if path_p["type"] == "sine":
            self.path = SinePath(path_p["x_multiplier"], path_p["y_offset"])
        elif path_p["type"] == "straight_line":
            self.path = StraightLinePath(path_p["slope"], path_p["intercept"])
        else:
            raise ValueError("Unsupported path type")

        self.mpc = MPCController(model_p, mpc_p, self.path)
        self.initial_guess_state = np.zeros(self.mpc.nx)
        self.initial_guess_control = np.zeros(self.mpc.nu)

        self.cmd_pub = rospy.Publisher("/mpc_force", Force, queue_size=1)
        rospy.Subscriber("odometry/filtered", Odometry, self.odom_cb)

        # Handling obstacles
        num_obstacles = mpc_p["num_obstacles"]
        self.dummy_x = mpc_p["dummy_x"]
        self.dummy_y = mpc_p["dummy_y"]
        self.dummy_radius = mpc_p["dummy_radius"]
        self.obstacle_x = np.ones(num_obstacles)*mpc_p["dummy_x"]
        self.obstacle_y = np.ones(num_obstacles)*mpc_p["dummy_y"]
        self.obstacle_radius = np.ones(num_obstacles)*mpc_p["dummy_radius"]
        rospy.Subscriber("/buoy_array", BuoyArray, self.buoy_array_cb)

        self.path_pub = rospy.Publisher(
            "/desired_path",
            Path,
            queue_size=1,
            latch=True   # important
        )
        self.publish_path()
        rospy.Timer(rospy.Duration(1.0), lambda _: self.publish_path())

        # Control loop at 10 Hz
        self.current_state = None
        self.control_timer = rospy.Timer(
        rospy.Duration(0.1),   # 10 Hz control
        self.control_loop
        )


    def buoy_array_cb(self, msg):
        obstacles = []

        for b in msg.buoys:
            # Extract position from Odometry
            x = b.odom.pose.pose.position.x
            y = -b.odom.pose.pose.position.y

            # Radius comes from your custom message
            r = b.radius

            obstacles.append((x, y, r))

        # Find indices of num_obs closest obstacles
        if len(obstacles) > 0 and self.current_state is not None:
            distances = [np.hypot(x - self.current_state[0], y - self.current_state[1]) for x, y, r in obstacles]
            closest_indices = np.argsort(distances)[:self.mpc.num_obs]
        
            # select only the closest num_obs obstacles (but keeping the order in the original list to maintain consistency)
            obstacles_selected = []
            for i in range(len(obstacles)):
                if i in closest_indices:
                    obstacles_selected.append(obstacles[i])

        else:
                obstacles_selected = []
        
        while len(obstacles_selected) < self.mpc.num_obs:
                obstacles_selected.append((self.dummy_x, self.dummy_y, self.dummy_radius))

        # Update obstacle parameters for MPC
        for i, (x, y, r) in enumerate(obstacles_selected):
            self.obstacle_x[i] = x
            self.obstacle_y[i] = y
            self.obstacle_radius[i] = r
            
        
    def odom_cb(self, msg):
        self.current_state = self.odom_to_state(msg)
        # u = self.mpc.solve(x, [1000, 1000], [1000, 1000], 
        #                        [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], 1.0, 1.0)
        # print("Control output from MPC:", u)                              
        # self.publish_cmd(u)
        # print something to show that the callback is working
        # print("Received odometry message, current state:", x)

    def odom_to_state(self, msg):
        
        """
        Handle odometry messages.
        Transform from ROS/ENU to MPC/NED frame and compute path parameter s.
        """
        # Position: x stays, y inverts
        x_ned = msg.pose.pose.position.x
        y_ned = -msg.pose.pose.position.y  # ENU->NED

        # Orientation: yaw inverts
        q = msg.pose.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_ned = -yaw_enu  # ENU->NED

        # Body velocities: u stays, v inverts, r inverts
        u = msg.twist.twist.linear.x   # surge (forward)
        v = -msg.twist.twist.linear.y  # sway: left->right
        r = -msg.twist.twist.angular.z # yaw rate inverts

        s = minimize(self.path.distance_cost, 
                        0, method='Nelder-Mead', 
                        args=(x_ned, y_ned),
                        options={'xatol': 1e-8, 'disp': False}).x[0]

        return np.array([
            x_ned,
            y_ned,
            yaw_ned,
            u,
            v,
            r,
            s
        ])        

    def control_loop(self, event):
        if self.current_state is None:
            return
        print("Current state for MPC:", self.current_state)
        print("Current obstacles for MPC:", list(zip(self.obstacle_x, self.obstacle_y, self.obstacle_radius)))
        u, U, X = self.mpc.solve(self.current_state, self.obstacle_x, self.obstacle_y, 
                               self.obstacle_radius, 0.5*np.ones(self.mpc.num_obs), 0.5*np.ones(self.mpc.num_obs),
                               self.initial_guess_state, self.initial_guess_control)  
        self.initial_guess_state = X 
        self.initial_guess_control = U 
        print("Control output from MPC:", u)                              
        self.publish_cmd(u)

    def publish_cmd(self, u):
        cmd_msg = Force()
        cmd_msg.data = u
        self.cmd_pub.publish(cmd_msg)

    def load_params(self):
        model = rospy.get_param("parameters_model")
        mpc = rospy.get_param("parameters_mpc")
        path = rospy.get_param("path")

        return model, mpc, path

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
    rospy.init_node("mpc_controller")
    MPCNode()
    rospy.spin()
