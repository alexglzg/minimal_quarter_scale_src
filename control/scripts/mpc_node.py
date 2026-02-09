#!/usr/bin/env python3
import rospy
import numpy as np


from mpc_rockit_core import MPCController
from roboat_core.msg import Force
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from path import SinePath

from scipy.optimize import minimize

class MPCNode:
    def __init__(self):
        model_p, mpc_p, scenario_p, cbf_p, path_p = self.load_params()

        if path_p["type"] == "sine":
            self.path = SinePath(path_p["x_multiplier"], path_p["y_offset"])
        else:
            raise ValueError("Unsupported path type")

        self.mpc = MPCController(model_p, mpc_p, scenario_p, self.path)

        self.cmd_pub = rospy.Publisher("/mpc_force", Force, queue_size=1)
        rospy.Subscriber("odometry/filtered", Odometry, self.odom_cb)

        self.path_pub = rospy.Publisher(
            "/desired_path",
            Path,
            queue_size=1,
            latch=True   # important
        )
        self.publish_path()
        
        # Contr loop at 10 Hz
        self.current_state = None
        self.control_timer = rospy.Timer(
        rospy.Duration(0.1),   # 10 Hz control
        self.control_loop
        )

        
    def odom_cb(self, msg):
        self.current_state = self.odom_to_state(msg)
        # u = self.mpc.solve(x, [1000, 1000], [1000, 1000], 
        #                        [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], 1.0, 1.0)
        # print("Control output from MPC:", u)                              
        # self.publish_cmd(u)
        # print something to show that the callback is working
        # print("Received odometry message, current state:", x)

    def odom_to_state(self, msg):
        
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        s = minimize(self.path.distance_cost, 
                        0, method='Nelder-Mead', 
                        args=(msg.pose.pose.position.x, -msg.pose.pose.position.y),
                        options={'xatol': 1e-8, 'disp': True}).x[0]

        return np.array([
            msg.pose.pose.position.x,
            -msg.pose.pose.position.y,
            -yaw,
            msg.twist.twist.linear.x,
            -msg.twist.twist.linear.y,
            -msg.twist.twist.angular.z,
            s
        ])        

    def control_loop(self, event):
        if self.current_state is None:
            return
        print("Current state for MPC:", self.current_state)
        u = self.mpc.solve(self.current_state, [1000, 1000], [1000, 1000], 
                               [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], 1.0, 1.0)
        print("Control output from MPC:", u)                              
        self.publish_cmd(u)

    def publish_cmd(self, u):
        cmd_msg = Force()
        cmd_msg.data = u
        self.cmd_pub.publish(cmd_msg)

    def load_params(self):
        model = rospy.get_param("parameters_model")
        mpc = rospy.get_param("parameters_mpc")
        scenario = rospy.get_param("parameters_scenario")
        cbf = rospy.get_param("parameters_cbf")
        path = rospy.get_param("path")

        return model, mpc, scenario, cbf, path

    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "map"
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
