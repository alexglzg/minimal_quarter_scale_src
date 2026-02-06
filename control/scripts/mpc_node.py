#!/usr/bin/env python3
import rospy
import numpy as np


# from mpc_rockit_core import MPCController
from roboat_core.msg import Force
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from path import SinePath

from scipy.optimize import minimize

class MPCNode:
    def __init__(self):
        # config = self.load_params()
        # self.mpc = MPCController(config)

        # self.cmd_pub = rospy.Publisher("/mpc_force", Force, queue_size=1)
        rospy.Subscriber("odometry/filtered", Odometry, self.odom_cb)

        # path_param = rospy.get_param("path")
        self.path = SinePath()

    def odom_cb(self, msg):
        x = self.odom_to_state(msg)
        # u = self.mpc.solve(x)
        # self.publish_cmd(u)
        # print something to show that the callback is working
        print("Received odometry message, current state:", x)

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

    # def load_params():
    #     model = rospy.get_param("parameters_model")
    #     mpc = rospy.get_param("parameters_mpc")
    #     scenario = rospy.get_param("parameters_scenario")
    #     cbf = rospy.get_param("parameters_cbf")
    #     path = rospy.get_param("path")

    #     return model, mpc, scenario, cbf, path


if __name__ == "__main__":
    rospy.init_node("mpc_controller")
    MPCNode()
    rospy.spin()
