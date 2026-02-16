#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, Quaternion
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker

class FreeBuoySimulator:
    def __init__(self, name, R, h, initial_state):
        name = name.lower()
        self.R = R # radius of the buoy in meters
        self.h = h # draft of the buoy in meters (the submerged part)
        rho = 1000 # Density of water in kg/m^3

        V = np.pi * self.R**2 * self.h
        m = rho * V

        Ca = 1.0 # Added mass coefficient
        m_added = Ca * m

        self.m11 = m + m_added
        self.m22 = self.m11

        Iz = 0.5 * m * self.R**2
        Iz_added = 0.2 * Iz
        self.m33 = Iz + Iz_added

        Cd = 1.1 # Drag coefficient
        self.d11 = rho * Cd * self.R * self.h
        self.d22 = self.d11
        self.d33 = rho * Cd * self.R**3 * self.h

        # State: [x, y, psi, u, v, r]
        self.state = np.array(initial_state).reshape(6,-1)

        # Publisher of the buoy state
        self.odom_pub = rospy.Publisher(f"/{name}/odometry", Odometry, queue_size=10)
        # Publisher of the buoy marker for visualization in Rviz
        self.marker_pub = rospy.Publisher(f"/{name}/marker", Marker, queue_size=10)

        self.marker_id = hash(name) % 1000
        
        # Subscribers to the disturbances and currents
        rospy.Subscriber("/roboat_disturbance", Pose2D, self.disturbance_callback)
        rospy.Subscriber("/roboat_currents", Pose2D, self.current_callback)

        self.nu_u = 0.0
        self.nu_v = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_theta = 0.0

        self.t0 = rospy.get_time()
        rospy.Timer(rospy.Duration(0.01), self.update)

    def disturbance_callback(self, msg):
        self.delta_x = msg.x # North disturbance in Newtons
        self.delta_y = msg.y # East disturbance in Newtons
        self.delta_theta = msg.theta # Rotational disturbance in Nm
        # print(f"Received disturbance: delta_x={self.delta_x}, delta_y={self.delta_y}, delta_theta={self.delta_theta}")

    def current_callback(self, msg):
        V_c = msg.x # Current velocity in m/s
        beta_c = msg.theta # current direction in radians (0 means current is flowing in the positive x direction)
        self.nu_u = V_c * np.cos(beta_c - self.state[2]) # Current velocity in surge direction
        self.nu_v = V_c * np.sin(beta_c - self.state[2]) # Current velocity in sway direction
        # print(f"Received current: V_c={V_c}, beta_c={beta_c}, nu_u={self.nu_u}, nu_v={self.nu_v}")

    def update(self, event):
        # print("Updating buoy state...")
        dt = rospy.get_time() - self.t0
        self.t0 = rospy.get_time()

        # Unpack state
        psi = self.state[2]
        u = self.state[3]
        v = self.state[4]
        r = self.state[5]

        # Compute the derivatives using the equations of motion
        nedx_dot = np.cos(psi) * u - np.sin(psi) * v
        nedy_dot = np.sin(psi) * u + np.cos(psi) * v
        psi_dot = r
        u_dot = -self.d11/self.m11 * (u - self.nu_u) - self.delta_x / self.m11
        v_dot = -self.d22/self.m22 * (v - self.nu_v) - self.delta_y / self.m22
        r_dot = -self.d33/self.m33 * r - self.delta_theta / self.m33

        # Update state using Euler integration
        self.state += np.array([nedx_dot, nedy_dot, psi_dot, u_dot, v_dot, r_dot]) * dt

        # Publish the state as an Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "map"
        odom_msg.pose.pose.position.x = self.state[0,0]
        odom_msg.pose.pose.position.y = -self.state[1,0]
        odom_msg.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, -self.state[2,0])
        odom_msg.pose.pose.orientation = Quaternion(*quat)
        odom_msg.twist.twist.linear.x = self.state[3,0]
        odom_msg.twist.twist.linear.y = -self.state[4,0]
        odom_msg.twist.twist.angular.z = -self.state[5,0]
        self.odom_pub.publish(odom_msg)

        self.publish_marker()


    def publish_marker(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = "map"
        marker_msg.ns = "buoy"
        marker_msg.id = self.marker_id
        marker_msg.type = Marker.CYLINDER
        marker_msg.action = Marker.ADD
        marker_msg.pose.position.x = self.state[0,0]
        marker_msg.pose.position.y = -self.state[1,0]
        marker_msg.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, -self.state[2,0])
        marker_msg.pose.orientation = Quaternion(*quat)
        marker_msg.scale.x = self.R * 2 # Diameter of the cylinder
        marker_msg.scale.y = self.R * 2 # Diameter of the cylinder
        marker_msg.scale.z = self.h # Height of the cylinder
        marker_msg.color.a = 1.0 # Alpha
        marker_msg.color.r = 1.0 # Red
        marker_msg.color.g = 0.5 # Green
        marker_msg.color.b = 0.0 # Blue
        self.marker_pub.publish(marker_msg)

def main():
    rospy.init_node("buoy_simulator")

    # Get buoy list from ROS param
    buoys_param = rospy.get_param("~buoys", [
        {"name": "buoy1", "radius": 0.5, "draft": 0.3, "initial_state": [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "buoy2", "radius": 0.8, "draft": 0.4, "initial_state": [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]}
    ])

    buoys = []
    # print(buoys_param)
    for b in buoys_param:
        buoys.append(FreeBuoySimulator(name=b["name"], R=b["radius"], h=b["draft"], 
                                       initial_state=b["initial_state"]))

    rospy.loginfo(f"Spawned {len(buoys)} buoys")
    rospy.spin()


if __name__ == "__main__":
    main()
