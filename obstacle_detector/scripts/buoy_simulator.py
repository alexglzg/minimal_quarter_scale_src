#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from obstacle_detector.msg import Buoy, BuoyArray
from geometry_msgs.msg import Pose2D, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray
from scipy.optimize import fsolve


class FreeBuoySimulator:
    def __init__(self, name, R, m, initial_state):
        self.name = name.lower()
        self.R = R # radius of the buoy in meters
        self.m = m # mass of the buoy in kg
        self.rho = 1000 # Density of water in kg/m^3        
        self.Ca = 1.0 # Added mass coefficient
        self.Cd = 1.1 # Drag coefficient

        self.h = self.compute_draft() # draft of the buoy in meters (the submerged part)

        self.M, self.D = self.compute_buoy_model()

        # State: [x, y, psi, u, v, r]
        self.state = np.array(initial_state).reshape(6,-1)

        # Publisher of the buoy state
        self.odom_pub = rospy.Publisher(f"/{name}/odometry", Odometry, queue_size=10)
        # Publisher of the buoy marker for visualization in Rviz
        # self.marker_pub = rospy.Publisher(f"/{name}/marker", Marker, queue_size=10)

        self.marker_id = hash(name) % 1000
        
        # Subscribers to the disturbances and currents
        rospy.Subscriber("/roboat_disturbance", Pose2D, self.disturbance_callback)
        rospy.Subscriber("/roboat_currents", Pose2D, self.current_callback)

        # Subscriber to odometry of the vessel to model interaction
        # rospy.Subscriber("odometry/filtered", Odometry, self.vessel_odometry_callback)

        self.nu_u = 0.0
        self.nu_v = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_theta = 0.0
        self.F_body = np.array([0.0, 0.0])

        self.t0 = rospy.get_time()
        rospy.Timer(rospy.Duration(0.01), self.update)

    def vessel_odometry_callback(self, msg):

        sigma = 3.0
        sigma_r = 2.0
        k = 0.1
        kr = 0.25
        kd = 3.0

        x_vessel = msg.pose.pose.position.x
        y_vessel = -msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        psi_vessel = -euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]
        u_vessel = msg.twist.twist.linear.x
        v_vessel = -msg.twist.twist.linear.y
        V_vessel = np.sqrt(u_vessel**2 + v_vessel**2)

        dx = self.state[0,0] - x_vessel
        dy = self.state[1,0] - y_vessel
        d = np.sqrt(dx**2 + dy**2) + 1e-6 # distance between vessel and buoy, add small term to avoid division by zero
        e = np.array([np.cos(psi_vessel), np.sin(psi_vessel)]).flatten() # heading vector vessel
        
        # forward disturbance from vessel wake, modeled as a Gaussian centered at the vessel and aligned with the vessel heading
        u_forward = k * V_vessel * np.exp(-d**2/(2*sigma**2)) * e

        # sideways disturbance pushing water away from the vessel, modeled as a Gaussian centered at the vessel and pointing radially outward
        u_side = kr * np.exp(-d**2/(2*sigma_r**2)) * np.array([dx, dy]).flatten() / d

        # total water velocity disturbance at the buoy due to the vessel wake
        u_total =-u_side

        v_b_ned = np.array([
            np.cos(self.state[2,0])*self.state[3,0] - np.sin(self.state[2,0])*self.state[4,0],
            np.sin(self.state[2,0])*self.state[3,0] + np.cos(self.state[2,0])*self.state[4,0]
        ])

        rel = u_total - v_b_ned
        
        F = kd * rel

        R = np.array([
            [np.cos(self.state[2,0]), np.sin(self.state[2,0])],
            [-np.sin(self.state[2,0]), np.cos(self.state[2,0])]
        ])

        self.F_body = R @ F


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
        u_dot = -self.D[0,0]/self.M[0,0] * (u - self.nu_u) - (self.delta_x + self.F_body[0]) / self.M[0,0]
        v_dot = -self.D[1,1]/self.M[1,1] * (v - self.nu_v) - (self.delta_y + self.F_body[1]) / self.M[1,1]
        r_dot = -self.D[2,2]/self.M[2,2] * r - self.delta_theta / self.M[2,2]

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

        # self.publish_marker()

    def compute_draft(self):
        """Solve for draft h where buoyant force equals mass."""
        def draft_eq(h):
            return (np.pi * h**2 * (3*self.R - h)/3) - (self.m/self.rho) # submerged volume as a function of draft h is a spherical cap, solve for h where submerged volume * water density = mass of the buoy (Archimedes' principle)
        h_guess = self.R
        h_solution = fsolve(draft_eq, h_guess)[0]
        # clip to [0, 2*radius] to avoid unphysical values
        h_solution = np.clip(h_solution, 0, 2*self.R)
        rospy.loginfo(f"[{self.name}] radius={self.R:.3f}, mass={self.m:.1f}, draft={h_solution:.3f}")
        return h_solution

    def compute_buoy_model(self): # buoy model (sphere)
        V_sub = np.pi * self.h**2 * (3*self.R - self.h)/3 # submerged volume as a function of draft h is a spherical cap
        m_added = self.Ca * self.rho * V_sub
        m_total = self.m + m_added
        I_z = 2/5 * self.m * self.R**2 + 2/5 * m_added * self.R**2

        d11 = self.rho * self.Cd * self.R**2 * self.h
        d22 = d11
        d33 = self.rho * self.Cd * self.R**5  # approximate yaw damping

        M = np.diag([m_total, m_total, I_z])
        D = np.diag([d11, d22, d33])
        return M, D

    def publish_marker(self):
        marker_msg = Marker()
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.header.frame_id = "map"
        marker_msg.ns = "buoy"
        marker_msg.id = self.marker_id
        marker_msg.type = Marker.SPHERE
        marker_msg.action = Marker.ADD

        marker_msg.pose.position.x = self.state[0,0]
        marker_msg.pose.position.y = -self.state[1,0]
        marker_msg.pose.position.z = self.R - self.h # place the center of the sphere at the waterline (z=0), so we need to shift it down by the radius
        # quat = quaternion_from_euler(0, 0, -self.state[2,0])
        # marker_msg.pose.orientation = Quaternion(*quat)
        marker_msg.pose.orientation = Quaternion(0, 0, 0, 1)

        marker_msg.scale.x = self.R * 2 # Diameter of the sphere
        marker_msg.scale.y = self.R * 2 # Diameter of the sphere
        marker_msg.scale.z = self.R * 2 # Diameter of the sphere

        marker_msg.color.a = 1.0 # Alpha
        marker_msg.color.r = 1.0 # Red
        marker_msg.color.g = 0.5 # Green
        marker_msg.color.b = 0.0 # Blue
        self.marker_pub.publish(marker_msg)

def main():
    rospy.init_node("buoy_simulator")

    # Get buoy list from ROS param
    buoys_param = rospy.get_param("~buoys", [
        {"name": "buoy1", "radius": 0.5, "mass": 15, "initial_state": [5.0, 5.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "buoy2", "radius": 0.8, "mass": 20, "initial_state": [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]}
    ])

    buoys = []
    # print(buoys_param)
    for b in buoys_param:
        buoys.append(FreeBuoySimulator(name=b["name"], R=b["radius"], m=b["mass"], 
                                       initial_state=b["initial_state"]))
        
    buoy_array_pub = rospy.Publisher("/buoy_array", BuoyArray, queue_size=10)
    marker_array_pub = rospy.Publisher("/buoy_markers", MarkerArray, queue_size=10)
    
    def publish_all_buoys(event):
        array_msg = BuoyArray()
        array_msg.buoys = []

        marker_array = MarkerArray()
        marker_array.markers = []

        for buoy in buoys:
            # --- ODOM ENTRY ---
            buoy_msg = Buoy()
            buoy_msg.radius = buoy.R

            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "map"

            odom_msg.pose.pose.position.x = buoy.state[0,0]
            odom_msg.pose.pose.position.y = -buoy.state[1,0]
            odom_msg.pose.pose.position.z = 0.0

            quat = quaternion_from_euler(0, 0, -buoy.state[2,0])
            odom_msg.pose.pose.orientation = Quaternion(*quat)

            odom_msg.twist.twist.linear.x = buoy.state[3,0]
            odom_msg.twist.twist.linear.y = -buoy.state[4,0]
            odom_msg.twist.twist.angular.z = -buoy.state[5,0]
            
            buoy_msg.odom = odom_msg
            
            array_msg.buoys.append(buoy_msg)

            # --- MARKER ENTRY ---
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "map"
            marker.ns = "buoys"
            marker.id = buoy.marker_id

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = buoy.state[0,0]
            marker.pose.position.y = -buoy.state[1,0]
            marker.pose.position.z = buoy.R - buoy.h

            marker.pose.orientation = Quaternion(0, 0, 0, 1)

            marker.scale.x = buoy.R * 2
            marker.scale.y = buoy.R * 2
            marker.scale.z = buoy.R * 2

            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0

            marker_array.markers.append(marker)

            # --- FORCE ENTRY ---
            force_marker = Marker()
            force_marker.header.stamp = rospy.Time.now()
            force_marker.header.frame_id = "map"
            force_marker.ns = "buoy_forces"
            force_marker.id = buoy.marker_id
            force_marker.type = Marker.ARROW
            force_marker.action = Marker.ADD
            force_marker.pose.position.x = buoy.state[0,0]
            force_marker.pose.position.y = -buoy.state[1,0]
            force_marker.pose.position.z = 1.5 # place the force arrow above the buoy for better visibility
            force_marker.pose.orientation = Quaternion(0, 0, 0, 1)
            force_marker.scale.x = 0.1 # shaft diameter
            force_marker.scale.y = 0.2 # head diameter
            force_marker.scale.z = 0.2 # head length
            force_marker.color.a = 1.0
            force_marker.color.r = 0.0
            force_marker.color.g = 0.0
            force_marker.color.b = 1.0
            # Set the arrow direction and length based on the force
            F_total = np.sqrt(buoy.F_body[0]**2 + buoy.F_body[1]**2) + 1e-6 # total force magnitude, add small term to avoid division by zero
            force_marker.scale.x = 0.1 + 0.5 * F_total # scale the arrow length based on the force magnitude
            angle = np.arctan2(buoy.F_body[1], buoy.F_body[0]) # angle of the force vector
            quat = quaternion_from_euler(0, 0, angle)
            force_marker.pose.orientation = Quaternion(*quat)
            marker_array.markers.append(force_marker)

        # Publish both arrays
        buoy_array_pub.publish(array_msg)
        marker_array_pub.publish(marker_array)


    # Timer to publish array at 10 Hz
    rospy.Timer(rospy.Duration(0.1), publish_all_buoys)


    rospy.loginfo(f"Spawned {len(buoys)} buoys")
    rospy.spin()


if __name__ == "__main__":
    main()
