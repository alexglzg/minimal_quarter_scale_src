#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf

from geometry_msgs.msg import Twist, Point, Quaternion, Vector3, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class KinematicCarNode:
    def __init__(self):
        rospy.init_node('kinematic_car_sim')

        # --- Parameters (tuned to match the repo's Rectangle Geometry) ---
        self.dt = 0.02  # 50 Hz simulation rate
        self.wheelbase = rospy.get_param('~wheelbase', 0.25)
        self.width = rospy.get_param('~width', 0.15)
        self.length = rospy.get_param('~length', 0.3)
        self.max_steer = rospy.get_param('~max_steer', 0.6)
        self.max_accel = rospy.get_param('~max_accel', 2.0)

        # --- State [x, y, theta, v] ---
        self.x = 0.15
        self.y = 0.225
        self.theta = 0.0
        self.v = 0.0

        # --- Control Input [accel, steering_angle] ---
        self.accel_cmd = 0.0
        self.steer_cmd = 0.0

        # --- Publishers & Subscribers ---
        # Input: expecting linear.x = acceleration, angular.z = steering_angle
        self.sub_cmd = rospy.Subscriber('/car_cmd', Twist, self.cmd_callback)
        self.initial_pose_sub = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose_callback)
        
        self.pub_odom = rospy.Publisher('/odometry/filtered', Odometry, queue_size=10)
        self.pub_marker = rospy.Publisher('/car_marker', Marker, queue_size=10)
        
        self.tf_broadcaster = tf.TransformBroadcaster()

        # --- Main Loop ---
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.update_physics)

    def cmd_callback(self, msg):
        """
        Interpreting Twist message as Control Input u = [a, phi]
        linear.x  -> Acceleration
        angular.z -> Steering Angle (radians)
        """
        self.accel_cmd = np.clip(msg.linear.x, -self.max_accel, self.max_accel)
        self.steer_cmd = np.clip(msg.angular.z, -self.max_steer, self.max_steer)

    def initial_pose_callback(self, msg):
        """
        Set the initial pose of the car from a PoseWithCovarianceStamped message.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        orientation_q = msg.pose.pose.orientation
        (_, _, yaw) = tf.transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )
        self.theta = yaw
        self.v = 0.0  # Reset velocity

    def update_physics(self, event):
        # 1. Kinematic Car Dynamics (Bicycle Model)
        # x_dot = v * cos(theta)
        # y_dot = v * sin(theta)
        # theta_dot = (v / L) * tan(phi)
        # v_dot = a

        # Update State
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += (self.v / self.wheelbase) * math.tan(self.steer_cmd) * self.dt
        self.v += self.accel_cmd * self.dt

        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # 2. Publish Everything
        self.publish_state()
        self.publish_shape()

    def publish_state(self):
        current_time = rospy.Time.now()
        odom_quat = tf.transformations.quaternion_from_euler(0, 0, self.theta)

        # A. Broadcast Transform (odom -> base_link)
        self.tf_broadcaster.sendTransform(
            (self.x, self.y, 0.0),
            odom_quat,
            current_time,
            "base_link",
            "odom"
        )

        # B. Publish Odometry message
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # Position
        odom.pose.pose.position = Point(self.x, self.y, 0.0)
        odom.pose.pose.orientation = Quaternion(*odom_quat)

        # Velocity (Simulated local frame)
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = (self.v / self.wheelbase) * math.tan(self.steer_cmd)

        self.pub_odom.publish(odom)

    def publish_shape(self):
        """Publishes a rectangular marker representing the car geometry"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "car_shape"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # The marker center is relative to base_link (rear axle)
        # We shift it forward by half the length minus the rear_axle_offset (approx 0.05 in repo)
        rear_axle_offset = 0.05
        center_offset_x = (self.length / 2.0) - rear_axle_offset

        marker.pose.position.x = center_offset_x
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.1 # Lift slightly above ground
        
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.length
        marker.scale.y = self.width
        marker.scale.z = 0.1 # Height

        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 0.8

        self.pub_marker.publish(marker)

if __name__ == '__main__':
    try:
        node = KinematicCarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass