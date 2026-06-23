#!/usr/bin/env python3

import math

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import euler_from_quaternion
from obstacle_detector.msg import BuoyArray


def rect_circle_collision(local_x, local_y, half_length, half_width, radius):
    """True if a circle (radius, centered at local_x/y in the rectangle's frame)
    overlaps an axis-aligned rectangle spanning +/-half_length, +/-half_width."""
    closest_x = max(-half_length, min(half_length, local_x))
    closest_y = max(-half_width, min(half_width, local_y))
    return math.hypot(local_x - closest_x, local_y - closest_y) <= radius


class CollisionChecker:
    def __init__(self):
        self.half_length = rospy.get_param(
            "/parameters_model/ego_length", rospy.get_param("~boat_length", 0.9)
        ) / 2.0
        self.half_width = rospy.get_param(
            "/parameters_model/ego_width", rospy.get_param("~boat_width", 0.45)
        ) / 2.0

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.light_height = rospy.get_param("~light_height", 1.2)
        check_rate = rospy.get_param("~check_rate", 20.0)

        self.boat_pose = None  # (x, y, yaw)
        self.obstacles = []    # list of (x, y, radius)

        self.collision_pub = rospy.Publisher("/collision_detected", Bool, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("/collision_markers", MarkerArray, queue_size=1)

        rospy.Subscriber(rospy.get_param("~odom_topic", "odometry/filtered"), Odometry, self.odom_cb)
        rospy.Subscriber(rospy.get_param("~buoy_topic", "/buoy_array"), BuoyArray, self.buoy_cb)

        rospy.Timer(rospy.Duration(1.0 / check_rate), self.check)

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.boat_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def buoy_cb(self, msg):
        self.obstacles = [
            (b.odom.pose.pose.position.x, b.odom.pose.pose.position.y, b.radius)
            for b in msg.buoys
        ]

    def check(self, event):
        if self.boat_pose is None:
            return

        bx, by, yaw = self.boat_pose
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        collided = False
        for ox, oy, r in self.obstacles:
            dx, dy = ox - bx, oy - by
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            if rect_circle_collision(local_x, local_y, self.half_length, self.half_width, r):
                collided = True
                break

        self.collision_pub.publish(Bool(collided))
        self.publish_markers(bx, by, yaw, collided)

    def publish_markers(self, bx, by, yaw, collided):
        stamp = rospy.Time.now()
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        footprint = Marker()
        footprint.header.frame_id = self.frame_id
        footprint.header.stamp = stamp
        footprint.ns = "collision_checker"
        footprint.id = 0
        footprint.type = Marker.LINE_STRIP
        footprint.action = Marker.ADD
        footprint.pose.orientation.w = 1.0
        footprint.scale.x = 0.03
        footprint.color.a = 1.0
        if collided:
            footprint.color.r, footprint.color.g, footprint.color.b = 1.0, 0.0, 0.0
        else:
            footprint.color.r, footprint.color.g, footprint.color.b = 0.0, 0.4, 1.0

        corners_local = [
            (self.half_length, self.half_width),
            (self.half_length, -self.half_width),
            (-self.half_length, -self.half_width),
            (-self.half_length, self.half_width),
            (self.half_length, self.half_width),
        ]
        for lx, ly in corners_local:
            footprint.points.append(Point(
                x=bx + lx * cos_yaw - ly * sin_yaw,
                y=by + lx * sin_yaw + ly * cos_yaw,
                z=0.05,
            ))

        light = Marker()
        light.header.frame_id = self.frame_id
        light.header.stamp = stamp
        light.ns = "collision_checker"
        light.id = 1
        light.type = Marker.SPHERE
        light.action = Marker.ADD
        light.pose.position.x = bx
        light.pose.position.y = by
        light.pose.position.z = self.light_height
        light.pose.orientation.w = 1.0
        light.scale.x = light.scale.y = light.scale.z = 0.25
        light.color.r = 1.0
        light.color.g = 0.0
        light.color.b = 0.0
        light.color.a = 1.0 if collided else 0.0  # only lit up while colliding

        self.marker_pub.publish(MarkerArray(markers=[footprint, light]))


def main():
    rospy.init_node("collision_checker")
    CollisionChecker()
    rospy.spin()


if __name__ == "__main__":
    main()
