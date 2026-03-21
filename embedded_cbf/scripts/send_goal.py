#!/usr/bin/env python3
"""Publish a single goal to /move_base_simple/goal after a delay."""
import rospy
import math
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node('send_goal', anonymous=True)

    goal_x = rospy.get_param('~goal_x', 0.0)
    goal_y = rospy.get_param('~goal_y', 0.0)
    goal_yaw = rospy.get_param('~goal_yaw', 0.0)
    delay = rospy.get_param('~delay', 8.0)

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1, latch=True)

    rospy.loginfo(f"Waiting {delay}s before publishing goal [{goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.1f} deg]")
    rospy.sleep(delay)

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"
    msg.pose.position.x = goal_x
    msg.pose.position.y = goal_y
    msg.pose.orientation.z = math.sin(goal_yaw / 2.0)
    msg.pose.orientation.w = math.cos(goal_yaw / 2.0)

    pub.publish(msg)
    rospy.loginfo(f"Goal published: [{goal_x:.2f}, {goal_y:.2f}, {math.degrees(goal_yaw):.1f} deg]")

    # Keep alive so the latched message stays
    rospy.spin()

if __name__ == '__main__':
    main()