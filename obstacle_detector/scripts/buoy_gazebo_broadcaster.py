#!/usr/bin/env python3
"""Bridges /buoy_array -> /gazebo/set_model_state for each buoy, same pattern
as quarterscalesimulation/qs_gazebo_broadcaster.cpp does for the roboat itself
(subscribe to a recorded/simulated pose topic, republish as a Gazebo
ModelState so a spawned model visually tracks it). No name field on
obstacle_detector/Buoy, so buoys are matched by array index -> "buoy1",
"buoy2", "buoy3", matching the spawn order used when they were placed into
the world and matching buoy_simulator.py's own fixed publish order.
"""
import rospy
from gazebo_msgs.msg import ModelState
from obstacle_detector.msg import BuoyArray
from tf.transformations import quaternion_from_euler


_count = [0]


def callback(msg, pub):
    _count[0] += 1
    for i, b in enumerate(msg.buoys):
        state = ModelState()
        state.model_name = f"buoy{i + 1}"
        state.pose.position.x = b.odom.pose.pose.position.x
        state.pose.position.y = b.odom.pose.pose.position.y
        state.pose.position.z = 0.0
        state.pose.orientation = b.odom.pose.pose.orientation
        pub.publish(state)
    if _count[0] % 20 == 1:
        rospy.loginfo("buoy_bridge: callback #%d, n_buoys=%d, buoy1 x=%.2f y=%.2f, n_conn=%d",
                      _count[0], len(msg.buoys),
                      msg.buoys[0].odom.pose.pose.position.x if msg.buoys else -1,
                      msg.buoys[0].odom.pose.pose.position.y if msg.buoys else -1,
                      pub.get_num_connections())


if __name__ == "__main__":
    rospy.init_node("buoy_gazebo_broadcaster")
    pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
    rospy.loginfo("buoy_bridge: starting, waiting for /buoy_array")
    rospy.Subscriber("/buoy_array", BuoyArray, callback, pub)
    rospy.spin()
