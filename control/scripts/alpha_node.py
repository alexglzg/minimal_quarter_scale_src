#!/usr/bin/env python3
import rospy
import numpy as np

from control.msg import AlphaArray


class AlphaNode:
    """Publishes per-obstacle alpha1/alpha2 CBF gains for mpc_node to consume."""

    def __init__(self):
        mpc_p = rospy.get_param("parameters_mpc")
        alpha_p = rospy.get_param("parameters_alpha")

        self.num_obs = mpc_p["num_obstacles"]
        self.alpha1_min = alpha_p["alpha1_min"]
        self.alpha1_max = alpha_p["alpha1_max"]
        self.alpha2_min = alpha_p["alpha2_min"]
        self.alpha2_max = alpha_p["alpha2_max"]

        self.alpha_pub = rospy.Publisher("/cbf_alphas", AlphaArray, queue_size=1)

        update_rate = alpha_p["update_rate"]
        self.publish_alphas()
        rospy.Timer(rospy.Duration(1.0 / update_rate), lambda _: self.publish_alphas())

    def decide_alphas(self):
        # Placeholder policy: i.i.d. uniform samples per obstacle, resampled on a
        # timer. Intended drop-in replacement point for a learned (NN) policy later.
        alpha1 = np.random.uniform(self.alpha1_min, self.alpha1_max, self.num_obs)
        alpha2 = np.random.uniform(self.alpha2_min, self.alpha2_max, self.num_obs)
        return alpha1, alpha2

    def publish_alphas(self):
        alpha1, alpha2 = self.decide_alphas()
        msg = AlphaArray()
        msg.alpha1 = alpha1.tolist()
        msg.alpha2 = alpha2.tolist()
        self.alpha_pub.publish(msg)


if __name__ == "__main__":
    rospy.init_node("alpha_node")
    AlphaNode()
    rospy.spin()
