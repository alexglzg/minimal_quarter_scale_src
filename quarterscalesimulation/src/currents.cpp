/** ----------------------------------------------------------------------------
 * @file:     currents.cpp
 * @date:     Sep 7, 2022
 * @datemod:  Sep 7, 2022
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: Disturbances from currents. 
 * ---------------------------------------------------------------------------*/

#include <iostream>
#include <random>
#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include <math.h>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class Currents
{
public:
    float integral_step;

    float V_current;
    float beta_current;
    float mu;

    float V_c_dot;
    float V_c_dot_last;

    float scale_factor;

    float noise_cur;
    float mean_cur;
    float stddev_cur;
    float v_min;
    float v_max;
    std::default_random_engine generator_cur;

    geometry_msgs::Pose2D disturbance; //disturbance from currents

    Currents()
    {
        disturbance_pub = n.advertise<geometry_msgs::Pose2D>("/roboat_currents", 1);

        static const float r_beta_current = 0.0;
        static const float r_stddev_cur = 1.0;
        static const float r_mu = 0.9;
        static const float r_scale_factor = 1.0;
        static const float r_v_min = 0.0;
        static const float r_v_max = 1.0;

        n.param("currents/beta_current", beta_current, r_beta_current);
        n.param("currents/stddev_cur", stddev_cur, r_stddev_cur);
        n.param("currents/mu", mu, r_mu);
        n.param("currents/scale_factor", scale_factor, r_scale_factor);
        n.param("currents/v_min", v_min, r_v_min);
        n.param("currents/v_max", v_max, r_v_max);

        V_current = 0.0;
        mean_cur = 0.0;
        V_c_dot_last = 0.0;
    }

    void time_step()
    {
        std::normal_distribution<float> dist_cur(mean_cur, stddev_cur);

        noise_cur = dist_cur(generator_cur) * scale_factor;

        V_c_dot = noise_cur - mu*V_current;
        V_current = integral_step * (V_c_dot + V_c_dot_last)/2 + V_current;
        V_current = std::min(std::max(V_current, v_min), v_max);
        V_c_dot_last = V_c_dot;

        disturbance.x = V_current;
        disturbance.y = 0.0;
        disturbance.theta = beta_current;

        //Data publishing
        disturbance_pub.publish(disturbance);
    }

private:
    ros::NodeHandle n;

    ros::Publisher disturbance_pub;

};

//Main
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "currents");
    Currents currents;
    currents.integral_step = 0.01;
    int rate = 100;
    ros::Rate loop_rate(rate);

  while (ros::ok())
  {
    currents.time_step();
    ros::spinOnce();
    loop_rate.sleep();
  }

    return 0;
}