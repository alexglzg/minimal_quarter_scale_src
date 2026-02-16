/** ----------------------------------------------------------------------------
 * @file:     wind_and_waves.cpp
 * @date:     Sep 7, 2022
 * @datemod:  Sep 7, 2022
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: Disturbances from wind and waves. 
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

class WindWaves
{
public:
    float integral_step;

    float x;
    float y;
    float psi;
    float u;
    float v;
    float r;

    float delta_x;
    float delta_y;
    float delta_theta;
    float scale_factor;

    float F_wave;
    float X_wave;
    float Y_wave;
    float N_wave;

    float w0;
    float beta_wave;
    float lambda;
    float Kw;
    float we;
    float U;
    float g;

    float xF1;
    float xF2;
    float xF1_dot;
    float xF2_dot;
    float xF1_dot_last;
    float xF2_dot_last;

    float xN1;
    float xN2;
    float xN1_dot;
    float xN2_dot;
    float xN1_dot_last;
    float xN2_dot_last;

    float dF;
    float dF_dot_last;
    float dN;
    float dN_dot_last;

    float wF;
    float mean_wF;
    float stddev_wF;
    std::default_random_engine generator_wF;

    float dF_dot;
    float mean_dF;
    float stddev_dF;
    std::default_random_engine generator_dF;

    float wN;
    float mean_wN;
    float stddev_wN;
    std::default_random_engine generator_wN;

    float dN_dot;
    float mean_dN;
    float stddev_dN;
    std::default_random_engine generator_dN;

    float X_wind;
    float Y_wind;
    float N_wind;
    float beta_wind;
    float V_wind;
    float CX;
    float CY;
    float CN;
    float gamma_rw;
    float delta_wind;
    float rho;
    float CDlaf_0;
    float CDlaf_pi;
    float CDlaf;
    float CDl;
    float CDt;
    float AFW;
    float ALW;
    float LOA;
    float u_w;
    float v_w;
    float u_rw;
    float v_rw;
    float V_rw;

    geometry_msgs::Pose2D disturbance; //disturbance from wind and waves

    WindWaves()
    {
        disturbance_pub = n.advertise<geometry_msgs::Pose2D>("/roboat_disturbance", 1);

        pose_sub = n.subscribe("qs_dynamics/inertial_pose", 1000, &WindWaves::pose_callback, this);
        vel_sub = n.subscribe("qs_dynamics/body_vel", 1000, &WindWaves::vel_callback, this);

        static const float r_beta_wave = 0.0;
        static const float r_stddev_wF = 1.0;
        static const float r_stddev_dF = 0.5;
        static const float r_stddev_wN = 0.2;
        static const float r_stddev_dN = 0.1;
        static const float r_beta_wind = 0.0;
        static const float r_V_wind = 5.11;
        static const float r_scale_factor = 1.0;

        n.param("wind_and_waves/beta_wave", beta_wave, r_beta_wave);
        n.param("wind_and_waves/stddev_wF", stddev_wF, r_stddev_wF);
        n.param("wind_and_waves/stddev_dF", stddev_dF, r_stddev_dF);
        n.param("wind_and_waves/stddev_wN", stddev_wN, r_stddev_wN);
        n.param("wind_and_waves/stddev_dN", stddev_dN, r_stddev_dN);
        n.param("wind_and_waves/beta_wind", beta_wind, r_beta_wind);
        n.param("wind_and_waves/V_wind", V_wind, r_V_wind);
        n.param("wind_and_waves/scale_factor", scale_factor, r_scale_factor);

        psi = 0.0;
        u = 0.0;
        v = 0.0;

        delta_x = 0.0;
        delta_y = 0.0;
        delta_theta = 0.0;

        w0 = 0.8;
        lambda = 0.1;
        Kw = 0.64;
        g = 9.81;

        mean_wF = 0.0;
        mean_dF = 0.0;
        mean_wN = 0.0;
        mean_dN = 0.0;

        xF1 = 0.0;
        xF2 = 0.0;
        xF1_dot_last = 0.0;
        xF2_dot_last = 0.0;
        xN1 = 0.0;
        xN2 = 0.0;
        xN1_dot_last = 0.0;
        xN2_dot_last = 0.0;
        dF = 0.0;
        dN = 0.0;
        dF_dot_last = 0.0;
        dN_dot_last = 0.0;

        rho = 1.0;
        CDlaf_0 = 0.55;
        CDlaf_pi = 0.6;
        CDt = 0.9;
        AFW = 0.045;
        ALW = 0.09;
        LOA = 0.9;
        delta_wind = 0.6;
        
    }

    void pose_callback(const geometry_msgs::Pose2D::ConstPtr& _pose)
    {
        x = _pose->x; //North disturbance in Newtons
        y = _pose->y; //East disturbance in Newtons
        psi = _pose->theta; //Rotational disturbance in Nm
    }

    void vel_callback(const geometry_msgs::Pose2D::ConstPtr& _vel)
    {
        u = _vel->x; //North disturbance in Newtons
        v = _vel->y; //East disturbance in Newtons
        r = _vel->theta; //Rotational disturbance in Nm
    }

    void time_step()
    {
        std::normal_distribution<float> dist_wF(mean_wF, stddev_wF);
        std::normal_distribution<float> dist_wN(mean_wN, stddev_dF);
        std::normal_distribution<float> dist_dF(mean_dF, stddev_wN);
        std::normal_distribution<float> dist_dN(mean_dN, stddev_dN);

        wF = dist_wF(generator_wF);
        wN = dist_wN(generator_wN);
        dF_dot = dist_dF(generator_dF);
        dN_dot = dist_dN(generator_dN);

        U = std::pow(u*u + v*v, 0.5);
        we = std::abs(w0 - (w0*w0/g)*U*std::cos(beta_wave));

        xF2_dot = -we*we*xF1 - 2*lambda*we*xF2 + Kw*wF;
        xF2 = integral_step * (xF2_dot + xF2_dot_last)/2 + xF2;
        xF2_dot_last = xF2_dot;
        xF1_dot = xF2;
        xF1 = integral_step * (xF1_dot + xF1_dot_last)/2 + xF1;
        xF1_dot_last = xF1_dot;

        dF = integral_step * (dF_dot + dF_dot_last)/2 + dF;
        dF_dot_last = dF_dot;

        F_wave = (xF2 + dF)*scale_factor;

        xN2_dot = -we*we*xN1 - 2*lambda*we*xN2 + Kw*wN;
        xN2 = integral_step * (xN2_dot + xN2_dot_last)/2 + xN2;
        xN2_dot_last = xN2_dot;
        xN1_dot = xN2;
        xN1 = integral_step * (xN1_dot + xN1_dot_last)/2 + xN1;
        xN1_dot_last = xN1_dot;

        dN = integral_step * (dN_dot + dN_dot_last)/2 + dN;
        dN_dot_last = dN_dot;

        N_wave = (xN2 + dN)*scale_factor;

        X_wave = F_wave * std::cos(beta_wave - psi);
        Y_wave = F_wave * std::sin(beta_wave - psi);

        u_w = V_wind * std::cos(beta_wind - psi);
        v_w = V_wind * std::sin(beta_wind - psi);

        u_rw = u - u_w;
        v_rw = v - v_w;

        gamma_rw = std::atan2(v_rw,u_rw);
        V_rw = std::pow(u_rw*u_rw + v_rw*v_rw, 0.5);

        if (std::abs(gamma_rw) > 1.5708){
            CDlaf = CDlaf_pi;
        }
        else{
            CDlaf = CDlaf_0;
        }
        CDl = CDlaf*AFW/ALW;

        CX = CDlaf * std::cos(gamma_rw) / ( 1 - ((delta_wind /2) * (1 - (CDl/CDt)) * (std::sin(2*gamma_rw) * std::sin(2*gamma_rw))) );
        CY = CDt * std::sin(gamma_rw) / ( 1 - ((delta_wind /2) * (1 - (CDl/CDt)) * (std::sin(2*gamma_rw) * std::sin(2*gamma_rw))) );
        CN = -0.18*(gamma_rw - 3.1415/2)*CY;

        X_wind = 0.5*rho*V_rw*V_rw*CX*AFW*scale_factor;
        Y_wind = 0.5*rho*V_rw*V_rw*CY*ALW*scale_factor;
        N_wind = 0.5*rho*V_rw*V_rw*CN*AFW*LOA*scale_factor;

        delta_x = X_wave + X_wind;
        delta_y = Y_wave + Y_wind;
        delta_theta = N_wave + N_wind;;

        disturbance.x = delta_x;
        disturbance.y = delta_y;
        disturbance.theta = delta_theta;

        //Data publishing
        disturbance_pub.publish(disturbance);
    }

private:
    ros::NodeHandle n;

    ros::Publisher disturbance_pub;

    ros::Subscriber pose_sub;
    ros::Subscriber vel_sub;

};

//Main
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "wind_and_waves");
    WindWaves windWaves;
    windWaves.integral_step = 0.01;
    int rate = 100;
    ros::Rate loop_rate(rate);

  while (ros::ok())
  {
    windWaves.time_step();
    ros::spinOnce();
    loop_rate.sleep();
  }

    return 0;
}