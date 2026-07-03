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
#include <vector>
#include <string>
#include <boost/bind.hpp>
#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include <math.h>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

// Simplified wind-drag + wave-drift force model for a small symmetric (round) buoy.
// Unlike the roboat model, forces stay in the NED frame directly (no vessel heading to
// project against), and there is no oscillatory wave filter or yaw moment.
struct BuoyForceModel
{
    std::string name;
    float radius = 0.0;
    float A_proj = 0.0;  // projected wind area, pi*radius^2

    float dF = 0.0;       // wave drift state (Fossen d1/d2-style random-walk integrator)
    float dF_dot_last = 0.0;
    std::default_random_engine generator_dF;

    float vx_ned = 0.0;  // buoy's own velocity in NED frame, from its odometry
    float vy_ned = 0.0;

    ros::Publisher disturbance_pub;
    ros::Subscriber odom_sub;
};

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
    float scale_factor_wind;
    float scale_factor_wave;

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
    float dF_min;
    float dF_max;
    float dN;
    float dN_dot_last;
    float dN_min;
    float dN_max;

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
    float V_wind_knots;
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

    // Simplified per-buoy model, shared params + one BuoyForceModel per configured buoy
    float buoy_stddev_dF;
    float buoy_dF_min;
    float buoy_dF_max;
    float buoy_CD;
    std::vector<BuoyForceModel> buoys;

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
        static const float r_V_wind_knots = 9.93;  // ~5.11 m/s, previous default
        static const float r_scale_factor_wind = 1.0;
        static const float r_scale_factor_wave = 1.0;
        static const float r_dF_min = -5.0;
        static const float r_dF_max = 5.0;
        static const float r_dN_min = -5.0;
        static const float r_dN_max = 5.0;
        static const float r_buoy_stddev_dF = 0.05;
        static const float r_buoy_dF_min = -2.0;
        static const float r_buoy_dF_max = 2.0;
        static const float r_buoy_CD = 0.5;

        n.param("wind_and_waves/beta_wave", beta_wave, r_beta_wave);
        n.param("wind_and_waves/stddev_wF", stddev_wF, r_stddev_wF);
        n.param("wind_and_waves/stddev_dF", stddev_dF, r_stddev_dF);
        n.param("wind_and_waves/stddev_wN", stddev_wN, r_stddev_wN);
        n.param("wind_and_waves/stddev_dN", stddev_dN, r_stddev_dN);
        n.param("wind_and_waves/beta_wind", beta_wind, r_beta_wind);
        n.param("wind_and_waves/V_wind_knots", V_wind_knots, r_V_wind_knots);
        n.param("wind_and_waves/scale_factor_wind", scale_factor_wind, r_scale_factor_wind);
        n.param("wind_and_waves/scale_factor_wave", scale_factor_wave, r_scale_factor_wave);
        n.param("wind_and_waves/dF_min", dF_min, r_dF_min);
        n.param("wind_and_waves/dF_max", dF_max, r_dF_max);
        n.param("wind_and_waves/dN_min", dN_min, r_dN_min);
        n.param("wind_and_waves/dN_max", dN_max, r_dN_max);
        n.param("wind_and_waves/buoy_stddev_dF", buoy_stddev_dF, r_buoy_stddev_dF);
        n.param("wind_and_waves/buoy_dF_min", buoy_dF_min, r_buoy_dF_min);
        n.param("wind_and_waves/buoy_dF_max", buoy_dF_max, r_buoy_dF_max);
        n.param("wind_and_waves/buoy_CD", buoy_CD, r_buoy_CD);

        // Pierson-Moskowitz modal frequency derived from a 10 m reference wind speed:
        // convert to the 19.4 m reference height, then to the modal frequency w0.
        float V10_ms = V_wind_knots * 0.514444f;  // knots -> m/s
        float V194_ms = V10_ms * std::pow(19.4f / 10.0f, 1.0f / 7.0f);
        float B = 0.74f * std::pow(9.81f / V194_ms, 0.25f);
        w0 = std::pow((4.0f * B) / 5.0f, 0.25f);
        V_wind = V10_ms;

        generator_wF.seed(std::random_device{}());  // seed the random number generator with a random device, to avoid that the same numbers are generated for all quantities
        generator_dF.seed(std::random_device{}());
        generator_wN.seed(std::random_device{}());
        generator_dN.seed(std::random_device{}());

        // Simplified force publishers/subscribers for any buoys configured on the shared
        // "/buoys" param (same list buoy_simulator.py reads), so buoy geometry stays defined
        // in one place. Missing/absent param just means no buoys are simulated.
        XmlRpc::XmlRpcValue buoys_param;
        if (n.getParam("/buoys", buoys_param) && buoys_param.getType() == XmlRpc::XmlRpcValue::TypeArray)
        {
            for (int i = 0; i < buoys_param.size(); ++i)
            {
                XmlRpc::XmlRpcValue& b = buoys_param[i];
                BuoyForceModel buoy;
                buoy.name = static_cast<std::string>(b["name"]);
                buoy.radius = static_cast<double>(b["radius"]);
                buoy.A_proj = M_PI * buoy.radius * buoy.radius;
                buoy.generator_dF.seed(std::random_device{}());

                buoy.disturbance_pub = n.advertise<geometry_msgs::Pose2D>("/" + buoy.name + "/disturbance", 1);
                buoy.odom_sub = n.subscribe<nav_msgs::Odometry>("/" + buoy.name + "/odometry", 1,
                    boost::bind(&WindWaves::buoyOdomCallback, this, boost::placeholders::_1, (int)buoys.size()));

                buoys.push_back(buoy);
            }
            ROS_INFO("[wind_and_waves] loaded %zu buoy(s) for simplified wind/wave forcing", buoys.size());
        }

        psi = 0.0;
        u = 0.0;
        v = 0.0;

        delta_x = 0.0;
        delta_y = 0.0;
        delta_theta = 0.0;

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
        x = _pose->x; //Vessel North position in meters (unused elsewhere in this file)
        y = _pose->y; //Vessel East position in meters (unused elsewhere in this file)
        psi = _pose->theta; //Vessel heading in rad, NED-style; projects wave/wind direction into the body frame
    }

    void vel_callback(const geometry_msgs::Pose2D::ConstPtr& _vel)
    {
        u = _vel->x; //Vessel surge velocity in m/s; used for encounter frequency and relative wind speed
        v = _vel->y; //Vessel sway velocity in m/s; used for encounter frequency and relative wind speed
        r = _vel->theta; //Vessel yaw rate in rad/s (unused elsewhere in this file)
    }

    void buoyOdomCallback(const nav_msgs::Odometry::ConstPtr& msg, int idx)
    {
        // buoy_simulator.py negates y before publishing (ROS-facing convention); undo it
        // here to recover the buoy's NED-frame velocity, matching beta_wind/beta_wave.
        buoys[idx].vx_ned = msg->twist.twist.linear.x;
        buoys[idx].vy_ned = -msg->twist.twist.linear.y;
    }

    void time_step()
    {
        /******* Wave Roboat *******/
        std::normal_distribution<float> dist_wF(mean_wF, stddev_wF);
        std::normal_distribution<float> dist_wN(mean_wN, stddev_wN);
        std::normal_distribution<float> dist_dF(mean_dF, stddev_dF);
        std::normal_distribution<float> dist_dN(mean_dN, stddev_dN);

        // wF, wN drive the OSCILLATORY filters (xF, xN) -- fast, zero-mean noise input
        wF = dist_wF(generator_wF);
        wN = dist_wN(generator_wN);
        // dF_dot, dN_dot drive the DRIFT integrators (dF, dN) -- slow, unbounded random walk
        dF_dot = dist_dF(generator_dF);
        dN_dot = dist_dN(generator_dN);

        U = std::pow(u*u + v*v, 0.5);
        we = std::abs(w0 - (w0*w0/g)*U*std::cos(beta_wave - psi)); // Encounter frequency of the waves, based on the vessel's velocity and heading relative to the wave direction

        /* oscillatory surge/sway wave force: damped resonant filter, eq. (8.112)  */
        xF2_dot = -we*we*xF1 - 2*lambda*we*xF2 + Kw*wF;
        xF2 = integral_step * (xF2_dot + xF2_dot_last)/2 + xF2;
        xF2_dot_last = xF2_dot;
        xF1_dot = xF2;
        xF1 = integral_step * (xF1_dot + xF1_dot_last)/2 + xF1;
        xF1_dot_last = xF1_dot;

        /* drift surge/sway force: pure noise integrator, Fossen's d1/d2 (8.133-8.135) 
        nothing here pulls dF back to zero -- this is the term the book says needs
        "saturating elements to prevent di from exceeding a predescribed maximum physical limit"*/
        dF = integral_step * (dF_dot + dF_dot_last)/2 + dF;
        dF = std::min(std::max(dF, dF_min), dF_max);
        dF_dot_last = dF_dot;

        // final wave force = oscillatory component + drift component, summed and scaled
        F_wave = (xF2 + dF)*scale_factor_wave;


        /* oscillatory yaw moment */
        xN2_dot = -we*we*xN1 - 2*lambda*we*xN2 + Kw*wN;
        xN2 = integral_step * (xN2_dot + xN2_dot_last)/2 + xN2;
        xN2_dot_last = xN2_dot;
        xN1_dot = xN2;
        xN1 = integral_step * (xN1_dot + xN1_dot_last)/2 + xN1;
        xN1_dot_last = xN1_dot;
        /* drift yaw moment */
        dN = integral_step * (dN_dot + dN_dot_last)/2 + dN;
        dN = std::min(std::max(dN, dN_min), dN_max);
        dN_dot_last = dN_dot;
        /* final wave moment = oscillatory component + drift component, summed and scaled */
        N_wave = (xN2 + dN)*scale_factor_wave;
        /* final wave force components in the NED frame, projected from the wave direction */
        X_wave = F_wave * std::cos(beta_wave - psi);
        Y_wave = F_wave * std::sin(beta_wave - psi);

        /******* Wind Roboat *******/
        u_w = V_wind * std::cos(beta_wind - psi);
        v_w = V_wind * std::sin(beta_wind - psi);

        u_rw = u - u_w;
        v_rw = v - v_w;

        gamma_rw = -std::atan2(v_rw,u_rw);
        V_rw = std::pow(u_rw*u_rw + v_rw*v_rw, 0.5);

        // choosing one drag value depending if relative wind is hitting the front or the back of the boat (Fossen Section 8.1.3)
        if (std::abs(gamma_rw) > 1.5708){
            CDlaf = CDlaf_pi;
        }
        else{
            CDlaf = CDlaf_0;
        }
        CDl = CDlaf*AFW/ALW;

        CX = -CDlaf * std::cos(gamma_rw) / ( 1 - ((delta_wind /2) * (1 - (CDl/CDt)) * (std::sin(2*gamma_rw) * std::sin(2*gamma_rw))) );
        CY = CDt * std::sin(gamma_rw) / ( 1 - ((delta_wind /2) * (1 - (CDl/CDt)) * (std::sin(2*gamma_rw) * std::sin(2*gamma_rw))) );
        CN = -0.18*(gamma_rw - 3.1415/2)*CY;

        X_wind = 0.5*rho*V_rw*V_rw*CX*AFW*scale_factor_wind;
        Y_wind = 0.5*rho*V_rw*V_rw*CY*ALW*scale_factor_wind;
        N_wind = 0.5*rho*V_rw*V_rw*CN*ALW*LOA*scale_factor_wind;

        delta_x = X_wave + X_wind;
        delta_y = Y_wave + Y_wind;
        delta_theta = N_wave + N_wind;;

        disturbance.x = delta_x;
        disturbance.y = delta_y;
        disturbance.theta = delta_theta;

        //Data publishing
        disturbance_pub.publish(disturbance);

        /****** Buoys ******/
        // Simplified per-buoy wind-drag + wave-drift forcing, published in the NED frame
        // directly (buoys are symmetric/round, so there is no heading to project against).
        for (auto& buoy : buoys)
        {
            std::normal_distribution<float> dist_buoy_dF(0.0, buoy_stddev_dF);
            float buoy_dF_dot = dist_buoy_dF(buoy.generator_dF);
            buoy.dF = integral_step * (buoy_dF_dot + buoy.dF_dot_last)/2 + buoy.dF;
            buoy.dF = std::min(std::max(buoy.dF, buoy_dF_min), buoy_dF_max);
            buoy.dF_dot_last = buoy_dF_dot;

            float FX_wave = buoy.dF * std::cos(beta_wave) * scale_factor_wave;
            float FY_wave = buoy.dF * std::sin(beta_wave) * scale_factor_wave;

            float u_rw_buoy = V_wind * std::cos(beta_wind) - buoy.vx_ned;
            float v_rw_buoy = V_wind * std::sin(beta_wind) - buoy.vy_ned;
            float V_rw_buoy = std::pow(u_rw_buoy*u_rw_buoy + v_rw_buoy*v_rw_buoy, 0.5);

            float FX_wind = 0.0, FY_wind = 0.0;
            if (V_rw_buoy > 1e-6)
            {
                float q = 0.5*rho*V_rw_buoy*V_rw_buoy*buoy_CD*buoy.A_proj*scale_factor_wind;
                FX_wind = q * u_rw_buoy / V_rw_buoy;
                FY_wind = q * v_rw_buoy / V_rw_buoy;
            }

            geometry_msgs::Pose2D buoy_disturbance;
            buoy_disturbance.x = FX_wave + FX_wind; //NED-frame North disturbance force in Newtons
            buoy_disturbance.y = FY_wave + FY_wind; //NED-frame East disturbance force in Newtons
            buoy_disturbance.theta = 0.0; //Symmetric buoy: no yaw moment
            buoy.disturbance_pub.publish(buoy_disturbance);
        }
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