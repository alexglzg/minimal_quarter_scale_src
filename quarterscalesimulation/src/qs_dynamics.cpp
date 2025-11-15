/** ----------------------------------------------------------------------------
 * @file:     qs_dynamics.cpp
 * @date:     Aug 17, 2022
 * @datemod:  Aug 17, 2022
 * @author:   Alejandro Gonzalez-Garcia
 * @email:    alexglzg97@gmail.com
 * 
 * @brief: Dynamics for quarter scale. 
 * ---------------------------------------------------------------------------*/

#include <iostream>
#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include "std_msgs/UInt8.h"
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
//#include <quarterscalesimulation/Force.h>
#include <roboat_core/Force.h>

using namespace Eigen;

class DynamicModel
{
public:
    float integral_step;

    //Disturbances variables, currently not used
    float delta_x;
    float delta_y;
    float delta_theta;
    float V_c;
    float beta_c;
    Vector3f Delta;
    Vector3f nu_c;
    Vector3f nu_r;

    //Identified parameters
    float a;
    float b;
    float m;
    float Iz;
    float Xu_dot;
    float Yv_dot;
    float Yr_dot;
    float Nr_dot;
    float Xu;
    float Xuu;
    float Yv;
    float Yvv;
    float Nr;
    float Nrr;

    tf2::Quaternion myQuaternion;

    Vector3f nu;
    Vector3f nu_dot_last;
    Vector3f nu_dot;
    Vector3f eta;
    Vector3f eta_dot_last;
    Vector3f eta_dot;
    Matrix3f M;
    Matrix3f CRB;
	Matrix3f CA;
    Matrix3f C;
    Matrix3f Dl;
	Matrix3f Dn;
    Matrix3f D;
    VectorXf force;
    Vector3f tau;
    Matrix3f R;
    MatrixXf B;

    float x;
    float y;
    float psi;
    float u;
    float v;
    float r;

    geometry_msgs::Pose2D dynamic_pose; //inertial navigation system pose [North East Yaw] or [x y psi]
    geometry_msgs::Pose2D dynamic_vel; //velocity vector [u v r]
    nav_msgs::Odometry odom;

    DynamicModel()
    {
        //ROS Publishers for each required simulated ins_2d data
        inertial_pose_pub = n.advertise<geometry_msgs::Pose2D>("qs_dynamics/inertial_pose", 1);
        body_vel_pub = n.advertise<geometry_msgs::Pose2D>("qs_dynamics/body_vel", 1);
        boat_odom_pub = n.advertise<nav_msgs::Odometry>("odometry/filtered", 1);

        force_sub = n.subscribe("mpc_force", 1, &DynamicModel::force_callback, this);
        disturbance_sub = n.subscribe("/roboat_disturbance", 1000, &DynamicModel::dist_callback, this);
        currents_sub = n.subscribe("/roboat_currents", 1000, &DynamicModel::currents_callback, this);

        static const float starting_pose = 0.0;

        n.param("qs_dynamics/x", x, starting_pose);
        n.param("qs_dynamics/y", y, starting_pose);
        n.param("qs_dynamics/psi", psi, starting_pose);

        nu << 0.0, 0.0, 0.0;
        nu_dot_last << 0.0, 0.0, 0.0;
        eta << x, y, psi;
        eta_dot_last << 0.0, 0.0, 0.0;
        force = VectorXf::Zero(4);
        B = MatrixXf::Zero(3,4);
        nu_c << 0.0, 0.0, 0.0;
        nu_r << 0.0, 0.0, 0.0;
        V_c = 0.0;
        beta_c = 0.0;
        delta_x = 0.0;
        delta_y = 0.0;
        delta_theta = 0.0;

        //parameters
        a = 0.45;
        b = 0.9;
        m = 20;
        Iz = 1.5;
        Xu_dot = -1.5;
        Yv_dot = -1.108;
        Yr_dot = -0.4926;
        Nr_dot = -0.4;
        Xu = -5.214;
        Xuu = -10.91;
        Yv = -1.888;
        Yvv = -29.34;
        Nr = -0.197*0.0;
        Nrr = -1.5;

        //constant matrix M
        M << m - Xu_dot, 0.0, 0.0,
            0.0, m - Yv_dot, 0.0,
            0.0, 0.0, Iz - Nr_dot;
        //initial rotation matrix R
        R << cos(eta(2)), -sin(eta(2)), 0.0,
            sin(eta(2)), cos(eta(2)), 0.0,
            0.0, 0.0, 1;
        //allocation matrix B
        B << 1, 1, 0, 0,
            0, 0, 1, 1,
            a/2, -a/2, b/2, -b/2;
    }

    //void force_callback(const quarterscalesimulation::Force::ConstPtr& _force)
    void force_callback(const roboat_core::Force::ConstPtr& _force)
    {
        force(0) = _force->data[0];
        force(1) = _force->data[1];
        force(2) = _force->data[2];
        force(3) = _force->data[3];
    }

    void dist_callback(const geometry_msgs::Pose2D::ConstPtr& delta)
    {
        delta_x = delta->x; //North disturbance in Newtons
        delta_y = delta->y; //East disturbance in Newtons
        delta_theta = delta->theta; //Rotational disturbance in Nm
    }

    void currents_callback(const geometry_msgs::Pose2D::ConstPtr& cur)
    {
        V_c = cur->x; //Current magnitude in m/s
        beta_c = cur->theta; //Current direction in rad
    }

    void time_step()
    {
        //Vector of currents
        nu_c << V_c*cos(beta_c - eta(2)), V_c*sin(beta_c - eta(2)), 0.00;
        nu_r = nu - nu_c;
        //Vector of inertial disturbances
        Delta << delta_x, delta_y, delta_theta;
        //Vector of body disturbances
        Delta = R.inverse()*Delta;
        Nr = -0.197*sqrt(pow(nu_r(0),2)+pow(nu_r(1),2));

        //Coriolis matrix
        CRB << 0.00, 0.00, -m*nu(1),
            0.00, 0.00, m*nu(0),
            m*nu(1), -m*nu(0), 0.00;

        CA << 0.00, 0.00, (Yv_dot*nu_r(1)) + (Yr_dot*nu_r(2)),
            0.00, 0.00, -Xu_dot*nu_r(0),
            -(Yv_dot*nu_r(1)) - (Yr_dot*nu_r(2)), Xu_dot*nu_r(0), 0.00;

        //Drag matrix - linear
        Dl << -Xu, 0, 0,
            0, -Yv, 0,
            0, 0, -Nr;

        //Drag matrix - nonlinear
        Dn << Xuu*abs(nu_r(0)), 0.00, 0.00,
            0.00, Yvv*abs(nu_r(1)), 0.00,
            0.00, 0.00, Nrr*abs(nu_r(2));

        //Drag matrix
        D = Dl - Dn;

        tau = B*force;

        nu_dot =  M.inverse()*(tau - (CRB * nu) - (CA * nu_r) - (D * nu_r) + Delta); //acceleration vector [u' v' r']
        nu = integral_step * (nu_dot + nu_dot_last)/2 + nu; //integral [u v r]
        nu_dot_last = nu_dot;

        //Transformation matrix
        R << cos(eta(2)), -sin(eta(2)), 0.0,
            sin(eta(2)), cos(eta(2)), 0.0,
            0.0, 0.0, 1;

        eta_dot = R*nu; //transformation into inertial frame [x' y' psi']
        eta = integral_step*(eta_dot+eta_dot_last)/2 + eta; //integral [x y psi]
        eta_dot_last = eta_dot;

        x = eta(0); //position in x
        y = eta(1); //position in y
        psi = eta(2); //orientation psi
        //Wrap to [-pi pi]
        if (std::abs(psi) > 3.141592){
            psi = (psi/std::abs(psi))*(std::abs(psi)-2*3.141592);
            eta(2) = psi;
        }
        dynamic_pose.x = x;
        dynamic_pose.y = y;
        dynamic_pose.theta = psi;
        odom.pose.pose.position.x = x;
        odom.pose.pose.position.y = -y;
        odom.pose.pose.position.z = 0;

        myQuaternion.setRPY(0.0,0.0,-psi);

        odom.pose.pose.orientation.x = myQuaternion[0];
        odom.pose.pose.orientation.y = myQuaternion[1];
        odom.pose.pose.orientation.z = myQuaternion[2];
        odom.pose.pose.orientation.w = myQuaternion[3];

        u = nu(0); //surge velocity
        v = nu(1); //sway velocity
        r = nu(2); //yaw rate
        dynamic_vel.x = u;
        dynamic_vel.y = v;
        dynamic_vel.theta = r;
        odom.twist.twist.linear.x = u;
        odom.twist.twist.linear.y = -v;
        odom.twist.twist.linear.z = 0.0;

        odom.twist.twist.angular.x = 0.0;
        odom.twist.twist.angular.y = 0.0;
        odom.twist.twist.angular.z = -r;

        //Data publishing
        inertial_pose_pub.publish(dynamic_pose);
        body_vel_pub.publish(dynamic_vel);
        boat_odom_pub.publish(odom);

    }

private:
    ros::NodeHandle n;

    ros::Publisher inertial_pose_pub;
    ros::Publisher body_vel_pub;
    ros::Publisher boat_odom_pub;

    ros::Subscriber force_sub;
    ros::Subscriber disturbance_sub;
    ros::Subscriber currents_sub;

};

//Main
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "qs_dynamics");
    DynamicModel dynamicModel;
    dynamicModel.integral_step = 0.02;
    int rate = 50;
    ros::Rate loop_rate(rate);

  while (ros::ok())
  {
    dynamicModel.time_step();
    ros::spinOnce();
    loop_rate.sleep();
  }

    return 0;
}
