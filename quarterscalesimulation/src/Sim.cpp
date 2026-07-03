// ROS
#include "ros/ros.h"
#include <ros/package.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float32.h>
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include "std_msgs/UInt8.h"
#include <tf2/LinearMath/Quaternion.h>
#include <roboat_core/Force.h>

// C++
#include <boost/numeric/odeint.hpp>
#include <boost/ref.hpp>
#include "math.h"
#include <random>

// roboat
#include <Sim.hpp>

using namespace boost::numeric::odeint;

std::default_random_engine generator;
std::normal_distribution<double> distx(0.0, 0.0);
std::normal_distribution<double> disty(0.0, 0.0);
std::normal_distribution<double> distt(0.0, 0.0);

/**
 * Reference frames
 * -----------------
 * Internal simulation state (x[0..9], see state_type in operator() and Sim::state)
 * is expressed in a NED-style ship convention:
 *   x[0] = North position [m]
 *   x[1] = East position [m]
 *   x[2] = psi, heading [rad], positive CLOCKWISE from North (compass heading)
 *   x[3] = u, surge velocity [m/s]            (body x-axis, forward)
 *   x[4] = v, sway velocity [m/s]             (body y-axis, to starboard/right)
 *   x[5] = r, yaw rate = dpsi/dt [rad/s], same clockwise-positive sense as psi
 *   x[6..9] = individual thruster forces [N], held piecewise-constant between
 *             forceCallback() updates
 *
 * All internal computation -- this operator(), the current/wind/wave disturbance
 * inputs (nu_u, nu_v, delta_x, delta_y, delta_theta, all sourced from topics that
 * are themselves computed against this same psi), and the raw pose republished on
 * "qs_dynamics/inertial_pose" -- stays in this NED-style frame.
 *
 * The ROS-facing topics (filtered_pose, filtered_twist, odometry/filtered, and the
 * /initialpose subscriber) use the standard ROS convention instead (right-handed,
 * z-up, yaw CCW-positive about z, REP-103). Converting between the two frames
 * requires mirroring the East axis (y -> -y) AND negating psi (and r) together:
 * flipping one axis of a 2D frame flips its handedness, so the sign of positive
 * rotation must flip too, or headings/yaw-rates would come out mirrored.
 * See initialPoseCallback() (ROS -> internal) and the pose_pub/odom_pub/twist_pub
 * blocks in the constructor (internal -> ROS) for the matching negation pairs.
 *
 * Current/wind/wave direction parameters (beta_c here from /roboat_currents;
 * beta_wave, beta_wind in wind_and_waves.cpp) share the same North/East, psi
 * convention above: each is the compass-style angle (0 = North, clockwise-positive)
 * that the current/wind/waves flow TOWARDS -- the oceanographic sense, not the
 * meteorological "coming from" convention. currents_callback() below rotates that
 * vector into the body frame via (beta_c - state[2]), which is Fossen's ocean-current
 * model; wind_and_waves.cpp applies the identical (beta - psi) pattern for wind
 * (u_w, v_w) and waves (X_wave, Y_wave, the encounter frequency we), so all three
 * disturbance sources stay consistent with each other and with psi. Any beta_*
 * parameter set with the opposite ("coming from") convention will silently apply the
 * disturbance 180 degrees off.
 */
void Sim::operator()(const state_type &x, state_type &dxdt, const double /* t */)
{
  // velocities relative to current, used by the drag terms below
  double u_r = x[3] - u_c;
  double v_r = x[4] - v_c;

  double nu_r_dot[3] = {0, 0, 0};

  // model with Coriolis terms, current/wind/wave disturbances, and linear drag
  nu_r_dot[0] = 1 / m11 * (x[6] + x[7] + m22 * v_r * x[5] - d11 * u_r + delta_x);
  nu_r_dot[1] = 1 / m22 * (x[8] + x[9] - m11 * u_r * x[5] - d22 * v_r + delta_y);
  nu_r_dot[2] = 1 / m33 * (aa / 2 * x[6] - aa / 2 * x[7] + bb / 2 * x[8] - bb / 2 * x[9] + (m11 - m22) * u_r * v_r - d33 * x[5] + delta_theta);

  double u_c_dot = V_c_dot * cos(beta_c - x[2]) + V_c * sin(beta_c - x[2]) * x[5];
  double v_c_dot = V_c_dot * sin(beta_c - x[2]) - V_c * cos(beta_c - x[2]) * x[5];

  dxdt[0] = cos(x[2]) * x[3] - sin(x[2]) * x[4];
  dxdt[1] = sin(x[2]) * x[3] + cos(x[2]) * x[4];
  dxdt[2] = x[5];
  dxdt[3] = nu_r_dot[0] + u_c_dot;
  dxdt[4] = nu_r_dot[1] + v_c_dot;
  dxdt[5] = nu_r_dot[2];
  dxdt[6] = 0; // u1
  dxdt[7] = 0; // u2
  dxdt[8] = 0; // u3
  dxdt[9] = 0; // u4

  //  dxdt[0] = cos(x[2]) * x[3] - sin(x[2]) * x[4];
  // dxdt[1] = sin(x[2]) * x[3] + cos(x[2]) * x[4];
  // dxdt[2] = x[5];
  // dxdt[3] = -d11 / m11 * (x[3] - nu_u) + x[6] / m11 + x[7] / m11 - delta_x / m11;
  // dxdt[4] = -d22 / m22 * (x[4] - nu_v) + x[8] / m22 + x[9] / m22 - delta_y / m22;
  // dxdt[5] = -d33 / m33 * x[5] + aa / (2 * m33) * x[6] - aa / (2 * m33) * x[7] + bb / (2 * m33) * x[8] - bb / (2 * m33) * x[9] - delta_theta / m33;
  // dxdt[6] = 0;
  // dxdt[7] = 0;
  // dxdt[8] = 0;
  // dxdt[9] = 0;
}

void Sim::forceCallback(const roboat_core::Force::ConstPtr &msg)
{
  // last indices of state represent force
  for (int i = 0; i < 4; i++)
    state[i + 6] = msg->data[i];
}

void Sim::dist_callback(const geometry_msgs::Pose2D::ConstPtr &delta)
{
    delta_x = delta->x; //Body-frame surge (x) disturbance force in Newtons
    delta_y = delta->y; //Body-frame sway (y) disturbance force in Newtons
    delta_theta = delta->theta; //Yaw disturbance moment in Nm
}

void Sim::currents_callback(const geometry_msgs::Pose2D::ConstPtr &cur)
{
    V_c = cur->x; //Current magnitude in m/s
    V_c_dot = cur->y; //Effective derivative of current magnitude in m/s^2 (0 while saturated)
    beta_c = cur->theta; //Current direction in rad
    u_c = V_c*cos(beta_c - state[2]); // Current velocity in the body surge direction in m/s
    v_c = V_c*sin(beta_c - state[2]); // Current velocity in the body sway direction in m/s
}

void Sim::initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &msg)
{
  // last indices of state represent force
  double roll, pitch, yaw;
  tf::Matrix3x3 m;

  m = tf::Matrix3x3(tf::Quaternion(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                                   msg->pose.pose.orientation.z, msg->pose.pose.orientation.w));
  m.getRPY(roll, pitch, yaw);

  state[0] = msg->pose.pose.position.x;
  state[1] = -msg->pose.pose.position.y;
  state[2] = -yaw;
  state[3] = 0.0;
  state[4] = 0.0;
  state[5] = 0.0;
  state[6] = 0.0;
  state[7] = 0.0;
  state[8] = 0.0;
  state[9] = 0.0;
}

state_type Sim::integrate(state_type &x, double time)
{
  integrate_const(stepper, boost::ref(*this), x, 0.0, time, simStep);
  return x;
}

Sim::Sim(ros::NodeHandle n)
{
  state = std::vector<double>(10);
  std::fill(state.begin(), state.end(), 0);

  n.param("/x_0", state[0], 0.0);
  n.param("/y_0", state[1], -3.0);
  n.param("/psi", state[2], 0.0);
  n.param("system_dynamics/d11", d11);
  n.param("system_dynamics/d22", d22);
  n.param("system_dynamics/d33", d33);
  n.param("system_dynamics/m11", m11);
  n.param("system_dynamics/m22", m22);
  n.param("system_dynamics/m33", m33);
  n.param("system_dynamics/aa", aa);
  n.param("system_dynamics/bb", bb);
  // n.param("system_dynamics/step", step);
  step = 0.01;
  // publisher for x,y,theta + reference vector
  twist_pub = n.advertise<geometry_msgs::TwistStamped>("filtered_twist", 10);
  pose_pub = n.advertise<geometry_msgs::PoseStamped>("filtered_pose", 10);
  pub_VelocityRviz = n.advertise<std_msgs::Float32>("linear_velocity_viz", 10);
  odom_pub = n.advertise<nav_msgs::Odometry>("odometry/filtered", 10);
  inertial_pose_pub = n.advertise<geometry_msgs::Pose2D>("qs_dynamics/inertial_pose", 10);
  body_vel_pub = n.advertise<geometry_msgs::Pose2D>("qs_dynamics/body_vel", 10);
  
  // force from MPC, other controller, or manual rostopic
  force_sub = n.subscribe("/mpc_force", 10, &Sim::forceCallback, this);

  // disturbances from wind, waves, and currents
  disturbance_sub = n.subscribe("/roboat_disturbance", 1, &Sim::dist_callback, this);
  currents_sub = n.subscribe("/roboat_currents", 1, &Sim::currents_callback, this);

  // initial pose from rviz
  initialpose_sub = n.subscribe("/initialpose", 10, &Sim::initialPoseCallback, this);

  ros::Rate loop_rate(1 / step);

  ros::Time lastTime, currentTime = ros::Time::now();

  while (ros::ok())
  {
    ros::spinOnce();

    lastTime = currentTime;
    currentTime = ros::Time::now();

    // calculate new state as integration of state, over time using system model
    state = integrate(state, (currentTime - lastTime).toSec());

    // publish new twist (velocity)
    geometry_msgs::TwistStamped twist_msg;

    // twist_msg.twist.linear.x = cos(state[2])*state[3]-sin(state[2])*state[4];
    // twist_msg.twist.linear.y = sin(state[2])*state[3]+cos(state[2])*state[4];
    twist_msg.twist.linear.x = state[3];
    twist_msg.twist.linear.y = state[4];
    twist_msg.twist.linear.y *= -1;
    twist_msg.twist.angular.z = -state[5];

    twist_msg.header.stamp = currentTime;
    twist_msg.header.frame_id = "base_link";
    twist_pub.publish(twist_msg);

    // publish new pose
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.pose.position.x = state[0];// + distx(generator);
    pose_msg.pose.position.y = -state[1];// + disty(generator);
    pose_msg.pose.orientation = tf::createQuaternionMsgFromYaw(-state[2]);// + distt(generator));
    pose_msg.header.stamp = currentTime;
    pose_msg.header.frame_id = "map";
    pose_pub.publish(pose_msg);

    static tf::TransformBroadcaster odom_broadcaster;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = ros::Time::now();
    odom_trans.header.frame_id = "map";
    odom_trans.child_frame_id = "base_link";

    odom_trans.transform.translation.x = pose_msg.pose.position.x;
    odom_trans.transform.translation.y = pose_msg.pose.position.y;
    odom_trans.transform.translation.z = pose_msg.pose.position.z;
    odom_trans.transform.rotation = pose_msg.pose.orientation;

    // send the transform
    odom_broadcaster.sendTransform(odom_trans);

    std_msgs::Float32 vel_rviz;
    vel_rviz.data = twist_msg.twist.linear.x;
    pub_VelocityRviz.publish(vel_rviz);

    // publish new odom
    geometry_msgs::Pose2D dynamic_pose; //inertial navigation system pose [North East Yaw] or [x y psi]
    geometry_msgs::Pose2D dynamic_vel; //velocity vector [u v r]
    nav_msgs::Odometry odom;
    dynamic_pose.x = state[0];
    dynamic_pose.y = state[1];
    dynamic_pose.theta = state[2];
    
    odom.header.stamp = ros::Time::now();
    odom.header.frame_id = "map";      
    odom.child_frame_id = "base_link"; 
    odom.pose.pose.position.x = state[0];
    odom.pose.pose.position.y = -state[1];
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = tf::createQuaternionMsgFromYaw(-state[2]);

    dynamic_vel.x = state[3];
    dynamic_vel.y = state[4];
    dynamic_vel.theta = state[5];

    odom.twist.twist.linear.x = state[3];
    odom.twist.twist.linear.y = -state[4];
    odom.twist.twist.linear.z = 0.0;

    odom.twist.twist.angular.x = 0.0;
    odom.twist.twist.angular.y = 0.0;
    odom.twist.twist.angular.z = -state[5];

    //Data publishing
    inertial_pose_pub.publish(dynamic_pose);
    body_vel_pub.publish(dynamic_vel);
    odom_pub.publish(odom);

    ROS_DEBUG("[SIM_NODE] loop runtime: %fs, integration-time: %fs", ros::Time::now().toSec() - currentTime.toSec(),
              (currentTime - lastTime).toSec());

    loop_rate.sleep();
  }
}
