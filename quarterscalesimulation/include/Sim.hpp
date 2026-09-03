#ifndef SIM_HPP
#define SIM_HPP

// ROS
#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Pose2D.h"
#include "geometry_msgs/Vector3.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Float64.h"
#include "std_msgs/UInt8.h"
#include <tf2/LinearMath/Quaternion.h>
#include <roboat_core/Force.h>

// C++
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

typedef std::vector<double> state_type;

class Sim
{
private:
  std::vector<double> state;
  double simStep = 0.0001;

  /** default system dynamics parameters **/
  double d11 = 6;    // drag coff in the x direction
  double d22 = 8;    // drag coff in y direction
  double d33 = 0.6;  // drag torque coff
  double m11 = 12;   // mass plus added mass in the x direction
  double m22 = 24;   // mass plus added mass in the y direction
  double m33 = 3.0;  // moment of inertia plus added mass around the z axis
  double aa = 0.45;
  double bb = 0.9;
  double step = 0.1;

  double delta_x = 0.0;
  double delta_y = 0.0;
  double delta_theta = 0.0;
  double V_c = 0.0;
  double V_c_dot = 0.0;
  double beta_c = 0.0;
  double u_c = 0.0;
  double v_c = 0.0;

  // Held true only once a real /mpc_force command has been received. Some
  // scenarios seed state[3] (surge velocity, see /su_0) non-zero to match a
  // controller's training initial condition (e.g. anmpc_alpha/model.eqx was
  // trained from su=0.3); without this gate the sim would integrate that
  // residual momentum against zero thrust and drift forward on its own
  // before any controller starts commanding. integrate() is skipped (state
  // held exactly at its seeded values) until this flips true.
  bool force_received = false;

  ros::Publisher twist_pub;
  ros::Publisher pose_pub;
  ros::Publisher pub_VelocityRviz;
  ros::Publisher odom_pub;
  ros::Publisher inertial_pose_pub;
  ros::Publisher body_vel_pub;
  ros::Subscriber force_sub;
  ros::Subscriber initialpose_sub;
  ros::Subscriber disturbance_sub;
  ros::Subscriber currents_sub;

  runge_kutta4<state_type> stepper;
  state_type integrate(state_type& x, double time);
  void forceCallback(const roboat_core::Force::ConstPtr& msg);
  void initialPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
  void dist_callback(const geometry_msgs::Pose2D::ConstPtr& delta);
  void currents_callback(const geometry_msgs::Pose2D::ConstPtr& cur);

public:
  Sim(ros::NodeHandle n);
  void operator()(const state_type& x, state_type& dxdt, const double /* t */);
};

#endif
