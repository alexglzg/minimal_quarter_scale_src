#include <math.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "gazebo_msgs/ModelState.h"

class TfAndPath{
public:
  gazebo_msgs::ModelState states;
  TfAndPath(){
  gazebo_pub = node.advertise<gazebo_msgs::ModelState>("/gazebo/set_model_state", 1000);
  sub = node.subscribe("odometry/filtered", 1000, &TfAndPath::odomCallback, this);
  std::string model_name;
  std::string param_model_name;
  model_name = "roboat";

  node.param("roboat_gazebo_broadcaster/model", states.model_name, model_name);

  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
    
    states.pose.position.x = msg->pose.pose.position.x;
    states.pose.position.y = msg->pose.pose.position.y;
    states.pose.position.z = 0.0;

    states.pose.orientation.x = msg->pose.pose.orientation.x;
    states.pose.orientation.y = msg->pose.pose.orientation.y;
    states.pose.orientation.z = msg->pose.pose.orientation.z;
    states.pose.orientation.w = msg->pose.pose.orientation.w;
    
    gazebo_pub.publish(states);
  }

private:
  ros::NodeHandle node;
  ros::Publisher gazebo_pub;
  ros::Subscriber sub;

};

int main(int argc, char** argv){

  ros::init(argc, argv, "qs_gazebo_broadcaster");
  TfAndPath tfAndPath;

  while (ros::ok()){
    ros::Rate(100).sleep();
    ros::spinOnce();
  }

  return 0;
}