#include "ros/ros.h"

#include <Sim.hpp>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "sim_node");
  ros::NodeHandle n;
  Sim node(n);
  return 0;
}
