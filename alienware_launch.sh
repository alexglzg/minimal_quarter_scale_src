#!/bin/bash

# 1. Define the Docker execution prefix
# This is the "First Part" that runs at the start of every terminal
MY_IP=localhost
MASTER_IP=localhost

DOCKER_PREFIX="docker exec -it \
        --env ROS_HOSTNAME=$MY_IP \
        --env ROS_MASTER_URI=http://$MASTER_IP:11311 \
        quarterscale"

# 2. Define the list of commands
SOURCE_CMD="source /opt/ros/noetic/setup.bash && source /ros1_ws/devel/setup.bash"

#2b. ROS IPs
# ROS_CMD="export ROS_IP=192.168.0.107; export ROS_HOSTNAME=192.168.0.107; export ROS_MASTER_URI=http://192.168.0.107:11311"

# 3. Define the list of commands
COMMANDS=(
    "roslaunch gazebo_sim spawn_intersection.launch"
    "roslaunch gazebo_sim gazebo_roboat.launch"
    "roslaunch obstacle_detector pcl_filter.launch"
    # "rosrun my_decomp_test simple_decomp_node"
)

# 4. Loop through and launch
for CMD in "${COMMANDS[@]}"; do
    echo "Launching: $CMD"
    
    # We chain the sourcing and the command together inside the container
    gnome-terminal --tab -- bash -c "$DOCKER_PREFIX bash -c '$SOURCE_CMD && $CMD'; exec bash"
done
