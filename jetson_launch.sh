#!/bin/bash

# 1. Define the Docker execution prefix
# This is the "First Part" that runs at the start of every terminal
DOCKER_PREFIX="docker exec -it quarterscale"

# 2. Define the list of commands
SOURCE_CMD="source /opt/ros/noetic/setup.bash && source /ros1_ws/devel/setup.bash"

# 2b. Set ROS IPs
ROS_CMD="export ROS_IP=192.168.0.101; export ROS_HOSTNAME=192.168.0.101; export ROS_MASTER_URI=http://192.168.0.107:11311"

# 3. Define the list of commands
COMMANDS=(
    "rosrun firi_ros firi_node"
    "roslaunch embedded_cbf mpc_cbf.launch"
)

# 4. Loop through and launch
for CMD in "${COMMANDS[@]}"; do
    echo "Launching: $CMD"
    
    # We chain the sourcing and the command together inside the container
    gnome-terminal --tab -- bash -c "$DOCKER_PREFIX bash -c '$SOURCE_CMD && $ROS_CMD && $CMD'; exec bash"	
done
