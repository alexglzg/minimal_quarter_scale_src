#!/bin/bash

# 1. Define the Docker execution prefix
# This is the "First Part" that runs at the start of every terminal
DOCKER_PREFIX="docker exec -it quarterscale bash"

# 2. Define the list of commands
# This is the "Second Part" (one command per line)
COMMANDS=(
    "roslaunch gazebo_sim spawn_intersection.launch"
    "roslaunch gazebo_sim gazebo_roboat.launch"
    "roslaunch obstacle_detector pcl_filter.launch"
    "rosrun my_decomp_test simple_decomp_node"
    "roslaunch embedded_cbf mpc_cbf.launch"
)

# 3. Loop through and launch
for CMD in "${COMMANDS[@]}"; do
    echo "Launching: $CMD"
    
    # We use 'bash -c' to chain the docker command and your specific command.
    # The final '; bash' keeps the terminal window open after the command finishes.
    gnome-terminal -- bash -c "$DOCKER_PREFIX $CMD; bash"
done