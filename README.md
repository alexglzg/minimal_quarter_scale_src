# Simulation Environment for Quarter Scale Roboat ASV

## To run lake environment

roslaunch gazebo_sim lake.launch

- Disable gazebo physics for it to work

- Check for multiple obstacle scenarios inside iros2026_scenarios including lanes, intersections, buoys, etc.

## To run roboat simulation

roslaunch gazebo_sim gazebo_roboat.launch

- Spawns roboat with velodyne lidar and custom dynamics
- Check launch file to include disturbances from wind, waves, and currents

## Multiple perception systems

roslaunch roboat_planning run_boat.launch

- Filters LiDAR data
- Creates occupancy grid maps
- Check for different occupancy grid-based perception system inside obstacle_detector package, which fits ellipses and circles to obstacles
- Used for IROS 2025 publication

roslaunch obstacle_detector pcl_filter.launch
- Launches pcl-based pointcloud filter
- Used for LiDAR-based polygonal corridor construction

roslaunch obstacle_detector map.launch
- Launches occupancy grid map

## Multiple polygonal corridor geometry

Alternative nodes:

rosrun my_decomp_test grid_decomp_node
- Builds free-space polytopes on a grid map

rosrun my_decomp_test simple_decomp_node
- Builds free-space polyopes using pointcloud data

rosrun firi_ros firi_node
- Builds free-space polytopes using poincloud data

These nodes take odometry information to build the seed.


## CBF node

roslaunch embedded_cbf mpc_cbf.launch


## Known Dependencies
- DecompROS
- DecompUtil
- catkin_simple
- Eigen3
- OsqpEigen
- osqp
- NLOPT
