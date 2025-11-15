# Simulation Environment for Quarter Scale Roboat ASV

## To run lake environment

roslaunch gazebo_sim lake.launch

- TODO: check why lake model doesn't show up in some installations

- Check for multiple obstacle scenarios inside iros2026_scenarios including lanes, intersections, buoys, etc.

## To run roboat simulation

roslaunch gazebo_sim gazebo_roboat.launch

- Spawns roboat with velodyne lidar and custom dynamics
- Check launch file to include disturbances from wind, waves, and currents

## Multiple perception systems

roslaunch roboat_planning run_boat

- Filters LiDAR data
- Creates occupancy grid maps

- Check for different occupancy grid-based perception system inside obstacle_detector package, which fits ellipses and circles to obstacles
