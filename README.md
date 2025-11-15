# MPCs-for-QuarterScale-Roboat
BioPd, MPC, Adaptive MPC and Robust MPC

roslaunch roboat_launch loam_master.launch

roslaunch roboat_loam online.launch

# MPC

roslaunch roboat_core mpc_dist_sim.launch

# MPC_CBF

roslaunch roboat_core mpc_cbf_dist_sim.launch

# Sliding Mode for Quarter Scale Roboat
## For experiments:

roslaunch roboat_launch loam_master.launch

roslaunch roboat_loam online.launch

roslaunch roboat_core antsm.launch

rosbags are automatically created

## For simulations:

roslaunch quarterscalesimulation dist_sim_node.launch

roslaunch roboat_core antsm.launch


## Order of experiments:

### Experiment 1: Trajectory tracking without payload

After running loam_master and slam, place the robot approximately in [0,-3], facing forward

run roslaunch roboat_core antsm.launch

### Experiment 2: Trajectory tracking with 10-pound payload 

After running loam_master and slam, place the robot approximately in [0,-3], facing forward

Place 10 pound payload on top

run roslaunch roboat_core antsm_10p.launch

### Experiment 3: Trajectory tracking with 20-pound payload 

After running loam_master and slam, place the robot approximately in [0,-3], facing forward

Place 20 pound payload on top

run roslaunch roboat_core antsm_20p.launch

### Tuning

If the performance is too aggressive or oscilatory:
Reduce k_x, k_y, k_psi. Note: k_x and k_y should have the same value.
Increase mu_x, mu_y, mu_psi. Note: mu_x and mu_y should have the same value.

k relates to how fast the gain adapts 
mu is the expected accuracy of the performance