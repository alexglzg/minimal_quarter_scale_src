import rosbag
import numpy as np
# import matplotlib
# matplotlib.use("Agg")   # non-GUI backend

import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
from scipy.optimize import minimize

bag = rosbag.Bag("../../rosbag/mpc_test.bag")

import numpy as np

class Path:
    def x(self, s_var):
        raise NotImplementedError

    def y(self, s_var):
        raise NotImplementedError

    def x_dot(self, s_var):
        raise NotImplementedError
    
    def y_dot(self, s_var):
        raise NotImplementedError
    
    def distance_cost(self, s_var, xpos, ypos):
        return (xpos - self.x(s_var))**2 + (ypos - self.y(s_var))**2

class SinePath(Path):
    def __init__(self, x_multiplier=0.2, y_offset=3.0):
        self.k = x_multiplier
        self.y_offset = y_offset

    def x(self, s_var):
        return self.k * s_var

    def y(self, s_var):
        return (np.sin(self.k * s_var) + self.y_offset)

    def x_dot(self, s_var):
        return self.k
    
    def y_dot(self, s_var):
        return self.k * np.cos(self.k * s_var)

    def distance_cost(self, s_var, xpos, ypos):
        return super().distance_cost(s_var, xpos, ypos)

class StraightLinePath(Path):
    def __init__(self, slope=0.0, intercept=0.0):
        self.slope = slope
        self.intercept = intercept

    def x(self, s_var):
        return s_var

    def y(self, s_var):
        return self.slope * s_var + self.intercept

    def x_dot(self, s_var):
        return 1.0
    
    def y_dot(self, s_var):
        return self.slope

    def distance_cost(self, s_var, xpos, ypos):
        return super().distance_cost(s_var, xpos, ypos)


# Define the path
path = SinePath(x_multiplier=0.2, y_offset=0.0)

def odom_to_state(msg):
        
        """
        Handle odometry messages.
        Transform from ROS/ENU to MPC/NED frame and compute path parameter s.
        """
        # Position: x stays, y inverts
        x_ned = msg.pose.pose.position.x
        y_ned = -msg.pose.pose.position.y  # ENU->NED

        # Orientation: yaw inverts
        q = msg.pose.pose.orientation
        _, _, yaw_enu = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_ned = -yaw_enu  # ENU->NED

        # Body velocities: u stays, v inverts, r inverts
        u = msg.twist.twist.linear.x   # surge (forward)
        v = -msg.twist.twist.linear.y  # sway: left->right
        r = -msg.twist.twist.angular.z # yaw rate inverts

        s = minimize(path.distance_cost, 
                        0, method='Nelder-Mead', 
                        args=(x_ned, y_ned),
                        options={'xatol': 1e-8, 'disp': False}).x[0]

        return np.array([
            x_ned,
            y_ned,
            yaw_ned,
            u,
            v,
            r,
            s
        ]) 

# Storage
t_odom = []
states = []

t_force = []
forces = []

# see which topics are in the bag
print("Topics in bag:", bag.get_type_and_topic_info().topics.keys())

# Loop through bag
for topic, msg, t in bag.read_messages(topics=['odometry/filtered', '/mpc_force']):

    if topic == 'odometry/filtered':
        t_odom.append(t.to_sec())
        states.append(odom_to_state(msg))
       

    elif topic == '/mpc_force':
        t_force.append(t.to_sec())
        forces.append(msg.data)  

bag.close()

forces = np.array(forces)
states = np.array(states)

print("States shape:", states.shape)
print("Forces shape:", forces.shape)

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(t_odom, states[:, 0], label='x (NED)')
plt.plot(t_odom, states[:, 1], label='y (NED)')
plt.plot(t_odom, states[:, 2], label='yaw (NED)')
plt.plot(t_odom, states[:, 3], label='u (surge)')
plt.plot(t_odom, states[:, 4], label='v (sway)')
plt.plot(t_odom, states[:, 5], label='r (yaw rate)')
plt.title('Vehicle State Over Time')
plt.xlabel('Time (s)')
plt.ylabel('State')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t_force, forces[:, 0], label='Force 1')
plt.plot(t_force, forces[:, 1], label='Force 2')
plt.plot(t_force, forces[:, 2], label='Force 3')
plt.plot(t_force, forces[:, 3], label='Force 4')
plt.title('Control Forces Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.legend()
plt.tight_layout()
# plt.savefig("mpc_debug_plots_states_controls.png")
plt.show()

#%%
x_d = path.x(states[:, -1])  
y_d = path.y(states[:, -1]) 
x_dot = path.x_dot(states[:, -1])
y_dot = path.y_dot(states[:, -1])    
gamma_p = np.arctan2(y_dot, x_dot)
ye = -(states[:, 0]-x_d)*np.sin(gamma_p)+(states[:, 1]-y_d)*np.cos(gamma_p)

# Plotting cross-track error, desired path, and actual path, gamma_p
plt.figure()
plt.plot(t_odom, ye, label='Cross-track error (ye)')
plt.title('Cross-track Error Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Cross-track Error (m)')

plt.figure()
plt.plot(x_d, y_d, label='Desired Path')
plt.plot(states[:, 0], states[:, 1], label='Actual Path')
plt.title('Desired vs Actual Path')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t_odom, gamma_p, label='Path Tangent Angle (gamma_p)')
plt.title('Path Tangent Angle Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Gamma_p (radians)')
plt.legend()

plt.figure()
plt.plot(t_odom, states[:, -1], '-', label='Path Parameter (s)')
plt.title('Path Parameter Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Path Parameter (s)')
plt.grid()

plt.figure()
plt.plot(t_odom, np.sin(states[:,2])-np.sin(gamma_p), label='sin(yaw) - sin(gamma_p)')
plt.plot(t_odom, np.cos(states[:,2])-np.cos(gamma_p), label='cos(yaw) - cos(gamma_p)')
plt.title('Orientation Error Components Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Orientation Error Components')
plt.legend()
plt.tight_layout()
plt.show()

# plot control forces
plt.figure()
plt.plot(t_force, forces[:, 0], label='Force 1')
plt.plot(t_force, forces[:, 1], label='Force 2')
plt.plot(t_force, forces[:, 2], label='Force 3')
plt.plot(t_force, forces[:, 3], label='Force 4')
plt.title('Control Forces Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Force')
plt.legend()
plt.tight_layout()
plt.show()

# %%
