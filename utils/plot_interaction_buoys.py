import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------

sigma = 3.0
sigma_r = 2.0
k = 0.5
kr = 0.8

vessel_speed = 1.5
psi = 0.0
heading = np.array([np.cos(psi), np.sin(psi)])

vessel_pos = np.array([0.0, 0.0])

# buoy example
buoy_pos = np.array([2.0, 1.5])
buoy_vel = np.array([0.0, 0.0])
kd = 3.0

# -----------------------------
# Grid for flow field
# -----------------------------

x = np.linspace(-6, 6, 25)
y = np.linspace(-6, 6, 25)
X, Y = np.meshgrid(x, y)

U = np.zeros_like(X)
V = np.zeros_like(Y)

# compute velocity field
for i in range(X.shape[0]):
    for j in range(X.shape[1]):

        point = np.array([X[i, j], Y[i, j]])
        dx = point - vessel_pos
        d = np.linalg.norm(dx) + 1e-6

        # forward wake flow
        u_forward = (
            k
            * vessel_speed
            * np.exp(-(d**2)/(2*sigma**2))
            * heading
        )

        # sideways push
        u_side = (
            kr
            * np.exp(-(d**2)/(2*sigma_r**2))
            * dx / d
        )

        u = u_forward + u_side

        U[i, j] = u[0]
        V[i, j] = u[1]

# -----------------------------
# Flow at buoy
# -----------------------------

dx = buoy_pos - vessel_pos
d = np.linalg.norm(dx)

u_forward = (
    k
    * vessel_speed
    * np.exp(-(d**2)/(2*sigma**2))
    * heading
)

u_side = (
    kr
    * np.exp(-(d**2)/(2*sigma_r**2))
    * dx / d
)

u = u_forward + u_side

F = kd * (u - buoy_vel)

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(8,8))

# vector field
plt.quiver(X, Y, U, V)

# vessel
plt.scatter(*vessel_pos, s=200, label="vessel")

# buoy
plt.scatter(*buoy_pos, s=200, label="buoy")

# force on buoy
plt.arrow(
    buoy_pos[0],
    buoy_pos[1],
    F[0],
    F[1],
    head_width=0.2,
    color="red"
)

plt.gca().set_aspect('equal')
plt.xlim(-6,6)
plt.ylim(-6,6)

plt.title("Vessel-induced flow field")
plt.legend()
plt.grid()

plt.show()