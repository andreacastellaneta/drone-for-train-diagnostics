import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import control as ctrl
from quadcopter import Quadcopter6DOF

"""
In this experiment, we exploit the performances of the DJI Phantom 4 Pro Quadcopter
https://www.dji.com/it/phantom-4-pro/info
"""

# Physical parameters
g = 9.81            # Gravity acceleration [m/s^2]
m_train = 60000     # Train mass [kg]
m_drone = 1.38      # Quadcopter mass [kg]
m_motor = 0.06      # Quadcopter motor mass [kg]
l_arm = 0.175       # Quadcopter length [m]

"""

The following 3D scenario is examined: 

The drone has already left the train and has just finished the data acquisition. The train has a constant velocity of 
around 40 km/h and it is distant 900 m from the drone. The drone has a very tiny velocity along x and y axis. The top of
the train is at an altitude of 3 m, while the drone performs measures at 5 m along the positive z axis.

"""

# Initial conditions
x0_t = np.array([500, 0, 3, 40 / 3.6, 0, 0])                     # Train init condition
x0_d = np.array([[0], [750], [6], [0], [0], [0],
                 [5 / 3.6], [0.1 / 3.6], [0], [0], [0], [0]])    # Drone init condition

# State Space 6DOF model of the Drone
dt = 0.01   # Integration step
drone = Quadcopter6DOF(m_drone, m_motor, l_arm, dt)

# Equilibrium point
x_bar = np.array([0, 750, 5, 0, 0, 0, 5/3.6, 2/3.6, 0, 0, 0, 0])
u_bar = np.array([drone.m * g, 0, 0, 0])

# Controllability and Observability of Quadcopter State Space system
A, B = drone.linearize(x_bar, u_bar)
rank_C = np.linalg.matrix_rank(ctrl.ctrb(A, B))         # Fully controllable system
rank_O = np.linalg.matrix_rank(ctrl.obsv(A, drone.C))   # Fully observable system

"""

The Leader-Follower Consensus Algorithm is performed in order to make the drone catch the train and synchronize with it.
From the analysis of experimental data, the quadcopter is given 20 min to stabilize with consensus
The following specs of the quadcopter are taken into consideration:
-   Drone Autonomy =~ 30 min
-   Drone max velocity = 72 km/h

"""

# Parameters of the Simulation
t_consensus = np.arange(0, 1201, 1)    # Time of the simulation [s]
xt = np.zeros((6, 1201))
xt[:, 0] = x0_t                        # Train state vector [m]
xd = x0_d                              # Quadcopter state vector [m]


# Train Trajectory
# Sinusoidal Part
sep = 900
xt[0, 0:sep] = 17 * t_consensus[0:sep] + x0_t[0]
xt[1, 0:sep] = 500 * np.sin(2 * np.pi / sep * t_consensus[0:sep]) + x0_t[1]
xt[3, 0:sep] = 17
xt[4, 0:sep] = 500 * 2 * np.pi / sep * np.cos(2 * np.pi / sep * t_consensus[0:sep])
# Rectilinear Part
xt[0, sep:len(t_consensus)] = xt[0, sep-1] + xt[3, sep-1] * t_consensus[0:len(t_consensus) - sep]
xt[1, sep:len(t_consensus)] = xt[1, sep-1] + xt[4, sep-1] * t_consensus[0:len(t_consensus) - sep]
xt[3, sep:len(t_consensus)] = np.full((1, (len(t_consensus) - sep)), xt[3, sep-1])
xt[4, sep:len(t_consensus)] = np.full((1, (len(t_consensus) - sep)), xt[4, sep-1])

xt[2, :] = np.full((1, len(t_consensus)), x0_t[2])

# Directed Graph of the Leader-Follower Problem
"""
Leader:     Train
Follower:   Drone
"""
plt.figure()
G = nx.DiGraph()
G.add_edge("Drone", "Train")
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=300)
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
nx.draw_networkx_labels(G, pos)

# Parameters for Leader-Follower Consensus Algorithm
k = 1.2
Kr = np.diag([1, 1, 1, 1, 1, 1])
Kv = np.diag([1.25, 1.3, 1, 1, 1, 1])

for i in range(0, len(t_consensus) - 1):
    """
    Apply Bounded Consensus Leader-Follower Algorithm to axis x, y and z (the drone is brought to 1 m ahead of the train
    in terms of altitude)
    (cfr. page 117 of Wei Ren PhD, Randal W. Beard PhD - Distributed Consensus in Multi-vehicle Cooperative Control
    Theory and Applications-Springer-Verlag London (2008))
    """
    term1 = np.tanh(np.array([xd[0, i] - xt[0, i] - 153 + 9.741939995372377,
                              xd[1, i] - xt[1, i] - 31.415160951928783 + 9.997470413188239,
                              xd[2, i] - xt[2, i] - 1,
                              xd[3, i], xd[4, i], xd[5, i]]))
    term2 = np.tanh(np.array([xd[6, i] - xt[3, i], xd[7, i] - xt[4, i], xd[8, i] - xt[5, i],
                              xd[9, i], xd[10, i], xd[11, i]]))
    term1 = np.reshape(term1, (6, 1))
    term2 = np.reshape(term2, (6, 1))

    # Calculate Control Input
    a_drone = - (1 / k) * np.dot(Kr, term1) - (1 / k) * np.dot(Kv, term2)    # Accelerations
    a = np.block([[np.zeros((6, 1))], [a_drone]])

    # Apply Control Input
    xd_prev = np.array(xd[:, i]).reshape((12, 1))
    v = np.array(xd_prev[[6, 7, 8, 9, 10, 11]])
    vd_new = v + a_drone
    x = np.array(xd_prev[[0, 1, 2, 3, 4, 5]])
    xd_new = x + vd_new
    xd_new = np.concatenate((xd_new, vd_new), axis=0).reshape((12, 1))

    # Update positions
    xd = np.append(xd, xd_new, axis=1)

# print(xt[:, -1])    # Train state after consensus
# print(xd[:, -1])    # Drone state after consensus
xt_init = xt[0, -1]
yt_init = xt[1, -1]

"""

Before the landing, the drone arrives ahead of the train. In this case, the drone must intercept the train on the last 
wagon. The LQR algorithm with 2 Riccati equations is performed in order to make the drone land on the train. The 
quadcopter is given 10 s.

"""

# Initial condition
z0 = [xd[0, -1], xd[1, -1], xd[2, -1], 0, 0, 0, 0, 0, 0, 0, 0, 0]
u0 = np.array([drone.m * g, 0, 0, 0])

# Final condition
x_disp = 10
zbar = np.array([xd[0, -1] + x_disp, xd[1, -1] + x_disp, xt[2, -1], 0, 0, 0, 0, 0, 0, 0, 0, 0])
ubar = np.array([drone.m * g, 0, 0, 0])

# Parameters for the LQR Algorithm
R = np.identity(4) * 2
Q = np.identity(12) * 100
Q[6, 6] = 500
horizon_length = 1000

P = Q.copy()
K_mat_1 = np.empty([12, horizon_length - 1])
K_mat_2 = np.empty([12, horizon_length - 1])
K_mat_3 = np.empty([12, horizon_length - 1])
K_mat_4 = np.empty([12, horizon_length - 1])

# Discretization and linearization of quadcopter matrices
A_discr, B_discr = drone.linearize_discr(zbar, ubar)

# Riccati equations resolution
for s in range(0, horizon_length - 1):
    K = -np.dot((np.dot((np.dot((np.linalg.inv(np.dot((np.dot(B_discr.T, P)), B_discr) + R)), B_discr.T)), P)), A_discr)
    P = Q + np.dot((np.dot(A_discr.T, P)), A_discr) + np.dot((np.dot((np.dot(A_discr.T, P)), B_discr)), K)
    K_12 = K.T.copy()
    K_mat_1[:, horizon_length - s - 2] = (K_12[:, 0]).copy()
    K_mat_2[:, horizon_length - s - 2] = (K_12[:, 1]).copy()
    K_mat_3[:, horizon_length - s - 2] = (K_12[:, 2]).copy()
    K_mat_4[:, horizon_length - s - 2] = (K_12[:, 3]).copy()

A_mat = np.empty([12, horizon_length])
z1 = np.empty([12, horizon_length])
z1[:, 0] = z0
u_opt = np.zeros([4, horizon_length])
u_opt[:, 0] = u0.copy()

# Apply LQR
for i in range(0, horizon_length - 1):
    A_mat[:, i] = drone.model_integrator(z1[:, i], u_opt[:, i])
    z1[:, i + 1] = A_mat[:, i]
    K_mat_tot = np.concatenate((K_mat_1[:, i], K_mat_2[:, i], K_mat_3[:, i], K_mat_4[:, i]), axis=0)
    K_mat_tot = K_mat_tot.reshape((4, 12))
    u_opt[:, i + 1] = np.dot(K_mat_tot, (z1[:, i + 1] - zbar)) + ubar  # Control law

# Update train state
xt_new = np.zeros((6, 10))
xt_new[0, :] = xt[0, len(t_consensus) - 1] + xt[3, len(t_consensus) - 1] * t_consensus[0:len(xt_new[0, :])]
xt_new[1, :] = xt[1, len(t_consensus) - 1] + xt[4, len(t_consensus) - 1] * t_consensus[0:len(xt_new[0, :])]
xt_new[2, :] = np.full((1, len(xt_new[2, :])), x0_t[2])
xt_new[3, :] = np.full((1, len(xt_new[3, :])), xt[3, len(t_consensus) - 1])
xt_new[4, :] = np.full((1, len(xt_new[4, :])), xt[4, len(t_consensus) - 1])
xt = np.append(xt, xt_new, axis=1)

x1 = np.empty([12, 10])
for j in range(0, 10):
    x1[:, j] = z1[:, 100 * j]
bar = np.zeros((3, len(z1[0, :])))
bar[0, :] = np.arange(xd[0, -1], xd[0, -1] + x_disp, x_disp/horizon_length)
bar[1, :] = np.arange(xd[1, -1], xd[1, -1] + x_disp, x_disp/horizon_length)
bar[2, :] = np.arange(4, 3, -1/horizon_length)

# Complete state vector of the drone
xd = np.concatenate((xd, x1), axis=1)

print(xt[0, -1] - xt_init)      # Train displacement from consensus to end (along x)
print(xt[1, -1] - yt_init)      # Train displacement from consensus to end (along y)
print(xd[[0, 1, 2], -1])        # Drone position after LQR
print(xt[[0, 1, 2], -1])        # Train last state
print(xd[[6, 7, 8], -1])        # Drone velocity after LQR
print(xd[0, -1] - xt[0, -1])    # Distance of drone from train after LQR (along x)
print(xd[1, -1] - xt[1, -1])    # Distance of drone from train after LQR (along y)
print(xd[2, -1] - xt[2, -1])    # Distance of drone from train after LQR (along z)

# Total time
t = np.arange(0, len(xd[0, :]), 1)

# Velocity Graph
fig6 = plt.figure()
plt.plot(t, np.sqrt(np.power(xt[3, :], 2) + np.power(xt[4, :], 2) + np.power(xt[5, :], 2)) * 3.6, 'b', label='Train')
plt.plot(t, np.sqrt(np.power(xd[6, :], 2) + np.power(xd[7, :], 2) + np.power(xd[8, :], 2)) * 3.6, 'r', label='Drone')
plt.title('Velocity Graph')
plt.grid()
plt.legend()

# State Variables Plot
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t, xt[0, :], color='g', label='TRAIN')
axs[0, 0].plot(t_consensus, xd[0, 0:t_consensus[-1]+1], color='r', label='DRONE CONSENSUS')
axs[0, 0].plot(t[t_consensus[-1]+1:-1], xd[0, t_consensus[-1]+1:t[-1]], color='b', label='DRONE LQR')
axs[0, 0].set(xlabel='t [s]', ylabel='x [m]')
axs[0, 0].grid()
axs[0, 0].legend(loc="upper left")
axs[1, 0].plot(t, xt[1, :], color='g')
axs[1, 0].plot(t_consensus, xd[1, 0:t_consensus[-1]+1], color='r')
axs[1, 0].plot(t[t_consensus[-1]+1:-1], xd[1, t_consensus[-1]+1:t[-1]], color='b')
axs[1, 0].set(xlabel='t [s]', ylabel='y [m]')
axs[1, 0].grid()
axs[2, 0].plot(t, xt[2, :], color='g')
axs[2, 0].plot(t_consensus, xd[2, 0:t_consensus[-1]+1], color='r')
axs[2, 0].plot(t[t_consensus[-1]+1:-1], xd[2, t_consensus[-1]+1:t[-1]], color='b')
axs[2, 0].set(xlabel='t [s]', ylabel='z [m]')
axs[2, 0].grid()
axs[0, 1].plot(t, xt[3, :], color='g')
axs[0, 1].plot(t_consensus, xd[6, 0:t_consensus[-1]+1], color='r')
axs[0, 1].plot(t[t_consensus[-1]+1:-1], xd[6, t_consensus[-1]+1:t[-1]], color='b')
axs[0, 1].set(xlabel='t [s]', ylabel='vx [m/s]')
axs[0, 1].grid()
axs[1, 1].plot(t, xt[4, :], color='g')
axs[1, 1].plot(t_consensus, xd[7, 0:t_consensus[-1]+1], color='r')
axs[1, 1].plot(t[t_consensus[-1]+1:-1], xd[7, t_consensus[-1]+1:t[-1]], color='b')
axs[1, 1].set(xlabel='t [s]', ylabel='vy [m/s]')
axs[1, 1].grid()
axs[2, 1].plot(t, xt[5, :], color='g')
axs[2, 1].plot(t_consensus, xd[8, 0:t_consensus[-1]+1], color='r')
axs[2, 1].plot(t[t_consensus[-1]+1:-1], xd[8, t_consensus[-1]+1:t[-1]], color='b')
axs[2, 1].set(xlabel='t [s]', ylabel='vz [m/s]')
axs[2, 1].grid()
fig.tight_layout()

# 3D Trajectory Plot with Colormap based on Velocity
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
# Calculate Velocity Magnitude [km/h]
vt = np.sqrt(np.power(xt[3, :], 2) + np.power(xt[4, :], 2) + np.power(xt[5, :], 2)) * 3.6   # Train velocity
vd = np.sqrt(np.power(xd[6, :], 2) + np.power(xd[7, :], 2) + np.power(xd[8, :], 2)) * 3.6   # Drone velocity
max_vel = max(np.max(vt), np.max(vd))
min_vel = min(np.min(vt), np.min(vd))
# Train Plot
ct = mpl.cm.turbo(vt / max_vel)
ax1.scatter(xt[0, 0], xt[1, 0], xt[2, 0], color=ct[0], marker='D')
ax1.scatter(xt[0, -1], xt[1, -1], xt[2, -1], color=ct[-1], marker='D')
for i in t[0:-3]:
    ax1.plot([xt[0, i], xt[0, i+1]], [xt[1, i], xt[1, i+1]], [xt[2, i], xt[2, i+1]], color=ct[i])
ax1.plot([xt[0, -2], xt[0, -1]], [xt[1, -2], xt[1, -1]],
         [xt[2, -2], xt[2, -1]], color=ct[t[-2]])
ax1.text(16500, - 300, 3, s='Train')
# Drone Consensus Plot
cd = mpl.cm.turbo(vd / max_vel)
ax1.plot(xd[0, 0], xd[1, 0], xd[2, 0], color=cd[0], marker='+', mew=2, ms=9)
ax1.plot(xd[0, t_consensus[-1]], xd[1, t_consensus[-1]], xd[2, t_consensus[-1]], color=cd[len(t_consensus)],
         marker='+', mew=2, ms=9)
for i in t[0:t_consensus[-3]]:
    ax1.plot([xd[0, i], xd[0, i+1]], [xd[1, i], xd[1, i+1]], [xd[2, i], xd[2, i+1]], color=cd[i])
ax1.plot([xd[0, t_consensus[-2]], xd[0, t_consensus[-1]]], [xd[1, t_consensus[-2]], xd[1, t_consensus[-1]]],
         [xd[2, t_consensus[-2]], xd[2, t_consensus[-1]]], color=cd[t[-2]])
# Drone Landing Plot
ax1.scatter(xd[0, t_consensus[-1]+1], xd[1, t_consensus[-1]+1], xd[2, t_consensus[-1]+1], color=cd[t_consensus[-1]+1],
            marker='o')
ax1.scatter(xd[0, -1], xd[1, -1], xd[2, -1], color=cd[-1], marker='o')
for i in t[t_consensus[-1]+1:-3]:
    ax1.plot([xd[0, i], xd[0, i+1]], [xd[1, i], xd[1, i+1]], [xd[2, i], xd[2, i+1]], '--', color=cd[i])
ax1.plot([xd[0, t[-2]], xd[0, t[-1]]], [xd[1, t[-2]], xd[1, t[-1]]],
         [xd[2, t[-2]], xd[2, t[-1]]], '--', color=ct[t[-2]])
ax1.text(500, 1200, 3.9, s='Drone')
cmap_v = mpl.cm.turbo
norm_v = mpl.colors.Normalize(vmin=min_vel, vmax=max_vel)
fig1.colorbar(mpl.cm.ScalarMappable(norm=norm_v, cmap=cmap_v), label='Velocity [km/h]')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_zlabel('z [m]')
plt.grid()
plt.title('3D Trajectory colormap-based on Velocity')

# 3D Trajectory Plot with Time Reference
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection="3d")
# Train Plot
ax2.plot(xt[0, 0], xt[1, 0], xt[2, 0], color='y', marker='|',
         mew=3, ms=9)
ax2.plot(xt[0, -1], xt[1, -1], xt[2, -1], color='black', marker='|',
         mew=3, ms=9)
ax2.plot(xt[0, 850], xt[1, 850], xt[2, 850],
         color='c', marker='|', mew=3, ms=9)
ax2.plot(xt[0, t_consensus[-1]], xt[1, t_consensus[-1]], xt[2, t_consensus[-1]],
         color='m', marker='|', mew=3, ms=9)
ax2.plot(xt[0, :], xt[1, :], xt[2, :], '-.', color='g', label='TRAIN')
# Drone Consensus Plot
ax2.plot(xd[0, 0], xd[1, 0], xd[2, 0], color='y', marker='|', label='0 s',
         mew=3, ms=9)
ax2.plot(xd[0, 850], xd[1, 850], xd[2, 850],
         color='c', marker='|', mew=3, ms=9, label='850 s')
ax2.plot(xd[0, 0:t_consensus[-1]], xd[1, 0:t_consensus[-1]], xd[2, 0:t_consensus[-1]],
         color='r', label='DRONE CONSENSUS')
ax2.plot(xd[0, t_consensus[-1]], xd[1, t_consensus[-1]], xd[2, t_consensus[-1]],
         color='m', marker='|', label='1200 s',
         mew=3, ms=9)
# Drone Landing Plot
ax2.plot(xd[0, t_consensus[-1]+1], xd[1, t_consensus[-1]+1], xd[2, t_consensus[-1]+1], color='m', marker='|',
         mew=3, ms=9)
ax2.plot(xd[0, t_consensus[-1]+1:t[-1]], xd[1, t_consensus[-1]+1:t[-1]], xd[2, t_consensus[-1]+1:t[-1]], '--',
         color='blue', label='DRONE LQR')
ax2.plot(xd[0, -1], xd[1, -1], xd[2, -1], color='black', marker='|', label='1210 s',
         mew=3, ms=9)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.set_zlabel('z [m]')
plt.grid()
plt.title('3D Trajectory with Time Reference')
ax2.legend(loc="upper right")

# 3D Trajectory Plot with colormap based on Time
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
# Train Plot
ct = mpl.cm.winter(t / t[-1])
ax3.scatter(xt[0, 0], xt[1, 0], xt[2, 0], color=ct[0], marker='D')
ax3.scatter(xt[0, -1], xt[1, -1], xt[2, -1], color=ct[-1], marker='D')
for i in t[0:-3]:
    ax3.plot([xt[0, i], xt[0, i+1]], [xt[1, i], xt[1, i+1]], [xt[2, i], xt[2, i+1]], color=ct[i])
ax3.plot([xt[0, -2], xt[0, -1]], [xt[1, -2], xt[1, -1]],
         [xt[2, -2], xt[2, -1]], color=ct[t[-2]])
ax3.text(16500, - 300, 3, s='Train', fontweight='bold')
# Drone Consensus Plot
cd = mpl.cm.winter(t / t[-1])
ax3.plot(xd[0, 0], xd[1, 0], xd[2, 0], color=cd[0], marker='+', mew=2, ms=9)
ax3.plot(xd[0, t_consensus[-1]], xd[1, t_consensus[-1]], xd[2, t_consensus[-1]], color=cd[len(t_consensus)],
         marker='+', mew=2, ms=9)
for i in t[0:t_consensus[-3]]:
    ax3.plot([xd[0, i], xd[0, i+1]], [xd[1, i], xd[1, i+1]], [xd[2, i], xd[2, i+1]], color=cd[i])
ax3.plot([xd[0, t_consensus[-2]], xd[0, t_consensus[-1]]], [xd[1, t_consensus[-2]], xd[1, t_consensus[-1]]],
         [xd[2, t_consensus[-2]], xd[2, t_consensus[-1]]], color=cd[t[-2]])
# Drone Landing Plot
ax3.scatter(xd[0, t_consensus[-1]+1], xd[1, t_consensus[-1]+1], xd[2, t_consensus[-1]+1], color=cd[t_consensus[-1]+1],
            marker='o')
ax3.scatter(xd[0, -1], xd[1, -1], xd[2, -1], color=cd[-1], marker='o')
for i in t[t_consensus[-1]+1:-3]:
    ax3.plot([xd[0, i], xd[0, i+1]], [xd[1, i], xd[1, i+1]], [xd[2, i], xd[2, i+1]], '--', color=cd[i])
ax3.plot([xd[0, t[-2]], xd[0, t[-1]]], [xd[1, t[-2]], xd[1, t[-1]]],
         [xd[2, t[-2]], xd[2, t[-1]]], '--', color=ct[t[-2]])
ax3.text(500, 1200, 3.9, s='Drone', fontweight='bold')
cmap_t = mpl.cm.winter
norm_t = mpl.colors.Normalize(vmin=t[0], vmax=t[-1])
fig3.colorbar(mpl.cm.ScalarMappable(norm=norm_t, cmap=cmap_t), label='Time [s]')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_zlabel('z [m]')
plt.grid()
plt.title('3D Trajectory Colormap based on Time')

# 2D Trajectory Graph
fig3 = plt.figure()
plt.plot(xt[0, :], xt[1, :], 'g', label='TRAIN')
plt.plot(xd[0, :], xd[1, :], 'r', label='DRONE')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Trajectory along x and y')
plt.legend()

plt.show()
