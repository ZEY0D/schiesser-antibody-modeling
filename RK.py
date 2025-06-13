# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Parameters
# D = 1e-10       # Diffusion coefficient (m²/s)
# kf = 1e5        # Forward binding rate (m³/mol/s)
# kr = 1e1        # Reverse binding rate (1/s)
# cbsat = 1.66e-9 # Saturation concentration (mol/m²)
# cbulk = 4.48e-5 # Bulk concentration (mol/m³)
# h = 5e-5        # Domain height (m)
# c0 = 0.0        # Initial concentration (mol/m³)
# cb0 = 0.0       # Initial bound concentration (mol/m²)

# # Spatial grid
# N = 21          # Number of grid points
# z = np.linspace(0, h, N)
# dz = z[1] - z[0]

# # Time grid
# t_start = 0
# t_end = 100     # End time (s)
# dt = 0.01       # Time step (must satisfy CFL: dt ≤ dz²/(2D))
# num_steps = int(t_end / dt) + 1
# t = np.linspace(t_start, t_end, num_steps)

# # Initialize solutions
# c = np.zeros((N, num_steps))  # Spatial concentration
# cb = np.zeros(num_steps)      # Surface concentration
# c[:, 0] = c0                  # Initial condition c(z,0) = c0
# cb[0] = cb0                   # Initial condition cb(0) = cb0

# # RK4 Solver
# for step in range(num_steps - 1):
#     current_c = c[:, step]
#     current_cb = cb[step]

#     # Stage 1: Compute k1
#     dcdt = np.zeros(N)
#     # PDE: Interior points (central difference)
#     for i in range(1, N-1):
#         dcdt[i] = D * (current_c[i+1] - 2*current_c[i] + current_c[i-1]) / dz**2
#     # Robin BC at z=0 (using ghost point)
#     flux = kf * current_c[0] * (cbsat - current_cb) - kr * current_cb
#     c_ghost = current_c[1] - (2*dz/D) * flux
#     dcdt[0] = D * (current_c[1] - 2*current_c[0] + c_ghost) / dz**2
#     # Dirichlet BC at z=h
#     current_c[-1] = cbulk
#     dcdt[-1] = 0
#     # ODE for cb
#     dcbdt = kf * current_c[0] * (cbsat - current_cb) - kr * current_cb
#     k1_c = dcdt
#     k1_cb = dcbdt

#     # Stage 2: Compute k2 (mid-step)
#     mid_c = current_c + 0.5 * dt * k1_c
#     mid_cb = current_cb + 0.5 * dt * k1_cb
#     dcdt = np.zeros(N)
#     for i in range(1, N-1):
#         dcdt[i] = D * (mid_c[i+1] - 2*mid_c[i] + mid_c[i-1]) / dz**2
#     flux = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
#     c_ghost = mid_c[1] - (2*dz/D) * flux
#     dcdt[0] = D * (mid_c[1] - 2*mid_c[0] + c_ghost) / dz**2
#     mid_c[-1] = cbulk
#     dcdt[-1] = 0
#     dcbdt = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
#     k2_c = dcdt
#     k2_cb = dcbdt

#     # Stage 3: Compute k3 (another mid-step)
#     mid_c = current_c + 0.5 * dt * k2_c
#     mid_cb = current_cb + 0.5 * dt * k2_cb
#     dcdt = np.zeros(N)
#     for i in range(1, N-1):
#         dcdt[i] = D * (mid_c[i+1] - 2*mid_c[i] + mid_c[i-1]) / dz**2
#     flux = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
#     c_ghost = mid_c[1] - (2*dz/D) * flux
#     dcdt[0] = D * (mid_c[1] - 2*mid_c[0] + c_ghost) / dz**2
#     mid_c[-1] = cbulk
#     dcdt[-1] = 0
#     dcbdt = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
#     k3_c = dcdt
#     k3_cb = dcbdt

#     # Stage 4: Compute k4 (full-step)
#     next_c = current_c + dt * k3_c
#     next_cb = current_cb + dt * k3_cb
#     dcdt = np.zeros(N)
#     for i in range(1, N-1):
#         dcdt[i] = D * (next_c[i+1] - 2*next_c[i] + next_c[i-1]) / dz**2
#     flux = kf * next_c[0] * (cbsat - next_cb) - kr * next_cb
#     c_ghost = next_c[1] - (2*dz/D) * flux
#     dcdt[0] = D * (next_c[1] - 2*next_c[0] + c_ghost) / dz**2
#     next_c[-1] = cbulk
#     dcdt[-1] = 0
#     dcbdt = kf * next_c[0] * (cbsat - next_cb) - kr * next_cb
#     k4_c = dcdt
#     k4_cb = dcbdt

#     # Update solutions using RK4 weights
#     c[:, step+1] = current_c + (dt / 6) * (k1_c + 2*k2_c + 2*k3_c + k4_c)
#     cb[step+1] = current_cb + (dt / 6) * (k1_cb + 2*k2_cb + 2*k3_cb + k4_cb)

# # Compute fractional coverage
# theta = cb / cbsat

# # Plot results
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(t, c[0, :])
# plt.title('c(0,t) vs t')
# plt.xlabel('Time (s)')
# plt.ylabel('Concentration (mol/m³)')

# plt.subplot(2, 2, 2)
# plt.plot(t, cb)
# plt.title('cb(t) vs t')
# plt.xlabel('Time (s)')
# plt.ylabel('Surface Concentration (mol/m²)')

# plt.subplot(2, 2, 3)
# plt.plot(t, theta)
# plt.title('θ(t) vs t')
# plt.xlabel('Time (s)')
# plt.ylabel('Fractional Coverage')
# # #rate vs time
# plt.subplot(2, 2, 1)
# plt.plot(t, c[0, :])
# plt.title('c(0,t) vs t')
# plt.xlabel('Time (s)')
# plt.ylabel('Concentration (mol/m³)')

# # plt.subplot(2, 2, 4, projection='3d')
# # Z, T = np.meshgrid(z, t)
# # plt.plot_surface(Z, T, c.T, cmap='viridis')
# # plt.title('c(z,t) Profile')
# # plt.xlabel('z (m)')
# # plt.ylabel('Time (s)')
# # plt.tight_layout()
# # # plt.show()
# # Get the 3D axes object and call plot_surface on it
# ax = plt.subplot(2, 2, 4, projection='3d')
# Z, T = np.meshgrid(z, t)
# ax.plot_surface(Z, T, c.T, cmap='viridis') # Call plot_surface on the axes object
# ax.set_title('c(z,t) Profile')
# ax.set_xlabel('z (m)')
# ax.set_ylabel('Time (s)')
# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
D = 1e-10       # Diffusion coefficient (m²/s)
kf = 1e5        # Forward binding rate (m³/mol/s)
kr = 1e1        # Reverse binding rate (1/s)
cbsat = 1.66e-9 # Saturation concentration (mol/m²)
cbulk = 4.48e-5 # Bulk concentration (mol/m³)
h = 5e-5        # Domain height (m)
c0 = 0.0        # Initial concentration (mol/m³)
cb0 = 0.0       # Initial bound concentration (mol/m²)

# Spatial grid
N = 21          # Number of grid points
z = np.linspace(0, h, N)
dz = z[1] - z[0]

# Time grid
t_start = 0
t_end = 100     # End time (s)
dt = 0.01       # Time step (must satisfy CFL: dt ≤ dz²/(2D))
num_steps = int(t_end / dt) + 1
t = np.linspace(t_start, t_end, num_steps)

# Initialize solutions
c = np.zeros((N, num_steps))  # Spatial concentration
cb = np.zeros(num_steps)      # Surface concentration
rate = np.zeros(num_steps)    # Net binding rate
c[:, 0] = c0                  # Initial condition c(z,0) = c0
cb[0] = cb0                   # Initial condition cb(0) = cb0
rate[0] = 0                   # Initial rate is zero

# RK4 Solver
for step in range(num_steps - 1):
    current_c = c[:, step]
    current_cb = cb[step]

    # Stage 1: Compute k1
    dcdt = np.zeros(N)
    # PDE: Interior points (central difference)
    for i in range(1, N-1):
        dcdt[i] = D * (current_c[i+1] - 2*current_c[i] + current_c[i-1]) / dz**2
    # Robin BC at z=0 (using ghost point)
    current_flux = kf * current_c[0] * (cbsat - current_cb) - kr * current_cb
    c_ghost = current_c[1] - (2*dz/D) * current_flux
    dcdt[0] = D * (current_c[1] - 2*current_c[0] + c_ghost) / dz**2
    # Dirichlet BC at z=h
    current_c[-1] = cbulk
    dcdt[-1] = 0
    # ODE for cb
    dcbdt = current_flux  # Same as the flux calculation
    k1_c = dcdt
    k1_cb = dcbdt

    # Stage 2: Compute k2 (mid-step)
    mid_c = current_c + 0.5 * dt * k1_c
    mid_cb = current_cb + 0.5 * dt * k1_cb
    dcdt = np.zeros(N)
    for i in range(1, N-1):
        dcdt[i] = D * (mid_c[i+1] - 2*mid_c[i] + mid_c[i-1]) / dz**2
    mid_flux = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
    c_ghost = mid_c[1] - (2*dz/D) * mid_flux
    dcdt[0] = D * (mid_c[1] - 2*mid_c[0] + c_ghost) / dz**2
    mid_c[-1] = cbulk
    dcdt[-1] = 0
    dcbdt = mid_flux
    k2_c = dcdt
    k2_cb = dcbdt

    # Stage 3: Compute k3 (another mid-step)
    mid_c = current_c + 0.5 * dt * k2_c
    mid_cb = current_cb + 0.5 * dt * k2_cb
    dcdt = np.zeros(N)
    for i in range(1, N-1):
        dcdt[i] = D * (mid_c[i+1] - 2*mid_c[i] + mid_c[i-1]) / dz**2
    mid_flux = kf * mid_c[0] * (cbsat - mid_cb) - kr * mid_cb
    c_ghost = mid_c[1] - (2*dz/D) * mid_flux
    dcdt[0] = D * (mid_c[1] - 2*mid_c[0] + c_ghost) / dz**2
    mid_c[-1] = cbulk
    dcdt[-1] = 0
    dcbdt = mid_flux
    k3_c = dcdt
    k3_cb = dcbdt

    # Stage 4: Compute k4 (full-step)
    next_c = current_c + dt * k3_c
    next_cb = current_cb + dt * k3_cb
    dcdt = np.zeros(N)
    for i in range(1, N-1):
        dcdt[i] = D * (next_c[i+1] - 2*next_c[i] + next_c[i-1]) / dz**2
    next_flux = kf * next_c[0] * (cbsat - next_cb) - kr * next_cb
    c_ghost = next_c[1] - (2*dz/D) * next_flux
    dcdt[0] = D * (next_c[1] - 2*next_c[0] + c_ghost) / dz**2
    next_c[-1] = cbulk
    dcdt[-1] = 0
    dcbdt = next_flux
    k4_c = dcdt
    k4_cb = dcbdt

    # Update solutions using RK4 weights
    c[:, step+1] = current_c + (dt / 6) * (k1_c + 2*k2_c + 2*k3_c + k4_c)
    cb[step+1] = current_cb + (dt / 6) * (k1_cb + 2*k2_cb + 2*k3_cb + k4_cb)
    
    # Calculate and store the current net binding rate
    rate[step+1] = kf * c[0, step+1] * (cbsat - cb[step+1]) - kr * cb[step+1]

# Compute fractional coverage
theta = cb / cbsat

# Plot results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(t, c[0, :])
plt.title('Surface Concentration c(0,t) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/m³)')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(t, cb)
plt.title('Bound Concentration cb(t) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Surface Concentration (mol/m²)')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(t, theta)
plt.title('Fractional Coverage θ(t) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Fractional Coverage')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(t, rate)
plt.title('Net Binding Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Rate (mol/m²/s)')
plt.grid(True)

ax = plt.subplot(2, 3, (5,6), projection='3d')
Z, T = np.meshgrid(z, t)
ax.plot_surface(Z, T, c.T, cmap='viridis')
ax.set_title('Concentration Profile c(z,t)')
ax.set_xlabel('z (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Concentration (mol/m³)')

plt.tight_layout()
plt.show()