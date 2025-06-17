import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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

true_data = {
    't': [0, 2, 4, 6, 8, 10, 90, 92, 94, 96, 98, 100],
    'c(0,t)': [0.000e+000, 3.735e-007, 3.050e-006, 7.035e-006, 1.121e-005, 1.516e-005, 
               4.470e-005, 4.472e-005, 4.473e-005, 4.474e-005, 4.474e-005, 4.475e-005],
    'cb(t)': [0.000e+000, 5.121e-012, 4.645e-011, 1.063e-010, 1.648e-010, 2.165e-010, 
              5.127e-010, 5.129e-010, 5.130e-010, 5.131e-010, 5.131e-010, 5.132e-010],
    'theta': [0.000e+000, 3.085e-003, 2.798e-002, 6.403e-002, 9.927e-002, 1.304e-001, 
              3.089e-001, 3.089e-001, 3.090e-001, 3.091e-001, 3.091e-001, 3.092e-001],
    'rate': [0.000e+000, 1.060e-011, 2.772e-011, 3.014e-011, 2.757e-011, 2.301e-011, 
             1.430e-012, 1.262e-012, 8.640e-013, 4.094e-013, 2.026e-013, 3.637e-014]
}
# Start measuring execution time
start_time = time.time()
# RK4 Solver
for step in range(num_steps - 1):
    current_c = c[:, step]
    current_cb = cb[step]

    # Stage 1: Compute k1
    dcdt = np.zeros(N)
    # PDE: Interior points (Method of lines)
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
# Stop measuring execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.4f} seconds")

 # Extract computed values at the same time points
time_indices = [np.argmin(np.abs(t - time)) for time in true_data['t']]
computed_data = {
    't': true_data['t'],
    'c(0,t)': [c[0, idx] for idx in time_indices],
    'cb(t)': [cb[idx] for idx in time_indices],
    'theta': [theta[idx] for idx in time_indices],
    'rate': [rate[idx] for idx in time_indices]
}

# Calculate error (%)
error_percentage = {
    't': true_data['t'],
    'c(0,t)': [abs((true - comp) / true) * 100 if true != 0 else 0 
               for true, comp in zip(true_data['c(0,t)'], computed_data['c(0,t)'])],
    'cb(t)': [abs((true - comp) / true) * 100 if true != 0 else 0 
              for true, comp in zip(true_data['cb(t)'], computed_data['cb(t)'])],
    'theta': [abs((true - comp) / true) * 100 if true != 0 else 0 
              for true, comp in zip(true_data['theta'], computed_data['theta'])],
    'rate': [abs((true - comp) / true) * 100 if true != 0 else 0 
             for true, comp in zip(true_data['rate'], computed_data['rate'])]
}
# Print the error table
print("\nTrue Error Percentage (%):")
print("{:<5} {:<12} {:<12} {:<12} {:<12}".format("t", "c(0,t)", "cb(t)", "theta", "rate"))
for i in range(len(true_data['t'])):
    print("{:<5} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e}".format(
        error_percentage['t'][i], 
        error_percentage['c(0,t)'][i], 
        error_percentage['cb(t)'][i], 
        error_percentage['theta'][i], 
        error_percentage['rate'][i]
    ))
# Find the maximum error across all variables and time points
max_error = {
    'c(0,t)': max(error_percentage['c(0,t)']),
    'cb(t)': max(error_percentage['cb(t)']),
    'theta': max(error_percentage['theta']),
    'rate': max(error_percentage['rate'])
}

overall_max_error = max(max_error.values())

# Print results
print("Maximum Error Percentage (%):")
print("{:<10} {:<10}".format("Variable", "Max Error"))
for var, err in max_error.items():
    print("{:<10} {:<10.3e}".format(var, err))
print("\nOverall Maximum Error: {:.3e}%".format(overall_max_error))    
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

plt.subplot(2, 3, 4)
plt.plot(t, theta)
plt.title('Fractional Coverage θ(t) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Fractional Coverage')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(t, rate)
plt.title('Net Binding Rate vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Rate (mol/m²/s)')
plt.grid(True)

ax = plt.subplot(2, 3, (6,7), projection='3d')
Z, T = np.meshgrid(z, t)
ax.plot_surface(Z, T, c.T, cmap='viridis')
ax.set_title('Concentration Profile c(z,t)')
ax.set_xlabel('z (m)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Concentration (mol/m³)')

plt.tight_layout()
plt.show()
