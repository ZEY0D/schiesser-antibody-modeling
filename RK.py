import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1
T_final = 0.1
alpha = 0.01
Nx = 100
Nt = 100
dx = L / (Nx - 1)
dt = T_final / Nt

# Spatial and temporal grids
x = np.linspace(0, L, Nx)
t = np.linspace(0, T_final, Nt + 1)

# Initial condition: u(x,0) = sin(pi*x)
u = np.sin(np.pi * x)

# Solution matrix U(x,t)
U = np.zeros((Nx, Nt + 1))
U[:, 0] = u

# Time integration using RK4
for n in range(Nt):
    k1 = np.zeros(Nx)
    k2 = np.zeros(Nx)
    k3 = np.zeros(Nx)
    k4 = np.zeros(Nx)

    # Boundary conditions
    u[0] = 0
    u[-1] = 0

    # k1
    for i in range(1, Nx - 1):
        k1[i] = alpha * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

    u_temp = u + dt / 2 * k1
    for i in range(1, Nx - 1):
        k2[i] = alpha * (u_temp[i - 1] - 2 * u_temp[i] + u_temp[i + 1]) / dx**2

    u_temp = u + dt / 2 * k2
    for i in range(1, Nx - 1):
        k3[i] = alpha * (u_temp[i - 1] - 2 * u_temp[i] + u_temp[i + 1]) / dx**2

    u_temp = u + dt * k3
    for i in range(1, Nx - 1):
        k4[i] = alpha * (u_temp[i - 1] - 2 * u_temp[i] + u_temp[i + 1]) / dx**2

    # Update u for next time step using RK4
    u += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    U[:, n + 1] = u

# Plotting the solution
X, T = np.meshgrid(x, t)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U.T, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Solution of the Heat Equation using RK4')
plt.show()