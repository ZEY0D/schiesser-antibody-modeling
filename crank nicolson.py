import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import solve

# Global parameters (Table 2.2)
D = 1e-10
kf = 1e5
kr = 1e1
cbsat = 1.66e-9
cbulk = 4.48e-5
h = 5e-5
zl = 0
zu = h
zn = 21
dz = (zu - zl) / (zn - 1)
dz2 = dz**2
tl = 0
tu = 100
tn = 51
dt = (tu - tl) / (tn - 1)
n = zn - 1  # Number of spatial points excluding boundary

def crank_nicolson_solver():
    """
    Solve the PDE-ODE system using Crank-Nicolson method
    """
    # Time and space grids
    t = np.linspace(tl, tu, tn)
    z = np.linspace(zl, zu, zn)
    
    # Initialize solution arrays
    c = np.zeros((tn, n))  # c(z,t) for interior points
    cb = np.zeros(tn)      # cb(t) surface bound concentration
    
    # Initial conditions (all zeros)
    c[0, :] = 0.0
    cb[0] = 0.0
    
    # Crank-Nicolson coefficient
    r = D * dt / (2 * dz2)
    
    # Time stepping
    for it in range(1, tn):
        # Current time step
        t_curr = t[it]
        t_prev = t[it-1]
        
        # Previous values
        c_old = c[it-1, :].copy()
        cb_old = cb[it-1]
        
        # Iterative solution for coupled PDE-ODE system
        # Initial guess for new time step
        c_new = c_old.copy()
        cb_new = cb_old
        
        # Newton-Raphson iterations for nonlinear coupling
        for newton_iter in range(10):  # Max 10 iterations
            c_prev_iter = c_new.copy()
            cb_prev_iter = cb_new
            
            # Set up the linear system for concentration field
            # A * c_new = b
            A = np.zeros((n, n))
            b = np.zeros(n)
            
            # Interior points (i = 1 to n-2)
            for i in range(1, n-1):
                A[i, i-1] = -r
                A[i, i] = 1 + 2*r
                A[i, i+1] = -r
                b[i] = r*c_old[i-1] + (1-2*r)*c_old[i] + r*c_old[i+1]
            
            # Boundary condition at z=0 (i=0)
            # Using the flux boundary condition with Crank-Nicolson
            # c[1] - c[-1] = (2*dz/D) * (kf * c[0] * (cbsat - cb) - kr * cb)
            # This gives us: c[-1] = c[1] - (2*dz/D) * flux
            # For Crank-Nicolson, we average the flux terms
            
            flux_old = kf * c_old[0] * (cbsat - cb_old) - kr * cb_old
            flux_new = kf * c_new[0] * (cbsat - cb_new) - kr * cb_new
            flux_avg = 0.5 * (flux_old + flux_new)
            
            # Ghost point: c[-1] = c[1] - (2*dz/D) * flux_avg
            # Discretization at i=0: (c[1] - 2*c[0] + c[-1])/dz^2
            # Substituting c[-1]: (c[1] - 2*c[0] + c[1] - (2*dz/D)*flux_avg)/dz^2
            # = (2*c[1] - 2*c[0] - (2*dz/D)*flux_avg)/dz^2
            
            A[0, 0] = 1 + 2*r + r*(2*dz*kf*(cbsat - cb_new)/D)
            A[0, 1] = -2*r
            b[0] = (1 - 2*r - r*(2*dz*kf*(cbsat - cb_old)/D))*c_old[0] + 2*r*c_old[1] + \
                   r*(2*dz/D)*(kr*cb_old + kr*cb_new)
            
            # Boundary condition at z=h (i=n-1): c[n-1] = cbulk
            A[n-1, n-1] = 1
            b[n-1] = cbulk
            
            # Solve for c_new
            c_new = solve(A, b)
            
            # Update cb using Crank-Nicolson for the ODE
            # dcb/dt = kf * c[0] * (cbsat - cb) - kr * cb
            # Crank-Nicolson: (cb_new - cb_old)/dt = 0.5 * (RHS_old + RHS_new)
            
            RHS_old = kf * c_old[0] * (cbsat - cb_old) - kr * cb_old
            RHS_new = kf * c_new[0] * (cbsat - cb_new) - kr * cb_new
            
            # Rearrange to solve for cb_new
            # cb_new - cb_old = 0.5 * dt * (RHS_old + RHS_new)
            # cb_new - 0.5*dt*(kf*c_new[0]*(cbsat - cb_new) - kr*cb_new) = cb_old + 0.5*dt*RHS_old
            # cb_new + 0.5*dt*(kf*c_new[0]*cb_new + kr*cb_new) = cb_old + 0.5*dt*RHS_old + 0.5*dt*kf*c_new[0]*cbsat
            # cb_new * (1 + 0.5*dt*(kf*c_new[0] + kr)) = cb_old + 0.5*dt*RHS_old + 0.5*dt*kf*c_new[0]*cbsat
            
            denominator = 1 + 0.5*dt*(kf*c_new[0] + kr)
            numerator = cb_old + 0.5*dt*RHS_old + 0.5*dt*kf*c_new[0]*cbsat
            cb_new = numerator / denominator
            
            # Check convergence
            c_change = np.max(np.abs(c_new - c_prev_iter))
            cb_change = abs(cb_new - cb_prev_iter)
            
            if c_change < 1e-10 and cb_change < 1e-12:
                break
        
        # Store results
        c[it, :] = c_new
        cb[it] = cb_new
    
    return t, z, c, cb

# Solve the system
t, z, c, cb = crank_nicolson_solver()

# Extract data for plotting (same as original)
c_plot = c[:, 0]       # c at z=0 over time
cb_plot = cb           # surface bound concentration cb over time
theta_plot = cb_plot / cbsat
rate_plot = kf * c_plot * (cbsat - cb_plot) - kr * cb_plot

# Print output table (same format as original)
nout = len(t)
print(f"{'t':>6} {'c(0,t)':>12} {'cb(t)':>12} {'theta':>12} {'rate':>12}")
for it in range(nout):
    print(f"{t[it]:6.0f} {c_plot[it]:12.3e} {cb_plot[it]:12.3e} "
          f"{theta_plot[it]:12.3e} {rate_plot[it]:12.3e}")

# Plotting (same as original)
plt.figure(figsize=(12, 10))

# subplot 1: c(0,t)
plt.subplot(2, 2, 1)
plt.plot(t, c_plot)
plt.title('c(0,t) vs t')
plt.xlabel('t (s)')
plt.ylabel('c(0,t)')
plt.grid(True)

# subplot 2: cb(t)
plt.subplot(2, 2, 2)
plt.plot(t, cb_plot)
plt.title('cb(t) vs t')
plt.xlabel('t (s)')
plt.ylabel('cb(t)')
plt.grid(True)

# subplot 3: theta(t)
plt.subplot(2, 2, 3)
plt.plot(t, theta_plot)
plt.title('theta(t) vs t')
plt.xlabel('t (s)')
plt.ylabel('theta(t)')
plt.grid(True)

# subplot 4: rate(t)
plt.subplot(2, 2, 4)
plt.plot(t, rate_plot)
plt.title('rate(t) vs t')
plt.xlabel('t (s)')
plt.ylabel('rate(t)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D surface plot of concentration c(z,t)
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

Z, T = np.meshgrid(z[:n], t)  # meshgrid for plotting

# c shape is (time, space)
surf = ax.plot_surface(Z, T, c, cmap='viridis')

ax.set_xlabel('z (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('c(z,t) (moles/m^3)')
ax.set_title('c(z,t) over space and time')

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()