import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

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
ncall = 0  # Global call counter for PDE function

def PDE_1(t, u):
    global ncall
    c = u[:n].copy()
    cb = u[n]
    
    # Boundary condition at z=0 with flux term cf
    cf = c[1] - (2*dz/D) * (kf * c[0] * (cbsat - cb) - kr * cb)
    # Boundary condition at z=h concentration = cbulk
    c[-1] = cbulk
    
    # PDE discretization for interior points
    ct = np.zeros(n)
    for i in range(n):
        if i == 0:
            ct[0] = D * (c[1] - 2*c[0] + cf) / dz2
        elif i == n-1:
            ct[-1] = 0  # zero flux or Dirichlet enforced via c[-1]=cbulk already
        else:
            ct[i] = D * (c[i+1] - 2*c[i] + c[i-1]) / dz2

    # ODE for surface concentration cb
    cbt = kf * c[0] * (cbsat - cb) - kr * cb
    
    ut = np.zeros(n + 1)
    ut[:n] = ct
    ut[n] = cbt
    
    ncall += 1
    return ut

# Initial conditions: all zeros
u0 = np.zeros(n + 1)

# Time points for output
tout = np.linspace(tl, tu, tn)

# Solve PDE-ODE system with stiff BDF method
sol = solve_ivp(PDE_1, [tl, tu], u0, t_eval=tout, method='BDF', rtol=1e-7, atol=1e-7)

t = sol.t
u = sol.y.T

# Extract data for plotting
c_plot = u[:, 0]       # c at z=0 over time
cb_plot = u[:, n]      # surface bound concentration cb over time
theta_plot = cb_plot / cbsat
rate_plot = kf * c_plot * (cbsat - cb_plot) - kr * cb_plot

# Print output table
nout = len(t)
print(f"{'t':>6} {'c(0,t)':>12} {'cb(t)':>12} {'theta':>12} {'rate':>12}")
for it in range(nout):
    print(f"{t[it]:6.0f} {c_plot[it]:12.3e} {cb_plot[it]:12.3e} "
          f"{theta_plot[it]:12.3e} {rate_plot[it]:12.3e}")
print(f"\n ncall = {ncall:4d}")

# Spatial grid for concentration profile (excluding last point)
z = np.linspace(zl, zu, zn)
c_3D = u[:, :n]  # c(z,t) for all spatial points except last (boundary)

# Plotting

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

Z, T = np.meshgrid(z[:n], t)  # meshgrid for plotting, note z[:n] excludes last point

# c_3D shape is (time, space), transpose for plotting as (space, time)
surf = ax.plot_surface(Z, T, c_3D, cmap='viridis')

ax.set_xlabel('z (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('c(z,t) (moles/m^3)')
ax.set_title('c(z,t) over space and time')

fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
