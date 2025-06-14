import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D

# Parameters (Table 2.2)
D = 1e-10           # Diffusivity (m^2/s)
kf = 1e5            # Forward reaction rate (1/M.s)
kr = 1e1            # Reverse reaction rate (1/s)
cbsat = 1.66e-9     # Saturation surface concentration (mol/m^2)
cbulk = 4.48e-5     # Bulk concentration (mol/m^3)
h = 5e-5            # Thickness (m)

# Finite Element Mesh
zl = 0
zu = h
ne = 20  # Number of elements
nn = ne + 1  # Number of nodes
z = np.linspace(zl, zu, nn)
he = h / ne  # Element length

# Time setup
tl = 0
tu = 100
tn = 51  # Same as solve_ivp version
dt = (tu - tl) / (tn - 1)
t = np.linspace(tl, tu, tn)

# Initial conditions
c = np.zeros(nn)  # Concentration at all nodes
cb = 0.0  # Surface concentration
c_save = np.zeros((tn, nn))
cb_save = np.zeros(tn)

def linear_shape_functions(xi):
    """Linear shape functions for 1D elements"""
    N1 = 0.5 * (1 - xi)  # Shape function for left node
    N2 = 0.5 * (1 + xi)  # Shape function for right node
    return N1, N2

def linear_shape_derivatives():
    """Derivatives of linear shape functions (constant in natural coordinates)"""
    dN1_dxi = -0.5
    dN2_dxi = 0.5
    return dN1_dxi, dN2_dxi

def assemble_matrices():
    """Assemble global mass and stiffness matrices"""
    M = np.zeros((nn, nn))  # Mass matrix
    K = np.zeros((nn, nn))  # Stiffness matrix
    
    # Gauss quadrature points and weights for 2-point quadrature
    xi_gauss = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w_gauss = np.array([1, 1])
    
    # Assembly loop over elements
    for e in range(ne):
        # Global node numbers for element e
        node1 = e
        node2 = e + 1
        
        # Element matrices
        Me = np.zeros((2, 2))
        Ke = np.zeros((2, 2))
        
        # Numerical integration using Gauss quadrature
        for ip in range(2):  # Integration points
            xi = xi_gauss[ip]
            w = w_gauss[ip]
            
            # Shape functions and derivatives
            N1, N2 = linear_shape_functions(xi)
            dN1_dxi, dN2_dxi = linear_shape_derivatives()
            
            # Jacobian (transformation from natural to physical coordinates)
            J = he / 2
            
            # Shape function derivatives in physical coordinates
            dN1_dz = dN1_dxi / J
            dN2_dz = dN2_dxi / J
            
            # Element mass matrix contribution
            Me[0, 0] += N1 * N1 * J * w
            Me[0, 1] += N1 * N2 * J * w
            Me[1, 0] += N2 * N1 * J * w
            Me[1, 1] += N2 * N2 * J * w
            
            # Element stiffness matrix contribution (diffusion term)
            Ke[0, 0] += D * dN1_dz * dN1_dz * J * w
            Ke[0, 1] += D * dN1_dz * dN2_dz * J * w
            Ke[1, 0] += D * dN2_dz * dN1_dz * J * w
            Ke[1, 1] += D * dN2_dz * dN2_dz * J * w
        
        # Assemble into global matrices
        nodes = [node1, node2]
        for i in range(2):
            for j in range(2):
                M[nodes[i], nodes[j]] += Me[i, j]
                K[nodes[i], nodes[j]] += Ke[i, j]
    
    return M, K

# Assemble matrices
M, K = assemble_matrices()

# Time stepping using backward Euler for stability
for it in range(tn):
    # Store results
    c_save[it, :] = c.copy()
    cb_save[it] = cb
    
    if it == tn - 1:  # Last iteration, just store
        break
    
    # Set up system: (M + dt*K)*c_new = M*c + boundary terms
    A = M + dt * K
    b = M @ c
    
    # Apply boundary condition at z=0 (reactive flux)
    # Natural boundary condition: -D * dc/dz|_{z=0} = kf * c(0) * (cbsat - cb) - kr * cb
    # This contributes to the RHS as a boundary flux term
    flux_bc = kf * c[0] * (cbsat - cb) - kr * cb
    b[0] += dt * flux_bc
    
    # Apply boundary condition at z=h (Dirichlet: c = cbulk)
    # Modify the system to enforce c[nn-1] = cbulk
    A[nn-1, :] = 0
    A[nn-1, nn-1] = 1
    b[nn-1] = cbulk
    
    # Solve the linear system
    c_new = spsolve(A, b)
    
    # Update surface concentration using implicit scheme
    # dcb/dt = kf * c[0] * (cbsat - cb) - kr * cb
    cb_new = (cb + dt * kf * c_new[0] * cbsat) / (1 + dt * (kf * c_new[0] + kr))
    
    # Update for next time step
    c = c_new.copy()
    cb = cb_new

# Extract for plotting
c_plot = c_save[:, 0]  # Concentration at z=0
cb_plot = cb_save
theta_plot = cb_plot / cbsat
rate_plot = kf * c_plot * (cbsat - cb_plot) - kr * cb_plot

# Tabular output
print(f"{'t':>6} {'c(0,t)':>12} {'cb(t)':>12} {'theta':>12} {'rate':>12}")
for i in range(tn):
    print(f"{t[i]:6.0f} {c_plot[i]:12.3e} {cb_plot[i]:12.3e} "
          f"{theta_plot[i]:12.3e} {rate_plot[i]:12.3e}")

# 2D plots
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, c_plot, color='blue', linewidth=2)
plt.title('c(0,t) vs t - Finite Element Method')
plt.xlabel('t (s)')
plt.ylabel('c(0,t) [mol/m³]')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, cb_plot, color='green', linewidth=2)
plt.title('cb(t) vs t - Finite Element Method')
plt.xlabel('t (s)')
plt.ylabel('cb(t) [mol/m²]')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, theta_plot, color='purple', linewidth=2)
plt.title('θ(t) vs t - Finite Element Method')
plt.xlabel('t (s)')
plt.ylabel('θ(t)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, rate_plot, color='red', linewidth=2)
plt.title('rate(t) vs t - Finite Element Method')
plt.xlabel('t (s)')
plt.ylabel('rate(t) [mol/m²·s]')
plt.grid(True)

plt.tight_layout()
plt.show()

# 3D plot of c(z,t)
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
Z, T = np.meshgrid(z, t)
surf = ax.plot_surface(Z, T, c_save, cmap='viridis', alpha=0.8)
ax.set_xlabel('z (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('c(z,t) [mol/m³]')
ax.set_title('Finite Element Method: c(z,t)')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()