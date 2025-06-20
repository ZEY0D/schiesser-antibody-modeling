# 🧪 Antibody Binding Kinetics — SBE2250 Course Project (Spring 2025)
## 📌 Project Overview

The goal of this project is to model and simulate the **kinetics of antibody-antigen binding** in a flow chamber using:
- 🧮 **Numerical methods** (Finite Difference - Crank Nicolson - Runge-Kutta)
- 🤖 **Machine Learning methods** (Physics-Informed Neural Networks - PINNs)

We reproduce and extend the results from **Chapter 2 of _Schiesser – PDE – MATLAB – 2013_**, which models the spatial-temporal concentration of antibodies undergoing **diffusion** and **surface reaction binding**.

---

## 🧠 Problem Formulation
We solve a coupled PDE-ODE system modeling the binding kinetics of free analyte (e.g., antibodies) diffusing in a fluid and binding to receptors on a sensor surface.

Governing PDE (Diffusion of Free Analyte c(z,t))
∂c/∂t = D ∂²c/∂z²

This describes the one-dimensional diffusion of the free analyte in a domain of length L.

Initial and Boundary Conditions
Initial Condition:
c(z, 0) = 1
(Initial concentration is uniform throughout the domain)

Dirichlet Boundary Condition at z = 1 (top boundary):
c(1, t) = 1
(Maintains the analyte concentration at the top boundary at a constant bulk value)

Reactive Boundary Condition at z = 0 (sensor surface):
D ∂c/∂z (0, t) = kf * c(0, t) * (csat - cb(t)) - kr * cb(t)
(Models the binding reaction flux at the surface)

Here:

c(z,t) is the free analyte concentration in the fluid

cb(t) is the surface-bound complex concentration

csat is the saturation value of cb

kf and kr are the forward and reverse rate constants

D is the diffusion coefficient

Coupled ODE (Surface Reaction Kinetics)
dc_b/dt = kf * c(0, t) * (csat - cb(t)) - kr * cb(t)

This ODE governs the time evolution of the surface-bound complex cb(t), driven by the analyte concentration at the surface (z = 0)

## 📚 Literature Review

- **Physics-Informed Neural Networks (PINNs):**  
  Raissi et al., [Nature Reviews Physics, 2021](https://www.nature.com/articles/s42254-021-00314-5)

- **Survey of Physics-Based ML:**  
  Xiaowei et al., [Physics-based Modeling + ML Survey](https://beiyulincs.github.io/teach/fall_2020/papers/xiaowei.pdf)

- **Deep Learning for PDEs:**  
  [https://physicsbaseddeeplearning.org](https://physicsbaseddeeplearning.org)

These studies validate the effectiveness of using PINNs for biomedical PDEs, especially in low-data regimes.


## 🔧 How to Run This Project

   ```bash
   git clone https://github.com/ZEY0D/schiesser-antibody-modeling.git
   cd schiesser-antibody-modeling



