# 🧪 Antibody Binding Kinetics — SBE2250 Course Project (Spring 2025)
## 📌 Project Overview

The goal of this project is to model and simulate the **kinetics of antibody-antigen binding** in a flow chamber using:
- 🧮 **Numerical methods** (Finite Difference - Crank Nicolson - Runge-Kutta)
- 🤖 **Machine Learning methods** (Physics-Informed Neural Networks - PINNs)

We reproduce and extend the results from **Chapter 2 of _Schiesser – PDE – MATLAB – 2013_**, which models the spatial-temporal concentration of antibodies undergoing **diffusion** and **surface reaction binding**.

---

## 🧠 Problem Formulation

We solve the following **partial differential equation (PDE)** for the concentration \( c(z,t) \) of free analyte (e.g. antibodies):

\[
\frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial z^2}
\]

With boundary and initial conditions:
- **Initial condition:** \( c(z,0) = 1 \)
- **Dirichlet boundary (z = 1):** \( c(1,t) = 1 \)
- **Surface reaction (z = 0):**
  \[
  D \frac{\partial c}{\partial z}(0,t) = k_f c(0,t)(c_{\text{sat}} - c_b(t)) - k_r c_b(t)
  \]

Where \( c_b(t) \) is the **bound surface complex**, and is solved as a separate ODE:
\[
\frac{d c_b}{dt} = k_f c(0,t)(c_{\text{sat}} - c_b(t)) - k_r c_b(t)
\]
---

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



