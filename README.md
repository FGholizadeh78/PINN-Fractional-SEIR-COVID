# 🦠 PINN-Fractional-SEIR-COVID

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A Physics-Informed Neural Network (PINN) framework for solving the SEIR epidemic model across two countries (Ireland and Italy), supporting both integer-order and fractional-order derivatives.

</div>

---

## 📖 Overview

This project provides a comprehensive framework for parameter identification and forecasting of the first COVID-19 wave in Ireland and Italy. It implements Physics-Informed Neural Networks (PINNs) that are trained to satisfy the SEIR ODEs while fitting observed daily new cases. 

To capture memory effects in epidemic dynamics, the framework supports three mathematical approaches for the time derivative:
1. **AD (Automatic Differentiation)**: Standard integer-order SEIR model (alpha = 1).
2. **L1 Scheme**: Caputo fractional derivative.
3. **GL Scheme**: Grünwald-Letnikov fractional derivative.

Results are validated against classical SciPy least-squares baselines and evaluated using an 80/20 train/test temporal split.

## 🔬 Mathematical Model

The classic SEIR compartments are defined by:
- dS/dt = -beta * S * I
- dE/dt = beta * S * I - sigma * E
- dI/dt = sigma * E - gamma * I
- dR/dt = gamma * I

For fractional-order extensions (alpha < 1), the standard time derivative is replaced by a fractional derivative (Caputo or Grünwald-Letnikov). 
Observed daily incidence: y_hat = sigma * E * N_eff

## 📁 Project Structure

The repository contains 6 main notebooks divided by country and method, along with their respective datasets:

- **autodiff Ireland.ipynb** & **autodiff Italy.ipynb**: Integer-order SEIR modeling using PyTorch autograd.
- **L1Ireland.ipynb** & **L1Italy.ipynb**: Fractional-order SEIR using the L1 scheme for the Caputo derivative.
- **GLIreland.ipynb** & **GLItaly.ipynb**: Fractional-order SEIR using the Grünwald-Letnikov derivative.
- **COVID_19_Ireland.csv** & **COVID_19_Italy.csv**: Daily COVID-19 case data for both countries.

## ✨ Features

- ✅ Multi-country modeling (Ireland & Italy).
- ✅ Multi-method approach (Autodiff, L1, GL).
- ✅ PINN SEIR solver using PyTorch for time derivatives.
- ✅ Two-stage optimization: Adam warm-up + L-BFGS.
- ✅ Mass-conservation loss (S + E + I + R = 1) enforced during training.
- ✅ Classical ODE baseline via SciPy.
- ✅ Reproducible (fixed seeds, deterministic).

## 🛠 Requirements

```bash
pip install numpy pandas torch matplotlib scipy jupyter
