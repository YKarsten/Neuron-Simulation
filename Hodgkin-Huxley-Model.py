# -*- coding: utf-8 -*-
"""
Originally Created on Wed May  8 09:48:37 2019

Updated on Wed August 23 14:45:42 2023
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Constants
np.random.seed(1000)
tmin = 0.0
tmax = 50.0
gK = 36.0  # Potassium conductivity [ms/cm2]
gNa = 120.0  # Sodium conductivity [ms/cm2]
gL = 0.3  # Leak conductivity [ms/cm2]
Cm = 1.0  # Membrane potential [uF/cm2]
VK = -12.0  # Nernst Potential Potassium [mV]
VNa = 115.0  # Nernst Potential Sodium [mV]
Vl = 10.6  # Leakage potential [mV]
T = np.linspace(tmin, tmax, 10000)


# Ion-channel rate functions
def alpha_n(Vm):
    return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)


def beta_n(Vm):
    return 0.125 * np.exp(-Vm / 80.0)


def alpha_m(Vm):
    return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)


def beta_m(Vm):
    return 4.0 * np.exp(-Vm / 18.0)


def alpha_h(Vm):
    return 0.07 * np.exp(-Vm / 20.0)


def beta_h(Vm):
    return 1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0)


# Steady-state values
def n_inf(Vm=0.0):
    return alpha_n(Vm) / (alpha_n(Vm) + beta_n(Vm))


def m_inf(Vm=0.0):
    return alpha_m(Vm) / (alpha_m(Vm) + beta_m(Vm))


def h_inf(Vm=0.0):
    return alpha_h(Vm) / (alpha_h(Vm) + beta_h(Vm))


# Input stimulus
def Id(t):
    if 0.0 < t < 1.0:
        return 150.0
    elif 10.0 < t < 11.0:
        return 50.0
    return 0.0


# Compute derivatives
def compute_derivatives(y, t0):
    Vm, n, m, h = y

    GK = (gK / Cm) * n ** 4.0
    GNa = (gNa / Cm) * m ** 3.0 * h
    GL = gL / Cm

    dy = [
        (Id(t0) / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl)),
        alpha_n(Vm) * (1.0 - n) - beta_n(Vm) * n,
        alpha_m(Vm) * (1.0 - m) - beta_m(Vm) * m,
        alpha_h(Vm) * (1.0 - h) - beta_h(Vm) * h
    ]
    return dy


# Initial conditions
Y0 = [0.0, n_inf(), m_inf(), h_inf()]

# Solve ODE system (ordinary differential equations)
Vy = odeint(compute_derivatives, Y0, T)

# Input stimulus
Idv = np.vectorize(Id)(T)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].plot(T, Idv, linestyle= "dashed", color="black" )
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Current density (uA/$cm^2$)')
axes[0, 0].set_title('Stimulus (Current density)')
axes[0, 0].grid()

axes[0, 0].plot(T, Vy[:, 0])
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Vm (mV)')
axes[0, 0].set_title('Neuron potential with two spikes')
axes[0, 0].grid()

axes[1, 0].plot(Vy[:, 0], Vy[:, 1], label='Vm - n')
axes[1, 0].plot(Vy[:, 0], Vy[:, 2], label='Vm - m')
axes[1, 0].set_title('Limit cycles')
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].plot(T, Vy[:, 1], label='T - n')
axes[1, 1].plot(T, Vy[:, 2], label='T - m')
axes[1, 1].plot(T, Vy[:, 3], label='T - h')
axes[1, 1].set_title('Limit cycles')
axes[1, 1].set_xlabel('Time (ms)')
axes[1, 1].set_ylabel('Gating Value')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()
