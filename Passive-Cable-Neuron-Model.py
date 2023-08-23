# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:02:41 2019

@author: Yannik
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
# time
dt = 0.01
t = np.arange(dt, 50 + dt, dt)

# compartments
n = 100

# membrane
Cm = 1
Rm = 10

# membrane voltage
Vm = np.ones((n, len(t)))

# axon
p = 0.1
L = 0.1e-4
r = 2e-4
Ra = (p * L) / (np.pi * r ** 2)

c = np.ones(n) * -2
c[0] = -1
c[-1] = -1

C = np.diag(c) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)

A = np.eye(n) - (dt / (Cm * Ra)) * C

Istim = np.zeros((n, len(t)))
Istim[n // 2, :] = 10

for idx in range(len(t) - 1):
    Ihh = Vm[:, idx] / Rm
    b = Vm[:, idx] + dt / Cm * (-Ihh + Istim[:, idx])
    Vm[:, idx + 1] = np.linalg.solve(A, b)

lambda_val = np.max(Vm[49, :]) * 0.37

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Color scaled image of compartments
ax = axes[0, 0]
im = ax.imshow(Vm, aspect='auto', cmap='viridis')
fig.colorbar(im, ax=ax)
ax.set_xlabel('Time in ms', fontsize=16)
ax.set_ylabel('Compartment Nr.', fontsize=16)
ax.set_title('Color scaled image of 100 compartments in passive multi compartment model', fontsize=18)
ax.set_xticks(np.arange(0, len(t), len(t) // 10))
ax.set_xticklabels(np.arange(0, 50, 5))
ax.tick_params(labelsize=14)

# Constant stimulation of compartment 50
ax = axes[0, 1]
ax.plot(t, Vm[49, :], linewidth=2, color='black')
#ax.set_xlabel('time [ms]', fontsize=16)
#ax.set_ylabel('V_m [mV]', fontsize=16)
#ax.set_title('Constant stimulation (Istim = 10 μA) of compartment no. 50', fontsize=18)
#ax.grid(True)
#ax.set_xticks(np.arange(0, len(t), 1000))
#ax.set_xticklabels(np.arange(0, 50, 10))
#ax.tick_params(labelsize=14)

# Exponential decay of Vm along the axon at steady state (t = 4000 ms)
ax = axes[1, 0]
ax.plot(Vm[:, 3999], linewidth=2, color='black')
#ax.set_xlabel('Compartment no.', fontsize=16)
#ax.set_ylabel('V_m [mV]', fontsize=16)
#ax.set_title('Vm along the axon in steady state (t = 20 ms)', fontsize=18)
#ax.set_xticks([0, 20, 35, 50, 65, 80, 100])
#ax.set_xticklabels(['0', '20', f'{lambda_val:.2f}', '50', f'{lambda_val:.2f}', '80', '100'])
#ax.tick_params(labelsize=14)
#ax.set_ylim(0.01, 3.5)

plt.tight_layout()
plt.show()
