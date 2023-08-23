# -*- coding: utf-8 -*-
"""
Originally Created on Tue May  7 14:33:49 2019
Updated on Wed August 23 14:14:23 2023

@author: Yannik
"""

import numpy as np
import matplotlib.pyplot as plt


def membrane_voltage(vm, i_stim, cm, g_leak, params):
    r = 1 / g_leak
    tau = cm * r

    for index in range(len(params["time"]) - 1):
        if vm[index] < params["v_thr"]:
            vm[index + 1] = (((-vm[index] + params["v_rest"]) / tau) + (i_stim[index] / cm)) * \
                            params["step_size"] + vm[index]
        elif vm[index] == params["v_spike"]:
            vm[index + 1] = params["v_rest"]
        elif vm[index] > params["v_thr"]:
            vm[index + 1] = params["v_spike"]

    return vm


# Parameters
params = {
    "v_rest": -70e-3, # Resting membrane potential (mV)
    "v_thr": -30e-3, # Threshold for firing an action potential (mV)
    "v_spike": 20e-3, # Voltage during an action potential (mV)
    "step_size": 30e-6, # Time step size (ms)
}

params["time"] = np.arange(0, 0.1, params["step_size"])

vm = np.zeros(len(params["time"]))
vm[0] = params["v_rest"]

i_stim = np.zeros(len(params["time"]))
i_stim[:4999] = 10e-6

# Plotting
fig, axes = plt.subplots(2, 2, sharex='col', clear=True)
fig.suptitle('LIF-model: Influence of cm and g_leak on firing rate', fontsize=16)

conditions = [
    {"title": 'cm: high', "cm": 10e-6, "g_leak": 200e-6},
    {"title": 'cm: low', "cm": 5e-6, "g_leak": 200e-6},
    {"title": 'g_leak: high', "cm": 10e-6, "g_leak": 300e-6},
    {"title": 'g_leak: low', "cm": 10e-6, "g_leak": 50e-6}
]

for ax, condition in zip(axes.flatten(), conditions):
    vm_copy = np.copy(vm)  # Create a copy of vm for each condition
    ax.plot(params["time"], membrane_voltage(vm_copy, i_stim, condition["cm"], condition["g_leak"], params), label='')
    ax.hlines(params["v_thr"], params["time"][0], params["time"][-1], linestyles='dashed', label='threshold')
    ax.set_title(condition["title"])

plt.show()
