import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


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


def update_plot(i_stim_val, g_leak_val):
    vm = np.zeros(len(params["time"]))
    vm[0] = params["v_rest"]

    i_stim = np.zeros(len(params["time"]))
    i_stim[:4999] = i_stim_val

    vm = membrane_voltage(vm, i_stim, params["cm"], g_leak_val, params)

    ax.clear()
    ax.plot(params["time"], vm)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Membrane Voltage (V)")
    ax.set_title("Membrane Voltage with Sliders")
    ax.grid()

    plt.draw()


params = {
    "v_rest": -70e-3,
    "v_thr": -30e-3,
    "v_spike": 20e-3,
    "step_size": 30e-6,
    "cm": 1e-6,  # Update with the correct membrane capacitance value
}

params["time"] = np.arange(0, 0.1, params["step_size"])

# Create sliders
i_stim_slider = widgets.FloatSlider(value=10e-6, min=0, max=50e-6, step=1e-6, description="I_stim (A)")
g_leak_slider = widgets.FloatSlider(value=1e-8, min=1e-9, max=1e-7, step=1e-9, description="g_leak (S)")

# Create a plot for the initial values
fig, ax = plt.subplots()
update_plot(i_stim_slider.value, g_leak_slider.value)


# Create a callback function for slider changes
def on_slider_change(change):
    i_stim_val = i_stim_slider.value
    g_leak_val = g_leak_slider.value
    update_plot(i_stim_val, g_leak_val)


i_stim_slider.observe(on_slider_change, names='value')
g_leak_slider.observe(on_slider_change, names='value')

# Display the sliders
widgets.VBox([i_stim_slider, g_leak_slider])
