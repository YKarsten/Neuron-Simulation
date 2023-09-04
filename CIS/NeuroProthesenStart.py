import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from hodgkin_huxley import simulate_hodgkin_huxley
from scipy.signal import butter, filtfilt

# Build signal
fs = 100000
f = 1000
f2 = 1200
duration = 100e-3  # sec
t = np.arange(0, fs * duration) / fs
signal = np.sin(2 * np.pi * t * f)
signal2 = np.sin(2 * np.pi * t * f2)
mod_sin = (1 - np.cos(2 * np.pi * t * 50))
mod_sin2 = (1 - np.cos(2 * np.pi * t * 100))
signal = signal2 * mod_sin2 + signal * mod_sin

# CIS
# Middle ear
def middle_ear(input, fs):
    order = 2
    fcut = np.array([500, 4000]) / (fs / 2)
    b, a = butter(order, fcut, btype='band')
    out = filtfilt(b, a, input)
    return out

signal = middle_ear(signal, fs)  

# Filterbank
def basilarmembran(signal_in, fs):
    # Only one channel
    signal = signal_in

    # Third octave middle frequencies from 500(-3) to 4k (6) (-1/+1 for border calculation!)
    to_f = 1000 * 2.0 ** np.arange(-4, 8) / 3.0  # Use floating-point numbers

    # Lower/upper border frequencies for the defined to_f
    band_edges = np.sqrt(to_f[:-1] * to_f[1:])

    # Cut the +1/-1
    to_f = to_f[1:-1]

    # Preallocation
    signal_out = np.zeros((len(signal_in), len(to_f)))

    # Filter order
    order = 2

    # Loop over frequency band + filter
    for idx in range(len(to_f)):
        fcut = [band_edges[idx] / (fs / 2), band_edges[idx + 1] / (fs / 2)]  # Divide each element by fs/2
        b, a = butter(order, fcut, btype='band')
        signal_out[:, idx] = filtfilt(b, a, signal)

    return signal_out

filterbank = basilarmembran(signal, fs) 

# # Inner ear
# def haircell(input, fs):
#     cutoff = 200
#     order = 4
#     compress_power = 1
    
#     # Half-wave rectification
#     rect = np.maximum(input, 0)
    
#     # Compression
#     output = rect ** compress_power
    
#     # Lowpass filtering
#     fcut = cutoff / (fs / 2)

#     # Calculate the required padlen based on the filter order
#     padlen = (order * 2) + 1

#     # Apply the low-pass filter
#     b, a = butter(order, fcut, btype='low')
#     output = filtfilt(b, a, output, padlen=padlen)
    
#     return output

# Inner ear
def haircell(input, fs):
    cutoff = 200
    order = 4
    compress_power = 1
    
    # Initialize the output envelope
    output = np.zeros_like(input)
    
    for channel in range(input.shape[1]):
        # Half-wave rectification
        rect = np.maximum(input[:, channel], 0)
        
        # Compression
        output[:, channel] = rect ** compress_power
        
        # Lowpass filtering
        fcut = cutoff / (fs / 2)

        # Calculate the required padlen based on the filter order
        padlen = (order * 2) + 1

        # Apply the low-pass filter
        b, a = butter(order, fcut, btype='low')
        output[:, channel] = filtfilt(b, a, output[:, channel], padlen=padlen)
    
    return output


envelope = haircell(filterbank, fs)  

# Kernel
kernel_duration = 200e-6
kernel = np.concatenate((np.ones(int((kernel_duration / 2) * fs)),
                         -np.ones(int((kernel_duration / 2) * fs))))

# Only sizes
n1, n2 = envelope.shape

# PPS
pps = 800

# Period duration in samples
T = round(fs / pps)

# Dirac train (no zero at the first entry!)
pulses = np.zeros(n1)
pulses[range(T - 1, n1, T)] = 1

# Preallocation
final_stim = np.zeros((n1, n2))

for idx in range(n2):
    # Delay for the different electrodes
    phase = 1 / pps * idx / n2
    phase_samples = round(fs * phase)

    # Sampled envelope
    sampled_envelope = pulses * envelope[:, idx]

    # Add the delay
    stim_plus_phase = np.concatenate((np.zeros(phase_samples), np.convolve(sampled_envelope, kernel)))

    # Cut the stim to the right length
    final_stim[:, idx] = stim_plus_phase[:n1]



# Generate modulated pulse trains with a frequency of 800 Hz
def generate_modulated_pulse_trains(fs, duration, pps):
    t = np.arange(0, duration, 1/fs)
    pulse_train = np.zeros(len(t))
    pulse_period = int(fs / pps)
    pulse_train[::pulse_period] = 1
    return pulse_train

# Generate modulated pulse trains for each channel
pps = 800
modulated_pulse_trains = [generate_modulated_pulse_trains(fs, duration, pps) for _ in range(n2)]

# Fit the modulated pulse trains to the envelope
modulated_stim = np.zeros((n1, n2))
for idx in range(n2):
    modulated_stim[:, idx] = modulated_pulse_trains[idx][:n1] * envelope[:, idx]

# Adapt the stimulus from Pascal to μA
final_stim = modulated_stim * 700  # μA

# Plot the results
plt.figure(figsize=[20, 8])
plt.subplot(4, 1, 1)
plt.plot(signal)

plt.subplot(4, 1, 2)
plt.plot(filterbank[:, 4])

plt.subplot(4, 1, 3)
plt.plot(envelope[:, 4])

plt.subplot(4, 1, 4)
plt.plot(envelope[:, 4])
plt.plot(modulated_stim[:, 4])

plt.show()

# Single neuron version
def simulate_hodgkin_huxley(input_signal, fs):
    # Constants
    Cm = 1.0  # Membrane capacitance (uF/cm^2)
    V_Na = 50.0  # Sodium Nernst potential (mV)
    V_K = -77.0  # Potassium Nernst potential (mV)
    V_L = -54.4  # Leak Nernst potential (mV)

    G_Na = 120.0  # Sodium conductance (mS/cm^2)
    G_K = 36.0  # Potassium conductance (mS/cm^2)
    G_L = 0.3  # Leak conductance (mS/cm^2)

    # Time parameters
    dt = 0.01  # Time step (ms)
    t_max = 100.0  # Maximum simulation time (ms)
    num_steps = int(t_max / dt)

    # Initialize variables
    t = np.arange(0, t_max, dt)
    Vm = np.zeros(num_steps)  # Membrane potential (mV)
    m = np.zeros(num_steps)  # Sodium activation gate
    h = np.zeros(num_steps)  # Sodium inactivation gate
    n = np.zeros(num_steps)  # Potassium activation gate

    # Initial conditions
    Vm[0] = -65.0  # Resting membrane potential (mV)
    m[0] = 0.05  # Initial sodium activation
    h[0] = 0.6  # Initial sodium inactivation
    n[0] = 0.32  # Initial potassium activation

    # External current stimulation (uA/cm^2)
    I_stim = np.zeros(num_steps)
    I_stim[1000:4000] = 10.0  # Inject a current pulse from 100 to 400 ms

    # Simulation loop
    for i in range(1, num_steps):
        # Hodgkin-Huxley equations
        alpha_m = 0.1 * (Vm[i - 1] + 40.0) / (1.0 - np.exp(-0.1 * (Vm[i - 1] + 40.0)))
        beta_m = 4.0 * np.exp(-(Vm[i - 1] + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(Vm[i - 1] + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-0.1 * (Vm[i - 1] + 35.0)))
        alpha_n = 0.01 * (Vm[i - 1] + 55.0) / (1.0 - np.exp(-0.1 * (Vm[i - 1] + 55.0)))
        beta_n = 0.125 * np.exp(-(Vm[i - 1] + 65.0) / 80.0)

        m[i] = m[i - 1] + dt * (alpha_m * (1.0 - m[i - 1]) - beta_m * m[i - 1])
        h[i] = h[i - 1] + dt * (alpha_h * (1.0 - h[i - 1]) - beta_h * h[i - 1])
        n[i] = n[i - 1] + dt * (alpha_n * (1.0 - n[i - 1]) - beta_n * n[i - 1])

        I_Na = G_Na * m[i] ** 3 * h[i] * (Vm[i - 1] - V_Na)
        I_K = G_K * n[i] ** 4 * (Vm[i - 1] - V_K)
        I_L = G_L * (Vm[i - 1] - V_L)

        # Update membrane potential using the equation Cm * dV/dt = I_stim - I_Na - I_K - I_L
        Vm[i] = Vm[i - 1] + (dt / Cm) * (I_stim[i] - I_Na - I_K - I_L)

    return Vm, t


Vm, t = simulate_hodgkin_huxley(final_stim, fs)

# # Plot the results
# plt.figure(1)

# # First subplot (Vm)
# plt.subplot(2, 1, 1)
# plt.plot(t, Vm)
# plt.ylabel('Vm in mV')
# plt.title('CIS Stimulation response in a single Neuron - 4')

# # Create a second y-axis for the stimulation current
# plt2 = plt.twinx()

# # Plot the stimulation current for channel 4 on the second y-axis
# line1, = plt2.plot(t, final_stim[:, 4], ':', linewidth=2, color='red', label='channel 4 (Stim)')
# plt2.set_ylabel('Stimulus in μA')

# # Second subplot (Stimulus for multiple channels)
# plt.subplot(2, 1, 2)
# plt.plot(t, final_stim[:, 4], ':', linewidth=2, label='channel 4')
# plt.plot(t, final_stim[:, 6], ':', linewidth=2, label='channel 5')
# plt.xlim(8, 15)
# plt.legend(['channel 4', 'channel 5', 'channel 6', 'channel 7'])
# plt.xlabel('Time in ms')
# plt.ylabel('Stimulus in μA')

# # Get the legend handles and labels for both subplots
# handles1, labels1 = plt.gca().get_legend_handles_labels()
# handles2, labels2 = plt2.get_legend_handles_labels()

# # Combine legend handles and labels from both subplots
# handles = handles1 + handles2
# labels = labels1 + labels2

# # Add a legend using the combined handles and labels
# plt.legend(handles, labels, loc='upper right')

# plt.tight_layout()
# plt.show()

# # Multi-neuron version
# neurons = 100

# # Determine the desired size of the window
# window_size = n2

# # Only a simple triangle for the radiation behavior
# window = np.concatenate((np.arange(0, window_size) / window_size, np.arange(window_size - 2, -1, -1) / window_size))


# # Weighting matrix for radiation behavior (windows per electrode)
# weighting_matrix = np.zeros((n2, neurons))

# # Index for the middle of the window
# middle_idx = np.round(np.linspace(len(window) / 2, neurons - len(window) / 2, n2)).astype(int)

# # Set the triangles in the right place in the matrix
# for idx in range(n2):
#     idx1 = middle_idx[idx] - len(window) // 2
#     idx2 = middle_idx[idx] + len(window) // 2
#     weighting_matrix[idx, idx1:idx2 + 1] = window

# # Matrix multiplication (tricky: dimensions/order! only this works!)
# final_stim = np.dot(weighting_matrix.T, final_stim.T).T


# # Initialize Vm and t
# Vm = np.zeros((len(final_stim), neurons))
# t = np.zeros(len(final_stim))

# for idx in range(n2):
#     Vm[:, idx], t = simulate_hodgkin_huxley(final_stim[idx, :], fs)  

# # Plot the results
# plt.figure(2)
# extent = [0, t[-1], 0, neurons]  # Set the extent based on the available data
# plt.imshow(Vm.T, extent=extent, aspect='auto', cmap='viridis')
# plt.yticks(range(0, 101, 5), range(500, 4001, 175))
# plt.ylabel('Frequency in Hz')
# plt.xlabel('Time in ms')
# plt.title('CIS Stimulation response in multiple Neurons')
# plt.colorbar(label='Vm in mV')

# # Adds horizontal dots
# for idx in middle_idx:
#     plt.plot([0, t[-1]], [idx, idx], 'w:', linewidth=1.5)


# plt.ylim(30, 55)

# plt.show()

# # Plot the weighting matrix
# plt.figure(3)
# plt.imshow(weighting_matrix, cmap='viridis')
# plt.xlabel('Neurons')
# plt.ylabel('Electrode channel')
# plt.title('Weighting matrix')
# plt.colorbar()
# plt.show()
