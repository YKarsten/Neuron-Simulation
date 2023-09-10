import numpy as np
import matplotlib.pyplot as plt

from hodgkin_huxley import simulate_hodgkin_huxley
from scipy.signal import butter, filtfilt, convolve
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Build signal
fs = 100_000 # sampling frequency
f = 1000
f2 = 1200
duration = 100e-3  # 100 ms
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
        signal_out[:, idx] = filtfilt(b, a, signal_in)

    return signal_out

filterbank = basilarmembran(signal, fs) 

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
T = round(fs / pps) # 125

# Dirac train (no zero at the first entry!)
pulses = np.zeros(n1)
pulses[range(T - 1, n1, T)] = 1

# Preallocate final_stim
final_stim = np.zeros((n1, n2))

for idx in range(1, n2 + 1):
    # Delay for the different electrodes
    phase = (1 / pps) * idx / n2
    phase_samples = int(fs * phase)
    
    # Sampled envelope
    sampled_envelope = pulses * envelope[:, idx - 1]  # Adjust for 0-based indexing
    
    # Add the delay
    stim_plus_phase = np.concatenate((np.zeros(phase_samples), convolve(sampled_envelope, kernel)))

    # Cut the stim to the right length
    final_stim[:, idx - 1] = stim_plus_phase[0: n1]

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
final_stim = final_stim * 700  # μA

# Plot the results
# plt.figure(1, figsize=(10, 8))

# # Create subplots with shared x-axis
# ax1 = plt.subplot(4, 1, 1)
# plt.plot(signal)
# plt.title("Modified sinusoid signal", fontsize=16)

# ax2 = plt.subplot(4, 1, 2, sharex=ax1)
# plt.plot(filterbank[:, 4])
# plt.title("Signal - Bandpass filtered", fontsize=16)

# ax3 = plt.subplot(4, 1, 3, sharex=ax1)
# plt.plot(envelope[:, 4])
# plt.title("Envelope extraction", fontsize=16)

# ax4 = plt.subplot(4, 1, 4, sharex=ax1)
# plt.plot(envelope[:, 4])
# plt.plot(modulated_stim[:, 4])
# plt.title("Modulated pulse trains", fontsize=16)

# # Remove x-axis labels from upper plots
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax3.get_xticklabels(), visible=False)

# plt.xlabel('Time (s)', fontsize=14)  # Label only the bottom subplot
# plt.xlim(0, 10_000)

# plt.tight_layout()  # Adjust subplot layout for better spacing
# plt.show()

# Hodgkin-Huxley-Model

# Constants
np.random.seed(1000)
tmin = 0.0
tmax = 100.0
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


idx = 0
channel = 4
stim = final_stim[:, channel]

# Compute derivatives
def compute_derivatives(y, t0):
    global idx
    idx+=1

    Vm, n, m, h = y

    GK = (gK / Cm) * n ** 4.0
    GNa = (gNa / Cm) * m ** 3.0 * h
    GL = gL / Cm

    if t0>= 100:
        t0 = 99.99
    
    dy = [
        (stim[int(t0*100)] / Cm) - (GK * (Vm - VK)) - (GNa * (Vm - VNa)) - (GL * (Vm - Vl)),
        alpha_n(Vm) * (1.0 - n) - beta_n(Vm) * n,
        alpha_m(Vm) * (1.0 - m) - beta_m(Vm) * m,
        alpha_h(Vm) * (1.0 - h) - beta_h(Vm) * h
    ]
    return dy

# Initial conditions
Y0 = [0.0, n_inf(), m_inf(), h_inf()]


Vy = odeint(compute_derivatives, Y0, T)


# Single neuron - plot membrane potential and stimulation current
fig, ax = plt.subplots(2,1)

ax[0].plot(T, Vy[:, 0]*10, 
           label = "channel 5", 
           color="black", 
           linewidth=2,
           zorder=3,
           alpha=0.8)

ax[0].set_title("CIS stimulation response in a single Neuron", fontsize=16)
ax[0].set_ylabel("Vm in mV", fontsize=16)
ax[0].set_xlim(0, 100)

ax2 = ax[0].twinx()
ax2.plot(T, final_stim[:, channel]*3, 
         color= "orange", 
         label='final_stim', 
         linestyle= ":", 
         linewidth=2,
         alpha=0.8)

ax2.set_ylabel(r'Stimulus in $\mu$A', color="orange", fontsize=16)

# Customize the color of the tick labels on the second x-axis
for tick in ax2.get_yticklabels():
    tick.set_color("orange")

ax[1].plot(T[1000:1500], final_stim[1000:1500, 3]*180, 
           color="blue", 
           label='channel 5', 
           linestyle=":")

ax[1].plot(T[1000:1500], final_stim[1000:1500, 4], 
           color="orange", 
           label='channel 6', 
           linestyle=":")

ax[1].set_title("Interleaved pulse trains", fontsize=16)
ax[1].set_xlim(10,15)
ax[1].set_ylabel(r'Stimulus in $\mu$A', fontsize=14)
ax[1].set_xlabel("time in ms", fontsize=14)

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper right')
ax2.legend(loc='upper right')

plt.show()


# # # Single neuron version
# def simulate_hodgkin_huxley(input_signal):
#     # Constants
#     Cm = 1.0  # Membrane capacitance (uF/cm^2)
#     V_Na = 50.0  # Sodium Nernst potential (mV)
#     V_K = -77.0  # Potassium Nernst potential (mV)
#     V_L = -54.4  # Leak Nernst potential (mV)

#     G_Na = 120.0  # Sodium conductance (mS/cm^2)
#     G_K = 36.0  # Potassium conductance (mS/cm^2)
#     G_L = 0.3  # Leak conductance (mS/cm^2)

#     # Time parameters
#     dt = 0.01  # Time step (ms)
#     t_max = 100.0  # Maximum simulation time (ms)
#     num_steps = int(t_max / dt)

#     # Initialize variables
#     t = np.arange(0, t_max, dt)
#     Vm = np.zeros(num_steps)  # Membrane potential (mV)
#     m = np.zeros(num_steps)  # Sodium activation gate
#     h = np.zeros(num_steps)  # Sodium inactivation gate
#     n = np.zeros(num_steps)  # Potassium activation gate

#     # Initial conditions
#     Vm[0] = -65.0  # Resting membrane potential (mV)
#     m[0] = 0.05  # Initial sodium activation
#     h[0] = 0.6  # Initial sodium inactivation
#     n[0] = 0.32  # Initial potassium activation

#     # Simulation loop
#     for i in range(1, num_steps-1):
#         # Hodgkin-Huxley equations
#         alpha_m = (2.5 - 0.1 * Vm[i - 1]) / (np.exp(2.5 - 0.1 * Vm[i - 1]) - 1)
#         beta_m = 4 * np.exp(-Vm[i - 1] / 18)
#         alpha_h = 0.07 * np.exp(-Vm[i - 1] / 20)
#         beta_h = 1 / (np.exp(3 - 0.1 * Vm[i - 1]) + 1)
#         alpha_n = (0.1 - 0.01 * Vm[i - 1]) / (np.exp(1 - 0.1 * Vm[i - 1]) - 1)
#         beta_n = 0.125 * np.exp(-Vm[i - 1] / 80)

#         m[i] = m[i - 1] + dt * (alpha_m * (1.0 - m[i - 1]) - beta_m * m[i - 1])
#         h[i] = h[i - 1] + dt * (alpha_h * (1.0 - h[i - 1]) - beta_h * h[i - 1])
#         n[i] = n[i - 1] + dt * (alpha_n * (1.0 - n[i - 1]) - beta_n * n[i - 1])

#         I_Na = G_Na * m[i] ** 3 * h[i] * (Vm[i - 1] - V_Na)
#         I_K = G_K * n[i] ** 4 * (Vm[i - 1] - V_K)
#         I_L = G_L * (Vm[i - 1] - V_L)

#         # Update membrane potential using the equation Cm * dV/dt = I_stim - I_Na - I_K - I_L
#         Vm[i] = Vm[i - 1] + (dt / Cm) * (input_signal[i] - I_Na - I_K - I_L)

#     return Vm, t


# # Number of channels
# num_channels = final_stim.shape[1]

# # Create subplots with twinx() for secondary y-axis
# fig, axs = plt.subplots(num_channels, 1, figsize=(8, 2*num_channels), sharex=True)

# # Plot each channel with two y-axes
# for channel in range(num_channels):
#     # Create a twin axes for the secondary y-axis
#     ax1 = axs[channel]
#     ax2 = ax1.twinx()

#     # Plot Vm on the primary y-axis (left)
#     Vm, t = simulate_hodgkin_huxley(final_stim[:, channel])
#     ax1.plot(t, Vm, 'b', label='Vm')
#     ax1.set_ylabel(f'Channel {channel - 1}')
#     ax1.set_title(f'Channel {channel - 1}')
#     ax1.set_xlim(0,100)

#     # Plot final_stim on the secondary y-axis (right)
#     ax2.plot(t, final_stim[:, channel], 'r', label='final_stim')
#     ax2.set_ylabel('final_stim')

#     # Add legends for both y-axes
#     ax1.legend(loc='upper left')
#     ax2.legend(loc='upper right')

# # Set common x-axis label
# axs[-1].set_xlabel('Time (s)')

# # Adjust spacing between subplots
# plt.tight_layout()

# # Plot the results
# plt.figure()

# # First subplot (Vm)
# plt.subplot(2, 1, 1)
# plt.plot(t, Vm)
# plt.ylabel('Vm in mV')
# plt.title('CIS Stimulation response in a single Neuron - 4')
# plt.xticks(np.arange(0,101,5))
# plt.xlim(0,100)
# # Create a second y-axis for the stimulation current
# plt2 = plt.twinx()

# # Plot the stimulation current for channel 4 on the second y-axis
# line1, = plt2.plot(t, final_stim[:, 4], ':', linewidth=2, color='red', label='channel 4 (Stim)')
# plt2.set_ylabel('Stimulus in μA')

# # Second subplot (Stimulus for multiple channels)
# plt.subplot(2, 1, 2)
# plt.plot(t, final_stim[:, 4], ':', linewidth=2, label='channel 4')
# plt.plot(t, final_stim[:, 5], ':', linewidth=2, label='channel 5')
# plt.xlim(8, 15)
# plt.legend(['channel 4', 'channel 5'])
# plt.xlabel('Time in ms')
# plt.ylabel('Stimulus in μA')
# plt.xticks(np.arange(10,15,0.5))

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

# # # # Multi-neuron version
# # # neurons = 100

# # # # Initialize Vm and t for multiple neurons
# # # Vm = np.zeros((len(final_stim), neurons, n2))
# # # t = np.zeros(len(final_stim))

# # # # Determine the desired size of the window
# # # window_size = n2

# # # # Only a simple triangle for the radiation behavior
# # # window = np.concatenate((np.arange(0, window_size) / window_size, np.arange(window_size - 2, -1, -1) / window_size))


# # # # Weighting matrix for radiation behavior (windows per electrode)
# # # weighting_matrix = np.zeros((n2, neurons))

# # # # Index for the middle of the window
# # # middle_idx = np.round(np.linspace(len(window) / 2, neurons - len(window) / 2, n2)).astype(int)

# # # # Set the triangles in the right place in the matrix
# # # for idx in range(n2):
# # #     idx1 = middle_idx[idx] - len(window) // 2
# # #     idx2 = middle_idx[idx] + len(window) // 2
# # #     weighting_matrix[idx, idx1:idx2 + 1] = window

# # # # Matrix multiplication (tricky: dimensions/order! only this works!)
# # # final_stim = np.dot(weighting_matrix.T, final_stim.T).T


# # # # Initialize Vm and t
# # # Vm = np.zeros((len(final_stim), neurons))
# # # t = np.zeros(len(final_stim))

# # # for idx in range(n2):
# # #     Vm[:, idx], t = simulate_hodgkin_huxley(final_stim[idx, :])  

# # # # # Plot the results
# # # plt.figure(2)
# # # extent = [0, t[-1], 0, neurons]  # Set the extent based on the available data
# # # plt.imshow(Vm.T, extent=extent, aspect='auto', cmap='viridis')
# # # plt.yticks(range(0, 101, 5), range(500, 4001, 175))
# # # plt.ylabel('Frequency in Hz')
# # # plt.xlabel('Time in ms')
# # # plt.title('CIS Stimulation response in multiple Neurons')
# # # plt.colorbar(label='Vm in mV')

# # # # Adds horizontal dots
# # # for idx in middle_idx:
# # #     plt.plot([0, t[-1]], [idx, idx], 'w:', linewidth=1.5)


# # # plt.ylim(30, 55)

# # # plt.show()

# # # # Plot the weighting matrix
# # # # plt.figure(3, figsize=(6, 6))  
# # # # plt.imshow(weighting_matrix, cmap='viridis', aspect='auto', extent=[0, 100, 0, 100])  # Set extent to control axis limits
# # # # plt.xlabel('Neurons')
# # # # plt.ylabel('Electrode channel')
# # # # plt.title('Weighting matrix')
# # # # plt.colorbar()
# # # # plt.show()