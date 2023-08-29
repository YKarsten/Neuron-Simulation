import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# channel to plot 
x = 4

# built signal
sampling_freq = 100_000
f = 1000
f2 = 1200
duration = 100e-3
t = np.arange(0, int(sampling_freq * duration)) / sampling_freq
signal = np.sin(2 * np.pi * t * f)
signal2 = np.sin(2 * np.pi * t * f2)
mod_sin = (1 - np.cos(2 * np.pi * t * 50))
mod_sin2 = (1 - np.cos(2 * np.pi * t * 100))
signal = signal2 * mod_sin2 + signal * mod_sin

## Cochlea Implant Stimulus

# middle ear
def middle_ear(signal, sampling_freq):
    # low order: broader filter
    order = 1
    
    # Cut-off frequencies
    # lower limit = 500 Hz; upper limit = 4000 Hz
    fcut = np.array([500, 4000]) / (sampling_freq / 2)
    
    # bandpass filter
    b, a = butter(order, fcut, btype='band')
    output = filtfilt(b, a, signal)
    
    return output

signal = middle_ear(signal, sampling_freq)


# basilar membrane/ filterbank
def basilar_membrane(signal, sampling_freq):
    n = np.arange(-4, 8)
    terz_mid_freq = 1000 * 2**(n / 3)

    # Upper cut-off frequency
    f_o = terz_mid_freq[:-1]

    # Lower cut-off frequency
    f_u = terz_mid_freq[1:]

    f_grenz = np.sqrt((f_o * f_u))

    terz_mid_freq = terz_mid_freq[1:-1]
    order = 2

    # allocate output matrix
    output = np.zeros((len(signal), 10))

    for idx, terz_freq in enumerate(terz_mid_freq):
        fcut = np.array([f_grenz[idx], f_grenz[idx + 1]]) / (sampling_freq / 2)

        b, a = butter(order, fcut, btype='band')

        output[:, idx] = filtfilt(b, a, signal)

    return output

output_matrix = basilar_membrane(signal, sampling_freq)

# hair cell
def hair_cell(signal, sampling_freq):
    compress_power = 1
    order = 4 # changed from 4
    # Tiefpass 200 Hz
    cutoff = 200

    # Halbwellen Gleichrichtung. Negative Werte werden = 0 gesetzt.
    rect = np.maximum(signal, 0)

    # compression
    compressed_signal = rect ** compress_power

    # lowpass filtering
    b, a = butter(order, cutoff / (sampling_freq / 2), btype='low')

    # Das wird später als Strom für die Neuron Modelle genutzt. I_stim
    # Stark vereinfacht, denn im Grunde ist das ja ein akustisches Signal

    envelope = filtfilt(b, a, compressed_signal, padlen = 0)

    return envelope

envelope = hair_cell(output_matrix, sampling_freq)


plt.figure()
plt.plot(output_matrix[:, x-1])
plt.title('basilar membrane')

plt.figure()
plt.plot(envelope[:, x-1])
plt.title('hair cell')

## electric signal

# electric pulse/ kernel
pulse_duration = 200e-6
pps = 800

sample = int(sampling_freq * pulse_duration)
kernel = np.ones(sample)
kernel[sample // 2:] = -1

# time between pulses
inter_sample = sampling_freq / pps
pulse_train = np.zeros(int(1e4))
pulse_train[np.arange(int(inter_sample), int(1e4), int(inter_sample))] = 1

stim_speicher = np.zeros((len(np.convolve(envelope[:, 0] * pulse_train, kernel)), 10))
envelope_pulsed_speicher = np.zeros_like(envelope)

for n in range(envelope.shape[1]):
    delay = (1 / pps) * (n / envelope.shape[1])
    sample_delay = round(delay * sampling_freq)
    envelope_pulsed = envelope[:, n] * pulse_train

    weighted_pulse = np.convolve(envelope_pulsed, kernel)
    weighted_pulse = np.concatenate((np.zeros(sample_delay), weighted_pulse))
    weighted_pulse = weighted_pulse[:len(weighted_pulse) - sample_delay]

    stim_speicher[:len(weighted_pulse), n] = weighted_pulse
    envelope_pulsed_speicher[:, n] = envelope_pulsed

## visualization

plt.subplot(4, 1, 1)
plt.plot(signal)
plt.title('Modified sinusoid signal', fontsize=14)

plt.subplot(4, 1, 2)
plt.plot(output_matrix[:, x-1])
plt.title('Signal - Bandpass filtered', fontsize=14)

plt.subplot(4, 1, 3)
plt.plot(envelope[:, x-1])
plt.title('Envelope extraction', fontsize=14)

plt.subplot(4, 1, 4)
plt.plot(envelope[:, x-1])
plt.title('Modulated pulse trains', fontsize=14)
plt.plot(envelope_pulsed_speicher[:, x-1])
plt.xlabel('time [ms]', fontsize=16)

## Stimulus in HH-Model

final_stim = stim_speicher * 700

# Placeholder for demonstration
t = np.arange(len(final_stim))
Vm = np.random.rand(len(final_stim), x)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('modified sinus signal')
plt.plot(envelope_pulsed_speicher[:, x-1])

plt.subplot(2, 1, 2)
plt.plot(t, Vm)
plt.title('HH-Model simulation')
plt.xlim([1, 100])

## single neuron version

# Placeholder for demonstration
t = np.arange(len(final_stim[:, 3]))
Vm = np.random.rand(len(final_stim[:, 3]))

plt.figure()
plt.plot(t, Vm, 'k-', linewidth=2)
plt.plot(t, final_stim[:, 3], ':', color=[0.4, 0.4, 0.04], linewidth=1)
# plt.ylim([-20, 120])
# plt.xlim([0, 100])
plt.xlabel('Time in ms')
plt.ylabel('Vm in mV')
plt.legend(['V_m', 'Stimulus'])
plt.show()
