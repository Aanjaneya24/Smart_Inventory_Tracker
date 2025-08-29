import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.special import erfc
# Convolutional Encoder
def convolutional_encode(data):
state = [0] * 7 # Adjusted to match polynomial length
encoded_data = []
g1 = [1, 0, 1, 1, 0, 1, 1] # Polynomial 171
g2 = [1, 1, 1, 0, 1, 1, 0] # Polynomial 133
for bit in data:
state.insert(0, bit)
state.pop()
output1 = np.mod(np.dot(g1, state), 2)
output2 = np.mod(np.dot(g2, state), 2)
encoded_data.extend([output1, output2])
return np.array(encoded_data)
# QPSK Modulation
def qpsk_modulate(data):
symbol_map = {
(0, 0): (1 + 1j),
(0, 1): (-1 + 1j),
(1, 0): (1 - 1j),
(1, 1): (-1 - 1j)
}
symbols = [symbol_map[tuple(data[i:i+2])] for i in range(0, len(data), 2)]
return np.array(symbols) / np.sqrt(2)
# AWGN Channel
def awgn(signal, snr_db):
snr_linear = 10**(snr_db / 10)
power_signal = np.mean(np.abs(signal) ** 2)
noise_power = power_signal / snr_linear
noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j *
np.random.randn(*signal.shape))
return signal + noise
# QPSK Demodulation
def qpsk_demodulate(symbols):
decisions = np.array([(np.real(s) > 0, np.imag(s) > 0) for s in symbols], dtype=int)
return decisions.flatten()
# Generate Eye Diagram
def plot_eye_diagram(signal, samples_per_symbol=10):
plt.figure(figsize=(8, 4))
for i in range(0, len(signal) - samples_per_symbol, samples_per_symbol):
plt.plot(signal[i:i + samples_per_symbol], 'b', alpha=0.5)
plt.title("Eye Diagram")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
# Simulating the SDR system
data = np.random.randint(0, 2, 100)
encoded_data = convolutional_encode(data)
modulated_signal = qpsk_modulate(encoded_data)
noisy_signal = awgn(modulated_signal, 15)
demodulated_data = qpsk_demodulate(noisy_signal)
# Plot Constellation Diagram
plt.figure(figsize=(6, 6))
plt.scatter(noisy_signal.real, noisy_signal.imag, color='blue', alpha=0.5)
plt.xlabel('In-Phase')
plt.ylabel('Quadrature')
plt.title('QPSK Constellation Diagram with Noise')
plt.grid()
plt.show()
# Generate Eye Diagram
plot_eye_diagram(np.real(noisy_signal))
# Input-Output Waveforms
plt.figure(figsize=(10, 4))
plt.plot(data[:50], label="Input Data", linestyle='dashed', marker='o')
plt.plot(demodulated_data[:50], label="Received Data", linestyle='dashed', marker='x')
plt.xlabel("Time")
plt.ylabel("Bit Value")
plt.title("Input vs. Output Waveform")
plt.legend()
plt.grid()
plt.show()