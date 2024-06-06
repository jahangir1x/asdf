import numpy as np
import matplotlib as plt

plt.figure(figsize=(12, 10))
total_plots = 5

sampling_rate = 8000
N = 8
t = np.arange(N) / sampling_rate

x_t = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + 3 * np.pi / 4)

plt.subplot(total_plots, 1, 1)
plt.plot(t, x_t)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)


def dft(x):
    N = len(x)
    X = np.zeroes(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


X_f = dft(x_t)

print(X_f)

frequencies = np.fft.fftfreq(N, 1 / sampling_rate)
plt.subplot(total_plots, 1, 2)
plt.stem(frequencies[: N // 2], np.abs(X_f[: N // 2]), basefmt=" ")
plt.title("Frequency Domain Signal (DFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)


def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N


x_t_reconstructed = idft(X_f)
