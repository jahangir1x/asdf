import numpy as np
import matplotlib.pyplot as plt

fs = 8000
N = 8
t = np.arange(N) / fs

X_t = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(
    2 * np.pi * 2000 * t + (3 * np.pi / 4)
)

print(len(X_t))
print(X_t)


def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N


X_f = DFT(X_t)


magnitude_spectrum = np.abs(X_f)
X_t_reconstructed = IDFT(X_f)

print(X_t_reconstructed)
phase_spectrum = np.angle(X_f)

frequencies = np.fft.fftfreq(N, 1 / fs)

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(t, X_t, "o-")
plt.title("Time-Domain Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 2)
plt.stem(frequencies, magnitude_spectrum, "b")
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()


plt.subplot(3, 2, 3)
plt.plot(t, X_t_reconstructed.real, "o-")
plt.title("Reconstructed Time-Domain Signal from IDFT")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(3, 2, 4)
plt.stem(frequencies, phase_spectrum, "g")
plt.title("Phase Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [radians]")
plt.grid()

plt.tight_layout()
plt.show()
