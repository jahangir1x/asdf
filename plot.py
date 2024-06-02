import numpy as np
import matplotlib.pyplot as plt


def convolution(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    for n in range(len(y)):
        for k in range(len(y)):
            if n - k >= 0 and n - k < len(h) and k < len(x):
                y[n] += x[k] * h[n - k]
    return y


x = np.array([1, 2, 3, 4, 5])
h = np.array([3, 2, 1])

y = convolution(x, h)

n_x = np.arange(len(x))
n_h = np.arange(len(h))
n_y = np.arange(len(y))

# fig, axs = plt.subplots(1, 4, figsize=(20, 5))
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].stem(n_x, x)
axs[0].set_xlabel("n")
axs[0].set_ylabel("x(n)")
axs[0].set_title("x(n)")
axs[0].grid(True)

axs[1].stem(n_h, h)
axs[1].set_xlabel("n")
axs[1].set_ylabel("h(n)")
axs[1].set_title("h(n)")
axs[1].grid(True)

axs[2].stem(n_y, y)
axs[2].set_xlabel("n")
axs[2].set_ylabel("y(n)")
axs[2].set_title("y(n)")
axs[2].grid(True)

# y_builtin = np.convolve(x, h, mode="full")
# n_y_builtin = np.arange(len(y_builtin))
# axs[3].stem(n_y_builtin, y_builtin)
# axs[3].set_xlabel("n")
# axs[3].set_ylabel("y(n)")
# axs[3].set_title("y(n) with np.convolve")
# axs[3].grid(True)

plt.tight_layout()
plt.show()
