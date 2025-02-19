import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrix, N):
    assert N * N == matrix.shape[0]
    plt.xlim(0,N)
    plt.ylim(0,N)
    plt.xticks(np.arange(0,N + N/10, N/10), np.arange(-5,6,1) / 10)
    plt.yticks(np.arange(0,N + N/10, N/10), np.arange(-5,6,1) / 10)
    plt.imshow(matrix.reshape((N,N)), cmap = 'gnuplot2')
    # plt.colorbar(fraction = 0.045, pad = 0.05)
    plt.show()

matrix = np.loadtxt("matrix.txt")
plot_matrix(matrix, 1000)
