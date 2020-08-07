import numpy as np
from IB_function import iterative_ib
import matplotlib.pyplot as plt

SNR_db = 6  # SNR in db
Nx = 8  # cardinality of source signal
Ny = 64  # cardinality of quantizer input
Nz = 16  # cardinality of quantizer output
alphabet = np.arange(-7, 9, 2)
p_x = 1 / Nx * np.ones(Nx)  # p(x)
# AWGN Channel
SNR_lin = 10 ** (SNR_db / 10)  # SNR in linear scale
var_X = np.dot((alphabet ** 2), p_x)  # Variance of signal
var_N = var_X / SNR_lin  # Variance of noise
sigma_N_2 = np.sqrt(var_N / 2)
n = (np.random.randn(np.size(alphabet)) + 1j * np.random.randn(np.size(alphabet))) * sigma_N_2

# sampling interval
dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(var_N)) / Ny

# discretized y
y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(var_N)), (np.amax(alphabet) + 5 * np.sqrt(var_N)), dy).reshape(
    Ny, 1)

py_x = np.ones((Ny, Nx))  # initialized p(y|x)
tmp = np.zeros((Ny, 1))  # initialize the temporary matrix to store conditional pdf

for i in range(0, Nx):
    v = (1 / np.sqrt(2 * np.pi * var_N)) * np.exp(-(y - alphabet[i]) ** 2 / (2 * var_N))
    tmp = np.append(tmp, v, axis=1)

py = tmp[:, -Nx:]  # remove the zero from the first column of the matrix
py_x = py_x * py  # obtain the conditional pdf

norm_sum = np.sum(py_x, 0)  # normalization for the pdf
py_x = py_x / np.tile(norm_sum, (Ny, 1))  # p(y|x)
p_x_y = py_x * p_x  # p(x,y) joint probability of x and y
p_y = np.sum(p_x_y, 1)  # p(y)
px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))  # Bayes rule

# Iterative IB Algorithm
convergence_param = 10 ** -4
I_x_y = np.sum(
    p_x_y * (np.log2(p_x_y) - np.tile(np.expand_dims(np.log2(np.tile(p_x, Ny // Nx) * p_y), axis=1), (1, Nx))))
H_y = -np.sum(p_y * np.log2(p_y))
pz_y = np.zeros((Nz, Ny))
for k in range(0, Nz):
    temp = np.arange((k * np.floor(Ny / Nz)), min(((k + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
    pz_y[k, temp] = 1
if temp[-1] < Ny:
    pz_y[k, temp[-1] + 1:] = 1
p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, Nx)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
count = 0
target_rate = 2
accuracy_bs = 1e-31
iter_bs = 0
residual_bs = 1
beta_min = 1
beta_max = 50
beta_values = []
Iyz = []
while residual_bs > accuracy_bs:
    beta_interval = np.array([beta_min, beta_max])
    beta = np.mean(beta_interval)
    while True:
        KL = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_z_expanded + 1e-31)) * px_y_expanded, 2)  # KL divergence
        exponential_term = np.exp(-(beta * KL))
        numerator = np.tile(np.expand_dims(p_z, axis=1), (1, Ny)) * exponential_term
        denominator = np.sum(numerator, 0)
        pz_y1 = numerator / denominator  # updated p(z|y)
        p_z1 = np.sum(p_y * pz_y1, 1)  # updated p(z)
        g = np.tile(np.expand_dims(pz_y1, axis=2), (1, 1, Nx)) * p_x_y_expanded
        g1 = np.sum(g, 1) / np.tile(np.expand_dims(p_z1 + 1e-31, axis=1), (1, Nx))
        px_z1 = g1  # updated p(x|z)
        px_z_new_3D = np.tile(np.expand_dims(px_z1, axis=1), (1, Ny, 1))
        pi = [0.5, 0.5]
        p = pi[0] * pz_y1 + pi[1] * pz_y
        KL1 = np.sum((np.log2(pz_y1 + 1e-31) - np.log2(p + 1e-31)) * pz_y1)
        KL2 = np.sum((np.log2(pz_y + 1e-31) - np.log2(p + 1e-31)) * pz_y)
        JS = pi[0] * KL1 + pi[1] * KL2  # JS Divergence
        if JS <= convergence_param:
            p_x_z = px_z1 * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx))
            w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx))
            w1 = np.log2(p_x_z + 1e-31) - np.log2(w + 1e-31)
            I_x_z = np.sum(p_x_z * w1)  # I(X;Z)
            p_y_z = pz_y1 * np.tile(np.expand_dims(p_y, axis=0), (Nz, 1))
            w2 = np.tile(np.expand_dims(p_y, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z1, axis=1), (1, Ny))
            w3 = np.log2(p_y_z + 1e-31) - np.log2(w2 + 1e-31)
            I_y_z = np.sum(p_y_z * w3)  # I(Y;Z)
            break
        else:
            p_z = p_z1
            pz_y = pz_y1
            px_z_expanded = px_z_new_3D
    if I_y_z < target_rate:
        beta_min = beta
    else:
        beta_max = beta
    residual_bs = np.abs(beta_max - beta_min)
    beta_values.append(beta)
    Iyz.append(I_y_z)
    count = count + 1
    if count > 200:
        break
plt.plot(range(count), beta_values, linewidth=2)
plt.grid()
plt.title('Lagrangian multiplier versus number of iterations')
plt.xlabel('number of iterations for bs')
plt.ylabel('lagrangian multiplier')
plt.show()

plt.plot(range(count), Iyz, linewidth=2)
plt.grid()
plt.title('Compression rate versus number of iterations')
plt.xlabel('number of iterations for bs')
plt.ylabel('I(Y;Z)')
plt.show()
