import numpy as np
import matplotlib.pyplot as plt

SNR_db = 25  # SNR in db
Nx = 4  # cardinality of source signal
Ny = 64  # cardinality of quantizer input
Nz = 16  # cardinality of quantizer output
alphabet = np.arange(-(Nx - 1), (Nx + 1), 2)
p_x = 1 / Nx * np.ones(Nx)  # p(x)
# AWGN Channel
SNR_lin = 10 ** (SNR_db / 10)  # SNR in linear scale
var_X = np.dot(alphabet ** 2, p_x)  # Variance of signal
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
convergence_param = 10 ** -4
I_x_y = np.sum(
    p_x_y * (np.log2(p_x_y + 1e-31) - np.tile(np.expand_dims(np.log2(np.tile(p_x + 1e-31, Ny // Nx) * p_y), axis=1),
                                              (1, Nx))))
pz_y = np.zeros((Nz, Ny))
for k in range(0, Ny):
    pz_y[np.int(np.floor(np.random.rand() * Nz)), k] = 1
count = 0
beta = 170
Ixz = []
while True:
    p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
    pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
    p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
    px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, Nx)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
    px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
    px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
    KL = np.sum((np.log2(px_y_expanded) - np.log2(px_z_expanded)) * px_y_expanded, 2)  # KL divergence
    exponential_term = np.exp(-(beta * KL))
    numerator = np.tile(np.expand_dims(p_z, axis=1), (1, Ny)) * exponential_term
    denominator = np.sum(numerator, 0)
    pz_y1 = numerator / denominator
    pz_y1[np.isnan(pz_y1)] = 0
    pi = [0.5, 0.5]
    p = pi[0] * pz_y1 + pi[1] * pz_y
    KL1 = np.sum((np.log2(pz_y1 + 1e-31) - np.log2(p + 1e-31)) * pz_y1)
    KL2 = np.sum((np.log2(pz_y + 1e-31) - np.log2(p + 1e-31)) * pz_y)
    JS = pi[0] * KL1 + pi[1] * KL2  # JS Divergence
    if JS <= convergence_param and count == 20:
        break
    else:
        px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, Nx)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
        p_x_z = px_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
        w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
        w1 = np.log2(p_x_z) - np.log2(w)
        I_x_z = np.sum(p_x_z * w1)
        Ixz.append(I_x_z)
        pz_y = pz_y1
        count = count + 1
with open('IB.txt', 'w') as f:
    for item in Ixz:
        f.write("{}\n".format(item))
