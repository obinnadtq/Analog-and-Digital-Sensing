import numpy as np
from IB_function import iterative_ib
import matplotlib.pyplot as plt

SNR_db = 3  # SNR in db
Nx = 4  # cardinality of source signal
Ny = 64  # cardinality of quantizer input
Nz = 32  # cardinality of quantizer output
alphabet = np.array([-3, -1, 1, 3])
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
Ixz = []
Iyz = []
LG = []
counter = 0
pz_y = np.zeros((Nz, Ny))
for k in range(0, Nz):
    temp = np.arange((k * np.floor(Ny / Nz)), min(((k + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
    pz_y[k, temp] = 1
if temp[-1] < Ny:
    pz_y[k, temp[-1] + 1:] = 1
p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))  # p(x,y) expanded dimension
px_z = np.tile(np.expand_dims(1 / p_z, axis=1), (1, 4)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
IIB = iterative_ib(p_x_y=p_x_y_expanded, px_y=px_y_expanded, p_y=p_y, px_z=px_z_expanded, p_z=p_z, pz_y=pz_y,
                   beta=50, Nx=Nx, Ny=Ny, Nz=Nz, convg=convergence_param, p_x=p_x)
