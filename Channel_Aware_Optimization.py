import numpy as np
import matplotlib.pyplot as plt
SNR_db = 1  # SNR in db
Nx = 8  # cardinality of source signal
Ny = 64  # cardinality of quantizer input
Nz = 16  # cardinality of quantizer output
Nz_bar = 16  # cardinality of DMC channel
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
p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))
p_y = np.sum(p_x_y, 1)  # p(y)
px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))  # Bayes rule
px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
pz_y = np.zeros((Nz, Ny))
for k in range(0, Nz):
    temp = np.arange((k * np.floor(Ny / Nz)), min(((k + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
    pz_y[k, temp] = 1
if temp[-1] < Ny:
    pz_y[k, temp[-1] + 1:] = 1
I_x_y = np.sum(
    p_x_y * (np.log2(p_x_y) - np.tile(np.expand_dims(np.log2(np.tile(p_x, Ny // Nx) * p_y), axis=1), (1, Nx))))
pe = 0.01
convg_param = 10 ** -4
pzbar_z = np.ones((Nz_bar, Nz))
for idx1 in range(len(pzbar_z)):
    for idx2 in range(len(pzbar_z)):
        if idx1 == idx2:
            pzbar_z[idx1, idx2] = 1 - (Nz - 1) * pe
        else:
            pzbar_z[idx1, idx2] = pe
C = np.ones((Nz_bar, Ny))
count = 0
IXZ = []
while True:
    temp_z = np.argmin(np.sum(
        np.tile(np.expand_dims(C, axis=1), (1, Nz, 1)) * np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)),
        0), axis=0)
    pz_y1 = np.zeros((Nz, Ny))
    for idx3 in range(Nz):
        for idx4 in range(len(temp_z)):
            if idx3 == temp_z[idx4]:
                pz_y1[idx3, idx4] = 1
            else:
                pz_y1[idx3, idx4] = 0

    pzbar_y = np.sum(np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)) * np.tile(np.expand_dims(pz_y1, axis=0),
                                                                                    (Nz_bar, 1, 1)), axis=1)
    p_zbar = np.sum(pzbar_y * p_y, axis=1)

    px_zbar = (1 / np.tile(np.expand_dims(p_zbar, axis=1), (1, Nx))) * np.sum(np.tile(np.expand_dims(p_x_y, axis=0),
                                                                                   (Nz_bar, 1, 1)) * np.tile(
        np.expand_dims(pzbar_y, axis=2), (1, 1, Nx)), axis=1)
    px_zbar_expanded = np.tile(np.expand_dims(px_zbar, axis=1), (1, Ny, 1))
    C = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_zbar_expanded + 1e-31)) * px_y_expanded, 2)
    pi = [0.5, 0.5]
    p = pi[0] * pz_y1 + pi[1] * pz_y
    KL1 = np.sum((np.log2(pz_y1 + 1e-31) - np.log2(p + 1e-31)) * pz_y1)
    KL2 = np.sum((np.log2(pz_y + 1e-31) - np.log2(p + 1e-31)) * pz_y)
    JS = pi[0] * KL1 + pi[1] * KL2  # JS Divergence
    if JS <= convg_param:
        break
    else:
        p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y1, 1)
        pz_y_expanded = np.tile(np.expand_dims(pz_y1, axis=2), (1, 1, Nx))
        px_z = np.tile(np.expand_dims(1 / (p_z + 1e-31), axis=1), (1, Nx)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
        p_x_z = px_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
        w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
        w1 = np.log2(p_x_z + 1e-31) - np.log2(w + 1e-31)
        I_x_z = np.sum(p_x_z * w1)  # I(X;Z)
        IXZ.append(I_x_z)
        pz_y = pz_y1
    count = count + 1
plt.plot(range(count), IXZ, linewidth=2)
plt.grid()
plt.title('Relevant Information versus number of iterations')
plt.xlabel('iterations')
plt.ylabel('I(x;z)[bit]')
plt.show()
