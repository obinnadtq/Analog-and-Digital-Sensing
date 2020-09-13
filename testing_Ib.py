import numpy as np
import matplotlib.pyplot as plt

Nx = [4, 8]
Ny = 64
Nz = 16
SNR_db = np.arange(-20, 31, 1)
for idx7 in range(len(Nx)):
    Ixz = []
    Ixy = []
    for index in range(len(SNR_db)):
        alphabet = np.arange(-(Nx[idx7] - 1), (Nx[idx7] + 1), 2)
        p_x = 1 / Nx[idx7] * np.ones(Nx[idx7])
        SNR_lin = 10 ** (SNR_db[index] / 10)
        var_X = np.dot(alphabet ** 2, p_x)
        var_N = var_X / SNR_lin
        sigma_N_2 = np.sqrt(var_N / 2)
        n = (np.random.randn(np.size(alphabet)) + 1j * np.random.randn(np.size(alphabet))) * sigma_N_2
        dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(var_N)) / Ny
        y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(var_N)), (np.amax(alphabet) + 5 * np.sqrt(var_N)), dy).reshape(
            Ny, 1)
        py_x = np.ones((Ny, Nx[idx7]))
        tmp = np.zeros((Ny, 1))
        for i in range(0, Nx[idx7]):
            v = (1 / np.sqrt(2 * np.pi * var_N)) * np.exp(-(y - alphabet[i]) ** 2 / (2 * var_N))
            tmp = np.append(tmp, v, axis=1)
        py = tmp[:, -Nx[idx7]:]
        py_x = py_x * py
        norm_sum = np.sum(py_x, 0)
        py_x = py_x / np.tile(norm_sum, (Ny, 1))
        p_x_y = py_x * p_x
        p_y = np.sum(p_x_y, 1)
        px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx[idx7]) / p_y, axis=1), (1, Nx[idx7]))
        convergence_param = 10 ** -4
        I_x_y = np.sum(
            p_x_y * (np.log2(p_x_y + 1e-31) - np.tile(
                np.expand_dims(np.log2(np.tile(p_x + 1e-31, Ny // Nx[idx7]) * p_y), axis=1), (1, Nx[idx7]))))
        Ixy.append(I_x_y)
        H_y = -np.sum(p_y * np.log2(p_y))
        pz_y = np.zeros((Nz, Ny))
        for k in range(0, Nz):
            temp = np.arange((k * np.floor(Ny / Nz)), min(((k + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
            pz_y[k, temp] = 1
        if temp[-1] < Ny:
            pz_y[k, temp[-1] + 1:] = 1
        p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
        pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx[idx7]))
        p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))
        px_z = np.tile(np.expand_dims(1 / (p_z + 1e-31), axis=1), (1, Nx[idx7])) * np.sum(
            pz_y_expanded * p_x_y_expanded, 1)
        px_z_expanded = np.tile(np.expand_dims(px_z, axis=1), (1, Ny, 1))
        px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
        count = 0
        beta = 50
        while True:
            KL = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_z_expanded + 1e-31)) * px_y_expanded,
                        2)
            exponential_term = np.exp(-(beta * KL))
            numerator = np.tile(np.expand_dims(p_z + 1e-31, axis=1), (1, Ny)) * exponential_term
            denominator = np.sum(numerator, 0)
            pz_y1 = numerator / denominator
            pz_y1[np.isnan(pz_y1)] = 0
            p_z1 = np.sum(p_y * pz_y1, 1)
            g = np.tile(np.expand_dims(pz_y1, axis=2), (1, 1, Nx[idx7])) * p_x_y_expanded
            g1 = np.sum(g, 1) / np.tile(np.expand_dims(p_z1 + 1e-31, axis=1), (1, Nx[idx7]))
            px_z1 = g1
            px_z_new_3D = np.tile(np.expand_dims(px_z1, axis=1), (1, Ny, 1))
            pi = [0.5, 0.5]
            p = pi[0] * pz_y1 + pi[1] * pz_y
            KL1 = np.sum((np.log2(pz_y1 + 1e-31) - np.log2(p + 1e-31)) * pz_y1)
            KL2 = np.sum((np.log2(pz_y + 1e-31) - np.log2(p + 1e-31)) * pz_y)
            JS = pi[0] * KL1 + pi[1] * KL2
            if JS <= convergence_param:
                p_x_z = px_z1 * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx[idx7]))
                w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z1, axis=1), (1, Nx[idx7]))
                w1 = np.log2(p_x_z + 1e-31) - np.log2(w + 1e-31)
                I_x_z = np.sum(p_x_z * w1)
                Ixz.append(I_x_z)
                break
            else:
                pz_y = pz_y1

    with open('IXY' + str(Nx[idx7]) + '.txt', 'w') as f:
        for item in Ixy:
            f.write("{}\n".format(item))

    with open('IXZ' + str(Nx[idx7]) + '.txt', 'w') as f:
        for item in Ixz:
            f.write("{}\n".format(item))
