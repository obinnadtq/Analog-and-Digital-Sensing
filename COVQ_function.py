import numpy as np
import matplotlib.pyplot as plt
import math
SNR_db = [8]
Nx = 4
Ny = 64
Nz = 8
Nz_bar = Nz
alphabet = np.arange(-(Nx-1), (Nx+1), 2)
p_x = 1 / Nx * np.ones(Nx)
MI = []
Ixz = []
Ixy = []
best = np.zeros(20).tolist()
for idx7 in range(len(SNR_db)):
    SNR_lin = 10 ** (SNR_db[idx7] / 10)
    var_X = np.dot(alphabet ** 2, p_x)
    var_N = var_X / SNR_lin
    sigma_N_2 = np.sqrt(var_N / 2)
    n = (np.random.randn(np.size(alphabet)) + 1j * np.random.randn(np.size(alphabet))) * sigma_N_2

    dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(var_N)) / Ny

    y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(var_N)), (np.amax(alphabet) + 5 * np.sqrt(var_N)), dy).reshape(
        Ny, 1)

    py_x = np.ones((Ny, Nx))
    tmp = np.zeros((Ny, 1))

    for i in range(0, Nx):
        v = (1 / np.sqrt(2 * np.pi * var_N)) * np.exp(-(y - alphabet[i]) ** 2 / (2 * var_N))
        tmp = np.append(tmp, v, axis=1)

    py = tmp[:, -Nx:]
    py_x = py_x * py
    norm_sum = np.sum(py_x, 0)  # normalization for the pdf
    py_x = py_x / np.tile(norm_sum, (Ny, 1))  # p(y|x)
    p_x_y = py_x * p_x  # p(x,y) joint probability of x and y
    p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))
    p_y = np.sum(p_x_y, 1)  # p(y)
    px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))  # Bayes rule
    px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
    pzbar_z = np.ones((Nz_bar, Nz))
    # pe = 0
    # for idx1 in range(len(pzbar_z)):
    #     for idx2 in range(len(pzbar_z)):
    #         if idx1 == idx2:
    #             pzbar_z[idx1, idx2] = 1 - (Nz - 1) * pe
    #         else:
    #             pzbar_z[idx1, idx2] = pe

    SNR_dmc = 20
    SNR_dmc_lin = 10 ** (SNR_dmc / 10)
    alphabet2 = np.arange(-(Nz - 1), Nz + 1, 2)
    alphabet3 = np.arange(-(Nz_bar - 1), Nz_bar + 1, 2)
    p_z = 1 / Nz * np.ones(Nz)
    SNR_dmc_lin_norm = SNR_dmc_lin / (np.dot(alphabet2 ** 2, p_z))
    error = []
    for idx1 in range(len(pzbar_z)):
        for idx2 in range(len(pzbar_z)):
            if idx1 != idx2:
                pzbar_z[idx1, idx2] = 0.5 * math.erfc(
                    np.sqrt((((alphabet3[idx1] - alphabet2[idx2]) / 2) ** 2) * SNR_dmc_lin_norm))
                error.append(pzbar_z[idx1, idx2])
    for j in range(len(pzbar_z)):
        for i in range(len(pzbar_z[j])):
            if i == j:
                allElementsInRow = pzbar_z[:, j]
                allElementsWithoutDiagonal = np.delete(allElementsInRow, [j])
                pzbar_z[i, j] = pzbar_z[i, j] - np.sum(allElementsWithoutDiagonal)


    I_x_y = np.sum(
        p_x_y * (np.log2(p_x_y + 1e-31) - np.tile(
            np.expand_dims(np.log2(np.tile(p_x + 1e-31, Ny // Nx) * p_y), axis=1),
            (1, Nx))))
    for index in range(0, 100):
        best_Ixzbar = []
        best_Ixz = []
        C = np.random.random((Nz_bar, Ny))
        Co = 1000
        convergence_params = 10 ** -4
        count = 0
        while True:
            ptr = np.argmin(np.sum(
                np.tile(np.expand_dims(C, axis=1), (1, Nz_bar, 1)) * np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)),
                0), axis=0)
            pz_y = np.zeros((Nz, Ny))
            pz_y[ptr, np.arange(ptr.size, dtype=int)] = 1
            pzbar_y = np.sum(
                np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)) * np.tile(np.expand_dims(pz_y, axis=0),
                                                                               (Nz_bar, 1, 1)), axis=1)
            p_zbar = np.sum(pzbar_y * p_y, axis=1)

            px_zbar = (1 / np.tile(np.expand_dims(p_zbar + 1e-31, axis=1), (1, Nx))) * np.sum(
                np.tile(np.expand_dims(p_x_y, axis=0),
                        (Nz_bar, 1, 1)) * np.tile(
                    np.expand_dims(pzbar_y, axis=2), (1, 1, Nx)), axis=1)
            px_zbar_expanded = np.tile(np.expand_dims(px_zbar, axis=1), (1, Ny, 1))
            C = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_zbar_expanded + 1e-31)) * px_y_expanded, 2)
            Cm = np.sum(p_y) * np.sum(pzbar_y * C, 0)
            eff = (Co - (Cm + 1e-31)) / (Cm + 1e-31)
            if np.all(eff) <= convergence_params and count == 20:
                break
            else:
                Co = Cm
                count = count + 1
                p_x_zbar = px_zbar * np.tile(np.expand_dims(p_zbar, axis=1), (1, Nx))
                w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_zbar, axis=1),
                                                                            (1, Nx))
                w1 = np.log2(p_x_zbar + 1e-31) - np.log2(w + 1e-31)
                I_x_zbar = np.sum(p_x_zbar * w1)
                p_z = np.sum(np.tile(p_y, (Nz, 1)) * pz_y, 1)
                pz_y_expanded = np.tile(np.expand_dims(pz_y, axis=2), (1, 1, Nx))  # p(z|y) expanded dimension
                px_z = np.tile(np.expand_dims(1 / (p_z + 1e-31), axis=1), (1, Nx)) * np.sum(pz_y_expanded * p_x_y_expanded, 1)
                p_x_z = px_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
                w = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))
                w1 = np.log2(p_x_z + 1e-31) - np.log2(w + 1e-31)
                I_x_z = np.sum(p_x_z * w1)
                best_Ixzbar.append(I_x_zbar)
                best_Ixz.append(I_x_z)
            if best_Ixzbar[-1] > best[-1]:
                best = best_Ixzbar
    plt.plot(range(count), best, '+-', label=str(SNR_db[idx7]) + ' dB')
plt.grid()
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('I(x;ẑ) (bits)')
plt.show()
    # Ixzbar_best = np.max(best_Ixzbar)
    # Ixz_best = np.max(best_Ixz)
    # MI.append(Ixzbar_best)
    # Ixy.append(I_x_y)
    # Ixz.append(Ixz_best)
# plt.plot(SNR_db, MI, '+-', label="I(x;ẑ)")
# plt.plot(SNR_db, Ixz, 'o-', label="I(x;z)")
# plt.plot(SNR_db, Ixy, 'x-', label="I(x;y)")
# plt.grid()
# plt.xlabel('SNR(dB)')
# plt.ylabel('Relevant Information (bits)')
# plt.legend()
# plt.show()
# with open('COVQ.txt', 'w') as f:
#     for item in MI:
#         f.write("{}\n".format(item))