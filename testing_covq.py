import numpy as np
import matplotlib.pyplot as plt
import math

Nx = [4, 8]  # cardinality of source signal
Ny = 64  # cardinality of quantizer input
Nz = 16  # cardinality of quantizer output
Nz_bar = 16  # cardinality of DMC channel
SNR_db = np.arange(-20, 31, 1)
for idx7 in range(len(Nx)):
    Ixy = []
    Ixzbar = []
    for idx8 in range(len(SNR_db)):
        alphabet = np.arange(-(Nx[idx7] - 1), (Nx[idx7] + 1), 2)
        p_x = 1 / Nx[idx7] * np.ones(Nx[idx7])  # p(x)
        # AWGN Channel
        SNR_lin = 10 ** (SNR_db[idx8] / 10)  # SNR in linear scale
        var_X = np.dot(alphabet ** 2, p_x)
        var_N = var_X / SNR_lin
        sigma_N_2 = np.sqrt(var_N / 2)
        n = (np.random.randn(np.size(alphabet)) + 1j * np.random.randn(np.size(alphabet))) * sigma_N_2
        # sampling interval
        dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(var_N)) / Ny
        # discretized y
        y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(var_N)), (np.amax(alphabet) + 5 * np.sqrt(var_N)), dy).reshape(
            Ny, 1)
    
        py_x = np.ones((Ny, Nx[idx7]))  # initialized p(y|x)
        tmp = np.zeros((Ny, 1))  # initialize the temporary matrix to store conditional pdf
    
        for i in range(0, Nx[idx7]):
            v = (1 / np.sqrt(2 * np.pi * var_N)) * np.exp(-(y - alphabet[i]) ** 2 / (2 * var_N))
            tmp = np.append(tmp, v, axis=1)
    
        py = tmp[:, -Nx[idx7]:]  # remove the zero from the first column of the matrix
        py_x = py_x * py  # obtain the conditional pdf
    
        norm_sum = np.sum(py_x, 0)  # normalization for the pdf
        py_x = py_x / np.tile(norm_sum, (Ny, 1))  # p(y|x)
        p_x_y = py_x * p_x  # p(x,y) joint probability of x and y
        p_x_y_expanded = np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1))
        p_y = np.sum(p_x_y, 1)  # p(y)
        px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx[idx7]) / p_y, axis=1), (1, Nx[idx7]))  # Bayes rule
        px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz, 1, 1))
        I_x_y = np.sum(
            p_x_y * (np.log2(p_x_y + 1e-31) - np.tile(
                np.expand_dims(np.log2(np.tile(p_x + 1e-31, Ny // Nx[idx7]) * p_y), axis=1),
                (1, Nx[idx7]))))
        convg_param = 10 ** -4
        SNR_dmc = 20
        SNR_dmc_lin = 10 ** (SNR_dmc / 10)
        alphabet2 = np.arange(-(Nz - 1), Nz + 1, 2)
        alphabet3 = np.arange(-(Nz_bar - 1), Nz_bar + 1, 2)
        p_z_dmc = 1 / Nz * np.ones(Nz)
        SNR_dmc_lin_norm = SNR_dmc_lin / (np.dot(alphabet2 ** 2, p_z_dmc))
        pzbar_z = np.ones((Nz_bar, Nz))
        for idx1 in range(len(pzbar_z)):
            for idx2 in range(len(pzbar_z)):
                if idx1 != idx2:
                    pzbar_z[idx1, idx2] = 0.5 * math.erfc(
                        np.sqrt((((alphabet3[idx1] - alphabet2[idx2]) / 2) ** 2) * SNR_dmc_lin_norm))
        for idx6 in range(len(pzbar_z)):
            for idx5 in range(len(pzbar_z[idx6])):
                if idx5 == idx6:
                    allElementsInRow = pzbar_z[:, idx6]
                    allElementsWithoutDiagonal = np.delete(allElementsInRow, [idx6])
                    pzbar_z[idx5, idx6] = pzbar_z[idx5, idx6] - np.sum(allElementsWithoutDiagonal)
        KL = np.zeros((Nz_bar, Ny))
        Co = 100
        count = 0
        eff = 100
        while True:
            pz_y = np.zeros((Nz, Ny))
            for k in range(0, Nz):
                temp = np.arange((k * np.floor(Ny / Nz)), min(((k + 1) * np.floor(Ny / Nz)), Ny), dtype=int)
                pz_y[k, temp] = 1
            if temp[-1] < Ny:
                pz_y[k, temp[-1] + 1:] = 1
    
            pzbar_y = np.sum(
                np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)) * np.tile(np.expand_dims(pz_y, axis=0),
                                                                               (Nz_bar, 1, 1)), axis=1)
            pzbar_y_expanded = np.tile(np.expand_dims(pzbar_y, axis=2), (1, 1, Nx[idx7]))
            p_zbar = np.sum(pzbar_y * p_y, axis=1)
            px_zbar = np.tile(np.expand_dims(p_zbar**-1, axis=1), (1, Nx[idx7])) * np.sum(p_x_y_expanded * pzbar_y_expanded, 1)
            px_zbar_expanded = np.tile(np.expand_dims(px_zbar, axis=1), (1, Ny, 1))
            KL = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_zbar_expanded + 1e-31)) * px_y_expanded, 2)
            Cm = np.sum(p_y) * np.sum(pzbar_y * KL, 0)
            eff = (Co - Cm) / Cm
            if np.all(eff) <= convg_param and count == 20:
                p_x_zbar = px_zbar * np.tile(np.expand_dims(p_zbar, axis=1), (1, Nx[idx7]))
                w = np.tile(np.expand_dims(p_x, axis=0), (Nz_bar, 1)) * np.tile(np.expand_dims(p_zbar, axis=1),
                                                                            (1, Nx[idx7]))
                w1 = np.log2(p_x_zbar) - np.log2(w)
                I_x_zbar = np.sum(p_x_zbar * w1)
                Ixzbar.append(I_x_zbar)
                Ixy.append(I_x_y)
                break
            else:
                Co = Cm
                count = count + 1
    with open('IXZbar_' + str(Nx[idx7]) + '.txt', 'w') as f:
        for item in Ixzbar:
            f.write("{}\n".format(item))
