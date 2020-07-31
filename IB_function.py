import numpy as np
import matplotlib.pyplot as plt


def iterative_ib(p_x_y, px_y, p_y, px_z, pz_y, p_z, beta, Nx, Ny, Nz, convg, p_x):
    while True:
        kl = np.sum((np.log2(px_y + 1e-31) - np.log2(px_z + 1e-31)) * px_y, 2)
        exponential_term = np.exp(-(beta * kl))
        numerator = np.tile(np.expand_dims(p_z, axis=1), (1, Ny)) * exponential_term
        denominator = np.sum(numerator, 0)
        pz_y1 = numerator / denominator
        p_z1 = np.sum(p_y * pz_y1, 1)
        g = np.tile(np.expand_dims(pz_y1, axis=2), (1, 1, Nx)) * p_x_y
        g1 = np.sum(g, 1) / np.tile(np.expand_dims(p_z1 + 1e-31, axis=1), (1, 4))
        px_z1 = g1
        px_z_new_3D = np.tile(np.expand_dims(px_z1, axis=1), (1, Ny, 1))
        pi = [0.5, 0.5]
        p = pi[0] * pz_y1 + pi[1] * pz_y
        KL1 = np.sum((np.log2(pz_y1 + 1e-31) - np.log2(p + 1e-31)) * pz_y1)
        KL2 = np.sum((np.log2(pz_y + 1e-31) - np.log2(p + 1e-31)) * pz_y)
        JS = pi[0] * KL1 + pi[1] * KL2
        if JS <= convg:
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
            px_z = px_z_new_3D
    obinna = 5