import numpy as np

Nx = 4
Ny = 4
Nz = 2
Nz_bar = 2
p_x = 1 / Nx * np.ones(Nx)
pe = 0.3
pe1 = 0.4
convg_param = 10 ** -4
py_x = np.ones((Ny, Nx))
for i in range(len(py_x)):
    for j in range(len(py_x)):
        if i == j:
            py_x[i, j] = 1 - 3 * pe
        else:
            py_x[i, j] = pe

p_x_y = py_x * p_x
p_y = np.sum(p_x_y, 1)
px_y = py_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))
px_y_expanded = np.tile(np.expand_dims(px_y, axis=0), (Nz_bar, 1, 1))
pzbar_z = np.ones((Nz_bar, Nz))
for idx1 in range(len(pzbar_z)):
    for idx2 in range(len(pzbar_z)):
        if idx1 == idx2:
            pzbar_z[idx1, idx2] = 1 - pe1
        else:
            pzbar_z[idx1, idx2] = pe1
C = np.ones((Nz_bar, Ny))
Cm1 = 10
while True:
    temp_z = np.argmin(np.sum(
        np.tile(np.expand_dims(C, axis=1), (1, Nz, 1)) * np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)),
        0), axis=0)
    pz_y = np.zeros((Nz, Ny))
    for idx3 in range(Nz):
        for idx4 in range(len(temp_z)):
            if idx3 == temp_z[idx4]:
                pz_y[idx3, idx4] = 1
            else:
                pz_y[idx3, idx4] = 0

    pzbar_y = np.sum(np.tile(np.expand_dims(pzbar_z, axis=2), (1, 1, Ny)) * np.tile(np.expand_dims(pz_y, axis=0),
                                                                                    (Nz_bar, 1, 1)), axis=1)
    p_zbar = np.sum(pzbar_y * p_y, axis=1)

    px_zbar = (1 / np.tile(np.expand_dims(p_zbar, axis=1), (1, Nx))) * np.sum(np.tile(np.expand_dims(p_x_y, axis=0),
                                                                                   (Nz_bar, 1, 1)) * np.tile(
        np.expand_dims(pzbar_y, axis=2), (1, 1, Nx)), axis=1)
    px_zbar_expanded = np.tile(np.expand_dims(px_zbar, axis=1), (1, Ny, 1))
    C_updated = np.sum((np.log2(px_y_expanded + 1e-31) - np.log2(px_zbar_expanded + 1e-31)) * px_y_expanded, 2)
