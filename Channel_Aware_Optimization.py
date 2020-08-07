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
            py_x[i, j] = 1 - 3*pe
        else:
            py_x[i, j] = pe

p_x_y = py_x * p_x

pzbar_z = np.ones((Nz_bar, Nz))
for idx1 in range(len(pzbar_z)):
    for idx2 in range(len(pzbar_z)):
        if idx1 == idx2:
            pzbar_z[idx1, idx2] = 1 - pe1
        else:
            pzbar_z[idx1, idx2] = pe1
C = np.ones((Nz_bar, Ny))
