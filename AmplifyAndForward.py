# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt

# SNR in dB
SNR_db_1 = [6, 12, 20]

for index in range(len(SNR_db_1)):
    # SNR in linear scale
    SNR_lin_1 = 10 ** (SNR_db_1[index] / 10)

    # cardinality of AWGN channel input signal
    Nx = 4

    # cardinality of AWGN channel output signal
    Ny = 256

    # cardinality of sensor output signal. The value is set to be the same as the cardinality of Y
    Nz = 256

    # input alphabet 4-ASK
    alphabet = np.array([-3, -1, 1, 3])

    # p(x)
    p_x = 1 / Nx * np.ones(Nx)

    # variance of input signal
    sigma2_X = np.dot(((alphabet - np.mean(alphabet)) ** 2), p_x)

    # variance of noise
    sigma2_N = sigma2_X / SNR_lin_1

    # standard deviation of each part of the complex noise
    sigma_N_2 = np.sqrt(sigma2_N / 2)

    # noise
    n = (np.random.randn(np.size(alphabet)) + 1j * np.random.randn(np.size(alphabet))) * sigma_N_2

    # sampling interval
    dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(sigma2_N)) / Ny

    # discretized y
    y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(sigma2_N)), (np.amax(alphabet) + 5 * np.sqrt(sigma2_N)), dy).reshape(
        Ny, 1)

    # initialized p(y|x)
    p_y_x = np.ones((Ny, Nx))

    # initialize the temporary matrix to store conditional pdf
    temp = np.zeros((Ny, 1))

    # obtain the conditional pdf p(y|x) for a Gaussian channel
    for idx in range(0, Nx):
        tmp = (1 / np.sqrt(2 * np.pi * sigma2_N)) * np.exp(-(y - alphabet[idx]) ** 2 / (2 * sigma2_N))
        temp = np.append(temp, tmp, axis=1)

    # remove the zero from the first column of the matrix
    py_remove_initial_zeroes = temp[:, -Nx:]

    # obtain the conditional pdf
    p_y_given_x = p_y_x * py_remove_initial_zeroes

    # normalization for the pdf
    norm_sum = np.sum(p_y_given_x, 0)

    # p(y|x)
    p_y_given_x = p_y_given_x / np.tile(norm_sum, (Ny, 1))

    # p(x,y) joint probability of x and y
    p_x_y = p_y_given_x * p_x

    # p(y)
    p_y = np.sum(p_x_y, 1)

    # p(x|y) obtained using Bayes rule
    p_x_given_y = p_y_given_x * np.tile(np.expand_dims(np.tile(p_x, Ny // Nx) / p_y, axis=1), (1, Nx))

    p_x_times_p_y = np.tile(np.expand_dims(p_x, axis=0), (Ny, 1)) * np.tile(np.expand_dims(p_y, axis=1), (1, Nx))

    w1 = np.log2(p_x_y + 1e-31) - np.log2(p_x_times_p_y + 1e-31)

    # I(X;Y)
    I_x_y = np.sum(p_x_y * w1)

    # Amplify and Forward

    # SNR for second channel in sensor in dB
    SNR_db_2 = np.arange(0, 20, 1)
    IXZ = []

    # SNR in linear scale
    for i in range(len(SNR_db_2)):
        SNR_lin_2 = 10 ** (SNR_db_2[i] / 10)

        # variance of signal into the sensor
        sigma2_Y = np.dot(np.transpose((y-np.mean(y)) ** 2), p_y)[0]

        # variance of noise in sensor
        sigma2_N_sensor = sigma2_Y / SNR_lin_2

        # standard deviation of each part of the complex noise
        sigma_N_2_sensor = np.sqrt(sigma2_N_sensor / 2)

        # AWGN for Amplify and Forward sensing
        n_2 = (np.random.randn(np.size(y)) + 1j * np.random.randn(np.size(y))) * sigma_N_2_sensor

        # sampling interval
        dz = 2 * (np.amax(y) + 5 * np.sqrt(sigma2_N_sensor)) / Nz

        # discretized z
        z = np.arange(-(np.amax(y) + 5 * np.sqrt(sigma2_N_sensor)), (np.amax(y) + 5 * np.sqrt(sigma2_N_sensor)),
                      dz).reshape(
            Nz, 1)

        # initialized p(z|y)
        p_z_y = np.ones((Nz, Ny))

        # initialize the temporary matrix to store conditional pdf
        temp1 = np.zeros((Nz, 1))

        # obtain the conditional pdf p(y|x) for a Gaussian channel
        for idx in range(0, Ny):
            tmp1 = (1 / np.sqrt(2 * np.pi * sigma2_N_sensor)) * np.exp(-(z - y[idx]) ** 2 / (2 * sigma2_N_sensor))
            temp1 = np.append(temp1, tmp1, axis=1)

        # remove the zero from the first column of the matrix
        pz_remove_initial_zeroes = temp1[:, -Ny:]

        # obtain the conditional pdf
        p_z_given_y = p_z_y * pz_remove_initial_zeroes

        # normalization for the pdf
        norm_sum_1 = np.sum(p_z_given_y, 0)

        # p(z|y)
        p_z_given_y = p_z_given_y / np.tile(norm_sum_1, (Nz, 1))

        # p(z)
        p_z = np.sum(p_z_given_y * p_y, 1)

        # p(z|y) expanded
        p_z_given_y_expanded = np.tile(np.expand_dims(p_z_given_y, axis=2), (1, 1, Nx))

        g = np.sum(np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1)) * p_z_given_y_expanded, 1)

        # p(x|z)
        p_x_given_z = 1 / np.tile(np.expand_dims(p_z + 1e-31, axis=1), (1, Nx)) * g

        # p(x,z)
        p_x_z = p_x_given_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))

        p_x_times_p_z = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))

        w1 = np.log2(p_x_z + 1e-31) - np.log2(p_x_times_p_z + 1e-31)

        # I(X;Z)
        I_x_z = np.sum(p_x_z * w1)

        IXZ.append(I_x_z)

    plt.plot(SNR_db_2, IXZ, linewidth=2, label=str(SNR_db_1[index]) + ' dB')
    plt.title('Relevant Information versus SNR for Amplify and Forward Sensing for 4-ASK')
    plt.grid()
    plt.legend()
    plt.xlim(0, 20)
    plt.xlabel('SNR of sensor channel in dB')
    plt.ylabel('I(X;Z)')
plt.show()

