# Import the necessary modules
import numpy as np
import matplotlib.pyplot as plt

# SNR in dB
SNR_db_channel = [6, 12, 20]

for index in range(len(SNR_db_channel)):
    # SNR in linear scale
    SNR_linear_channel = 10 ** (SNR_db_channel[index] / 10)

    # cardinality of AWGN channel input signal
    Nx = 8

    # cardinality of AWGN channel output signal
    Ny = 1024

    # cardinality of sensor output signal. The value is set to be the same as the cardinality of Y
    Nz = 1024

    # input alphabet 64-ASK
    alphabet = np.arange(-7, 9, 2)


    # p(x) uniform distribution
    p_x = 1 / Nx * np.ones(Nx)

    # variance of input signal
    var_X = np.dot((alphabet**2), p_x)

    # variance of channel noise
    var_N = var_X / SNR_linear_channel

    # standard deviation of each part of the complex noise

    # sampling interval
    dy = 2 * (np.amax(alphabet) + 5 * np.sqrt(var_N)) / Ny

    # discretized y
    y = np.arange(-(np.amax(alphabet) + 5 * np.sqrt(var_N)), (np.amax(alphabet) + 5 * np.sqrt(var_N)),
                  dy).reshape(
        Ny, 1)

    # initialized p(y|x)
    p_y_x = np.ones((Ny, Nx))

    # initialize the temporary matrix to store conditional pdf
    temp = np.zeros((Ny, 1))

    # obtain the conditional pdf p(y|x) for a Gaussian channel
    for idx in range(0, Nx):
        tmp = (1 / np.sqrt(2 * np.pi * var_N)) * np.exp(-(y[np.arange(0, Ny)] - alphabet[idx]) ** 2 / (2 * var_N))
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

    C = 0.5 * np.log2(1 + SNR_linear_channel)

    # Amplify and Forward

    # SNR for second channel in sensor in dB
    SNR_db_sensor = np.arange(0, 40, 1)
    IXZ = []
    cap = []

    # SNR in linear scale
    for i in range(len(SNR_db_sensor)):
        SNR_lin_sensor = 10 ** (SNR_db_sensor[i] / 10)

        # variance of input signal into the sensor
        sigma2_X_sensor = np.var(y)

        scaling_factor = np.sqrt(sigma2_X_sensor / (var_X + var_N))

        # input into sensor
        x_sensor = scaling_factor * y

        # variance of noise in sensor
        sigma2_N_sensor = sigma2_X_sensor / SNR_lin_sensor

        # standard deviation of each part of the complex noise
        sigma_N_2_sensor = np.sqrt(sigma2_N_sensor / 2)

        # AWGN for Amplify and Forward sensing
        n_sensor = (np.random.randn(np.size(x_sensor)) + 1j * np.random.randn(np.size(x_sensor))) * sigma_N_2_sensor

        # sampling interval
        dz = 2 * (np.amax(x_sensor) + 5 * np.sqrt(sigma2_N_sensor)) / Nz

        # discretized z
        z = np.arange(-(np.amax(x_sensor) + 5 * np.sqrt(sigma2_N_sensor)),
                      (np.amax(x_sensor) + 5 * np.sqrt(sigma2_N_sensor)),
                      dz).reshape(Nz, 1)

        # initialized p(z|y)
        p_z_x_sensor = np.ones((Nz, Ny))

        # initialize the temporary matrix to store conditional pdf
        temp1 = np.zeros((Nz, 1))

        # obtain the conditional pdf p(y|x) for a Gaussian channel
        for idx in range(0, Ny):
            tmp1 = (1 / np.sqrt(2 * np.pi * sigma2_N_sensor)) * np.exp(
                -(z - x_sensor[idx]) ** 2 / (2 * sigma2_N_sensor))
            temp1 = np.append(temp1, tmp1, axis=1)

        # remove the zero from the first column of the matrix
        pz_remove_initial_zeroes = temp1[:, -Ny:]

        # obtain the conditional pdf
        p_z_given_x_sensor = p_z_x_sensor * pz_remove_initial_zeroes

        # normalization for the pdf
        norm_sum_1 = np.sum(p_z_given_x_sensor, 0)

        # p(z|x_sensor)
        p_z_given_x_sensor = p_z_given_x_sensor / np.tile(norm_sum_1, (Nz, 1))

        # p(z)
        p_z = np.sum(p_z_given_x_sensor * p_y, 1)

        # p(z|y) expanded
        p_z_given_x_sensor_expanded = np.tile(np.expand_dims(p_z_given_x_sensor, axis=2), (1, 1, Nx))

        g = np.sum(np.tile(np.expand_dims(p_x_y, axis=0), (Nz, 1, 1)) * p_z_given_x_sensor_expanded, 1)

        # p(x|z)
        p_x_given_z = 1 / np.tile(np.expand_dims(p_z + 1e-31, axis=1), (1, Nx)) * g

        # p(x,z)
        p_x_z = p_x_given_z * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))

        p_x_times_p_z = np.tile(np.expand_dims(p_x, axis=0), (Nz, 1)) * np.tile(np.expand_dims(p_z, axis=1), (1, Nx))

        w1 = np.log2(p_x_z + 1e-31) - np.log2(p_x_times_p_z + 1e-31)

        # I(X;Z)
        I_x_z = np.sum(p_x_z * w1)

        C = 0.5 * np.log2(1 + (SNR_linear_channel * SNR_lin_sensor) / (SNR_linear_channel + SNR_lin_sensor + 1))

        IXZ.append(I_x_z)
        cap.append(C)
    plt.plot(SNR_db_sensor, IXZ, linewidth=2, label=str(SNR_db_channel[index]) + ' dB - 8-ASK')
    plt.plot(SNR_db_sensor, cap, '-o', linewidth=2, label=str(SNR_db_channel[index]) + ' dB - Gaussian')
    plt.title('Relevant Information versus SNR for Amplify and Forward Sensing')
    plt.grid()
    plt.legend()
    plt.xlim(0, 40)
    plt.xlabel('SNR of sensor channel in dB')
    plt.ylabel('I(X;Z)')
plt.show()
