import matplotlib.pyplot as plt
import numpy as np

IXY4 = []
IXY8 = []
IXZ4 = []
IXZ8 = []
IXZbar_4 = []
IXZbar_8 = []
IXZbar15_4 = []
IXZbar15_8 = []
SNR = np.arange(-20, 31, 1)
with open("IXY4.txt") as file1:
    lines = file1.readlines()
    ixy4 = [line.split()[0] for line in lines]

with open("IXY8.txt") as file1:
    lines = file1.readlines()
    ixy8 = [line.split()[0] for line in lines]

with open("IXZ4.txt") as file1:
    lines = file1.readlines()
    ixz4 = [line.split()[0] for line in lines]

with open("IXZ8.txt") as file1:
    lines = file1.readlines()
    ixz8 = [line.split()[0] for line in lines]

with open("IXZbar_4.txt") as file1:
    lines = file1.readlines()
    ixzbar_4 = [line.split()[0] for line in lines]

with open("IXZbar_8.txt") as file1:
    lines = file1.readlines()
    ixzbar_8 = [line.split()[0] for line in lines]

with open("IXZbar_154.txt") as file1:
    lines = file1.readlines()
    ixzbar15_4 = [line.split()[0] for line in lines]

with open("IXZbar_158.txt") as file1:
    lines = file1.readlines()
    ixzbar15_8 = [line.split()[0] for line in lines]

for i in ixy4:
    IXY4.append(float(i))

for i in ixy8:
    IXY8.append(float(i))

for i in ixz4:
    IXZ4.append(float(i))

for i in ixz8:
    IXZ8.append(float(i))

for i in ixzbar_4:
    IXZbar_4.append(float(i))

for i in ixzbar_8:
    IXZbar_8.append(float(i))

for i in ixzbar15_4:
    IXZbar15_4.append(float(i))

for i in ixzbar15_8:
    IXZbar15_8.append(float(i))


# plt.plot(SNR, IXY4, label="I(x;y), 4-ASK")
# plt.plot(SNR, IXZ4, label="I(x;z), 4-ASK")
# plt.plot(SNR, IXZbar_4, label="I(x;ẑ), 4-ASK TX-SNR = 20 dB")
# plt.plot(SNR, IXZbar15_4, label="I(x;ẑ), 4-ASK TX-SNR = 15 dB")
plt.plot(SNR, IXY8, label="I(x;y), 8-ASK")
plt.plot(SNR, IXZ8, label="I(x;z), 8-ASK")
plt.plot(SNR, IXZbar_8, label="I(x;ẑ), 8-ASK TX-SNR = 20 dB")
plt.plot(SNR, IXZbar15_8, label="I(x;ẑ), 8-ASK TX-SNR = 15 dB")
plt.xlabel('SNR (dB)')
plt.title('Effect of Quantization and Transmission')
plt.ylabel('Mutual Information')
plt.xticks(np.arange(-20, 32, 5))
plt.legend()
plt.grid()
plt.show()
