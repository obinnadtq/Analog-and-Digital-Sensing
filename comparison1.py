import matplotlib.pyplot as plt
import numpy as np
IB6 = []
COVQ6 = []
IB12 = []
COVQ12 = []
IB20 = []
COVQ20 = []
with open("IB_6.txt") as file1:
    lines = file1.readlines()
    ib6 = [line.split()[0] for line in lines]
with open("IB_12.txt") as file1:
    lines = file1.readlines()
    ib12 = [line.split()[0] for line in lines]
with open("IB_20.txt") as file1:
    lines = file1.readlines()
    ib20 = [line.split()[0] for line in lines]

with open("COVQ_6.txt") as file2:
    lines = file2.readlines()
    covq6 = [line.split()[0] for line in lines]
with open("COVQ_12.txt") as file2:
    lines = file2.readlines()
    covq12 = [line.split()[0] for line in lines]
with open("COVQ_20.txt") as file2:
    lines = file2.readlines()
    covq20 = [line.split()[0] for line in lines]


for i in ib6:
    IB6.append(float(i))

for i in ib12:
    IB12.append(float(i))

for i in ib20:
    IB20.append(float(i))

for i in covq6:
    COVQ6.append(float(i))

for i in covq12:
    COVQ12.append(float(i))

for i in covq20:
    COVQ20.append(float(i))

plt.plot(range(len(IB6)), IB6, linewidth=2, label="IIB, SNR - 6 dB")
plt.plot(range(len(COVQ6)), COVQ6, linewidth=2, label="CA, SNR - 6 dB ")
plt.plot(range(len(IB12)), IB12, linewidth=2, label="IIB, SNR - 12 dB")
plt.plot(range(len(COVQ12)), COVQ12, linewidth=2, label="CA, SNR - 12 dB")
plt.plot(range(len(IB20)), IB20, linewidth=2, label="IIB, SNR - 20 dB")
plt.plot(range(len(COVQ20)), COVQ20, linewidth=2, label="CA, SNR - 20 dB")
plt.xlabel('No. of iterations')
plt.ylabel('Relevant Information')
plt.title('Comparison between Iterative IB and COVQ')
plt.xticks(np.arange(0, 22, 2))
plt.legend()
plt.grid()
plt.show()
