import numpy as np
import matplotlib.pyplot as plt

fft = np.loadtxt("/Users/aoji/Desktop/未命名文件夹/fft.txt", skiprows=1)
freq, reference, freq, signal, freq, drude = np.hsplit(fft, 6)
print(freq.shape)
plt.figure("signal/reference")
plt.plot(freq, signal / reference)
plt.figure("-np.log(signal/reference)/0.01")
plt.plot(freq, -np.log(signal / reference) / 0.01)

coef = np.loadtxt("/Users/aoji/Desktop/未命名文件夹/coef.txt", skiprows=1)
freq, coef = np.hsplit(coef, 2)
plt.figure("coef")
plt.plot(freq, coef)

plt.show()
