import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

folder_txt = "/Users/aoji/Documents/THz/data/vi-coef/red-down/%d.txt"  # 输入文件夹
folder_peak = "/Users/aoji/Desktop/红下/peak%d.txt"  # 输出文件夹
folder_valley = "/Users/aoji/Desktop/红下/valley%d.txt"  # 输出文件夹
image = "%d"

peak_x_all = np.zeros([0])
valley_x_all = np.zeros([0])

for i in range(1, 101):  # 文件范围
    file_txt = folder_txt % i
    file_peak = folder_peak % i
    file_valley = folder_valley % i

    txt = np.loadtxt(file_txt, comments='f', dtype=np.float)
    frequency, coef = np.hsplit(txt, [1])

    # plt.figure(image % i)
    plt.figure("coef")
    plt.plot(frequency, coef)
    plt.xlim(0.2, 1.2)
    plt.ylim(-3, 3)

    peak_x = np.zeros([0])
    peak_y = np.zeros([0])
    for i in range(1, coef.shape[0] - 1):
        if coef[i - 1] < coef[i] and coef[i] > coef[i + 1]:
            peak_x = np.append(peak_x, frequency[i])
            peak_y = np.append(peak_y, coef[i])
    peak_x_all = np.append(peak_x_all, peak_x)
    peak_x = peak_x.reshape(-1, 1)
    peak_y = peak_y.reshape(-1, 1)
    peak = np.hstack([peak_x, peak_y])

    valley_x = np.zeros([0])
    valley_y = np.zeros([0])
    for i in range(1, coef.shape[0] - 1):
        if coef[i - 1] > coef[i] and coef[i] < coef[i + 1]:
            valley_x = np.append(valley_x, frequency[i])
            valley_y = np.append(valley_y, coef[i])
    valley_x_all = np.append(valley_x_all, valley_x)
    valley_x = valley_x.reshape(-1, 1)
    valley_y = valley_y.reshape(-1, 1)
    valley = np.hstack([valley_x, valley_y])

    # np.savetxt(file_peak, peak, fmt='%.8f')
    # np.savetxt(file_valley, valley, fmt='%.8f')

    # peak_y[:] = 0
    # valley_y[:] = 0
    plt.figure("peak", (10, 5))
    plt.scatter(peak_x, peak_y, c="r", alpha=0.02)
    plt.xlim(0, 1.5)

    plt.figure("valley", (10, 5))
    plt.scatter(valley_x, valley_y, c="r", alpha=0.02)
    plt.xlim(0, 1.5)

print(Counter(peak_x_all))
print(Counter(valley_x_all))
plt.show()
