import numpy as np
import matplotlib.pyplot as plt

g_num = 100
g_absorb_load_dir = "./data/红上/吸收/%d.txt"
g_absorb_save_dir = "./data/红上/absorb.txt"
g_absorb_stack_X_dir = "./data/X_absorb.txt"
g_absorb_stack_y_dir = "./data/y_absorb.txt"


def LoadData():
    # absorb
    txt = np.loadtxt(g_absorb_load_dir % 1, skiprows=1, dtype=np.float)
    x, y = np.hsplit(txt, [1])
    x_absorb = x[(x < 1.2) & (x > 0.2)].reshape(1, -1)  # [1,17]
    print(x_absorb.shape)

    y_absorb = np.zeros([0])
    for i in range(1, g_num + 1):  # 100个样本
        txt = np.loadtxt(g_absorb_load_dir % i, skiprows=1, dtype=np.float)
        x = txt[:, 0]
        y = txt[(x > 0.2) & (x < 1.2)][:, 1]
        y_absorb = np.append(y_absorb, y)
    y_absorb = y_absorb.reshape(g_num, x_absorb.shape[1])  # [100,17] 100个样本17个特征
    print(y_absorb.shape)
    np.savetxt(g_absorb_save_dir, y_absorb, fmt='%.8f')

    # refraction
    # ...
    return None


def Stack():
    X1 = np.loadtxt("./data/红上/absorb.txt", dtype=np.float)
    X2 = np.loadtxt("./data/红下/absorb.txt", dtype=np.float)
    X3 = np.loadtxt("./data/蓝/absorb.txt", dtype=np.float)
    X = np.vstack([X1, X2, X3])

    y1 = np.zeros([100], dtype=np.int)
    y2 = np.zeros([100], dtype=np.int) + 1
    y3 = np.zeros([50], dtype=np.int) + 2
    y = np.hstack([y1, y2, y3]).reshape(-1, 1)

    print(X.shape)
    print(y.shape)
    np.savetxt(g_absorb_stack_X_dir, X, fmt="%.8f")
    np.savetxt(g_absorb_stack_y_dir, y, fmt="%d")

    return X


def Show(X):
    plt.figure()
    for i in range(0, 100):
        plt.plot(X[i, :], c='r')
    plt.xticks([])
    plt.yticks([])

    plt.figure()
    for i in range(100, 200):
        plt.plot(X[i, :], c='g')
    plt.xticks([])
    plt.yticks([])

    plt.figure()
    for i in range(200, 250):
        plt.plot(X[i, :], c='b')
    plt.xticks([])
    plt.yticks([])

    plt.figure()
    plt.plot(X[20, :], c='r', label="Positive")
    plt.plot(X[150, :], c='g', label="Negative")
    plt.plot(X[220, :], c='b', label="Blood vessel")
    plt.xlabel("Feature")
    plt.ylabel("Absorption Coefficient")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # LoadData()
    X = Stack()
    Show(X)
