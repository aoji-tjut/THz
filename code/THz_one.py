import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys

txt = 0


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setWindowTitle("THz")
        self.setFixedSize(450, 95)

        self.mylabel1 = QtWidgets.QLabel(self)
        self.mylabel1.setText("browse:")
        self.mylabel1.setFixedSize(100, 30)
        self.mylabel1.move(5, 15)

        self.mytext1 = QtWidgets.QTextEdit(self)
        self.mytext1.setReadOnly(True)
        self.mytext1.setFixedSize(330, 50)
        self.mytext1.move(60, 5)

        self.myButton1 = QtWidgets.QPushButton(self)
        self.myButton1.setFixedSize(60, 60)
        self.myButton1.move(390, 2)
        self.myButton1.setText("...")
        self.myButton1.clicked.connect(self.browse)

        self.myButton2 = QtWidgets.QPushButton(self)
        self.myButton2.setFixedSize(450, 40)
        self.myButton2.move(0, 55)
        self.myButton2.setText("确定")
        self.myButton2.setEnabled(False)
        self.myButton2.clicked.connect(self.confirm)

    def browse(self):
        global txt
        fileName, filetype = QFileDialog.getOpenFileName(self, "browse", "/Users/aoji/Documents/太赫兹数据/data/")
        if fileName == "":
            self.mytext1.setText("")
            self.myButton2.setEnabled(False)
            return
        self.mytext1.setText(fileName)
        self.myButton2.setEnabled(True)
        txt = np.loadtxt(fileName, comments='f', dtype=np.float32)

    def confirm(self):
        t1, t2 = np.hsplit(txt, [2])
        t2, t3 = np.hsplit(t2, [2])
        frequency, reference = np.hsplit(t1, [1])
        frequency, signal = np.hsplit(t2, [1])
        frequency, drude = np.hsplit(t3, [1])

        result = np.true_divide(signal, reference)
        divide = np.hstack([frequency, result])
        np.savetxt("divide.txt", divide, fmt='%.8f')

        peak_x = np.zeros([0])
        peak_y = np.zeros([0])
        for i in range(1, result.shape[0] - 1):
            if result[i - 1] < result[i] and result[i] > result[i + 1]:
                peak_x = np.append(peak_x, frequency[i])
                peak_y = np.append(peak_y, result[i])
        peak_x = peak_x.reshape(-1, 1)
        peak_y = peak_y.reshape(-1, 1)
        peak = np.hstack([peak_x, peak_y])
        np.savetxt("peak.txt", peak, fmt='%.8f')

        valley_x = np.zeros([0])
        valley_y = np.zeros([0])
        for i in range(1, result.shape[0] - 1):
            if result[i - 1] > result[i] and result[i] < result[i + 1]:
                valley_x = np.append(valley_x, frequency[i])
                valley_y = np.append(valley_y, result[i])
        valley_x = valley_x.reshape(-1, 1)
        valley_y = valley_y.reshape(-1, 1)
        valley = np.hstack([valley_x, valley_y])
        np.savetxt("valley.txt", valley, fmt='%.8f')

        plt.close('all')

        plt.figure("reference", (10, 5))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.xlim(0, 2)
        plt.plot(frequency, reference, c="b", lw=0.5)

        plt.figure("signal", (10, 5))
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.xlim(0, 2)
        plt.plot(frequency, signal, c="r", lw=0.5)

        plt.figure("divide", (10, 5))
        plt.xlabel("Frequency")
        plt.ylabel("signal/reference")
        plt.xlim(0, 2)
        plt.plot(frequency, result, c="g")

        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
