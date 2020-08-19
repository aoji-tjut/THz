import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys

frequency = reference_amplitude = signal_amplitude = divide = 0
fileName1 = fileName2 = 0


class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setWindowTitle("THz")
        self.setFixedSize(450, 150)

        self.mylabel1 = QtWidgets.QLabel(self)
        self.mylabel1.setText("reference:")
        self.mylabel1.setFixedSize(100, 30)
        self.mylabel1.move(5, 15)

        self.mylabel2 = QtWidgets.QLabel(self)
        self.mylabel2.setText("signal:")
        self.mylabel2.setFixedSize(100, 30)
        self.mylabel2.move(5, 70)

        self.mytext1 = QtWidgets.QTextEdit(self)
        self.mytext1.setReadOnly(True)
        self.mytext1.setFixedSize(290, 50)
        self.mytext1.move(100, 5)

        self.mytext2 = QtWidgets.QTextEdit(self)
        self.mytext2.setReadOnly(True)
        self.mytext2.setFixedSize(290, 50)
        self.mytext2.move(100, 60)

        self.myButton1 = QtWidgets.QPushButton(self)
        self.myButton1.setFixedSize(60, 60)
        self.myButton1.move(390, 2)
        self.myButton1.setText("...")
        self.myButton1.clicked.connect(self.load_reference)

        self.myButton2 = QtWidgets.QPushButton(self)
        self.myButton2.setFixedSize(60, 60)
        self.myButton2.move(390, 57)
        self.myButton2.setText("...")
        self.myButton2.clicked.connect(self.load_signal)

        self.myButton3 = QtWidgets.QPushButton(self)
        self.myButton3.setFixedSize(450, 40)
        self.myButton3.move(0, 110)
        self.myButton3.setText("确定")
        self.myButton3.setEnabled(False)
        self.myButton3.clicked.connect(self.confirm)

    def load_reference(self):
        global frequency, reference_amplitude, fileName1
        fileName1, filetype = QFileDialog.getOpenFileName(self, "browse", "/Users/aoji/Documents/太赫兹数据/data/")
        if fileName1 == "":
            self.mytext1.setText("")
            self.myButton3.setEnabled(False)
            return
        self.mytext1.setText(fileName1)
        reference = np.loadtxt(fileName1, comments='#', dtype=np.float32)
        frequency, reference_amplitude = np.hsplit(reference, [1])
        if fileName1 != "" and fileName2 != "":
            self.myButton3.setEnabled(True)

    def load_signal(self):
        global frequency, signal_amplitude, fileName2
        fileName2, filetype = QFileDialog.getOpenFileName(self, "browse", "/Users/aoji/Documents/太赫兹数据/data/")
        if fileName2 == "":
            self.mytext2.setText("")
            self.myButton3.setEnabled(False)
            return
        self.mytext2.setText(fileName2)
        signal = np.loadtxt(fileName2, comments='#', dtype=np.float32)
        frequency, signal_amplitude = np.hsplit(signal, [1])
        if fileName1 != "" and fileName2 != "":
            self.myButton3.setEnabled(True)

    def confirm(self):
        global frequency, signal_amplitude, reference_amplitude
        divide = np.true_divide(signal_amplitude, reference_amplitude)
        result = np.hstack([frequency, divide])
        np.savetxt("divide.txt", result, fmt='%.8f')

        peak_x = np.zeros([0])
        peak_y = np.zeros([0])
        for i in range(1, divide.shape[0] - 1):
            if divide[i - 1] < divide[i] and divide[i] > divide[i + 1]:
                peak_x = np.append(peak_x, frequency[i])
                peak_y = np.append(peak_y, divide[i])
        peak_x = peak_x.reshape(-1, 1)
        peak_y = peak_y.reshape(-1, 1)
        peak = np.hstack([peak_x, peak_y])
        np.savetxt("peak.txt", peak, fmt='%.8f')

        valley_x = np.zeros([0])
        valley_y = np.zeros([0])
        for i in range(1, divide.shape[0] - 1):
            if divide[i - 1] < divide[i] and divide[i] < divide[i + 1]:
                valley_x = np.append(valley_x, frequency[i])
                valley_y = np.append(valley_y, divide[i])
        valley_x = valley_x.reshape(-1, 1)
        valley_y = valley_y.reshape(-1, 1)
        valley = np.hstack([valley_x, valley_y])
        np.savetxt("valley.txt", valley, fmt='%.8f')

        plt.close('all')

        plt.figure("reference", (10, 5))
        plt.xlabel("Frequency(THz)")
        plt.ylabel("Intensity(dB)")
        plt.plot(frequency, reference_amplitude, c="b", lw=0.5)

        plt.figure("signal", (10, 5))
        plt.xlabel("Frequency(THz)")
        plt.ylabel("Intensity(dB)")
        plt.plot(frequency, signal_amplitude, c="r", lw=0.5)

        plt.figure("divide", (10, 5))
        plt.xlabel("Frequency(THz)")
        plt.ylabel("signal/reference")
        plt.plot(frequency, divide, c="g")

        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())
