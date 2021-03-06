import tensorflow as tf
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import pprint

ref = "/Users/aoji/Documents/THz/data/aj-7-2/coef/人/心肌/1.txt"
# folder_txt1 = "/Users/aoji/Documents/THz/data/aj-7-2/coef/人/心肌/all.txt"
# folder_txt2 = "/Users/aoji/Documents/THz/data/aj-7-2/coef/人/血管/all.txt"
# folder_txt3 = "/Users/aoji/Documents/THz/data/aj-7-2/coef/猪/心肌/all.txt"
# folder_txt4 = "/Users/aoji/Documents/THz/data/aj-7-2/coef/猪/血管/all.txt"


folder_txt1 = "/Users/aoji/Desktop/未命名文件夹/水.txt"
folder_txt2 = "/Users/aoji/Desktop/未命名文件夹/酒精.txt"

all1 = []
all2 = []

ref = np.loadtxt(ref, skiprows=1)
sig1 = np.loadtxt(folder_txt1)
sig2 = np.loadtxt(folder_txt2)
print(sig1.shape)

for i in range(100):
    plt.plot(sig1[i,:],c='r')
    plt.plot(sig2[i,:],c='b')
plt.show()

# sig3 = np.loadtxt(folder_txt3)
# sig4 = np.loadtxt(folder_txt4)
#
# x = ref[:, 0]
#
# ref = ref[:, 1].reshape(1, -1)
# sig1 = np.mean(sig1, 0).reshape(-1, 1)
# sig2 = np.mean(sig2, 0).reshape(-1, 1)
# sig3 = np.mean(sig3, 0).reshape(1, -1)
# sig4 = np.mean(sig4, 0).reshape(1, -1)
#
# stack1 = np.vstack([sig1, sig3])
# stack2 = np.vstack([sig2, sig4])
#
# plt.figure()
# plt.plot(x, stack1[0, :], label="Diseased", c='r')
# plt.plot(x, stack1[1, :], label="Normal", c='b')
# plt.legend()
# plt.title("Mean Of Absorption Coefficient Spectrum Of The Myocardium")
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
# plt.xlabel("Frequency(THz)")
# plt.ylabel("Absorption Coefficient(1/cm)")
# plt.show()
#
# plt.figure()
# plt.plot(x, stack2[0, :], label="Diseased", c='r')
# plt.plot(x, stack2[1, :], label="Normal", c='b')
# plt.legend()
# plt.title("Mean Of Absorption Coefficient Spectrum Of The Blood Vessel")
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
# plt.xlabel("Frequency(THz)")
# plt.ylabel("Absorption Coefficient(1/cm)")
# plt.show()
