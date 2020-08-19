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

folder_txt = "/Users/aoji/Documents/THz/data/aj-7-2/freq/猪/血管/%d.txt"
save_txt = "/Users/aoji/Documents/THz/data/aj-7-2/freq/猪/血管/all.txt"

x = np.loadtxt(folder_txt % 1, skiprows=1)
x = x[:, 0]

all = []

# 直接导出
for i in range(50):
    txt = np.loadtxt(folder_txt % (i + 1), skiprows=1)
    value = txt[:, 3].reshape(1, -1)
    all = np.append(all, value)
all = all.reshape(50, -1)
print(all.shape)
np.savetxt(save_txt, all)

fig = plt.figure()
for i in range(50):
    if (i == 0):
        plt.plot(x, all[i, :])
        continue
    plt.plot(x, all[i, :])
plt.title("Lesion Vessel Frequency Domain Spectrum")
plt.xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
plt.xlabel("Frequency(THz)")
plt.ylabel("Amplitude(mV)")
plt.show()
