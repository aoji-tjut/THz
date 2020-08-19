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

X1 = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/coef/人/血管/all.txt", dtype=np.float)
X2 = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/coef/猪/血管/all.txt", dtype=np.float)
X = np.vstack([X1, X2])

y = [1] * 50 + [0] * 50
y = np.asarray(y)
y.reshape(-1, 1)

print(X.shape)
print(y.shape)
np.savetxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/X.txt", X, fmt="%.8f")
np.savetxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/y.txt", y, fmt="%d")

# x轴
xx = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/coef/人/心肌/1.txt", skiprows=1)
xx = xx[:, 0]

plt.figure()
for i in range(0, 100):
    if i < 50:
        if (i == 0):
            plt.plot(xx, X[i, :], c='r', label="Diseased")
            continue
        plt.plot(xx, X[i, :], c='r')
    else:
        if (i == 50):
            plt.plot(xx, X[i, :], c='b', label="Normal")
            continue
        plt.plot(xx, X[i, :], c='b')

plt.legend()
plt.title("Absorption Coefficient Spectrum Of The Blood Vessel")
# plt.title("Absorption Coefficient Spectrum Of The Myocardium")
plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
plt.xlabel("Frequency(THz)")
plt.ylabel("Absorption Coefficient(1/cm)")
plt.show()
