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

txt1 = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/coef/猪/血管/all.txt")
# txt2 = np.loadtxt("/Users/aoji/Desktop/aj-7-2/coef/猪/肥-瘦-血管/瘦/all.txt")
fig = plt.figure()
for i in range(50):
    # if(i==0):
        # plt.plot(txt1[i, :], c='r',label="fat")
        # plt.plot(txt2[i, :], c='g',label="thin")
        # continue
    plt.plot(txt1[i, :], c='r')
    # plt.plot(txt2[i, :], c='g')
    # plt.plot(txt3[i, :], c='b')
    # plt.plot(txt4[i, :], c='y')
plt.legend()
plt.show()


#man = np.loadtxt("/Users/aoji/Desktop/a/人.txt", skiprows=1)
#_, ref1, _, sig1, _, drude1 = np.hsplit(man, 6)
#
#pig = np.loadtxt("/Users/aoji/Desktop/a/猪.txt", skiprows=1)
#_, ref2, _, sig2, _, drude2 = np.hsplit(pig, 6)
#
#fig = plt.figure()
plt.plot(_, sig2, c='b', label="Normal")
#plt.plot(_, sig1, c='r', label="Diseased")
#plt.legend()

#plt.title("Mean Of Frequency Domain Spectrum Of The Blood Vessel")
#plt.xticks([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,1.2])
#plt.xlabel("Frequency(THz)")
#plt.ylabel("Amplitude(mV)")
#plt.show()
