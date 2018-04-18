#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:01:09 2018

@author: whb17
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import preprocessing as prep
from sklearn.decomposition import KernelPCA

#toy data
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim005.csv', delimiter=',', header=0)

#categories
y = pd.read_csv('../../data/simulated/mvnsim/target005.csv', delimiter=',', header=0)
y = y[y.columns[1]]
#print(X)
#plt.figure(figsize=(8,6))
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()

#print(y)



#plot graph
lin_kpca = KernelPCA(n_components=2, kernel='linear')
X_lin_kpca = lin_kpca.fit_transform(X)

#print(X_lin_kpca)

#plot kpca graph

plt.figure(figsize=(8,6))

for i in range(len(y)):
    #for category 1 samples
    if y[i] == 0:
        plt.scatter(X_lin_kpca[:, 0][i],
        X_lin_kpca[:, 1][i],
        color ='red',
        marker='s',     #square marker
        alpha=0.5,
        )
    #for category 2 examples    
    elif y[i] == 1:
        plt.scatter(X_lin_kpca[:, 0][i],
        X_lin_kpca[:, 1][i],
        color='blue',
        marker='^',     #triangle marker
        alpha=0.5,
        )


plt.show()
plt.close()
