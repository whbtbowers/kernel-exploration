#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:40:59 2018

@author: whb17
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Select current toy dataset
dataset = '010'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')

plt.figure(figsize=(6, 5))

cata = plt.scatter(X[y==0, 0],
                   X[y==0, 1],
                   color='red',
                   alpha=0.5
                   )

catb = plt.scatter(X[y==1, 0],
                   X[y==1, 1],
                   color='blue',
                   alpha=0.5)

plt.title('Initial data')
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.legend([cata, catb],['Category A', 'Category B'])

plt.show()