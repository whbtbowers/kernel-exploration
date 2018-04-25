#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:00:02 2018

@author: whb17
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import preprocessing as prep
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_classification

#Select current toy dataset
dataset = '010'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')


# Plot initial data

plt.figure(figsize=(8,6))

set1 = plt.scatter(X[y==0, 0],
            X[y==0, 1], 
            color='red', 
            alpha=0.5)

set2 = plt.scatter(X[y==1, 0],
            X[y==1, 1],
            color='blue',
            alpha=0.5)

plt.title('Simulated multivariate dataset %s' % dataset)
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.legend([set1, set2], ['Category 1', 'Category 1'])

#plt.show()
plt.savefig('../../figs/mvnfigs/kpcatestinitial%s.png' % dataset)
plt.close()

# Linear PCA

l_kpca = KernelPCA(n_components=2, eigen_solver='arpack', kernel='linear')
X_l_kpca = l_kpca.fit_transform(X)

plt.figure(figsize=(8,6))

cat1 = plt.scatter(X_l_kpca[y==0, 0],
            X_l_kpca[y==0, 1], 
            color='red',
            marker = 'o',
            alpha=0.5)

cat2 = plt.scatter(X_l_kpca[y==1, 0], 
            X_l_kpca[y==1, 1], 
            color='blue',
            marker = '^',
            alpha=0.5)

plt.legend([cat1, cat2], ['Category 1', 'Category 2'])
plt.title('PCA of simulated dataset %s with linear kernel' % dataset)
plt.ylabel('Principal component 1')
plt.xlabel('Principal component 2')

#plt.show()
plt.savefig('../../figs/mvnfigs/kpcatestlinear%s.png' % dataset)
plt.close()

# KPCA with RBF kernel

#gamma value
g = 0.002

rbf_kpca = KernelPCA(n_components=2, kernel='rbf', gamma = g)
X_rbf_kpca = rbf_kpca.fit_transform(X)

plt.figure(figsize=(8,6))

cat1 = plt.scatter(X_rbf_kpca[y==0, 0],
            X_rbf_kpca[y==0, 1], 
            color='red',
            marker = 'o',
            alpha=0.5)

cat2 = plt.scatter(X_rbf_kpca[y==1, 0], 
            X_rbf_kpca[y==1, 1], 
            color='blue',
            marker = '^',
            alpha=0.5)

plt.legend([cat1, cat2], ['Category 1', 'Category 2'])
plt.title('PCA of simulated dataset %s with RBF kernel' % dataset)
plt.ylabel('Principal component 1')
plt.xlabel('Principal component 2')
#display gamma value t0 3 decimal places
plt.text(-0.6, 0.3, 'gamma = %.3f' % g, fontsize=12)

#plt.show()
plt.savefig('../../figs/mvnfigs/kpcatestrbf%s.png'% dataset)
plt.close()
