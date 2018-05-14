#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import KernelPCA, PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import laplacian_kernel
from scipy import interp
from sklearn.datasets import load_iris

'''
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
Y = iris.target


def my_kernel(X, Y):
    """
    We create a custom kernel:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


h = .02  # step size in the mesh

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()
'''
def make_meshgrid(x, y, h=.002):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out






X = pd.read_csv('../../data/simulated/mvnsim/mvnsim023.csv', sep=',', header=0, index_col=0)
y = np.load('../../data/simulated/mvnsim/target023.npy')

X_scaled = scale(X)

X_kpca = KernelPCA(n_components=2, kernel='linear').fit_transform(X_scaled)



'''
X_scaled = scale(X)

gamma = 0.002

folds = 10

h = 0.002
    
cv = StratifiedKFold(n_splits=folds, random_state=10)

svm = SVC(kernel='rbf', gamma=gamma, probability=True)


for train, test in cv.split(X_kpca, y):
    
    X_svc = svm.fit(X_kpca[train], y[train])
    probas_ = X_svc.predict_proba(X_kpca[test])
    
    x_min, x_max = X_kpca[:, 0].min() - 0.1, X_kpca[:, 0].max() + 0.1
    y_min, y_max = X_kpca[:, 1].min() - 0.1, X_kpca[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    fig = plt.figure(figsize=(8, 6))
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
                   
    cata = plt.scatter(X_kpca[y==0, 0],
                       X_kpca[y==0, 1],
                       color='green',
                       marker = '^',
                       alpha=0.8,
                       )
    
    catb = plt.scatter(X_kpca[y==1, 0],
                       X_kpca[y==1, 1], 
                       color='yellow',
                       marker = 's',
                       alpha=0.8,
                       )

    plt.show()
               
    
fig = plt.figure(figsize=(8, 6))

cata = plt.scatter(X_kpca[y==0, 0],
                   X_kpca[y==0, 1],
                   color='red',
                   marker = '^',
                   alpha=0.5
                   )

catb = plt.scatter(X_kpca[y==1, 0],
                   X_kpca[y==1, 1],
                   color='blue',
                   marker = 's',
                   alpha=0.5)

plt.title('title')
plt.xlabel('x_label')
plt.ylabel('y_label')    
gamma_label = mpatches.Patch(color='white', label='gamma')
plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category 1', 'Category 0'])

plt.show()
plt.close() 
'''