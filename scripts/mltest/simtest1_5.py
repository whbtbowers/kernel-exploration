#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:19:41 2018

@author: whb17
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
#from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy import interp

#Select current toy dataset
dataset = '010'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')

## EXTRACT TRAINING AND TEST FOLDS ##


skf = StratifiedKFold(n_splits=10, random_state=10)
skf.get_n_splits(X, y)

X_train = []
X_test = []
y_train = []
y_test = []

for train_index, test_index in skf.split(X,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    
    #append lists
    X_train.append(X_tr)
    X_test.append(X_te)
    y_train.append(y_tr)
    y_test.append(y_te)
 
#convert lists to arrays
X_train = np.array([X_train])
X_test = np.array([X_train])
y_train = np.array([X_train])
y_test = np.array([X_train])

print('X train:', str(X_train.shape))
print('X test:', str(X_test.shape))
print('y train:', str(y_train.shape))
print('y test:', str(y_test.shape))

'''
#Logistic regression with default settings
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
y_score = classifier.fit(X_test, y_train).decision_function(X_test)
print(y_score)


# Compute ROC curve and ROC area for each class
n_classes = len(np.unique(y))

fpr = dict()
tpr = dict()
roc_auc = dict()


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


#Plot single class ROC curve
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

print('\ntrain X')
print(X_train)
print(X_train.shape)

print('\ntest X')
print(X_train)

print('\ntrain y')
print(y_train)

print('\ntest y')
print(y_test)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
print('y score')
print(y_score)

# Compute ROC curve and ROC area for each class
n_classes = len(np.unique(y))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold = roc_curve(y_test, y_score)
    #print('fpr:')
    #print(fpr[i])
    #print('tpr:')
    #print(tpr[i])
    #print('Underscore')
    #print(_)
    roc_auc[i] = auc(fpr[i], tpr[i])
    #print(roc_auc[i])
    
#print('area under curve')
#print(roc_auc)

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# Plot of a ROC curve for a specific class

roc_c = plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''