#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:30:54 2018

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

from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import preprocessing as prep
#from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Select current toy dataset
dataset = '010'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)#.as_matrix()
y_target = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')

n_samples, n_features = X.shape

#y_target = pd.DataFrame(y)

#X['target'] = y_df.values

cols = X.columns

#y_target = X['target']



#print(X.size)

#print("Toy data set dimensions : {}".format(X.shape))

#print(X.groupby('target').size())

#Plot works with numpy array but not pandas dataframe

'''
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

plt.show()
'''



'''
X_scaled = prep.scale(X)

# Plot scaled data

plt.figure(figsize=(8,6))

set1 = plt.scatter(X_scaled[y==0, 0],
            X_scaled[y==0, 1], 
            color='red', 
            alpha=0.5)

set2 = plt.scatter(X_scaled[y==1, 0],
            X_scaled[y==1, 1],
            color='blue',
            alpha=0.5)

plt.title('Simulated multivariate dataset %s scaled' % dataset)
plt.ylabel('y coordinate')
plt.xlabel('x coordinate')
plt.legend([set1, set2], ['Category 1', 'Category 1'])

plt.show()
'''

## PREPROCESSING ##

#Show columns with missing or null datapoints
#print(X.isnull().sum())

'''
#Show distributions of variables.
#X.hist(figsize=(20, 20))
plt.title('Distributions of variables in toy dataset %s' % dataset)
X.groupby('target').hist(figsize=(35, 35))
plt.savefig('../../figs/pandas/distribution%s.png' % dataset)
plt.close()

#scale data
X_scaled = prep.scale(X)

#Normalise data
X_normed = prep.normalize(X_scaled)


# Use single model for initial ROC curve
kfold = KFold(n_splits=10, random_state=10)
fold_score = cross_val_score(LogisticRegression(), X_normed, y_target, cv=kfold, scoring='accuracy')
score = cross_val_score(LogisticRegression(), X_normed, y_target, cv=kfold, scoring='accuracy').mean()
y_predict = cross_val_predict(LogisticRegression(), X_normed, y_target, cv=kfold)

y_score = LogisticRegression().fit(X_normed, y_target).predict_proba(X)

#classifier = LogisticRegression()
#cv = cross_validate(LogisticRegression(), X_normed, y, cv=kfold) 
#X_train, X_test, y_train, y_test = cross_validate(LogisticRegression(), X_normed, y, cv=kfold)
#y_score = cv.fit(X, y_target).decision_function(X)

#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)



#print("Scores for individual folds: ")
#print(fold_score)
#print("Mean score across folds: " + str(score))
print(y_predict)
print(y_score)


#Initiate models with default parameters
models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

#K-fold cross-validation
names = []
scores = []

for name, model in models:
    
    kfold = KFold(n_splits=10, random_state=10)
    fold_scores = cross_val_score(model, X_normed, y, cv=kfold, scoring='accuracy')
    score = cross_val_score(model, X_normed, y, cv=kfold, scoring='accuracy').mean()
    
    names.append(name)
    scores.append(score)

#Show scores from classifiers in dataframes
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
#print(kf_cross_val)

n_classes = 2

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold = roc_curve(y_predict[i], y_score[i])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()
#plt.savefig('../../figs/pandas/distribution%s.png' % dataset)
plt.close()
'''

random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = SVC(kernel='linear', probability=True,
                     random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()