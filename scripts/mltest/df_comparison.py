#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:19:01 2018

@author: whb17
"""

import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

from p2funcs import plot_scatter, target_split, distribution_boxplot


from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import KernelPCA, PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from scipy import interp

def target_split(df, col):
    
    df_mat = df.as_matrix()
    #df_mat = df_mat.T
    rows, cols = df_mat.shape    
    target = df_mat[:, [col]]
    target = target.reshape(1, len(target))    
    data = df_mat[:, 0:cols-1]
    
    return(data, target[0])
    
def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', output=None, path=None):
    
    plt.figure(figsize=(8, 6))
    
    cata = plt.scatter(x,
                       y,
                       color='red',
                       marker = '^',
                       alpha=0.5
                       )
    
    catb = plt.scatter(x,
                       y,
                       color='blue',
                       marker = 's',
                       alpha=0.5)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('y coordinate')    
    gamma_label = mpatches.Patch(color='white', label='gamma')
    plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category A', 'Category B'])
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    else:
        pass
        
    plt.close() 

def distribution_boxplot(df, targ, a_title, b_title, output=None, path=None):
    
    plt.figure(figsize=(50,15))
    
    plt.subplot(2, 1, 1)
    img1 = sns.boxplot(data=df[targ==1])
    plt.title(a_title, fontsize=20)
    
    plt.subplot(2, 1, 2)
    img2 = sns.boxplot(data=df[targ==0])
    plt.title(b_title, fontsize=20)
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    else:
        pass
        
    plt.close() 
    
inp_csv = pd.read_csv('../../data/simulated/mvnsim/mvnsim015.csv', sep=',', header=0, index_col=0)
inp_target = np.load('../../data/simulated/mvnsim/target015.npy')

X = pd.read_csv('../../data/simulated/mvnsim/mvnsim013.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target013.npy')

print('\nShape of unsplit mvn dataframe: %s\n' % (inp_csv.shape,))
#print(inp_csv)
print('\nShape of make_classification dataframe: %s\n' % (X.shape,))
#print(X)
print('\nShape of make_classification target array: %s\n' % (y.shape,))
#print(y)

#X2, y2 = target_split(inp_csv, 500)
print('\nShape of mvn dataframe: %s\n' % (X2.shape,))
#print(X2)
print('\nShape of make_classification target array: %s\n' % (y2.shape,))
#print(y2)

plt.figure(figsize=(50,15))

plt.subplot(2, 1, 1)
img1 = sns.boxplot(data=inp_csv[inp_target==1])
plt.title("Category A distribution of dataset 015", fontsize=20)

plt.subplot(2, 1, 2)
img2 = sns.boxplot(data=inp_csv[inp_target==0])
plt.title("Category B distribution of dataset 015", fontsize=20)

#plt.savefig('../../data/simulated/mvnsim/mvnsim%sdist.png' % simname)
plt.show()

plt.close()

distribution_boxplot(inp_csv,
                     inp_target,
                     "Category A distribution of dataset 015",
                     "Category B distribution of dataset 015",
                     output='show',
                     )
