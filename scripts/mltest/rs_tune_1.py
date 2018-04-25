#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:35:39 2018

@author: whb17
"""

import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy import interp

def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', output='show', path=None):
    
    plt.figure(figsize=(8, 6))
    
    cata = plt.scatter(x[y==0, 0],
                       x[y==0, 1],
                       color='red',
                       alpha=0.5
                       )
    
    catb = plt.scatter(X[y==1, 0],
                       X[y==1, 1],
                       color='blue',
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
        
    plt.close()    
    

#To show runtime of script
StartTime = time.time()

# Name of script to trace where images came from
scriptname = 'hp_tune1'

#Select current toy dataset
dataset = '013'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
 
#Plot initial data
plot_scatter(X, 
             y, 
             'Initial data', 
             x_label='x coordinate', 
             y_label='y coordinate', 
             #output='save',
             #path='../../figs/out/%s/%s/initial.png' % (scriptname, dataset)
             )






#Calculate and display time taken or script to run 
EndTime = (time.time() - StartTime)
print('Time taken for script to run is %.2f seconds' % EndTime)