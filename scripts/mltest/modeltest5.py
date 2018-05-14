#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:31:23 2018

@author: whb17
"""
import operator
import time
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import p2funcs as p2f
from scipy import interp

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
from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel

from scipy import interp


#To show runtime of script
StartTime = time.time()

# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")

# Name of script to trace where images came from
scriptname = 'modeltest5'

#List of toy datasets to test
dataset_list = ['022', '023', '024']
#dataset_list = ['024']
#dataset_list = ['mesa']

#provide values of gamma to test for all kernel methods
t1_gamma_list = [2e-7, 0.000002, 0.00002, 0.0002, 0.002, 0.02, 0.2, 2.0]

# Collect optimal gamma from each dataset
opt_t1_gammas = []
opt_t2_gammas = []

for dataset in dataset_list: 

    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    #Import data and target
    if dataset == 'mesa':
        X = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',')
        
        X = p2f.filt_imp(X, 0.1)
        
        y = np.load('../../data/mesa/mesatarget.npy')
    else:
        X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)
        y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
                
        #Scale initial data to centre data    
        X_scaled = scale(X)
        
        #Declare list for t1 mean mean area under ROC curve (mma) values        
        t1_mmas = []
        
        # Create initial variables to update
        max_t1_mma = 0
        opt_t1_gamma = 0
        
        for gamma in t1_gamma_list:
            
            t1_auc_mat, t1_kpcas, t1_models  = p2f.m_test5(X_scaled, y, gamma, dataset)
            
            t1_mmas.append(t1_auc_mat.mean())
        
        # Select optimal t1 gamma 
        for i in range(len(t1_mmas)):
            if t1_mmas[i] > max_t1_mma:
                max_t1_mma = t1_mmas[i]
                opt_t1_gamma = t1_gamma_list[i]
                
        # Show optimal gamma
        #print('Optimal Tier 1 gamma for dataset %s found to be %s' %(dataset, opt_t1_gamma))
        opt_t1_gammas.append(opt_t1_gamma)

    # End of dataset run        
    print('\n###################################################################')

for i in range(len(opt_t1_gammas)):
    print('Optimal Tier 1 gamma for dataset %s found to be %s' %(dataset_list[i], opt_t1_gammas[i]))       
    
gcount_dict = dict((x,opt_t1_gammas.count(x)) for x in set(opt_t1_gammas))

gamma_consensus = max(gcount_dict, key=gcount_dict.get)

gamma_i = t1_gamma_list.index(gamma_consensus)

t2_gamma_list = list(p2f.frange(t1_gamma_list[gamma_i], t1_gamma_list[gamma_i+1], t1_gamma_list[gamma_i]))
 
print(t2_gamma_list)   
    
#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print("\nTime taken for script to run is %.2f seconds\n" % EndTime)