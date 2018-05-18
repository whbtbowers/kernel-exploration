#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 14:32:26 2018

@author: whb17


"""

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
scriptname = 'roc_kpca2_2_mesa'

#List of toy datasets to test
dataset_list = ['022', '023', '024']
#dataset_list = ['024']
dataset_list = ['MESA']

for dataset in dataset_list: 

    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/mesa/' % (scriptname, nowdate)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    #Import toy data and target
    inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcolOC.csv', header=None, index_col=0, sep=',')
    #y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
    
    #print(inp_df)
    
    X_imp = p2f.filt_imp(inp_df, 0.1)
    
    X_imp_df = pd.DataFrame.from_records(X_imp)
    
    print(X_imp_df)
    
    X, y = p2f.tsplit(X_imp_df)
    y = y.astype(int)
    n_cols, n_rows = X.shape
    
    
    '''    
    p2f.distribution_boxplot(X,
                         y,
                         "Initial category 1 distribution of dataset %s" % dataset,
                         "Initial category 0 distribution of dataset %s" % dataset,
                         #output='show',
                         #output='plotly',
                         #ply_title="Initial distribution of dataset %s" % dataset,
                         output='save',
                         path='%sinitialdist.png' % filepath,
                         )
    
    print('\nBoxplot of initial data for dataset %s saved.' % dataset)
    
    '''
    
    ## PREPROCESSING ##
    
    #Scale initial data to centre data
    
    X_scaled = scale(X)
    print('\nData have been scaled')
    #print(X_scaled)
    #X_scaled_df = pd.DataFrame.from_records(X_scaled)
    
    '''
    distribution_boxplot(X_scaled_df,
                         y,
                         "Scaled category 1 distribution of dataset %s" % dataset,
                         "Scaled category 0 distribution of dataset %s" % dataset,
                         #output='show'
                         output='save',
                         path='../../figs/out/%s/%s/scaledist.png' % (scriptname, dataset)
                         )
    #print(X_scaled.shape)
    print('\nBoxplot of scaled data for dataset %s saved.' % dataset)
    '''
    
    #Initiate KPCAwith various kernels
    
    # As I'm using 500 variables, 0.002 is the default gamma (1/n_variables)
    # I only explicitly state it at this point so I can display it on graphs
    gamma = 1/n_cols
    
    #compute kernels not preloaded into kpca
    #laplacian
    kpca_lap = laplacian_kernel(X_scaled, gamma=gamma) 
    #kpca_chi = chi2_kernel(X, gamma=gamma)
    
    kpcas = []
    
    #Use standard PCA for comparison
    
    #kpcas.append(('standard ', 'std_', PCA(n_components=2)))
    
    #Linear kernal has no need for gamma
    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'prec_lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    #kpcas.append(('Chi Squared KPCA', 'prec_chi_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Polynomial KPCA', 'ply_kpca', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))
    
    #Initiate models with default parameters
    
    models = []
    
    models.append(('SVC', SVC(kernel='linear', probability=True)))
    
    
    cv = StratifiedKFold(n_splits=10, random_state=10)
    
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    
    
    # Declare KPCA kernels deployed
    
    kpca_kernels = []
    
    for kernel, abbreviation, kpca in kpcas:

        # To utilise precomputed kernel(s)
        if kernel == 'Laplacian KPCA':
            X_kpca = kpca.fit_transform(kpca_lap)
        #elif kernel == 'Chi Squared KPCA':
        #    X_kpca = kpca.fit_transform(kpca_chi)
        else:
            X_kpca = kpca.fit_transform(X_scaled)
    
        p2f.plot_scatter(X_kpca,
                         y,
                         'First 2 principal components after %s' % kernel,
                         gamma=gamma,
                         x_label='Principal component 1',
                         y_label='Principal component 2',
                         #output = 'show',
                         output='save',
                         path='%s%s_%s_gamma%s.png' % (filepath, nowtime, abbreviation, gamma)
                         )
        
        print('\nScatter plot of first two principal components after %s for dataset %s saved.' % (kernel, dataset))
    
        kpca_kernels.append(kernel)
    
    
        # Declare names of models deployed
        mdl_names = []
        plt.figure(figsize=(15, 9))
    
        for name, model in models:
    
            mdl_names.append(name)
            print('\nPerforming ' + name + ' with ' + kernel)
            #print(mdl_names)
    
            # To count number of folds
            i = 0
    
            for train, test in cv.split(X_kpca, y):
    
                probas_ = model.fit(X_kpca[train], y[train]).predict_proba(X_kpca[test])
    
                # Compute ROC curve and area the curve
    
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
    
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
            plt.title('Receiver operating characteristic (Using %s, γ = %s)' % (kernel, gamma))
            plt.legend(loc="lower right")
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #plt.show()
            plt.savefig('%s%s_roc_%s_gamma%s.png' % (filepath, nowtime, abbreviation, gamma))
            plt.close()

    
            
#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print('\nTime taken for script to run is %.2f seconds\n' % EndTime)
