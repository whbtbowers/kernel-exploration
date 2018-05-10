#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:40:45 2018

@author: whb17
"""

import datetime
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import p2funcs as p2 

import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

#from p2funcs import plot_scatter, target_split, distribution_boxplot, filt_imp


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
from scipy import interp


#To show runtime of script
StartTime = time.time()

# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")

# Name of script to trace where images came from
scriptname = 'dectree1'

#Select current toy dataset
#dataset = '018'

#mesa = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

#List of toy datasets to test
dataset_list = ['017', '018', '019', '020', '021', '022', '023', '024']
#dataset_list = ['022']
#dataset_list = ['MESA']


for dataset in dataset_list:    
       
    print('\nRunning %s with %s:' % (scriptname, dataset))
    
    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Import  dataset and target
    if dataset == 'MESA':        
        X = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)
        
        X = p2.filt_imp(X, 0.1)     
       
    else:
        X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)
    
        y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
    #print(y)
    #print(y.shape)
    #print(X.shape)
    
     
    
    '''
    p2.distribution_boxplot(X,
                         y,
                         "Initial category 1 distribution of dataset %s" % dataset,
                         "Initial category 0 distribution of distribution %s" % dataset,
                         output='show'
                         #output='plotly',
                         #ply_title="Initial distribution of dataset %s" % dataset,
                         #output='save',
                         path='%s%s_initialdist.png' % (filepath, nowtime)
                         )
    '''
    #print('\nBoxplot of initial data for dataset %s saved.' % dataset)


    ## PREPROCESSING ##
    
    #Scale initial data to centre data
    
    X_scaled = scale(X)
    
    '''
    X_scaled_df = pd.DataFrame.from_records(X_scaled)
    
    p2.distribution_boxplot(X_scaled_df,
                         y,
                         "Scaled category 1 distribution of dataset %s" % dataset,
                         "Scaled category 0 distribution of dataset %s" % dataset,
                         #output='show'
                         #output='save',
                         #path='%s%s_scaledist.png' % (filepath, nowtime)
                         )
    #print(X_scaled.shape)
    #print('\nBoxplot of scaled data for dataset %s saved.' % dataset)
    '''
    #Initiate KPCA with various kernels
    
    # As I'm using 500 variables, 0.002 is the default gamma (1/n_variables)
    gamma = 0.002
    
    kpcas = []
    
    #Use standard PCA for comparison
    
    #kpcas.append(('standard ', 'std_', PCA(n_components=2)))
    
    #Linear kernal has no need for gamma
    kpcas.append(('Linear K', 'lin_k', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF K', 'rbf_k',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    #kpcas.append(('Polynomial K', 'ply_k', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
    #kpcas.append(('Sigmoid K', 'sig_k', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    #kpcas.append(('Cosine K', 'cos_k',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))
    
    #Initiate models with default parameters
    
    models = []
    
    #models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    #models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    #models.append(('K-Nearest Neighbour', 'knn', KNeighborsClassifier()))
    #models.append(('Logistic Regression', 'log_reg', LogisticRegression()))
    models.append(('Decision Tree using Gini Impurity Criterion', 'dec_tree_gini', DecisionTreeClassifier()))
    models.append(('Decision Tree using Information gain criterion', 'dec_tree_ent', DecisionTreeClassifier(criterion='entropy')))
    models.append(('Decision Tree using best random split', 'dec_tree_rndsplt', DecisionTreeClassifier(splitter='random')))
    models.append(('Decision Tree using Information gain criterion with best random split', 'dec_tree_ent_rndsplt', DecisionTreeClassifier(criterion='entropy', splitter='random')))
    #models.append(('Gaussian Naive Bayes', 'gnb', GaussianNB()))
    #models.append(('Random Forest', 'rf', RandomForestClassifier()))
    #models.append(('Gradient Boosting with deviance loss', 'gb_dev', GradientBoostingClassifier()))
    #models.append(('Gradient Boosting with exponential loss', 'gb_exp', GradientBoostingClassifier(loss='exponential')))

    #models.append(('PLS', PLSRegression())) # Scale=False as data already scaled.
        
    cv = StratifiedKFold(n_splits=10, random_state=10)
    
    # Declare KPCA kernels deployed
    
    kpca_kernels = []
    
    for kernel, abbreviation, kpca in kpcas:
    
        X_kpca = kpca.fit_transform(X_scaled)
    
        p2.plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %sPCA' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     output = 'show',
                     #output='save',
                     #path='%s%s_%spca_gamma%s.png' % (filepath, nowtime, abbreviation, gamma)
                     )
        #print('\nScatter plot of first two principal components after %sPCA for dataset %s saved.' % (kernel, dataset))
    
        kpca_kernels.append(kernel)
    
        # Declare names of models deployed and ROC AUC for each model
        #mdl_names = []
        #mean_aucs = []        
        
        for model_name, model_abv, model in models:
            
            #p2.cv_mra(X, y, cv, model, model_name, kernel)
            
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            
            mdl_names.append(model_name)
            print('\nPerforming ' + model_name + ' after ' + kernel + 'PCA')
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
    
                i += 1
    
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            mean_aucs.append(mean_auc)
            std_auc = np.std(aucs)
                
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)         
                       
            # Display mean roc auc
            print("Mean area under curve for %sPCA followed by %s: %0.2f" % (kernel, model_name, mean_auc))
             
        fig = plt.figure(figsize=(8, 6))
        
        plt.bar(mdl_names,
                mean_aucs,
                )
        
        plt.xticks(mdl_names,
                   rotation='vertical',
                   )
        
        plt.show()
        plt.close()
        
        
        
    # End of dataset run        
    print('\n###################################################################')


#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print("\nTime taken for script to run is %.2f seconds\n" % EndTime)
