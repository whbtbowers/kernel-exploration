#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:35:31 2018

@author: whb17
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
sns.set(style="ticks")
sns.set(style='white')

import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

import pandas as pd
import numpy as np
 
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import scale, normalize, Imputer
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


def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', output=None, path=None, ply_title=None):
    
    fig = plt.figure(figsize=(8, 6))
    
    cata = plt.scatter(x[y==0, 0],
                       x[y==0, 1],
                       color='red',
                       marker = '^',
                       alpha=0.5
                       )
    
    catb = plt.scatter(x[y==1, 0],
                       x[y==1, 1],
                       color='blue',
                       marker = 's',
                       alpha=0.5)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)    
    gamma_label = mpatches.Patch(color='white', label='gamma')
    plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), 'Category 1', 'Category 0'])
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    else:
        pass
        
    plt.close() 

def target_split(df, col):
    
    df_mat = df.as_matrix()
    #df_mat = df_mat.T
    rows, cols = df_mat.shape    
    target = df_mat[:, [col]]
    target = target.reshape(1, len(target))    
    data = df_mat[:, 0:cols-1]
    
    return(data.T, target[0])

def distribution_boxplot(df, targ, a_title, b_title, output=None, path=None, ply_title=None):
    
    boxplot = plt.figure(figsize=(50,15))
    
    plt.subplot(2, 1, 1)
    sns.boxplot(data=df[targ==1])
    plt.title(a_title, fontsize=20)
    
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df[targ==0])
    plt.title(b_title, fontsize=20)
    
    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
    elif output == 'plotly':
        plotly_fig = tls.mpl_to_plotly(boxplot)
        py.iplot(boxplot, filename=ply_title)
    else:
        pass
        
    plt.close()
    
def filt_imp(X, threshold):
    
    # Make list of sample IDs
    row_names = X.index.values.tolist()
    
    # Binarise 'Yes' and 'No'
    X = X.replace('Yes', 1)
    X = X.replace('No', 0)
    
    # Display metrics for initial data
    n_cols, n_rows = X.shape
    print('\nInitial data contains %s columns and %s rows.' % (n_cols, n_rows))
    col_names = list(X)
    #print('Categories:')
    #print(col_names)
    
    # Remove ID column and display metrics
    #X = X.drop(columns='study')
    
    #n_cols, n_rows = X.shape
    #print("\nAfter dropping 'study' column, data contains %s columns and %s rows." % (n_cols, n_rows))
    
    #col_names = list(X)
    
    # Initiate counter to count number of null values per column
    dropped_cols = 0
    
    # Find columns containing <10% of filled fields
    for i in col_names:
        
        # Initiate counter to count number of null values per column
        null_cells = 0
        
        for j in X[i]:
            if pd.isnull(j) == True:
                null_cells += 1
    
        # Remove column if more than 10% values empty
        if null_cells / len(X[i]) >= 1-threshold:
            X = X.drop(columns=i)
            dropped_cols += 1
    
    n_cols, n_rows = X.shape        
    print('\n%s columns in dataset removed due to ≤10%% of cells populated.' % dropped_cols)
    print('\nAfter columns ≤ 10%% populated removed, data contains %s columns and %s rows.' % (n_cols, n_rows))
    
    col_names = list(X)
    
    # Count number of rows removed
    dropped_rows = 0
    
    #Find rows containing <10% of filled fields
    for i in row_names:
    
    #    print(X.loc[i])
        # Initiate counter to count number of null values per row
        nulls = 0
    
        for j in X.loc[i]:
            if pd.isnull(j) == True:
                nulls += 1
        # Remove row if more than 10% values empty
        if nulls/len(X.loc[i]) >= 1-threshold:
            print(i)
            X = X.drop(index=i)
            dropped_rows += 1
    
    n_cols, n_rows = X.shape
    print('\n%s rows in remaining dataset removed due to ≤10%% of cells populated.' % dropped_rows)
    print('\nAfter columns and rows ≤ 10%% populated removed, data contains %s columns and %s rows.' % (n_cols, n_rows))
    
    
    # Convert dataframe to numpy matrix for scikit learn
    X = X.as_matrix()
    
    # Uses mean as imputation strategy
    impute = Imputer(strategy='mean')
    X_imputed = impute.fit_transform(X)
    
    #print(X_imputed.shape)
    n_cols, n_rows = X_imputed.shape
    print('\nAfter imputation, data contains %s columns and %s rows.' % (n_cols, n_rows))
    
    return(X_imputed)

# Like range but for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump    

def writetext(content, filename, path):
	os.chdir(path)
	text_file = open(filename, "w")
	text_file.write(content)
	text_file.close()
    
def m_test5(X, y, gamma, dataset):
    
    #Record all mean roc_aucs for each gamma value
    auc_mat = []

    
    #compute kernels not preloaded into kpca
    #laplacian
    K_lap = laplacian_kernel(X, gamma=gamma)
    
    kpcas = []

    #Use standard PCA for comparison
    
    #kpcas.append(('standard PCA', 'std_', PCA(n_components=2)))
    
    #Linear kernal has no need for gamma
    kpcas.append(('Linear KPCA', 'lin_k', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_k',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'lap_k',KernelPCA(n_components=2, kernel='precomputed')))    
    kpcas.append(('Polynomial KPCA', 'ply_k', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
    kpcas.append(('Sigmoid KPCA', 'sig_k', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_k',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))
    
    #Initiate models with default parameters

    models = []
    
    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('Polynomial Kernel SVM','ply_svc', SVC(kernel='poly', gamma=gamma, probability=True)))
    models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))
    models.append(('K-Nearest Neighbour', 'knn', KNeighborsClassifier()))
    models.append(('Logistic Regression', 'log_reg', LogisticRegression()))
    models.append(('Decision Tree', 'dec_tree', DecisionTreeClassifier()))
    #models.append(('Gaussian Naive Bayes', 'gnb', GaussianNB()))
    models.append(('Random Forest', 'rf', RandomForestClassifier()))
    models.append(('Gradient Boosting', 'gb', GradientBoostingClassifier()))
    
    # Initiate cross-validation
    folds = 10    
    cv = StratifiedKFold(n_splits=folds, random_state=10)
    
    # Declare KPCA kernels deployed

    kpca_kernels = []
    
    for kernel, abbreviation, kpca in kpcas:
        
        # To utilise precomputed kernel(s)
        if kernel == 'Laplacian KPCA':
            X_kpca = kpca.fit_transform(K_lap)
        else:
            X_kpca = kpca.fit_transform(X)
        
        '''        
        p2f.plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%s_%spca_gamma%s.png' % (filepath, nowtime, abbreviation, gamma)
                     )
        print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))
        '''    
        
        kpca_kernels.append(kernel)
    
        # Declare names of models deployed
        mdl_names = []        
        
        #Record mean_aucs
        auc_mat_row = []
               
        for model_name, model_abv, model in models:
            
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            
            mdl_names.append(model_name)
            print('\nPerforming %s followed by %s for dataset %s\n' % (kernel, model_name, dataset))
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
            auc_mat_row.append(mean_auc)
            std_auc = np.std(aucs)
                
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    
               
            # Display mean roc auc
            #print("Mean area under curve for %s followed by %s (γ = %s): %0.2f" % (kernel, model_name, gamma, mean_auc))
    
        auc_mat.append(auc_mat_row)     

    auc_mat = np.array(auc_mat)
    
    return(auc_mat, kpca_kernels, mdl_names)
