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


# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")

# Name of script to trace where images came from
scriptname = 'mesa_lin_kpca'

#List of datasets to test
dataset_list = ['024']

for dataset in dataset_list:
    
    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)
    
    y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
    
    X_cols, X_rows = X.shape
    
    print(X_cols)
    '''   
    gamma 
    
    kpcas = []
    
    kpcas.append(('Linear KPCA', 'lin_k', KernelPCA(n_components=2, kernel='linear')))
    
    for kernel, abbreviation, kpca in kpcas:

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
    '''