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
scriptname = 'tbgenplot'

X = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcol_diabetes.csv', sep=',', header=None, index_col=0)

X_imp = p2f.filt_imp(X, 0.1)

X, y = p2f.tsplit(X_imp)

X_scaled = scale(X)

dataset_list, y = p2f.toybox_gen(X_scaled)

print(dataset_list)

'''
for ds_label, dataset in dataset_list:
    
    print('\n##### Now running dataset %s #####' %dataset)
    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        

    
    p2f.pca_plot(X_scaled, y, dataset, filepath, '%s: 1' % dataset, '%s: 0' % dataset)
    
    # Display user current time
    CurrTime = (time.time() - StartTime)
    print('\nTime taken for script to run so far is %.2f seconds' % CurrTime)

#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print('\nTime taken for script to run is %.2f seconds\n' % EndTime)
'''