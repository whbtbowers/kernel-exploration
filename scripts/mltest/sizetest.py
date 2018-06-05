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
scriptname = 'sizetest'

inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',', header=None, index_col=0)
    #Define size of component using input data

inp_df = p2f.filt_imp(inp_df, 0.1)

ds_list, y_list = p2f.multisize_toybox(inp_df)

for i in range(len(y_list)):

    y = y_list[i]
    ds_label, X = ds_list[i]

    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, ds_label)
    plotpath = '%splotting/' % filepath

    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.makedirs(plotpath)

    X_rows, X_cols =  X.shape

    def_gamma = 1/X_cols

    gamma_list = [def_gamma/100, def_gamma/10, def_gamma]

    for gamma in gamma_list:
        
        auc_mat, kpcas, models  = p2f.m_test5_2_rocplot(X, y, gamma, ds_label, filepath, plotpath, 'gamma_%s' %gamma)

        p2f.plot_mpl_heatmap(auc_mat, kpcas, models, cmap="autumn", cbarlabel="Mean area under ROC curve after 10-fold cross validation", output='save', path='%s%s_tier2_heatmap_gamma_%s.png' % (filepath, nowtime, gamma))

        p2f.js_heatmap(auc_mat, kpcas, models, "hmapgamma%s" % (gamma), "%s%sheatmap_gamma%s.js" % (plotpath, nowtime, gamma))

#Calculate and display time taken or script to run
print("\nTime taken for script to run is %.2f seconds\n" % (time.time() - StartTime))

p2f.endalert('final2_3 successfully completed', "Hey me,\n\nIt's pretty cool because it's successfully finished.\n\nMuch love,\n\nMe")
