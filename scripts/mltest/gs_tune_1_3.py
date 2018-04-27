#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:43:25 2018

@author: whb17
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:35:39 2018

@author: whb17
"""

import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

from p2funcs import plot_scatter

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
    

#To show runtime of script
StartTime = time.time()

# Name of script to trace where images came from
scriptname = 'gs_tune_1_3'

#Select current toy dataset
dataset = '013'

#Create directory if directory does not exist
filepath = '../../figs/out/%s/%s/' % (scriptname, dataset)

if not os.path.exists(filepath):
    os.makedirs(filepath)

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
 
#Plot initial data
plot_scatter(X, 
             y, 
             'Initial data', 
             x_label='x coordinate', 
             y_label='y coordinate',
             output='save',
             path='%sinitial.png' % filepath
             )


## PREPROCESSING ##

#Scale initial data to centre data

X_scaled = scale(X)

plot_scatter(X_scaled,
             y, 'Scaled data',
             x_label='x coordinate',
             y_label='y coordinate',             
             output='save',
             path='%sscaled.png' % filepath
             )

# Array for all areas under ROC curve
mean_aucs = dict()

# hyperparameters to test
k_list = list(range(5, 31, 5))
gamma_list = [2e-10, 2e-9, 2e-8, 2e-7, 2e-6, 2e-5, 2e-4]#, 2e-3, 2e-2, 0.2, 2.0]

# Single model chosen for now
svc = SVC(kernel='linear', probability=True)

#Declare lists for drawing ROC
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for gamma in gamma_list:
    
    #RBF KPCA with each gamma
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    X_kpca = kpca.fit_transform(X_scaled)
    
    # Plot each kpca
    plot_scatter(X_kpca,
                 y,
                 'First 2 principal components after RBF KPCA',
                 gamma=gamma,
                 x_label='Principal component 1',
                 y_label='Principal component 2',
                 output='save',
                 path='%srbf_kpca_gamma%s.png' % (filepath, gamma)
                 )
    
    # Declare list of mean ROC AUC for each value of gamma
    mean_auc_row = []
    
    # Update dictionary of all AUC means
    
    mean_aucs.update({gamma:mean_auc_row})
    
    for k in k_list:
        
                cv = StratifiedKFold(n_splits=k)
                
                #print('Peforming %s-fold cross-validated SVC after RBF KPCA (γ = %s)' % (k, gamma))
                
                #plt.figure(figsize=(10, 7))

                
                # To count number of folds
                i = 0
                
                for train, test in cv.split(X_kpca, y):
                       
                    probas_ = svc.fit(X_kpca[train], y[train]).predict_proba(X_kpca[test])
            
                    # Compute ROC curve and area the curve
                    
                    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                    tprs.append(interp(mean_fpr, fpr, tpr))
                    tprs[-1][0] = 0.0
                    roc_auc = auc(fpr, tpr)
                    aucs.append(roc_auc)
                    #plt.plot(fpr, tpr, lw=1, alpha=0.3,
                    #         label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
                
                    i += 1
                        
                #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                #         label='Luck', alpha=.8)
                
                #Calculate means
                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                mean_auc = auc(mean_fpr, mean_tpr)
                mean_auc_row.append(mean_auc)
                
                std_auc = np.std(aucs)
                #plt.plot(mean_fpr, mean_tpr, color='b',
                #         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                #         lw=2, alpha=.8)
                
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                #                 label=r'$\pm$ 1 std. dev.')
                
                #plt.xlim([-0.05, 1.05])
                #plt.ylim([-0.05, 1.05])
                #plt.xlabel('False Positive Rate')
                #plt.ylabel('True Positive Rate')
                #plt.title('Receiver operating characteristic (Using RBF KPCA, γ = %s, k = %s)' % (gamma, k))
                #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                #plt.show()
                #plt.savefig('%sroc_rbf_kpca_gamma%s.png' % (filepath, gamma))
                #plt.close()
        
#print(mean_aucs.keys())

# Plot surface
ma_array = []

for i in gamma_list:
    ma_array.append(mean_aucs[i])

ma_array = np.array(ma_array)

ma_array = ma_array.reshape(len(k_list),len(gamma_list))



X, Y = np.meshgrid(gamma_list, k_list)

print('\nShape of mean auc list:' + str(ma_array.shape) + '\nDimensions in mean auc list: ' + str(ma_array.ndim))
print(ma_array)
print('\nShape of gamma list:' + str(X.shape) + '\nDimensions in gamma list: ' + str(X.ndim))
print(X)
print('\nShape of k list:' + str(Y.shape) + '\nDimensions in k list: ' + str(Y.ndim))
print(Y)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, 
                       Y,
                       Z=ma_array,
                       cmap=cm.coolwarm,                       
                       #linewidth=0,
                       antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('$\gamma$', fontsize=20)
ax.set_ylabel('Number of folds in cross-validation', fontsize=20)
ax.set_zlabel('Mean area under ROC curve', fontsize=20)

plt.title('Change in area under ROC curve with varying gamma in RBF KPCA and varying K in k-fold cross-validation')

plt.show()
#Calculate and display time taken or script to run 
EndTime = (time.time() - StartTime)
print('\nTime taken for script to run is %.2f seconds' % EndTime)
