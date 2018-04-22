import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from scipy import interp


#Select current toy dataset
dataset = '010'

#Import toy data and target
X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0).as_matrix()
y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')

#Initiate KPCAwith various kernels

# As I'm using 500 variables, this is the default gamma (1/n_variables)
# I only explicitly state it at this point so I can dispaly it on graphs
gamma = 0.002

kpcas = []

#Linear kernal has no need for gamma
kpcas.append(('Linear', KernelPCA(n_components=2, kernel='linear',)))
kpcas.append(('RBF', KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
kpcas.append(('Polynomial', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
kpcas.append(('Sigmoid', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
kpcas.append(('Cosine', KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

#Initiate models with default parameters#

models = []

models.append(('SVC', SVC(kernel='linear', probability=True)))


cv = StratifiedKFold(n_splits=10, random_state=10)


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0

# Declare KPCA kernels deployed

kpca_kernels = []

for kernel, kpca in kpcas:
    
    X_kpca = kpca.fit_transform(X)
    
    kpca_kernels.append(kernel)
    print(kpca_kernels)
      
    # Declare names of models deployed
    mdl_names = []
    plt.figure(figsize=(10, 10))
    
    for name, model in models:
        
        mdl_names.append(name)
        print(mdl_names)
        
        for train, test in cv.split(X_kpca, y):
               
            probas_ = model.fit(X_kpca[train], y[train]).predict_proba(X_kpca[test])
    
            # Compute ROC curve and area the curve
            
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
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
        plt.title('Receiver operating characteristic (Using %s KPCA)' % kernel)
        plt.legend(loc="lower right")
        #plt.show()
        plt.savefig('../../figs/mvnfigs/roc_curves/roc_%s_kpca_dataset%s.png' % (kernel, dataset))
        plt.close()
