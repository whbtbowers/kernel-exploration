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
scriptname = 'final1'

#List of datasets to test
#inp_dataset_list = ['diabetes', 'sex', 'cac_binomial', 'cac_extremes', 'family_hx_diabetes', 'parent_cvd_65_hx', 'family_hx_cvd', 'bp_treatment', 'diabetes_treatment', 'lipids_treatment', 'plaque']
#inp_dataset_list = ['diabetes', 'sex', 'cac_binomial']
inp_dataset_list = ['diabetes']

#Split data list
spl_data = []

# Collect optimal gamma from each dataset
opt_t1_gammas = []
opt_t2_gammas = []

#Using first input dataset to generate toy datasets
inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcol_%s.csv' % inp_dataset_list[0], sep=',', header=None, index_col=0)

print('\nUsing %s dataset to generate toy datasets for the purpose of tuning algorithms and hyperperameters.' %inp_dataset_list[0] )
X_imp = p2f.filt_imp(inp_df, 0.1)
X, y = p2f.tsplit(X_imp)
toy_dataset_list, toy_y = p2f.toybox_gen(X)

for toy_label, toy_X in toy_dataset_list:
    
    print('\n##### Now running dataset %s #####' %toy_label)
          
    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, toy_label)
    plotpath = '%splotting/' % filepath
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.makedirs(plotpath) 
    
    toy_X_scaled = scale(toy_X)
    
    toy_X_rows, toy_X_cols = toy_X_scaled.shape
    
    #default gamma
    def_gamma = 1/toy_X_cols
    
    #Tier1 gamma values
    t1_gamma_list = [def_gamma/10000, def_gamma/1000, def_gamma/100, def_gamma/10, def_gamma, def_gamma*10, def_gamma*100, def_gamma*1000, def_gamma*10000, def_gamma*10000]
    
    print('\n### TIER 1 GRID SEARCH ###')
    
    #Declare list for t1 mean mean area under ROC curve (mma) values        
    t1_mmas = []
    
    # Create initial variables to update
    max_t1_mma = 0
    opt_t1_gamma = 0
    
    for gamma in t1_gamma_list:
        
        t1_auc_mat, t1_kpcas, t1_models  = p2f.m_test5_2(toy_X_scaled, toy_y, gamma, toy_label, filepath, 'tier1')
        
        t1_mmas.append(np.mean(t1_auc_mat))
        
    # Select optimal t1 gamma 
    for i in range(len(t1_mmas)):
        if t1_mmas[i] > max_t1_mma:
            max_t1_mma = t1_mmas[i]
            opt_t1_gamma = t1_gamma_list[i]
            
    # Show optimal gamma
    #print('Optimal Tier 1 gamma for dataset %s found to be %s' %(dataset, opt_t1_gamma))
    opt_t1_gammas.append(opt_t1_gamma)
    print("\n%.2f seconds elapsed so far\n" % (time.time() - StartTime))
    
# End of dataset run     
print('\n###################################################################')

# Print aggregate gamma values
for i in range(len(opt_t1_gammas)):
    print('\nOptimal Tier 1 gamma for dataset %s found to be %s' %(toy_dataset_list[i], opt_t1_gammas[i]))       

# Count number of each gamma    
t1_gcount_dict = dict((x,opt_t1_gammas.count(x)) for x in set(opt_t1_gammas))

# Find most frequent gamma value
t1_gamma_consensus = max(t1_gcount_dict, key=t1_gcount_dict.get)

#Just in case last value selected:
if t1_gamma_consensus == t1_gamma_list[-1]:
    t1_gamma_consensus = t1_gamma_list[-2]

# Create tier 2 gamma list
gamma_i_t1 = t1_gamma_list.index(t1_gamma_consensus)
t2_gamma_list = list(p2f.frange(t1_gamma_list[gamma_i_t1-1], t1_gamma_list[gamma_i_t1], t1_gamma_list[gamma_i_t1-1])) + list(p2f.frange(t1_gamma_list[gamma_i_t1], t1_gamma_list[gamma_i_t1+1], t1_gamma_list[gamma_i_t1]))   


amat_dict_list = []
t2_kpcas = 0
t2_models = 0

for toy_label, toy_X in toy_dataset_list:
    
    # Dict of gammas w/ t2 Matrices
    amat_dict = dict()
    
    #Scale initial data to centre    
    toy_X_scaled = scale(toy_X)
    
    #Declare list for t2 mean mean area under ROC curve (mma) values        
    t2_mmas = []
    
    # Create initial variables to update
    max_t2_mma = 0
    opt_t2_gamma = 0
    
    print('\n### TIER 2 GRID SEARCH ###')
    
    for gamma in t2_gamma_list:
        
        amat_dict.update({gamma:[]})
        
        t2_auc_mat, t2_kpcas, t2_models  = p2f.m_test5_2_rocplot(toy_X_scaled, toy_y, gamma, toy_label, filepath, 'tier2', plotpath)
        
        amat_dict[gamma].append(t2_auc_mat)
        
        t2_mmas.append(np.mean(t2_auc_mat))

    amat_dict_list.append(amat_dict)
    
    # Select optimal t2 gamma 
    for i in range(len(t2_mmas)):
        if t2_mmas[i] > max_t2_mma:
            max_t2_mma = t2_mmas[i]
            opt_t2_gamma = t2_gamma_list[i]
    
    # Record optimal gamma for given dataset        
    opt_t2_gammas.append(opt_t2_gamma)
    print("\n%.2f seconds elapsed so far\n" % (time.time() - StartTime))
    
# Count number of each gamma    
t2_gcount_dict = dict((x,opt_t2_gammas.count(x)) for x in set(opt_t2_gammas))

# Find most frequent gamma value
t2_gamma_consensus = max(t2_gcount_dict, key=t2_gcount_dict.get)
gamma_i_t2 = t2_gamma_list.index(t2_gamma_consensus)
opt_gamma = t2_gamma_list[gamma_i_t2]

#Declare arrays to find most common methods
opt_kpcas_byg = []
opt_models_byg = []

#Find highest mean AUC in each dataset for 'optimal' gamma
for i in range(len(amat_dict_list)):
    mat_dict = amat_dict_list[i]
    choice_auc_mat = mat_dict[opt_gamma]
    max_auc = np.max(choice_auc_mat)
    kpca_index, model_index = np.where(choice_auc_mat == max_auc)
    opt_kpca_byg = t2_kpcas[kpca_index]
    opt_kpcas_byg.append(opt_kpca_byg)
    opt_model_byg = t2_models[model_index]
    opt_models_byg.append(opt_model_byg)
    
    print('\nAt γ = %s, optimal KPCA is %s, whereas optimal KSVM is %s' % (opt_gamma, opt_kpca_byg, opt_model_byg))

opt_kpca = p2f.most_common(opt_kpcas_byg)
opt_model = p2f.most_common(opt_models_byg)

print('\nOverall optimal KPCA/KSVM combination determined to be %s and %s respectively (γ = %s).' %s (opt_kpca, opt_model, gamma))    
            
# End of dataset run       
print("\n%.2f seconds elapsed so far\n" % (time.time() - StartTime)) 
print('\n###################################################################\n')
      
print("~~~~~~~~~~~~ REAL DATA ~~~~~~~~~~~~")

rrun_aucs = []

for dataset in inp_dataset_list:
    
    print('\nNow running with %s dataset.' % dataset)
    inp_df = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcol_%s.csv' % dataset, sep=',', header=None, index_col=0)
    X_imp = p2f.filt_imp(inp_df, 0.1)
    X, y = p2f.tsplit(X_imp)
    
    rrun_mean_auc, rrun_kpca, rrun_model  = m_run5_3(X, y, opt_gamma, opt_kernel, opt_model, dataset, filepath, 'rrun')
    
    rrun_aucs.append(rrun_mean_auc)
    
    print('After this run with %s followed by %s (γ = %s), mean auc is %s' % (rrun_kpca, rrun_model, opt_gamma, rrun_mean_auc))

#Calculate and display time taken or script to run
print("\nTime taken for script to run is %.2f seconds\n" % (time.time() - StartTime))