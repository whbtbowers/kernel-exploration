import datetime
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.tools as tls
tls.set_credentials_file(username='whtbowers', api_key='skkoCIGowBQdx7ZTJMzM')

import p2funcs as p2f


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
from sklearn.metrics.pairwise import laplacian_kernel
from scipy import interp


#To show runtime of script
StartTime = time.time()

# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")

# Name of script to trace where images came from
scriptname = 'gs_tune_2'

#Select current toy dataset
#dataset = '018'

#mesa = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

#List of toy datasets to test
#dataset_list = ['017', '018', '019', '020', '021', '022', '023', '024']
dataset_list = ['023']
#dataset_list = ['MESA']

# hyperparameters to test
#gamma_list = [2e-10, 2e-9, 2e-8, 2e-7, 2e-6, 2e-5, 2e-4, 2e-3, 2e-2, 0.2, 2.0]
gamma_list = list(p2f.frange(0.002, 0.02, 0.002))    

for dataset in dataset_list:    
       
    print('\nRunning %s with %s:' % (scriptname, dataset))
    
    #Create directory if directory does not exist
    #filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    filepath = '../../figs/out/%s/%s/' % (scriptname, nowdate)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Import  dataset and target
    if dataset == 'MESA':        
        X = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)
        
        X = p2f.filt_imp(X, 0.1)     
       
    else:
        X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)
    
        y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
    #print(y)
    #print(y.shape)
    #print(X.shape)
    
    #dictionary for comparison of gamma values with AUC
    model_dict = dict()
    
    # Collect mean roc area under curve
    mean_aucs = []
    
    '''
    distribution_boxplot(X,
                         y,
                         "Initial category 1 distribution of dataset %s" % dataset,
                         "Initial category 0 distribution of distribution %s" % dataset,
                         #output='show'
                         #output='plotly',
                         #ply_title="Initial distribution of dataset %s" % dataset,
                         #output='save',
                         #path='%sinitialdist_%s.png' % (filepath, nowtime)
                         )

    #print('\nBoxplot of initial data for dataset %s saved.' % dataset)
    '''

    ## PREPROCESSING ##
    
    #Scale initial data to centre data
    
    X_scaled = scale(X)
    #X_scaled_df = pd.DataFrame.from_records(X_scaled)
    
    '''
    distribution_boxplot(X_scaled_df,
                         y,
                         "Scaled category 1 distribution of dataset %s" % dataset,
                         "Scaled category 0 distribution of dataset %s" % dataset,
                         #output='show'
                         #output='save',
                         #path='%sscaledist_%s.png' % (filepath, nowtime)
                         )
    #print(X_scaled.shape)
    #print('\nBoxplot of scaled data for dataset %s saved.' % dataset)
    '''
    
    #Initiate KPCAwith various kernels
    
    # As I'm using 500 variables, 0.002 is the default gamma (1/n_variables)
    # I only explicitly state it at this point so I can display it on graphs
    #gamma = 0.002
    
    for gamma in gamma_list:
        
        # Collect mean roc area under curve
        #mean_aucs = []
        
        #Add entry to dictionary to compare effect of changing k
        #gamma_dict.update({gamma : []})
        
        #compute kernels not preloaded into kpca
        #laplacian
        K_lap = laplacian_kernel(X_scaled, gamma=gamma)   
        
        kpcas = []
        
        #Use standard PCA for comparison
        
        #kpcas.append(('standard PCA', 'std_', PCA(n_components=2)))
        
        #Linear kernal has no need for gamma
        #kpcas.append(('Linear KPCA', 'lin_k', KernelPCA(n_components=2, kernel='linear')))
        kpcas.append(('RBF KPCA', 'rbf_k',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
        #kpcas.append(('Laplacian KPCA', 'lap_k',KernelPCA(n_components=2, kernel='precomputed')))
        
        #kpcas.append(('Polynomial KPCA', 'ply_k', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
        #kpcas.append(('Sigmoid KPCA', 'sig_k', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
        #kpcas.append(('Cosine KPCA', 'cos_k',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))
        
        #Initiate models with default parameters
        
        models = []
        
        models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
        #models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
        #models.append(('K-Nearest Neighbour', 'knn', KNeighborsClassifier()))
        #models.append(('Logistic Regression', 'log_reg', LogisticRegression()))
        #models.append(('Decision Tree', 'dec_tree', DecisionTreeClassifier()))
        #models.append(('Gaussian Naive Bayes', 'gnb', GaussianNB()))
        #models.append(('Random Forest', 'rf', RandomForestClassifier()))
        #models.append(('Gradient Boosting', 'gb', GradientBoostingClassifier()))
    
        #models.append(('PLS', PLSRegression())) # Scale=False as data already scaled.
        
        folds = 2
        
        cv = StratifiedKFold(n_splits=folds, random_state=10)
        
        # Declare KPCA kernels deployed
        
        kpca_kernels = []
        
        for kernel, abbreviation, kpca in kpcas:
            
            # To utilise precomputed kernel(s)
            if kernel == 'Laplacian KPCA':
                X_kpca = kpca.fit_transform(K_lap)
            else:
                X_kpca = kpca.fit_transform(X_scaled)
                
            '''
            plot_scatter(X_kpca,
                         y,
                         'First 2 principal components after %s' % kernel,
                         gamma=gamma,
                         x_label='Principal component 1',
                         y_label='Principal component 2',
                         #output = 'show',
                         #output='save',
                         #path='%s%s_%spca_gamma%s.png' % (filepath, nowtime, abbreviation, gamma)
                         )
            print('\nScatter plot of first two principal components after %s for dataset %s saved.' % (kernel, dataset))
            '''
            
            kpca_kernels.append(kernel)
        
            # Declare names of models deployed
            mdl_names = []
            mdl_abvs = []
            #mean_aucs = []
        
            for model_name, model_abv, model in models:
                
                model_dict.update({model_name:[]})
                
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)
                
                mdl_names.append(model_name)
                mdl_abvs.append(model_abv)
                print('\nPerforming %s after %s ' % (model_name, kernel))
                #print(mdl_names)
        
                # To count number of folds
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
                print("Mean area under curve for %s followed by %s: %0.2f" % (kernel, model_name, mean_auc))
            
                # Add ROC and mean aucs to gamma_dict lists 
                #gamma_dict[gamma].append(mean_aucs)
                #gamma_dict[gamma].append(mdl_names)
                
                #Update model list
                model_dict[model_name].append(mean_auc)
                model_dict[model_name].append(gamma)
                
            
    #print(mdl_names)
    #print(mean_aucs)
            
    #print(gamma_dict)
    
    for i in range(len(mdl_names)):
        
        fig = plt.figure(figsize=(8, 6))
        
        plt.bar(list(range(len(gamma_list))),
                mean_aucs,
                )
        
        plt.xticks(np.arange(len(gamma_list)), gamma_list,
                   rotation='vertical',
                   )     
        
        plt.title('Gamma vs Area under ROC curve for %s' % mdl_names[i])
        plt.xlabel('$\gamma$')
        plt.ylabel('Area under ROC curve')
        #plt.ylabel('some numbers')
        
        #plt.show()
        plt.savefig('%s%s_%s_%spca_%s_gammavroc.png' % (filepath, nowtime, dataset, abbreviation, mdl_abvs[i]))
        plt.close()
         
            
    # End of dataset run        
    print('\n###################################################################')


#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print("\nTime taken for script to run is %.2f seconds\n" % EndTime)
