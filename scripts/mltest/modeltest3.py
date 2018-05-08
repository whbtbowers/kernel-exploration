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

from p2funcs import plot_scatter, target_split, distribution_boxplot, filt_imp


from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, cross_val_predict, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
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
scriptname = 'modeltest3'

#Select current toy dataset
#dataset = '018'

mesa = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO_non-verbose.csv', sep=',', header=0, index_col=1)

#List of toy datasets to test
#dataset_list = [('dataset_017', '017'), ('dataset_018','018'), ('datase_019', '019'), ('dataset_020', '020'), ('dataset_021', '021'), ('dataset_022', '022'), ('dataset_023', '023'), ('dataset_024', '024')]
#dataset_list = [('dataset_021', '021')]
dataset_list = [('MESA_dataset', mesa)]


for dataset_name, dataset in dataset_list:    
       
    print('\nRunning %s with %s:' % (scriptname, dataset_name))
    
    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset_name)
    
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    
    # Import  dataset and target
    if dataset_name == 'MESA_dataset':        
        X = mesa
        
        X = filt_imp(X, 0.1)     
        
    else:
        X = pd.read_csv('../../data/simulated/mvnsim/mvnsim' + dataset + '.csv', sep=',', header=0, index_col=0)
    
    y = np.load('../../data/simulated/mvnsim/target' + dataset + '.npy')
    #print(y)
    #print(y.shape)
    #print(X.shape)
    
     
    
    
    distribution_boxplot(X,
                         y,
                         "Initial category 1 distribution of %s" % dataset_name,
                         "Initial category 0 distribution of %s" % dataset_name,
                         #output='show'
                         #output='plotly',
                         #ply_title="Initial distribution of dataset %s" % dataset,
                         output='save',
                         path='%sinitialdist_%s.png' % (filepath, nowtime)
                         )

    print('\nBoxplot of initial data for %s saved.' % dataset)

   
    ## PREPROCESSING ##
    
    #Scale initial data to centre data
    
    X_scaled = scale(X)
    X_scaled_df = pd.DataFrame.from_records(X_scaled)
    
    distribution_boxplot(X_scaled_df,
                         y,
                         "Scaled category 1 distribution of dataset %s" % dataset,
                         "Scaled category 0 distribution of dataset %s" % dataset,
                         #output='show'
                         output='save',
                         path='%sscaledist_%s.png' % (filepath, nowtime)
                         )
    #print(X_scaled.shape)
    print('\nBoxplot of scaled data for dataset %s saved.' % dataset)
    
    #Initiate KPCAwith various kernels
    
    # As I'm using 500 variables, 0.002 is the default gamma (1/n_variables)
    # I only explicitly state it at this point so I can display it on graphs
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
    
    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('K-Nearest Neighbour', 'knn', KNeighborsClassifier()))
    models.append(('Logistic Regression', 'log_reg', LogisticRegression()))
    models.append(('Decision Tree', 'dec_tree', DecisionTreeClassifier()))
    models.append(('Gaussian Naive Bayes', 'gnb', GaussianNB()))
    models.append(('Random Forest', 'rf', RandomForestClassifier()))
    models.append(('Gradient Boosting', 'gb', GradientBoostingClassifier()))

    #models.append(('PLS', PLSRegression())) # Scale=False as data already scaled.
    
    folds = 10
    
    cv = StratifiedKFold(n_splits=folds, random_state=10)
    
    # Declare KPCA kernels deployed
    
    kpca_kernels = []
    
    for kernel, abbreviation, kpca in kpcas:
    
        X_kpca = kpca.fit_transform(X_scaled)
    
        plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %sPCA' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%spca_gamma%s_%s.png' % (filepath, abbreviation, gamma, nowtime)
                     )
        print('\nScatter plot of first two principal components after %sPCA for dataset %s saved.' % (kernel, dataset))
    
        kpca_kernels.append(kernel)
    
        # Declare names of models deployed
        mdl_names = []        
    
        for model_name, model_abv, model in models:
            
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            
            mdl_names.append(model_name)
            print('\nPerforming ' + model_name + ' after ' + kernel + 'PCA')
            #print(mdl_names)
    
            # To count number of folds
            i = 0
            
            #Initiate plot
            fig = plt.figure(figsize=(15, 9))
            
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
            plt.title('Receiver operating characteristic (Using %sPCA with %s, γ = %s)' % (kernel, model_name, gamma))
            plt.legend()
            #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            #plt.legend('lower right')
            #plt.show()
            plt.savefig('%sroc_%spca_%s_gamma%s_%s.png' % (filepath, abbreviation, model_abv, gamma, nowtime))
            plt.close()
'''
'''
            #Convert to plotly object
            plotly_fig = tls.mpl_to_plotly(fig)
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(plotly_fig['layout'])
            py.iplot(plotly_fig, filename='roc-curve')
'''

'''            
            # Display mean roc auc
            print("Mean area under curve for %sPCA followed by %s: %0.2f" % (kernel, model_name, mean_auc))
            
    # End of dataset run        
    print('\n###################################################################')
'''

#Calculate and display time taken or script to run
EndTime = (time.time() - StartTime)
print("\nTime taken for script to run is %.2f seconds\n" % EndTime)
