import os
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time
import datetime

import seaborn as sns
sns.set(style="ticks")
sns.set(style='white')
import smtplib
from email.mime.text import MIMEText as text

import pandas as pd
import numpy as np
from numpy import linalg as la
from numpy.random import multivariate_normal

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
from sklearn.metrics.pairwise import laplacian_kernel, chi2_kernel, polynomial_kernel


from scipy import interp
from pylab import get_cmap

# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")


def plot_write(content, path):
    text_file = open(path, "a")
    text_file.write(content)
    text_file.close()

def plot_scatter(x, y, title, gamma=None, x_label='x coordinate', y_label='y coordinate', cat1='Category 1', cat0='Category 0', output=None, path=None, jspath=None, divname = None, dataset=None, kernel=None):

    fig = plt.figure(figsize=(8, 6))

    cata = plt.scatter(x[y==0, 0],
                       x[y==0, 1],
                       color='red',
                       marker = '^',
                       alpha=0.5
                       )

    trace1 = js_scatter_trace(x[y==0, 0],
                              x[y==0, 1],
                              1,
                              cat0
                              )

    catb = plt.scatter(x[y==1, 0],
                       x[y==1, 1],
                       color='blue',
                       marker = 's',
                       alpha=0.3
                       )

    trace2 = js_scatter_trace(x[y==1, 0],
                              x[y==1, 1],
                              2,
                              cat1
                              )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    gamma_label = mpatches.Patch(color='white', label='gamma')
    plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), cat0, cat1])

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)
        js_construct_scatter(divname, jspath, trace1, trace2)
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
    n_rows, n_cols = X.shape
    print('\nInitial data contains %s rows and %s columns.' % (n_rows, n_cols))
    col_names = list(X)


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

    n_rows, n_cols = X.shape
    print('\n%s columns in dataset removed due to ≤10%% of cells populated.' % dropped_cols)
    print('\nAfter columns ≤ 10%% populated removed, data contains %s rows and %s columns.' % (n_rows, n_cols))

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

    n_rows, n_cols = X.shape
    print('\n%s rows in remaining dataset removed due to ≤10%% of cells populated.' % dropped_rows)
    print('\nAfter columns and rows ≤ 10%% populated removed, data contains %s rows and %s columns.' % (n_rows, n_cols))


    # Convert dataframe to numpy matrix for scikit learn
    #X = X.as_matrix()

    # Uses mean as imputation strategy
    impute = Imputer(strategy='median')
    X_imputed = impute.fit_transform(X)
    X_imputed_df = pd.DataFrame.from_records(X_imputed)
    #print(X_imputed.shape)
    n_rows, n_cols = X_imputed.shape
    print('\nAfter imputation, data contains %s rows and %s columns.\n' % (n_rows, n_cols))

    return(X_imputed_df)

# Like range but for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def m_test5(X, y, gamma, dataset, filepath, signifier):

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
    #models.append(('Polynomial Kernel SVM','ply_svc', SVC(kernel='poly', gamma=gamma, probability=True)))
    #models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))
    #models.append(('K-Nearest Neighbour', 'knn', KNeighborsClassifier()))
    #models.append(('Logistic Regression', 'log_reg', LogisticRegression()))
    #models.append(('Decision Tree', 'dec_tree', DecisionTreeClassifier()))
    #models.append(('Gaussian Naive Bayes', 'gnb', GaussianNB()))
    #models.append(('Random Forest', 'rf', RandomForestClassifier()))
    #models.append(('Gradient Boosting', 'gb', GradientBoostingClassifier()))

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


        plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     #output='save',
                     #path='%s%s_%spca_gamma%s_%s.png' % (filepath, nowtime, abbreviation, gamma, signifier)
                     )
        print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))


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

#To generate toy datasets based on size and covariance of test data
def toybox_gen(inp_df):

    #Define size of component using input data
    inp_df = inp_df.T
    cols, rows = inp_df.shape
    df_cov = np.cov(inp_df)
    print('\nCovariance calculated\n')
    cov1 = np.array([df_cov/40, df_cov/20, df_cov/10, df_cov/5, df_cov/2, df_cov/1.5, df_cov/1.5, df_cov, df_cov, df_cov, df_cov, df_cov, df_cov])
    # cov1 = np.array([df_cov/40])
    cov2 = np.array(df_cov)

    init_mean = inp_df.mean(axis=1).tolist()
    print('Means calculated\n')
    mean1 = np.array([init_mean, init_mean, init_mean, init_mean, init_mean, init_mean, np.add(init_mean, 0.5), np.add(init_mean, 0.5), np.add(init_mean, 1.0), np.add(init_mean, 1.5), np.add(init_mean, 2.0), np.add(init_mean, 2.5), np.add(init_mean, 3.0)])
    # mean1 = np.array([init_mean])
    mean2 =  init_mean

    dataset_list = []

    # counter for labelling dataset
    counter = 1

    # Second component consistent. Generate first to save time.
    d2_x = multivariate_normal(mean2, cov2, int(round(rows/2)))
    d2rows, d2cols = d2_x.shape

    #Set up target array
    target = np.zeros(d2rows*2, dtype=int)
    target[0:int(round(d2rows))] = '1'
    print('Simulated outcomes generated\n')

    for i in range(len(cov1)):

        d1_x= multivariate_normal(mean1[i], cov1[i], int(round(rows/2)))


        #Put components together
        mvn_sim = np.vstack((d1_x, d2_x))
        mvn_sim_df = pd.DataFrame.from_records(mvn_sim)
        dataset_list.append(('ds00%d' % counter, mvn_sim_df))
        print('Simulated dataset ds%d generated' % counter)
        counter += 1



    return(dataset_list, target)

def multisize_toybox(inp_df):

    df_cov = np.cov(inp_df)
    print('\nCovariance calculated\n')

    n_var = 500

    toy_cov = df_cov[0:n_var, 0:n_var]

    n_sam_list = [500, 1000, 1500, 2000, 2500, 3000, 3500]

    cov1 = toy_cov/1.75
    cov2 = toy_cov

    mean = inp_df.mean(axis=1).tolist()[:n_var]

    print('Means calculated\n')

    dataset_list = []
    target_list = []

    # counter for labelling dataset
    counter = 1

    for size in n_sam_list:

        d1_x = multivariate_normal(mean, cov1, int(size/2))
        d2_x = multivariate_normal(mean, cov2, int(size/2))

        #Put components together
        mvn_sim = np.vstack((d1_x, d2_x))
        mvn_sim_df = pd.DataFrame.from_records(mvn_sim)
        dataset_list.append(('ds00%d' % counter, mvn_sim_df))
        print('Simulated dataset ds%d generated' % counter)
        counter += 1

        # Generate target array
        target = np.zeros(size, dtype=int)
        target[0:int((size/2))] = '1'
        target_list.append(target)
        print('Simulated outcome for ds%d generated\n' % counter)

    return(dataset_list, target_list)

def tsplit(inp_df):

    target = inp_df[0].as_matrix()
    data =  inp_df.drop(columns=0)

    target = target.astype(int)

    return(data, target)

def pca_plot(X, y, dataset, filepath, cat1, cat0):

    X_rows, X_cols = X.shape

    #print(X_cols)

    gamma = 1/X_cols

    kpcas = []

    kpca_lap = laplacian_kernel(X, gamma=gamma)

    kpcas = []

    #Use standard PCA for comparison

    #kpcas.append(('standard ', 'std_', PCA(n_components=2)))

    #Linear kernal has no need for gamma
    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'prec_lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Polynomial KPCA', 'ply_kpca', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    for kernel, abbreviation, kpca in kpcas:

        if kernel == 'Laplacian KPCA':
            X_kpca = kpca.fit_transform(kpca_lap)
        else:
            X_kpca = kpca.fit_transform(X)

        plot_scatter(X_kpca,
                         y,
                         'First 2 principal components after %s' % kernel,
                         gamma=gamma,
                         x_label='Principal component 1',
                         y_label='Principal component 2',
                         cat1=cat1,
                         cat0=cat0,
                         #output = 'show',
                         output='save',
                         path='%s%s_%s_%s_gamma%s.png' % (filepath, nowtime, dataset, abbreviation, gamma)
                         )

        print('\nScatter plot of first two principal components after %s for dataset %s saved.' % (kernel, dataset))

def gs_pca_plot(X, y, dataset, filepath, cat1, cat0):

    X_rows, X_cols = X.shape

    #print(X_cols)

    init_gamma = 1/X_cols

    #gamma_list = [init_gamma/10000, init_gamma/1000, init_gamma/100, init_gamma/10, init_gamma, init_gamma*10, init_gamma*100, init_gamma*1000, init_gamma*10000, init_gamma*100000]
    gamma_list = [init_gamma/10, init_gamma, init_gamma*10, init_gamma*100, init_gamma*1000, init_gamma*10000, init_gamma*100000]
    for gamma in gamma_list:

        kpcas = []

        kpca_lap = laplacian_kernel(X, gamma=gamma)

        kpcas = []

        #Use standard PCA for comparison

        #kpcas.append(('standard ', 'std_', PCA(n_components=2)))

        #Linear kernal has no need for gamma
        kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
        kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
        #kpcas.append(('Laplacian KPCA', 'prec_lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
        #kpcas.append(('Polynomial KPCA', 'ply_kpca', KernelPCA(n_components=2, kernel='poly', gamma=gamma)))
        #kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
        #kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

        for kernel, abbreviation, kpca in kpcas:

            if kernel == 'Laplacian KPCA':
                X_kpca = kpca.fit_transform(kpca_lap)
            else:
                X_kpca = kpca.fit_transform(X)

            plot_scatter(X_kpca,
                             y,
                             'First 2 principal components after %s' % kernel,
                             gamma=gamma,
                             x_label='Principal component 1',
                             y_label='Principal component 2',
                             cat1=cat1,
                             cat0=cat0,
                             #output = 'show',
                             output='save',
                             path='%s%s_%s_%s_gamma%s.png' % (filepath, nowtime, dataset, abbreviation, gamma)
                             )

            print('\nScatter plot of first two principal components after %s for dataset %s with gamma = %s saved.' % (kernel, dataset, gamma))

def m_test5_2(X, y, gamma, dataset, filepath, jspath, signifier):

    #Record all mean roc_aucs for each gamma value
    auc_mat = []


    #compute kernels not preloaded into kpca
    #laplacian
    K_lap = laplacian_kernel(X, gamma=gamma)

    kpcas = []

    #Use standard PCA for comparison

    #kpcas.append(('standard PCA', 'std_', PCA(n_components=2)))

    #Linear kernal has no need for gamma
    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    #Initiate models with default parameters

    models = []

    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))

    # Initiate cross-validation
    folds = 10
    cv = StratifiedKFold(n_splits=folds, random_state=10)

    # Declare KPCA kernels deployed

    kpca_kernels = []

    for kernel, abbreviation, kpca in kpcas:

        # To utilise precomputed kernel(s)
        if kernel == 'Laplacian KPCA':
            X_kpca = kpca.fit_transform(K_lap)
        elif kernel == 'Polynomial KPCA':
            X_kpca = kpca.fit_transform(K_ply)
        else:
            X_kpca = kpca.fit_transform(X)


        plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%s_%s_%sgamma%s.png' % (filepath, nowtime, abbreviation, signifier, gamma),
                     jspath='%s%s_%s_%sgamma%s.js' % (jspath, nowtime, abbreviation, signifier, gamma),
                     )
        print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))


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

def m_test5_2_rocplot(X, y, gamma, dataset, filepath, jspath, signifier):

    #Record all mean roc_aucs for each gamma value
    auc_mat = []


    #compute kernels not preloaded into kpca
    #laplacian
    K_lap = laplacian_kernel(X, gamma=gamma)

    kpcas = []

    #Use standard PCA for comparison

    #kpcas.append(('standard PCA', 'std_', PCA(n_components=2)))

    #Linear kernal has no need for gamma
    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    #Initiate models with default parameters

    models = []

    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))

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


        plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%s_%s_%sgamma%s.png' % (filepath, nowtime, abbreviation, signifier, gamma),
                     jspath='%s%s_%s_%sgamma%s.js' % (jspath, nowtime, abbreviation, signifier, gamma),
                     )
        print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))


        kpca_kernels.append(kernel)

        # Declare names of models deployed
        mdl_names = []

        #Record mean_aucs
        auc_mat_row = []

        for model_name, model_abv, model in models:

            raw_tprs = []
            raw_fprs = []
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            mdl_names.append(model_name)
            print('\nPerforming %s followed by %s for dataset %s\n' % (kernel, model_name, dataset))
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
                raw_tprs.append(tpr)
                raw_fprs.append(fpr)
                aucs.append(roc_auc)
                plt.plot(fpr, tpr, lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

                i += 1

            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Luck', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            auc_mat_row.append(mean_auc)
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
            plt.title('Receiver operating characteristic (Using %s with %s, γ = %s)' % (kernel, model_name, gamma))
            plt.legend()
            #plt.show()
            plt.savefig('%s%sroc_%s_%s_%s_gamma%s.png' % (filepath, nowtime, abbreviation, model_abv, signifier, gamma))
            plt.close()

            trace_list, traces = js_fold_line(raw_fprs, raw_tprs, aucs)
            trace_list, traces = js_mean_trace(mean_fpr, mean_tpr, mean_auc, std_auc, trace_list, traces)
            trace_list, traces = js_luck_trace(trace_list, traces)
            trace_list, traces = js_tpr_std(tprs_upper, tprs_lower, mean_fpr, trace_list, traces)

            js_construct_roc('ROCPLOT', 'rocplot%s_%s_%s_%s' % (nowtime, dataset, abbreviation, model_abv), trace_list, traces, '%s%s_%s_%s_gamma%s_roc.js' % (jspath, nowtime, abbreviation, model_abv, gamma))

        auc_mat.append(auc_mat_row)

    auc_mat = np.array(auc_mat)


    return(auc_mat, kpca_kernels, mdl_names)

def m_run5_3(X, y, gamma, opt_kernel, opt_model, dataset, filepath, jspath, signifier):

    #Record AUC
    auc_mat = []

    #compute kernels not preloaded into kpca
    #laplacian
    K_lap = laplacian_kernel(X, gamma=gamma)

    kpcas = []

    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    #Initiate models with default parameters

    models = []

    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))

    # Initiate cross-validation
    folds = 10
    cv = StratifiedKFold(n_splits=folds, random_state=10)

    # Declare KPCA kernel and model deployed

    kpca_kernel = 0
    mdl_name = 0

    # Declare list for expected variances
    exp_vars = []

    for kernel, abbreviation, kpca in kpcas:

        if opt_kernel == kernel:

            kpca_kernels = kernel
            # To utilise precomputed kernel(s)
            if kernel == 'Laplacian KPCA':
                X_kpca = kpca.fit_transform(K_lap)
            else:
                X_kpca = kpca.fit_transform(X)


            plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%s_%s_%sgamma%s.png' % (filepath, nowtime, abbreviation, signifier, gamma),
                     jspath='%s%s_%s_%sgamma%s.js' % (jspath, nowtime, abbreviation, signifier, gamma),
                     )
            print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))



            # Declare names of models deployed
            mdl_names = []

            #Record mean_aucs
            auc_mat_row = []

            for model_name, model_abv, model in models:

                if opt_model == model_name:

                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0, 1, 100)

                    mdl_names = model_name

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

def m_run5_3_rocplot(X, y, gamma, opt_kernel, opt_model, dataset, filepath, jspath, signifier):

    #Record AUC
    auc_mat = []

    #compute kernels not preloaded into kpca
    #laplacian
    K_lap = laplacian_kernel(X, gamma=gamma)

    kpcas = []

    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    #Initiate models with default parameters

    models = []

    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))

    # Initiate cross-validation
    folds = 10
    cv = StratifiedKFold(n_splits=folds, random_state=10)

    # Declare KPCA kernel and model deployed

    kpca_kernel = 0
    mdl_name = 0

    # Declare array for explained variances
    #exp_vars = []


    for kernel, abbreviation, kpca in kpcas:

        if kernel == opt_kernel:

            kpca_kernel = kernel

            X_kpca = 0

            # To utilise precomputed kernel(s)
            if kernel == 'Laplacian KPCA':
                X_kpca = kpca.fit_transform(K_lap)
            else:
                X_kpca = kpca.fit_transform(X)

            exp_var = np.var(X_kpca, axis=0)
            print('\nExplained variance of first principal component: %s' % exp_var[0])
            print('\nExplained variance of second principal component: %s' % exp_var[1])
            #exp_vars.append(exp_var)

            plot_scatter(X_kpca,
                     y,
                     'First 2 principal components after %s' % kernel,
                     gamma=gamma,
                     x_label='Principal component 1',
                     y_label='Principal component 2',
                     #output = 'show',
                     output='save',
                     path='%s%s_%s_%sgamma%s.png' % (filepath, nowtime, abbreviation, signifier, gamma),
                     jspath='%s%s_%s_%sgamma%s.js' % (jspath, nowtime, abbreviation, signifier, gamma),
                     )
            print('\nScatter plot of first two principal components after %s for dataset %s (γ = %s) saved.' % (kernel, dataset, gamma))

            #Record mean_aucs
            auc_mat_row = []

            for model_name, model_abv, model in models:

                if opt_model == model_name:

                    mdl_name = model_name

                    raw_tprs = []
                    raw_fprs = []
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0, 1, 100)


                    print('\nPerforming %s followed by %s for dataset %s\n' % (kernel, model_name, dataset))
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
                        raw_tprs.append(tpr)
                        raw_fprs.append(fpr)
                        aucs.append(roc_auc)
                        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                                 label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        
                        i += 1
        
                    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                             label='Luck', alpha=.8)
        
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_tpr[-1] = 1.0
                    mean_auc = auc(mean_fpr, mean_tpr)
                    auc_mat_row.append(mean_auc)
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
                    plt.title('Receiver operating characteristic (Using %s with %s, γ = %s)' % (kernel, model_name, gamma))
                    plt.legend()
                    #plt.show()
                    plt.savefig('%s%sroc_%s_%s_%s_gamma%s.png' % (filepath, nowtime, abbreviation, model_abv, signifier, gamma))
                    plt.close()

                    trace_list, traces = js_fold_line(raw_tprs, raw_fprs, aucs)
                    trace_list, traces = js_mean_trace(mean_fpr, mean_tpr, mean_auc, std_auc, trace_list, traces)
                    trace_list, traces = js_luck_trace(trace_list, traces)
                    trace_list, traces = js_tpr_std(tprs_upper, tprs_lower, mean_fpr, trace_list, traces)

                    js_construct_roc('ROCPLOT', 'rocplot%s_%s_%s_%s' % (nowtime, dataset, abbreviation, model_abv), trace_list, traces, '%s%s_%s_%s_%s_gamma%s_roc.js' % (jspath, nowtime, dataset, kernel, model_name, gamma))

            auc_mat.append(auc_mat_row)

    auc_mat = np.array(auc_mat)
    print(auc_mat)

    return(auc_mat, kpca_kernel, mdl_name)

# Find which element occurs most commonly in a 1D array
def most_common(inp_list):
    inp_list_count_dict = dict((x,inp_list.count(x)) for x in set(inp_list))
    consensus = max(inp_list_count_dict, key=inp_list_count_dict.get)
    return(consensus)

### Set of functions for building plotly ROC curve


def js_fold_line(x_vals_list, y_vals_list, aucs):

    #Declare trace list for calling on data
    trace_list = []
    traces = []
    #To make numbering of reads in legend look nice
    trinv= list(reversed(list(range(len(x_vals_list)))))
    
    for i in range(len(x_vals_list)):

        trace = 'trace%s' % str(i+4)
        trace_list.append(trace)
        open_trace = 'var %s = {' % trace
        close_trace = "\n\ttype: 'scatter'\n};\n\n"
        trace_x = '\n\tx: ' + str(x_vals_list[i].tolist()) + ','
        trace_y = '\n\ty: ' + str(y_vals_list[i].tolist()) + ','
        leg_name = "\n\tname: 'ROC fold %s (AUC = %0.2f)'," % (trinv[i] + 1, aucs[i])
        line = "\n\tline:{\n\t\twidth: 2,\n\t},"
        full_trace = open_trace + trace_x + trace_y +  leg_name + line + close_trace
        traces.append(full_trace)

    return(trace_list, traces)

def js_mean_trace(mean_x, mean_y, mean_auc, std_auc, trace_list, traces):

    mean_trace = "trace" + str(int(trace_list[-1][5:]) + 1)
    trace_list.append(mean_trace)
    mean_trace_x = '\n\tx: ' + str(mean_x.tolist()) + ','
    mean_trace_y = '\n\ty: ' + str(mean_y.tolist()) + ','
    mean_trace_close ="\n\tname: 'Mean (AUC = %0.2f ± %0.2f)',\n\tline: {\n\t\tcolor: 'rgb(0, 0, 225)',\n\t\twidth: 8,\n\t},\n\tmode: 'lines',\n\ttype: 'scatter'\n};" % (mean_auc, std_auc)
    mean_trace_full = "var %s = {%s%s%s" % (mean_trace, mean_trace_x, mean_trace_y, mean_trace_close)
    traces.append(mean_trace_full)

    return(trace_list, traces)

def js_construct_roc(chartname, divname, trace_list, traces, path):

    #To remove inverted commas around each trace
    class MyStr(str):
        def __repr__(self):
            return super(MyStr, self).__repr__().strip("'")

    plot_write("%s = document.getElementById('%s');\n\n" % (chartname, divname), path)
    for trace in traces:
        plot_write(trace, path)

    tl_stripped = []

    for label in trace_list:
        tl_stripped.append(MyStr(label))

    plot_write('\nvar data = %s;\n' % str(tl_stripped), path)
    plot_write("var layout = {\n\txaxis: {\n\t\ttitle: 'False positive rate',\n\t\ttitlefont: {\n\t\t\tfamily: 'Courier New, monospace',\n\t\t\tsize: 18,\n\t\t\tcolor: '#7f7f7f'\n\t\t},\n\t\tzeroline: true\n\t},\n\tyaxis: {\n\t\ttitle: 'True positive rate',\n\t\ttitlefont: {\n\t\t\tfamily: 'Courier New, monospace',\n\t\t\tcolor: '#7f7f7f'\n\t\t},\n\t\tzeroline: true\n\t},\n};\n\n", path)
    plot_write('Plotly.newPlot(%s, data, layout);' % chartname, path)
    #print("%s = document.getElementById('%s');\n\n" % (chartname, divname))
    #for trace in traces:
    #    print(trace)
    #print('\nvar data = %s;\n' % str(tl_stripped))
    #print("var layout = {\n\txaxis: {\n\t\ttitle: 'False positive rate',\n\t\ttitlefont: {\n\t\t\tfamily: 'Courier New, monospace',\n\t\t\tsize: 18,\n\t\t\tcolor: '#7f7f7f'\n\t\t},\n\t\tzeroline: true\n\t},\n\tyaxis: {\n\t\ttitle: 'True positive rate',\n\t\ttitlefont: {\n\t\t\tfamily: 'Courier New, monospace',\n\t\t\tcolor: '#7f7f7f'\n\t\t},\n\t\tzeroline: true\n\t},\n};\n\n")
    #print('Plotly.newPlot(%s, data, layout);' % chartname)

def js_luck_trace(trace_list, traces):

    trace_list = ['trace3'] + trace_list
    traces = ["var trace3 = {\n\tx: [0, 1],\n\ty: [0, 1],\n\tname: 'Luck',\n\tline: {\n\t\tdash: 'dot',\n\t\tcolor: 'rgb(255, 0, 0)',\n\t},\n\tmode: 'lines',\n\ttype: 'scatter'\n};\n\n"] + traces
    return(trace_list, traces)

def js_tpr_std(tpr_std_upper, tpr_std_lower, fpr_std, trace_list, traces):

    trace_list = ['trace2'] + trace_list
    traces = ["var trace2 = {\n\tx: %s,\n\ty: %s,\n\tname: 'Mean ±1 standard deviation',\n\tline:{\n\t\twidth: 0,\n\t\tcolor: '#808080'\n\t},\n\tfill:'tonexty',\n\tmode: 'lines',\n\ttype: 'scatter'\n};\n\n" % (str(tpr_std_upper.tolist()), str(fpr_std.tolist()))] + traces

    trace_list = ['trace1'] + trace_list
    traces = ["var trace1 = {\n\tx: %s,\n\ty: %s,\n\tname: '',\n\tline:{\n\t\twidth: 0,\n\t},\n\tfill: 'none',\n\tmode: 'lines',\n\ttype: 'scatter'\n};\n\n" % (str(tpr_std_lower.tolist()), str(fpr_std.tolist()))] + traces
    return(trace_list, traces)

#Create matplotlib heatmap
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return(im, cbar)

# Plot and show matplotlib heatmap
def plot_mpl_heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", output=None, path=None, **kwargs, ):

    fig, ax = plt.subplots()

    im, cbar = heatmap(data, row_labels, col_labels, ax=ax,
            cbar_kw={}, cbarlabel="", **kwargs)

    fig.tight_layout()

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)

    plt.close()

def js_scatter_trace(X, Y, n, category):

    trace = "var trace%s = {\n\tx: %s,\n\ty: %s,\n\tmarker: {\n\t\tsize: 6,\n\t\topacity: 0.5,\n\t\tsymbol: 'square',\n\t},\n\tmode: 'markers',\n\tname: '%s',\n\ttype: 'scatter',\n};\n" % (n, X.tolist(), Y.tolist(), category)
    return(trace)

def js_construct_scatter(divname, path, *traces):

    plot_write("SCATTER = document.getElementById('%s');\n\n" % divname, path)

    #To remove inverted commas around each trace
    class MyStr(str):
        def __repr__(self):
            return super(MyStr, self).__repr__().strip("'")

    t_list = []

    for i in range(len(traces)):
        plot_write(traces[i], path)
        t_list.append(MyStr('trace%s' % str(i + 1)))

    plot_write("var data = %s;\n\nPlotly.plot(SCATTER, data);" % str(t_list), path)

def js_heatmap(data, X_labels, Y_labels, divname, path):
    plot_write("HEATMAP = document.getElementById('%s');\n\nvar trace1 = {\n\tx: %s,\n\ty: %s,\n\tz: %s,\n\tcolorscale: 'YIOrRd',\n\ttype: 'heatmap',\n\tcolorbar:{\n\t\ttitle:'Mean area under ROC curve after 10-flod cross validation',\n\t\ttitleside:'right',\n\t},\n};\n\nvar data = [trace1];\n\nvar layout = {\n\tlegend: {\n\t\tbgcolor: '#FFFFFF',\n\t\tfont: {color: '#4D5663'}\n\t},\n\tpaper_bgcolor: '#FFFFFF',\n\tplot_bgcolor: '#FFFFFF',\n\txaxis1: {\n\t\tgridcolor: '#E1E5ED',\n\t\ttickfont: {color: '#4D5663'},\n\t\ttitle: '',\n\t\ttitlefont: {color: '#4D5663'},\n\t\tzerolinecolor: '#E1E5ED'\n\t},\n\tyaxis1: {\n\t\tgridcolor: '#E1E5ED',\n\t\ttickfont: {color: '#4D5663'},\n\t\ttitle: '',\n\t\ttitlefont: {color: '#4D5663'},\n\t\tzeroline: false,\n\t\tzerolinecolor: '#E1E5ED'\n\t}\n};\n\nPlotly.plot(HEATMAP, data, layout);" % (divname, X_labels, Y_labels, str(data.tolist())), path)

def js_bars(X_data, Y_data, path):
    plot_write("var summBar = document.getElementById('summary-bar');\n\nvar x = %s;\nvar y = %s;\n\nvar trace1 = {\n\tx:x,\n\ty:y,\n\tmarker: {\n\t\tcolor: col_list,\n\t\tline: {\n\t\t\twidth: 1.0\n\t\t}/n/t},/n/topacity: 1,\n\torientation: 'v',\n\ttype: 'bar',\n\txaxis: 'x1',\n\tyaxis: 'y1'\n};\n\nvar data = [trace1];\n\nPlotly.plot(summBar, data);\n\nvar strList = INSERT ABV LIST;\n\n summBar.on('plotly_click', function(data){\n\tvar char = x.indexOf(data.points[0].x);\n\tvar corr = strList[char];\n\twindow.open('GIVE HTML');\n});" % (X_data, Y_data), path)

def mpl_simplebar(x, y, xlab, ylab, col_list, output=None, path=None):



    fig, ax = plt.subplots(figsize=(10,7))
    plt.bar(np.arange(len(x)), y, color=col_list)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(np.arange(len(x)), x, rotation='vertical')

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig(path)

    plt.close()

def get_col_list(colormap, n_cols):

    colours = []

    cmap = get_cmap(colormap, n_cols)

    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 to get rgb
        colours.append(mcolors.rgb2hex(rgb))

    return(colours)

def endalert(subject, content):

    fromaddr = "whb17@ic.ac.uk"
    toaddrs  = "whtbowers@gmail.com"
    subject = 'subject'

    msg = text(content)

    msg['Subject'] = subject
    msg['From'] = fromaddr
    msg['To'] = toaddrs

    server = smtplib.SMTP('localhost')
    server.set_debuglevel(1)
    server.sendmail(fromaddr, toaddrs, msg.as_string())
    server.quit()
