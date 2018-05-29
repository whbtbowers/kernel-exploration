import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC

import p2funcs as p2f

# Time to differentiate images
now = datetime.datetime.now()
nowdate = now.strftime("%Y-%m-%d")
nowtime = now.strftime("%H-%M")

# Name of script to trace where images came from
scriptname = 'varex'

dataset_list = ['021']

for dataset in dataset_list:

    X = pd.read_csv('../../data/simulated/mvnsim/mvnsim%s.csv' % dataset, sep=',', header=0, index_col=0)

    y = np.load('../../data/simulated/mvnsim/target%s.npy' % dataset)

    #Create directory if directory does not exist
    filepath = '../../figs/out/%s/%s/%s/' % (scriptname, nowdate, dataset)
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    gamma = 0.002

    kpcas = []

    kpcas.append(('Linear KPCA', 'lin_kpca', KernelPCA(n_components=2, kernel='linear')))
    kpcas.append(('RBF KPCA', 'rbf_kpca',KernelPCA(n_components=2, kernel='rbf', gamma=gamma)))
    #kpcas.append(('Laplacian KPCA', 'lap_kpca',KernelPCA(n_components=2, kernel='precomputed')))
    kpcas.append(('Sigmoid KPCA', 'sig_kpca', KernelPCA(n_components=2, kernel='sigmoid', gamma=gamma)))
    kpcas.append(('Cosine KPCA', 'cos_kpca',KernelPCA(n_components=2, kernel='cosine', gamma=gamma)))

    models = []

    models.append(('Linear SVM', 'lin_svc', SVC(kernel='linear', probability=True)))
    #models.append(('RBF Kernel SVM','rbf_svc', SVC(kernel='rbf', gamma=gamma, probability=True)))
    #models.append(('Sigmoid Kernel SVM','sig_svc', SVC(kernel='sigmoid', gamma=gamma, probability=True)))

    kpca_kernels = []

    for kernel, abbreviation, kpca in kpcas:

        #X_kpca = kpca.fit_transform(X_scaled)
        X_kpca = kpca.fit_transform(X)

        # Get explained variance
        exp_var = np.var(X_kpca, axis=0)
        print('Explained variance of first principal component: %0.4f' % exp_var[0])
        print('Explained variance of second principal component: %0.4f' % exp_var[1])

        p2f.plot_scatter(X_kpca,
                        y,
                        'First 2 principal components after %s' % kernel,
                        gamma=gamma,
                        x_label='Principal component 1',
                        y_label='Principal component 2',
                        output = 'show',
                        #output='save',
                        #path='%s%spca_gamma%s_%s.png' % (filepath, abbreviation, gamma, nowtime)
                        )
        print('\nScatter plot of first two principal components after %s for dataset %s saved.' % (kernel, dataset))

        kpca_kernels.append(kernel)

        # Declare names of models deployed
        mdl_names = []

        for model_name, model_abv, model in models:

            X_svm = model.fit(X_kpca, y)

            # get the separating hyperplane
            w = X_svm.coef_[0]
            a = -w[0] / w[1]
            xx = np.linspace(-5, 5)
            yy = a * xx - (X_svm.intercept_[0]) / w[1]

            # plot the parallels to the separating hyperplane that pass through the
            # support vectors
            b_up = X_svm.support_vectors_[0]
            yy_down = a * xx + (b_up[1] - a * b_up[0])
            b_down = X_svm.support_vectors_[-1]
            yy_up = a * xx + (b_down[1] - a * b_down[0])

            #Hyperplane shizness
            '''
            p2f.plot_scatter(X_kpca,
                            y,
                            'First 2 principal components after %sPCA' % kernel,
                            gamma=gamma,
                            x_label='Principal component 1',
                            y_label='Principal component 2',
                            output = 'show',
                            #output='save',
                            #path='%s%spca_gamma%s_%s.png' % (filepath, abbreviation, gamma, nowtime)
                            )

            fig = plt.figure(figsize=(8, 6))
            '''
            cata = plt.scatter(X_kpca[y==0, 0],
                               X_kpca[y==0, 1],
                               color='red',
                               marker = '^',
                               alpha=0.5
                               )

            catb = plt.scatter(X_kpca[y==1, 0],
                               X_kpca[y==1, 1],
                               color='blue',
                               marker = 's',
                               alpha=0.3
                               )

            h_plane = plt.plot(xx, yy, 'k-')
            hbound_up = plt.plot(xx, yy_down, 'k--')
            hbound_down = plt.plot(xx, yy_up, 'k--')

            #If rolling out irl, make more dynamic.
            plt.xlim(xmin=-1, xmax=1)
            plt.ylim(ymin=-1, ymax=1)

            plt.title('title')
            plt.xlabel('x_label')
            plt.ylabel('y_label')
            #gamma_label = mpatches.Patch(color='white', label='gamma')
            #plt.legend([gamma_label,cata, catb],['γ = '+str(gamma), cat0, cat1])
            plt.show()
            plt.close()
'''
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            mdl_names.append(model_name)
            print('\nPerforming  ' + model_name + ' after ' + kernel)

            for train, test in cv.split(X_kpca, y):

                probas_ = model.fit(X_kpca[train], y[train]).predict_proba(X_kpca[test])
'''
