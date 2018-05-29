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

    kpca_kernels = []
    p1_exvar = []
    p2_exvar = []

    for kernel, abbreviation, kpca in kpcas:

        kpca_kernels.append(kernel)

        #X_kpca = kpca.fit_transform(X_scaled)
        X_kpca = kpca.fit_transform(X)

        # Get explained variance
        exp_var = np.var(X_kpca, axis=0)
        p1_exvar.append(exp_var[0])
        p1_exvar.append(exp_var[1])
        print('Explained variance of first principal component: %0.4f' % exp_var[0])
        print('Explained variance of second principal component: %0.4f' % exp_var[1])

    fig, ax = plt.subplots()

    #ind = np.arange(N)    # the x locations for the groups
    width = 0.35         # the width of the bars
    #p1 = ax.bar(ind, menMeans, width, color='r', bottom=0*cm, yerr=menStd)
    p1 = ax.bar(np.arange(len(kpca_kernels)), p1_exvar, width, color='r')

    #womenMeans = (145*cm, 149*cm, 172*cm, 165*cm, 200*cm)
    #womenStd = (30*cm, 25*cm, 20*cm, 31*cm, 22*cm)
    #p2 = ax.bar(ind + width, womenMeans, width, color='y', bottom=0*cm, yerr=womenStd)
    p2 = ax.bar(np.arange(len(kpca_kernels)), p1_exvar, width, color='b')

    #ax.set_title('Scores by group and gender')
    #ax.set_xticks(ind + width / 2)
    #ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))

    # ax.legend((p1[0], p2[0]), ('Men', 'Women'))
    #ax.yaxis.set_units(inch)
    ax.autoscale_view()

    plt.show()
    plt.close()
