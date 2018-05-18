#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:16:11 2018

@author: whb17
"""
import p2funcs as p2f

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal
#from numpy.random import multivariate_normal
from numpy.linalg import cholesky

X = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcol_diabetes.csv', sep=',', header=None, index_col=0)

X_imp = p2f.filt_imp(X, 0.1)

X_imp_df = pd.DataFrame.from_records(X_imp)

X, y = p2f.tsplit(X_imp_df)

n_cols, n_rows = X.shape

#print(n_cols)
#print(y)

means = X.mean(axis=0).tolist()

#print(means)

X_cov = np.cov(X.T)
X_cov_pd = p2f.nearestPD(X_cov)
p2f.isPD(X_cov_pd)
#print(X_cov.shape)
#numpy format
#d2_x, d2_y = multivariate_normal(means, X_cov, [n_cols, n_rows], check_valid='ignore').T  

#scipy format
d2_x, d2_y = multivariate_normal([n_cols, n_rows], means, X_cov_pd).T  

#print(d2_X)