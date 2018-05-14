#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 10:13:49 2018

@author: whb17
"""

import numpy as np
import p2funcs as p2f
import pandas as pd

#X = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_VarInfo.csv', sep=',')#, header=0, index_col=0)

#print(X)

#with open('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.txt', 'r') as mesadata:
#    print(mesadata.read())

#X = pd.read_table('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.txt', sep='\t', header=None)#, header=0, index_col=0)

#print(X)

#outpath = '../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv'

#X.to_csv(path_or_buf=outpath, sep=',', header=False)

#X = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_Data.csv', sep=',')

#X_filt = p2f.filt_imp(X, 0.1)

X = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO.csv', sep=',')

diabetes = pd.Series(X['diabetes']).as_matrix()

np.save('../../data/mesa/mesatarget.npy', diabetes)