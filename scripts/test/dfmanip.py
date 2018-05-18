#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:05:38 2018

@author: whb17
"""

import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)

mydata = pd.read_csv('../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_collated_2.csv', header=None, sep=',')

bigsheet = pd.read_csv('../../data/mesa/MESA_Clinical_data_full_COMBI-BIO.csv', header=0, index_col=None, sep=',')

IDs = bigsheet['ID'].as_matrix()

# Binary outcomes
diabetes = ('diabetes', bigsheet['diabetes'].as_matrix())
sex = ('sex', bigsheet['sex'].as_matrix())
cac_binomial = ('cac_binomial', bigsheet['cac_binomial'].as_matrix())
cac_extremes = ('cac_extremes', bigsheet['cac_extremes'].as_matrix())
family_hx_diabetes = ('family_hx_diabetes', bigsheet['family_hx_diabetes'].as_matrix())
parent_cvd_65_hx = ('parent_cvd_65_hx', bigsheet['parent_cvd_65_hx'].as_matrix())
family_hx_cvd = ('family_hx_cvd', bigsheet['family_hx_cvd'].as_matrix())
bp_treatment = ('bp_treatment', bigsheet['bp_treatment'].as_matrix())
diabetes_treatment = ('diabetes_treatment', bigsheet['diabetes_treatment'].as_matrix())
lipids_treatment = ('lipids_treatment', bigsheet['lipids_treatment'].as_matrix())
mi_stroke_hx = ('mi_stroke_hx', bigsheet['mi_stroke_hx'].as_matrix())
plaque = ('plaque', bigsheet['plaque'].as_matrix())

outcomes = [diabetes, sex, cac_binomial, cac_extremes, family_hx_diabetes, parent_cvd_65_hx, family_hx_cvd, bp_treatment, diabetes_treatment, lipids_treatment, mi_stroke_hx, plaque]

for abv, column in outcomes:
    
    IDsID = mydata[0].as_matrix()
    
    OC = []
    
    count = 0
    
    for k in range(len(IDs)):
        if not IDs[k] in IDsID:
            count += 1
        elif IDs[k] in IDsID:
            OC.append(column[k])
            
    
    
    OC = np.array([OC])
    
    mydata_df = pd.DataFrame.from_records(mydata)
    
    UD = mydata.drop(columns=0)
    
    
    
    OC.reshape((1, len(OC[0])))
    
    OC_df = pd.DataFrame.from_records(OC).transpose()
    
    
    udoc = pd.concat([OC_df, UD], axis=1)
    
    print(udoc)
    
    udoc.to_csv(path_or_buf='../../data/mesa/MESA_CPMG_MBINV2_ManuallyBinnedData_BatchCorrected_LogTransformed_1stcol_%s.csv' % abv, sep=',', index_label=False, header=False)
    
    print('Saved for %s.' % abv)